"""Holdout evaluation of the Task 1 GBM + day-similarity tensor-completion stack (v3 test).

Same protocol as run_eval.py (fit on train days < n_days-holdout; predict the
published target rows of the last holdout days from the masked view), but the
feature matrix is extended with DaySimFeatures (same-tod similarity-weighted
completion over the month's other days + neighbour-day same-tod values), and the
script also scores the stand-alone tensor estimate and GBM/tensor blends.

Usage:
  python run_eval_ds.py --panel D7_I10_E --holdout-days 60 [--train-cap 400000]
                        [--alpha-sim 0.25] [--dow-boost 2.0] [--t-decay 0]
                        [--blend-grid]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np  # noqa: E402
import traffic  # noqa: E402

from t1lib import score, fmt  # noqa: E402
from t1gbm import LGB_PARAMS, clip_predictions  # noqa: E402
from run_eval import prepare, mem_gb  # noqa: E402
from daysim import DaySimFeatures, day_dow  # noqa: E402
import lightgbm as lgb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="D7_I10_E")
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--train-cap", type=int, default=400000)
    ap.add_argument("--eval-cap", type=int, default=800000)
    ap.add_argument("--n-estimators", type=int, default=LGB_PARAMS["n_estimators"])
    ap.add_argument("--num-leaves", type=int, default=LGB_PARAMS["num_leaves"])
    ap.add_argument("--learning-rate", type=float, default=LGB_PARAMS["learning_rate"])
    ap.add_argument("--alpha-sim", type=float, default=0.25)
    ap.add_argument("--dow-boost", type=float, default=2.0)
    ap.add_argument("--t-decay", type=float, default=0.0)
    ap.add_argument("--blend-grid", action="store_true")
    ap.add_argument("--lrt", action="store_true", help="also add low-rank tensor completion features")
    ap.add_argument("--lrt-rank", type=int, default=12)
    ap.add_argument("--lrt-iters", type=int, default=10)
    ap.add_argument("--no-stack", action="store_true", help="score only base+GBM (control)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    b = prepare(args.panel, args.holdout_days, args.train_cap, args.eval_cap, args.seed)
    fb, masked = b["fb"], b["masked"]
    fi, hi = b["fit_idx"], b["hold_idx"]
    rows = np.concatenate([fi, hi])
    ts_rows = b["ts_all"][rows]
    ss_rows = b["ss_all"][rows]
    reg_rows = b["reg_all"][rows]
    D = fb.build(ts_rows, ss_rows, reg_rows)
    nf = len(fi)
    base_s_all = D["base_s"]
    base_q_all = D["base_q"]
    X = D["X"]
    del D
    gc.collect()
    print(f"  base features {X.shape} {time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)

    ds_names = []
    ds_cols = {}
    if not args.no_stack:
        dd = day_dow(fb.dow, fb.days, fb.n_days)
        ds = DaySimFeatures(masked, dd, alpha_sim=args.alpha_sim,
                            dow_boost=args.dow_boost, t_decay=args.t_decay)
        Fd = ds.build(ts_rows, ss_rows)
        ds_names = ds.extra_names()
        for k in ds_names:
            ds_cols[k] = Fd[k]
        del Fd, ds
        gc.collect()
        print(f"  day-sim features {len(ds_names)} {time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)
        if args.lrt:
            from daysim import LRTFeatures
            lrt = LRTFeatures(masked, rank=args.lrt_rank, iters=args.lrt_iters, seed=args.seed)
            Fl = lrt.build(ts_rows, ss_rows)
            for k in lrt.extra_names():
                ds_names.append(k)
                ds_cols[k] = Fl[k]
            del Fl, lrt
            gc.collect()
            print(f"  +LRT features {len(ds_names)} {time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)
        Xd = np.column_stack([ds_cols[k] for k in ds_names])
        Xnew = np.empty((X.shape[0], X.shape[1] + Xd.shape[1]), np.float32)
        Xnew[:, :X.shape[1]] = X
        Xnew[:, X.shape[1]:] = Xd
        del X, Xd
        gc.collect()
        X = Xnew

    Xf, Xh = X[:nf], X[nf:]
    params = dict(LGB_PARAMS)
    params.update({"n_estimators": args.n_estimators, "num_leaves": args.num_leaves,
                   "learning_rate": args.learning_rate, "seed": args.seed})
    res_extra = {}
    ms = lgb.LGBMRegressor(**params)
    ms.fit(Xf, b["true_s_fit"] - base_s_all[:nf])
    print(f"  speed model {time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)
    mq = lgb.LGBMRegressor(**params)
    mq.fit(Xf, b["true_q_fit"] - base_q_all[:nf])
    print(f"  flow model {time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)
    pred_s = base_s_all[nf:] + ms.predict(Xh)
    pred_q = base_q_all[nf:] + mq.predict(Xh)
    del X, Xf, Xh
    gc.collect()

    ss_h = b["ss_all"][hi]
    reg_h = b["reg_all"][hi]
    lanes = masked["lanes"][ss_h].astype(np.float64)
    free_speed = masked["free_speed"][ss_h].astype(np.float64)

    def score_clip(ps, pq):
        ps, pq = clip_predictions(ps, pq, free_speed, b["lo_s"][ss_h], b["hi_s"][ss_h], b["hi_q"][ss_h])
        return score(b["true_s_hold"], b["true_q_hold"], ps, pq, lanes, reg_h), ps, pq

    res_gbm, ps_clip, pq_clip = score_clip(pred_s, pred_q)
    print("\n[%s gbm%s] %s" % (args.panel, "+ds" if not args.no_stack else "", fmt(res_gbm)), flush=True)

    if not args.no_stack:
        # stand-alone tensor estimate and blends
        t_s = ds_cols["ds_sim_adj_s"][nf:].astype(np.float64)
        t_q = ds_cols["ds_sim_adj_q"][nf:].astype(np.float64)
        res_t, _, _ = score_clip(t_s, t_q)
        print("[%s tensor stand-alone] %s" % (args.panel, fmt(res_t)), flush=True)
        if args.blend_grid:
            best = (res_gbm["macro"], 1.0)
            for w in (0.95, 0.9, 0.85, 0.8, 0.7):
                bs = np.where(np.isfinite(t_s), w * ps_clip + (1 - w) * t_s, ps_clip)
                bq = np.where(np.isfinite(t_q), w * pq_clip + (1 - w) * t_q, pq_clip)
                rb, _, _ = score_clip(bs, bq)
                print("  blend w=%.2f %s" % (w, fmt(rb)), flush=True)
                if rb["macro"] > best[0]:
                    best = (rb["macro"], w)
            print("  best blend w=%.2f macro=%.4f" % (best[1], best[0]), flush=True)
        if args.lrt:
            l_s = ds_cols["lrt_s"][nf:].astype(np.float64)
            l_q = ds_cols["lrt_q"][nf:].astype(np.float64)
            res_l, _, _ = score_clip(l_s, l_q)
            print("[%s LRT stand-alone] %s" % (args.panel, fmt(res_l)), flush=True)
            if args.blend_grid:
                best = (res_gbm["macro"], 1.0)
                for w in (0.95, 0.9, 0.85, 0.8, 0.7):
                    bs = np.where(np.isfinite(l_s), w * ps_clip + (1 - w) * l_s, ps_clip)
                    bq = np.where(np.isfinite(l_q), w * pq_clip + (1 - w) * l_q, pq_clip)
                    rb, _, _ = score_clip(bs, bq)
                    print("  LRT blend w=%.2f %s" % (w, fmt(rb)), flush=True)
                    if rb["macro"] > best[0]:
                        best = (rb["macro"], w)
                print("  best LRT blend w=%.2f macro=%.4f" % (best[1], best[0]), flush=True)
    print(f"total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
