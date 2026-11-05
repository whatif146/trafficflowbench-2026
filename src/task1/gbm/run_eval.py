"""Holdout evaluation of the Task 1 interp/smoother + LightGBM residual model.

Protocol (mirrors work/task1/eval.py):
  fit on train days < n_days - holdout_days; predict the published target rows
  of the held-out days using only the masked view; score with the official
  Task 1 formula (t1lib.score).  Predictions for the held-out rows can be
  written as a CSV that work/task1/eval.py scores directly.

Memory-lean: plain is loaded only to extract labels/profiles/clip bounds and is
freed before the masked layer is loaded; features are built once for the union
of fit and holdout rows.

Usage:
  python run_eval.py --panel D7_I10_E --holdout-days 60 [--train-cap 400000]
                     [--csv out_holdout.csv] [--no-gbm] [--n-estimators 600]
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
import pandas as pd  # noqa: E402
import traffic  # noqa: E402

from t1lib import (day_index, tod_index, dow_index, link_perm, score, fmt)  # noqa: E402
from t1gbm import (build_profiles, per_day_station_means, per_link_clip_bounds,  # noqa: E402
                   FeatureBuilder, fit_models, clip_predictions, LGB_PARAMS)


def mem_gb():
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = M()
        st.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 1e9
    except Exception:
        return float("nan")


def prepare(panel, holdout_days, train_cap, eval_cap, seed):
    """Load plain (labels/profiles), then masked (features); returns a bundle."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    plain = traffic._load_or_build(panel, "train", "plain")
    times = plain["times"]
    days = day_index(times)
    tod = tod_index(times)
    dow = dow_index(times)
    n_days = int(days.max()) + 1
    day_lo = n_days - holdout_days
    fit_day_t = days < day_lo
    tg = traffic.load_targets(panel, "train")
    ts_all = tg["t_idx"].astype(np.int64)
    ss_all = tg["s_idx"].astype(np.int64)
    reg_all = tg["regime"] + 1
    day_r = days[ts_all]
    finite = np.isfinite(plain["speed"][ts_all, ss_all]) & np.isfinite(plain["flow"][ts_all, ss_all])
    is_fit = (day_r < day_lo) & finite
    is_hold = (day_r >= day_lo) & finite
    profiles = build_profiles(plain, fit_day_t, dow, tod)
    lo_s, hi_s, hi_q = per_link_clip_bounds(plain, fit_day_t)
    fit_idx = np.where(is_fit)[0]
    if len(fit_idx) > train_cap:
        fit_idx = np.sort(rng.choice(fit_idx, size=train_cap, replace=False))
    hold_idx = np.where(is_hold)[0]
    if len(hold_idx) > eval_cap:
        hold_idx = np.sort(rng.choice(hold_idx, size=eval_cap, replace=False))
    true_s_fit = plain["speed"][ts_all[fit_idx], ss_all[fit_idx]].astype(np.float64)
    true_q_fit = plain["flow"][ts_all[fit_idx], ss_all[fit_idx]].astype(np.float64)
    true_s_hold = plain["speed"][ts_all[hold_idx], ss_all[hold_idx]].astype(np.float64)
    true_q_hold = plain["flow"][ts_all[hold_idx], ss_all[hold_idx]].astype(np.float64)
    del plain
    gc.collect()

    masked = traffic._load_or_build(panel, "train", "masked")
    masked.pop("elig", None)  # not needed for features
    masked.pop("regime", None)
    order_p, inv = link_perm(masked)
    day_means = per_day_station_means(masked, days, n_days)
    attrs = {k: masked[k] for k in ("lanes", "free_speed", "capacity", "k_jam", "milepost", "length_km")}
    fb = FeatureBuilder(masked, profiles, day_means, attrs, order_p, inv, n_days, dow, tod)
    print(f"[{panel}] loaded {time.time()-t0:.1f}s n_days={n_days} holdout={holdout_days} "
          f"fit_rows={len(fit_idx)} hold_rows={len(hold_idx)} avail={mem_gb():.2f}GB", flush=True)
    return {
        "panel": panel, "fb": fb, "ts_all": ts_all, "ss_all": ss_all, "reg_all": reg_all,
        "fit_idx": fit_idx, "hold_idx": hold_idx, "true_s_fit": true_s_fit,
        "true_q_fit": true_q_fit, "true_s_hold": true_s_hold, "true_q_hold": true_q_hold,
        "lo_s": lo_s, "hi_s": hi_s, "hi_q": hi_q, "masked": masked, "times": times, "t0": t0,
    }


def build_union(b):
    """Features for fit+hold rows in a single smoother pass set."""
    fb, ts_all, ss_all, reg_all = b["fb"], b["ts_all"], b["ss_all"], b["reg_all"]
    fi, hi = b["fit_idx"], b["hold_idx"]
    rows = np.concatenate([fi, hi])
    D = fb.build(ts_all[rows], ss_all[rows], reg_all[rows])
    nf = len(fi)
    out = {"X_fit": D["X"][:nf], "X_hold": D["X"][nf:],
           "base_s_fit": D["base_s"][:nf], "base_q_fit": D["base_q"][:nf],
           "base_s_hold": D["base_s"][nf:], "base_q_hold": D["base_q"][nf:]}
    del D
    gc.collect()
    print(f"  features built {time.time()-b['t0']:.1f}s X={out['X_fit'].shape[0]+out['X_hold'].shape[0]}x"
          f"{out['X_fit'].shape[1]} avail={mem_gb():.2f}GB", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="D7_I10_E")
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--train-cap", type=int, default=400000)
    ap.add_argument("--eval-cap", type=int, default=700000)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=LGB_PARAMS["n_estimators"])
    ap.add_argument("--num-leaves", type=int, default=LGB_PARAMS["num_leaves"])
    ap.add_argument("--learning-rate", type=float, default=LGB_PARAMS["learning_rate"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    b = prepare(args.panel, args.holdout_days, args.train_cap, args.eval_cap, args.seed)
    F = build_union(b)
    reg_h = b["reg_all"][b["hold_idx"]]
    ss_h = b["ss_all"][b["hold_idx"]]
    lanes = b["masked"]["lanes"][ss_h].astype(np.float64)
    free_speed = b["masked"]["free_speed"][ss_h].astype(np.float64)

    if args.no_gbm:
        pred_s, pred_q = F["base_s_hold"], F["base_q_hold"]
    else:
        params = dict(LGB_PARAMS)
        params["n_estimators"] = args.n_estimators
        params["num_leaves"] = args.num_leaves
        params["learning_rate"] = args.learning_rate
        import lightgbm as lgb

        ms = lgb.LGBMRegressor(**params)
        ms.fit(F["X_fit"], b["true_s_fit"] - F["base_s_fit"])
        print(f"  speed model fit {time.time()-b['t0']:.1f}s avail={mem_gb():.2f}GB", flush=True)
        mq = lgb.LGBMRegressor(**params)
        mq.fit(F["X_fit"], b["true_q_fit"] - F["base_q_fit"])
        print(f"  flow model fit {time.time()-b['t0']:.1f}s avail={mem_gb():.2f}GB", flush=True)
        pred_s = F["base_s_hold"] + ms.predict(F["X_hold"])
        pred_q = F["base_q_hold"] + mq.predict(F["X_hold"])

    res_pre = score(b["true_s_hold"], b["true_q_hold"], pred_s, pred_q, lanes, reg_h)
    print("\n[{}{}] PRE-CLIP {}".format(args.panel, "" if args.no_gbm else " gbm", fmt(res_pre)), flush=True)
    pred_s, pred_q = clip_predictions(pred_s, pred_q, free_speed,
                                      b["lo_s"][ss_h], b["hi_s"][ss_h], b["hi_q"][ss_h])
    res = score(b["true_s_hold"], b["true_q_hold"], pred_s, pred_q, lanes, reg_h)
    print("[{}{}] POST-CLIP {}".format(args.panel, "" if args.no_gbm else " gbm", fmt(res)), flush=True)
    print("  nan preds:", int(np.isnan(pred_s).sum()), flush=True)

    # ---- diagnostic: RMSE split by adjacency to a long blank run (Task-2 gap) ----
    names = FeatureBuilder.FEATURES
    i_dtb, i_dta = names.index("s_dt_before"), names.index("s_dt_after")
    dtb = F["X_hold"][:, i_dtb]
    dta = F["X_hold"][:, i_dta]
    adj = (dtb >= 8) | (dta >= 8)
    for tag, mask in (("gap-adjacent", adj), ("other", ~adj)):
        if mask.sum() == 0:
            continue
        es = np.abs(pred_s[mask] - b["true_s_hold"][mask])
        eq = np.abs(pred_q[mask] - b["true_q_hold"][mask])
        esb = np.abs(F["base_s_hold"][mask] - b["true_s_hold"][mask])
        print("  %-13s n=%7d  gbm speed RMSE=%.3f flow RMSE=%.1f | base speed RMSE=%.3f" % (
            tag, int(mask.sum()), float(np.sqrt(np.mean(es ** 2))),
            float(np.sqrt(np.mean((eq / lanes[mask]) ** 2))), float(np.sqrt(np.mean(esb ** 2)))), flush=True)

    if args.csv:
        hold = b["ts_all"][b["hold_idx"]]
        out = pd.DataFrame({
            "panel": args.panel,
            "timestamp": traffic.epoch_to_iso(b["times"][hold]),
            "station_id": b["masked"]["stations"][ss_h].astype(str),
            "link_id": b["masked"]["links"][ss_h].astype(str),
            "mask_regime": np.array(traffic.REGIME_NAMES)[reg_h - 1],
            "speed_kmh": pred_s,
            "flow_vph": pred_q,
        })
        out.to_csv(args.csv, index=False)
        print(f"wrote {args.csv} rows={len(out)}", flush=True)
    print(f"total {time.time()-b['t0']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
