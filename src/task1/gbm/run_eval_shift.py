"""Distribution-shift experiment: Task-2 horizon blanks in the released masked input.

Measured pattern (probe_shift.py): on validation/private the released masked view
additionally blanks 18 consecutive 5-min time steps (all links, both channels)
starting at forecast_origin+1..+18 for each of the 10 Task-2 windows of the
split-month.  Train has ZERO such blanks (mask simulation only blanks the R1/R2/R3
target cells).  So a model trained on train masks sees a different missingness
pattern on the scored splits.

This script measures, on the last-60-days holdout of train:
  * S_state of the shipped v1 model under the plain R-mask (reference),
  * S_state of the same model when the holdout input additionally carries
    validation-like horizon blocks (rate 10 windows per 31 days, 18 full steps),
  * S_state of a model retrained with the same horizon blocks injected into the
    fit-day input (augmented training mask, R target masks unchanged).
Per-panel and per-regime; optional task3 physics proxy when the ship bar is met.

  python run_eval_shift.py [--panels ...] [--train-cap 400000] [--origins-rate 10/31]
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

from t1lib import score, fmt  # noqa: E402
from t1gbm import LGB_PARAMS, FeatureBuilder, clip_predictions  # noqa: E402
from run_eval import prepare, mem_gb  # noqa: E402
import lightgbm as lgb  # noqa: E402

BLOCK_STEPS = 18


def sample_origins(days_pool, n_orig, rng, n_tod=288):
    """Return Time indices t0 such that t0+1..t0+18 are all inside the same day."""
    days = rng.choice(np.asarray(days_pool), size=n_orig, replace=True)
    steps = rng.integers(0, n_tod - BLOCK_STEPS, size=n_orig)
    return (days.astype(np.int64) * n_tod + steps).astype(np.int64)


def add_blocks(obs: dict, origins: np.ndarray, T: int) -> dict:
    """Copy speed/flow and NaN out the 18 steps after each origin (in place on copies)."""
    out = dict(obs)
    sp = np.array(obs["speed"], copy=True)
    fl = np.array(obs["flow"], copy=True)
    flat_s = sp.reshape(-1)
    flat_f = fl.reshape(-1)
    S = sp.shape[1]
    for t0 in origins:
        a = (t0 + 1) * S
        b = (t0 + 1 + BLOCK_STEPS) * S
        if b > flat_s.size:
            continue
        flat_s[a:b] = np.nan
        flat_f[a:b] = np.nan
    out["speed"] = sp
    out["flow"] = fl
    return out


def build_ds(masked, fb, ts, ss, reg, dd=None, ds_cache=None):
    attrs = {k: masked[k] for k in ("lanes", "free_speed", "capacity", "k_jam",
                                    "milepost", "length_km")}
    f = FeatureBuilder(masked, (fb.ps, fb.pq), (fb.dsm, fb.dqm), attrs,
                       fb.order_p, fb.inv, fb.n_days, fb.dow, fb.tod)
    D = f.build(ts, ss, reg)
    del f
    gc.collect()
    if dd is not None:
        from daysim import DaySimFeatures
        ds = DaySimFeatures(masked, dd, alpha_sim=0.25, dow_boost=2.0)
        Fd = ds.build(ts, ss)
        names = ds.extra_names()
        del ds
        gc.collect()
        Xd = np.column_stack([Fd[k] for k in names])
        X = np.empty((D["X"].shape[0], D["X"].shape[1] + Xd.shape[1]), np.float32)
        X[:, :D["X"].shape[1]] = D["X"]
        X[:, D["X"].shape[1]:] = Xd
        D["X"] = X
        del Xd, Fd
        gc.collect()
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*",
                    default=["D7_I10_E", "D12_I5_N", "D7_I405_S", "D7_I210_W"])
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--train-cap", type=int, default=400000)
    ap.add_argument("--eval-cap", type=int, default=800000)
    ap.add_argument("--n-estimators", type=int, default=700)
    ap.add_argument("--origins-per-month", type=float, default=10.0)
    ap.add_argument("--fit-mult", type=float, default=1.0,
                    help="multiply the training-side horizon-block rate")
    ap.add_argument("--ds-features", action="store_true",
                    help="append the day-similarity/neighbour-day features")
    ap.add_argument("--out", default="shift_holdout.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out_rows = []
    for panel in args.panels:
        t0 = time.time()
        b = prepare(panel, args.holdout_days, args.train_cap, args.eval_cap, args.seed)
        fb = b["fb"]
        masked = b["masked"]
        n_days = fb.n_days
        day_lo = n_days - args.holdout_days
        T = masked["speed"].shape[0]
        n_fit_days = day_lo
        n_hold_days = n_days - day_lo
        n_orig_fit = max(1, int(round(n_fit_days * args.origins_per_month / 31.0 * args.fit_mult)))
        n_orig_hold = max(1, int(round(n_hold_days * args.origins_per_month / 31.0)))
        fit_days = np.arange(0, day_lo)
        hold_days = np.arange(day_lo, n_days)
        orig_fit = sample_origins(fit_days, n_orig_fit, rng)
        orig_hold = sample_origins(hold_days, n_orig_hold, rng)
        rows_fit = b["ts_all"][b["fit_idx"]]
        ss_fit = b["ss_all"][b["fit_idx"]]
        reg_fit = b["reg_all"][b["fit_idx"]]
        rows_hold = b["ts_all"][b["hold_idx"]]
        ss_hold = b["ss_all"][b["hold_idx"]]
        reg_hold = b["reg_all"][b["hold_idx"]]

        # ---- reference features/labels (plain R mask) ----
        Dp_fit = build_ds(masked, fb, rows_fit, ss_fit, reg_fit)
        Dp_hold = build_ds(masked, fb, rows_hold, ss_hold, reg_hold)
        # ---- augmented mask (plain R mask + horizon blocks) ----
        aug = add_blocks(masked, np.concatenate([orig_fit, orig_hold]), T)
        S_ = masked["speed"].shape[1]
        h0 = day_lo * 288
        nb_hold = int(np.isnan(aug["speed"][h0:]).sum() - np.isnan(masked["speed"][h0:]).sum())
        nb_fit = int(np.isnan(aug["speed"][:h0]).sum() - np.isnan(masked["speed"][:h0]).sum())
        in_block_hold = np.zeros(len(rows_hold), dtype=bool)
        for t0 in orig_hold:
            in_block_hold |= (rows_hold > t0) & (rows_hold <= t0 + BLOCK_STEPS)
        n_aff = int(in_block_hold.sum())
        print(f"[{panel}] origins fit={n_orig_fit} hold={n_orig_hold}; new blank cells "
              f"hold={nb_hold} ({nb_hold/(n_hold_days*288*S_)*100:.2f}% of holdout grid), "
              f"fit={nb_fit}; holdout targets inside blocks={n_aff} "
              f"({n_aff/len(rows_hold)*100:.2f}% of holdout targets) "
              f"{time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)

        params = dict(LGB_PARAMS)
        params.update({"n_estimators": args.n_estimators, "seed": args.seed})
        dd = None
        if args.ds_features:
            from daysim import day_dow
            dd = day_dow(fb.dow, fb.days, fb.n_days)

        def fit_pair(masked_view, rows, ss, reg):
            D = build_ds(masked_view, fb, rows, ss, reg, dd=dd)
            ms = lgb.LGBMRegressor(**params)
            ms.fit(D["X"], b["true_s_fit"] - D["base_s"])
            mq = lgb.LGBMRegressor(**params)
            mq.fit(D["X"], b["true_q_fit"] - D["base_q"])
            del D
            gc.collect()
            return (ms, mq)

        ss_h = b["ss_all"][b["hold_idx"]]
        lanes = masked["lanes"][ss_h].astype(np.float64)
        free_speed = masked["free_speed"][ss_h].astype(np.float64)
        lo_s, hi_s, hi_q = b["lo_s"][ss_h], b["hi_s"][ss_h], b["hi_q"][ss_h]

        def evaluate(masked_view, model):
            D = build_ds(masked_view, fb, rows_hold, ss_hold, reg_hold, dd=dd)
            ps = D["base_s"] + model[0].predict(D["X"])
            pq = D["base_q"] + model[1].predict(D["X"])
            del D
            gc.collect()
            ps, pq = clip_predictions(ps, pq, free_speed, lo_s, hi_s, hi_q)
            return score(b["true_s_hold"], b["true_q_hold"], ps, pq, lanes, reg_hold)

        mA = fit_pair(masked, rows_fit, ss_fit, reg_fit)
        print(f"  model A (plain mask) fit {time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)
        mB = fit_pair(aug, rows_fit, ss_fit, reg_fit)
        print(f"  model B (augmented) fit {time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)
        ref = evaluate(masked, mA)
        cur = evaluate(aug, mA)
        ret = evaluate(aug, mB)
        ref_b = evaluate(masked, mB)
        del aug, mA, mB, b
        gc.collect()
        print("  [%s] A plain-mask      %s" % (panel, fmt(ref)), flush=True)
        print("  [%s] A val-like-mask   %s" % (panel, fmt(cur)), flush=True)
        print("  [%s] B retrained+aug   %s" % (panel, fmt(ret)), flush=True)
        print("  [%s] B on plain-mask   %s" % (panel, fmt(ref_b)), flush=True)
        out_rows += [
            dict(panel=panel, model="A_plainmask", S_state=ref["macro"],
                 R1=ref[1]["S_state"], R2=ref[2]["S_state"], R3=ref[3]["S_state"]),
            dict(panel=panel, model="A_vallikemask", S_state=cur["macro"],
                 R1=cur[1]["S_state"], R2=cur[2]["S_state"], R3=cur[3]["S_state"]),
            dict(panel=panel, model="B_retrained_aug_vallikemask", S_state=ret["macro"],
                 R1=ret[1]["S_state"], R2=ret[2]["S_state"], R3=ret[3]["S_state"]),
            dict(panel=panel, model="B_plainmask", S_state=ref_b["macro"],
                 R1=ref_b[1]["S_state"], R2=ref_b[2]["S_state"], R3=ref_b[3]["S_state"]),
        ]
        print(f"  panel done {time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(os.path.join(HERE, args.out), index=False)
    piv = df.pivot(index="panel", columns="model", values="S_state")
    print("\n=== S_state (holdout) ===")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    d_shift = piv["A_vallikemask"] - piv["A_plainmask"]
    d_retrain = piv["B_retrained_aug_vallikemask"] - piv["A_vallikemask"]
    print("\nshift cost (A val-like - A plain):", d_shift.round(4).to_dict(),
          " mean=%.5f" % d_shift.mean())
    print("retrain gain (B - A under val-like):", d_retrain.round(4).to_dict(),
          " mean=%.5f" % d_retrain.mean(), " max regression=%.5f" % max(0.0, -d_retrain.min()))
    if "B_plainmask" in piv.columns:
        d_plain = piv["B_plainmask"] - piv["A_plainmask"]
        print("plain-mask regression (B - A):", d_plain.round(4).to_dict(),
              " mean=%.5f" % d_plain.mean(), " max regression=%.5f" % max(0.0, -d_plain.min()))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
