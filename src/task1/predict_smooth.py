"""Production Task 1 predictor: historical profile + smoothed residual field.

For each panel and split:
  * the weekday x time-of-day speed/flow profile is learned from the train
    unmasked observation layer (never from the split being predicted);
  * the residual field (observation - profile) of each day is reconstructed
    from the observed cells by Jacobi smoothing in (time, space), which fills
    the masked cells from their neighbours;
  * predictions are the profile plus the smoothed residual, clipped to physical
    bounds.

The output CSV has exactly the rows of task1/<panel>/<split>/sample_submission_state.csv
in template order.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
import traffic
import proto_smooth as P
from tune_smooth import _smooth_w

COLUMNS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def profiles_from_train(panel):
    plain = traffic.load_mainline(panel, "train")["plain"]
    fit = np.ones(len(plain["times"]), dtype=bool)
    return P.build_profiles(plain, fit)


def predict_panel(panel, split, lam_t=1.0, lam_s_speed=0.10, lam_s_flow=0.30, iters=60):
    masked = traffic.load_mainline(panel, split)["masked"]
    P_s, P_q = profiles_from_train(panel)
    times = pd.to_datetime(masked["times"], unit="s", utc=True)
    T, S = masked["speed"].shape
    wd = times.weekday.to_numpy()
    obs_s = masked["speed"].astype(np.float64)
    obs_q = masked["flow"].astype(np.float64)
    obs_mask = np.isfinite(obs_s) & np.isfinite(obs_q)
    wgt = (masked["pct"].astype(np.float64) / 100.0).clip(0.0, 1.0)
    pred_s = np.full((T, S), np.nan, dtype=np.float32)
    pred_q = np.full((T, S), np.nan, dtype=np.float32)
    n_days = T // 288
    for d in range(n_days):
        sl = slice(d * 288, (d + 1) * 288)
        w = wd[sl][0]
        m = obs_mask[sl]
        ww = np.where(m, wgt[sl], 0.0)
        base_s = P_s[:, w, :].T
        base_q = P_q[:, w, :].T
        r_s = np.where(m, obs_s[sl] - base_s, 0.0)
        r_q = np.where(m, obs_q[sl] - base_q, 0.0)
        r_s_hat = _smooth_w(r_s, ww, lam_t, lam_s_speed, iters)
        r_q_hat = _smooth_w(r_q, ww, lam_t, lam_s_flow, iters)
        pred_s[sl] = np.maximum(base_s + r_s_hat, 1.0)
        pred_q[sl] = np.maximum(base_q + r_q_hat, 0.0)
    # physical clipping
    free = masked["free_speed"].astype(np.float32)[None, :]
    cap = masked["capacity"].astype(np.float32)[None, :]
    pred_s = np.clip(pred_s, 1.0, np.maximum(free * 1.35, 20.0))
    pred_q = np.clip(pred_q, 0.0, np.maximum(cap * 1.5, 100.0))
    tg = traffic.load_targets(panel, split)
    ti = tg["t_idx"].astype(np.int64)
    si = tg["s_idx"].astype(np.int64)
    return {
        "panel": panel,
        "timestamp": traffic.epoch_to_iso(masked["times"][ti]),
        "station_id": masked["stations"][si].astype(str),
        "link_id": masked["links"][si].astype(str),
        "mask_regime": np.array(traffic.REGIME_NAMES)[tg["regime"].astype(np.int64)],
        "speed_kmh": pred_s[ti, si],
        "flow_vph": pred_q[ti, si],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation", "private"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--panels", nargs="*", default=traffic.PANELS)
    ap.add_argument("--lam-t", type=float, default=1.0)
    ap.add_argument("--lam-s-speed", type=float, default=0.10)
    ap.add_argument("--lam-s-flow", type=float, default=0.30)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    t0 = time.time()
    first = True
    total = 0
    for panel in args.panels:
        t1 = time.time()
        try:
            data = predict_panel(panel, args.split, args.lam_t, args.lam_s_speed, args.lam_s_flow, args.iters)
        except FileNotFoundError:
            print(f"{panel}: no cache for {args.split}, skipped", flush=True)
            continue
        out = pd.DataFrame(data, columns=COLUMNS)
        out.to_csv(args.output, mode="w" if first else "a", header=first, index=False)
        first = False
        total += len(out)
        print(f"{panel:12s} {len(out):8d} rows  {time.time()-t1:5.1f}s", flush=True)
    print(f"Wrote {total:,} rows to {args.output} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
