"""Generate Task 1 state submissions for validation and private in one pass.

The profile is learned once per panel from the train unmasked layer, then both
split views are reconstructed (each day's residual field smoothed over the
observed cells).  Processing panel by panel keeps memory small.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import gc

import numpy as np
import pandas as pd
import traffic
import proto_smooth as P
from tune_smooth import _smooth_w

COLUMNS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def predict_view(panel, split, P_s, P_q, lam_t=1.0, lam_s_speed=0.10, lam_s_flow=0.30, iters=60):
    masked = traffic.load_mainline(panel, split)["masked"]
    times = pd.to_datetime(masked["times"], unit="s", utc=True)
    T, S = masked["speed"].shape
    wd = times.weekday.to_numpy()
    obs_s = masked["speed"].astype(np.float64)
    obs_q = masked["flow"].astype(np.float64)
    obs_mask = np.isfinite(obs_s) & np.isfinite(obs_q)
    wgt = (masked["pct"].astype(np.float64) / 100.0).clip(0.0, 1.0)
    pred_s = np.full((T, S), np.nan, dtype=np.float32)
    pred_q = np.full((T, S), np.nan, dtype=np.float32)
    for d in range(T // 288):
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
    free = masked["free_speed"].astype(np.float32)[None, :]
    cap = masked["capacity"].astype(np.float32)[None, :]
    pred_s = np.clip(pred_s, 1.0, np.maximum(free * 1.35, 20.0))
    pred_q = np.clip(pred_q, 0.0, np.maximum(cap * 1.5, 100.0))
    tg = traffic.load_targets(panel, split)
    ti = tg["t_idx"].astype(np.int64)
    si = tg["s_idx"].astype(np.int64)
    out = pd.DataFrame({
        "panel": panel,
        "timestamp": traffic.epoch_to_iso(masked["times"][ti]),
        "station_id": masked["stations"][si].astype(str),
        "link_id": masked["links"][si].astype(str),
        "mask_regime": np.array(traffic.REGIME_NAMES)[tg["regime"].astype(np.int64)],
        "speed_kmh": pred_s[ti, si],
        "flow_vph": pred_q[ti, si],
    }, columns=COLUMNS)
    del masked, obs_s, obs_q, obs_mask, wgt, pred_s, pred_q
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["validation", "private"])
    ap.add_argument("--panels", nargs="*", default=traffic.PANELS)
    ap.add_argument("--outdir", default=r"C:\Users\lenovo\Desktop\kaggle\work\submissions")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    paths = {s: os.path.join(args.outdir, f"state_{s}.csv") for s in args.splits}
    headers = {s: True for s in args.splits}
    total = {s: 0 for s in args.splits}
    t0 = time.time()
    for panel in args.panels:
        t1 = time.time()
        plain = traffic.load_mainline(panel, "train")["plain"]
        P_s, P_q = P.build_profiles(plain, np.ones(len(plain["times"]), dtype=bool))
        del plain
        gc.collect()
        for split in args.splits:
            out = predict_view(panel, split, P_s, P_q)
            out.to_csv(paths[split], mode="w" if headers[split] else "a",
                       header=headers[split], index=False)
            headers[split] = False
            total[split] += len(out)
            del out
            gc.collect()
        print(f"{panel:12s} {time.time()-t1:5.1f}s", flush=True)
    for s in args.splits:
        print(f"Wrote {total[s]:,} rows to {paths[s]}", flush=True)
    print(f"done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
