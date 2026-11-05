"""Regenerate Task 1 state CSVs from the packed dense arrays (fast recovery).

Writes the template-order CSVs (panel,timestamp,station_id,link_id,mask_regime,
speed_kmh,flow_vph) from work/submissions/<dense>/<panel>_<split>.npz, for
validation then private, one panel at a time.

Usage:
  python work/pipeline/state_dense_to_csv.py --dense dense_gbm \
      --out-validation work/task1/gbm/state_gbm_validation.csv \
      --out-private    work/task1/gbm/state_gbm_private.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
import traffic

COLS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]


def build(split, dense_dir, panels):
    frames = []
    for panel in panels:
        z = np.load(os.path.join(dense_dir, f"{panel}_{split}.npz"))
        speed, flow, mask = z["speed"], z["flow"], z["mask"]
        masked = traffic.load_mainline(panel, split)["masked"]
        tg = traffic.load_targets(panel, split)
        ti, si = tg["t_idx"].astype(np.int64), tg["s_idx"].astype(np.int64)
        if not mask[ti, si].all():
            raise SystemExit(f"{panel} {split}: dense mask incomplete "
                             f"({int(mask[ti, si].sum())}/{len(ti)})")
        frames.append(pd.DataFrame({
            "panel": panel,
            "timestamp": traffic.epoch_to_iso(masked["times"][ti]),
            "station_id": masked["stations"][si].astype(str),
            "link_id": masked["links"][si].astype(str),
            "mask_regime": np.array(traffic.REGIME_NAMES)[tg["regime"].astype(np.int64)],
            "speed_kmh": speed[ti, si],
            "flow_vph": flow[ti, si],
        }, columns=COLS))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="dense_gbm")
    ap.add_argument("--out-validation", default=os.path.join(
        traffic.WORK, "task1", "gbm", "state_gbm_validation.csv"))
    ap.add_argument("--out-private", default=os.path.join(
        traffic.WORK, "task1", "gbm", "state_gbm_private.csv"))
    args = ap.parse_args()
    dense_dir = os.path.join(traffic.WORK, "submissions", args.dense)
    for split, out in (("validation", args.out_validation), ("private", args.out_private)):
        df = build(split, dense_dir, traffic.PANELS)
        df.to_csv(out, index=False)
        print(f"{split}: {len(df):,} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()
