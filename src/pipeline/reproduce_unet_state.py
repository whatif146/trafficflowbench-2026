"""Reproduce the deployed U-Net state blend from frozen predictions.

Inputs:
  work/task1/gbm/state_gbm_{validation,private}.csv        (Task-1 LightGBM output)
  work/task1/unet/deploy_npz/pred_unet_refit_<PANEL>_<split>.npz
      keys v, q: seed-averaged U-Net predictions for that panel/split template rows
  built by task1-unet's deploy_refit.py (seeds 43+44, refit on all 273 train days).

Output:
  <out-dir>/state_gbm_unet_final_<split>.csv  (same columns/order as the shipped
  state files; values = w*GBM + (1-w)*U-Net)

Verified to reproduce the deployed state_gbm_unet_*.csv to CSV rounding
(speed max |diff| 5e-4 km/h, flow max |diff| 5e-3 vph).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
WORK = os.path.join(ROOT, "work")
GBM_DIR = os.path.join(WORK, "task1", "gbm")
UNET_DIR = os.path.join(WORK, "task1", "unet")
NPZ_DIR = os.path.join(UNET_DIR, "deploy_npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=float, default=0.6, help="GBM weight in the blend")
    ap.add_argument("--gbm-dir", default=GBM_DIR)
    ap.add_argument("--npz-dir", default=NPZ_DIR)
    ap.add_argument("--out-dir", default=UNET_DIR)
    ap.add_argument("--splits", nargs="+", default=["validation", "private"])
    args = ap.parse_args()
    for split in args.splits:
        src = os.path.join(args.gbm_dir, f"state_gbm_{split}.csv")
        df = pd.read_csv(src)
        cols = list(df.columns)
        out_frames = []
        for panel, g in df.groupby("panel", sort=True):
            p = os.path.join(args.npz_dir, f"pred_unet_refit_{panel}_{split}.npz")
            if not os.path.exists(p):
                raise SystemExit(f"missing U-Net prediction: {p}")
            z = np.load(p)
            if len(z["v"]) != len(g):
                raise SystemExit(f"{panel} {split}: npz rows {len(z['v'])} != gbm rows {len(g)}")
            g = g.copy()
            g["speed_kmh"] = args.w * g["speed_kmh"].to_numpy() + (1 - args.w) * z["v"]
            g["flow_vph"] = args.w * g["flow_vph"].to_numpy() + (1 - args.w) * z["q"]
            out_frames.append(g)
            print(f"{panel:12s} {split:10s} rows {len(g):>7,} blended", flush=True)
        out = pd.concat(out_frames, ignore_index=True)[cols]
        dst = os.path.join(args.out_dir, f"state_gbm_unet_final_{split}.csv")
        out.to_csv(dst, index=False)
        print(f"wrote {dst} rows={len(out):,}")


if __name__ == "__main__":
    main()
