"""Build arbitrary-weight U-Net blends by linear extrapolation from the w=0.5 files.

state_w = (1-w)*GBM + w*U  =>  U = GBM + (state_0.5 - GBM)/0.5
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = r"C:\Users\lenovo\Desktop\kaggle"
UNET = os.path.join(ROOT, "work", "task1", "unet")
GBM = os.path.join(ROOT, "work", "task1", "gbm")


def build(prefix, w, tag):
    for split in ("validation", "private"):
        g = pd.read_csv(os.path.join(GBM, f"state_gbm_{split}.csv"))
        b = pd.read_csv(os.path.join(UNET, f"{prefix}_w05_{split}.csv"))
        assert (g[["panel", "timestamp", "station_id"]].values ==
                b[["panel", "timestamp", "station_id"]].values).all(), "row misalignment"
        u = g.copy()
        u["speed_kmh"] = g["speed_kmh"] + (b["speed_kmh"] - g["speed_kmh"]) / 0.5
        out = g.copy()
        out["speed_kmh"] = ((1 - w) * g["speed_kmh"] + w * u["speed_kmh"]).round(3)
        out["flow_vph"] = ((1 - w) * g["flow_vph"] + w * u["flow_vph"]).round(1)
        dst = os.path.join(UNET, f"{prefix}_{tag}_{split}.csv")
        out.to_csv(dst, index=False)
        print(split, len(out), "->", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="state_gbm_unet")
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    build(a.prefix, a.w, a.tag)
