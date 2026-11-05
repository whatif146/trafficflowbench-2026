"""Verify that the packaged U-Net prediction npz reproduces the deployed blend.

For one panel/split: 0.6*shipped GBM state + 0.4*U-Net npz prediction should equal
the deployed state_gbm_unet CSV (written by deploy_refit.py) row-for-row.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

UNET = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\work\task1\unet"
GBM = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\work\task1\gbm"
PANEL = "D7_I10_E"
SPLIT = "validation"


def main():
    cols = ["panel", "timestamp", "station_id", "speed_kmh", "flow_vph"]
    g = pd.read_csv(os.path.join(GBM, f"state_gbm_{SPLIT}.csv"), usecols=cols)
    g = g[g.panel == PANEL].reset_index(drop=True)
    u = np.load(os.path.join(UNET, f"pred_unet_refit_{PANEL}_{SPLIT}.npz"))
    out = pd.read_csv(os.path.join(UNET, f"state_gbm_unet_{SPLIT}.csv"), usecols=cols)
    out = out[out.panel == PANEL].reset_index(drop=True)
    print("rows gbm/unet/out:", len(g), len(u["v"]), len(out))
    if not (len(g) == len(u["v"]) == len(out)):
        raise SystemExit("row mismatch")
    v = 0.6 * g.speed_kmh.to_numpy() + 0.4 * u["v"]
    q = 0.6 * g.flow_vph.to_numpy() + 0.4 * u["q"]
    dv = np.abs(v - out.speed_kmh.to_numpy())
    dq = np.abs(q - out.flow_vph.to_numpy())
    print(f"speed max|diff| {dv.max():.6g} mean {dv.mean():.3g}")
    print(f"flow  max|diff| {dq.max():.6g} mean {dq.mean():.3g}")
    print("MATCH" if dv.max() < 5e-3 and dq.max() < 5e-2 else "MISMATCH")


if __name__ == "__main__":
    main()
