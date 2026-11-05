"""Apples-to-apples: the previously delivered median adjuster on the same GBM
holdout panels, for comparison with the notebook density correction."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
for p in (os.path.join(ROOT, "libs"), os.path.join(ROOT, "work", "lib"),
          os.path.join(ROOT, "work", "task3")):
    sys.path.insert(0, p)
import local_eval as le
import adjust as ad
import density_adjust as da

PANELS = ["D7_I10_E", "D7_I210_W", "D7_I405_S", "D12_I5_N"]
rows = []
for panel in PANELS:
    pdata, cs, cf = da.load_holdout_panel(panel)
    flux = le.build_frozen_flux(pdata, "projected", 1, flux_jitter=da.JITTER)
    base = da.metrics(pdata, flux, cs, cf)
    for strength in (0.5, 0.75):
        _, cf2, diag = ad.adjust_panel(pdata, cs, cf, method="median", window=5,
                                       strength=strength)
        m = da.metrics(pdata, flux, cs, cf2)
        d = {k: m[k] - base[k] for k in base}
        est = 0.35 * d["S_state"] + 0.15 * (d["S_FD"] / 3 + 2 * d["S_LWR"] / 3)
        rows.append({"panel": panel, "method": f"median_w5_s{strength}", "dS_state": d["S_state"],
                     "dS_FD": d["S_FD"], "dS_LWR": d["S_LWR"], "dS_physics": d["S_physics"],
                     "est_LB": est, "mean_abs_dq": diag["mean_abs_dq"]})
        print(rows[-1], flush=True)
    del pdata, cs, cf, flux
df = pd.DataFrame(rows)
agg = df.groupby("method")[["dS_state", "dS_FD", "dS_LWR", "dS_physics", "est_LB"]].mean()
print("\n=== median adjuster, macro over 4 holdout panels ===")
print(agg.round(5).to_string())
df.to_csv(os.path.join(da.WORK3, "reports", "median_adjuster_holdout4.csv"), index=False)
