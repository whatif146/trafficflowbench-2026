"""Test the flux-integration adjuster (Lead's hypothesis) on GBM holdout rows."""
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

PANEL = "D7_I10_E"
CSV = os.path.join(ROOT, "work", "task1", "gbm", "holdout_D7_I10_E.csv")
pdata = le.load_panel(PANEL, "train")
rows = le._read_submission_rows(CSV, [PANEL], "train")
cs, cf = le._rows_to_arrays(rows, pdata)
fluxes = {j: le.build_frozen_flux(pdata, "projected", 1, flux_jitter=j) for j in (0.0, 0.06, 0.12)}


def metric(cs_, cf_, label):
    st = ad.task1_state_scores(pdata, cs_, cf_)
    out = {"label": label, "S_state": st["macro"]}
    for j, fx in fluxes.items():
        r = le.evaluate({PANEL: pdata}, lambda pd_: (cs_, cf_), {PANEL: fx})
        r = r[r.level == "panel"].iloc[0]
        out[f"S_FD_{j}"] = r.S_FD
        out[f"S_LWR_{j}"] = r.S_LWR
        out[f"S_phys_{j}"] = r.S_physics
    print(f"  {label:26s} S_state={out['S_state']:.4f} "
          f"S_FD(.06)={out['S_FD_0.06']:.4f} "
          f"S_LWR(0/.06/.12)={out['S_LWR_0.0']:.4f}/{out['S_LWR_0.06']:.4f}/{out['S_LWR_0.12']:.4f} "
          f"S_phys(.06)={out['S_phys_0.06']:.4f}", flush=True)
    return out


base = metric(cs, cf, "gbm_raw")
res = [base]
for strength in (0.25, 0.5, 1.0):
    _, cf2, diag = ad.adjust_panel(pdata, cs, cf, method="flux", strength=strength)
    out = metric(cs, cf2, f"flux_s{strength:g}")
    out["mean_abs_dq"] = diag["mean_abs_dq"]
    out["dS_state"] = out["S_state"] - base["S_state"]
    for j in fluxes:
        out[f"dS_phys_{j}"] = out[f"S_phys_{j}"] - base[f"S_phys_{j}"]
    print(f"      mean|dq|={diag['mean_abs_dq']:.1f} vph  dS_state={out['dS_state']:+.4f}  "
          + "  ".join(f"dS_phys(j{j:g})={out[f'dS_phys_{j}']:+.4f}" for j in fluxes), flush=True)
    res.append(out)
pd.DataFrame(res).to_csv(os.path.join(le.WORK3, "reports", "flux_adjust_test.csv"), index=False)
print("wrote reports/flux_adjust_test.csv")
