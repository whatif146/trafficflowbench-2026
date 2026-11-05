"""Evaluate the GBM Task 1 holdout predictions and the adjuster on them.

Protocol as requested: work/task1/gbm/holdout_D7_I10_E.csv contains only the
holdout-day rows, so the proxy naturally scores the holdout cells (target cells
without a prediction are dropped by the evaluator, exactly like the official
Task 3 frame; Task 1 S_state here is the holdout-day regime aggregation).
"""
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
print(f"[{PANEL}] T={pdata['T']} L={pdata['L']}", flush=True)
rows = le._read_submission_rows(CSV, [PANEL], "train")
cs, cf = le._rows_to_arrays(rows, pdata)
print(f"GBM holdout rows mapped: {int(np.isfinite(cs).sum()):,}", flush=True)
fluxes = {j: le.build_frozen_flux(pdata, "projected", 1, flux_jitter=j)
          for j in (0.0, 0.06, 0.12)}


def metric(cs_, cf_, label):
    st = ad.task1_state_scores(pdata, cs_, cf_)
    out = {"label": label, "S_state": st["macro"],
           "rmse_speed": np.mean([r["rmse_speed"] for r in st["rows"]]),
           "rmse_flow": np.mean([r["rmse_flow"] for r in st["rows"]])}
    for j, fx in fluxes.items():
        r = le.evaluate({PANEL: pdata}, lambda pd_: (cs_, cf_), {PANEL: fx})
        r = r[r.level == "panel"].iloc[0]
        out[f"S_FD_j{j:g}"] = r.S_FD
        out[f"S_LWR_j{j:g}"] = r.S_LWR
        out[f"S_phys_j{j:g}"] = r.S_physics
        out[f"n_trans_j{j:g}"] = int(r.n_trans)
    print(f"  {label:22s} S_state={out['S_state']:.4f} (rmse v {out['rmse_speed']:.2f}, "
          f"q/lane {out['rmse_flow']:.1f})  S_FD(j.06)={out['S_FD_j0.06']:.4f}  "
          f"S_LWR(j0/.06/.12)={out['S_LWR_j0']:.4f}/{out['S_LWR_j0.06']:.4f}/"
          f"{out['S_LWR_j0.12']:.4f}  S_phys(j.06)={out['S_phys_j0.06']:.4f}", flush=True)
    return out


res = [metric(cs, cf, "gbm_raw")]
for strength in (0.75, 0.5, 1.0):
    _, cf2, diag = ad.adjust_panel(pdata, cs, cf, method="median", window=5,
                                   strength=strength)
    out = metric(cs, cf2, f"gbm_adj_s{strength:g}")
    out["mean_abs_dq"] = diag["mean_abs_dq"]
    out["dS_state"] = out["S_state"] - res[0]["S_state"]
    for j in fluxes:
        out[f"dS_phys_j{j:g}"] = out[f"S_phys_j{j:g}"] - res[0][f"S_phys_j{j:g}"]
        out[f"tradeoff_j{j:g}"] = 0.35 * out["dS_state"] + 0.15 * out[f"dS_phys_j{j:g}"]
    print(f"      dS_state={out['dS_state']:+.4f}  "
          + "  ".join(f"dS_phys(j{j:g})={out[f'dS_phys_j{j:g}']:+.4f}/"
                      f"tradeoff={out[f'tradeoff_j{j:g}']:+.5f}" for j in fluxes),
          flush=True)
    res.append(out)

df = pd.DataFrame(res)
out = os.path.join(le.WORK3, "reports", "gbm_holdout_adjust.csv")
df.to_csv(out, index=False)
print(f"wrote {out}")
