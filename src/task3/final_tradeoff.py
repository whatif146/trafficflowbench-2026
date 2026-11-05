"""Final trade-off report for the delivered adjuster settings.

Reports on train holdout, for each candidate:
  base vs adjusted  ->  S_state (Task 1 formula), S_FD / S_LWR / S_physics under
  the calibrated proxy with residual levels 0 / 0.06 / 0.12 (0.06 is the
  anchor-calibrated setting), plus the competition trade-off
  0.35*dS_state + 0.15*dS_physics, and adjustment diagnostics.

Two candidates are used:
  * the Lead's real smoother on D7_I10_E (train CSV)
  * a synthetic noisy reconstruction on a second panel (robustness / lower bound)
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

METHOD, WINDOW, STRENGTH = "median", 5, 0.75
JITTERS = (0.0, 0.06, 0.12)
rows = []


def measure(panel, cs, cf, label):
    csm, cfm, diag = ad.adjust_panel(pd_data, cs, cf, method=METHOD,
                                     window=WINDOW, strength=STRENGTH)
    st0 = ad.task1_state_scores(pd_data, cs, cf)
    st1 = ad.task1_state_scores(pd_data, csm, cfm)
    out = {"panel": panel, "candidate": label,
           "S_state_base": st0["macro"], "S_state_adj": st1["macro"],
           "dS_state": st1["macro"] - st0["macro"],
           "mean_abs_dq": diag.get("mean_abs_dq", np.nan),
           "n_adjusted": diag.get("n_adjusted", 0)}
    for j in JITTERS:
        flux = le.build_frozen_flux(pd_data, "projected", 1, flux_jitter=j)
        r0 = le.evaluate({panel: pd_data}, lambda pd_: (cs, cf), {panel: flux})
        r0 = r0[r0.level == "panel"].iloc[0]
        r1 = le.evaluate({panel: pd_data}, lambda pd_: (csm, cfm), {panel: flux})
        r1 = r1[r1.level == "panel"].iloc[0]
        tag = f"j{j:g}"
        out[f"S_FD_base_{tag}"] = r0.S_FD
        out[f"S_FD_adj_{tag}"] = r1.S_FD
        out[f"S_LWR_base_{tag}"] = r0.S_LWR
        out[f"S_LWR_adj_{tag}"] = r1.S_LWR
        out[f"S_phys_base_{tag}"] = r0.S_physics
        out[f"S_phys_adj_{tag}"] = r1.S_physics
        out[f"dS_phys_{tag}"] = r1.S_physics - r0.S_physics
        out[f"tradeoff_{tag}"] = 0.35 * out["dS_state"] + 0.15 * out[f"dS_phys_{tag}"]
    rows.append(out)
    print(f"\n[{panel}] {label}: S_state {st0['macro']:.4f} -> {st1['macro']:.4f} "
          f"(dS {out['dS_state']:+.4f}); adjusted {out['n_adjusted']:,} cells, "
          f"mean|dq|={out['mean_abs_dq']:.2f} vph")
    for j in JITTERS:
        tag = f"j{j:g}"
        print(f"   {tag:6s} S_FD {out[f'S_FD_base_{tag}']:.4f}->{out[f'S_FD_adj_{tag}']:.4f}  "
              f"S_LWR {out[f'S_LWR_base_{tag}']:.4f}->{out[f'S_LWR_adj_{tag}']:.4f}  "
              f"S_phys {out[f'S_phys_base_{tag}']:.4f}->{out[f'S_phys_adj_{tag}']:.4f}  "
              f"dS_phys={out[f'dS_phys_{tag}']:+.4f}  tradeoff={out[f'tradeoff_{tag}']:+.5f}")


# candidate 1: the Lead's real smoother, D7_I10_E train
panel = "D7_I10_E"
pd_data = le.load_panel(panel, "train")
csv = os.path.join(ROOT, "work", "submissions", "state_train_D7_I10_E.csv")
rows_csv = le._read_submission_rows(csv, [panel], "train")
cs, cf = le._rows_to_arrays(rows_csv, pd_data)
measure(panel, cs, cf, "lead_smoother_train")
del pd_data, cs, cf

# candidate 2: synthetic noisy reconstruction on a second (small) panel
panel = "D7_I210_W"
pd_data = le.load_panel(panel, "train")
cs, cf = le.build_candidate(pd_data, "noisy:3:0.06")
measure(panel, cs, cf, "noisy:3:0.06")
del pd_data, cs, cf

df = pd.DataFrame(rows)
out = os.path.join(le.WORK3, "reports", "adjust_final_tradeoff.csv")
df.to_csv(out, index=False)
print(f"\nwrote {out}")
