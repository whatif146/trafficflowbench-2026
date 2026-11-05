"""Experiment 4 — pick the post-processing estimator / window / strength.

Candidate: a real Task 1 train submission (or synthetic corruption).  Reports
dS_state (Task 1 formula) and dS_physics (calibrated proxy) for a grid of
local-accumulation estimators, plus the competition-weighted trade-off.

    python experiment4.py D7_I10_E <train_submission_csv>
"""
import os
import sys

import numpy as np

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
for p in (os.path.join(ROOT, "libs"), os.path.join(ROOT, "work", "lib"),
          os.path.join(ROOT, "work", "task3")):
    sys.path.insert(0, p)
import local_eval as le
import adjust as ad

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
csv = sys.argv[2] if len(sys.argv) > 2 else None
pdata = le.load_panel(panel, "train")
print(f"[{panel}] T={pdata['T']} L={pdata['L']}", flush=True)
fluxes = {"cal": le.build_frozen_flux(pdata, "projected", 1, flux_jitter=0.06),
          "j0": le.build_frozen_flux(pdata, "projected", 1, flux_jitter=0.0)}
if csv:
    rows = le._read_submission_rows(csv, [panel], "train")
    cs, cf = le._rows_to_arrays(rows, pdata)
    name = os.path.basename(csv)
else:
    cs, cf = le.build_candidate(pdata, "noisy:4:0.08")
    name = "noisy:4:0.08"


def ev(cs_, cf_, label):
    res = le.evaluate({panel: pdata}, lambda pd_: (cs_, cf_), {panel: fluxes["cal"]})
    r = res[res.level == "panel"].iloc[0]
    st = ad.task1_state_scores(pdata, cs_, cf_)
    print(f"  {label:34s} S_state={st['macro']:.4f} S_FD={r.S_FD:.4f} "
          f"S_LWR={r.S_LWR:.4f} S_phys={r.S_physics:.4f}", flush=True)
    return st["macro"], r.S_physics


s0, p0 = ev(cs, cf, name + " base")
grid = []
for m, w in (("median", 3), ("median", 5), ("median", 7),
             ("mean", 3), ("mean", 5), ("linear", 5), ("linear", 7)):
    for s in (0.5, 0.75, 1.0):
        grid.append((m, w, s))
best = None
for m, w, s in grid:
    _, cf2, diag = ad.adjust_panel(pdata, cs, cf, method=m, window=w, strength=s)
    s1, p1 = ev(cs, cf2, f"{m} w={w} s={s}")
    trade = 0.35 * (s1 - s0) + 0.15 * (p1 - p0)
    print(f"      dS_state={s1 - s0:+.4f} dS_phys={p1 - p0:+.4f} tradeoff={trade:+.5f}", flush=True)
    if best is None or trade > best[0]:
        best = (trade, m, w, s, s1 - s0, p1 - p0)
print(f"\nBEST tradeoff={best[0]:+.5f} at method={best[1]} window={best[2]} strength={best[3]} "
      f"(dS_state={best[4]:+.4f}, dS_phys={best[5]:+.4f})")
