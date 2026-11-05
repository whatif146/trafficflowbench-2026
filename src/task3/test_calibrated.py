"""Quick check of the calibrated frozen-flux proxy on one small panel:
rhs = exact projection onto released observations (window 1) + 6% calibrated residual.
"""
import os
import sys

import numpy as np

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "libs"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task3"))
import local_eval as le

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I210_W"
pdata = le.load_panel(panel, "train")
print(f"[{panel}] T={pdata['T']} L={pdata['L']}")
cands = ["observed", "historical", "noisy:5:0.10", "lagged:1", "smooth:5"]
arrs = {c: le.build_candidate(pdata, c) for c in cands}
for jitter in (0.0, 0.06):
    flux = le.build_frozen_flux(pdata, "projected", 1, flux_jitter=jitter)
    print(f"\nflux=projected/obs-projection window=1 jitter={jitter}")
    for c in cands:
        res = le.evaluate({panel: pdata}, lambda pd_, c=c: arrs[c], {panel: flux})
        r = res[res.level == "panel"].iloc[0]
        print(f"  {c:16s} S_FD={r.S_FD:.4f} S_LWR={r.S_LWR:.4f} E={r.E_LWR:.4f} S_physics={r.S_physics:.4f}")
