"""Second experiment: how much physics headroom is left for a good smooth
predictor (ceiling check), and does the adjuster pay off more when the Task 1
candidate is noisier?

Runs on one small panel to keep RAM low.
"""
import os
import sys

import numpy as np

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "libs"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task3"))
import local_eval as le
import adjust as ad

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I210_W"
pdata = le.load_panel(panel, "train")
print(f"[{panel}] T={pdata['T']} L={pdata['L']}", flush=True)
flux = le.build_frozen_flux(pdata, "projected", 1, flux_jitter=0.06)


def eval_cand(cs, cf, label):
    res = le.evaluate({panel: pdata}, lambda pd_: (cs, cf), {panel: flux})
    r = res[res.level == "panel"].iloc[0]
    st = ad.task1_state_scores(pdata, cs, cf)
    print(f"  {label:22s} S_state={st['macro']:.4f} S_FD={r.S_FD:.4f} S_LWR={r.S_LWR:.4f} "
          f"S_phys={r.S_physics:.4f}", flush=True)
    return st["macro"], r.S_physics


print("=== ceiling / ordering ===")
for spec in ("observed", "smooth:3", "smooth:9", "historical"):
    cs, cf = le.build_candidate(pdata, spec)
    eval_cand(cs, cf, spec)

print("=== adjuster value vs candidate noise ===")
for spec in ("noisy:2:0.04", "noisy:5:0.10"):
    cs, cf = le.build_candidate(pdata, spec)
    s0, p0 = eval_cand(cs, cf, spec + " base")
    for strength, iters in ((0.5, 20), (1.0, 40)):
        _, cf2, diag = ad.adjust_panel(pdata, cs, cf, strength=strength, iterations=iters,
                                       flux_mode="projected", flux_window=1)
        s1, p1 = eval_cand(cs, cf2, f"{spec} adj s={strength} i={iters}")
        print(f"      dS_state={s1 - s0:+.4f} dS_phys={p1 - p0:+.4f} "
              f"tradeoff={0.35 * (s1 - s0) + 0.15 * (p1 - p0):+.5f}", flush=True)
