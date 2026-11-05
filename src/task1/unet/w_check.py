"""Blend-weight edge check: S_state of w in {0.5,0.55,0.6,0.65,0.7,0.8} on the
4 holdout panels (single seed 42). Cheap: no physics proxy, just re-blending the
saved predictions.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "unet"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "gbm"))

import unetlib as U  # noqa: E402
from t1lib import score  # noqa: E402
from run_holdout import load_gbm_rows, GBM_DIR  # noqa: E402

WS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
PANELS = ["D7_I210_W", "D7_I405_S", "D7_I10_E", "D12_I5_N"]
res = {}
for panel in PANELS:
    c = U.load_cache(panel, "train_hold60")
    pz = np.load(os.path.join(U.HERE, f"pred_unet_{panel}_u80.npz"))
    hold_rows = pz["rows"]
    _, tpl_pos, ps, pq = load_gbm_rows(panel, c, os.path.join(GBM_DIR, f"holdout_{panel}.csv"))
    tv, tq, lanes, reg = U.truth_rows(c, tpl_pos)
    ix = np.searchsorted(hold_rows, tpl_pos)
    vu, qu = pz["v"][ix], pz["q"][ix]
    for w in WS:
        vs = w * ps + (1 - w) * vu
        vq = w * pq + (1 - w) * qu
        r = score(tv, tq, vs, vq, lanes, reg)
        res.setdefault(w, {})[panel] = r["macro"]
    del c
print("S_state by blend weight (single seed 42):")
print("panel      " + "".join(f"{w:>8.2f}" for w in WS))
for panel in PANELS:
    print(f"{panel:11s}" + "".join(f"{res[w][panel]:8.4f}" for w in WS))
print(f"{'mean':11s}" + "".join(f"{np.mean([res[w][p] for p in PANELS]):8.4f}" for w in WS))
base = np.mean([res[0.6][p] for p in PANELS])
print("\ndelta vs w=0.6: " + "  ".join(
    f"w={w:.2f}: {np.mean([res[w][p] for p in PANELS]) - base:+.4f}" for w in WS))
