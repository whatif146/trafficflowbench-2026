"""Calibrate the boundary-flux proxy against the two published official anchors.

Anchors (docs/RUN_LOCAL.md, docs/BASELINES.md):
  * perfect answer : S_physics 0.9636 (10-panel validation mean), 0.9598 on
                     D12_I5_N validation.  S_FD is computable exactly from the
                     observations, so the organizer-flux S_LWR of an exact answer
                     is ~0.95 (implied 0.947-0.955 on the panels probed).
  * naive baseline : S_physics 0.3549 (validation mean), 0.3242 on D12_I5_N.

A faithful frozen-flux proxy must give S_LWR(observed truth) ~ 0.95 and
S_LWR(historical mean) ~ 0.05 while keeping the right ordering for every
candidate (truth > good reconstruction > historical > noisy > lagged).

Runs one panel at a time and frees it before loading the next (RAM is tight).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "libs"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task3"))
import local_eval as le

CANDIDATES = ["observed", "historical", "noisy:5:0.10", "lagged:1"]
WINDOWS = (3, 5, 9, 15, 21)


def flux_variants():
    out = [("raw", {"kind": "raw"}), ("fallback", {"kind": "fallback"})]
    for w in WINDOWS:
        out.append((f"smooth{w}", {"kind": "smooth", "window": w}))
        out.append((f"proj{w}", {"kind": "projected", "window": w}))
        out.append((f"blend{w}", {"kind": "blend", "window": w, "blend_lambda": 0.5}))
    return out


def main():
    panels = sys.argv[1:] or ["D7_I210_W", "D7_I10_E"]
    variants = flux_variants()
    rows = []
    for panel in panels:
        if not le._panel_cache_ready(panel, "train"):
            print(f"skip {panel}: cache not ready")
            continue
        pdata = le.load_panel(panel, "train")
        print(f"[{panel}] T={pdata['T']} L={pdata['L']}", flush=True)
        cand_arrays = {c: le.build_candidate(pdata, c) for c in CANDIDATES}
        for name, kw in variants:
            flux = ({"kind": "fallback"} if name == "fallback"
                    else le.build_frozen_flux(pdata, **kw))
            line = []
            for cand in CANDIDATES:
                def fn(pd_, cand=cand):
                    return cand_arrays[cand]
                res = le.evaluate({panel: pdata}, fn, {panel: flux})
                r = res[res.level == "panel"].iloc[0]
                rows.append({"panel": panel, "flux": name, "candidate": cand,
                             "S_FD": r.S_FD, "S_LWR": r.S_LWR, "E_LWR": r.E_LWR,
                             "S_physics": r.S_physics})
                line.append(f"{cand[:6]}:FD={r.S_FD:.3f},LWR={r.S_LWR:.3f}")
            print(f"  {name:9s} " + " | ".join(line), flush=True)
            del flux
        del pdata, cand_arrays
    df = pd.DataFrame(rows)
    out = os.path.join(le.WORK3, "reports", "calibrate_flux.csv")
    df.to_csv(out, index=False)
    for cand in CANDIDATES:
        sub = df[df.candidate == cand]
        print(f"\n=== candidate {cand}: S_LWR by flux ===")
        print(sub.pivot(index="panel", columns="flux", values="S_LWR").round(4).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
