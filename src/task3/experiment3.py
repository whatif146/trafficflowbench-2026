"""Experiment 3 — what is the physics ceiling for an honest smooth predictor, and
which post-processing form captures it?

On the calibrated proxy S_LWR punishes the *level error of N=(q/v)*L at the
submitted (target) cells, so the useful adjustment is to replace the candidate's
N at target cells by the best estimate of the (noiseless) observed N, derived
from the surrounding released observations.  Two estimators are compared:

  A. conservation Jacobi correction (adjust.py adjust_panel)
  B. plain temporal smoothing of the accumulation series, observations kept fixed
     (the "smooth N then invert q = N*v/L" recipe)

Candidates: observed truth (oracle), temporal smoothers of the truth (estimate of
the noiseless field -> the ceiling for smooth predictors), historical mean, and
the Lead's smoother train CSV.

    python experiment3.py D7_I10_E <lead_smoother_csv>
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

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
lead_csv = sys.argv[2] if len(sys.argv) > 2 else None
pdata = le.load_panel(panel, "train")
T, L = pdata["T"], pdata["L"]
print(f"[{panel}] T={T} L={L}", flush=True)
length = pdata["p_arr"]["length_km"].astype(np.float64)
lanes = np.clip(pdata["p_arr"]["lanes"], 1.0, None)
flux = le.build_frozen_flux(pdata, "projected", 1, flux_jitter=0.06)
flux0 = le.build_frozen_flux(pdata, "projected", 1, flux_jitter=0.0)


def eval_cand(cs, cf, label):
    res = le.evaluate({panel: pdata}, lambda pd_: (cs, cf), {panel: flux})
    r = res[res.level == "panel"].iloc[0]
    st = ad.task1_state_scores(pdata, cs, cf)
    print(f"  {label:30s} S_state={st['macro']:.4f} S_FD={r.S_FD:.4f} "
          f"S_LWR={r.S_LWR:.4f} S_phys={r.S_physics:.4f}", flush=True)
    return st["macro"], r.S_physics


def smooth_n_adjust(cs, cf, w=5, lam=1.0, max_rel=0.35, max_abs=400.0):
    cand_mask = np.isfinite(cs) & np.isfinite(cf)
    state_speed = np.where(cand_mask, cs, pdata["masked_speed"]).astype(np.float32)
    state_flow = np.where(cand_mask, cf, pdata["masked_flow"]).astype(np.float32)
    N = le.accum_from(state_speed, state_flow, length)
    Nf = le.fill_time_series(N).astype(np.float64)
    Ns = le.moving_average(Nf, w)
    Nnew = np.where(cand_mask & np.isfinite(Ns), N + lam * (Ns - N), N)
    q_new = (Nnew * state_speed.astype(np.float64) / length[None, :]).astype(np.float32)
    delta = np.where(cand_mask, q_new - cf, 0.0)
    delta = np.clip(delta, -max_abs, max_abs)
    rel = np.abs(delta) / np.maximum(np.abs(cf), 1.0)
    delta = np.where(rel > max_rel, np.sign(delta) * max_rel * np.maximum(np.abs(cf), 1.0), delta)
    out = np.where(cand_mask, cf + delta, cf).astype(np.float32)
    return out


print("=== oracle / ceiling ===")
for spec in ("observed", "smooth:3", "smooth:5", "smooth:9", "smooth:15", "historical"):
    cs, cf = le.build_candidate(pdata, spec)
    eval_cand(cs, cf, spec)

candidates = {}
if lead_csv:
    rows = le._read_submission_rows(lead_csv, [panel], "train")
    candidates["lead_smoother"] = le._rows_to_arrays(rows, pdata)
else:
    candidates["noisy:6:0.15"] = le.build_candidate(pdata, "noisy:6:0.15")

print("=== candidates and adjustments ===")
for name, (cs, cf) in candidates.items():
    s0, p0 = eval_cand(cs, cf, name + " base")
    for w, lam in ((3, 0.5), (5, 0.5), (5, 1.0), (9, 0.75), (9, 1.0)):
        cf2 = smooth_n_adjust(cs, cf, w=w, lam=lam)
        s1, p1 = eval_cand(cs, cf2, f"{name} smoothN w={w} lam={lam}")
        print(f"      dS_state={s1 - s0:+.4f} dS_phys={p1 - p0:+.4f} "
              f"tradeoff={0.35 * (s1 - s0) + 0.15 * (p1 - p0):+.5f}", flush=True)
    for strength, iters in ((0.5, 20), (1.0, 40)):
        _, cf3, diag = ad.adjust_panel(pdata, cs, cf, strength=strength, iterations=iters,
                                       flux_mode="projected", flux_window=1)
        s1, p1 = eval_cand(cs, cf3, f"{name} jacobi s={strength} i={iters}")
        print(f"      dS_state={s1 - s0:+.4f} dS_phys={p1 - p0:+.4f} "
              f"tradeoff={0.35 * (s1 - s0) + 0.15 * (p1 - p0):+.5f}", flush=True)
