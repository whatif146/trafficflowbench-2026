"""Onset decision-rule experiments: smarter set-size adaptation than a fixed
threshold or fixed top-k.  Parameters chosen on January only; February (protocol
B) and the released June windows (protocol A) are scored once.

The model half is the January-selected v3+v5 probability blend (w=0.5 onset),
retrained here exactly as in run_final_cv.py so rule and model are consistent.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC
import run_final_cv as FC
import run_exp as EX

W_ON = 0.5          # January-selected blend weight from run_final_cv.py
TH0 = 0.30          # reference threshold for the count that drives adaptive k


# --------------------------------------------------------------------------- #
# rule families: each takes a probability vector and returns a boolean mask
# --------------------------------------------------------------------------- #
def _mask_from_k(p, k):
    L = len(p)
    k = int(max(0, min(L, k)))
    m = np.zeros(L, bool)
    if k:
        m[np.argsort(-p)[:k]] = True
    return m


def _mask_th(p, th):
    return p >= th


def rule_th(p, th):
    return _mask_th(p, th)


def rule_th_min1(p, th):
    m = _mask_th(p, th)
    if not m.any():
        m[int(np.argmax(p))] = True
    return m


def rule_th_topm(p, th, mM):
    m = _mask_th(p, th)
    need = mM - int(m.sum())
    if need > 0:
        for j in np.argsort(-p):
            if not m[j]:
                m[j] = True
                need -= 1
                if need == 0:
                    break
    return m


def rule_adaptive_k(p, a, b):
    n30 = int((p >= TH0).sum())
    k = int(round(a + b * n30))
    return _mask_from_k(p, max(1, k))


def rule_sum_alpha(p, alpha):
    k = int(round(alpha * float(p.sum())))
    return _mask_from_k(p, max(1, k))


def rule_count_quantile(p, th_lo, maxk):
    """Keep everything above a low threshold, but cap the set at maxk."""
    m = _mask_th(p, th_lo)
    k = int(m.sum())
    if k > maxk:
        idx = np.argsort(-p[p >= th_lo])[:maxk]
        pos = np.where(p >= th_lo)[0][idx]
        m[:] = False
        m[pos] = True
    return m


KNOWN = {
    "th": [("th", round(float(t), 2)) for t in np.arange(0.15, 0.56, 0.05)],
    "th_min1": [("th_min1", round(float(t), 2)) for t in np.arange(0.15, 0.56, 0.05)],
    "th_topm": [("th_topm", round(float(t), 2), m)
                for t in np.arange(0.20, 0.46, 0.05) for m in (1, 2, 3)],
    "adaptive_k": [("adaptive_k", float(a), float(b))
                   for a in (0.0, 1.0, 2.0, 3.0) for b in (0.25, 0.5, 0.75, 1.0)],
    "sum_alpha": [("sum_alpha", float(al)) for al in (0.4, 0.6, 0.8, 1.0, 1.25, 1.5)],
    "count_cap": [("count_cap", round(float(t), 2), m)
                  for t in (0.15, 0.20, 0.25, 0.30) for m in (2, 3, 4, 5, 6)],
}


def apply_rule(p, spec):
    kind = spec[0]
    if kind == "th":
        return rule_th(p, spec[1])
    if kind == "th_min1":
        return rule_th_min1(p, spec[1])
    if kind == "th_topm":
        return rule_th_topm(p, spec[1], spec[2])
    if kind == "adaptive_k":
        return rule_adaptive_k(p, spec[1], spec[2])
    if kind == "sum_alpha":
        return rule_sum_alpha(p, spec[1])
    if kind == "count_cap":
        return rule_count_quantile(p, spec[1], spec[2])
    raise ValueError(kind)


def eval_jan(probs: dict, loader) -> dict:
    """mean IoU per rule spec over the January block."""
    acc = {spec: [] for fam in KNOWN.values() for spec in fam}
    for p, d in probs.items():
        truth, elig, L = loader(p, d["idx"])
        for j in range(len(d["idx"])):
            pv = d["prob"][j]
            for spec in acc:
                pred = np.zeros((LH.HORIZON, L), bool)
                pred[LH.HORIZON - 1] = apply_rule(pv, spec)
                acc[spec].append(LH.iou(pred, truth[j], elig[j]))
    return {spec: float(np.mean(v)) for spec, v in acc.items() if v}


def eval_feb_june(probs: dict, loader, spec) -> float:
    ious = []
    for p, d in probs.items():
        truth, elig, L = loader(p, d["idx"])
        for j in range(len(d["idx"])):
            pred = np.zeros((LH.HORIZON, L), bool)
            pred[LH.HORIZON - 1] = apply_rule(d["prob"][j], spec)
            ious.append(LH.iou(pred, truth[j], elig[j]))
    return float(np.mean(ious))


def eval_june(jprobs: dict, spec) -> float:
    ious = []
    for p, d in jprobs.items():
        L = d["L"]
        for j in range(len(d["idx"])):
            pred = np.zeros((LH.HORIZON, L), bool)
            pred[LH.HORIZON - 1] = apply_rule(d["prob"][j][LH.HORIZON - 1], spec)
            ious.append(LH.iou(pred, d["truth"][j], d["elig"][j]))
    return float(np.mean(ious))


def main():
    t0 = time.time()
    # ---- models: pre-Jan (protocol B) ----
    probs_jan, probs_feb = {}, {}
    for feat in ("v3", "v5"):
        X, y = FC.gather_lean("on", feat, exclude_june=False)[:2]
        m = FC.train_model("on", feat, X, y, RC.ON_ROUNDS)
        probs_jan[feat] = EX.block_probs("on", FC.COLS[feat], m, "tune")
        probs_feb[feat] = EX.block_probs("on", FC.COLS[feat], m, "feb")
        del m, X, y
        gc.collect()
    pj = EX.blend_probs(probs_jan["v3"], probs_jan["v5"], W_ON)
    pf = EX.blend_probs(probs_feb["v3"], probs_feb["v5"], W_ON)
    loader = lambda p, idx: EX._truth("on", p, idx)  # noqa: E731
    jan = eval_jan(pj, loader)
    # ---- models: pre-Jan minus June (protocol A) ----
    jprobs = {}
    for feat in ("v3", "v5"):
        X, y = FC.gather_lean("on", feat, exclude_june=True)[:2]
        m = FC.train_model("on", feat, X, y, RC.ON_ROUNDS)
        jprobs[feat] = EX.june_probs("on", FC.COLS[feat], m)
        del m, X, y
        gc.collect()
    jb = {}
    for p in jprobs["v3"]:
        a, b = jprobs["v3"][p], jprobs["v5"][p]
        jb[p] = {"idx": a["idx"], "L": a["L"], "truth": a["truth"], "elig": a["elig"],
                 "win": a["win"],
                 "prob": W_ON * a["prob"] + (1.0 - W_ON) * b["prob"]}

    rows = []
    for fam, specs in KNOWN.items():
        scores = {s: jan[s] for s in specs if s in jan}
        best = max(scores, key=scores.get)
        feb = eval_feb_june(pf, loader, best)
        june = eval_june(jb, best)
        rows.append({"family": fam, "spec": str(best), "jan": scores[best],
                     "feb_onset": feb, "june_onset": june,
                     "top3": [(str(s), round(v, 4)) for s, v in
                              sorted(scores.items(), key=lambda kv: -kv[1])[:3]]})
        print(f"{fam:>10}: best {best} | jan {scores[best]:.4f} | FEB onset {feb:.4f} | "
              f"JUNE onset {june:.4f}", flush=True)
    out = {"w_on": W_ON, "th0": TH0, "rows": rows,
           "baseline_blend": {"feb_onset": eval_feb_june(pf, loader, ("th", 0.30)),
                              "june_onset": eval_june(jb, ("th", 0.30))},
           "jan_top": {str(k): v for k, v in sorted(jan.items(), key=lambda kv: -kv[1])[:15]}}
    with open(os.path.join(LH.OUT, "rule_exp.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\nbaseline blend th 0.30: FEB onset "
          f"{out['baseline_blend']['feb_onset']:.4f} | JUNE onset "
          f"{out['baseline_blend']['june_onset']:.4f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
