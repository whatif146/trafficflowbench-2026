"""Honest CV of onset THRESHOLD + FALLBACK rules (the family queued for LB probes).

Fits the v3 and long-history (v5) onset models on the same pseudo-window data as
run_cv.py (origin before 2031-01-01, every June-2030 origin excluded), blends the
two probability vectors at w=0.5 (the shipped blend), and scores threshold and
threshold+top-k-fallback rules on:
  B: the untouched February pseudo-window block (honest, out-of-sample month), and
  A: the 80 released June-2030 train windows (out-of-sample origins).
Reports mean IoU per rule so the LB probe order can be chosen from train evidence.

Usage: python work/task2_seq/eval_fallback_cv.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC


def predict_rules(prob, th, fb):
    """prob is (HORIZON, L) for released windows or (L,) for pseudo windows."""
    if prob.ndim == 1:
        pred = np.zeros((LH.HORIZON, len(prob)), bool)
        pred[LH.HORIZON - 1] = prob >= th
        if fb and not pred[LH.HORIZON - 1].any():
            k = min(int(fb), len(prob))
            pred[LH.HORIZON - 1, np.argsort(-prob)[:k]] = True
        return pred
    pred = np.zeros_like(prob, bool)
    pred[LH.HORIZON - 1] = prob[LH.HORIZON - 1] >= th
    if fb and not pred[LH.HORIZON - 1].any():
        k = min(int(fb), len(prob[LH.HORIZON - 1]))
        pred[LH.HORIZON - 1, np.argsort(-prob[LH.HORIZON - 1])[:k]] = True
    return pred


THS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
RULES = [(f"th{t:.3f}_none", t, 0) for t in THS] + \
        [(f"th{t:.3f}_fb1", t, 1) for t in THS] + \
        [(f"th{t:.3f}_fb2", t, 2) for t in THS if t >= 0.50]


def probs_pseudo(model3, model5, where):
    p3 = RC.pseudo_probs("on", "v3", model3, where)
    p5 = RC.pseudo_probs("on", "v5", model5, where)
    out = {}
    for panel in p3:
        out[panel] = {
            "idx": p3[panel]["idx"],
            "prob": 0.5 * p3[panel]["prob"] + 0.5 * p5[panel]["prob"],
            "L": p3[panel]["L"],
        }
    return out


def probs_released(model3, model5):
    out = {}
    for panel in LH.PANELS:
        z = RC.load_rel(panel)
        sel = np.where((z["split"] == "train") & (z["condition"] == "queue_onset"))[0]
        if not len(sel):
            z.close()
            continue
        L = int(z["n_links"])
        per = LH.HORIZON * L
        x3 = RC.rel_X("on", "v3", z)
        x5 = RC.rel_X("on", "v5", z)
        rows = (sel[:, None] * per + np.arange(per)[None, :]).reshape(-1)
        b3 = model3.predict(np.ascontiguousarray(x3[rows]))
        b5 = model5.predict(np.ascontiguousarray(x5[rows]))
        prob = (0.5 * b3 + 0.5 * b5).astype(np.float32).reshape(len(sel), LH.HORIZON, L)
        out[panel] = {"sel": sel, "prob": prob, "truth": z["truth"][sel],
                      "elig": z["elig"][sel], "L": L}
        del x3, x5
        z.close()
    return out


def main():
    print("fitting onset models (same protocol as run_cv) ...", flush=True)
    X, y = RC.gather_train("on", "v5", exclude_june=True, stride=1)
    m5 = RC.train_model("on", "v5", X, y, RC.ON_ROUNDS)
    del X, y
    X, y = RC.gather_train("on", "v3", exclude_june=True, stride=1)
    m3 = RC.train_model("on", "v3", X, y, RC.ON_ROUNDS)
    del X, y
    print("models fitted", flush=True)

    results = {}
    # protocol B
    pb = probs_pseudo(m3, m5, "eval_feb")
    acc = {name: [] for name, _, _ in RULES}
    for panel, d in pb.items():
        truth, elig, L = RC._panel_truth("on", panel, d["idx"])
        for j in range(len(d["idx"])):
            for name, th, fb in RULES:
                pred = predict_rules(d["prob"][j], th, fb)
                acc[name].append(LH.iou(pred, truth[j], elig[j]))
    results["B_feb"] = {k: float(np.mean(v)) for k, v in acc.items()}
    del pb

    # protocol A
    pa = probs_released(m3, m5)
    acc = {name: [] for name, _, _ in RULES}
    for panel, d in pa.items():
        for j in range(d["prob"].shape[0]):
            for name, th, fb in RULES:
                pred = predict_rules(d["prob"][j], th, fb)
                acc[name].append(LH.iou(pred, d["truth"][j], d["elig"][j]))
    results["A_june"] = {k: float(np.mean(v)) for k, v in acc.items()}
    del pa

    df = pd.DataFrame(results)
    df["mean"] = df.mean(axis=1)
    df = df.sort_values("mean", ascending=False)
    df.to_csv(os.path.join(LH.OUT, "fallback_cv.csv"))
    print(df.round(4).to_string())
    print("\nbaseline shipped rule th0.30 none:")
    print(df.loc["th0.300_none"] if "th0.300_none" in df.index else df.head())


if __name__ == "__main__":
    main()
