"""Honest CV grid over the v3/v5 blend weight for the onset rule.

The shipped blend uses w=0.5 (onset), selected on the February pseudo-CV only.
Here the same two honest protocols are evaluated over w in {0.3,...,0.8} at
thresholds 0.25/0.30, with and without the top-1 fallback, so a better weight can
be chosen only if it improves BOTH protocols.

Usage: python work/task2_seq/eval_blend_weight_cv.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC


def predict_rule(prob, th, fb):
    if prob.ndim == 1:
        pred = np.zeros((LH.HORIZON, len(prob)), bool)
        pred[LH.HORIZON - 1] = prob >= th
        if fb and not pred[LH.HORIZON - 1].any():
            pred[LH.HORIZON - 1, int(np.argmax(prob))] = True
        return pred
    pred = np.zeros_like(prob, bool)
    pred[LH.HORIZON - 1] = prob[LH.HORIZON - 1] >= th
    if fb and not pred[LH.HORIZON - 1].any():
        pred[LH.HORIZON - 1, int(np.argmax(prob[LH.HORIZON - 1]))] = True
    return pred


def main():
    print("fitting onset models ...", flush=True)
    X, y = RC.gather_train("on", "v5", exclude_june=True, stride=1)
    m5 = RC.train_model("on", "v5", X, y, RC.ON_ROUNDS)
    del X, y
    X, y = RC.gather_train("on", "v3", exclude_june=True, stride=1)
    m3 = RC.train_model("on", "v3", X, y, RC.ON_ROUNDS)
    del X, y
    print("models fitted", flush=True)

    # protocol B: February pseudo-window block
    p3 = RC.pseudo_probs("on", "v3", m3, "eval_feb")
    p5 = RC.pseudo_probs("on", "v5", m5, "eval_feb")
    b = {p: (p3[p]["prob"], p5[p]["prob"], p3[p]["idx"], p3[p]["L"]) for p in p3}

    # protocol A: released June windows
    a = {}
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
        q3 = m3.predict(np.ascontiguousarray(x3[rows])).astype(np.float32).reshape(len(sel), LH.HORIZON, L)
        q5 = m5.predict(np.ascontiguousarray(x5[rows])).astype(np.float32).reshape(len(sel), LH.HORIZON, L)
        a[panel] = (q3, q5, sel, z["truth"][sel], z["elig"][sel], L)
        del x3, x5
        z.close()

    ws = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    rules = [(0.25, 0), (0.25, 1), (0.30, 0), (0.30, 1)]
    rows = []
    for w in ws:
        for th, fb in rules:
            bvals = []
            for panel, (q3, q5, idx, L) in b.items():
                truth, elig, L = RC._panel_truth("on", panel, idx)
                for j in range(len(idx)):
                    prob = w * q3[j] + (1 - w) * q5[j]
                    bvals.append(LH.iou(predict_rule(prob, th, fb), truth[j], elig[j]))
            avals = []
            for panel, (q3, q5, sel, truth, elig, L) in a.items():
                for j in range(q3.shape[0]):
                    prob = w * q3[j] + (1 - w) * q5[j]
                    avals.append(LH.iou(predict_rule(prob, th, fb), truth[j], elig[j]))
            rows.append({"w": w, "th": th, "fb": fb, "B_feb": float(np.mean(bvals)),
                         "A_june": float(np.mean(avals)),
                         "mean": 0.5 * (float(np.mean(bvals)) + float(np.mean(avals)))})
    df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    df.to_csv(os.path.join(LH.OUT, "blend_weight_cv.csv"), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
