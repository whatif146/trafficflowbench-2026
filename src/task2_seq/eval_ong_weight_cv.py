"""Honest CV grid over the v3/v5 blend weight for the ONGOING queue model.

The shipped blend uses w_ong=0.6 (selected on the February pseudo-CV). This script
evaluates w_ong in {0.2,...,0.8} on the same two honest protocols as run_cv.py
(B = untouched Feb pseudo-window block, A = released June windows; models refit
with every June origin excluded) at thresholds 0.20/0.25/0.30.

Stride 2 is used to keep memory bounded; only the relative ordering matters.

Usage: python work/task2_seq/eval_ong_weight_cv.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC


def main():
    print("fitting ongoing models (stride 2) ...", flush=True)
    X, y = RC.gather_train("ong", "v5", exclude_june=True, stride=2)
    print("v5 rows", X.shape, flush=True)
    m5 = RC.train_model("ong", "v5", X, y, RC.ONG_ROUNDS)
    del X, y
    X, y = RC.gather_train("ong", "v3", exclude_june=True, stride=2)
    print("v3 rows", X.shape, flush=True)
    m3 = RC.train_model("ong", "v3", X, y, RC.ONG_ROUNDS)
    del X, y
    print("models fitted", flush=True)

    # protocol B
    p3 = RC.pseudo_probs("ong", "v3", m3, "eval_feb")
    p5 = RC.pseudo_probs("ong", "v5", m5, "eval_feb")
    # protocol A
    a = {}
    for panel in LH.PANELS:
        z = RC.load_rel(panel)
        sel = np.where((z["split"] == "train") & (z["condition"] == "queue_ongoing"))[0]
        if not len(sel):
            z.close()
            continue
        L = int(z["n_links"])
        per = LH.HORIZON * L
        x3 = RC.rel_X("ong", "v3", z)
        x5 = RC.rel_X("ong", "v5", z)
        rows = (sel[:, None] * per + np.arange(per)[None, :]).reshape(-1)
        q3 = m3.predict(np.ascontiguousarray(x3[rows])).astype(np.float32).reshape(len(sel), LH.HORIZON, L)
        q5 = m5.predict(np.ascontiguousarray(x5[rows])).astype(np.float32).reshape(len(sel), LH.HORIZON, L)
        a[panel] = (q3, q5, z["truth"][sel], z["elig"][sel], L)
        del x3, x5
        z.close()

    rows = []
    for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        for th in (0.20, 0.25, 0.30):
            bv = []
            for panel, d in p3.items():
                q3 = p3[panel]["prob"].reshape(len(p3[panel]["idx"]), LH.HORIZON, p3[panel]["L"])
                q5 = p5[panel]["prob"].reshape(len(p5[panel]["idx"]), LH.HORIZON, p5[panel]["L"])
                truth, elig, L = RC._panel_truth("ong", panel, p3[panel]["idx"])
                for j in range(q3.shape[0]):
                    prob = w * q3[j] + (1 - w) * q5[j]
                    bv.append(LH.iou(prob >= th, truth[j], elig[j]))
            av = []
            for panel, (q3, q5, truth, elig, L) in a.items():
                for j in range(q3.shape[0]):
                    prob = w * q3[j] + (1 - w) * q5[j]
                    av.append(LH.iou(prob >= th, truth[j], elig[j]))
            rows.append({"w_ong": w, "th": th, "B_feb": float(np.mean(bv)),
                         "A_june": float(np.mean(av)),
                         "mean": 0.5 * (float(np.mean(bv)) + float(np.mean(av)))})
    df = pd.DataFrame(rows).sort_values("mean", ascending=False)
    df.to_csv(os.path.join(LH.OUT, "ong_weight_cv.csv"), index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
