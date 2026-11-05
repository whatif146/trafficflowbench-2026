"""Compare fixed-threshold vs top-k onset rules on the released train windows.

In-sample for the shipped boosters (relative comparison only). Rules:
  * threshold p >= th (last step only) for th in {0.45,0.50,0.55,0.60,0.65}
  * global top-k for k in {1,2,3,4,5}
  * per-panel top-k, k = round(mean truth set size of that panel's released
    onset windows)  [the expected-set-size rule]
  * per-window k = round( predicted-ish size ) ... not available without history
    features, so not tested here.
Reports mean IoU and mean predicted set size.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC
import ship_blend as SB

import lightgbm as lgb


def iou(pred, truth):
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def main():
    cv = json.load(open(os.path.join(LH.OUT, "final_cv_stride1.json")))
    w_on = float(cv["on"]["blend_B"]["w"])
    m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_onset.txt"))
    v3_on, _ = SB.load_v3_boosters()
    ths = [0.45, 0.50, 0.55, 0.60, 0.65]
    ks = [1, 2, 3, 4, 5]
    res = {}
    sizes = {}
    per_panel = {}
    for p in LH.PANELS:
        st = LH.static_panel(p)
        L = st["n_links"]
        z = RC.load_rel(p)
        sel = np.where((z["split"] == "train") & (z["condition"] == "queue_onset"))[0]
        X3 = z["v3"]
        X5 = RC.rel_X("on", "v5", z)
        P = LH.HORIZON * L
        probs, truths = [], []
        for j in sel:
            rr = np.arange(j * P, (j + 1) * P)
            a = v3_on.predict(np.ascontiguousarray(X3[rr]))
            b = m_on.predict(np.ascontiguousarray(X5[rr]))
            prob = (w_on * a + (1 - w_on) * b).reshape(LH.HORIZON, L)
            probs.append(prob[LH.HORIZON - 1])
            truths.append(z["truth"][j, LH.HORIZON - 1].astype(bool) & z["elig"][j, LH.HORIZON - 1])
        sizes[p] = float(np.mean([t.sum() for t in truths]))
        per_panel[p] = (probs, truths)
        del X3, X5
        z.close()

    def add(name, value, size):
        res.setdefault(name, []).append(value)
        res.setdefault(name + "__sz", []).append(size)

    for p, (probs, truths) in per_panel.items():
        kp = max(1, int(round(sizes[p])))
        for th in ths:
            for pr, tr in zip(probs, truths):
                pred = pr >= th
                add(f"th{th:.2f}", iou(pred, tr), pred.sum())
        for k in ks:
            for pr, tr in zip(probs, truths):
                pred = np.zeros_like(pr, bool)
                pred[np.argsort(-pr)[:k]] = True
                add(f"top{k}", iou(pred, tr), pred.sum())
        for pr, tr in zip(probs, truths):
            pred = np.zeros_like(pr, bool)
            pred[np.argsort(-pr)[:kp]] = True
            add("topk_panel", iou(pred, tr), pred.sum())
    print("per-panel June truth mean size:", {k: round(v, 2) for k, v in sizes.items()})
    base = float(np.mean(res["th0.50"]))
    rows = []
    for name in {n for n in res if not n.endswith("__sz")}:
        m = float(np.mean(res[name]))
        rows.append((name, m, m - base, float(np.mean(res[name + "__sz"]))))
    rows.sort(key=lambda x: -x[1])
    print(f"{'rule':12s} {'IoU':>7s} {'d_vs_th.50':>9s} {'mean set':>8s}")
    for name, m, d, s in rows:
        print(f"{name:12s} {m:7.4f} {d:+9.4f} {s:8.2f}")


if __name__ == "__main__":
    main()
