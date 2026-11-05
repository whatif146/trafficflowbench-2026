"""Offline evaluation of soft probability smoothing for the ongoing queue model.

The hard morphological closing of the binary mask lost 0.014 on the LB; a softer
alternative is to smooth the *probabilities* along the corridor (and/or time)
before thresholding. Evaluated on the released train windows (truth available),
in-sample for the shipped boosters, so only the relative ordering of rules is
meaningful.

Rules tested: baseline; spatial smoothing p <- a*p + (1-a)*mean(p_up1,p_dn1)
for a in {0.7,0.8,0.9}; temporal smoothing across the 6 steps with b in {0.7,0.8};
thresholds 0.20/0.25/0.30/0.35.
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


def neighbors_mean(p, L, rank_to_idx):
    order = np.asarray(rank_to_idx, np.int64)
    inv = np.argsort(order)
    q = p[:, order]
    nb = np.zeros_like(q)
    nb[:, 1:-1] = 0.5 * (q[:, :-2] + q[:, 2:])
    nb[:, 0] = q[:, 1]
    nb[:, -1] = q[:, -2]
    return nb[:, inv]


def main():
    cv = json.load(open(os.path.join(LH.OUT, "final_cv_stride1.json")))
    w_ong = float(cv["ong"]["blend_B"]["w"])
    m_ong = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_ongoing.txt"))
    _, v3_ong = SB.load_v3_boosters()
    rules = []
    for th in (0.20, 0.25, 0.30, 0.35):
        rules.append((f"th{th:.2f}_raw", th, None, None))
        for a in (0.9, 0.8, 0.7):
            rules.append((f"th{th:.2f}_sp{a}", th, a, None))
        for b in (0.9, 0.8):
            rules.append((f"th{th:.2f}_tp{b}", th, None, b))
        rules.append((f"th{th:.2f}_sp0.8_tp0.9", th, 0.8, 0.9))
    res = {k: [] for k, _, _, _ in rules}
    for p in LH.PANELS:
        st = LH.static_panel(p)
        L = st["n_links"]
        z = RC.load_rel(p)
        sel = np.where((z["split"] == "train") & (z["condition"] == "queue_ongoing"))[0]
        X3 = z["v3"]
        X5 = RC.rel_X("on", "v5", z)
        P = LH.HORIZON * L
        for j in sel:
            rr = np.arange(j * P, (j + 1) * P)
            a3 = v3_ong.predict(np.ascontiguousarray(X3[rr]))
            b5 = m_ong.predict(np.ascontiguousarray(X5[rr]))
            prob = (w_ong * a3 + (1 - w_ong) * b5).reshape(LH.HORIZON, L)
            truth = z["truth"][j].astype(bool) & z["elig"][j]
            nb = neighbors_mean(prob, L, st["rank_to_idx"])
            nb_t = np.zeros_like(prob)
            nb_t[1:-1] = 0.5 * (prob[:-2] + prob[2:])
            nb_t[0] = prob[1]
            nb_t[-1] = prob[-2]
            for name, th, a, b in rules:
                q = prob
                if a is not None:
                    q = a * q + (1 - a) * nb
                if b is not None:
                    q = b * q + (1 - b) * nb_t
                pred = q >= th
                res[name].append(iou(pred, truth))
        del X3, X5
        z.close()
    base = np.mean(res["th0.25_raw"])
    rows = []
    for name, _, _, _ in rules:
        m = float(np.mean(res[name]))
        rows.append((name, m - base))
    rows.sort(key=lambda x: -x[1])
    print(f"ongoing windows: baseline th0.25 raw IoU={base:.4f}")
    print("top rules by delta (in-sample, relative only):")
    for name, d in rows[:15]:
        print(f"  {name:22s} delta={d:+.4f}")


if __name__ == "__main__":
    main()
