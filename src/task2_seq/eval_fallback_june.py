"""Offline evaluation of onset fallback variants on the released train windows.

The shipped blend boosters are refit on all train windows, so this is an
in-sample check of *relative* fallback behaviour (not an absolute score). It
measures how often the fallback link is correct and the IoU each rule would get
on the 40 released train onset windows, for:
  baseline th 0.30,
  th + top-1 fallback by blended probability,
  th + top-1 fallback by train climatology (per-link queue rate at the target TOD),
  th + top-1 fallback by argmax(p * rate),
and thresholds 0.30 / 0.40 / 0.50.
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


def slot_of(epoch, offset_steps=6):
    ts = pd.Timestamp(epoch + 300 * offset_steps, unit="s", tz="UTC")
    return (ts.hour * 60 + ts.minute) // 5


def iou(pred, truth):
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    return 1.0 if union == 0 else float(inter) / float(union)


def main():
    cv = json.load(open(os.path.join(LH.OUT, "final_cv_stride1.json")))
    w_on = float(cv["on"]["blend_B"]["w"])
    m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_onset.txt"))
    v3_on, _ = SB.load_v3_boosters()
    rules = []
    for th in (0.50, 0.55, 0.60, 0.65, 0.70):
        rules.append((f"th{th:.2f}_none", th, "none"))
        rules.append((f"th{th:.2f}_fbp", th, "fbp"))
        rules.append((f"th{th:.2f}_fbclim", th, "fbclim"))
        rules.append((f"th{th:.2f}_fbmix", th, "fbmix"))
    res = {k: [] for k, _, _ in rules}
    hits = {k: [] for k, _, _ in rules}
    for p in LH.PANELS:
        st = LH.static_panel(p)
        L = st["n_links"]
        z = RC.load_rel(p)
        sel = np.where((z["split"] == "train") & (z["condition"] == "queue_onset"))[0]
        if not len(sel):
            z.close()
            continue
        X3 = z["v3"]
        X5 = RC.rel_X("on", "v5", z)
        P = LH.HORIZON * L
        for j in sel:
            rr = np.arange(j * P, (j + 1) * P)
            a = v3_on.predict(np.ascontiguousarray(X3[rr]))
            b = m_on.predict(np.ascontiguousarray(X5[rr]))
            prob = (w_on * a + (1 - w_on) * b).reshape(LH.HORIZON, L)
            truth = z["truth"][j, LH.HORIZON - 1].astype(bool) & z["elig"][j, LH.HORIZON - 1]
            pp = prob[LH.HORIZON - 1]
            slot = slot_of(int(z["time"][j]))
            rate = st["rate"][slot]
            for name, th, mode in rules:
                pred = np.zeros(L, bool)
                pred = pp >= th
                if mode != "none" and not pred.any():
                    if mode == "fbp":
                        pred[int(np.argmax(pp))] = True
                    elif mode == "fbclim":
                        pred[int(np.argmax(rate))] = True
                    else:
                        pred[int(np.argmax(pp * rate))] = True
                    hit = bool(pred[truth].any()) if truth.any() else None
                    hits[name].append(hit)
                elif mode != "none":
                    hits[name].append(None)
                res[name].append(iou(pred, truth))
        del X3, X5
        z.close()
    print(f"released train onset windows evaluated; blend w_on={w_on}")
    for name, _, _ in rules:
        v = np.asarray(res[name], dtype=float)
        h = [x for x in hits[name] if x is not None]
        extra = f" fallback_hit={np.mean(h):.2f} (n={len(h)})" if h else ""
        print(f"{name:16s} IoU mean={v.mean():.4f} median={np.median(v):.3f}{extra}")


if __name__ == "__main__":
    main()
