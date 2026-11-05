"""End-to-end check: frozen v3 models on the task2_seq released-window features.

Reproduces the v3 "released-train-window proxy" (0.879) from the long-history
pipeline's rel_*.npz v3 feature block.  Agreement validates that the feature
function and truth convention in lhcore/build_released match work/task2.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import lightgbm as lgb

META = json.load(open(os.path.join(LH.WORK, "task2", "models", "v3", "gbm_meta.json")))
m_on = lgb.Booster(model_file=os.path.join(LH.WORK, "task2", "models", "v3", "gbm_onset.txt"))
m_ong = lgb.Booster(model_file=os.path.join(LH.WORK, "task2", "models", "v3", "gbm_ongoing.txt"))
on_rule = META["onset_rule"]
ong_th = float(META["ongoing_th"])
print("v3 rule", on_rule, ong_th)

rows = []
for p in LH.PANELS:
    z = np.load(os.path.join(LH.CACHE, f"rel_{p}.npz"), allow_pickle=False)
    sel = z["split"] == "train"
    X = z["v3"]
    truth, elig = z["truth"], z["elig"]
    L = int(z["n_links"])
    cond = z["condition"]
    P = 6 * L
    prob = np.zeros((len(X),), np.float32)
    for j in np.where(sel & (cond == "queue_onset"))[0]:
        sl = slice(j * P + 5 * L, j * P + 6 * L)
        prob[sl] = m_on.predict(X[sl])
    for j in np.where(sel & (cond == "queue_ongoing"))[0]:
        prob[j * P:(j + 1) * P] = m_ong.predict(X[j * P:(j + 1) * P])
    for j in np.where(sel)[0]:
        pp = prob[j * P:(j + 1) * P]
        if cond[j] == "queue_onset":
            pred = np.zeros((6, L), bool)
            pred[5, pp[5 * L:] >= on_rule[1]] = True
        else:
            pred = pp.reshape(6, L) >= ong_th
        rows.append({"panel": p, "condition": cond[j], "window_id": str(z["window_ids"][j]),
                     "iou": LH.iou(pred, truth[j], elig[j]), "truth_n": int(truth[j].sum())})
    z.close()

d = pd.DataFrame(rows)
agg = LH.aggregate(d)
print(f"released-train proxy with frozen v3: overall {agg['overall']:.4f} | "
      f"onset {agg['onset']:.4f} | ongoing {agg['ongoing']:.4f}")
print("per family:", {k: round(v, 3) for k, v in agg["per_family"].items()})
print(d.groupby("condition").iou.agg(["mean", "count"]).round(3).to_string())
