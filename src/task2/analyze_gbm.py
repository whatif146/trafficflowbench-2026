"""Diagnostics for the GBM models on the held-out February block.

Reports, per condition, how the decision rule (probability threshold / top-k)
trades off, and how much of the loss is intrinsic (an oracle rule over the same
probabilities).  Also reports the distribution of truth-set sizes per window so
the predicted set size can be calibrated.
"""
from __future__ import annotations

import json
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C
import features as FT
import train_gbm as TG


def win_iou(pred6, truth6, elig6):
    p = pred6.astype(bool) & elig6
    t = truth6.astype(bool)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / float(union)


def main():
    import lightgbm as lgb
    on, ong = TG.load_dataset()
    m_on = lgb.Booster(model_file=os.path.join(C.MODEL2, "gbm_onset_pre.txt"))
    m_ong = lgb.Booster(model_file=os.path.join(C.MODEL2, "gbm_ongoing_pre.txt"))
    rows = []
    # ---- onset ---------------------------------------------------------
    onset_rules = [("th", round(th, 2)) for th in np.arange(0.05, 0.96, 0.05)] + \
                  [("top", k) for k in range(1, 16)]
    per_rule = {r: [] for r in onset_rules}
    oracle = []
    for i, it in enumerate(on):
        L = it["L"]
        d = it["date"]
        ev = np.where(d >= it["cut_day"])[0]
        if len(ev) == 0:
            continue
        probs = m_on.predict(it["X"]).astype(np.float32) if len(it["X"]) else np.zeros(0)
        for r in ev:
            p = probs[r * L:(r + 1) * L]
            truth, elig = it["truth"][r], it["elig"][r]
            best = -1.0
            for rule in onset_rules:
                kind, val = rule
                pred = np.zeros((C.HORIZON, L), bool)
                if kind == "th":
                    pred[C.HORIZON - 1, p >= val] = True
                else:
                    pred[C.HORIZON - 1, np.argsort(-p)[:min(int(val), L)]] = True
                v = win_iou(pred, truth, elig)
                per_rule[rule].append(v)
                best = max(best, v)
            oracle.append(best)
            rows.append({"panel": it["panel"], "condition": "queue_onset",
                         "truth_n": int(truth.sum()), "pred_n_th10": int((p >= 0.1).sum()),
                         "oracle_rule_iou": best})
    print("== ONSET (Feb) rule sweep, mean IoU over windows ==")
    sweep = sorted(((r, float(np.mean(v))) for r, v in per_rule.items()), key=lambda kv: -kv[1])
    for r, v in sweep[:12]:
        print(f"  {r}: {v:.4f}")
    print(f"  best-rule-per-window oracle: {np.mean(oracle):.4f}")
    on_df = pd.DataFrame([x for x in rows if x["condition"] == "queue_onset"])
    print("  truth size describe:", on_df.truth_n.describe().round(2).to_dict())
    print("  mean pred_n at th=0.1:", on_df.pred_n_th10.mean().round(2))
    per_panel = on_df.groupby("panel")[["truth_n", "pred_n_th10", "oracle_rule_iou"]].mean().round(2)
    print(per_panel.to_string())

    # ---- ongoing -------------------------------------------------------
    ong_rules = [round(th, 2) for th in np.arange(0.1, 0.86, 0.05)]
    per_rule_ong = {th: [] for th in ong_rules}
    ong_rows = []
    for i, it in enumerate(ong):
        L = it["L"]
        d = it["date"]
        ev = np.where(d >= it["cut_day"])[0]
        if len(ev) == 0:
            continue
        probs = m_ong.predict(it["X"]).astype(np.float32) if len(it["X"]) else np.zeros(0)
        for r in ev:
            p = probs[r * 6 * L:(r + 1) * 6 * L].reshape(C.HORIZON, L)
            truth, elig = it["truth"][r], it["elig"][r]
            best = -1
            for th in ong_rules:
                pred = p >= th
                v = win_iou(pred, truth, elig)
                per_rule_ong[th].append(v)
                best = max(best, v)
            ong_rows.append({"panel": it["panel"], "truth_n": int(truth.sum()),
                             "pred_n_th25": int((p >= 0.25).sum()), "oracle": best})
    print("\n== ONGOING (Feb) threshold sweep ==")
    for th, v in sorted(per_rule_ong.items(), key=lambda kv: -np.mean(kv[1]))[:8]:
        print(f"  th {th}: {np.mean(v):.4f}")
    print(f"  best-th-per-window oracle: {np.mean([r['oracle'] for r in ong_rows]):.4f}")
    ong_df = pd.DataFrame(ong_rows)
    print("  truth_n:", ong_df.truth_n.describe().round(1).to_dict())
    print(ong_df.groupby("panel")[["truth_n", "pred_n_th25", "oracle"]].mean().round(1).to_string())


if __name__ == "__main__":
    main()
