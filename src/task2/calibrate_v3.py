"""Test per-panel probability calibration for the frozen v3 model.

The v3 artefacts are used unchanged; only the decision threshold is adjusted per
panel, using the January block.  February is scored once, so the comparison with
the global-threshold v3 number is honest.

    python calibrate_v3.py
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
import gbm_predict as GP


def win_iou(pred6, truth6, elig6):
    p = pred6.astype(bool) & elig6
    t = truth6.astype(bool)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / float(union)


def onset_pred(p, L, th):
    pred = np.zeros((C.HORIZON, L), bool)
    pred[C.HORIZON - 1, p >= th] = True
    return pred


def main():
    meta = GP.load_meta("v3")
    idx = [FT.FEATURES.index(n) for n in meta["features"]]
    m_on, m_ong = GP.load_models("v3")
    on, ong = TG.load_dataset()
    # keep the pieces needed for scoring
    for items, kind in ((on, "on"), (ong, "ong")):
        for it in items:
            it["is_tune"] = it["acc"] & (it["date"] >= it["tune_day"]) & (it["date"] < it["cut_day"])
            it["is_eval"] = it["acc"] & (it["date"] >= it["cut_day"])
    probs_on = [m_on.predict(it["X"][:, idx]).astype(np.float32) if len(it["X"]) else np.zeros(0, np.float32) for it in on]
    probs_ong = [m_ong.predict(it["X"][:, idx]).astype(np.float32) if len(it["X"]) else np.zeros(0, np.float32) for it in ong]

    def gather_probs(items, probs, kind, which):
        out = {}
        for i, it in enumerate(items):
            L = it["L"]
            per = L if kind == "on" else 6 * L
            for r in range(len(it["date"])):
                if not it[which][r]:
                    continue
                blk = probs[i][r * per:(r + 1) * per]
                out.setdefault(it["panel"], []).append(
                    (blk.reshape((1, L)) if kind == "on" else blk.reshape(C.HORIZON, L),
                     it["truth"][r], it["elig"][r]))
        return out

    results = {}
    base_on, base_ong = 0.3, 0.25
    # ---- baseline (v3 rules) -------------------------------------------------
    def score(onset_ths: dict, ongoing_ths: dict):
        tot = []
        for key, items, probs, kind, ths, base in (
                ("on", on, probs_on, "on", onset_ths, base_on),
                ("ong", ong, probs_ong, "ong", ongoing_ths, base_ong)):
            for i, it in enumerate(items):
                L = it["L"]
                per = L if kind == "on" else 6 * L
                for r in range(len(it["date"])):
                    if not it["is_eval"][r]:
                        continue
                    p = probs[i][r * per:(r + 1) * per]
                    th = ths.get(it["panel"], base)
                    if kind == "on":
                        pred = onset_pred(p, L, th)
                    else:
                        pred = p.reshape(C.HORIZON, L) >= th
                    tot.append({"panel": it["panel"],
                                "condition": "queue_onset" if kind == "on" else "queue_ongoing",
                                "iou": win_iou(pred, it["truth"][r], it["elig"][r])})
        return pd.DataFrame(tot)

    base_detail = score({}, {})
    agg = C.aggregate_iou(base_detail)
    print(f"baseline v3 (global th): overall {agg['overall']:.4f} onset {agg['onset']:.4f} ongoing {agg['ongoing']:.4f}")

    # ---- per-panel threshold chosen on January -------------------------------
    jan_on = gather_probs(on, probs_on, "on", "is_tune")
    jan_ong = gather_probs(ong, probs_ong, "ong", "is_tune")
    grid = np.arange(0.10, 0.86, 0.05)
    on_ths, ong_ths = {}, {}
    for panel, recs in jan_on.items():
        best, bv = None, -1
        for th in grid:
            vals = [win_iou(onset_pred(p[0], p.shape[1], th), t, e) for p, t, e in recs]
            v = float(np.mean(vals))
            if v > bv:
                best, bv = th, v
        on_ths[panel] = best
    for panel, recs in jan_ong.items():
        best, bv = None, -1
        for th in grid:
            vals = [win_iou(p >= th, t, e) for p, t, e in recs]
            v = float(np.mean(vals))
            if v > bv:
                best, bv = th, v
        ong_ths[panel] = best
    print("per-panel thresholds from January:")
    print("  onset:", {k: round(v, 2) for k, v in on_ths.items()})
    print("  ongoing:", {k: round(v, 2) for k, v in ong_ths.items()})
    detail = score(on_ths, ong_ths)
    agg2 = C.aggregate_iou(detail)
    print(f"per-panel th: overall {agg2['overall']:.4f} onset {agg2['onset']:.4f} ongoing {agg2['ongoing']:.4f}")

    # ---- shrunk per-panel threshold (average with the global) ----------------
    def shrink(ths, base, w=0.5):
        return {k: w * base + (1 - w) * v for k, v in ths.items()}
    for w in (0.3, 0.5, 0.7):
        detail = score(shrink(on_ths, base_on, w), shrink(ong_ths, base_ong, w))
        a = C.aggregate_iou(detail)
        print(f"shrunk w={w}: overall {a['overall']:.4f} onset {a['onset']:.4f} ongoing {a['ongoing']:.4f}")
    base_detail.to_csv(os.path.join(C.OUT2, "calib_baseline.csv"), index=False)


if __name__ == "__main__":
    main()
