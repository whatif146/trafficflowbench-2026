"""Analyse the onset-step6 queue set structure and test static predictors.

The scorer keeps only eligible truth cells and drops predictions on cells that
are not in the truth file, so predicting a *superset* of the true queue set is
cheap only where the extra cells are ineligible; predicting the exact queue set
gives IoU 1.0.  This script measures how stable that set is.
"""
from __future__ import annotations

import gc
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C

CUT = pd.Timestamp("2031-02-01", tz="UTC")


def iou_masked(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> float:
    p = pred.astype(bool) & elig
    t = truth.astype(bool)
    union = float(np.logical_or(p, t).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / union


def analyse(panel: str, cut=CUT):
    d = C.load_train_panel(panel)
    cand = C.candidate_table(d)
    acc = C.greedy_select(cand).sort_values("time_dt").reset_index(drop=True)
    onset = acc[acc.condition == "queue_onset"]
    fit = onset[onset.time_dt < cut]
    ev = onset[onset.time_dt >= cut]
    step = C.HORIZON - 1
    L = d["n_links"]
    # raw queue set at step 6 for fit windows
    raw = np.zeros((len(fit), L), np.int8)
    for j, r in enumerate(fit.itertuples(index=False)):
        raw[j] = d["Qobs"][r.ti + 1 + step]
    freq_raw = raw.mean(axis=0)
    sets_fit = [tuple(np.where(raw[j] > 0)[0].tolist()) for j in range(len(fit))]
    from collections import Counter
    cnt = Counter(sets_fit)
    modal_set, modal_n = (cnt.most_common(1)[0] if cnt else ((), 0))
    core = np.where(freq_raw >= 0.5)[0]
    # eligible-masked truth for eval
    truth = []
    for r in ev.itertuples(index=False):
        truth.append((d["Q"][r.ti + 1:r.ti + 1 + C.HORIZON],
                      d["elig"][r.ti + 1:r.ti + 1 + C.HORIZON]))
    res = {"panel": panel, "n_fit": len(fit), "n_ev": len(ev),
           "raw_size_fit_mean": float(np.mean([len(s) for s in sets_fit])) if sets_fit else np.nan,
           "modal_set_size": len(modal_set), "modal_set_share": modal_n / max(len(fit), 1),
           "core_set_size": len(core),
           "raw_size_ev_mean": float(np.mean([int(d["Qobs"][r.ti + 1 + step].sum()) for r in ev.itertuples(index=False)])) if len(ev) else np.nan}

    def score_links(links, name):
        pred = np.zeros((C.HORIZON, L), bool)
        pred[step, list(links)] = True
        vals = [iou_masked(pred, t, e) for t, e in truth]
        res[name] = float(np.mean(vals)) if vals else np.nan
        return res[name]

    # strategies
    for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12):
        top = np.argsort(-freq_raw)[:k]
        score_links(top, f"top{k}")
    score_links(modal_set, "modal")
    score_links(core, "core")
    sel1 = np.where(freq_raw >= 0.1)[0]
    score_links(sel1, "freq10")
    sel2 = np.where(freq_raw >= 0.02)[0]
    score_links(sel2, "freq02")
    # step 5+6 variants
    for k in (3, 5, 7):
        top = np.argsort(-freq_raw)[:k]
        pred = np.zeros((C.HORIZON, L), bool)
        pred[4:, list(top)] = True
        vals = [iou_masked(pred, t, e) for t, e in truth]
        res[f"top{k}_last2"] = float(np.mean(vals)) if vals else np.nan
    # note freq index of core links
    res["core_links"] = ",".join(d["links"][core])
    res["modal_links"] = ",".join(d["links"][list(modal_set)])
    tops = np.argsort(-freq_raw)[:8]
    res["top_links"] = "|".join(f"{d['links'][i]}:{freq_raw[i]:.2f}" for i in tops)
    # stability: split fit into first/second half
    half = len(fit) // 2
    f1 = raw[:half].mean(axis=0)
    f2 = raw[half:].mean(axis=0)
    res["halves_corr"] = float(np.corrcoef(f1, f2)[0, 1]) if half > 3 else np.nan
    res["top5_first_half"] = ",".join(d["links"][np.argsort(-f1)[:5]])
    res["top5_second_half"] = ",".join(d["links"][np.argsort(-f2)[:5]])
    return res


def main():
    panels = [p for p in C.PANELS if os.path.exists(C.traffic.split_cache_path(p, "train"))]
    rows = []
    for p in panels:
        rows.append(analyse(p))
        gc.collect()
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 300)
    cols = [c for c in df.columns if not c.startswith(("top_links", "core_links", "modal_links", "top5"))]
    print(df[cols].to_string())
    print()
    for c in ("top_links", "core_links", "modal_links", "top5_first_half", "top5_second_half"):
        print(f"--- {c}")
        for r in df.itertuples(index=False):
            print(f"  {r.panel:11s} {getattr(r, c)}")
    df.to_csv(os.path.join(C.OUT2, "onset_structure.csv"), index=False)


if __name__ == "__main__":
    main()
