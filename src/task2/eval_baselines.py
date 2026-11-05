"""Baseline predictors and pseudo-CV for Task 2.

Predictors implemented here (all causal: use only history <= T):
  * persistence      : queue flags at the last history step, repeated for 6 steps
  * onset_static_k   : for queue_onset windows, the k most frequent step-6 queue
                       links seen in the fit-period onset windows, predicted at
                       step 6 only (the greedy selector puts the onset there)
  * ongoing_static   : persistence plus links whose fit-period queue rate for the
                       target time-of-day is high
Run:  python eval_baselines.py [--panels ...]
"""
from __future__ import annotations

import argparse
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C

CUT = pd.Timestamp("2031-02-01", tz="UTC")


def iou(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> float:
    """Space-time IoU over eligible truth cells (mirrors score_task2)."""
    p = pred.astype(bool) & elig
    t = truth.astype(bool)
    inter = float(np.logical_and(p, t).sum())
    union = float(np.logical_or(p, t).sum())
    return 1.0 if union == 0 else inter / union


def onset_freq(d: dict, acc: pd.DataFrame, step: int = C.HORIZON - 1) -> np.ndarray:
    """Per-link frequency of being queued at the last horizon step at onset."""
    freq = np.zeros(d["n_links"], np.float64)
    sub = acc[acc.condition == "queue_onset"]
    if len(sub) == 0:
        return freq
    for r in sub.itertuples(index=False):
        q = d["Q"][r.ti + 1 + step]
        freq += (q > 0)
    return freq / len(sub)


def tod_queue_rate(d: dict, fit_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-(link, minute-of-day) queue rate from the train days."""
    times = d["time_dt"]
    tod = (times.hour * 60 + times.minute).to_numpy()
    Q = d["Q"]
    L = d["n_links"]
    rate = np.zeros((288, L), np.float64)
    cnt = np.zeros(288, np.float64)
    slots = tod // 5
    if fit_mask is not None:
        idx = np.where(fit_mask)[0]
    else:
        idx = np.arange(len(times))
    np.add.at(rate, slots[idx], Q[idx].astype(np.float64))
    np.add.at(cnt, slots[idx], 1.0)
    return rate / np.maximum(cnt, 1)[:, None]


def eval_panel(panel: str, verbose: bool = True):
    d = C.load_train_panel(panel)
    cand = C.candidate_table(d)
    acc = C.greedy_select(cand)
    acc = acc.sort_values("time_dt").reset_index(drop=True)
    is_fit = acc.time_dt < CUT
    fit = acc[is_fit]
    ev = acc[~is_fit]
    if verbose:
        print(f"{panel}: accepted {len(acc)} (fit {len(fit)} / eval {len(ev)}) "
              f"onset fit {int((fit.condition=='queue_onset').sum())} "
              f"eval {int((ev.condition=='queue_onset').sum())}")

    freq = onset_freq(d, fit)
    # also fit-period ongoing last-step link frequency (structural hotspots)
    ong_freq = np.zeros(d["n_links"], np.float64)
    sub = fit[fit.condition == "queue_ongoing"]
    for r in sub.itertuples(index=False):
        ong_freq += (d["Q"][r.ti - 1] > 0)
    if len(sub):
        ong_freq /= len(sub)
    rate = tod_queue_rate(d)

    rows = []
    for r in ev.itertuples(index=False):
        ti = r.ti
        truth = d["Q"][ti + 1:ti + 1 + C.HORIZON]
        eligf = d["elig"][ti + 1:ti + 1 + C.HORIZON]
        L = d["n_links"]
        pred_pers = np.repeat(d["Qobs"][ti - 1][None, :], C.HORIZON, axis=0).astype(bool)
        rec = {"panel": panel, "condition": r.condition, "time_dt": r.time_dt,
               "truth_n": int(truth.sum()), "persistence": iou(pred_pers, truth, eligf)}
        if r.condition == "queue_onset":
            for k in (1, 2, 3, 4, 5, 6, 7, 8):
                top = np.argsort(-freq)[:k]
                pred = np.zeros((C.HORIZON, L), bool)
                pred[C.HORIZON - 1, top] = True
                rec[f"onset_static{k}"] = iou(pred, truth, eligf)
            # same but including ongoing-hotspot frequency
            comb = freq + 0.5 * ong_freq
            for k in (2, 3, 4, 5, 6, 7):
                top = np.argsort(-comb)[:k]
                pred = np.zeros((C.HORIZON, L), bool)
                pred[C.HORIZON - 1, top] = True
                rec[f"onset_comb{k}"] = iou(pred, truth, eligf)
            # same but predict last two steps
            top = np.argsort(-freq)[:5]
            pred = np.zeros((C.HORIZON, L), bool)
            pred[4:, top] = True
            rec["onset_static5_last2"] = iou(pred, truth, eligf)
        else:
            for th in (0.05, 0.10, 0.15, 0.20, 0.30):
                pred = np.zeros((C.HORIZON, L), bool)
                for k in range(C.HORIZON):
                    tod_min = d["time_dt"][ti + 1 + k]
                    slot = (tod_min.hour * 60 + tod_min.minute) // 5
                    pred[k] = (rate[slot] >= th) | pred_pers[k]
                rec[f"ongoing_rate{th}"] = iou(pred, truth, eligf)
            # persistence with 1-step dilation (upstream+downstream in rank)
            for dxy in (0, 1, 2, 3):
                pr = pred_pers.copy()
                if dxy:
                    qr = pr[:, d["rank_to_idx"]]
                    for s in range(1, dxy + 1):
                        pr[:, d["rank_to_idx"]] |= np.roll(qr, s, axis=1) | np.roll(qr, -s, axis=1)
                rec[f"ongoing_pers_dil{dxy}"] = iou(pr, truth, eligf)
        rows.append(rec)
    detail = pd.DataFrame(rows)
    return detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(C.OUT2, "baselines_cv.csv"))
    args = ap.parse_args()
    C.ensure_dirs()
    panels = args.panels or [p for p in C.PANELS
                             if os.path.exists(C.traffic.split_cache_path(p, "train"))]
    details = [eval_panel(p) for p in panels]
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    cols = [c for c in detail.columns if c not in ("panel", "condition", "time_dt", "truth_n")]
    # eval set only has Feb windows; aggregate with the official nesting
    view = detail[detail.condition == "queue_onset"]
    print("\n== queue_onset (Feb pseudo-CV) n=", len(view))
    print(view[cols].mean().sort_values(ascending=False).to_string())
    view = detail[detail.condition == "queue_ongoing"]
    print("\n== queue_ongoing (Feb pseudo-CV) n=", len(view))
    print(view[cols].mean().sort_values(ascending=False).to_string())
    # equal-condition mean for a couple of candidates
    print("\n== equal-condition mean of persistence =",
          round(float(detail.groupby("condition").persistence.mean().mean()), 4))


if __name__ == "__main__":
    main()

