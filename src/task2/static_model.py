"""Static (climatology + persistence) Task 2 predictors.

Fitted per panel from the train split only.  Two blocks:

  onset   : the greedy selector puts the first queue of a day at the LAST horizon
            step, so we predict a small set of recurrent hotspot links at that
            step.  Membership probability comes from the accepted train onset
            windows; per-window history signals can refine it (see refine methods).
  ongoing : the queue already exists at T-5.  Predict a robust estimate of the
            current queue set (3-step median speed <= v_cut), optionally
            expanded/shifted upstream and unioned with the time-of-day
            climatology of the panel.
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


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def tod_queue_rate(d: dict, day_mask: np.ndarray | None = None) -> np.ndarray:
    """(288, L) queue rate by 5-min slot of the day from train days."""
    tod = (d["time_dt"].hour * 60 + d["time_dt"].minute).to_numpy(dtype=np.int32)
    slots = tod // 5
    L = d["n_links"]
    Q = d["Q"]
    if day_mask is None:
        rate_sum = np.zeros((288, L), np.float64)
        np.add.at(rate_sum, slots, Q.astype(np.float64))
        cnt = np.bincount(slots, minlength=288).astype(np.float64)
    else:
        idx = np.where(day_mask)[0]
        rate_sum = np.zeros((288, L), np.float64)
        np.add.at(rate_sum, slots[idx], Q[idx].astype(np.float64))
        cnt = np.bincount(slots[idx], minlength=288).astype(np.float64)
    return rate_sum / np.maximum(cnt, 1.0)[:, None]


def fit_panel(d: dict, onset_fit: pd.DataFrame, ongoing_fit: pd.DataFrame,
              day_mask: np.ndarray | None = None) -> dict:
    """Fit the static parameters of one panel."""
    step = C.HORIZON - 1
    L = d["n_links"]
    raw6 = np.zeros((len(onset_fit), L), np.float32)
    for j, r in enumerate(onset_fit.itertuples(index=False)):
        raw6[j] = d["Qobs"][r.ti + 1 + step]
    onset_freq = raw6.mean(axis=0) if len(raw6) else np.zeros(L, np.float32)
    # frequency of being queued at last history step in ongoing windows
    ong_freq = np.zeros(L, np.float32)
    for r in ongoing_fit.itertuples(index=False):
        ong_freq += d["Qobs"][r.ti - 1]
    if len(ongoing_fit):
        ong_freq /= len(ongoing_fit)
    rate = tod_queue_rate(d, day_mask)
    elig_rate = d["elig"].mean(axis=0).astype(np.float32)
    return {
        "panel": d["panel"], "links": d["links"], "n_links": L,
        "rank": d["rank"], "rank_to_idx": d["rank_to_idx"],
        "v_cut": d["v_cut"], "free_speed": d["free_speed"],
        "capacity": d["capacity"], "lanes": d["lanes"], "k_jam": d["k_jam"],
        "onset_freq": onset_freq, "ong_freq": ong_freq,
        "rate": rate.astype(np.float32), "elig_rate": elig_rate,
        "n_onset_fit": len(onset_fit), "n_ongoing_fit": len(ongoing_fit),
    }


# --------------------------------------------------------------------------- #
# history / prediction helpers (shared with predict.py)
# --------------------------------------------------------------------------- #
def history_arrays(speed: np.ndarray, flow: np.ndarray, elig: np.ndarray,
                   v_cut: np.ndarray) -> dict:
    """Derive the causal per-link state used by the predictors.

    ``speed``/``flow``/``elig`` are (12, L) history arrays T-60..T-5.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        s = speed.astype(np.float32)
        f = flow.astype(np.float32)
        ratio = s / v_cut[None, :]
        dens = np.where(np.isfinite(s) & (s > 0), f / np.maximum(s, 1e-3), np.nan)
    last = s[-1]
    med3 = np.nanmedian(s[-3:], axis=0)
    med6 = np.nanmedian(s[-6:], axis=0)
    medall = np.nanmedian(s, axis=0)
    qnow = np.isfinite(last) & (last <= v_cut)
    qmed3 = np.isfinite(med3) & (med3 <= v_cut)
    qmed6 = np.isfinite(med6) & (med6 <= v_cut)
    # mean queue flag over the last k steps
    qflags = (np.isfinite(s) & (s <= v_cut[None, :])).astype(np.float32)
    qmean3 = qflags[-3:].mean(axis=0)
    qmean6 = qflags[-6:].mean(axis=0)
    nobs = np.isfinite(s).sum(axis=0)
    trend = last - med6
    return {
        "speed": s, "flow": f, "elig": elig, "ratio": ratio, "dens": dens,
        "s_last": last, "s_med3": med3, "s_med6": med6, "s_medall": medall,
        "f_last": f[-1], "f_mean": np.nanmean(f, axis=0),
        "qnow": qnow, "qmed3": qmed3, "qmed6": qmed6,
        "qmean3": qmean3, "qmean6": qmean6, "nobs": nobs, "trend": trend,
    }


def predict_onset(hist: dict, p: dict, k: int = 3, th: float | None = None,
                  shift_steps: int = 1) -> np.ndarray:
    """(6, L) bool prediction for a queue_onset window."""
    L = p["n_links"]
    pred = np.zeros((C.HORIZON, L), bool)
    freq = p["onset_freq"]
    if th is not None:
        sel = np.where(freq >= th)[0]
        if len(sel) == 0:
            sel = np.argsort(-freq)[:max(k, 1)]
    else:
        sel = np.argsort(-freq)[:k]
    pred[C.HORIZON - 1, sel] = True
    return pred


def _shift_rank(mask2d: np.ndarray, rank_to_idx: np.ndarray, shift: int,
                fill: bool = False) -> np.ndarray:
    """Shift a (T, L) mask along the rank axis by ``shift`` positions."""
    if shift == 0:
        return mask2d
    out = mask2d.copy()
    out[:, rank_to_idx] = np.roll(mask2d[:, rank_to_idx], shift, axis=1)
    if not fill:
        if shift > 0:
            out[:, rank_to_idx[:shift]] = False
        else:
            out[:, rank_to_idx[shift:]] = False
    return out


def predict_ongoing(hist: dict, p: dict, mode: str = "robust", dil: int = 0,
                    rate_th: float | None = None, shift_per_step: int = 0,
                    union_now: bool = True) -> np.ndarray:
    """(6, L) bool prediction for a queue_ongoing window."""
    L = p["n_links"]
    if mode == "persist":
        base = hist["qnow"]
    elif mode == "robust":
        base = hist["qmed3"]
    elif mode == "robust6":
        base = hist["qmed6"]
    else:
        raise ValueError(mode)
    pred = np.repeat(base[None, :], C.HORIZON, axis=0)
    if shift_per_step:
        for k in range(C.HORIZON):
            pred[k] = _shift_rank(pred[k][None, :], p["rank_to_idx"],
                                  shift_per_step * (k + 1))[0]
    if dil:
        pred = _shift_rank(pred, p["rank_to_idx"], dil) | pred
        pred = _shift_rank(pred, p["rank_to_idx"], -dil) | pred
    if rate_th is not None:
        tod = None
        for k in range(C.HORIZON):
            # caller passes target slots through hist['target_slots']
            slot = hist["target_slots"][k]
            pred[k] |= p["rate"][slot] >= rate_th
    return pred

