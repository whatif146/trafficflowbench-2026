"""v6 feature block: long-horizon recurrence, incident context, imbalance, shockwaves.

All features are functions of released observations with timestamp <= T (the
split's masked layer, NaN-aware) plus a train-split 24 h time-of-day profile
(legal past data).  No imputation model, no future rows.

Per window, ~26 features per link:
  recurrence (24 h / 72 h same time-of-day): queued flags on the previous 1-3
  days, their mean and a smoothed version, the previous-days speed level and its
  gap to the current level, hours since the link was last observed queued
  (censored at 72 h);
  incident context: hours since the link last had a >30 km/h / 15 min speed
  drop, panel-level event counts in the last 6 h / 24 h, minutes since the last
  event anywhere in the panel, fraction of links currently >20 % below their
  24 h profile, and the mean current/profile speed ratio;
  demand-capacity imbalance: upstream-3 and downstream-3 mean flow/capacity at T
  and their 1 h trends, the downstream-minus-upstream difference;
  shockwaves: LWR wave speed w=(q_dn-q_up)/(k_dn-k_up) clipped, its rank
  neighbours and the panel median, queue-front motion over the last 30/60 min.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH

V6_FEATURES = [
    # --- 24 h / 72 h recurrence ---
    "rec_q1", "rec_q2", "rec_q3", "rec_qmean", "rec_qsm", "rec_r", "rec_gap",
    "since_q_h", "ev_since_h",
    # --- incident context (panel level) ---
    "p_ev6", "p_ev24", "p_since_h", "frac_below24", "r_profile_ratio",
    # --- demand / capacity imbalance ---
    "up3_fc", "dn3_fc", "diff_dn_up_fc", "up3_fc_trend", "dn3_fc_trend",
    # --- shockwaves / front motion ---
    "w_shock", "w_up1", "w_dn1", "p_w_med", "d_up_60", "front_up_links_h",
    "d_dn_60",
]
N_V6 = len(V6_FEATURES)
V6_INDEX = {n: i for i, n in enumerate(V6_FEATURES)}

CAP_STEPS = 864          # 72 h lookback cap
EV_DROP_KMH = 30.0       # speed drop ...
EV_WIN_STEPS = 3         # ... within 15 min


def _tod_slot(times: np.ndarray) -> np.ndarray:
    dti = pd.to_datetime(times, unit="s", utc=True)
    return ((dti.hour * 60 + dti.minute) // 5).to_numpy(np.int64)


def profile_24h(panel: str) -> dict:
    """Train-split, masked, NaN-aware 24 h time-of-day profile (288, L)."""
    d = LH.split_data(panel, "train")
    st = d["st"]
    L = st["n_links"]
    sp = d["masked"]["speed"]
    vc = st["v_cut"][None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = sp / vc
        fc = d["masked"]["flow"] / np.maximum(st["capacity"], 1e-3)[None, :]
    slots = _tod_slot(d["times"])
    T = len(slots)
    flat = slots.repeat(L) * L + np.tile(np.arange(L), T)
    out = {}
    for name, arr in (("r", r), ("fc", fc)):
        a = np.asarray(arr, np.float32).ravel()
        v = np.isfinite(a)
        s = np.zeros(288 * L, np.float64)
        c = np.zeros(288 * L, np.float64)
        np.add.at(s, flat[v], a[v])
        np.add.at(c, flat[v], 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            prof = (s / c).reshape(288, L).astype(np.float32)
        out[name] = prof
    return out


def panel_series(panel: str, split: str) -> dict:
    """Precomputed per-row derivations for one panel/split (one panel at a time)."""
    d = LH.split_data(panel, split)
    st = d["st"]
    m = d["masked"]
    sp, fl = m["speed"], m["flow"]
    T, L = sp.shape
    vc = st["v_cut"][None, :]
    cap = np.maximum(st["capacity"], 1e-3)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (sp / vc).astype(np.float32)
        fc = (fl / cap).astype(np.float32)
        dens = (fl / np.maximum(sp, 1e-3) / np.maximum(st["k_jam"], 1e-3)[None, :]).astype(np.float32)
    obs = np.isfinite(sp)
    qf = np.where(obs, (sp <= vc).astype(np.float32), np.nan).astype(np.float32)
    # >30 km/h speed drop within 15 min, observed on both ends
    delta = np.full((T, L), np.nan, np.float32)
    delta[EV_WIN_STEPS:] = sp[EV_WIN_STEPS:] - sp[:-EV_WIN_STEPS]
    ev = np.isfinite(delta) & (delta <= -EV_DROP_KMH)
    ev_per_row = ev.sum(axis=1).astype(np.int32)
    prefix = np.zeros(T + 1, np.int64)
    prefix[1:] = np.cumsum(ev_per_row)
    # last observed queued row per link / last event row per link / panel-level last event
    lastq = np.full((T, L), -10 ** 6, np.int32)
    lastev = np.full((T, L), -10 ** 6, np.int32)
    lastev_row = np.full(T, -10 ** 6, np.int32)
    cur_q = np.full(L, -10 ** 6, np.int32)
    cur_e = np.full(L, -10 ** 6, np.int32)
    cur_er = -10 ** 6
    for t in range(T):
        q = qf[t]
        cur_q = np.where(np.isfinite(q) & (q > 0.5), t, cur_q)
        e = ev[t]
        cur_e = np.where(e, t, cur_e)
        if e.any():
            cur_er = t
        lastq[t] = cur_q
        lastev[t] = cur_e
        lastev_row[t] = cur_er
    return {"panel": panel, "split": split, "times": d["times"], "st": st,
            "r": r, "fc": fc, "dens": dens, "obs": obs, "qf": qf,
            "ev": ev, "ev_prefix": prefix, "lastq": lastq, "lastev": lastev,
            "lastev_row": lastev_row, "L": L, "T": T}


def v6_features(series: dict, ti: int, prof: dict) -> np.ndarray:
    """(L, N_V6) float32 features for one window origin ti."""
    st = series["st"]
    L = series["L"]
    r2i = st["rank_to_idx"]
    qf = series["qf"]
    r = series["r"]
    fc = series["fc"]
    dens = series["dens"]
    obs = series["obs"]
    T = series["T"]

    def qrow(t):
        return qf[t] if t >= 0 else np.full((L,), np.nan, np.float32)

    # ---- 24 h / 72 h recurrence -------------------------------------------
    q1 = qrow(ti - 288)
    q2 = qrow(ti - 576)
    q3 = qrow(ti - 864)
    stack = np.vstack([q1, q2, q3])
    rec_qmean = np.nanmean(stack, axis=0)
    sm_rows = []
    for d in range(3):
        base = ti - 288 * (d + 1)
        for j in (-2, -1, 0, 1, 2):
            sm_rows.append(qrow(base + j))
    rec_qsm = np.nanmean(np.vstack(sm_rows), axis=0)
    rec_r = np.nanmean(np.vstack([r[ti - 288] if ti >= 288 else np.full(L, np.nan),
                                  r[ti - 576] if ti >= 576 else np.full(L, np.nan),
                                  r[ti - 864] if ti >= 864 else np.full(L, np.nan)]), axis=0)
    r_T = r[ti]
    rec_gap = r_T - rec_r
    since_q_h = np.clip(ti - series["lastq"][ti], 0, CAP_STEPS).astype(np.float32) / 12.0
    ev_since_h = np.clip(ti - series["lastev"][ti], 0, CAP_STEPS).astype(np.float32) / 12.0

    # ---- incident context --------------------------------------------------
    pre = series["ev_prefix"]
    lo6 = max(0, ti - 72)
    lo24 = max(0, ti - 288)
    p_ev6 = np.full(L, float(pre[ti + 1] - pre[lo6]), np.float32)
    p_ev24 = np.full(L, float(pre[ti + 1] - pre[lo24]), np.float32)
    p_since_h = np.full(L, min(max(ti - series["lastev_row"][ti], 0), CAP_STEPS * 7) / 12.0,
                        np.float32)
    slot = int(_tod_slot(series["times"][ti:ti + 1])[0])
    ref_r = prof["r"][slot]
    o = obs[ti] & np.isfinite(r_T) & np.isfinite(ref_r)
    if o.any():
        frac_below = np.full(L, float(np.mean(r_T[o] < 0.8 * ref_r[o])), np.float32)
        ratio = np.full(L, float(np.nanmean(r_T[o] / np.maximum(ref_r[o], 1e-3))), np.float32)
    else:
        frac_below = np.zeros(L, np.float32)
        ratio = np.zeros(L, np.float32)

    # ---- demand / capacity imbalance --------------------------------------
    fcT = fc[ti]
    fc0 = fc[ti - 12] if ti >= 12 else np.full(L, np.nan, np.float32)
    up_vals = [LH.shift_rank(fcT, r2i, -d, fill=np.nan) for d in (1, 2, 3)]
    dn_vals = [LH.shift_rank(fcT, r2i, d, fill=np.nan) for d in (1, 2, 3)]
    up_old = [LH.shift_rank(fc0, r2i, -d, fill=np.nan) for d in (1, 2, 3)]
    dn_old = [LH.shift_rank(fc0, r2i, d, fill=np.nan) for d in (1, 2, 3)]
    up3 = np.nanmean(np.vstack(up_vals), axis=0)
    dn3 = np.nanmean(np.vstack(dn_vals), axis=0)
    up3_old = np.nanmean(np.vstack(up_old), axis=0)
    dn3_old = np.nanmean(np.vstack(dn_old), axis=0)
    diff_dn_up = dn3 - up3
    up3_trend = up3 - up3_old
    dn3_trend = dn3 - dn3_old

    # ---- shockwaves and front motion --------------------------------------
    cap_v = np.maximum(st["capacity"], 1e-3)
    kjam_v = np.maximum(st["k_jam"], 1e-3)
    q_up = LH.shift_rank(series["fc"][ti] * cap_v, r2i, -1, fill=np.nan)
    q_dn = LH.shift_rank(series["fc"][ti] * cap_v, r2i, 1, fill=np.nan)
    k_up = LH.shift_rank(series["dens"][ti] * kjam_v, r2i, -1, fill=np.nan)
    k_dn = LH.shift_rank(series["dens"][ti] * kjam_v, r2i, 1, fill=np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (q_dn - q_up) / (k_dn - k_up)
    w = np.where(np.isfinite(w) & (np.abs(k_dn - k_up) > 1.0),
                 np.clip(w, -80.0, 20.0), np.nan).astype(np.float32)
    w_up1 = LH.shift_rank(w, r2i, -1, fill=np.nan)
    w_dn1 = LH.shift_rank(w, r2i, 1, fill=np.nan)
    p_w_med = np.full(L, float(np.nanmedian(w)) if np.isfinite(w).any() else np.nan, np.float32)

    qT = np.nan_to_num(qf[ti], nan=0.0)
    d_T = LH._dist_to_queued(qT, r2i, -1)
    d_30 = LH._dist_to_queued(np.nan_to_num(qf[ti - 6], nan=0.0), r2i, -1) if ti >= 6 else d_T
    d_60 = LH._dist_to_queued(np.nan_to_num(qf[ti - 12], nan=0.0), r2i, -1) if ti >= 12 else d_T
    dT_dn = LH._dist_to_queued(qT, r2i, 1)
    d60_dn = (LH._dist_to_queued(np.nan_to_num(qf[ti - 12], nan=0.0), r2i, 1)
              if ti >= 12 else dT_dn)
    d_up_60 = (d_T - d_60).astype(np.float32)
    d_dn_60 = (dT_dn - d60_dn).astype(np.float32)
    front_up = (d_up_60 * 12.0).astype(np.float32)   # links/hour (upstream front motion)

    out = np.stack([
        q1, q2, q3, rec_qmean, rec_qsm, rec_r, rec_gap, since_q_h, ev_since_h,
        p_ev6, p_ev24, p_since_h, frac_below, ratio,
        up3, dn3, diff_dn_up, up3_trend, dn3_trend,
        w, w_up1, w_dn1, p_w_med, d_up_60, front_up, d_dn_60,
    ], axis=0).astype(np.float32)
    return np.ascontiguousarray(out.T)


def d_for_dn_default(series):
    return LH._dist_to_queued(np.nan_to_num(series["qf"][max(0, series["T"] - 1)], nan=0.0),
                              series["st"]["rank_to_idx"], 1)
