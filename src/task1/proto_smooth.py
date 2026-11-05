"""Prototype Task 1 reconstruction: historical profile + residual field smoothing.

Holdout protocol: fit the profile on all train days except a trailing block;
predict the held-out days' published target rows using only the masked layer;
score with the official Task 1 formula per regime.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
import traffic


def build_profiles(plain, time_mask, max_pct=100):
    """Mean speed/flow by (station, weekday, time-of-day) over selected times."""
    speed, flow = plain["speed"], plain["flow"]
    elig = plain["elig"]
    times = pd.to_datetime(plain["times"], unit="s", utc=True)
    wd = times.weekday.to_numpy()
    tod = (times.hour * 12 + times.minute // 5).to_numpy()
    S = speed.shape[1]
    P_s = np.zeros((S, 7, 288), dtype=np.float64)
    C_s = np.zeros((S, 7, 288), dtype=np.float64)
    P_q = np.zeros((S, 7, 288), dtype=np.float64)
    C_q = np.zeros((S, 7, 288), dtype=np.float64)
    valid = elig & time_mask[:, None]
    day_index = np.arange(len(times)) // 288
    for d in np.unique(day_index):
        rows = np.flatnonzero(day_index == d)
        if len(rows) == 0:
            continue
        w = wd[rows[0]]
        sl = slice(rows[0], rows[-1] + 1)
        m = valid[sl]
        v_s = np.where(m, speed[sl], 0.0)
        v_q = np.where(m, flow[sl], 0.0)
        mf = m.astype(np.float64)
        # (288, S) -> (S, 288)
        P_s[:, w, :] += v_s.T
        C_s[:, w, :] += mf.T
        P_q[:, w, :] += v_q.T
        C_q[:, w, :] += mf.T
    with np.errstate(invalid="ignore", divide="ignore"):
        P_s = np.where(C_s > 0, P_s / np.maximum(C_s, 1), np.nan)
        P_q = np.where(C_q > 0, P_q / np.maximum(C_q, 1), np.nan)
    # fallbacks: weekday-independent profile, then station mean
    P_s_tod = np.nanmean(P_s, axis=1)
    P_q_tod = np.nanmean(P_q, axis=1)
    fill_s = np.nanmean(P_s_tod, axis=1)
    fill_q = np.nanmean(P_q_tod, axis=1)
    for l in range(S):
        m = ~np.isfinite(P_s[l])
        P_s[l][m] = np.take(P_s_tod[l], np.nonzero(m)[1])
        P_q[l][m] = np.take(P_q_tod[l], np.nonzero(m)[1])
    P_s = np.where(np.isfinite(P_s), P_s, fill_s[:, None, None])
    P_q = np.where(np.isfinite(P_q), P_q, fill_q[:, None, None])
    return P_s, P_q


def temporal_fill(x, mask):
    """Linear interpolation along axis 0 where mask is False, per column."""
    T, S = x.shape
    out = x.copy()
    t = np.arange(T)
    for l in range(S):
        m = mask[:, l]
        if m.sum() < 2:
            out[:, l] = 0.0
            continue
        out[~m, l] = np.interp(t[~m], t[m], x[m, l])
    return out


def smooth_fill(resid, mask, lam_t=1.0, lam_s=0.6, iters=60):
    """Jacobi smoothing of a partly observed residual field (T x S)."""
    T, S = resid.shape
    w = mask.astype(np.float64)
    x = temporal_fill(np.where(mask, resid, 0.0), mask)
    for _ in range(iters):
        t_avg = np.empty_like(x)
        t_avg[1:-1] = 0.5 * (x[:-2] + x[2:])
        t_avg[0] = x[1]
        t_avg[-1] = x[-2]
        s_avg = np.empty_like(x)
        s_avg[:, 1:-1] = 0.5 * (x[:, :-2] + x[:, 2:])
        s_avg[:, 0] = x[:, 1]
        s_avg[:, -1] = x[:, -2]
        x = (w * resid + lam_t * t_avg + lam_s * s_avg) / (w + lam_t + lam_s)
    return x


def predict_panel(panel, split="train", holdout_days=60):
    data = traffic.load_mainline(panel, split)
    masked = data["masked"]
    plain = data.get("plain")
    times = pd.to_datetime(masked["times"], unit="s", utc=True)
    S = len(masked["stations"])
    T = len(times)
    day_index = np.arange(T) // 288
    n_days = day_index.max() + 1
    hold_days = np.arange(n_days - holdout_days, n_days)
    time_mask_fit = ~np.isin(day_index, hold_days)
    wd = times.weekday.to_numpy()
    tod = (times.hour * 12 + times.minute // 5).to_numpy()
    P_s, P_q = build_profiles(plain, time_mask_fit)

    obs_s = masked["speed"]
    obs_q = masked["flow"]
    obs_mask = np.isfinite(obs_s) & np.isfinite(obs_q)

    pred_s = np.full((T, S), np.nan, dtype=np.float32)
    pred_q = np.full((T, S), np.nan, dtype=np.float32)
    for d in hold_days:
        sl = slice(d * 288, (d + 1) * 288)
        w = wd[sl][0]
        base_s = P_s[:, w, :].T  # (288, S)
        base_q = P_q[:, w, :].T
        m = obs_mask[sl]
        r_s = np.where(m, obs_s[sl] - base_s, 0.0)
        r_q = np.where(m, obs_q[sl] - base_q, 0.0)
        r_s_hat = smooth_fill(r_s, m)
        r_q_hat = smooth_fill(r_q, m)
        pred_s[sl] = np.maximum(base_s + r_s_hat, 1.0)
        pred_q[sl] = np.maximum(base_q + r_q_hat, 0.0)

    return masked, pred_s, pred_q, hold_days


def score_holdout(panel, masked, pred_s, pred_q, hold_days):
    plain = traffic.load_mainline(panel, "train")["plain"]
    true_s = plain["speed"].astype(np.float64)
    true_q = plain["flow"].astype(np.float64)
    lanes = masked["lanes"].astype(np.float64)
    tg = traffic.load_targets(panel, "train")
    t_idx = tg["t_idx"].astype(np.int64)
    s_idx = tg["s_idx"].astype(np.int64)
    reg = tg["regime"].astype(np.int64)
    day = t_idx // 288
    keep = np.isin(day, hold_days)
    t_idx, s_idx, reg = t_idx[keep], s_idx[keep], reg[keep]
    ps = pred_s[t_idx, s_idx].astype(np.float64)
    pq = pred_q[t_idx, s_idx].astype(np.float64)
    ts = true_s[t_idx, s_idx]
    tq = true_q[t_idx, s_idx]
    ok = np.isfinite(ps) & np.isfinite(pq) & np.isfinite(ts) & np.isfinite(tq)
    rows = {}
    for r in range(3):
        m = ok & (reg == r)
        ds = ps[m] - ts[m]
        dq = (pq[m] - tq[m]) / lanes[s_idx[m]]
        rs = float(np.sqrt(np.mean(ds ** 2)))
        rq = float(np.sqrt(np.mean(dq ** 2)))
        ss = max(0.0, 1 - rs / 25.0)
        sq = max(0.0, 1 - rq / 600.0)
        rows[f"R{r+1}"] = dict(n=int(m.sum()), rmse_s=rs, rmse_q=rq, S_speed=ss, S_flow=sq,
                               S_state=0.54 * ss + 0.46 * sq)
    macro = float(np.mean([v["S_state"] for v in rows.values()]))
    return rows, macro


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="D7_I10_E")
    ap.add_argument("--holdout-days", type=int, default=60)
    args = ap.parse_args()
    t0 = time.time()
    masked, pred_s, pred_q, hold = predict_panel(args.panel, "train", args.holdout_days)
    print(f"predicted in {time.time()-t0:.1f}s", flush=True)
    rows, macro = score_holdout(args.panel, masked, pred_s, pred_q, hold)
    for k, v in rows.items():
        print(k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()})
    print("macro S_state", round(macro, 4))
