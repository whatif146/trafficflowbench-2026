"""Tune the profile + residual-smoothing reconstruction on a train holdout."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
import traffic
import proto_smooth as P


def predict(masked, plain, hold_days, lam_t, lam_s, iters, pct_weight=False, profile_from_plain=True):
    times = pd.to_datetime(masked["times"], unit="s", utc=True)
    S = len(masked["stations"])
    T = len(times)
    day_index = np.arange(T) // 288
    n_days = day_index.max() + 1
    fit_time = ~np.isin(day_index, hold_days)
    wd = times.weekday.to_numpy()
    P_s, P_q = P.build_profiles(plain, fit_time)

    obs_s = masked["speed"].astype(np.float64)
    obs_q = masked["flow"].astype(np.float64)
    obs_mask = np.isfinite(obs_s) & np.isfinite(obs_q)
    wgt = np.ones_like(obs_s)
    if pct_weight:
        wgt = (masked["pct"].astype(np.float64) / 100.0).clip(0.0, 1.0)
    pred_s = np.full((T, S), np.nan, dtype=np.float32)
    pred_q = np.full((T, S), np.nan, dtype=np.float32)
    for d in hold_days:
        sl = slice(d * 288, (d + 1) * 288)
        w = wd[sl][0]
        base_s = P_s[:, w, :].T
        base_q = P_q[:, w, :].T
        m = obs_mask[sl]
        ww = np.where(m, wgt[sl], 0.0)
        r_s = np.where(m, obs_s[sl] - base_s, 0.0)
        r_q = np.where(m, obs_q[sl] - base_q, 0.0)
        r_s_hat = P.smooth_fill(r_s, m, lam_t, lam_s, iters) if not pct_weight else _smooth_w(r_s, ww, lam_t, lam_s, iters)
        r_q_hat = P.smooth_fill(r_q, m, lam_t, lam_s, iters) if not pct_weight else _smooth_w(r_q, ww, lam_t, lam_s, iters)
        pred_s[sl] = np.maximum(base_s + r_s_hat, 1.0)
        pred_q[sl] = np.maximum(base_q + r_q_hat, 0.0)
    return pred_s, pred_q


def _smooth_w(resid, w, lam_t, lam_s, iters):
    T, S = resid.shape
    m = w > 0
    x = P.temporal_fill(np.where(m, resid, 0.0), m)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="D7_I10_E")
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--lam-t", type=float, nargs="+", default=[1.0])
    ap.add_argument("--lam-s", type=float, nargs="+", default=[0.6])
    ap.add_argument("--iters", type=int, nargs="+", default=[60])
    ap.add_argument("--pct-weight", action="store_true")
    args = ap.parse_args()
    panel = args.panel
    data = traffic.load_mainline(panel, "train")
    masked, plain = data["masked"], data["plain"]
    times = pd.to_datetime(masked["times"], unit="s", utc=True)
    n_days = (np.arange(len(times)) // 288).max() + 1
    hold = np.arange(n_days - args.holdout_days, n_days)

    # profile-only baseline
    fit_time = ~np.isin(np.arange(len(times)) // 288, hold)
    wd = times.weekday.to_numpy()
    P_s, P_q = P.build_profiles(plain, fit_time)
    tg = traffic.load_targets(panel, "train")
    day = tg["t_idx"] // 288
    keep = np.isin(day, hold)
    t_idx, s_idx, reg = tg["t_idx"][keep].astype(np.int64), tg["s_idx"][keep].astype(np.int64), tg["regime"][keep]
    true_s = plain["speed"].astype(np.float64)[t_idx, s_idx]
    true_q = plain["flow"].astype(np.float64)[t_idx, s_idx]
    lanes = masked["lanes"].astype(np.float64)[s_idx]

    def report(name, ps, pq):
        ok = np.isfinite(ps) & np.isfinite(pq) & np.isfinite(true_s) & np.isfinite(true_q)
        scores = []
        for r in range(3):
            m = ok & (reg == r)
            ds = ps[m] - true_s[m]
            dq = (pq[m] - true_q[m]) / lanes[m]
            rs, rq = float(np.sqrt(np.mean(ds ** 2))), float(np.sqrt(np.mean(dq ** 2)))
            ss, sq = max(0, 1 - rs / 25), max(0, 1 - rq / 600)
            scores.append(0.54 * ss + 0.46 * sq)
        print(f"{name:38s} " + " ".join(f"{s:.4f}" for s in scores) + f"  macro={np.mean(scores):.4f}", flush=True)

    bp_s = np.full(true_s.shape, np.nan)
    bp_q = np.full(true_q.shape, np.nan)
    for i in range(len(t_idx)):
        bp_s[i] = P_s[s_idx[i], wd[t_idx[i]], (pd.Timestamp(times[t_idx[i]]).hour * 12 + pd.Timestamp(times[t_idx[i]]).minute // 5)]
        bp_q[i] = P_q[s_idx[i], wd[t_idx[i]], (pd.Timestamp(times[t_idx[i]]).hour * 12 + pd.Timestamp(times[t_idx[i]]).minute // 5)]
    report("profile-only", bp_s, bp_q)

    for lam_t in args.lam_t:
        for lam_s in args.lam_s:
            for iters in args.iters:
                t0 = time.time()
                pred_s, pred_q = predict(masked, plain, hold, lam_t, lam_s, iters, args.pct_weight)
                ps = pred_s[t_idx, s_idx].astype(np.float64)
                pq = pred_q[t_idx, s_idx].astype(np.float64)
                report(f"lam_t={lam_t} lam_s={lam_s} it={iters}", ps, pq)
                print(f"   ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
