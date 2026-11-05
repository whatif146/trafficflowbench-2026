"""More Task 1 variants: density-based flow, separate spatial smoothing, pct weights."""
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
import tune_smooth as T


def predict_variant(masked, plain, hold, lam_t, lam_s_speed, lam_s_flow, iters,
                    mode="direct", pct_weight=False, cross_day=False):
    times = pd.to_datetime(masked["times"], unit="s", utc=True)
    S = len(masked["stations"])
    Tn = len(times)
    day_index = np.arange(Tn) // 288
    fit_time = ~np.isin(day_index, hold)
    wd = times.weekday.to_numpy()
    tod = (times.hour * 12 + times.minute // 5).to_numpy()
    P_s, P_q = P.build_profiles(plain, fit_time)
    # density profile
    with np.errstate(invalid="ignore", divide="ignore"):
        k_plain = np.where(plain["speed"] > 1, plain["flow"] / plain["speed"], np.nan)
    P_k = _build_k_profile(k_plain, plain, fit_time)

    obs_s = masked["speed"].astype(np.float64)
    obs_q = masked["flow"].astype(np.float64)
    obs_mask = np.isfinite(obs_s) & np.isfinite(obs_q)
    wgt = np.ones_like(obs_s)
    if pct_weight:
        wgt = (masked["pct"].astype(np.float64) / 100.0).clip(0.0, 1.0)

    pred_s = np.full((Tn, S), np.nan, dtype=np.float32)
    pred_q = np.full((Tn, S), np.nan, dtype=np.float32)
    for d in hold:
        sl = slice(d * 288, (d + 1) * 288)
        w = wd[sl][0]
        m = obs_mask[sl]
        ww = np.where(m, wgt[sl], 0.0)
        base_s = P_s[:, w, :].T
        r_s = np.where(m, obs_s[sl] - base_s, 0.0)
        r_s_hat = T._smooth_w(r_s, ww, lam_t, lam_s_speed, iters)
        pred_s[sl] = np.maximum(base_s + r_s_hat, 1.0)
        if mode == "direct":
            base_q = P_q[:, w, :].T
            r_q = np.where(m, obs_q[sl] - base_q, 0.0)
            r_q_hat = T._smooth_w(r_q, ww, lam_t, lam_s_flow, iters)
            pred_q[sl] = np.maximum(base_q + r_q_hat, 0.0)
        else:
            base_k = P_k[:, w, :].T
            obs_k = np.where(m, obs_q[sl] / np.maximum(obs_s[sl], 1.0), 0.0)
            r_k = np.where(m, obs_k - base_k, 0.0)
            r_k_hat = T._smooth_w(r_k, ww, lam_t, lam_s_flow, iters)
            k_hat = np.maximum(base_k + r_k_hat, 0.5)
            pred_q[sl] = np.maximum(k_hat * (base_s + r_s_hat), 0.0)
    return pred_s, pred_q


def _build_k_profile(k_plain, plain, time_mask):
    times = pd.to_datetime(plain["times"], unit="s", utc=True)
    wd = times.weekday.to_numpy()
    S = k_plain.shape[1]
    Pk = np.zeros((S, 7, 288))
    Ck = np.zeros((S, 7, 288))
    day_index = np.arange(len(times)) // 288
    for d in np.unique(day_index):
        rows = np.flatnonzero(day_index == d)
        w = wd[rows[0]]
        sl = slice(rows[0], rows[-1] + 1)
        m = np.isfinite(k_plain[sl]) & time_mask[sl][:, None]
        v = np.where(m, k_plain[sl], 0.0).astype(np.float64)
        Pk[:, w, :] += v.T
        Ck[:, w, :] += m.T.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        Pk = np.where(Ck > 0, Pk / np.maximum(Ck, 1), np.nan)
    station_fill = np.nanmean(Pk, axis=(1, 2))
    Pk = np.where(np.isfinite(Pk), Pk, station_fill[:, None, None])
    return Pk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="D7_I10_E")
    ap.add_argument("--holdout-days", type=int, default=60)
    args = ap.parse_args()
    panel = args.panel
    data = traffic.load_mainline(panel, "train")
    masked, plain = data["masked"], data["plain"]
    times = pd.to_datetime(masked["times"], unit="s", utc=True)
    n_days = (np.arange(len(times)) // 288).max() + 1
    hold = np.arange(n_days - args.holdout_days, n_days)
    tg = traffic.load_targets(panel, "train")
    keep = np.isin(tg["t_idx"] // 288, hold)
    t_idx = tg["t_idx"][keep].astype(np.int64)
    s_idx = tg["s_idx"][keep].astype(np.int64)
    reg = tg["regime"][keep]
    true_s = plain["speed"].astype(np.float64)[t_idx, s_idx]
    true_q = plain["flow"].astype(np.float64)[t_idx, s_idx]
    lanes = masked["lanes"].astype(np.float64)[s_idx]

    def report(name, ps, pq):
        ok = np.isfinite(ps) & np.isfinite(pq)
        sc = []
        detail = []
        for r in range(3):
            m = ok & (reg == r)
            ds = ps[m] - true_s[m]
            dq = (pq[m] - true_q[m]) / lanes[m]
            rs, rq = float(np.sqrt(np.mean(ds ** 2))), float(np.sqrt(np.mean(dq ** 2)))
            ss, sq = max(0, 1 - rs / 25), max(0, 1 - rq / 600)
            sc.append(0.54 * ss + 0.46 * sq)
            detail.append((rs, rq))
        print(f"{name:52s} " + " ".join(f"{x:.4f}" for x in sc) +
              f"  macro={np.mean(sc):.4f}  rmse_s={np.mean([d[0] for d in detail]):.2f} rmse_q={np.mean([d[1] for d in detail]):.2f}", flush=True)

    variants = [
        ("direct lt1.0 lss.15 lsf.15", dict(mode="direct", lam_t=1.0, lam_s_speed=0.15, lam_s_flow=0.15)),
        ("direct lt1.0 lss.10 lsf.30", dict(mode="direct", lam_t=1.0, lam_s_speed=0.10, lam_s_flow=0.30)),
        ("direct lt1.0 lss.25 lsf.10", dict(mode="direct", lam_t=1.0, lam_s_speed=0.25, lam_s_flow=0.10)),
        ("density  lt1.0 lss.15 lsf.15", dict(mode="density", lam_t=1.0, lam_s_speed=0.15, lam_s_flow=0.15)),
        ("density  lt1.0 lss.10 lsf.30", dict(mode="density", lam_t=1.0, lam_s_speed=0.10, lam_s_flow=0.30)),
        ("direct lt1.0 lss.15 lsf.15 pct", dict(mode="direct", lam_t=1.0, lam_s_speed=0.15, lam_s_flow=0.15, pct_weight=True)),
        ("direct lt2.0 lss.15 lsf.15", dict(mode="direct", lam_t=2.0, lam_s_speed=0.15, lam_s_flow=0.15)),
    ]
    for name, kw in variants:
        t0 = time.time()
        ps, pq = predict_variant(masked, plain, hold, iters=60, **kw)
        report(name, ps[t_idx, s_idx], pq[t_idx, s_idx])
        print(f"    ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
