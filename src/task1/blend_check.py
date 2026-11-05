"""Offline check: does blending the Task-1 LightGBM with the profile smoother help?

Uses the GBM holdout prediction CSVs produced by task1-gbm (holdout-day rows only)
and the Lead's smoother on the same holdout days, scored with the official Task-1
formula on those rows.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
import traffic
import proto_smooth as P
import tune_smooth as T

PANEL = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
gbm_path = os.path.join(traffic.WORK, "task1", "gbm", f"holdout_{PANEL}.csv")

data = traffic.load_mainline(PANEL, "train")
masked, plain = data["masked"], data["plain"]
times = pd.to_datetime(masked["times"], unit="s", utc=True)
n_days = (np.arange(len(times)) // 288).max() + 1
hold = np.arange(n_days - 60, n_days)
fit = ~np.isin(np.arange(len(times)) // 288, hold)
Ps, Pq = P.build_profiles(plain, fit)
wd = times.weekday.to_numpy()
obs_s = masked["speed"].astype(np.float64)
obs_q = masked["flow"].astype(np.float64)
m0 = np.isfinite(obs_s) & np.isfinite(obs_q)
pct = masked["pct"].astype(np.float64) / 100.0
sp = np.full_like(obs_s, np.nan)
fl = np.full_like(obs_q, np.nan)
for d in hold:
    sl = slice(d * 288, (d + 1) * 288)
    w = wd[sl][0]
    m = m0[sl]
    ww = np.where(m, pct[sl], 0.0)
    bs = Ps[:, w, :].T
    bq = Pq[:, w, :].T
    rs = np.where(m, obs_s[sl] - bs, 0.0)
    rq = np.where(m, obs_q[sl] - bq, 0.0)
    sp[sl] = np.maximum(bs + T._smooth_w(rs, ww, 1.0, 0.10, 60), 1.0)
    fl[sl] = np.maximum(bq + T._smooth_w(rq, ww, 1.0, 0.30, 60), 0.0)

gbm = pd.read_csv(gbm_path, dtype={"panel": str, "timestamp": str, "station_id": str,
                                   "link_id": str, "mask_regime": str})
main = masked
iso = traffic.epoch_to_iso(main["times"])
ti_map = {s: i for i, s in enumerate(iso)}
si_map = {s: i for i, s in enumerate(main["stations"].astype(str))}
ti = pd.to_datetime(gbm.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ").map(ti_map).to_numpy()
si = gbm.station_id.astype(str).map(si_map).to_numpy()
gs = gbm.speed_kmh.to_numpy(np.float64)
gq = gbm.flow_vph.to_numpy(np.float64)
ss = sp[ti, si]
sq = fl[ti, si]
true_s = plain["speed"].astype(np.float64)[ti, si]
true_q = plain["flow"].astype(np.float64)[ti, si]
lanes = masked["lanes"].astype(np.float64)[si]
reg = np.array([traffic.REGIME_NAMES.index(r) for r in gbm.mask_regime], dtype=np.int64)


def score(ps, pq, tag):
    ok = np.isfinite(ps) & np.isfinite(pq) & np.isfinite(true_s) & np.isfinite(true_q)
    out = []
    for r in range(3):
        m = ok & (reg == r)
        ds = ps[m] - true_s[m]
        dq = (pq[m] - true_q[m]) / lanes[m]
        ss_ = max(0.0, 1 - float(np.sqrt(np.mean(ds ** 2))) / 25.0)
        sq_ = max(0.0, 1 - float(np.sqrt(np.mean(dq ** 2))) / 600.0)
        out.append(0.54 * ss_ + 0.46 * sq_)
    print(f"{tag:22s} R1={out[0]:.4f} R2={out[1]:.4f} R3={out[2]:.4f} macro={np.mean(out):.4f}", flush=True)
    return float(np.mean(out))


print("panel", PANEL, "holdout rows", len(gbm))
score(gs, gq, "GBM")
score(ss, sq, "smoother")
for w in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    score(w * gs + (1 - w) * ss, w * gq + (1 - w) * sq, f"blend gbm={w}")
