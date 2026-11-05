"""Task 1 evaluator and baseline for the team.

Usage:
    python work/task1/eval.py --submission pred.csv --panel D7_I10_E --holdout-days 60
    python work/task1/eval.py --submission pred.csv            # all panels with a cache
If --submission is omitted the profile-only historical baseline is scored.

The holdout protocol: the last ``--holdout-days`` days of train are held out; a
prediction CSV produced *without using those days' truth* can be scored on the
published target rows of those days. The script also reports the full-train
score (official style) for reference.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
import traffic


def profile_predictions(panel, split, hold_days=None):
    data = traffic.load_mainline(panel, split)
    if split == "train":
        plain, masked = data["plain"], data["masked"]
    else:
        return None
    times = pd.to_datetime(masked["times"], unit="s", utc=True)
    day = np.arange(len(times)) // 288
    fit = np.ones(len(times), dtype=bool) if hold_days is None else ~np.isin(day, hold_days)
    # profile from plain layer
    import proto_smooth as P

    P_s, P_q = P.build_profiles(plain, fit)
    wd = times.weekday.to_numpy()
    tod = (times.hour * 12 + times.minute // 5).to_numpy()
    tg = traffic.load_targets(panel, split)
    t_idx, s_idx = tg["t_idx"].astype(np.int64), tg["s_idx"].astype(np.int64)
    ps = P_s[s_idx, wd[t_idx], tod[t_idx]]
    pq = P_q[s_idx, wd[t_idx], tod[t_idx]]
    return t_idx, s_idx, tg["regime"].astype(np.int64), ps.astype(np.float64), pq.astype(np.float64)


def score(panel, pred, hold_days=None):
    """pred: DataFrame with timestamp/station_id/link_id/mask_regime/speed/flow."""
    plain = traffic.load_mainline(panel, "train")["plain"]
    masked = traffic.load_mainline(panel, "train")["masked"]
    times = pd.to_datetime(masked["times"], unit="s", utc=True)
    time_index = pd.Series(np.arange(len(times)), index=pd.DatetimeIndex(times))
    stations = pd.Index(masked["stations"].astype(str))
    links = pd.Index(masked["links"].astype(str))
    p = pred[pred.panel == panel].copy()
    p["ti"] = pd.to_datetime(p.timestamp, utc=True).map(time_index)
    p["si"] = stations.get_indexer(p.station_id.astype(str))
    p = p[(p.ti.notna()) & (p.si >= 0)]
    ti = p.ti.to_numpy(dtype=np.int64)
    si = p.si.to_numpy(dtype=np.int64)
    if hold_days is not None:
        keep = np.isin(ti // 288, hold_days)
        ti, si = ti[keep], si[keep]
        p = p[keep]
    reg = np.array([traffic.REGIME_NAMES.index(r) for r in p.mask_regime.astype(str)], dtype=np.int64)
    ps = pd.to_numeric(p.speed_kmh, errors="coerce").to_numpy(dtype=np.float64)
    pq = pd.to_numeric(p.flow_vph, errors="coerce").to_numpy(dtype=np.float64)
    true_s = plain["speed"].astype(np.float64)[ti, si]
    true_q = plain["flow"].astype(np.float64)[ti, si]
    lanes = masked["lanes"].astype(np.float64)[si]
    ok = np.isfinite(true_s) & np.isfinite(true_q)  # truth usable
    rows = {}
    for r in range(3):
        m = ok & (reg == r)
        if m.sum() == 0:
            rows[f"R{r+1}"] = dict(n=0, rmse_speed=np.nan, rmse_flow=np.nan, S_speed=0.0, S_flow=0.0, S_state=0.0)
            continue
        # missing predictions count as zero prediction, like the official scorer
        ppf_s = np.where(np.isfinite(ps[m]), ps[m], 0.0)
        ppf_q = np.where(np.isfinite(pq[m]), pq[m], 0.0)
        ds = ppf_s - true_s[m]
        dq = (ppf_q - true_q[m]) / lanes[m]
        rs = float(np.sqrt(np.mean(ds ** 2)))
        rq = float(np.sqrt(np.mean(dq ** 2)))
        ss = max(0.0, 1 - rs / 25.0)
        sq = max(0.0, 1 - rq / 600.0)
        rows[f"R{r+1}"] = dict(n=int(m.sum()), rmse_speed=rs, rmse_flow=rq, S_speed=ss, S_flow=sq,
                               S_state=0.54 * ss + 0.46 * sq)
    macro = float(np.mean([v["S_state"] for v in rows.values()]))
    return rows, macro, int(ok.sum()), int((~np.isfinite(ps)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", type=str, default=None)
    ap.add_argument("--panels", nargs="*", default=traffic.PANELS)
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--full-train", action="store_true", help="score on all train days instead of the holdout")
    args = ap.parse_args()
    pred = pd.read_csv(args.submission) if args.submission else None
    panel_macros = []
    for panel in args.panels:
        try:
            masked = traffic.load_mainline(panel, "train")["masked"]
        except FileNotFoundError:
            print(f"{panel}: no cache, skipped")
            continue
        n_days = len(masked["times"]) // 288
        hold = np.arange(n_days - args.holdout_days, n_days) if not args.full_train else None
        if pred is None:
            ti, si, reg, ps, pq = profile_predictions(panel, "train", hold)
            sub = pd.DataFrame({
                "panel": panel,
                "timestamp": traffic.epoch_to_iso(masked["times"][ti]),
                "station_id": masked["stations"][si].astype(str),
                "link_id": masked["links"][si].astype(str),
                "mask_regime": np.array(traffic.REGIME_NAMES)[reg],
                "speed_kmh": ps,
                "flow_vph": pq,
            })
        else:
            sub = pred
        rows, macro, n, nmiss = score(panel, sub, hold)
        panel_macros.append(macro)
        detail = " ".join(f"{k}={v['S_state']:.4f}" for k, v in rows.items())
        print(f"{panel:12s} macro={macro:.4f} {detail} n={n} missing={nmiss}", flush=True)
    print(f"\nALL PANELS mean S_state = {np.mean(panel_macros):.4f}")


if __name__ == "__main__":
    main()
