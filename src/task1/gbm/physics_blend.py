"""Physics-motivated blend experiment for Task 3 calibration on train.

Builds  pred(alpha) = alpha * GBM + (1 - alpha) * base  files over the full train
template of one panel (GBM = the 2-fold cross-fitted full-train predictions, so
no row is in-sample), reports the official Task-1 S_state for each alpha, and
writes one CSV per alpha for work/task3/local_eval.py.

  python physics_blend.py --panel D7_I10_E \\
      --gbm state_gbm_train_D7_I10_E.csv --outdir blend_train \\
      --alphas 0.9 0.8 0.7 0.5
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import traffic  # noqa: E402

from t1lib import (day_index, link_perm, temporal_neighbour_rows, interp_from_neighbours,
                   spatial_nb_mean_rows, smooth_field, score, fmt)


def base_rows(masked, ts, ss):
    order_p, inv = link_perm(masked)
    sp, fl = masked["speed"], masked["flow"]
    nbs = temporal_neighbour_rows(sp, ts, ss)
    nbq = temporal_neighbour_rows(fl, ts, ss)
    s_intr = interp_from_neighbours(nbs)
    q_intr = interp_from_neighbours(nbq)
    del nbs, nbq
    s_nbr = spatial_nb_mean_rows(sp, ts, ss, order_p, inv)
    q_nbr = spatial_nb_mean_rows(fl, ts, ss, order_p, inv)
    base_s = np.where(np.isfinite(s_nbr), 0.9 * s_intr + 0.1 * s_nbr, s_intr)
    q_anchor = np.where(np.isfinite(q_nbr), 0.9 * q_intr + 0.1 * q_nbr, q_intr)
    f = smooth_field(fl, 1.5, 1.0, order=1, order_p=order_p, inv=inv)
    q_sm = f[ts, ss]
    del f
    gc.collect()
    base_q = np.where(np.isfinite(q_sm), 0.5 * q_sm + 0.5 * q_anchor, q_anchor)
    return base_s.astype(np.float64), base_q.astype(np.float64)


def write_state_csv(path, panel, ts_str, st_str, lk_str, reg_str, speed, flow, chunk=300000):
    n = len(speed)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"])
        for lo in range(0, n, chunk):
            hi = min(n, lo + chunk)
            for i in range(lo, hi):
                w.writerow((panel, ts_str[i], st_str[i], lk_str[i], reg_str[i],
                            format(speed[i], ".3f"), format(flow[i], ".2f")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="D7_I10_E")
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--gbm", default=None)
    ap.add_argument("--outdir", default=os.path.join(HERE, "blend_train"))
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.9, 0.8, 0.7, 0.5])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    gbm_path = args.gbm or os.path.join(HERE, f"state_gbm_train_{args.panel}.csv")

    plain = traffic._load_or_build(args.panel, "train", "plain")
    masked = traffic._load_or_build(args.panel, "train", "masked")
    masked.pop("elig", None)
    masked.pop("regime", None)
    times = masked["times"]
    days = day_index(times)
    n_days = int(days.max()) + 1
    tg = traffic.load_targets(args.panel, "train")
    ts, ss = tg["t_idx"].astype(np.int64), tg["s_idx"].astype(np.int64)
    reg = tg["regime"].astype(np.int8) + 1
    true_s = plain["speed"][ts, ss].astype(np.float64)
    true_q = plain["flow"][ts, ss].astype(np.float64)
    lanes = masked["lanes"][ss].astype(np.float64)
    with np.errstate(all="ignore"):
        lo_s = np.nanpercentile(plain["speed"], 0.5, axis=0).astype(np.float32)
        hi_s = np.nanpercentile(plain["speed"], 99.5, axis=0).astype(np.float32)
        hi_q = np.nanpercentile(plain["flow"], 99.5, axis=0).astype(np.float32)
    del plain
    gc.collect()

    base_s, base_q = base_rows(masked, ts, ss)
    print(f"[{args.panel}] base rows {len(ts)}", flush=True)
    gbm = pd.read_csv(gbm_path, usecols=["speed_kmh", "flow_vph"])
    if len(gbm) != len(ts):
        raise SystemExit(f"GBM csv rows {len(gbm)} != template rows {len(ts)}")
    gbm_s = gbm["speed_kmh"].to_numpy(np.float64)
    gbm_q = gbm["flow_vph"].to_numpy(np.float64)
    del gbm
    gc.collect()

    def clip(speed, flow):
        lo = np.maximum(lo_s[ss].astype(np.float64), 1.0)
        hi = np.where(np.isfinite(hi_s[ss]), hi_s[ss], masked["free_speed"][ss] * 1.1).astype(np.float64)
        hq = np.where(np.isfinite(hi_q[ss]), hi_q[ss], 1e9).astype(np.float64)
        return np.clip(speed, lo, hi), np.clip(flow, 0.0, hq)

    base_s, base_q = clip(base_s, base_q)
    gbm_s, gbm_q = clip(gbm_s, gbm_q)

    ts_str = traffic.epoch_to_iso(times[ts])
    st_str = masked["stations"][ss].astype(str)
    lk_str = masked["links"][ss].astype(str)
    reg_str = np.array(traffic.REGIME_NAMES)[reg - 1]

    print("\n alpha   S_state    S_speed    S_flow   speedRMSE flowRMSE  file")
    rows = [("gbm", gbm_s, gbm_q)]
    for a in args.alphas:
        rows.append((f"a{a:.1f}", a * gbm_s + (1 - a) * base_s, a * gbm_q + (1 - a) * base_q))
    for tag, ps, pq in rows:
        res = score(true_s, true_q, ps, pq, lanes, reg)
        sm = float(np.mean([res[r]["S_speed"] for r in (1, 2, 3)]))
        qm = float(np.mean([res[r]["S_flow"] for r in (1, 2, 3)]))
        rs = float(np.sqrt(np.mean([res[r]["rmse_speed"] ** 2 for r in (1, 2, 3)])))
        rq = float(np.sqrt(np.mean([res[r]["rmse_flow"] ** 2 for r in (1, 2, 3)])))
        out = os.path.join(args.outdir, f"state_{tag}_train_{args.panel}.csv")
        write_state_csv(out, args.panel, ts_str, st_str, lk_str, reg_str, ps, pq)
        print("%-8s %.4f     %.4f     %.4f   %.2f      %.1f   %s" % (
            tag, res["macro"], sm, qm, rs, rq, out), flush=True)


if __name__ == "__main__":
    main()
