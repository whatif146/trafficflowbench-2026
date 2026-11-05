"""Ensemble test: blend the GBM holdout predictions with the base estimator.

For each panel with a saved holdout_<PANEL>.csv (produced by run_eval.py), the
base-only prediction (temporal interp + 10% spatial-neighbour mean, flow = 50/50
with spatial smoother) is recomputed for the same holdout rows, and
  pred(w) = w * GBM + (1-w) * base
is scored with the official Task 1 formula for w in the grid.

Alignment between the CSV (template holdout rows, possibly a random cap) and the
cache index arrays is done through (t_idx, station) keys, never row position.
"""
from __future__ import annotations

import argparse
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


def base_predictions(masked, ts, ss):
    """Base estimator rows for the given (t, s) rows (see t1gbm.FeatureBuilder)."""
    order_p, inv = link_perm(masked)
    sp, fl = masked["speed"], masked["flow"]
    nbs = temporal_neighbour_rows(sp, ts, ss)
    nbq = temporal_neighbour_rows(fl, ts, ss)
    s_intr = interp_from_neighbours(nbs)
    q_intr = interp_from_neighbours(nbq)
    s_nbr = spatial_nb_mean_rows(sp, ts, ss, order_p, inv)
    q_nbr = spatial_nb_mean_rows(fl, ts, ss, order_p, inv)
    base_s = np.where(np.isfinite(s_nbr), 0.9 * s_intr + 0.1 * s_nbr, s_intr)
    base_q_anchor = np.where(np.isfinite(q_nbr), 0.9 * q_intr + 0.1 * q_nbr, q_intr)
    f = smooth_field(fl, 1.5, 1.0, order=1, order_p=order_p, inv=inv)
    q_sm = f[ts, ss]
    del f
    gc.collect()
    base_q = np.where(np.isfinite(q_sm), 0.5 * q_sm + 0.5 * base_q_anchor, base_q_anchor)
    return base_s.astype(np.float64), base_q.astype(np.float64)


def align_csv(csv_path, masked, times, ts_hold, ss_hold):
    """Map the CSV rows onto the holdout-row positions; returns (pos, df, speed, flow)."""
    S = len(masked["stations"])
    df = pd.read_csv(csv_path)
    t_epoch = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("s").astype("int64").to_numpy()
    ti = np.searchsorted(times, t_epoch)
    ti = np.clip(ti, 0, len(times) - 1)
    if not np.all(times[ti] == t_epoch):
        raise SystemExit(f"{csv_path}: timestamps not on the cache grid")
    st_index = pd.Index(masked["stations"].astype(str))
    si = st_index.get_indexer(df["station_id"].astype(str))
    if (si < 0).any():
        raise SystemExit(f"{csv_path}: unknown station ids")
    key = ti.astype(np.int64) * S + si.astype(np.int64)
    hold_key = ts_hold.astype(np.int64) * S + ss_hold.astype(np.int64)
    order = np.argsort(hold_key)
    pos = np.searchsorted(hold_key[order], key)
    if (pos >= len(order)).any() or not np.all(hold_key[order][np.clip(pos, 0, len(order) - 1)] == key):
        raise SystemExit(f"{csv_path}: CSV keys not found in the holdout target rows")
    return order[pos], df, \
        pd.to_numeric(df["speed_kmh"], errors="coerce").to_numpy(np.float64), \
        pd.to_numeric(df["flow_vph"], errors="coerce").to_numpy(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*",
                    default=["D12_I5_N", "D7_I10_E", "D7_I405_S", "D7_I210_W"])
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--csv-dir", default=HERE)
    ap.add_argument("--w", nargs="*", type=float, default=[1.0, 0.95, 0.9, 0.8, 0.7, 0.6])
    args = ap.parse_args()

    w_grid = args.w
    totals = {w: [] for w in w_grid}
    per_panel = {}
    for panel in args.panels:
        csv_path = os.path.join(args.csv_dir, f"holdout_{panel}.csv")
        if not os.path.exists(csv_path):
            print(f"{panel}: no {csv_path}, skipped", flush=True)
            continue
        plain = traffic._load_or_build(panel, "train", "plain")
        masked = traffic._load_or_build(panel, "train", "masked")
        masked.pop("elig", None)
        masked.pop("regime", None)
        times = masked["times"]
        days = day_index(times)
        n_days = int(days.max()) + 1
        tg = traffic.load_targets(panel, "train")
        ts_all = tg["t_idx"].astype(np.int64)
        ss_all = tg["s_idx"].astype(np.int64)
        reg_all = tg["regime"].astype(np.int8) + 1
        fin = np.isfinite(plain["speed"][ts_all, ss_all]) & np.isfinite(plain["flow"][ts_all, ss_all])
        hold = (days[ts_all] >= n_days - args.holdout_days) & fin
        hold_pos = np.where(hold)[0]
        ts_h, ss_h, reg_h = ts_all[hold_pos], ss_all[hold_pos], reg_all[hold_pos]
        true_s = plain["speed"][ts_h, ss_h].astype(np.float64)
        true_q = plain["flow"][ts_h, ss_h].astype(np.float64)
        lanes = masked["lanes"][ss_h].astype(np.float64)
        del plain
        gc.collect()

        base_s, base_q = base_predictions(masked, ts_h, ss_h)
        pos, df, gbm_s, gbm_q = align_csv(csv_path, masked, times, ts_h, ss_h)
        # save the base-only predictions on exactly the rows of holdout_<PANEL>.csv
        base_frame = df[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
        base_frame["speed_kmh"] = np.round(base_s[pos], 3)
        base_frame["flow_vph"] = np.round(base_q[pos], 2)
        out_base = os.path.join(args.csv_dir, f"holdout_base_{panel}.csv")
        base_frame.to_csv(out_base, index=False)
        print(f"  wrote {out_base} rows={len(base_frame)}", flush=True)
        del base_frame
        full_s = np.full(len(hold_pos), np.nan)
        full_q = np.full(len(hold_pos), np.nan)
        full_s[pos] = gbm_s
        full_q[pos] = gbm_q
        mask_score = np.isfinite(full_s) | np.isfinite(full_q)
        print(f"\n[{panel}] holdout rows={len(hold_pos)} csv rows={len(pos)} scored={mask_score.sum()}",
              flush=True)
        rows = {}
        def blend(g, b, w):
            return np.where(np.isfinite(g) & np.isfinite(b), w * g + (1 - w) * b,
                            np.where(np.isfinite(g), g, b))
        for w in w_grid:
            ps = blend(full_s, base_s, w)
            pq = blend(full_q, base_q, w)
            res = score(true_s[mask_score], true_q[mask_score], ps[mask_score], pq[mask_score],
                        lanes[mask_score], reg_h[mask_score])
            rows[w] = res
            totals[w].append(res["macro"])
            print("  w=%.2f  macro=%.4f  R1=%.4f R2=%.4f R3=%.4f  rmse_s=%.2f rmse_q=%.0f" % (
                w, res["macro"], res[1]["S_state"], res[2]["S_state"], res[3]["S_state"],
                np.sqrt(np.mean([res[r]["rmse_speed"] ** 2 for r in (1, 2, 3)])),
                np.sqrt(np.mean([res[r]["rmse_flow"] ** 2 for r in (1, 2, 3)]))), flush=True)
        per_panel[panel] = rows
        del masked, base_s, base_q, full_s, full_q
        gc.collect()

    print("\n=== mean S_state across panels ===")
    for w in w_grid:
        print("  w=%.2f  mean=%.4f" % (w, float(np.mean(totals[w]))))
    best_w = max(w_grid, key=lambda w: np.mean(totals[w]))
    print(f"best w={best_w} mean={np.mean(totals[best_w]):.4f}")


if __name__ == "__main__":
    main()
