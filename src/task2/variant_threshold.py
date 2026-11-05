"""Generate Task 2 variants from the frozen v3 boosters (no retraining).

Variants: decision-threshold changes, onset top-k rule, and an optional
spatial morphological closing of the ongoing queue mask along the corridor
(milepost order), which encodes the physical fact that a jam is a contiguous
block of links.

Usage:
  python work/task2/variant_threshold.py --split both --on-th 0.30 --ong-th 0.40 \
      --close 1 --output work/task2/out/q_t40c1.csv
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C
import gbm_predict as GP


def closing(mask: np.ndarray, rank_to_idx: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological closing along the corridor order (fill gaps <= 2*radius)."""
    order = np.asarray(rank_to_idx, dtype=np.int64)          # rank position -> link index
    inv = np.argsort(order)                                   # link index -> rank position
    out = mask.copy()
    for h in range(out.shape[0]):
        m = out[h, order].copy()
        for i in range(len(m)):
            if m[i]:
                continue
            left = i - 1
            while left >= 0 and not m[left] and i - left <= 2 * radius:
                left -= 1
            right = i + 1
            while right < len(m) and not m[right] and right - i <= 2 * radius:
                right += 1
            if left >= 0 and right < len(m) and m[left] and m[right]:
                out[h, order[i]] = True
    return out


def predict_split(split: str, on_rule: list, ong_th: float, close: int = 0) -> pd.DataFrame:
    windows = C.released_windows()
    windows = windows[windows.split == split].copy()
    template = C.traffic.load_queue_template()
    template["window_id"] = template.window_id.astype(str)
    template["link_id"] = template.link_id.astype(str)
    template["timestamp"] = pd.to_datetime(template.timestamp, utc=True)
    template = template[template.window_id.isin(set(windows.window_id))].copy()
    history = C.released_history()
    history = history[history.window_id.isin(set(windows.window_id))]
    meta = GP.load_meta("v3")
    meta = copy.deepcopy(meta)
    meta["onset_rule"] = list(on_rule)
    meta["ongoing_th"] = float(ong_th)
    m_on, m_ong = GP.load_models("v3")
    rows = []
    for panel, wp in windows.groupby("panel", sort=True):
        clim = GP.attach_ramps(GP.load_clim(panel), split)
        links = np.asarray(clim["links"], dtype=str)
        r2i = np.asarray(clim["rank_to_idx"], dtype=np.int64)
        hp = history[history.window_id.isin(set(wp.window_id))]
        tp = template[template.window_id.isin(set(wp.window_id))]
        for r in wp.itertuples(index=False):
            wid = str(r.window_id)
            h = hp[hp.window_id == wid]
            tpl = tp[tp.window_id == wid]
            step_times = pd.DatetimeIndex(np.sort(tpl.timestamp.unique()))
            pred = GP.predict_window(h, clim, meta, m_on, m_ong, step_times)
            if close and r.condition == "queue_ongoing":
                pred = closing(pred, r2i, radius=close)
            li = tpl.link_id.map({l: i for i, l in enumerate(links)})
            ti = tpl.timestamp.map({t: i for i, t in enumerate(step_times)})
            vals = np.zeros(len(tpl), np.int8)
            ok = li.notna().to_numpy() & ti.notna().to_numpy()
            vals[ok] = pred[ti[ok].astype(int).to_numpy(), li[ok].astype(int).to_numpy()].astype(np.int8)
            sub = tpl[["window_id", "timestamp", "link_id"]].copy()
            sub["queue_pred"] = vals
            rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = out.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.sort_values(["window_id", "timestamp", "link_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "private", "both"], default="validation")
    ap.add_argument("--on-th", type=float, default=None)
    ap.add_argument("--on-top", type=int, default=None)
    ap.add_argument("--ong-th", type=float, required=True)
    ap.add_argument("--close", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rule = ["top", args.on_top] if args.on_top is not None else ["th", args.on_th]
    splits = ["validation", "private"] if args.split == "both" else [args.split]
    for split in splits:
        df = predict_split(split, rule, args.ong_th, args.close)
        out = args.output
        if len(splits) > 1:
            out = f"{args.output.rsplit('.csv', 1)[0]}_{split}.csv"
        df.to_csv(out, index=False)
        print(f"{split}: rows={len(df):,} positives={int(df.queue_pred.sum()):,} "
              f"windows={df.window_id.nunique()} -> {out}", flush=True)


if __name__ == "__main__":
    main()
