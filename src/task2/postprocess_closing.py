"""Apply a spatial morphological closing to a binary Task-2 queue submission.

Physical prior from the release analysis: a jam occupies a contiguous block of
mainline links (fixed head, tail growing upstream), so an isolated gap between
two queued neighbours of the same corridor is very likely queued as well.

Usage:
  python work/task2/postprocess_closing.py --input work/task2_seq/out/queue_v5_validation.csv \
      --output work/task2_seq/out/queue_v5c1_validation.csv --radius 1
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C


def close_mask(mask: np.ndarray, rank_to_idx: np.ndarray, radius: int) -> np.ndarray:
    order = np.asarray(rank_to_idx, dtype=np.int64)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--radius", type=int, default=1)
    args = ap.parse_args()
    d = pd.read_csv(args.input, dtype={"window_id": str, "link_id": str})
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
    windows = C.released_windows()
    win_panel = windows.set_index("window_id").panel.astype(str).to_dict()
    win_split = windows.set_index("window_id").split.astype(str).to_dict()
    clim_cache = {}
    total_changed = 0
    for wid, g in d.groupby("window_id", sort=True):
        panel = win_panel[wid]
        split = win_split[wid]
        key = (panel, split)
        if key not in clim_cache:
            base = C.load_masked_panel(panel, split)
            clim_cache[key] = C.panel_meta(panel, base["links"].astype(str))
        clim = clim_cache[key]
        links = list(clim["links"])
        idx = {str(l): i for i, l in enumerate(links)}
        steps = pd.DatetimeIndex(np.sort(g.timestamp.unique()))
        si = {t: i for i, t in enumerate(steps)}
        mask = np.zeros((len(steps), len(links)), bool)
        ti = g.timestamp.map(si).to_numpy()
        li = g.link_id.map(idx).to_numpy()
        ti = ti.astype(int)
        li = li.astype(int)
        mask[ti, li] = g.queue_pred.to_numpy(bool)
        closed = close_mask(mask, clim["rank_to_idx"], args.radius)
        new = closed[ti, li] != mask[ti, li]
        if new.any():
            d.loc[g.index[new], "queue_pred"] = 1
            total_changed += int(new.sum())
    d["timestamp"] = d.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    d.to_csv(args.output, index=False)
    print(f"closing radius {args.radius}: +{total_changed:,} cells set to 1; "
          f"positives {int(d.queue_pred.sum()):,} -> {args.output}")


if __name__ == "__main__":
    main()
