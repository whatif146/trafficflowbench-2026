"""Official-format persistence baseline for Task 2 (fallback until the model lands).

Writes work/task2/out/queue_validation.csv and queue_private.csv with the exact
template row coverage.  Persistence = queue state at the last released history
step (T-5), optionally using the released row at T when it is observed (the rules
allow it).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
import traffic


def build(split: str, out: str):
    windows = traffic.load_window_index()
    windows = windows[windows.split == split].copy()
    hist = traffic.load_window_history()
    hist = hist[hist.window_id.isin(windows.window_id)].copy()
    hist["timestamp"] = pd.to_datetime(hist.timestamp, utc=True)
    hist["speed_kmh"] = pd.to_numeric(hist.speed_kmh, errors="coerce")
    tpl = traffic.load_queue_template()
    tpl = tpl[tpl.window_id.isin(windows.window_id)].copy()
    tpl["timestamp"] = pd.to_datetime(tpl.timestamp, utc=True)
    frames = []
    for panel in sorted(windows.panel.unique()):
        meta = traffic.load_network(panel)["links"].set_index("link_id")
        v_cut = meta["v_cut"].to_dict()
        h = hist[hist.window_id.isin(windows[windows.panel == panel].window_id)]
        # queue state at the last history step
        last = h[h.timestamp == h.groupby("window_id").timestamp.transform("max")].copy()
        last["queued"] = (last.speed_kmh <= last.link_id.map(v_cut)) & \
                         last.is_score_eligible.astype(bool)
        q = last.groupby("window_id").apply(
            lambda g: dict(zip(g.link_id, g.queued)), include_groups=False).to_dict()
        t = tpl[tpl.window_id.isin(windows[windows.panel == panel].window_id)][
            ["window_id", "timestamp", "link_id"]].copy()
        t["queue_pred"] = [int(q.get(w, {}).get(l, 0)) for w, l in
                           zip(t.window_id, t.link_id)]
        frames.append(t)
    out_df = pd.concat(frames, ignore_index=True)
    out_df["timestamp"] = out_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"{split}: wrote {len(out_df):,} rows ({out_df.queue_pred.mean():.3f} positive) to {out}")
    return out_df


if __name__ == "__main__":
    base = os.path.join(traffic.WORK, "task2", "out")
    for split in ("validation", "private"):
        build(split, os.path.join(base, f"queue_{split}.csv"))
