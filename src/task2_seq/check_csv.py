"""Structural + determinism checks for the candidate CSVs."""
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import traffic


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


wi = traffic.load_window_index()
for split in ("validation", "private"):
    path = os.path.join(LH.OUT, f"queue_v5_{split}.csv")
    d = pd.read_csv(path, dtype={"window_id": str, "link_id": str})
    w = wi[wi.split == split]
    origin = dict(zip(w.window_id.astype(str), pd.to_datetime(w.forecast_origin, utc=True)))
    end = dict(zip(w.window_id.astype(str), pd.to_datetime(w.forecast_end, utc=True)))
    cond = dict(zip(w.window_id.astype(str), w.condition))
    ts = pd.to_datetime(d.timestamp, utc=True)
    d["is_end"] = ts == d.window_id.map(end)
    d["condition"] = d.window_id.map(cond)
    on = d[d.condition == "queue_onset"]
    print(f"\n{split}: {len(d):,} rows, {d.window_id.nunique()} windows")
    print("  onset positives outside the last step:",
          int((on[on.queue_pred == 1].is_end == False).sum()), "(must be 0)")
    g = on.groupby("window_id").queue_pred.sum()
    print("  onset positives/window: min", int(g.min()), "median", float(g.median()),
          "max", int(g.max()), "| windows with 0:", int((g == 0).sum()))
    og = d[d.condition == "queue_ongoing"].groupby("window_id").queue_pred.sum()
    L = len(d.link_id.unique())
    print("  ongoing positives/window: median", float(og.median()), "max", int(og.max()),
          f"| links {L}")
    print("  md5", md5(path))
