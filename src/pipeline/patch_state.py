"""Patch only the Task 1 (state) values of a merged submission, from dense arrays.

Streams the base submission and submission_key.csv in lockstep, so memory stays
small.  Used for fast A/B probes that change only the Task 1 numbers (T1 GBM
swap, physics-adjusted state, clipping variants).

Usage:
  python work/pipeline/patch_state.py --base work/submissions/submission_v2.csv \
      --state-dir work/submissions/dense --output work/submissions/submission_v3.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
import traffic

REL = traffic.REL
CHUNK = 500_000
KEY_COLS = ["submission_id", "task", "split", "panel", "timestamp", "station_id"]


def state_lookup(state_dir, panels):
    meta = {}
    for panel in panels:
        for split in ("validation", "private"):
            path = os.path.join(state_dir, f"{panel}_{split}.npz")
            if not os.path.exists(path):
                raise SystemExit(f"missing {path}")
            z = np.load(path)
            masked = traffic.load_mainline(panel, split)["masked"]
            iso = traffic.epoch_to_iso(masked["times"])
            meta[(panel, split)] = (z["speed"], z["flow"],
                                    z["mask"] if "mask" in z.files else None,
                                    {s: i for i, s in enumerate(iso)},
                                    {s: i for i, s in enumerate(masked["stations"].astype(str))})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--state-dir", default=r"C:\Users\lenovo\Desktop\kaggle\work\submissions\dense")
    ap.add_argument("--fallback-dir", default=None,
                    help="dense dir used where the primary has no value (coverage mask False)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--key", default=os.path.join(REL, "submission_key.csv"))
    args = ap.parse_args()
    smeta = state_lookup(args.state_dir, traffic.PANELS)
    fmeta = state_lookup(args.fallback_dir, traffic.PANELS) if args.fallback_dir else None
    base = pd.read_csv(args.base, chunksize=CHUNK)
    keys = pd.read_csv(args.key, usecols=KEY_COLS, chunksize=CHUNK,
                       dtype={"panel": str, "timestamp": str, "station_id": str, "task": str, "split": str})
    written = patched = fallback_used = 0
    first = True
    for b, k in zip(base, keys):
        if len(b) != len(k) or not (b.submission_id.to_numpy() == k.submission_id.to_numpy()).all():
            raise SystemExit("base/key not aligned")
        m = (k.task.to_numpy() == "state")
        if m.any():
            panel = k.panel.to_numpy()[m]
            split = k.split.to_numpy()[m]
            ts = k.timestamp.to_numpy()[m]
            st = k.station_id.to_numpy()[m]
            sp = np.zeros(int(m.sum()), np.float32)
            fl = np.zeros(int(m.sum()), np.float32)
            for pp, ss in set(zip(panel.tolist(), split.tolist())):
                idx = np.flatnonzero((panel == pp) & (split == ss))
                speed, flow, mask, times, stations = smeta[(pp, ss)]
                ti = np.fromiter((times.get(t, -1) for t in ts[idx]), np.int64, len(idx))
                si = np.fromiter((stations.get(s, -1) for s in st[idx]), np.int64, len(idx))
                good = (ti >= 0) & (si >= 0)
                sp[idx[good]] = speed[ti[good], si[good]]
                fl[idx[good]] = flow[ti[good], si[good]]
                patched += int(good.sum())
                if mask is not None and fmeta is not None:
                    have = np.zeros(len(idx), bool)
                    have[good] = mask[ti[good], si[good]]
                    need = ~have
                    if need.any():
                        fs, fq, _, ftimes, fstations = fmeta[(pp, ss)]
                        fti = np.fromiter(((ftimes.get(ts[idx][j], -1)) for j in np.flatnonzero(need)),
                                          np.int64, int(need.sum()))
                        fsi = np.fromiter(((fstations.get(st[idx][j], -1)) for j in np.flatnonzero(need)),
                                          np.int64, int(need.sum()))
                        ok = (fti >= 0) & (fsi >= 0)
                        sel = np.flatnonzero(need)[ok]
                        sp[sel] = fs[fti[ok], fsi[ok]]
                        fl[sel] = fq[fti[ok], fsi[ok]]
                        fallback_used += int(len(sel))
            b.loc[m, "speed_kmh"] = sp
            b.loc[m, "flow_vph"] = fl
        b.to_csv(args.output, mode="w" if first else "a", header=first, index=False)
        first = False
        written += len(b)
    print(f"wrote {written:,} rows, patched {patched:,} state rows "
          f"(fallback filled {fallback_used:,}) -> {args.output}")


if __name__ == "__main__":
    main()
