"""Fast, low-memory replacement for src/merge_submissions.py.

Streams `submission_key.csv` in chunks and fills the four value columns from
  * dense Task 1 arrays produced by work/pipeline/pack_state.py,
  * a Task 2 queue CSV (window_id,timestamp,link_id,queue_pred),
  * a Task 4 path-flow CSV (panel,departure_time,path_id,...,path_flow).
It then optionally verifies ids/task/row-count against sample_submission.csv.

Usage:
  python work/pipeline/fast_merge.py \
      --state-dir work/submissions/dense \
      --queue work/task2/out/queue_gbmv3_all.csv \
      --odme  work/task4/out/path_flow_validation_plus_private.csv \
      --output work/submissions/submission_v3.csv \
      --check
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
CHUNK = 750_000
OUT_COLS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
KEY_COLS = ["submission_id", "task", "split", "panel", "timestamp", "station_id", "link_id",
            "mask_regime", "window_id", "departure_time", "path_id"]


def state_lookup(state_dir, panels):
    meta = {}
    for panel in panels:
        for split in ("validation", "private"):
            path = os.path.join(state_dir, f"{panel}_{split}.npz")
            if not os.path.exists(path):
                continue
            z = np.load(path)
            speed, flow = z["speed"], z["flow"]
            masked = traffic.load_mainline(panel, split)["masked"]
            iso = traffic.epoch_to_iso(masked["times"])
            times = {s: i for i, s in enumerate(iso)}
            stations = {s: i for i, s in enumerate(masked["stations"].astype(str))}
            meta[(panel, split)] = (speed, flow, times, stations)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--odme", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--key", default=os.path.join(REL, "submission_key.csv"))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    smeta = state_lookup(args.state_dir, traffic.PANELS)
    q = pd.read_csv(args.queue, dtype={"window_id": str, "timestamp": str, "link_id": str})
    q["timestamp"] = pd.to_datetime(q.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    qindex = pd.MultiIndex.from_frame(q[["window_id", "timestamp", "link_id"]])
    qmap = pd.Series(pd.to_numeric(q.queue_pred, errors="coerce").to_numpy(np.float32), index=qindex)

    o = pd.read_csv(args.odme, dtype={"panel": str, "departure_time": str, "path_id": str})
    oindex = pd.MultiIndex.from_frame(o[["panel", "departure_time", "path_id"]])
    omap = pd.Series(pd.to_numeric(o.path_flow, errors="coerce").to_numpy(np.float32), index=oindex)

    reader = pd.read_csv(args.key, usecols=KEY_COLS, chunksize=CHUNK,
                         dtype={c: str for c in ("window_id", "panel", "timestamp", "station_id",
                                                 "link_id", "mask_regime", "departure_time",
                                                 "path_id", "task", "split")})
    written = 0
    first = True
    for k in reader:
        n = len(k)
        out = pd.DataFrame({"submission_id": k.submission_id.astype("int64"),
                            "task": k.task.astype(str)})
        for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
            out[c] = np.zeros(n, dtype=np.float64)

        # ---- state rows
        m = (out.task == "state").to_numpy()
        if m.any():
            panel = k.panel.to_numpy()[m]
            split = k.split.to_numpy()[m]
            ts = k.timestamp.to_numpy()[m]
            st = k.station_id.to_numpy()[m]
            sp = np.zeros(int(m.sum()), dtype=np.float32)
            fl = np.zeros(int(m.sum()), dtype=np.float32)
            for (pp, ss) in set(zip(panel.tolist(), split.tolist())):
                idx = np.flatnonzero((panel == pp) & (split == ss))
                if (pp, ss) not in smeta:
                    raise SystemExit(f"missing dense state for {pp} {ss}")
                speed, flow, times, stations = smeta[(pp, ss)]
                ti = np.fromiter((times.get(t, -1) for t in ts[idx]), dtype=np.int64, count=len(idx))
                si = np.fromiter((stations.get(s, -1) for s in st[idx]), dtype=np.int64, count=len(idx))
                good = (ti >= 0) & (si >= 0)
                sp[idx[good]] = speed[ti[good], si[good]]
                fl[idx[good]] = flow[ti[good], si[good]]
            out.loc[m, "speed_kmh"] = sp
            out.loc[m, "flow_vph"] = fl

        # ---- queue rows
        m = (out.task == "queue").to_numpy()
        if m.any():
            kk = pd.MultiIndex.from_frame(k.loc[m, ["window_id", "timestamp", "link_id"]])
            v = qmap.reindex(kk).to_numpy()
            out.loc[m, "queue_pred"] = np.where(np.isfinite(v), v, 0.0)

        # ---- odme rows
        m = (out.task == "odme").to_numpy()
        if m.any():
            kk = pd.MultiIndex.from_frame(k.loc[m, ["panel", "departure_time", "path_id"]])
            v = omap.reindex(kk).to_numpy()
            out.loc[m, "path_flow"] = np.where(np.isfinite(v), v, 0.0)

        out[OUT_COLS].to_csv(args.output, mode="w" if first else "a", header=first, index=False)
        first = False
        written += n
        print(f"  {written:,} rows", end="\\r", flush=True)
    print(f"\\nwrote {written:,} rows -> {args.output}", flush=True)

    if args.check:
        sample = pd.read_csv(os.path.join(REL, "sample_submission.csv"),
                             usecols=["submission_id", "task"], chunksize=CHUNK)
        mine = pd.read_csv(args.output, usecols=["submission_id", "task"], chunksize=CHUNK)
        n = 0
        for a, b in zip(sample, mine):
            n += len(a)
            if len(a) != len(b) or not (a.submission_id.to_numpy() == b.submission_id.to_numpy()).all():
                raise SystemExit("id mismatch with sample_submission")
            if not (a.task.to_numpy() == b.task.to_numpy()).all():
                raise SystemExit("task mismatch with sample_submission")
        print(f"check OK: {n:,} rows aligned with sample_submission")


if __name__ == "__main__":
    main()
