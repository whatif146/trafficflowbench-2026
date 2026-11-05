"""Patch one Task's values inside an already merged submission (fast A/B probes).

The merged submission has no natural keys, so this streams it in lockstep with
`submission_key.csv` (same row order), rebuilds the natural key of each row, and
rewrites only the rows of the chosen task.  Everything else is copied through,
so the result is identical to the base file outside that task.

Usage:
  python work/pipeline/patch_task.py --base work/submissions/submission_v1.csv \
      --task queue --source work/task2/out/queue_gbmv3_all.csv \
      --output work/submissions/submission_v2.csv
"""
from __future__ import annotations

import argparse
import sys
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
REL = os.environ.get("TFB_REL", os.path.join(ROOT, "kaggle_public"))
CHUNK = 1_000_000
KEY_COLS = ["submission_id", "task", "window_id", "panel", "timestamp", "link_id",
            "departure_time", "path_id"]


def load_map(task, source):
    if task == "queue":
        df = pd.read_csv(source, dtype={"window_id": str, "timestamp": str, "link_id": str})
        df["timestamp"] = pd.to_datetime(df.timestamp, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        keys = pd.MultiIndex.from_frame(df[["window_id", "timestamp", "link_id"]])
        return keys, pd.to_numeric(df.queue_pred, errors="coerce").to_numpy(np.float32), "queue_pred"
    if task == "odme":
        df = pd.read_csv(source, dtype={"panel": str, "departure_time": str, "path_id": str})
        keys = pd.MultiIndex.from_frame(df[["panel", "departure_time", "path_id"]])
        return keys, pd.to_numeric(df.path_flow, errors="coerce").to_numpy(np.float32), "path_flow"
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--key", default=os.path.join(REL, "submission_key.csv"))
    ap.add_argument("--task", required=True, choices=["queue", "odme"])
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    keys, values, col = load_map(args.task, args.source)
    lookup = pd.Series(values, index=keys)

    base_reader = pd.read_csv(args.base, chunksize=CHUNK)
    key_reader = pd.read_csv(args.key, usecols=KEY_COLS, chunksize=CHUNK,
                             dtype={"window_id": str, "panel": str, "timestamp": str,
                                    "link_id": str, "departure_time": str, "path_id": str})
    written = patched = 0
    first = True
    for b, k in zip(base_reader, key_reader):
        if len(b) != len(k) or not (b.submission_id.to_numpy() == k.submission_id.to_numpy()).all():
            raise SystemExit("base and submission_key are not aligned")
        task = k.task.astype(str).to_numpy()
        rows = task == args.task
        if rows.any():
            if args.task == "queue":
                kk = pd.MultiIndex.from_frame(
                    k.loc[rows, ["window_id", "timestamp", "link_id"]])
            else:
                kk = pd.MultiIndex.from_frame(
                    k.loc[rows, ["panel", "departure_time", "path_id"]])
            v = lookup.reindex(kk).to_numpy()
            found = np.isfinite(v)
            patched += int(found.sum())
            nr = int(rows.sum())
            b.loc[rows, col] = np.where(found, v, 0.0)
        if (task != args.task).any():
            pass
        b.to_csv(args.output, mode="w" if first else "a", header=first, index=False)
        first = False
        written += len(b)
    print(f"wrote {written:,} rows ({patched:,} {args.task} values patched, "
          f"{len(values) - patched} source rows unmatched) -> {args.output}")


if __name__ == "__main__":
    main()
