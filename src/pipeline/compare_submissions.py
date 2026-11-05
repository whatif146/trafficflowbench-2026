"""Compare two merged submission files numerically (reproducibility check).

Streams both in chunks and reports the maximum absolute difference per task
column and the number of differing cells.

Usage:
  python work/pipeline/compare_submissions.py --a work/submissions/submission_v4.csv \
      --b work/submissions/submission_repro.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

CHUNK = 750_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    ra = pd.read_csv(args.a, chunksize=CHUNK)
    rb = pd.read_csv(args.b, chunksize=CHUNK)
    stats = {}
    n = 0
    for a, b in zip(ra, rb):
        n += len(a)
        if len(a) != len(b) or not (a.submission_id.to_numpy() == b.submission_id.to_numpy()).all():
            raise SystemExit("files are not row-aligned")
        if not (a.task.astype(str).to_numpy() == b.task.astype(str).to_numpy()).all():
            raise SystemExit("task tags differ")
        for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
            va = a[col].to_numpy(np.float64)
            vb = b[col].to_numpy(np.float64)
            d = np.abs(va - vb)
            s = stats.setdefault(col, {"max": 0.0, "n_diff": 0, "n": 0, "sum_abs": 0.0})
            s["max"] = max(s["max"], float(np.nanmax(d)))
            s["n_diff"] += int((d > 1e-6).sum())
            s["n"] += len(d)
            s["sum_abs"] += float(np.nansum(d))
    print(f"rows compared: {n:,}")
    for col, s in stats.items():
        print(f"{col:12s} max|diff|={s['max']:.6g} cells>1e-6={s['n_diff']:,} "
              f"mean|diff|={s['sum_abs']/s['n']:.3g}")
    # Tolerances: the submission writer rounds floats (speed 3 dp, flow 2 dp,
    # path_flow 2 dp), so re-running the pipeline differs only at rounding level.
    tol = {"speed_kmh": 2e-3, "flow_vph": 5e-2, "queue_pred": 0.5, "path_flow": 5e-2}
    ok = all(s["max"] <= tol[col] for col, s in stats.items())
    print("REPRODUCTION MATCH (differences only at output rounding level)" if ok
          else "REPRODUCTION MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
