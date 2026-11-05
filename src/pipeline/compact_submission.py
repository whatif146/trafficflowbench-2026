"""Rewrite a merged submission compactly (fewer decimals) to shrink the upload.

Float columns are written with 3 decimals (ranges: speed ~1e2, flow ~1e3-1e4,
path_flow ~1e3; 3 dp costs nothing for the scores), queue_pred is cast to int so
it is written exactly as 0/1, and submission_id stays an int.

Usage:
  python work/pipeline/compact_submission.py --input work/submissions/submission_v4.csv \
      --output work/submissions/submission_v4_compact.csv
"""
from __future__ import annotations

import argparse
import os
import zipfile

import numpy as np
import pandas as pd

CHUNK = 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    first = True
    n = 0
    for chunk in pd.read_csv(args.input, chunksize=CHUNK):
        chunk["submission_id"] = chunk.submission_id.astype(np.int64)
        chunk["queue_pred"] = np.rint(chunk.queue_pred.to_numpy(np.float64)).astype(np.int64)
        chunk.to_csv(args.output, mode="w" if first else "a", header=first, index=False,
                     float_format="%.3f")
        first = False
        n += len(chunk)
    print(f"wrote {n:,} rows -> {args.output} ({os.path.getsize(args.output)/1e6:.1f} MB)")
    if args.zip:
        zpath = args.output[:-4] + ".zip" if args.output.endswith(".csv") else args.output + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            z.write(args.output, arcname="submission.csv")
        print(f"zipped -> {zpath} ({os.path.getsize(zpath)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
