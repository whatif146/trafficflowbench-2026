"""Write the submitted per-row component arrays to a small compressed npz.

The merged submission is order-based (no natural keys inside it); the arrays are
written in exactly the submission row order, so a verifier can compare the final
file column-by-column without the 640 MB submission_key.csv.

Usage:
  python work/pipeline/make_components_npz.py --submission work/submissions/submission_final.csv \
      --out work/submissions/package_data/components.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cols = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]
    parts = {c: [] for c in cols}
    n = 0
    for chunk in pd.read_csv(args.submission, usecols=cols, chunksize=1_000_000):
        for c in cols:
            parts[c].append(chunk[c].to_numpy(dtype=np.float32))
        n += len(chunk)
    arrs = {c: np.concatenate(parts[c]) for c in cols}
    np.savez_compressed(args.out, **arrs)
    print(f"rows {n:,} -> {args.out} ({__import__('os').path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
