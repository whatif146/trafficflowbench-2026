"""Verify the submitted final file against the shipped per-row component arrays.

The merged submission is order-based, so the package ships `components.npz` whose
arrays (`speed_kmh`, `flow_vph`, `queue_pred`, `path_flow`) are exactly the values
in submission row order. This script checks that the submission inside
`submissions/submission_final.zip` reproduces those arrays (CSV rounding only).

Usage (from the package root):
  python code/verify_final.py
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLS = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def main():
    final_zip = os.path.join(BASE, "submissions", "submission_final.zip")
    comp = np.load(os.path.join(BASE, "submissions", "package_data", "components.npz"))
    with zipfile.ZipFile(final_zip) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            df = pd.read_csv(io.BytesIO(fh.read()), usecols=COLS)
    assert len(df) == len(comp["speed_kmh"]), f"row mismatch {len(df)} vs {len(comp['speed_kmh'])}"
    ok = True
    for c in COLS:
        a = df[c].to_numpy(dtype=np.float64)
        b = comp[c].astype(np.float64)
        d = np.abs(a - b)
        rel = d.max()
        print(f"{c:12s} rows {len(a):,}  max|diff| {rel:.6g}")
        tol = {"speed_kmh": 5e-3, "flow_vph": 5e-2, "queue_pred": 1e-6, "path_flow": 1e-3}[c]
        ok = ok and rel < tol
    print("VERIFIED" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
