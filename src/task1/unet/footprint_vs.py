"""Footprint of a candidate state CSV vs another state CSV (same template rows).

    python footprint_vs.py --new state_gbm_unet3_validation.csv --ref state_gbm_unet_validation.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\lenovo\Desktop\kaggle"
HERE = os.path.join(ROOT, "work", "task1", "unet")


def compare(new_path: str, ref_path: str):
    d = pd.read_csv(new_path, dtype={"panel": str})
    r = pd.read_csv(ref_path, dtype={"panel": str})
    if len(d) != len(r):
        raise SystemExit(f"row count {len(d)} != {len(r)}")
    per = {}
    for panel in d["panel"].unique():
        m = d["panel"].to_numpy() == panel
        gs = r.loc[m, "speed_kmh"].to_numpy(np.float64)
        gq = r.loc[m, "flow_vph"].to_numpy(np.float64)
        vs = d.loc[m, "speed_kmh"].to_numpy(np.float64)
        vq = d.loc[m, "flow_vph"].to_numpy(np.float64)
        rel = np.abs(vq - gq) / np.maximum(np.abs(gq), 1.0)
        per[panel] = {
            "rows": int(m.sum()),
            "mean_abs_dv_kmh": float(np.mean(np.abs(vs - gs))),
            "mean_abs_dq_vph": float(np.mean(np.abs(vq - gq))),
            "frac_dq_gt1pct": float(np.mean(rel > 0.01)),
            "frac_dq_gt5pct": float(np.mean(rel > 0.05)),
        }
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    per = compare(os.path.join(HERE, args.new), os.path.join(HERE, args.ref))
    keys = ("mean_abs_dv_kmh", "mean_abs_dq_vph", "frac_dq_gt1pct", "frac_dq_gt5pct")
    mean = {k: float(np.mean([per[p][k] for p in per])) for k in keys}
    for panel, v in per.items():
        print(f"  {panel:12s} rows={v['rows']:8d} mean|dv|={v['mean_abs_dv_kmh']:.3f} "
              f"mean|dq|={v['mean_abs_dq_vph']:.2f} frac>1%={v['frac_dq_gt1pct']:.3f} "
              f"frac>5%={v['frac_dq_gt5pct']:.3f}")
    print(f"  MEAN         mean|dv|={mean['mean_abs_dv_kmh']:.3f} "
          f"mean|dq|={mean['mean_abs_dq_vph']:.2f} frac>1%={mean['frac_dq_gt1pct']:.3f} "
          f"frac>5%={mean['frac_dq_gt5pct']:.3f}")
    if args.json:
        out = {"ref": args.ref, "new": args.new, "per_panel": per, "mean": mean}
        json.dump(out, open(os.path.join(HERE, args.json), "w", encoding="utf-8"), indent=2)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
