"""Verify a deployed state CSV against the published template and the shipped GBM file.

Checks: row count, column order/names, and that panel/timestamp/station_id/link_id/
mask_regime are identical row-for-row to the shipped state_gbm_<split>.csv and to
the published sample_submission_state.csv; no NaNs in speed/flow; reports the
footprint vs the shipped GBM predictions.

    python verify_state.py --split validation
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
import traffic  # noqa: E402

GBM_DIR = os.path.join(ROOT, "work", "task1", "gbm")
HERE = os.path.join(ROOT, "work", "task1", "unet")
KEYS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
COLS = KEYS + ["speed_kmh", "flow_vph"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--path", default=None)
    args = ap.parse_args()
    path = args.path or os.path.join(HERE, f"state_gbm_unet_{args.split}.csv")
    d = pd.read_csv(path, dtype={"timestamp": str, "station_id": str, "link_id": str,
                                 "panel": str, "mask_regime": str})
    g = pd.read_csv(os.path.join(GBM_DIR, f"state_gbm_{args.split}.csv"),
                    dtype={"timestamp": str, "station_id": str, "link_id": str,
                           "panel": str, "mask_regime": str})
    print(f"{os.path.basename(path)}: {len(d)} rows, columns={list(d.columns)}")
    assert list(d.columns) == COLS, f"column order mismatch: {list(d.columns)}"
    assert len(d) == len(g), f"row count {len(d)} != shipped GBM {len(g)}"
    for k in KEYS:
        if not (d[k].to_numpy() == g[k].to_numpy()).all():
            raise SystemExit(f"key column {k} differs from the shipped GBM file")
    assert d["speed_kmh"].notna().all() and d["flow_vph"].notna().all(), "NaN in output"
    print("  keys/order identical to the shipped GBM file; no NaNs")
    # per-panel template check
    for panel in d["panel"].unique():
        p = d[d["panel"] == panel]
        tpl = pd.read_csv(os.path.join(traffic.REL, "task1", panel, args.split,
                                       "sample_submission_state.csv"),
                          dtype={"timestamp": str, "station_id": str, "link_id": str,
                                 "panel": str, "mask_regime": str})
        ok = (len(p) == len(tpl)
              and (p["timestamp"].to_numpy() == tpl["timestamp"].to_numpy()).all()
              and (p["station_id"].to_numpy() == tpl["station_id"].to_numpy()).all()
              and (p["link_id"].to_numpy() == tpl["link_id"].to_numpy()).all()
              and (p["mask_regime"].to_numpy() == tpl["mask_regime"].to_numpy()).all())
        print(f"  {panel:12s} rows={len(p):8d} template-exact={ok}")
        assert ok, panel
    ds = np.abs(d["speed_kmh"].to_numpy() - g["speed_kmh"].to_numpy())
    dq = np.abs(d["flow_vph"].to_numpy() - g["flow_vph"].to_numpy())
    rel = dq / np.maximum(np.abs(g["flow_vph"].to_numpy()), 1.0)
    print(f"footprint vs shipped GBM: mean|dv|={ds.mean():.3f} km/h p99={np.percentile(ds, 99):.2f} "
          f"mean|dq|={dq.mean():.2f} vph frac>1%={np.mean(rel > 0.01):.3f} "
          f"frac>5%={np.mean(rel > 0.05):.3f}")
    print("OK")


if __name__ == "__main__":
    main()
