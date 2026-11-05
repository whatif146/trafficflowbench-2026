"""Machine-readable template verification of a state CSV (for the deploy meta)."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\lenovo\Desktop\kaggle"
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
import traffic  # noqa: E402

HERE = os.path.join(ROOT, "work", "task1", "unet")
GBM_DIR = os.path.join(ROOT, "work", "task1", "gbm")
KEYS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
COLS = KEYS + ["speed_kmh", "flow_vph"]


def check(path: str, split: str) -> dict:
    d = pd.read_csv(os.path.join(HERE, path), dtype={"timestamp": str, "station_id": str,
                                                     "link_id": str, "panel": str,
                                                     "mask_regime": str})
    g = pd.read_csv(os.path.join(GBM_DIR, f"state_gbm_{split}.csv"),
                    dtype={"timestamp": str, "station_id": str, "link_id": str,
                           "panel": str, "mask_regime": str})
    out = {"file": path, "split": split, "rows": int(len(d)),
           "columns_ok": list(d.columns) == COLS,
           "row_count_matches_shipped_gbm": int(len(d)) == int(len(g)),
           "keys_match_shipped_gbm": all(
               (d[k].to_numpy() == g[k].to_numpy()).all() for k in KEYS),
           "no_nan": bool(d["speed_kmh"].notna().all() and d["flow_vph"].notna().all()),
           "panels": {}}
    for panel in d["panel"].unique():
        p = d[d["panel"] == panel]
        tpl = pd.read_csv(os.path.join(traffic.REL, "task1", panel, split,
                                       "sample_submission_state.csv"),
                          dtype={"timestamp": str, "station_id": str, "link_id": str,
                                 "panel": str, "mask_regime": str})
        ok = (len(p) == len(tpl)
              and (p["timestamp"].to_numpy() == tpl["timestamp"].to_numpy()).all()
              and (p["station_id"].to_numpy() == tpl["station_id"].to_numpy()).all()
              and (p["link_id"].to_numpy() == tpl["link_id"].to_numpy()).all()
              and (p["mask_regime"].to_numpy() == tpl["mask_regime"].to_numpy()).all())
        out["panels"][panel] = {"rows": int(len(p)), "template_exact": bool(ok)}
    out["all_panels_template_exact"] = all(v["template_exact"] for v in out["panels"].values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--json", default="verify_unet3.json")
    args = ap.parse_args()
    res = {}
    for f, s in zip(args.files, args.splits):
        r = check(f, s)
        res[f"{s}"] = r
        print(f"{f}: rows={r['rows']} cols_ok={r['columns_ok']} "
              f"keys_ok={r['keys_match_shipped_gbm']} no_nan={r['no_nan']} "
              f"template_exact={r['all_panels_template_exact']}")
        for p, v in r["panels"].items():
            print(f"   {p:12s} rows={v['rows']:8d} template_exact={v['template_exact']}")
    json.dump(res, open(os.path.join(HERE, args.json), "w", encoding="utf-8"), indent=2)
    print("wrote", args.json)


if __name__ == "__main__":
    main()
