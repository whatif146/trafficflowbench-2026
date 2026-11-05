"""Determinism check: does predict.py with its shipped default arguments reproduce
the shipped state_gbm_<split>.csv for the panel that was regenerated after the
flush/rounding patch (D12_I405_S, produced by `predict.py --split validation
private --panels D12_I405_S --outdir fix` with all-default args)?

Streams the shipped CSVs (low memory) and compares the D12_I405_S rows against
the fix/ shard: same rows/order/keys, and values equal after rounding.
"""
import csv
import sys

import numpy as np
import pandas as pd

BASE = r"C:\Users\lenovo\Desktop\kaggle\work\task1\gbm"


def compare(split: str):
    fix = pd.read_csv(f"{BASE}\\fix\\state_gbm_{split}.csv", dtype={"panel": str})
    assert (fix["panel"] == "D12_I405_S").all(), "fix shard should be one panel"
    print(f"{split}: fix shard rows={len(fix)}")
    n = 0
    max_ds = 0.0
    max_dq = 0.0
    key_ok = True
    with open(f"{BASE}\\state_gbm_{split}.csv", newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        for row in r:
            if row[0] != "D12_I405_S":
                continue
            f = fix.iloc[n]
            if (f["timestamp"] != row[1] or f["station_id"] != row[2]
                    or f["link_id"] != row[3] or f["mask_regime"] != row[4]):
                key_ok = False
                if n < 5:
                    print("  KEY MISMATCH at", n, row[:5], f[["timestamp", "station_id", "link_id", "mask_regime"]].tolist())
            ds = abs(float(row[5]) - float(f["speed_kmh"]))
            dq = abs(float(row[6]) - float(f["flow_vph"]))
            max_ds = max(max_ds, ds)
            max_dq = max(max_dq, dq)
            n += 1
    print(f"  shipped rows for panel={n}  keys_match={key_ok}  "
          f"max|d_speed|={max_ds:.6g}  max|d_flow|={max_dq:.6g}")
    ok = key_ok and n == len(fix) and max_ds <= 5e-4 and max_dq <= 5e-3
    print("  REPRODUCIBLE (same model config, rounded output only)" if ok else "  NOT REPRODUCIBLE")
    return ok


if __name__ == "__main__":
    ok1 = compare("validation")
    ok2 = compare("private")
    print("ALL OK" if (ok1 and ok2) else "CHECK FAILED")
