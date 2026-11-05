"""Combine the per-condition detail CSVs into the official nesting."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH

for proto, block in (("B", "feb"), ("A", "june")):
    print(f"== protocol {proto} ({block}) ==")
    for variant in ("v3", "v5", "blend"):
        a = pd.read_csv(os.path.join(LH.OUT, f"final_on_{variant}_{block}.csv"))
        b = pd.read_csv(os.path.join(LH.OUT, f"final_ong_{variant}_{block}.csv"))
        d = pd.concat([a[["panel", "condition", "iou"]], b[["panel", "condition", "iou"]]],
                      ignore_index=True)
        agg = LH.aggregate(d)
        print(f"{variant:>5}: overall {agg['overall']:.4f} | onset {agg['onset']:.4f} | "
              f"ongoing {agg['ongoing']:.4f} | panels "
              + " ".join(f"{k}={v:.3f}" for k, v in agg["per_panel"].items()))
        print(f"       families " + " ".join(f"{k}={v:.3f}" for k, v in agg["per_family"].items()))
