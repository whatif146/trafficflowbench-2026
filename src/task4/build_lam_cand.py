"""Build cand_c (lambda=100) and cand_c_lam10 combined + per-split files from the
ridge sweep outputs, validating key coverage and template shape."""
import os
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import odme

OUT = os.path.join(odme.WORK, "out")
CAND = os.path.join(odme.WORK, "candidates")
KEY = os.path.join(odme.REL, "submission_key.csv")

PAIRS = {
    "cand_c": "ridge100",
    "cand_c_lam10": "ridge10",
}


def build(letter, tag):
    key = pd.read_csv(KEY, dtype=str)
    kr = key[key.task == "odme"][["panel", "departure_time", "path_id"]]
    frames = []
    for split in ("validation", "private"):
        src = os.path.join(CAND, f"{tag}_{split}.csv")
        d = pd.read_csv(src, dtype={"panel": str, "path_id": str, "departure_time": str})
        parts = []
        for p in odme.PANELS:
            tpl = odme.load_template(p, split)
            parts.append(tpl[["panel", "departure_time", "path_id", "origin_zone",
                              "destination_zone"]])
        merged = pd.concat(parts, ignore_index=True).merge(
            d[["panel", "departure_time", "path_id", "path_flow"]],
            on=["panel", "departure_time", "path_id"], how="left")
        assert merged.path_flow.notna().all(), (letter, split)
        merged.to_csv(os.path.join(OUT, f"{letter}_{split}.csv"), index=False)
        frames.append(merged)
    comb = pd.concat(frames, ignore_index=True)
    m = comb.merge(kr, on=["panel", "departure_time", "path_id"], how="outer", indicator=True)
    f = comb.path_flow.to_numpy(float)
    comb.to_csv(os.path.join(OUT, f"{letter}_validation_plus_private.csv"), index=False)
    print(f"{letter}: rows={len(comb)} missing={int((m._merge=='right_only').sum())} "
          f"extra={int((m._merge=='left_only').sum())} finite={np.isfinite(f).all()} "
          f"min={f.min():.3f} dups="
          f"{int(comb.duplicated(['panel','departure_time','path_id']).sum())}")


if __name__ == "__main__":
    for letter, tag in PAIRS.items():
        build(letter, tag)
