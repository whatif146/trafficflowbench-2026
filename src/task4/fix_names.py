"""Create lowercase per-split probe files and remove the ambiguous uppercase ones."""
import os
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import odme

OUT = os.path.join(odme.WORK, "out")
CAND = os.path.join(odme.WORK, "candidates")

# letter -> method tag
MAP = {
    "a": "pool",
    "b": "ent_own",
    "b2": "ent_pool",
    "c": "lam100",          # created from out/lam100_<split>.csv
    "c_lam10": "lam10",
    "x_autoscale": "autoscale",
    "x_margcorr": "margcorr",
    "x_denoise_margcorr": "denoise_margcorr",
}

for letter, tag in MAP.items():
    for split in ("validation", "private"):
        src = os.path.join(CAND, f"path_flow_{tag}_{split}.csv")
        if not os.path.exists(src):
            src = os.path.join(OUT, f"{tag}_{split}.csv")
        if not os.path.exists(src):
            print("SKIP", letter, split, "no source", src)
            continue
        dst = os.path.join(OUT, f"cand_{letter}_{split}.csv")
        d = pd.read_csv(src, dtype={"panel": str, "path_id": str})
        tpl_parts = []
        for p in odme.PANELS:
            tpl = odme.load_template(p, split)
            tpl_parts.append(tpl[["panel", "departure_time", "path_id", "origin_zone",
                                  "destination_zone"]])
        tpl = pd.concat(tpl_parts, ignore_index=True)
        merged = tpl.merge(d[["panel", "departure_time", "path_id", "path_flow"]],
                           on=["panel", "departure_time", "path_id"], how="left")
        assert merged.path_flow.notna().all(), (letter, split)
        assert len(merged) == len(tpl)
        merged.to_csv(dst, index=False)
        print(f"cand_{letter}_{split}.csv rows={len(merged)} "
              f"sum={merged.path_flow.sum():.1f}")

# remove uppercase legacy files to avoid Windows case-insensitive collisions
for name in os.listdir(OUT):
    if name.startswith("cand_") and len(name) > 6 and name[5].isupper():
        os.remove(os.path.join(OUT, name))
        print("removed", name)
# remove intermediate lambda split files whose names collide with lowercase cand_c
for name in ("lam100_validation.csv", "lam100_private.csv", "lam10_validation.csv",
             "lam10_private.csv"):
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        os.remove(p)
        print("removed", name)
