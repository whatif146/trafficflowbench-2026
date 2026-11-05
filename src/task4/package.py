"""Copy the final probe candidates to work/task4/out with the Lead's naming and
validate key coverage / finiteness / non-negativity."""
import os
import shutil
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import odme

OUT = os.path.join(odme.WORK, "out")
CAND = os.path.join(odme.WORK, "candidates")
KEY = os.path.join(odme.REL, "submission_key.csv")

MAPPING = {
    "A": "ent_own",              # max-entropy projection, own prior (same scale as baseline)
    "B": "ent_pool",             # max-entropy projection, pooled prior (denoised)
    "C": "autoscale",            # prior rescaled to the cut level, then NNLS (scale axis)
    "D": "pool",                 # pooled-prior NNLS (Lead's requested A)
    "E": "denoise_margcorr",     # gravity-denoised prior + marginal correction (aggressive)
    "F": "margcorr",             # marginal correction only
}

DESC = {
    "A": "max-entropy/min-KL projection of the own-split prior onto the cut counts "
         "(42-100 segment potentials). Same total as baseline, different null-space allocation.",
    "B": "same entropy projection but prior = mean of the train/validation/private priors.",
    "C": "prior multiplied by the scalar that matches the cut-profile level, then NNLS like baseline.",
    "D": "baseline NNLS with the 3-split mean prior.",
    "E": "gravity-smoothed own prior (log-linear origin+destination+separation fit) then "
         "multiplicative marginal correction to the cuts.",
    "F": "own prior with per-zone multiplicative marginals corrected to the cuts.",
}


def main():
    key = pd.read_csv(KEY, dtype=str)
    kr = key[key.task == "odme"][["panel", "departure_time", "path_id"]]
    print("key odme rows:", len(kr))
    for letter, tag in MAPPING.items():
        src = os.path.join(CAND, f"path_flow_{tag}_validation_plus_private.csv")
        sub = pd.read_csv(src, dtype={"panel": str, "path_id": str, "departure_time": str})
        f = sub.path_flow.to_numpy(float)
        merged = sub.merge(kr, on=["panel", "departure_time", "path_id"], how="outer",
                           indicator=True)
        miss = int((merged._merge == "right_only").sum())
        extra = int((merged._merge == "left_only").sum())
        dups = int(sub.duplicated(["panel", "departure_time", "path_id"]).sum())
        ok = np.isfinite(f).all() and (f >= 0).all()
        dst = os.path.join(OUT, f"cand_{letter}_validation_plus_private.csv")
        shutil.copyfile(src, dst)
        for split in ("validation", "private"):
            shutil.copyfile(os.path.join(CAND, f"path_flow_{tag}_{split}.csv"),
                            os.path.join(OUT, f"cand_{letter}_{split}.csv"))
        print(f"cand_{letter} <= {tag}: rows={len(sub)} missing={miss} extra={extra} "
              f"dups={dups} finite_nonneg={ok}  -> {dst}")
    print("\nLegend:")
    for letter, tag in MAPPING.items():
        print(f"  cand_{letter} = {tag}: {DESC[letter]}")


if __name__ == "__main__":
    main()
