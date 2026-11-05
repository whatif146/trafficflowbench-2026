"""Clean simulation summary tables for the report + create the final recommended file."""
import os
import shutil
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import odme

REP = odme.REPORTS
OUT = os.path.join(odme.WORK, "out")

SCEN = {
    "own_s0.3": "A: truth = own prior x lognormal(0.3), projected onto cuts",
    "raw_s0.0": "B: truth = pooled prior (circular for ent_pool)",
    "raw_s0.4": "C: truth = pooled prior x lognormal(0.4), projected",
    "smooth_s0.0": "D: truth = gravity-smoothed pooled prior, projected",
    "smooth_s0.4": "E: truth = gravity-smoothed pooled prior x lognormal(0.4)",
}
LABEL = {
    "baseline": "baseline NNLS lambda=0.05",
    "pool": "cand_a  pooled prior NNLS",
    "ent_own": "cand_b  entropy, own prior",
    "ent_pool": "cand_b2 entropy, pooled prior",
    "margcorr": "cand_x_margcorr",
    "denoise_margcorr": "cand_x_denoise_margcorr",
    "autoscale": "cand_x_autoscale",
}


def main():
    d = pd.read_csv(os.path.join(REP, "simulation_study.csv"))
    for metric in ("S_ODME", "S_od", "S_dev", "S_attr"):
        piv = d.pivot_table(index="estimator", columns="dgp", values=metric)
        piv = piv.reindex([k for k in LABEL if k in piv.index])
        piv.index = [LABEL[k] for k in piv.index]
        piv.columns = [SCEN.get(c, c) for c in piv.columns]
        piv.to_csv(os.path.join(REP, f"sim_summary_{metric}.csv"))
        if metric == "S_ODME":
            print("=== mean S_ODME by estimator (4 panels) ===")
            print(piv.round(3).to_string())
        if metric == "S_od":
            print("\n=== mean S_od by estimator (4 panels) ===")
            print(piv.round(3).to_string())
    print("\n=== mean truth total / baseline total by scenario ===")
    d["ratio"] = d.truth_sum / d.est_sum
    print(d.groupby("dgp")[["truth_sum"]].mean().round(0).to_string())
    # final recommended file: entropy projection of the own prior (cand_b)
    src = os.path.join(OUT, "cand_b_validation_plus_private.csv")
    dst = os.path.join(OUT, "final_recommended_validation_plus_private.csv")
    shutil.copyfile(src, dst)
    d0 = pd.read_csv(dst)
    print(f"\nfinal_recommended_validation_plus_private.csv <= cand_b, rows={len(d0)}, "
          f"sum={d0.path_flow.sum():.1f}")
    for split in ("validation", "private"):
        shutil.copyfile(os.path.join(OUT, f"cand_b_{split}.csv"),
                        os.path.join(OUT, f"final_recommended_{split}.csv"))


if __name__ == "__main__":
    main()
