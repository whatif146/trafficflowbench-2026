"""Validate the packaged probe candidates against submission_key.csv."""
import os
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import odme

OUT = os.path.join(odme.WORK, "out")
CAND = os.path.join(odme.WORK, "candidates")
KEY = os.path.join(odme.REL, "submission_key.csv")

COPIES = {
    "cand_a": "path_flow_pool_validation_plus_private.csv",
    "cand_b": "path_flow_ent_own_validation_plus_private.csv",
    "cand_b2": "path_flow_ent_pool_validation_plus_private.csv",
    "cand_x_autoscale": "path_flow_autoscale_validation_plus_private.csv",
    "cand_x_denoise_margcorr": "path_flow_denoise_margcorr_validation_plus_private.csv",
    "cand_x_margcorr": "path_flow_margcorr_validation_plus_private.csv",
}


def main():
    key = pd.read_csv(KEY, dtype=str)
    kr = key[key.task == "odme"][["panel", "departure_time", "path_id"]]
    for name, src in COPIES.items():
        d0 = pd.read_csv(os.path.join(CAND, src), dtype={"panel": str, "path_id": str,
                                                         "departure_time": str})
        d0.to_csv(os.path.join(OUT, name + "_validation_plus_private.csv"), index=False)
        d = pd.read_csv(os.path.join(OUT, name + "_validation_plus_private.csv"),
                        dtype={"panel": str, "path_id": str, "departure_time": str})
        m = d.merge(kr, on=["panel", "departure_time", "path_id"], how="outer",
                    indicator=True)
        f = d.path_flow.to_numpy(float)
        print(f"{name}: rows={len(d)} missing={int((m._merge=='right_only').sum())} "
              f"extra={int((m._merge=='left_only').sum())} finite={np.isfinite(f).all()} "
              f"min={f.min():.3f} dups="
              f"{int(d.duplicated(['panel','departure_time','path_id']).sum())}")


if __name__ == "__main__":
    main()
