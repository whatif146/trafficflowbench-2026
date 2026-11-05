"""Combine the lambda-sweep split files into the Lead's cand_c combined file."""
import os
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import odme

OUT = os.path.join(odme.WORK, "out")
KEY = os.path.join(odme.REL, "submission_key.csv")


def combine(letter, val_path, priv_path):
    v = pd.read_csv(val_path, dtype={"panel": str, "path_id": str, "departure_time": str})
    p = pd.read_csv(priv_path, dtype={"panel": str, "path_id": str, "departure_time": str})
    sub = pd.concat([v, p], ignore_index=True)
    key = pd.read_csv(KEY, dtype=str)
    kr = key[key.task == "odme"][["panel", "departure_time", "path_id"]]
    m = sub.merge(kr, on=["panel", "departure_time", "path_id"], how="outer", indicator=True)
    f = sub.path_flow.to_numpy(float)
    dst = os.path.join(OUT, f"cand_{letter}_validation_plus_private.csv")
    sub.to_csv(dst, index=False)
    print(f"cand_{letter}: rows={len(sub)} missing={int((m._merge=='right_only').sum())} "
          f"extra={int((m._merge=='left_only').sum())} finite={np.isfinite(f).all()} "
          f"min={f.min():.3f} dups={int(sub.duplicated(['panel','departure_time','path_id']).sum())}"
          f" -> {dst}")
    return sub


if __name__ == "__main__":
    combine("c", os.path.join(OUT, "lam100_validation.csv"), os.path.join(OUT, "lam100_private.csv"))
    combine("c_lam10", os.path.join(OUT, "lam10_validation.csv"), os.path.join(OUT, "lam10_private.csv"))
