"""Ramp-jam-sensor deployment path: v5r (v3 + 80 long-history + 8 ramp ratios)
ongoing model, blended with the v3 boosters; onset half unchanged from the
shipped v5 blend.  Run only if the honest ramp CV is positive on both protocols.

    python fit_ramp.py        # joint refit of the v5r ongoing model, 700 trees
    python ship_ramp.py       # write out/queue_v5ramp_{validation,private}.csv
"""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC
import ramp_cv as RV

import lightgbm as lgb


def main():
    cond = "ong"
    names = RV.V5RN
    Xs = []
    for p in LH.PANELS:
        z = RC.load_panel(p)
        L = int(z["n_links"])
        X = RV.xsel(cond, z, names)
        Xs.append(X)
        z.close()
        gc.collect()
    X = np.concatenate(Xs, axis=0)
    del Xs
    gc.collect()
    ys = []
    for p in LH.PANELS:
        z = RC.load_panel(p)
        ys.append(z["ong_y"])
        z.close()
    y = np.concatenate(ys)
    del ys
    gc.collect()
    print(f"[v5r ongoing] {X.shape[0]:,} rows x {X.shape[1]}", flush=True)
    ds = lgb.Dataset(X, label=y, feature_name=list(names), free_raw_data=True,
                     params=RC.PARAMS)
    ds.construct()
    del X, y
    gc.collect()
    m = lgb.train(RC.PARAMS, ds, num_boost_round=RC.ONG_ROUNDS)
    del ds
    gc.collect()
    print("trees", m.num_trees())
    m.save_model(os.path.join(LH.MODELS, "v5r_ongoing.txt"))
    with open(os.path.join(LH.MODELS, "v5r_meta.json"), "w") as fh:
        json.dump({"features": list(names), "trees": int(m.num_trees()),
                   "kind": "ongoing ramp jam-sensor model"}, fh, indent=1)


if __name__ == "__main__":
    main()
