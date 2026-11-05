"""Refit the selected LightGBM configuration on every train window.

Reads out/gbm_meta.json (config, rounds, feature list) and the per-panel
datasets, trains one onset and one ongoing model on all windows, and saves
models/gbm_onset.txt / models/gbm_ongoing.txt.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C
import features as FT
import train_gbm as TG

import lightgbm as lgb


def main():
    with open(os.path.join(C.OUT2, "gbm_meta.json")) as fh:
        meta = json.load(fh)
    params = meta["params"]
    rounds = {"on": int(meta["on_rounds"]), "ong": int(meta["ong_rounds"])}
    on, ong = TG.load_dataset()
    for key, items, fname in (("on", on, "gbm_onset"), ("ong", ong, "gbm_ongoing")):
        Xs, ys = [], []
        for it in items:
            Xs.append(it["X"])
            ys.append(it["y"])
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        del Xs, ys
        ds = lgb.Dataset(X, label=y, feature_name=list(FT.FEATURES),
                         free_raw_data=True)
        ds.construct()
        del X, y
        gc.collect()
        print(f"training {fname} on {ds.num_data()} rows, {rounds[key]} rounds")
        model = lgb.train(params, ds, num_boost_round=rounds[key])
        model.save_model(os.path.join(C.MODEL2, f"{fname}.txt"))
        print(f"saved {fname}.txt ({model.num_trees()} trees)")
        del ds, model
        gc.collect()


if __name__ == "__main__":
    main()
