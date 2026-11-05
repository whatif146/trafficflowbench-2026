"""Joint (non-incremental) refit of the v5 deployment models on all train windows.

The per-panel incremental chaining in fit_final.py under-produced trees on some
panels (LightGBM continued-training quirk), so the deployment models are trained
jointly here with the exact CV configuration (450/700 rounds).  Memory: one
condition at a time, preallocated float32 matrix (~0.75 GB peak for ongoing).
"""
from __future__ import annotations

import gc
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC
import run_final_cv as FC

import lightgbm as lgb


def gather_all(cond: str, feat: str):
    cols = FC.COLS[feat]
    plan, total = [], 0
    for p in LH.PANELS:
        z = RC.load_panel(p)
        L = int(z["n_links"])
        per = RC.per_window_rows(cond, L)
        cnt = len(z[f"{'on' if cond == 'on' else 'ong'}_time"]) * per
        plan.append((p, per, cnt, total))
        total += cnt
        z.close()
    X = np.empty((total, len(cols)), np.float32)
    y = np.empty(total, np.int8)
    for p, per, cnt, start in plan:
        z = RC.load_panel(p)
        pre = "on" if cond == "on" else "ong"
        Xfull = FC.EX.X_sel(cond, z, cols)
        X[start:start + cnt] = Xfull
        y[start:start + cnt] = z[f"{pre}_y"]
        del Xfull
        z.close()
        gc.collect()
    return X, y, total


def main():
    for cond, rounds in (("on", RC.ON_ROUNDS), ("ong", RC.ONG_ROUNDS)):
        X, y, n = gather_all(cond, "v5")
        print(f"[{cond}] joint matrix {n:,} x {X.shape[1]}", flush=True)
        ds = lgb.Dataset(X, label=y, feature_name=RC.feature_names("v5"),
                         free_raw_data=True, params=RC.PARAMS)
        ds.construct()
        del X, y
        gc.collect()
        m = lgb.train(RC.PARAMS, ds, num_boost_round=rounds)
        del ds
        gc.collect()
        print(f"[{cond}] trees {m.num_trees()}", flush=True)
        m.save_model(os.path.join(LH.MODELS, f"v5_{'onset' if cond == 'on' else 'ongoing'}.txt"))
        del m
        gc.collect()


if __name__ == "__main__":
    main()
