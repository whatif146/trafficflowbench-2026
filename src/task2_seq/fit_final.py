"""Refit the long-history model on all train windows (deployment weights).

Memory-lean: one panel at a time, chained with ``init_model`` exactly like the
v3 refit (work/task2/tune_final.py), so the peak RSS stays well under 1 GB even
when eight panels are used.  All candidate windows of the fit block (before
2031-02-01) are used; February windows were never part of the v3 dataset, and
the deployment model may legally use every released train month anyway.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC

import lightgbm as lgb


def fit_condition(cond: str, feat: str, rounds: int) -> lgb.Booster:
    pre = "on" if cond == "on" else "ong"
    per_panel = max(20, int(round(rounds / len(LH.PANELS))))
    model = None
    for p in LH.PANELS:
        z = RC.load_panel(p)
        L = int(z["n_links"])
        per = RC.per_window_rows(cond, L)
        t = z[f"{pre}_time"]
        keep = np.ones(len(t), bool)       # deployment refit: every released train window
        rows = np.repeat(keep, per)
        X = RC.panel_X(cond, feat, z)
        Xk = np.ascontiguousarray(X[rows])
        yk = z[f"{pre}_y"][rows]
        del X
        z.close()
        ds = lgb.Dataset(Xk, label=yk, feature_name=RC.feature_names(feat),
                         free_raw_data=False, params=RC.PARAMS)
        ds.construct()
        del Xk, yk
        gc.collect()
        model = lgb.train(RC.PARAMS, ds, num_boost_round=per_panel, init_model=model)
        del ds
        gc.collect()
        print(f"  [{cond}] {p}: +{per_panel} trees (total {model.num_trees()})", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="v5")
    ap.add_argument("--on-rounds", type=int, default=RC.ON_ROUNDS)
    ap.add_argument("--ong-rounds", type=int, default=RC.ONG_ROUNDS)
    ap.add_argument("--meta", default=None,
                    help="json with the selected thresholds (default: cv_v5_B_s2.json)")
    args = ap.parse_args()
    meta_path = args.meta or os.path.join(LH.OUT, "cv_v5_B_s2.json")
    meta = json.load(open(meta_path))
    thr = meta["thresholds"]
    on_rule = thr["on"]
    ong_th = float(thr["ong"])
    print(f"thresholds from {os.path.basename(meta_path)}: onset {on_rule} ongoing {ong_th}")
    m_on = fit_condition("on", args.feat, args.on_rounds)
    m_on.save_model(os.path.join(LH.MODELS, f"{args.feat}_onset.txt"))
    del m_on
    gc.collect()
    m_ong = fit_condition("ong", args.feat, args.ong_rounds)
    m_ong.save_model(os.path.join(LH.MODELS, f"{args.feat}_ongoing.txt"))
    meta_out = {"feat": args.feat, "onset_rule": on_rule, "ongoing_th": ong_th,
                "params": RC.PARAMS, "on_rounds": args.on_rounds,
                "ong_rounds": args.ong_rounds,
                "features": RC.feature_names(args.feat),
                "cv_source": os.path.basename(meta_path),
                "cv_overall": meta.get("overall"), "cv_onset": meta.get("onset"),
                "cv_ongoing": meta.get("ongoing")}
    with open(os.path.join(LH.MODELS, f"{args.feat}_meta.json"), "w") as fh:
        json.dump(meta_out, fh, indent=1)
    print("saved models:", os.path.join(LH.MODELS, f"{args.feat}_onset.txt"))


if __name__ == "__main__":
    main()
