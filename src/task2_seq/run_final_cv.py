"""Final stride-1 CV: v3 vs v5 vs Jan-chosen v3+v5 blend, protocols A and B.

Memory-lean version (preallocated feature matrix, one model at a time).
Protocol B: fit on selector-valid pseudo windows with origin before 2031-01-01,
threshold (and blend weight) chosen on the January block, scored once on
February.  Protocol A: same fit block with every June 2030 origin removed, then
scored on the 80 released train windows (June 2030, horizon truth from the train
plain layer), using the parameters chosen on January only.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC
import run_exp as EX

import lightgbm as lgb

V3N = list(LH.v3_features())
ALLN = V3N + list(LH.LH_FEATURES)
NV3 = len(V3N)
FEATS = {"v3": V3N, "v5": ALLN}
COLS = {k: EX.cols_for(v) for k, v in FEATS.items()}


def gather_lean(cond: str, feat: str, exclude_june: bool):
    """Preallocated fitting matrix (avoids the list+concat double copy)."""
    cols = COLS[feat]
    plan, total = [], 0
    for p in LH.PANELS:
        z = RC.load_panel(p)
        pre = "on" if cond == "on" else "ong"
        L = int(z["n_links"])
        per = RC.per_window_rows(cond, L)
        t = z[f"{pre}_time"]
        keep = t < RC.TUNE_EPOCH
        if exclude_june:
            keep &= ~((t >= RC.JUNE0) & (t < RC.JUNE1))
        cnt = int(keep.sum()) * per
        plan.append((p, keep, per, cnt, total))
        total += cnt
        z.close()
    X = np.empty((total, len(cols)), np.float32)
    y = np.empty(total, np.int8)
    for p, keep, per, cnt, start in plan:
        z = RC.load_panel(p)
        pre = "on" if cond == "on" else "ong"
        Xfull = EX.X_sel(cond, z, cols)
        rows = np.repeat(keep, per)
        X[start:start + cnt] = Xfull[rows]
        y[start:start + cnt] = z[f"{pre}_y"][rows]
        del Xfull, rows
        z.close()
        gc.collect()
    return X, y, total


def train_model(cond: str, feat: str, X, y, rounds: int):
    ds = lgb.Dataset(X, label=y, feature_name=list(FEATS[feat]),
                     free_raw_data=True, params=RC.PARAMS)
    ds.construct()
    del X, y
    gc.collect()
    m = lgb.train(RC.PARAMS, ds, num_boost_round=rounds)
    del ds
    gc.collect()
    return m


def summ(detail: pd.DataFrame) -> dict:
    agg = LH.aggregate(detail)
    return {"overall": agg["overall"], "onset": agg["onset"], "ongoing": agg["ongoing"],
            "per_panel": agg["per_panel"], "per_family": agg["per_family"]}


def main():
    stride = 1
    out = {}
    for cond in ("on", "ong"):
        rounds = RC.ON_ROUNDS if cond == "on" else RC.ONG_ROUNDS
        res = {}
        store = {}
        # ---------------- protocol B (fit on pre-January) ----------------
        for feat in ("v3", "v5"):
            t0 = time.time()
            X, y, nrow = gather_lean(cond, feat, exclude_june=False)
            m = train_model(cond, feat, X, y, rounds)
            pj = EX.block_probs(cond, COLS[feat], m, "tune")
            th, jan, means = EX.tune_feb(cond, pj)
            pf = EX.block_probs(cond, COLS[feat], m, "feb")
            feb = EX.score_feb(cond, pf, th)
            store[feat] = {"pj": pj, "pf": pf}
            res[f"{feat}_B"] = {"th": th, "jan": jan, **summ(feb)}
            feb.to_csv(os.path.join(LH.OUT, f"final_{cond}_{feat}_feb.csv"), index=False)
            print(f"[{cond}/B] {feat}: {nrow:,} rows x {len(FEATS[feat])} feats | th {th} "
                  f"jan {jan:.4f} -> FEB {res[f'{feat}_B']['overall']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del m, X, y
            gc.collect()
        w, th_bl, jan_bl, _ = EX.tune_blend(cond, store["v3"]["pj"], store["v5"]["pj"])
        pf_bl = EX.blend_probs(store["v3"]["pf"], store["v5"]["pf"], w)
        feb_bl = EX.score_feb(cond, pf_bl, th_bl)
        res["blend_B"] = {"w": w, "th": th_bl, "jan": jan_bl, **summ(feb_bl)}
        feb_bl.to_csv(os.path.join(LH.OUT, f"final_{cond}_blend_feb.csv"), index=False)
        print(f"[{cond}/B] blend: w {w} th {th_bl} jan {jan_bl:.4f} -> "
              f"FEB {res['blend_B']['overall']:.4f}", flush=True)
        del store
        gc.collect()
        # ---------------- protocol A (June origins excluded) ----------------
        jstore = {}
        for feat in ("v3", "v5"):
            t0 = time.time()
            X, y, nrow = gather_lean(cond, feat, exclude_june=True)
            m = train_model(cond, feat, X, y, rounds)
            jp = EX.june_probs(cond, COLS[feat], m)
            jstore[feat] = jp
            june = EX.score_june(cond, jp, res[f"{feat}_B"]["th"])
            res[f"{feat}_A"] = {"th": res[f"{feat}_B"]["th"], **summ(june)}
            june.to_csv(os.path.join(LH.OUT, f"final_{cond}_{feat}_june.csv"), index=False)
            print(f"[{cond}/A] {feat}: {nrow:,} rows | th {res[f'{feat}_B']['th']} -> "
                  f"JUNE {res[f'{feat}_A']['overall']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            del m, X, y
            gc.collect()
        june_bl = EX.blend_june(jstore["v3"], jstore["v5"], w, th_bl, cond)
        res["blend_A"] = {"w": w, "th": th_bl, **summ(june_bl)}
        june_bl.to_csv(os.path.join(LH.OUT, f"final_{cond}_blend_june.csv"), index=False)
        print(f"[{cond}/A] blend: w {w} th {th_bl} -> JUNE "
              f"{res['blend_A']['overall']:.4f}", flush=True)
        del jstore
        gc.collect()
        out[cond] = res
    with open(os.path.join(LH.OUT, "final_cv_stride1.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n=== FINAL stride-1 summary ===")
    for cond in ("on", "ong"):
        for k, v in out[cond].items():
            print(f"{cond:>3} {k:>8}: overall {v['overall']:.4f} | onset {v['onset']:.4f} | "
                  f"ongoing {v['ongoing']:.4f}"
                  + (f" | w {v.get('w')} th {v.get('th')} jan {v.get('jan'):.4f}" if "w" in v else
                     f" | th {v.get('th')} jan {v.get('jan'):.4f}"))


if __name__ == "__main__":
    main()
