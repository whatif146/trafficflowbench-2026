"""v7 deployment: v3+v5 models refit on ALL train windows with DENOISED labels,
then the LB-verified fixed decision rules produce the two submission CSVs.

  python ship_v7.py     # -> out/queue_v7_{validation,private}.csv + footprint report

Rules: onset  p = 0.3*p_v3 + 0.7*p_v5, predict p >= 0.30 on the last step with
top-1 fallback when empty; ongoing p = 0.4*p_v3 + 0.6*p_v5, predict p >= 0.20 on
all six steps.
"""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC
import run_final_cv as FC
import run_v7_cv as V7
import run_exp as EX

import lightgbm as lgb

W_ON, W_ONG = V7.W_ON, V7.W_ONG
ON_TH, ONG_TH = V7.ON_TH, V7.ONG_TH


def gather_all(cond: str, feat: str):
    pre = V7.PRE[cond]
    cols = V7.FEATS[feat]
    plan, total = [], 0
    for p in LH.PANELS:
        z = RC.load_panel(p)
        L = int(z["n_links"])
        per = RC.per_window_rows(cond, L)
        cnt = len(z[f"{pre}_time"]) * per
        plan.append((p, per, cnt, total))
        total += cnt
        z.close()
    X = np.empty((total, len(cols)), np.float32)
    y = np.empty(total, np.int8)
    for p, per, cnt, start in plan:
        z = RC.load_panel(p)
        Xf = EX.X_sel(cond, z, FC.COLS[feat])
        X[start:start + cnt] = Xf
        v7 = V7.load_v7(p)
        yk = v7["on_y_dn"] if cond == "on" else v7["ong_y_dn"]
        y[start:start + cnt] = yk
        v7.close()
        del Xf
        z.close()
        gc.collect()
    return X, y, total


def train_and_save(cond: str, feat: str, rounds: int):
    X, y, n = gather_all(cond, feat)
    print(f"[v7 {cond}/{feat}] {n:,} rows x {X.shape[1]}", flush=True)
    ds = lgb.Dataset(X, label=y, feature_name=list(V7.FEATS[feat]),
                     free_raw_data=True, params=RC.PARAMS)
    ds.construct()
    del X, y
    gc.collect()
    m = lgb.train(RC.PARAMS, ds, num_boost_round=rounds)
    del ds
    gc.collect()
    tag = "onset" if cond == "on" else "ongoing"
    m.save_model(os.path.join(LH.MODELS, f"v7_{tag}_{feat}.txt"))
    print(f"  saved v7_{tag}_{feat}.txt ({m.num_trees()} trees)", flush=True)
    return m


def main():
    for cond, rounds in (("on", RC.ON_ROUNDS), ("ong", RC.ONG_ROUNDS)):
        ms = {}
        for feat in ("v3", "v5"):
            ms[feat] = train_and_save(cond, feat, rounds)
        del ms
        gc.collect()
    models = {}
    for cond in ("on", "ong"):
        tag = "onset" if cond == "on" else "ongoing"
        models[cond] = {f: lgb.Booster(model_file=os.path.join(LH.MODELS, f"v7_{tag}_{f}.txt"))
                        for f in ("v3", "v5")}
    import traffic
    tpl_all = traffic.load_queue_template()
    tpl_all["window_id"] = tpl_all.window_id.astype(str)
    tpl_all["link_id"] = tpl_all.link_id.astype(str)
    tpl_all = tpl_all.drop(columns=["queue_pred"], errors="ignore")
    for split in ("validation", "private"):
        frames = []
        for p in LH.PANELS:
            st = LH.static_panel(p)
            L = st["n_links"]
            z = np.load(os.path.join(LH.CACHE, f"rel_{p}.npz"), allow_pickle=False)
            sel = np.where(z["split"] == split)[0]
            if len(sel) == 0:
                z.close()
                continue
            X3 = z["v3"]
            X5 = RC.rel_X("on", "v5", z)
            P = LH.HORIZON * L
            cond = z["condition"]
            for j, wj in enumerate(sel):
                rr = np.arange(wj * P, (wj + 1) * P)
                wid = str(z["window_ids"][wj])
                t0 = int(z["time"][wj])
                ts = [t0 + 300 * (k + 1) for k in range(LH.HORIZON)]
                if cond[wj] == "queue_onset":
                    a = models["on"]["v3"].predict(np.ascontiguousarray(X3[rr]))
                    b = models["on"]["v5"].predict(np.ascontiguousarray(X5[rr]))
                    prob = (W_ON * a + (1 - W_ON) * b).reshape(LH.HORIZON, L)
                    pred = V7.onset_pred_blend(prob[LH.HORIZON - 1], L)
                else:
                    a = models["ong"]["v3"].predict(np.ascontiguousarray(X3[rr]))
                    b = models["ong"]["v5"].predict(np.ascontiguousarray(X5[rr]))
                    prob = (W_ONG * a + (1 - W_ONG) * b).reshape(LH.HORIZON, L)
                    pred = prob >= ONG_TH
                frames.append(pd.DataFrame({
                    "window_id": np.full(P, wid),
                    "timestamp": np.repeat(traffic.epoch_to_iso(np.asarray(ts, np.int64)), L),
                    "link_id": np.tile(st["links"], LH.HORIZON),
                    "queue_pred": pred.reshape(-1).astype(np.int8),
                }))
            del X3, X5
            z.close()
            gc.collect()
        pred = pd.concat(frames, ignore_index=True)
        tpl = tpl_all[tpl_all.window_id.isin(set(pred.window_id))].copy()
        out = tpl.merge(pred, on=["window_id", "timestamp", "link_id"], how="left",
                        validate="one_to_one", indicator=True)
        if int((out._merge != "both").sum()):
            raise RuntimeError("template rows without a prediction")
        out = out.drop(columns=["_merge"])
        out["queue_pred"] = out.queue_pred.astype(np.int8)
        out = out[["window_id", "timestamp", "link_id", "queue_pred"]]
        path = os.path.join(LH.OUT, f"queue_v7_{split}.csv")
        out.to_csv(path, index=False)
        print(f"{split}: {len(out):,} rows -> {os.path.basename(path)}", flush=True)
        # footprint vs the current best queue file
        best_path = os.path.join(LH.OUT, f"queue_w30ong40fb1_{split}.csv")
        if os.path.exists(best_path):
            best = pd.read_csv(best_path, dtype={"window_id": str, "link_id": str})
            m = best.merge(out, on=["window_id", "timestamp", "link_id"], suffixes=("_best", "_v7"))
            print(f"  vs queue_w30ong40fb1: best positives {int(best.queue_pred.sum()):,} "
                  f"({best.queue_pred.mean():.3%}) | v7 {int(out.queue_pred.sum()):,} "
                  f"({out.queue_pred.mean():.3%}) | differing cells "
                  f"{int((m.queue_pred_best != m.queue_pred_v7).sum()):,}", flush=True)
        else:
            print("  (queue_w30ong40fb1 reference file not found)", flush=True)


if __name__ == "__main__":
    main()

