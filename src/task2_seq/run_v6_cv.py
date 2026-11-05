"""v6 (long-horizon recurrence / incident / imbalance / shockwave) honest CV.

Stage 1: v6 single models on the exact v5 rows/config, protocols B and A.
Stage 2: v3+v6 probability blend vs the recorded v3+v5 blend baseline
        (from out/final_cv_stride1.json), same January-only selection.

Rows: stride 1, identical to run_final_cv.py.  v5 model features = 65 v3 + 80
long-history; v6 adds the 26 new features (171 total).
"""
from __future__ import annotations

import argparse
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
import v6core as V6

import lightgbm as lgb

V3N = list(LH.v3_features())
LHN = list(LH.LH_FEATURES)
V6N = list(V6.V6_FEATURES)
FEATS = {"v3": V3N, "v5": V3N + LHN, "v6": V3N + LHN + V6N}
TH_GRID = [round(float(t), 2) for t in np.arange(0.10, 0.81, 0.05)]
W_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def load_v6(panel: str):
    return np.load(os.path.join(LH.CACHE, f"v6_{panel}.npz"), allow_pickle=False)


def panel_X(cond: str, feat: str, z, z6) -> np.ndarray:
    if "v6" not in feat:
        return RC.panel_X(cond, feat, z)
    if cond == "on":
        return np.concatenate([z["on_v3"], z["on_lh"], z6["on_v6"]], axis=1)
    lh = z["ong_lh"]
    lh6 = np.broadcast_to(lh[:, None, :, :],
                          (lh.shape[0], LH.HORIZON, lh.shape[1], lh.shape[2])).reshape(-1, lh.shape[2])
    r6 = z6["ong_v6"]
    r6 = np.broadcast_to(r6[:, None, :, :],
                         (r6.shape[0], LH.HORIZON, r6.shape[1], r6.shape[2])).reshape(-1, r6.shape[2])
    return np.concatenate([z["ong_v3"], lh6, r6], axis=1)


def gather(cond: str, feat: str, exclude_june: bool):
    """Preallocated fitting matrix for one condition/feature set."""
    plan, total, ncol = [], 0, len(FEATS[feat])
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
    X = np.empty((total, ncol), np.float32)
    y = np.empty(total, np.int8)
    for p, keep, per, cnt, start in plan:
        z = RC.load_panel(p)
        z6 = load_v6(p)
        pre = "on" if cond == "on" else "ong"
        Xf = panel_X(cond, feat, z, z6)
        rows = np.repeat(keep, per)
        X[start:start + cnt] = Xf[rows]
        y[start:start + cnt] = z[f"{pre}_y"][rows]
        del Xf, rows
        z.close()
        z6.close()
        gc.collect()
    return X, y, total


def train(cond: str, feat: str, X, y, rounds: int):
    ds = lgb.Dataset(X, label=y, feature_name=list(FEATS[feat]),
                     free_raw_data=True, params=RC.PARAMS)
    ds.construct()
    del X, y
    gc.collect()
    m = lgb.train(RC.PARAMS, ds, num_boost_round=rounds)
    del ds
    gc.collect()
    return m


def probs_pseudo(cond: str, feat: str, model, where: str) -> dict:
    out = {}
    for p in LH.PANELS:
        z = RC.load_panel(p)
        z6 = load_v6(p)
        pre = "on" if cond == "on" else "ong"
        L = int(z["n_links"])
        t = z[f"{pre}_time"]
        acc = z[f"{pre}_acc"]
        per = RC.per_window_rows(cond, L)
        m = (acc & (t >= RC.TUNE_EPOCH) & (t < RC.CUT_EPOCH)) if where == "tune" else \
            (acc & (t >= RC.CUT_EPOCH) & (t < RC.MAR_EPOCH))
        idx = np.where(m)[0]
        if len(idx):
            X = panel_X(cond, feat, z, z6)
            rows = (idx[:, None] * per + np.arange(per)[None, :]).reshape(-1)
            out[p] = {"idx": idx, "time": t[idx], "L": L,
                      "prob": model.predict(np.ascontiguousarray(X[rows])).astype(np.float32).reshape(len(idx), per)}
            del X
        z.close()
        z6.close()
        gc.collect()
    return out


def probs_rel(cond: str, feat: str, model) -> dict:
    out = {}
    cond_name = "queue_onset" if cond == "on" else "queue_ongoing"
    for p in LH.PANELS:
        z = np.load(os.path.join(LH.CACHE, f"rel_{p}.npz"), allow_pickle=False)
        z6 = load_v6(p)
        assert len(z6["rel_ids"]) == len(z["window_ids"])
        sel = (z["split"] == "train") & (z["condition"] == cond_name)
        idx = np.where(sel)[0]
        L = int(z["n_links"])
        P = LH.HORIZON * L
        if feat == "v6":
            lh = z["lh"]
            lh6 = np.broadcast_to(lh[:, None, :, :],
                                  (lh.shape[0], LH.HORIZON, lh.shape[1], lh.shape[2])).reshape(-1, lh.shape[2])
            r6 = z6["rel_v6"]
            r6 = np.broadcast_to(r6[:, None, :, :],
                                 (r6.shape[0], LH.HORIZON, r6.shape[1], r6.shape[2])).reshape(-1, r6.shape[2])
            X = np.concatenate([z["v3"], lh6, r6], axis=1)
        else:
            X = RC.rel_X("on", feat, z)
        rows = (idx[:, None] * P + np.arange(P)[None, :]).reshape(-1)
        out[p] = {"idx": idx, "L": L, "truth": z["truth"][idx], "elig": z["elig"][idx],
                  "win": z["window_ids"][idx],
                  "prob": model.predict(np.ascontiguousarray(X[rows])).astype(np.float32).reshape(len(idx), LH.HORIZON, L)}
        del X
        z.close()
        z6.close()
        gc.collect()
    return out


def _truth(cond, p, idx):
    z = RC.load_panel(p)
    pre = "on" if cond == "on" else "ong"
    out = (z[f"{pre}_truth"][idx], z[f"{pre}_elig"][idx], int(z["n_links"]))
    z.close()
    return out


def tune(cond, probs):
    acc = {th: [] for th in TH_GRID}
    for p, d in probs.items():
        tr, el, L = _truth(cond, p, d["idx"])
        for j in range(len(d["idx"])):
            if cond == "on":
                for th in TH_GRID:
                    acc[th].append(LH.iou(RC.onset_pred(d["prob"][j], L, ("th", th)), tr[j], el[j]))
            else:
                pr = d["prob"][j].reshape(LH.HORIZON, L)
                for th in TH_GRID:
                    acc[th].append(LH.iou(pr >= th, tr[j], el[j]))
    means = {k: float(np.mean(v)) for k, v in acc.items()}
    best = max(means, key=means.get)
    return best, means[best]


def score_pseudo(cond, probs, th):
    rows = []
    for p, d in probs.items():
        tr, el, L = _truth(cond, p, d["idx"])
        for j in range(len(d["idx"])):
            if cond == "on":
                pred = RC.onset_pred(d["prob"][j], L, ("th", th))
            else:
                pred = d["prob"][j].reshape(LH.HORIZON, L) >= th
            rows.append({"panel": p, "condition": "queue_onset" if cond == "on" else "queue_ongoing",
                         "time": int(d["time"][j]), "iou": LH.iou(pred, tr[j], el[j])})
    return pd.DataFrame(rows)


def score_rel(cond, probs, th):
    rows = []
    for p, d in probs.items():
        L = d["L"]
        for j in range(len(d["idx"])):
            if cond == "on":
                pred = RC.onset_pred(d["prob"][j][LH.HORIZON - 1], L, ("th", th))
            else:
                pred = d["prob"][j] >= th
            rows.append({"panel": p, "condition": "queue_onset" if cond == "on" else "queue_ongoing",
                         "window_id": str(d["win"][j]),
                         "iou": LH.iou(pred, d["truth"][j], d["elig"][j])})
    return pd.DataFrame(rows)


def blend(a, b, w, key="prob"):
    out = {}
    for p in a:
        d = {"idx": a[p]["idx"], "L": a[p]["L"], "prob": w * a[p][key] + (1 - w) * b[p][key]}
        for k in ("time", "truth", "elig", "win"):
            if k in a[p]:
                d[k] = a[p][k]
        out[p] = d
    return out


def tune_blend(cond, pa, pb):
    rows = []
    for w in W_GRID:
        pbl = blend(pa, pb, w)
        for p, d in pbl.items():
            tr, el, L = _truth(cond, p, d["idx"])
            for j in range(len(d["idx"])):
                for th in TH_GRID:
                    if cond == "on":
                        pred = RC.onset_pred(d["prob"][j], L, ("th", th))
                    else:
                        pred = d["prob"][j].reshape(LH.HORIZON, L) >= th
                    rows.append({"w": w, "th": th, "iou": LH.iou(pred, tr[j], el[j])})
    g = pd.DataFrame(rows).groupby(["w", "th"]).iou.mean()
    (w, th) = g.idxmax()
    return float(w), float(th), float(g.max())


def train_cond(cond, feat, exclude_june, rounds):
    t0 = time.time()
    X, y, n = gather(cond, feat, exclude_june)
    m = train(cond, feat, X, y, rounds)
    print(f"  [{cond}/{feat}/{'A' if exclude_june else 'B'}] {n:,} rows x {len(FEATS[feat])} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    args = ap.parse_args()
    out = {}
    if args.stage == 1:
        for cond, rounds in (("on", RC.ON_ROUNDS), ("ong", RC.ONG_ROUNDS)):
            res = {}
            tag = "onset" if cond == "on" else "ongoing"
            # protocol B
            m = train_cond(cond, "v6", False, rounds)
            m.save_model(os.path.join(LH.MODELS, f"v6_{tag}_B.txt"))
            pj = probs_pseudo(cond, "v6", m, "tune")
            th, jan = tune(cond, pj)
            pf = probs_pseudo(cond, "v6", m, "feb")
            feb = score_pseudo(cond, pf, th)
            feb.to_csv(os.path.join(LH.OUT, f"v6_{cond}_feb.csv"), index=False)
            res["B"] = {"th": th, "jan": jan, "detail": feb}
            del m, pj, pf
            gc.collect()
            # protocol A
            m = train_cond(cond, "v6", True, rounds)
            m.save_model(os.path.join(LH.MODELS, f"v6_{tag}_A.txt"))
            jr = probs_rel(cond, "v6", m)
            june = score_rel(cond, jr, th)
            june.to_csv(os.path.join(LH.OUT, f"v6_{cond}_june.csv"), index=False)
            res["A"] = {"th": th, "detail": june}
            del m, jr
            gc.collect()
            out[cond] = res
        summ = {}
        for proto in ("B", "A"):
            d = pd.concat([out["on"][proto]["detail"], out["ong"][proto]["detail"]], ignore_index=True)
            agg = LH.aggregate(d)
            summ[proto] = {"overall": agg["overall"], "onset": agg["onset"], "ongoing": agg["ongoing"],
                           "per_panel": agg["per_panel"]}
            print(f"== v6 single protocol {proto}: overall {agg['overall']:.4f} | "
                  f"onset {agg['onset']:.4f} | ongoing {agg['ongoing']:.4f}", flush=True)
        base = json.load(open(os.path.join(LH.OUT, "final_cv_stride1.json")))
        print("\nv5 baseline (recorded): B %s | A %s" % (
            {k: round(base['on']['blend_B'][k], 4) for k in ('overall',)},
            {k: round(base['on']['blend_A'][k], 4) for k in ('overall',)}))
        with open(os.path.join(LH.OUT, "v6_stage1.json"), "w") as fh:
            json.dump({"v6_single": summ, "v5_single": {
                "B": {"overall": 0.8131, "onset": 0.7653, "ongoing": 0.8784},
                "A": {"overall": 0.8200, "onset": 0.8060, "ongoing": 0.8339}},
                "v5_blend": {"B": {"overall": 0.8195, "onset": 0.7714, "ongoing": 0.8812},
                             "A": {"overall": 0.8292, "onset": 0.8050, "ongoing": 0.8534}}},
                fh, indent=1, default=float)
    else:
        # stage 2: v3 + v6 blend vs recorded v3+v5 blend
        print("stage 2 not enabled in this run", flush=True)


if __name__ == "__main__":
    main()
