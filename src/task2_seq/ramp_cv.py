"""Ramp jam-sensor features in the ONGOING model: honest A/B.

Compares three ongoing halves on identical rows / config / January rule
selection: v3 (65 feats), v5 (65+80 long-history), v5r (65+80+8 ramp ratios),
each blended with the v3 probability at the January-selected weight, then scored
on February (protocol B) and the released June windows (protocol A).
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
import run_final_cv as FC
import run_exp as EX

import lightgbm as lgb


def train(cond, names, X, y, rounds):
    ds = lgb.Dataset(X, label=y, feature_name=list(names), free_raw_data=True,
                     params=RC.PARAMS)
    ds.construct()
    del X, y
    gc.collect()
    m = lgb.train(RC.PARAMS, ds, num_boost_round=rounds)
    del ds
    gc.collect()
    return m

V3N = list(LH.v3_features())
LHN = list(LH.LH_FEATURES)
RRN = list(LH.RAMP_FEATURES)
V5N = V3N + LHN
V5RN = V3N + LHN + RRN
NV3, NV5 = len(V3N), len(V5N)
W_GRID = [0.0001, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9999]
TH_GRID = [round(float(t), 2) for t in np.arange(0.10, 0.81, 0.05)]


def xsel(cond, z, names):
    v3c = [i for i, n in enumerate(names) if i < NV3]
    lhc = [i - NV3 for i, n in enumerate(names) if NV3 <= i < NV5]
    rrc = [i - NV5 for i, n in enumerate(names) if i >= NV5]
    out = [(z["on_v3"] if cond == "on" else z["ong_v3"])[:, v3c]] if v3c else []
    if lhc:
        if cond == "ong":
            lh = z["ong_lh"]
            nf = lh.shape[2]
            lh = np.broadcast_to(lh[:, None, :, :],
                                 (lh.shape[0], LH.HORIZON, lh.shape[1], nf)).reshape(-1, nf)
        else:
            lh = z["on_lh"]
        out.append(lh[:, lhc])
    if rrc:
        if cond == "ong":
            rr = z["ong_lhr"]
            nf = rr.shape[2]
            rr = np.broadcast_to(rr[:, None, :, :],
                                 (rr.shape[0], LH.HORIZON, rr.shape[1], nf)).reshape(-1, nf)
        else:
            rr = z["on_lhr"]
        out.append(rr[:, rrc])
    return np.ascontiguousarray(out[0] if len(out) == 1 else np.concatenate(out, axis=1))


def xsel_rel(cond, z, names):
    v3c = [i for i, n in enumerate(names) if i < NV3]
    lhc = [i - NV3 for i, n in enumerate(names) if NV3 <= i < NV5]
    rrc = [i - NV5 for i, n in enumerate(names) if i >= NV5]
    L = int(z["n_links"])
    P = LH.HORIZON * L
    out = [z["v3"][:, v3c]] if v3c else []
    if lhc:
        lh = z["lh"]
        nf = lh.shape[2]
        lh6 = np.broadcast_to(lh[:, None, :, :],
                              (lh.shape[0], LH.HORIZON, lh.shape[1], nf)).reshape(-1, nf)
        out.append(lh6[:, lhc])
    if rrc:
        rr = z["lhr"]
        nf = rr.shape[2]
        rr6 = np.broadcast_to(rr[:, None, :, :],
                              (rr.shape[0], LH.HORIZON, rr.shape[1], nf)).reshape(-1, nf)
        out.append(rr6[:, rrc])
    return np.ascontiguousarray(out[0] if len(out) == 1 else np.concatenate(out, axis=1))


def gather(cond, names, exclude_june):
    Xs, ys = [], []
    for p in LH.PANELS:
        z = RC.load_panel(p)
        pre = "on" if cond == "on" else "ong"
        L = int(z["n_links"])
        per = RC.per_window_rows(cond, L)
        t = z[f"{pre}_time"]
        keep = t < RC.TUNE_EPOCH
        if exclude_june:
            keep &= ~((t >= RC.JUNE0) & (t < RC.JUNE1))
        rows = np.repeat(keep, per)
        if rows.any():
            X = xsel(cond, z, names)
            Xs.append(X[rows])
            ys.append(z[f"{pre}_y"][rows])
            del X
        z.close()
        gc.collect()
    return np.concatenate(Xs, axis=0), np.concatenate(ys)


def probs_block(cond, names, model, where):
    out = {}
    for p in LH.PANELS:
        z = RC.load_panel(p)
        pre = "on" if cond == "on" else "ong"
        L = int(z["n_links"])
        t = z[f"{pre}_time"]
        acc = z[f"{pre}_acc"]
        per = RC.per_window_rows(cond, L)
        if where == "tune":
            m = acc & (t >= RC.TUNE_EPOCH) & (t < RC.CUT_EPOCH)
        else:
            m = acc & (t >= RC.CUT_EPOCH) & (t < RC.MAR_EPOCH)
        idx = np.where(m)[0]
        if len(idx):
            X = xsel(cond, z, names)
            rows = (idx[:, None] * per + np.arange(per)[None, :]).reshape(-1)
            out[p] = {"idx": idx, "time": t[idx], "L": L,
                      "prob": model.predict(X[rows]).astype(np.float32).reshape(len(idx), per)}
            del X
        z.close()
        gc.collect()
    return out


def probs_rel(cond, names, model):
    out = {}
    cond_name = "queue_onset" if cond == "on" else "queue_ongoing"
    for p in LH.PANELS:
        z = np.load(os.path.join(LH.CACHE, f"rel_{p}.npz"), allow_pickle=False)
        sel = (z["split"] == "train") & (z["condition"] == cond_name)
        idx = np.where(sel)[0]
        L = int(z["n_links"])
        P = LH.HORIZON * L
        X = xsel_rel(cond, z, names)
        rows = (idx[:, None] * P + np.arange(P)[None, :]).reshape(-1)
        out[p] = {"idx": idx, "L": L, "truth": z["truth"][idx], "elig": z["elig"][idx],
                  "win": z["window_ids"][idx],
                  "prob": model.predict(X[rows]).astype(np.float32).reshape(len(idx), LH.HORIZON, L)}
        del X
        z.close()
        gc.collect()
    return out


def truth(cond, p, idx):
    z = RC.load_panel(p)
    pre = "on" if cond == "on" else "ong"
    out = (z[f"{pre}_truth"][idx], z[f"{pre}_elig"][idx], int(z["n_links"]))
    z.close()
    return out


def score_pseudo(cond, probs, th):
    rows = []
    for p, d in probs.items():
        tr, el, L = truth(cond, p, d["idx"])
        for j in range(len(d["idx"])):
            pred = d["prob"][j].reshape(LH.HORIZON, L) >= th
            rows.append({"panel": p, "condition": "queue_ongoing", "iou": LH.iou(pred, tr[j], el[j])})
    return pd.DataFrame(rows)


def score_rel(cond, probs, th):
    rows = []
    for p, d in probs.items():
        L = d["L"]
        for j in range(len(d["idx"])):
            pred = d["prob"][j] >= th
            rows.append({"panel": p, "condition": "queue_ongoing",
                         "window_id": str(d["win"][j]),
                         "iou": LH.iou(pred, d["truth"][j], d["elig"][j])})
    return pd.DataFrame(rows)


def tune(cond, probs):
    rows = []
    for p, d in probs.items():
        tr, el, L = truth(cond, p, d["idx"])
        for j in range(len(d["idx"])):
            pr = d["prob"][j].reshape(LH.HORIZON, L)
            for th in TH_GRID:
                rows.append({"th": th, "iou": LH.iou(pr >= th, tr[j], el[j])})
    g = pd.DataFrame(rows).groupby("th").iou.mean()
    return float(g.idxmax()), float(g.max())


def blend(a, b, w):
    out = {}
    for p in a:
        out[p] = {"idx": a[p]["idx"], "L": a[p]["L"],
                  "prob": w * a[p]["prob"] + (1 - w) * b[p]["prob"]}
    return out


def main():
    t0 = time.time()
    cond = "ong"
    variants = {"v3": V3N, "v5": V5N, "v5r": V5RN}
    res = {"rows": {}, "B": {}, "A": {}, "thresholds": {}}
    for proto, exc in (("B", False), ("A", True)):
        store = {}
        for name, names in variants.items():
            X, y = gather(cond, names, exclude_june=exc)
            res["rows"][name] = int(X.shape[0])
            m = train(cond, names, X, y, RC.ONG_ROUNDS)
            del X, y
            gc.collect()
            if proto == "B":
                pj = probs_block(cond, names, m, "tune")
                pf = probs_block(cond, names, m, "feb")
                store[name] = {"pj": pj, "pf": pf}
                th, jan = tune(cond, pj)
                res["thresholds"][name] = th
                feb = score_pseudo(cond, pf, th)
                feb.to_csv(os.path.join(LH.OUT, f"ramp_{name}_feb.csv"), index=False)
                res["B"][name] = {"overall": LH.aggregate(feb)["overall"], "th": th, "jan": jan}
                print(f"[ong/B] {name}: {X.shape if False else res['rows'][name]:,} rows | "
                      f"th {th} jan {jan:.4f} -> FEB {res['B'][name]['overall']:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            else:
                store[name] = {"jr": probs_rel(cond, names, m)}
                res["A"][name] = {}
            del m
            gc.collect()
        if proto == "B":
            for partner in ("v5", "v5r"):
                pbj = blend(store["v3"]["pj"], store[partner]["pj"], 0.6)
                best = (None, -1.0)
                for th in TH_GRID:
                    rows = []
                    for p, d in pbj.items():
                        tr, el, L = truth(cond, p, d["idx"])
                        for j in range(len(d["idx"])):
                            pr = d["prob"][j].reshape(LH.HORIZON, L)
                            rows.append(LH.iou(pr >= th, tr[j], el[j]))
                    jan_avg = float(np.mean(rows))
                    if jan_avg > best[1]:
                        best = (th, jan_avg)
                th, jan = best
                pbf = blend(store["v3"]["pf"], store[partner]["pf"], 0.6)
                feb = score_pseudo(cond, pbf, th)
                feb.to_csv(os.path.join(LH.OUT, f"ramp_blend_{partner}_feb.csv"), index=False)
                res["B"][f"blend_{partner}"] = {"overall": LH.aggregate(feb)["overall"],
                                                "th": th, "jan": jan, "w": 0.6}
                print(f"[ong/B] blend v3+{partner} (w .6): th {th} jan {jan:.4f} -> "
                      f"FEB {res['B'][f'blend_{partner}']['overall']:.4f}", flush=True)
                store[f"blend_{partner}"] = {"th": th}
        else:
            for name in variants:
                jr = store[name]["jr"]
                th = res["thresholds"][name]
                june = score_rel(cond, jr, th)
                june.to_csv(os.path.join(LH.OUT, f"ramp_{name}_june.csv"), index=False)
                res["A"][name] = {"overall": LH.aggregate(june)["overall"], "th": th}
                print(f"[ong/A] {name}: th {th} -> JUNE {res['A'][name]['overall']:.4f}", flush=True)
            for partner in ("v5", "v5r"):
                th = res["B"][f"blend_{partner}"]["th"]
                jr = blend(store["v3"]["jr"], store[partner]["jr"], 0.6)
                june = score_rel(cond, jr, th)
                june.to_csv(os.path.join(LH.OUT, f"ramp_blend_{partner}_june.csv"), index=False)
                res["A"][f"blend_{partner}"] = {"overall": LH.aggregate(june)["overall"], "th": th}
                print(f"[ong/A] blend v3+{partner}: th {th} -> "
                      f"JUNE {res['A'][f'blend_{partner}']['overall']:.4f}", flush=True)
    with open(os.path.join(LH.OUT, "ramp_cv.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
