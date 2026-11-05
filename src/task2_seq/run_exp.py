"""Experiments: does the long-history block help on top of v3, and can a
v3+v5 blend / a lean long-history subset recover the onset regression?

Everything is selected on the January block only; February (pseudo-CV) and the
released June train windows (protocol A) are reported once.  Protocol A models
never see a June 2030 origin, and thresholds come from protocol B.
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

import lightgbm as lgb

V3N = list(LH.v3_features())
ALLN = V3N + list(LH.LH_FEATURES)
NV3 = len(V3N)

# lean long-history block: origin-row state + queue front + 3 h levels/trends
LITE_LH = [
    "r_T", "r_T_spat", "rT_obs", "q_T", "q_T_spat", "fc_T",
    "r_med_l65", "r_med_o60", "r_med_all", "rT_minus_medl", "dr_lo",
    "qfrac_l", "qfrac_all", "q_age", "t_since_q", "fc_slope",
    "dist_up_T", "dist_dn_T", "dist_up_30", "ddist_up", "ddist_dn",
    "qsum_up5_T", "qsum_dn5_T", "nq_T_panel", "qfa_panel_l", "d_qfa_panel",
    "on_l_cap", "on_slope_cap",
]
FEATSETS = {
    "v3": V3N,
    "v5lite": V3N + LITE_LH,
    "v5": ALLN,
}
TH_GRID = [round(float(t), 2) for t in np.arange(0.10, 0.81, 0.05)]
W_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def cols_for(names):
    return np.array([ALLN.index(n) for n in names], dtype=np.int64)


def X_sel(cond: str, z, cols: np.ndarray) -> np.ndarray:
    v3c = cols[cols < NV3]
    lhc = cols[cols >= NV3] - NV3
    out = []
    if len(v3c):
        out.append((z["on_v3"] if cond == "on" else z["ong_v3"])[:, v3c])
    if len(lhc):
        if cond == "on":
            lh = z["on_lh"]
        else:
            raw = z["ong_lh"]
            lh = np.broadcast_to(raw[:, None, :, :], (raw.shape[0], LH.HORIZON, raw.shape[1], raw.shape[2]))
            lh = lh.reshape(-1, raw.shape[2])
        out.append(lh[:, lhc])
    if len(out) == 1:
        return np.ascontiguousarray(out[0])
    return np.concatenate(out, axis=1)


def gather(cond: str, cols: np.ndarray, stride: int, exclude_june: bool):
    Xs, ys = [], []
    for p in LH.PANELS:
        z = RC.load_panel(p)
        pre = "on" if cond == "on" else "ong"
        L = int(z["n_links"])
        per = RC.per_window_rows(cond, L)
        t = z[f"{pre}_time"]
        keep = t < RC.TUNE_EPOCH             # fit block: pre-January only
        if exclude_june:
            keep &= ~((t >= RC.JUNE0) & (t < RC.JUNE1))
        if stride > 1:
            keep &= (np.arange(len(t)) % stride == 0)
        rows = np.repeat(keep, per)
        if rows.any():
            X = X_sel(cond, z, cols)
            Xs.append(X[rows])
            ys.append(z[f"{pre}_y"][rows])
            del X
        z.close()
        gc.collect()
    return np.concatenate(Xs, axis=0), np.concatenate(ys)


def train(cond: str, names: list[str], X, y, rounds: int):
    ds = lgb.Dataset(X, label=y, feature_name=list(names), free_raw_data=True, params=RC.PARAMS)
    ds.construct()
    del X, y
    gc.collect()
    m = lgb.train(RC.PARAMS, ds, num_boost_round=rounds)
    del ds
    gc.collect()
    return m


def block_probs(cond: str, cols: np.ndarray, model, where: str) -> dict:
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
        elif where == "feb":
            m = acc & (t >= RC.CUT_EPOCH) & (t < RC.MAR_EPOCH)
        else:
            raise ValueError(where)
        idx = np.where(m)[0]
        if len(idx):
            X = X_sel(cond, z, cols)
            rows = (idx[:, None] * per + np.arange(per)[None, :]).reshape(-1)
            prob = model.predict(np.ascontiguousarray(X[rows])).astype(np.float32)
            out[p] = {"idx": idx, "time": t[idx], "prob": prob.reshape(len(idx), per), "L": L}
            del X
        z.close()
        gc.collect()
    return out


def june_probs(cond: str, cols: np.ndarray, model) -> dict:
    out = {}
    cond_name = "queue_onset" if cond == "on" else "queue_ongoing"
    for p in LH.PANELS:
        z = RC.load_rel(p)
        sel = (z["split"] == "train") & (z["condition"] == cond_name)
        idx = np.where(sel)[0]
        if len(idx):
            L = int(z["n_links"])
            P = LH.HORIZON * L
            X = RC.rel_X("on", "v3", z)
            if np.any(cols >= NV3):
                lhc = cols[cols >= NV3] - NV3
                lh = z["lh"]
                lh6 = np.broadcast_to(lh[:, None, :, :], (lh.shape[0], LH.HORIZON, lh.shape[1], lh.shape[2]))
                lh6 = lh6.reshape(-1, lh.shape[2])[:, lhc]
                X = np.concatenate([X, lh6], axis=1)
            if np.any(cols < NV3):
                keep = np.where(cols < NV3)[0]
                X = np.concatenate([X[:, keep], X[:, NV3:]], axis=1) if np.any(cols >= NV3) else X[:, keep]
            rows = (idx[:, None] * P + np.arange(P)[None, :]).reshape(-1)
            prob = model.predict(np.ascontiguousarray(X[rows])).astype(np.float32)
            out[p] = {"idx": idx, "prob": prob.reshape(len(idx), LH.HORIZON, L), "L": L,
                      "truth": z["truth"][idx], "elig": z["elig"][idx],
                      "win": z["window_ids"][idx]}
            del X
        z.close()
        gc.collect()
    return out


def _truth(cond: str, p: str, idx: np.ndarray):
    z = RC.load_panel(p)
    pre = "on" if cond == "on" else "ong"
    out = (z[f"{pre}_truth"][idx], z[f"{pre}_elig"][idx], int(z["n_links"]))
    z.close()
    return out


def score_feb(cond: str, probs: dict, th: float) -> pd.DataFrame:
    rows = []
    for p, d in probs.items():
        truth, elig, L = _truth(cond, p, d["idx"])
        for j in range(len(d["idx"])):
            if cond == "on":
                pred = RC.onset_pred(d["prob"][j], L, ("th", th))
            else:
                pred = d["prob"][j].reshape(LH.HORIZON, L) >= th
            rows.append({"panel": p, "condition": "queue_onset" if cond == "on" else "queue_ongoing",
                         "time": int(d["time"][j]), "iou": LH.iou(pred, truth[j], elig[j])})
    return pd.DataFrame(rows)


def tune_feb(cond: str, probs: dict):
    rows = []
    for p, d in probs.items():
        truth, elig, L = _truth(cond, p, d["idx"])
        for j in range(len(d["idx"])):
            for th in TH_GRID:
                if cond == "on":
                    pred = RC.onset_pred(d["prob"][j], L, ("th", th))
                else:
                    pred = d["prob"][j].reshape(LH.HORIZON, L) >= th
                rows.append({"th": th, "iou": LH.iou(pred, truth[j], elig[j])})
    df = pd.DataFrame(rows)
    means = df.groupby("th").iou.mean()
    best = float(means.idxmax())
    return best, float(means.max()), means


def score_june(cond: str, probs: dict, th: float) -> pd.DataFrame:
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


def blend_probs(pv3: dict, pv5: dict, w: float) -> dict:
    out = {}
    for p in pv3:
        a, b = pv3[p], pv5[p]
        out[p] = {"idx": a["idx"], "time": a["time"], "L": a["L"],
                  "prob": w * a["prob"] + (1.0 - w) * b["prob"]}
    return out


def tune_blend(cond: str, pv3: dict, pv5: dict):
    rows = []
    for w in W_GRID:
        pb = blend_probs(pv3, pv5, w)
        for p, d in pb.items():
            truth, elig, L = _truth(cond, p, d["idx"])
            for j in range(len(d["idx"])):
                for th in TH_GRID:
                    if cond == "on":
                        pred = RC.onset_pred(d["prob"][j], L, ("th", th))
                    else:
                        pred = d["prob"][j].reshape(LH.HORIZON, L) >= th
                    rows.append({"w": w, "th": th, "iou": LH.iou(pred, truth[j], elig[j])})
    df = pd.DataFrame(rows)
    g = df.groupby(["w", "th"]).iou.mean()
    (w, th) = g.idxmax()
    return float(w), float(th), float(g.max()), g


def blend_june(pv3: dict, pv5: dict, w: float, th: float, cond: str) -> pd.DataFrame:
    out = {}
    for p in pv3:
        a, b = pv3[p], pv5[p]
        out[p] = {"idx": a["idx"], "L": a["L"], "truth": a["truth"], "elig": a["elig"],
                  "win": a["win"], "prob": w * a["prob"] + (1 - w) * b["prob"]}
    rows = []
    for p, d in out.items():
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


def run(cond: str, stride: int, feats=("v3", "v5lite", "v5")) -> dict:
    rounds = RC.ON_ROUNDS if cond == "on" else RC.ONG_ROUNDS
    res = {"cond": cond, "stride": stride}
    for protocol in ("B", "A"):
        store = {}
        for feat in feats:
            t0 = time.time()
            cols = cols_for(FEATSETS[feat])
            X, y = gather(cond, cols, stride, exclude_june=(protocol == "A"))
            model = train(cond, FEATSETS[feat], X, y, rounds)
            print(f"[{cond}/{protocol}] {feat}: {len(X):,} rows x {len(FEATSETS[feat])} feats "
                  f"({time.time() - t0:.0f}s)", flush=True)
            del X, y
            gc.collect()
            if protocol == "B":
                pj = block_probs(cond, cols, model, "tune")
                th, jan, means = tune_feb(cond, pj)
                pf = block_probs(cond, cols, model, "feb")
                feb = score_feb(cond, pf, th)
                store[feat] = {"th": th, "jan": jan, "feb": feb, "pj": pj, "pf": pf,
                               "jan_means": means.to_dict(), "cols": cols}
            else:
                jp = june_probs(cond, cols, model)
                store[feat] = {"jp": jp, "cols": cols}
            del model
            gc.collect()
        # single-feature summaries
        for feat in feats:
            if protocol == "B":
                agg = LH.aggregate(store[feat]["feb"])
                res[f"{feat}_B"] = {"th": store[feat]["th"], "jan": store[feat]["jan"],
                                    "overall": agg["overall"], "onset": agg["onset"],
                                    "ongoing": agg["ongoing"], "per_panel": agg["per_panel"]}
                print(f"[{cond}/B] {feat}: th {store[feat]['th']} jan {store[feat]['jan']:.4f} "
                      f"-> FEB {agg['overall']:.4f}", flush=True)
            else:
                th = store["v3"] if False else None
        if protocol == "B":
            # blends of v3 + v5
            for partner in ("v5lite", "v5"):
                w, th, jan, g = tune_blend(cond, store["v3"]["pj"], store[partner]["pj"])
                pb_feb = blend_probs(store["v3"]["pf"], store[partner]["pf"], w)
                feb = score_feb(cond, pb_feb, th)
                agg = LH.aggregate(feb)
                res[f"blend_v3_{partner}_B"] = {"w": w, "th": th, "jan": jan,
                                                "overall": agg["overall"], "onset": agg["onset"],
                                                "ongoing": agg["ongoing"]}
                print(f"[{cond}/B] blend v3+{partner}: w {w} th {th} jan {jan:.4f} "
                      f"-> FEB {agg['overall']:.4f}", flush=True)
            # keep detail of best blend for protocol A
            res["_store"] = store
        else:
            for feat in feats:
                th_b = res[f"{feat}_B"]["th"]
                agg = LH.aggregate(score_june(cond, store[feat]["jp"], th_b))
                res[f"{feat}_A"] = {"th": th_b, "overall": agg["overall"], "onset": agg["onset"],
                                    "ongoing": agg["ongoing"], "per_panel": agg["per_panel"]}
                print(f"[{cond}/A] {feat}: th {th_b} -> JUNE {agg['overall']:.4f}", flush=True)
            sb = res["_storeB"] if "_storeB" in res else None
            for partner in ("v5lite", "v5"):
                w, th, jan, _ = res[f"blend_v3_{partner}_B"]["w"], res[f"blend_v3_{partner}_B"]["th"], res[f"blend_v3_{partner}_B"]["jan"], None
                pj3 = res["_store"][partner]["jp"] if False else None
                # blend June probs using protocol-A single-feature probs for v3 and partner
                p3 = store["v3"]["jp"]
                pp = store[partner]["jp"]
                feb = blend_june(p3, pp, w, th, cond)
                agg = LH.aggregate(feb)
                res[f"blend_v3_{partner}_A"] = {"w": w, "th": th, "overall": agg["overall"],
                                                "onset": agg["onset"], "ongoing": agg["ongoing"]}
                print(f"[{cond}/A] blend v3+{partner}: w {w} th {th} -> JUNE {agg['overall']:.4f}", flush=True)
    return res


rows_n = 0


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--conds", nargs="*", default=["on", "ong"])
    args = ap.parse_args()
    # run both conditions for protocol B first, then A, so the thresholds exist
    out = {"on": run("on", args.stride), "ong": run("ong", args.stride)}
    for cond in ("on", "ong"):
        out[cond].pop("_store", None)
    with open(os.path.join(LH.OUT, f"exp_stride{args.stride}.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n=== summary (stride %d) ===" % args.stride)
    for cond in ("on", "ong"):
        for k, v in out[cond].items():
            if k in ("cond", "stride"):
                continue
            print(f"{cond} {k}: overall {v['overall']:.4f} (jan {v.get('jan', float('nan')):.4f})")


if __name__ == "__main__":
    main()
