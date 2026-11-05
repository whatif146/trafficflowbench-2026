"""Honest cross-validation for the long-history queue model vs the frozen v3.

Two protocols, identical window sets / rows / config for the two feature sets:

  B (pseudo, primary): fit on selector-valid pseudo windows with origin before
    2031-01-01, pick the decision threshold on the January block, then score the
    untouched February block with the official nesting.
  A (released, secondary): fit on the same pseudo windows but with every June
    2030 origin removed, then score the 80 released *train* windows (all in June
    2030) whose horizon truth is present in the train plain layer.  June truth is
    never used for fitting or threshold selection (thresholds come from B).

Feature sets:
  v3 : the frozen 65 v3 features only (baseline reused from the v3 dataset cache)
  v5 : v3 + the 80 long-history causal features (lhcore.LH_FEATURES)
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

import lightgbm as lgb

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, num_threads=12, verbose=-1, seed=42,
              deterministic=True, force_row_wise=True)
ON_ROUNDS = 450
ONG_ROUNDS = 700

TUNE_EPOCH = int(LH.TUNE_START.timestamp())
CUT_EPOCH = int(LH.CUT.timestamp())
MAR_EPOCH = int(pd.Timestamp("2031-03-01", tz="UTC").timestamp())
JUNE0 = int(pd.Timestamp("2030-06-01", tz="UTC").timestamp())
JUNE1 = int(pd.Timestamp("2030-07-01", tz="UTC").timestamp())

ONSET_RULES = [("th", round(float(th), 2)) for th in np.arange(0.10, 0.86, 0.05)] + \
              [("top", k) for k in range(1, 13)]
ONG_THS = [round(float(t), 2) for t in np.arange(0.10, 0.81, 0.05)]


def feature_names(feat: str) -> list[str]:
    names = list(LH.v3_features())
    if feat == "v5":
        names = names + list(LH.LH_FEATURES)
    return names


def load_panel(panel: str):
    return np.load(os.path.join(LH.CACHE, f"tr_{panel}.npz"), allow_pickle=False)


def load_rel(panel: str):
    return np.load(os.path.join(LH.CACHE, f"rel_{panel}.npz"), allow_pickle=False)


def panel_X(cond: str, feat: str, z) -> np.ndarray:
    if cond == "on":
        base = z["on_v3"]
        return np.concatenate([base, z["on_lh"]], axis=1) if feat == "v5" else base
    base = z["ong_v3"]
    if feat == "v5":
        lh = z["ong_lh"]
        lh = np.broadcast_to(lh[:, None, :, :], (lh.shape[0], LH.HORIZON, lh.shape[1], lh.shape[2]))
        return np.concatenate([base, lh.reshape(base.shape[0], LH.N_LH)], axis=1)
    return base


def rel_X(cond: str, feat: str, z) -> np.ndarray:
    base = z["v3"]
    if feat == "v5":
        lh = z["lh"]
        L = lh.shape[1]
        n = lh.shape[0]
        lh6 = np.broadcast_to(lh[:, None, :, :], (n, LH.HORIZON, L, LH.N_LH)).reshape(n * LH.HORIZON * L, LH.N_LH)
        return np.concatenate([base, lh6], axis=1)
    return base


def per_window_rows(cond: str, L: int) -> int:
    return L if cond == "on" else LH.HORIZON * L


def gather_train(cond: str, feat: str, exclude_june: bool, stride: int,
                 panels=None):
    panels = panels or LH.PANELS
    Xs, ys = [], []
    for p in panels:
        z = load_panel(p)
        pre = "on" if cond == "on" else "ong"
        L = int(z["n_links"])
        t = z[f"{pre}_time"]
        per = per_window_rows(cond, L)
        keep = t < TUNE_EPOCH
        if exclude_june:
            keep &= ~((t >= JUNE0) & (t < JUNE1))
        if stride > 1:
            keep &= (np.arange(len(t)) % stride == 0)
        rows = np.repeat(keep, per)
        if rows.any():
            X = panel_X(cond, feat, z)
            Xs.append(X[rows])
            ys.append(z[f"{pre}_y"][rows])
            del X
        z.close()
        gc.collect()
    return np.concatenate(Xs, axis=0), np.concatenate(ys)


def train_model(cond: str, feat: str, X: np.ndarray, y: np.ndarray,
                rounds: int) -> lgb.Booster:
    ds = lgb.Dataset(X, label=y, feature_name=feature_names(feat),
                     free_raw_data=True, params=PARAMS)
    ds.construct()
    del X, y
    gc.collect()
    m = lgb.train(PARAMS, ds, num_boost_round=rounds)
    del ds
    gc.collect()
    return m


def onset_pred(p, L, rule) -> np.ndarray:
    kind, val = rule
    pred = np.zeros((LH.HORIZON, L), bool)
    if kind == "th":
        pred[LH.HORIZON - 1, p >= val] = True
    else:
        pred[LH.HORIZON - 1, np.argsort(-p)[:min(int(val), L)]] = True
    return pred


def _block_mask(pre: str, t, acc, where: str):
    if where == "tune":
        return acc & (t >= TUNE_EPOCH) & (t < CUT_EPOCH)
    if where == "eval_feb":
        return acc & (t >= CUT_EPOCH) & (t < MAR_EPOCH)
    raise ValueError(where)


def pseudo_probs(cond: str, feat: str, model, where: str) -> dict:
    out = {}
    for p in LH.PANELS:
        z = load_panel(p)
        pre = "on" if cond == "on" else "ong"
        L = int(z["n_links"])
        t = z[f"{pre}_time"]
        acc = z[f"{pre}_acc"]
        per = per_window_rows(cond, L)
        m = _block_mask(pre, t, acc, where)
        idx = np.where(m)[0]
        if len(idx):
            X = panel_X(cond, feat, z)
            rows = (idx[:, None] * per + np.arange(per)[None, :]).reshape(-1)
            prob = model.predict(X[rows]).astype(np.float32)
            out[p] = {"idx": idx, "time": t[idx], "prob": prob.reshape(len(idx), per), "L": L}
            del X
        z.close()
        gc.collect()
    return out


def _panel_truth(cond: str, panel: str, idx: np.ndarray):
    z = load_panel(panel)
    pre = "on" if cond == "on" else "ong"
    L = int(z["n_links"])
    truth = z[f"{pre}_truth"][idx]
    elig = z[f"{pre}_elig"][idx]
    z.close()
    return truth, elig, L


def tune_pseudo(cond: str, feat: str, model, where: str = "tune"):
    probs = pseudo_probs(cond, feat, model, where)
    rules = ONSET_RULES if cond == "on" else ONG_THS
    acc = {r: [] for r in rules}
    for p, d in probs.items():
        truth, elig, L = _panel_truth(cond, p, d["idx"])
        for j in range(len(d["idx"])):
            prob = d["prob"][j]
            if cond == "on":
                for rule in rules:
                    acc[rule].append(LH.iou(onset_pred(prob, L, rule), truth[j], elig[j]))
            else:
                pr = prob.reshape(LH.HORIZON, L)
                for th in rules:
                    acc[th].append(LH.iou(pr >= th, truth[j], elig[j]))
    means = {k: float(np.mean(v)) for k, v in acc.items() if v}
    best = max(means, key=means.get)
    return best, means[best], means


def score_pseudo(cond: str, feat: str, model, where: str, rule) -> pd.DataFrame:
    probs = pseudo_probs(cond, feat, model, where)
    rows = []
    for p, d in probs.items():
        truth, elig, L = _panel_truth(cond, p, d["idx"])
        for j in range(len(d["idx"])):
            if cond == "on":
                pred = onset_pred(d["prob"][j], L, rule)
            else:
                pred = d["prob"][j].reshape(LH.HORIZON, L) >= rule
            rows.append({"panel": p, "condition": "queue_onset" if cond == "on" else "queue_ongoing",
                         "time": int(d["time"][j]), "iou": LH.iou(pred, truth[j], elig[j]),
                         "truth_n": int(truth[j].sum())})
    return pd.DataFrame(rows)


def score_released(cond: str, feat: str, model, rule) -> pd.DataFrame:
    out = []
    cond_name = "queue_onset" if cond == "on" else "queue_ongoing"
    for p in LH.PANELS:
        z = load_rel(p)
        sel = (z["split"] == "train") & (z["condition"] == cond_name)
        idx = np.where(sel)[0]
        if len(idx) == 0:
            z.close()
            continue
        L = int(z["n_links"])
        per = LH.HORIZON * L          # rel arrays always carry 6 steps per window
        X = rel_X(cond, feat, z)
        rows = (idx[:, None] * per + np.arange(per)[None, :]).reshape(-1)
        prob = model.predict(X[rows]).astype(np.float32).reshape(len(idx), LH.HORIZON, L)
        truth = z["truth"][idx]
        elig = z["elig"][idx]
        for j in range(len(idx)):
            if cond == "on":
                pred = onset_pred(prob[j][LH.HORIZON - 1], L, rule)
            else:
                pred = prob[j] >= rule
            out.append({"panel": p, "condition": cond_name,
                        "window_id": str(z["window_ids"][idx[j]]),
                        "iou": LH.iou(pred, truth[j], elig[j]),
                        "truth_n": int(truth[j].sum())})
        del X
        z.close()
        gc.collect()
    return pd.DataFrame(out)


def run_protocol(feat: str, protocol: str, stride: int, rounds_scale: float,
                 thresholds: dict | None) -> dict:
    on_r = max(20, int(ON_ROUNDS * rounds_scale))
    ong_r = max(20, int(ONG_ROUNDS * rounds_scale))
    details, thr, nrows = [], {}, {}
    for cond, rounds in (("on", on_r), ("ong", ong_r)):
        t0 = time.time()
        X, y = gather_train(cond, feat, exclude_june=(protocol == "A"), stride=stride)
        nrows[cond] = int(X.shape[0])
        model = train_model(cond, feat, X, y, rounds)
        print(f"[{feat}/{protocol}] {cond}: {nrows[cond]:,} rows x "
              f"{len(feature_names(feat))} feats ({time.time() - t0:.0f}s)", flush=True)
        del X, y
        gc.collect()
        if protocol == "B":
            best, best_score, means = tune_pseudo(cond, feat, model, "tune")
            thr[cond] = list(best) if isinstance(best, tuple) else float(best)
            top = sorted(means.items(), key=lambda kv: -kv[1])[:3]
            print(f"[{feat}/B] {cond}: tune best {best} -> {best_score:.4f} top3 {top}", flush=True)
            detail = score_pseudo(cond, feat, model, "eval_feb", best)
        else:
            detail = score_released(cond, feat, model, thresholds[cond])
        details.append(detail)
        del model
        gc.collect()
    detail = pd.concat(details, ignore_index=True)
    agg = LH.aggregate(detail)
    print(f"== {feat} protocol {protocol} == "
          f"overall {agg['overall']:.4f} | onset {agg['onset']:.4f} | "
          f"ongoing {agg['ongoing']:.4f}", flush=True)
    print("per panel:", {k: round(v, 3) for k, v in agg["per_panel"].items()})
    print("per family:", {k: round(v, 3) for k, v in agg["per_family"].items()}, flush=True)
    out = {"feat": feat, "protocol": protocol, "stride": stride,
           "rounds": {"on": on_r, "ong": ong_r},
           "thresholds": thr, "n_fit_rows": nrows,
           "overall": agg["overall"], "onset": agg["onset"], "ongoing": agg["ongoing"],
           "per_panel": agg["per_panel"], "per_family": agg["per_family"]}
    tag = f"{feat}_{protocol}_s{stride}"
    detail.to_csv(os.path.join(LH.OUT, f"cv_{tag}.csv"), index=False)
    with open(os.path.join(LH.OUT, f"cv_{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="*", default=["v3", "v5"])
    ap.add_argument("--protocols", nargs="*", default=["B", "A"])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--rounds-scale", type=float, default=1.0)
    args = ap.parse_args()
    results = {}
    for feat in args.feats:
        thresholds = None
        if "B" in args.protocols:
            r = run_protocol(feat, "B", args.stride, args.rounds_scale, None)
            results[(feat, "B")] = r
            thresholds = {k: (tuple(v) if isinstance(v, list) else v) for k, v in r["thresholds"].items()}
        if "A" in args.protocols:
            if thresholds is None:
                raise SystemExit("protocol A needs the thresholds selected by protocol B")
            results[(feat, "A")] = run_protocol(feat, "A", args.stride, args.rounds_scale, thresholds)
    print("\n=== summary (stride %d, rounds %s) ===" % (args.stride, args.rounds_scale))
    for (feat, prot), r in results.items():
        print(f"{feat} {prot}: overall {r['overall']:.4f} "
              f"(on {r['onset']:.4f} / ong {r['ongoing']:.4f}) thr={r['thresholds']}")


if __name__ == "__main__":
    main()
