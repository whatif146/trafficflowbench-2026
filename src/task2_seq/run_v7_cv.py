"""v7: retrain the v3+v5 queue models on DENOISED latent-state labels and score
both protocols against DENOISED truth (spec: official truth is the underlying
state, not the noisy observation).

Fixed, LB-verified decision rules for every arm (no tuning):
  onset : p = 0.3*p_v3 + 0.7*p_v5 ; predict p >= 0.30 on the last step;
          top-1 fallback when that set is empty.
  ongoing: p = 0.4*p_v3 + 0.6*p_v5 ; predict p >= 0.20 on all six steps.

Arms: protocol B/A x {models trained on noisy labels, models trained on denoised
labels}, each scored against denoised truth (primary) and noisy truth (reference).
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

W_ON, W_ONG = 0.3, 0.4
ON_TH, ONG_TH = 0.30, 0.20
FEATS = {"v3": list(LH.v3_features()), "v5": list(RC.feature_names("v5"))}
PRE = {"on": "on", "ong": "ong"}


def load_v7(panel: str):
    return np.load(os.path.join(LH.CACHE, f"v7_{panel}.npz"), allow_pickle=False)


def gather(cond: str, feat: str, exclude_june: bool, labels: str):
    cols = FEATS[feat]
    pre = PRE[cond]
    plan, total = [], 0
    for p in LH.PANELS:
        z = RC.load_panel(p)
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
        Xf = EX.X_sel(cond, z, FC.COLS[feat])
        rows = np.repeat(keep, per)
        X[start:start + cnt] = Xf[rows]
        if labels == "dn":
            v7 = load_v7(p)
            yk = v7["on_y_dn"] if cond == "on" else v7["ong_y_dn"]
            y[start:start + cnt] = yk[rows]
            v7.close()
        else:
            y[start:start + cnt] = z[f"{pre}_y"][rows]
        del Xf, rows
        z.close()
        gc.collect()
    return X, y, total


def train(feat: str, X, y, rounds):
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
    pre = PRE[cond]
    for p in LH.PANELS:
        z = RC.load_panel(p)
        L = int(z["n_links"])
        t = z[f"{pre}_time"]
        acc = z[f"{pre}_acc"]
        per = RC.per_window_rows(cond, L)
        m = (acc & (t >= RC.TUNE_EPOCH) & (t < RC.CUT_EPOCH)) if where == "tune" else \
            (acc & (t >= RC.CUT_EPOCH) & (t < RC.MAR_EPOCH))
        idx = np.where(m)[0]
        if len(idx):
            X = EX.X_sel(cond, z, FC.COLS[feat])
            rows = (idx[:, None] * per + np.arange(per)[None, :]).reshape(-1)
            out[p] = {"idx": idx, "time": t[idx], "L": L,
                      "prob": model.predict(np.ascontiguousarray(X[rows])).astype(np.float32).reshape(len(idx), per)}
            del X
        z.close()
        gc.collect()
    return out


def probs_rel(cond: str, feat: str, model) -> dict:
    out = {}
    cond_name = "queue_onset" if cond == "on" else "queue_ongoing"
    for p in LH.PANELS:
        z = np.load(os.path.join(LH.CACHE, f"rel_{p}.npz"), allow_pickle=False)
        sel = (z["split"] == "train") & (z["condition"] == cond_name)
        idx = np.where(sel)[0]
        L = int(z["n_links"])
        P = LH.HORIZON * L
        X = RC.rel_X("on", feat, z)
        rows = (idx[:, None] * P + np.arange(P)[None, :]).reshape(-1)
        out[p] = {"idx": idx, "L": L, "win": z["window_ids"][idx],
                  "prob": model.predict(np.ascontiguousarray(X[rows])).astype(np.float32).reshape(len(idx), LH.HORIZON, L)}
        del X
        z.close()
        gc.collect()
    return out


def onset_pred_blend(p, L):
    pred = np.zeros((LH.HORIZON, L), bool)
    sel = p >= ON_TH
    if not sel.any():
        sel[int(np.argmax(p))] = True
    pred[LH.HORIZON - 1] = sel
    return pred


def blend(a, b, w):
    return {p: {"idx": a[p]["idx"], "L": a[p]["L"],
                "prob": w * a[p]["prob"] + (1 - w) * b[p]["prob"]} for p in a}


def arm_truth(cond: str, protocol: str, panel: str, idx: np.ndarray, kind: str):
    pre = PRE[cond]
    if protocol == "A":
        z = np.load(os.path.join(LH.CACHE, f"rel_{panel}.npz"), allow_pickle=False)
        el = z["elig"][idx]
        if kind == "dn":
            v7 = load_v7(panel)
            tr = v7["rel_truth_dn"][idx]
            v7.close()
        else:
            tr = z["truth"][idx]
        z.close()
        return tr, el
    z = RC.load_panel(panel)
    el = z[f"{pre}_elig"][idx]
    if kind == "dn":
        v7 = load_v7(panel)
        tr = v7["on_truth_dn" if cond == "on" else "ong_truth_dn"][idx]
        v7.close()
    else:
        tr = z[f"{pre}_truth"][idx]
    z.close()
    return tr, el


def agg(detail: pd.DataFrame) -> dict:
    a = LH.aggregate(detail)
    return {"overall": a["overall"], "onset": a["onset"], "ongoing": a["ongoing"],
            "per_panel": a["per_panel"]}


def main():
    t0 = time.time()
    out = {}
    for protocol in ("B", "A"):
        excl = protocol == "A"
        for labels in ("noisy", "dn"):
            arm = {}
            for cond, rounds in (("on", RC.ON_ROUNDS), ("ong", RC.ONG_ROUNDS)):
                ms = {}
                for feat in ("v3", "v5"):
                    X, y, n = gather(cond, feat, exclude_june=excl, labels=labels)
                    ms[feat] = train(feat, X, y, rounds)
                    print(f"[{protocol}/{labels}/{cond}/{feat}] {n:,} rows ({time.time() - t0:.0f}s)",
                          flush=True)
                    del X, y
                    gc.collect()
                arm[cond] = {f: (probs_pseudo(cond, f, ms[f], "feb") if protocol == "B"
                                 else probs_rel(cond, f, ms[f])) for f in ("v3", "v5")}
                del ms
                gc.collect()
            for kind in ("dn", "noisy"):
                det = []
                for cond in ("on", "ong"):
                    w = W_ON if cond == "on" else W_ONG
                    pbl = blend(arm[cond]["v3"], arm[cond]["v5"], w)
                    rows = []
                    for p, d in pbl.items():
                        tr, el = arm_truth(cond, protocol, p, d["idx"], kind)
                        L = d["L"]
                        for j in range(len(d["idx"])):
                            if cond == "on":
                                pj = d["prob"][j]
                                p_last = pj[LH.HORIZON - 1] if pj.ndim == 2 else pj
                                pred = onset_pred_blend(p_last, L)
                            else:
                                pred = d["prob"][j].reshape(LH.HORIZON, L) >= ONG_TH
                            rows.append({"panel": p,
                                         "condition": "queue_onset" if cond == "on" else "queue_ongoing",
                                         "iou": LH.iou(pred, tr[j], el[j])})
                    det.append(pd.DataFrame(rows))
                a = agg(pd.concat(det, ignore_index=True))
                out.setdefault(protocol, {}).setdefault(labels, {})[kind] = a
                print(f"== {protocol} / {labels}-label models / {kind} truth: overall "
                      f"{a['overall']:.4f} | onset {a['onset']:.4f} | ongoing {a['ongoing']:.4f}",
                      flush=True)
            del arm
            gc.collect()
    with open(os.path.join(LH.OUT, "v7_cv.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n=== v7 decision table (denoised truth) ===", flush=True)
    passed = True
    for proto in ("B", "A"):
        base = out[proto]["noisy"]["dn"]
        new = out[proto]["dn"]["dn"]
        dlt = new["overall"] - base["overall"]
        cond_ok = all(new[c] - base[c] > -0.005 for c in ("onset", "ongoing"))
        ok = (dlt >= 0.01) and cond_ok
        passed &= ok
        print(f"{proto}: noisy-label blend {base['overall']:.4f} (on {base['onset']:.4f}/ong "
              f"{base['ongoing']:.4f}) -> v7 {new['overall']:.4f} (on {new['onset']:.4f}/ong "
              f"{new['ongoing']:.4f}) delta {dlt:+.4f} cond_ok {cond_ok} -> {'PASS' if ok else 'FAIL'}",
              flush=True)
    print(f"PASS={passed}", flush=True)


if __name__ == "__main__":
    main()

