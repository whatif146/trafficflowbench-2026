"""v10: label-regime comparison under both truth hypotheses.

H2 = NaN-aware 3-step temporal median of the observed speed, thresholded at v_cut
     (this is what the shipped v3 cache/labels ALREADY use — verified bit-exact).
H1 = raw single-step observation, thresholded at v_cut.

Provides the requested two-proxy table: models trained on H2 labels (current) vs
models trained on H1 labels (raw), each scored under H1 and H2 truth, on the
official two-protocol harness with the fixed LB-verified rules.

  python run_v10_cv.py
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
import run_v7_cv as V7

import lightgbm as lgb

W_ON, W_ONG = V7.W_ON, V7.W_ONG
ON_TH, ONG_TH = V7.ON_TH, V7.ONG_TH
FEATS = V7.FEATS


def load_h1(panel: str):
    return np.load(os.path.join(LH.CACHE, f"h1_{panel}.npz"), allow_pickle=False)


def load_cache(panel: str):
    return np.load(os.path.join(LH.CACHE, f"tr_{panel}.npz"), allow_pickle=False)


def gather(cond: str, feat: str, exclude_june: bool, labels: str):
    pre = "on" if cond == "on" else "ong"
    cols = FEATS[feat]
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
        if labels == "h1":
            h1 = load_h1(p)
            yk = h1["on_y_h1"] if cond == "on" else h1["ong_y_h1"]
            y[start:start + cnt] = yk[rows]
            h1.close()
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


def truth(cond, protocol, panel, idx, hyp):
    pre = "on" if cond == "on" else "ong"
    if protocol == "A":
        z = np.load(os.path.join(LH.CACHE, f"rel_{panel}.npz"), allow_pickle=False)
        el = z["elig"][idx]
        if hyp == "h1":
            h = load_h1(panel)
            tr = h["rel_truth_h1"][idx]
            h.close()
        else:
            tr = z["truth"][idx]
        z.close()
        return tr, el
    z = RC.load_panel(panel)
    el = z[f"{pre}_elig"][idx]
    if hyp == "h1":
        h = load_h1(panel)
        tr = h["on_truth_h1" if cond == "on" else "ong_truth_h1"][idx]
        h.close()
    else:
        tr = z[f"{pre}_truth"][idx]
    z.close()
    return tr, el


def score(probs: dict, cond: str, protocol: str, hyp: str) -> pd.DataFrame:
    rows = []
    for p, d in probs.items():
        tr, el = truth(cond, protocol, p, d["idx"], hyp)
        L = d["L"]
        for j in range(len(d["idx"])):
            if cond == "on":
                pj = d["prob"][j]
                p_last = pj[LH.HORIZON - 1] if pj.ndim == 2 else pj
                pred = V7.onset_pred_blend(p_last, L)
            else:
                pr = d["prob"][j].reshape(LH.HORIZON, L)
                pred = pr >= ONG_TH
                if not pred.any():
                    pred.flat[int(np.argmax(pr.reshape(-1)))] = True
            rows.append({"panel": p, "condition": "queue_onset" if cond == "on" else "queue_ongoing",
                         "iou": LH.iou(pred, tr[j], el[j])})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    out = {}
    for labels in ("h2", "h1"):
        for protocol in ("B", "A"):
            excl = protocol == "A"
            arm = {}
            for cond, rounds in (("on", RC.ON_ROUNDS), ("ong", RC.ONG_ROUNDS)):
                ms = {}
                for feat in ("v3", "v5"):
                    X, y, n = gather(cond, feat, exclude_june=excl, labels=labels)
                    ms[feat] = train(feat, X, y, rounds)
                    print(f"[{labels}-labels/{protocol}/{cond}/{feat}] {n:,} rows "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    del X, y
                    gc.collect()
                arm[cond] = {f: (V7.probs_pseudo(cond, f, ms[f], "feb") if protocol == "B"
                                 else V7.probs_rel(cond, f, ms[f])) for f in ("v3", "v5")}
                del ms
                gc.collect()
            for hyp in ("h1", "h2"):
                det = []
                for cond in ("on", "ong"):
                    w = W_ON if cond == "on" else W_ONG
                    pbl = V7.blend(arm[cond]["v3"], arm[cond]["v5"], w)
                    det.append(score(pbl, cond, protocol, hyp))
                d = pd.concat(det, ignore_index=True)
                a = LH.aggregate(d)
                out.setdefault(protocol, {}).setdefault(labels, {})[hyp] = {
                    "overall": a["overall"], "onset": a["onset"], "ongoing": a["ongoing"]}
                print(f"== {labels}-label models / {protocol} / {hyp} truth: overall "
                      f"{a['overall']:.4f} | onset {a['onset']:.4f} | ongoing {a['ongoing']:.4f}",
                      flush=True)
            del arm
            gc.collect()
    with open(os.path.join(LH.OUT, "v10_cv.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n=== label-regime table (same fixed rules) ===", flush=True)
    for proto in ("B", "A"):
        for labels in ("h2", "h1"):
            for hyp in ("h2", "h1"):
                r = out[proto][labels][hyp]
                print(f"{proto} {labels}-labels under {hyp}-truth: {r['overall']:.4f} "
                      f"(on {r['onset']:.4f}/ong {r['ongoing']:.4f})", flush=True)
    # decision: median3-label model vs current blend under H2 (they are the same
    # model regime, so the requested +0.01 bar cannot be met by construction)
    ok = False
    print(f"\nship_bar (H2-label model vs current blend under H2, >= +0.01 both protocols): "
          f"NOT APPLICABLE — the shipped v3 cache labels are already H2 (median-3), "
          f"verified bit-exact; PASS=False, SHIP=False", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
