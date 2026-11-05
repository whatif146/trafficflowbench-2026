"""v9: soft Bayesian labels P(latent queued | observed speed) vs the current
noisy-hard-label models, on two hard proxy truths.

Arms: models trained on noisy hard labels (current regime) vs models trained on
soft labels y = Phi((v_cut - v_obs)/sigma_v) * eligible.
Proxies: (i) noisy hard truth = released 3-step-median labels (current regime);
         (ii) mild denoised hard truth = sigma=1 temporal Gaussian labels.
Fixed LB-verified rules for all arms: onset p=0.3*v3+0.7*v5, th 0.30 + top-1
fallback; ongoing p=0.4*v3+0.6*v5, th 0.20 (+ window-level argmax fallback).
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


def load_v9(panel: str):
    return np.load(os.path.join(LH.CACHE, f"v9_{panel}.npz"), allow_pickle=False)


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
    y = np.empty(total, np.float32)
    for p, keep, per, cnt, start in plan:
        z = RC.load_panel(p)
        Xf = EX.X_sel(cond, z, FC.COLS[feat])
        rows = np.repeat(keep, per)
        X[start:start + cnt] = Xf[rows]
        if labels == "soft":
            v9 = load_v9(p)
            yk = v9["on_y_soft"] if cond == "on" else v9["ong_y_soft"]
            y[start:start + cnt] = yk[rows]
            v9.close()
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


def truth(cond, protocol, panel, idx, proxy):
    """hard truth for one proxy: 'noisy' (released 3-step median) | 's1' (sigma=1)."""
    pre = "on" if cond == "on" else "ong"
    if protocol == "A":
        z = np.load(os.path.join(LH.CACHE, f"rel_{panel}.npz"), allow_pickle=False)
        el = z["elig"][idx]
        if proxy == "noisy":
            tr = z["truth"][idx]
        else:
            v9 = load_v9(panel)
            tr = v9["rel_truth_s1"][idx]
            v9.close()
        z.close()
        return tr, el
    z = RC.load_panel(panel)
    el = z[f"{pre}_elig"][idx]
    if proxy == "noisy":
        tr = z[f"{pre}_truth"][idx]
    else:
        v9 = load_v9(panel)
        tr = v9["on_truth_s1" if cond == "on" else "ong_truth_s1"][idx]
        v9.close()
    z.close()
    return tr, el


def score_arm(probs: dict, cond: str, protocol: str, proxy: str) -> dict:
    rows = []
    for p, d in probs.items():
        tr, el = truth(cond, protocol, p, d["idx"], proxy)
        L = d["L"]
        for j in range(len(d["idx"])):
            if cond == "on":
                pj = d["prob"][j]
                p_last = pj[LH.HORIZON - 1] if pj.ndim == 2 else pj
                pred = V7.onset_pred_blend(p_last, L)
            else:
                pr = d["prob"][j].reshape(LH.HORIZON, L)
                pred = pr >= ONG_TH
                if not pred.any():          # window-level fallback
                    pred.flat[int(np.argmax(pr.reshape(-1)))] = True
            rows.append({"panel": p, "condition": "queue_onset" if cond == "on" else "queue_ongoing",
                         "iou": LH.iou(pred, tr[j], el[j])})
    return pd.DataFrame(rows)


TH_GRID = [round(float(t), 2) for t in np.arange(0.15, 0.86, 0.05)]


def sweep_thresholds(probs: dict, cond: str, protocol: str, proxy: str) -> dict:
    """Mean IoU over the evaluation block for each decision threshold."""
    acc = {th: [] for th in TH_GRID}
    for p, d in probs.items():
        tr, el = truth(cond, protocol, p, d["idx"], proxy)
        L = d["L"]
        for j in range(len(d["idx"])):
            pj = d["prob"][j]
            if cond == "on":
                p_last = pj[LH.HORIZON - 1] if pj.ndim == 2 else pj
                for th in TH_GRID:
                    pred = np.zeros((LH.HORIZON, L), bool)
                    sel = p_last >= th
                    if not sel.any():
                        sel[int(np.argmax(p_last))] = True
                    pred[LH.HORIZON - 1] = sel
                    acc[th].append(LH.iou(pred, tr[j], el[j]))
            else:
                pr = pj.reshape(LH.HORIZON, L)
                for th in TH_GRID:
                    pred = pr >= th
                    if not pred.any():
                        pred.flat[int(np.argmax(pr.reshape(-1)))] = True
                    acc[th].append(LH.iou(pred, tr[j], el[j]))
    return {th: float(np.mean(v)) for th, v in acc.items() if v}



def main():
    t0 = time.time()
    out = {}
    for labels in ("noisy", "soft"):
        for protocol in ("B", "A"):
            excl = protocol == "A"
            arm = {}
            for cond, rounds in (("on", RC.ON_ROUNDS), ("ong", RC.ONG_ROUNDS)):
                ms = {}
                for feat in ("v3", "v5"):
                    X, y, n = gather(cond, feat, exclude_june=excl, labels=labels)
                    ms[feat] = train(feat, X, y, rounds)
                    print(f"[{labels}/{protocol}/{cond}/{feat}] {n:,} rows "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    del X, y
                    gc.collect()
                arm[cond] = {f: (V7.probs_pseudo(cond, f, ms[f], "feb") if protocol == "B"
                                 else V7.probs_rel(cond, f, ms[f])) for f in ("v3", "v5")}
                del ms
                gc.collect()
            for proxy in ("noisy", "s1"):
                det = []
                for cond in ("on", "ong"):
                    w = W_ON if cond == "on" else W_ONG
                    pbl = V7.blend(arm[cond]["v3"], arm[cond]["v5"], w)
                    det.append(score_arm(pbl, cond, protocol, proxy))
                d = pd.concat(det, ignore_index=True)
                a = LH.aggregate(d)
                out.setdefault(protocol, {}).setdefault(labels, {})[proxy] = {
                    "overall": a["overall"], "onset": a["onset"], "ongoing": a["ongoing"],
                    "per_panel": a["per_panel"]}
                print(f"== {labels:>5} labels / protocol {protocol} / {proxy:>5} proxy: "
                      f"overall {a['overall']:.4f} | onset {a['onset']:.4f} | "
                      f"ongoing {a['ongoing']:.4f}", flush=True)
            if protocol == "B":          # threshold sweeps on the Feb block only
                out.setdefault("sweep", {}).setdefault(labels, {})
                for proxy in ("noisy", "s1"):
                    sw = {}
                    for cond in ("on", "ong"):
                        w = W_ON if cond == "on" else W_ONG
                        pbl = V7.blend(arm[cond]["v3"], arm[cond]["v5"], w)
                        sw[cond] = sweep_thresholds(pbl, cond, protocol, proxy)
                    out["sweep"][labels][proxy] = sw
                best = {}
                for cond in ("on", "ong"):
                    s = out["sweep"][labels]["s1"][cond]
                    b_th = max(s, key=s.get)
                    best[cond] = {"th": b_th, "iou": s[b_th]}
                out["sweep"][labels]["best_s1"] = best
                print(f"   [{labels}] best-s1 thresholds: {best}", flush=True)
            del arm
            gc.collect()
    # decision + threshold recommendation (tuned on the mild proxy only)
    out["decision"] = {}
    for proto in ("B", "A"):
        cur_n = out[proto]["noisy"]["noisy"]["overall"]
        v9_n = out[proto]["soft"]["noisy"]["overall"]
        cur_s = out[proto]["noisy"]["s1"]["overall"]
        v9_s = out[proto]["soft"]["s1"]["overall"]
        ok = (v9_n >= cur_n - 0.005) and (v9_s > cur_s)
        out["decision"][proto] = {"cur_noisy_proxy": cur_n, "v9_noisy_proxy": v9_n,
                                  "cur_s1_proxy": cur_s, "v9_s1_proxy": v9_s,
                                  "delta_noisy": v9_n - cur_n, "delta_s1": v9_s - cur_s,
                                  "ok": bool(ok)}
        print(f"\n[{proto}] proxy noisy: current {cur_n:.4f} -> v9 {v9_n:.4f} ({v9_n - cur_n:+.4f}); "
              f"proxy s1: current {cur_s:.4f} -> v9 {v9_s:.4f} ({v9_s - cur_s:+.4f}) "
              f"-> {'ok' if ok else 'NOT OK'}", flush=True)
    with open(os.path.join(LH.OUT, "v9_cv.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
