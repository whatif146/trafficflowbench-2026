"""Multi-month pseudo-validation simulation of the Task-2 decision rules.

For each held-out block (Nov 2030, Dec 2030, Jan 2031, Feb 2031) the v3 and
long-history (v5) onset/ongoing models are refit on every selector-valid pseudo
window with origin before the block, then the whole decision-rule grid is scored
on the block's windows. This emulates the real validation month as closely as the
released train data allows, and gives a distribution of expected gains for the
candidate queue layers (and the winner per block), so tomorrow's LB probes are
chosen with eyes open.

Outputs:
  work/task2_seq/out/sim_multimonth_rules.csv   (block, cond, w, th, fb, iou)
  work/task2_seq/out/sim_multimonth_candidates.csv (candidate, per-block delta IoU)

Usage: python work/task2_seq/sim_multimonth.py [blocks...]
"""
from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC

BLOCKS = [
    ("2030-11", "2030-11-01", "2030-12-01"),
    ("2030-12", "2030-12-01", "2031-01-01"),
    ("2031-01", "2031-01-01", "2031-02-01"),
    ("2031-02", "2031-02-01", "2031-03-01"),
]
ON_THS = [0.20, 0.25, 0.30, 0.35, 0.40]
ON_WS = [0.3, 0.4, 0.5, 0.6]
ON_FBS = [0, 1]
ONG_THS = [0.15, 0.20, 0.25, 0.30]
ONG_WS = [0.2, 0.3, 0.4, 0.5, 0.6]

# planned LB candidates: name -> (on_w, on_th, on_fb, ong_w, ong_th)
CANDIDATES = {
    "P4": (0.5, 0.30, 0, 0.6, 0.25),
    "q_w30ong40": (0.3, 0.30, 0, 0.4, 0.20),
    "q_w30": (0.3, 0.30, 0, 0.6, 0.25),
    "q_on025": (0.5, 0.25, 0, 0.6, 0.25),
    "q_onfb1": (0.5, 0.30, 1, 0.6, 0.25),
    "q_w30ong40_on025fb1": (0.3, 0.25, 1, 0.4, 0.20),
    "q_w30_on025fb1": (0.3, 0.25, 1, 0.6, 0.25),
}


def gather(cond, feat, cutoff_epoch, stride):
    Xs, ys = [], []
    for p in LH.PANELS:
        z = np.load(os.path.join(LH.CACHE, f"tr_{p}.npz"), allow_pickle=False)
        pre = "on" if cond == "on" else "ong"
        L = int(z["n_links"])
        t = z[f"{pre}_time"]
        per = L if cond == "on" else LH.HORIZON * L
        keep = t < cutoff_epoch
        if stride > 1:
            keep &= (np.arange(len(t)) % stride == 0)
        rows = np.repeat(keep, per)
        if rows.any():
            X = RC.panel_X(cond, feat, z)
            Xs.append(X[rows])
            ys.append(z[f"{pre}_y"][rows])
            del X
        z.close()
        gc.collect()
    return np.concatenate(Xs, axis=0), np.concatenate(ys)


def eval_probs(cond, feat, model, lo, hi):
    """window -> probability matrix for windows with lo <= origin < hi."""
    out = {}
    for p in LH.PANELS:
        z = np.load(os.path.join(LH.CACHE, f"tr_{p}.npz"), allow_pickle=False)
        pre = "on" if cond == "on" else "ong"
        L = int(z["n_links"])
        t = z[f"{pre}_time"]
        per = L if cond == "on" else LH.HORIZON * L
        sel = np.where((t >= lo) & (t < hi))[0]
        if len(sel):
            X = RC.panel_X(cond, feat, z)
            rows = (sel[:, None] * per + np.arange(per)[None, :]).reshape(-1)
            prob = model.predict(np.ascontiguousarray(X[rows])).astype(np.float32)
            prob = prob.reshape(len(sel), LH.HORIZON, L) if cond == "ong" else prob.reshape(len(sel), L)
            truth = z[f"{pre}_truth"][sel]
            elig = z[f"{pre}_elig"][sel]
            out[p] = (prob, truth, elig, L)
            del X
        z.close()
        gc.collect()
    return out


def eval_onset(probmap, w, th, fb):
    vals = []
    for p, (pr, truth, elig, L) in probmap.items():
        for j in range(pr.shape[0]):
            pred = np.zeros((LH.HORIZON, L), bool)
            pp = pr[j]
            pred[-1] = pp >= th
            if fb and not pred[-1].any():
                pred[-1, int(np.argmax(pp))] = True
            vals.append(LH.iou(pred, truth[j], elig[j]))
    return float(np.mean(vals))


def eval_ong(probmap, w, th, w3=1.0):
    vals = []
    for p, (pr, truth, elig, L) in probmap.items():
        for j in range(pr.shape[0]):
            vals.append(LH.iou(pr[j] >= th, truth[j], elig[j]))
    return float(np.mean(vals))


def main():
    blocks = BLOCKS
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        blocks = [b for b in BLOCKS if b[0] in wanted]
    rows = []
    cand_rows = []
    for name, lo_s, hi_s in blocks:
        lo = int(pd.Timestamp(lo_s, tz="UTC").timestamp())
        hi = int(pd.Timestamp(hi_s, tz="UTC").timestamp())
        print(f"=== block {name}", flush=True)
        models = {}
        for cond, rounds in (("on", RC.ON_ROUNDS), ("ong", RC.ONG_ROUNDS)):
            for feat in ("v3", "v5"):
                X, y = gather(cond, feat, lo, 2 if cond == "ong" else 1)
                m = RC.train_model(cond, feat, X, y, rounds)
                del X, y
                gc.collect()
                p3 = m if feat == "v3" else models.get(f"{cond}_v3")
                models[f"{cond}_{feat}"] = m
                print(f"  fitted {cond} {feat}", flush=True)
        for cond, ths, ws in (("on", ON_THS, ON_WS), ("ong", ONG_THS, ONG_WS)):
            p3 = eval_probs(cond, "v3", models[f"{cond}_v3"], lo, hi)
            p5 = eval_probs(cond, "v5", models[f"{cond}_v5"], lo, hi)
            merged = {}
            for p in p3:
                merged[p] = (p3[p][0], p5[p][0], p3[p][1], p3[p][2], p3[p][3])
            if cond == "on":
                for w in ws:
                    for th in ths:
                        for fb in ON_FBS:
                            vals = []
                            for p, (q3, q5, truth, elig, L) in merged.items():
                                for j in range(q3.shape[0]):
                                    prob = w * q3[j] + (1 - w) * q5[j]
                                    pred = np.zeros((LH.HORIZON, L), bool)
                                    pred[-1] = prob >= th
                                    if fb and not pred[-1].any():
                                        pred[-1, int(np.argmax(prob))] = True
                                    vals.append(LH.iou(pred, truth[j], elig[j]))
                            rows.append({"block": name, "cond": cond, "w": w, "th": th,
                                         "fb": fb, "iou": float(np.mean(vals))})
            else:
                for w in ws:
                    for th in ths:
                        vals = []
                        for p, (q3, q5, truth, elig, L) in merged.items():
                            for j in range(q3.shape[0]):
                                prob = w * q3[j] + (1 - w) * q5[j]
                                vals.append(LH.iou(prob >= th, truth[j], elig[j]))
                        rows.append({"block": name, "cond": cond, "w": w, "th": th,
                                     "fb": 0, "iou": float(np.mean(vals))})
            del merged, p3, p5
            gc.collect()
        del models
        gc.collect()
        df = pd.DataFrame([r for r in rows if r["block"] == name])
        print(df.sort_values(["cond", "iou"], ascending=[True, False]).head(8).to_string(index=False), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(LH.OUT, "sim_multimonth_rules.csv"), index=False)
    # candidate summary
    base = out[(out.cond == "on") & (out.w == 0.5) & (out.th == 0.30) & (out.fb == 0)]
    for cname, (onw, onth, onfb, ongw, ongth) in CANDIDATES.items():
        on = out[(out.cond == "on") & (out.w == onw) & (out.th == onth) & (out.fb == onfb)]
        ong = out[(out.cond == "ong") & (out.w == ongw) & (out.th == ongth)]
        m = on.merge(ong, on="block", suffixes=("_on", "_ong"))
        for _, r in m.iterrows():
            d_on = r["iou_on"] - float(base[base.block == r.block]["iou"].iloc[0])
            d_total = 0.30 * 0.5 * (d_on + (r["iou_ong"] - 0.0))
            cand_rows.append({"candidate": cname, "block": r["block"],
                              "d_on": d_on, "d_ong": r["iou_ong"], "d_total_est": d_total})
    cand = pd.DataFrame(cand_rows)
    cand.to_csv(os.path.join(LH.OUT, "sim_multimonth_candidates.csv"), index=False)
    print(cand.groupby("candidate")[["d_on", "d_total_est"]].agg(["mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
