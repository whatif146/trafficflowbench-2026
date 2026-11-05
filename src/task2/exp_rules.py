"""Honest rule experiments for Task 2: set-size calibration (top-k with a
predicted size), hybrid GBM+climatology ranking, and per-step rules.

Protocol:
  * probabilities come from a model trained on windows before 2031-01-01 only
    (the frozen v3 configuration), so the January tuning block and the February
    reporting block are both untouched by the model;
  * every rule is selected on January, then scored once on February with the
    official nesting;
  * the released-train-window proxy (optimistic, refit model) is reported as a
    secondary sanity check only.

    python exp_rules.py [--force-train]
"""
from __future__ import annotations

import gc
import json
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C
import features as FT
import train_gbm as TG
import gbm_predict as GP

import lightgbm as lgb

PRE_ON = os.path.join(C.MODEL2, "v3pre_onset.txt")
PRE_ONG = os.path.join(C.MODEL2, "v3pre_ongoing.txt")
PROB_PATH = os.path.join(C.CACHE2, "v3_pre_probs.npz")


def win_iou(pred6, truth6, elig6):
    p = pred6.astype(bool) & elig6
    t = truth6.astype(bool)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / float(union)


def train_pre(meta, on, ong, force=False):
    if not force and os.path.exists(PRE_ON) and os.path.exists(PRE_ONG):
        return lgb.Booster(model_file=PRE_ON), lgb.Booster(model_file=PRE_ONG)
    idx = [FT.FEATURES.index(n) for n in meta["features"]]
    params = dict(meta["params"])
    params["num_threads"] = 10
    models = []
    for items, rounds, path in ((on, int(meta["on_rounds"]), PRE_ON),
                                (ong, int(meta["ong_rounds"]), PRE_ONG)):
        Xs, ys = [], []
        for it in items:
            per = it["L"] if items is on else 6 * it["L"]
            m = it["date"] < it["tune_day"]
            rows = np.repeat(m, per)
            Xs.append(it["X"][rows][:, idx])
            ys.append(it["y"][rows])
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys)
        del Xs, ys
        ds = lgb.Dataset(X, label=y, feature_name=list(meta["features"]))
        ds.construct()
        del X, y
        gc.collect()
        print(f"training pre-Jan model on {ds.num_data()} rows -> {os.path.basename(path)}")
        model = lgb.train(params, ds, num_boost_round=rounds)
        model.save_model(path)
        del ds
        gc.collect()
        models.append(model)
    return models[0], models[1]


def main():
    meta = GP.load_meta("v3")
    on, ong = TG.load_dataset()
    for items in (on, ong):
        for it in items:
            it["is_tune"] = it["acc"] & (it["date"] >= it["tune_day"]) & (it["date"] < it["cut_day"])
            it["is_eval"] = it["acc"] & (it["date"] >= it["cut_day"])
    m_on, m_ong = train_pre(meta, on, ong)
    idx = [FT.FEATURES.index(n) for n in meta["features"]]

    # probability blocks per window, then free the feature matrices
    probs_on, probs_ong = [], []
    for items, model, key in ((on, m_on, "on"), (ong, m_ong, "ong")):
        per = (lambda L: L) if key == "on" else (lambda L: 6 * L)
        blocks = []
        for it in items:
            L = it["L"]
            p = model.predict(it["X"][:, idx]).astype(np.float32) if len(it["X"]) else np.zeros(0, np.float32)
            blocks.append(p)
        if key == "on":
            probs_on = blocks
        else:
            probs_ong = blocks
    for it in on + ong:
        it["X"] = None
    gc.collect()
    print("probabilities computed")

    # ---------------- rule definitions ---------------------------------------
    def onset_topk(p, L, k):
        pred = np.zeros((C.HORIZON, L), bool)
        pred[C.HORIZON - 1, np.argsort(-p)[:max(int(k), 0)]] = True
        return pred

    def onset_hybrid(p, L, panel, w, k):
        freq = ON_CLIM[panel]
        score = w * p + (1.0 - w) * freq
        pred = np.zeros((C.HORIZON, L), bool)
        pred[C.HORIZON - 1, np.argsort(-score)[:max(int(k), 0)]] = True
        return pred

    ON_CLIM, ONG_CLIM, RATE = {}, {}, {}
    for it in on:
        p = it["panel"]
        # climatology is stored on the panel dataset; rebuild per panel once
        pass
    # load the climatology arrays from the datasets (same for all windows/panels)
    for panel in sorted({it["panel"] for it in on}):
        z = np.load(os.path.join(C.CACHE2, f"ds_{panel}.npz"), allow_pickle=False)
        onf = z["onset_freq"].astype(np.float32)
        # normalise so it behaves like a probability ranking
        ON_CLIM[panel] = onf / max(float(onf.max()), 1e-6)
        RATE[panel] = z["rate"]
        z.close()
    for panel in sorted({it["panel"] for it in ong}):
        z = np.load(os.path.join(C.CACHE2, f"ds_{panel}.npz"), allow_pickle=False)
        og = z["ong_freq"].astype(np.float32)
        ONG_CLIM[panel] = og / max(float(og.max()), 1e-6)
        z.close()

    def eval_rule(kind, param, part, panel=None, which=None):
        """kind: th | top | alpha | hybrid  (onset) ; th | alpha (ongoing).

        ``which`` restricts the evaluation to "on" or "ong" rows.
        """
        detail = []
        # onset
        for i, it in enumerate(on if which != "ong" else []):
            L = it["L"]
            for r in range(len(it["date"])):
                if not it[part][r]:
                    continue
                if panel is not None and it["panel"] != panel:
                    continue
                p = probs_on[i][r * L:(r + 1) * L]
                truth, elig = it["truth"][r], it["elig"][r]
                key = it["panel"]
                if kind == "th":
                    pred = np.zeros((C.HORIZON, L), bool)
                    pred[C.HORIZON - 1, p >= param] = True
                elif kind == "top":
                    pred = onset_topk(p, L, param)
                elif kind == "alpha":
                    pred = onset_topk(p, L, round(param * float(p.sum())))
                elif kind == "hybrid":
                    ww, kk = param
                    pred = onset_hybrid(p, L, key, ww, kk)
                elif kind == "hybrid_alpha":
                    ww, aa = param
                    freq = ON_CLIM[key]
                    score = ww * p + (1 - ww) * freq
                    pred = onset_topk(score, L, round(aa * float(score.sum())))
                detail.append({"panel": key, "condition": "queue_onset",
                               "iou": win_iou(pred, truth, elig)})
        # ongoing
        for i, it in enumerate(ong if which != "on" else []):
            L = it["L"]
            for r in range(len(it["date"])):
                if not it[part][r]:
                    continue
                if panel is not None and it["panel"] != panel:
                    continue
                key = it["panel"]
                p = probs_ong[i][r * 6 * L:(r + 1) * 6 * L].reshape(C.HORIZON, L)
                truth, elig = it["truth"][r], it["elig"][r]
                if kind == "th":
                    pred = p >= param
                elif kind == "alpha":
                    aa = param
                    pred = np.zeros((C.HORIZON, L), bool)
                    for k in range(C.HORIZON):
                        kk = round(aa * float(p[k].sum()))
                        pred[k, np.argsort(-p[k])[:max(kk, 0)]] = True
                else:
                    raise ValueError(kind)
                detail.append({"panel": key, "condition": "queue_ongoing",
                               "iou": win_iou(pred, truth, elig)})
        d = pd.DataFrame(detail)
        if len(d) == 0:
            return np.nan, d
        return float(d.iou.mean()), d

    # ---------------- select rules on January --------------------------------
    print("\n== onset rule search (January) ==")
    on_cands = {}
    for th in (0.2, 0.25, 0.3, 0.35, 0.4):
        on_cands[f"th{th}"] = ("th", th)
    for k in (2, 3, 4, 5, 6, 7):
        on_cands[f"top{k}"] = ("top", k)
    for a in (0.6, 0.8, 1.0, 1.25, 1.5):
        on_cands[f"alpha{a}"] = ("alpha", a)
    for wv, k in ((0.5, 3), (0.5, 4), (0.5, 5), (0.7, 3), (0.7, 4), (0.7, 5), (0.3, 4)):
        on_cands[f"hyb{wv}_{k}"] = ("hybrid", (wv, k))
    for wv, a in ((0.5, 0.8), (0.5, 1.0), (0.7, 0.8), (0.7, 1.0)):
        on_cands[f"hyba{wv}_{a}"] = ("hybrid_alpha", (wv, a))
    on_tune = {}
    for name, (kind, param) in on_cands.items():
        v, _ = eval_rule(kind, param, "is_tune", which="on")
        on_tune[name] = v
    on_best = max(on_tune, key=on_tune.get)
    print(f"  best January: {on_best} -> {on_tune[on_best]:.4f}")
    for name, v in sorted(on_tune.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {name}: {v:.4f}")

    print("\n== ongoing rule search (January) ==")
    ong_cands = {f"th{th}": ("th", th) for th in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4)}
    ong_cands.update({f"alpha{a}": ("alpha", a) for a in (0.8, 1.0, 1.2, 1.5, 2.0)})
    ong_tune = {}
    for name, (kind, param) in ong_cands.items():
        v, _ = eval_rule(kind, param, "is_tune", which="ong")
        ong_tune[name] = v
    ong_best = max(ong_tune, key=ong_tune.get)
    print(f"  best January: {ong_best} -> {ong_tune[ong_best]:.4f}")
    for name, v in sorted(ong_tune.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {name}: {v:.4f}")

    # ---------------- February: baseline vs selected -------------------------
    def feb_report(label, on_sel, ong_sel):
        kind_on, param_on = on_cands[on_sel]
        kind_ong, param_ong = ong_cands[ong_sel]
        _, d1 = eval_rule(kind_on, param_on, "is_eval", which="on")
        s1 = d1.assign(condition="queue_onset")
        _, d2 = eval_rule(kind_ong, param_ong, "is_eval", which="ong")
        s2 = d2.assign(condition="queue_ongoing")
        detail = pd.concat([s1, s2], ignore_index=True)
        agg = C.aggregate_iou(detail)
        print(f"{label}: onset rule {on_sel} / ongoing {ong_sel}")
        print(f"  Feb overall {agg['overall']:.4f} | onset {agg['onset']:.4f} | ongoing {agg['ongoing']:.4f}")
        return agg, detail

    base_on = "th0.3" if "th0.3" in on_cands else on_best
    base_ong = "th0.25"
    agg_base, _ = feb_report("BASELINE v3 rules", base_on, base_ong)
    agg_sel, det_sel = feb_report("SELECTED on Jan", on_best, ong_best)
    # also report the best-possible Feb rule (diagnostic only, not a candidate)
    on_feb = {name: eval_rule(kind, param, "is_eval", which="on")[0] for name, (kind, param) in on_cands.items()}
    ong_feb = {name: eval_rule(kind, param, "is_eval", which="ong")[0] for name, (kind, param) in ong_cands.items()}
    print(f"  (diagnostic) best Feb onset rule {max(on_feb, key=on_feb.get)} "
          f"{on_feb[max(on_feb, key=on_feb.get)]:.4f}; best ongoing "
          f"{max(ong_feb, key=ong_feb.get)} {ong_feb[max(ong_feb, key=ong_feb.get)]:.4f}")

    # per-panel Feb comparison baseline vs selected
    print("\nper-panel Feb (baseline -> selected):")
    for panel in sorted({it["panel"] for it in on}):
        b, _ = eval_rule(on_cands[base_on][0], on_cands[base_on][1], "is_eval", panel=panel, which="on")
        s, _ = eval_rule(on_cands[on_best][0], on_cands[on_best][1], "is_eval", panel=panel, which="on")
        bo, _ = eval_rule(ong_cands[base_ong][0], ong_cands[base_ong][1], "is_eval", panel=panel, which="ong")
        so, _ = eval_rule(ong_cands[ong_best][0], ong_cands[ong_best][1], "is_eval", panel=panel, which="ong")
        print(f"  {panel}: onset {b:.3f}->{s:.3f} | ongoing {bo:.3f}->{so:.3f}")

    out = {"onset_selected": on_best, "ongoing_selected": ong_best,
           "onset_rules": on_cands, "ongoing_rules": ong_cands,
           "on_tune": on_tune, "ong_tune": ong_tune,
           "feb_baseline": {"overall": agg_base["overall"], "onset": agg_base["onset"],
                            "ongoing": agg_base["ongoing"]},
           "feb_selected": {"overall": agg_sel["overall"], "onset": agg_sel["onset"],
                            "ongoing": agg_sel["ongoing"],
                            "per_panel": {k: float(v) for k, v in agg_sel["per_panel"].items()}}}
    det_sel.to_csv(os.path.join(C.OUT2, "exp_rules_feb.csv"), index=False)
    with open(os.path.join(C.OUT2, "exp_rules.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\nsaved out/exp_rules.json")


if __name__ == "__main__":
    main()
