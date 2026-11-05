"""Final model selection: hyper-parameter + decision-rule tuning on the January
block, a single reporting pass on February, then a refit on all train windows.

Selection uses only the January block, so the February numbers stay honest.
Reports the official nesting (window -> panel x condition -> panel -> family).

Outputs: models/gbm_onset.txt, models/gbm_ongoing.txt, out/gbm_meta.json,
         out/gbm_cv.csv (February detail)
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

import lightgbm as lgb

BASE = dict(objective="binary", learning_rate=0.06, num_leaves=63,
            min_data_in_leaf=80, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, num_threads=10, verbose=-1, seed=42,
            deterministic=True, force_row_wise=True)

CONFIGS = {
    "base": dict(params=dict(BASE), on_rounds=350, ong_rounds=500),
    "big": dict(params=dict(BASE, num_leaves=127, min_data_in_leaf=40,
                            learning_rate=0.05), on_rounds=450, ong_rounds=700),
    "small": dict(params=dict(BASE, num_leaves=31, learning_rate=0.08),
                  on_rounds=300, ong_rounds=400),
}

ONSET_RULES = [("th", round(th, 2)) for th in np.arange(0.10, 0.86, 0.05)] + \
              [("top", k) for k in range(1, 13)]
ONG_THS = [round(t, 2) for t in np.arange(0.10, 0.81, 0.05)]


def win_iou(pred6, truth6, elig6):
    p = pred6.astype(bool) & elig6
    t = truth6.astype(bool)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / float(union)


def onset_pred(p, L, rule):
    kind, val = rule
    pred = np.zeros((C.HORIZON, L), bool)
    if kind == "th":
        pred[C.HORIZON - 1, p >= val] = True
    else:
        pred[C.HORIZON - 1, np.argsort(-p)[:min(int(val), L)]] = True
    return pred


def gather_train(items, key):
    Xs, ys = [], []
    for it in items:
        per = it["L"] if key == "on" else 6 * it["L"]
        m = it["date"] < it["tune_day"]
        rows = np.repeat(m, per)
        Xs.append(it["X"][rows])
        ys.append(it["y"][rows])
    return np.concatenate(Xs, axis=0), np.concatenate(ys)


def probs_by_window(items, model, key):
    """Return list (per item) of per-window probability blocks."""
    out = []
    for it in items:
        if len(it["X"]) == 0:
            out.append(np.zeros(0, np.float32))
            continue
        out.append(model.predict(it["X"]).astype(np.float32))
    return out


def tune_scores_onset(items, probs):
    per_rule = {r: [] for r in ONSET_RULES}
    for i, it in enumerate(items):
        L = it["L"]
        for r in range(len(it["date"])):
            if not it["is_tune"][r]:
                continue
            p = probs[i][r * L:(r + 1) * L]
            truth, elig = it["truth"][r], it["elig"][r]
            for rule in ONSET_RULES:
                per_rule[rule].append(win_iou(onset_pred(p, L, rule), truth, elig))
    means = {rule: float(np.mean(v)) for rule, v in per_rule.items() if v}
    best = max(means, key=means.get)
    return best, means[best]


def tune_scores_ongoing(items, probs):
    per_th = {th: [] for th in ONG_THS}
    for i, it in enumerate(items):
        L = it["L"]
        for r in range(len(it["date"])):
            if not it["is_tune"][r]:
                continue
            p = probs[i][r * 6 * L:(r + 1) * 6 * L].reshape(C.HORIZON, L)
            truth, elig = it["truth"][r], it["elig"][r]
            for th in ONG_THS:
                per_th[th].append(win_iou(p >= th, truth, elig))
    means = {th: float(np.mean(v)) for th, v in per_th.items() if v}
    best = max(means, key=means.get)
    return best, means[best]


def feb_detail(on, ong, probs_on, probs_ong, on_rule, ong_th):
    rows = []
    for i, it in enumerate(on):
        L = it["L"]
        ev = it["acc"] & (it["date"] >= it["cut_day"])
        for r in range(len(it["date"])):
            if not ev[r]:
                continue
            p = probs_on[i][r * L:(r + 1) * L]
            rows.append({"panel": it["panel"], "condition": "queue_onset",
                         "iou": win_iou(onset_pred(p, L, on_rule),
                                        it["truth"][r], it["elig"][r])})
    for i, it in enumerate(ong):
        L = it["L"]
        ev = it["acc"] & (it["date"] >= it["cut_day"])
        for r in range(len(it["date"])):
            if not ev[r]:
                continue
            p = probs_ong[i][r * 6 * L:(r + 1) * 6 * L].reshape(C.HORIZON, L)
            rows.append({"panel": it["panel"], "condition": "queue_ongoing",
                         "iou": win_iou(p >= ong_th, it["truth"][r], it["elig"][r])})
    return pd.DataFrame(rows)


def main():
    C.ensure_dirs()
    on, ong = TG.load_dataset()
    for it in on:
        it["kind"] = "on"
    for it in ong:
        it["kind"] = "ong"
        it["is_tune"] = it["acc"] & (it["date"] >= it["tune_day"]) & (it["date"] < it["cut_day"])
    for it in on:
        it["is_tune"] = it["acc"] & (it["date"] >= it["tune_day"]) & (it["date"] < it["cut_day"])
    print(f"onset windows {sum(len(it['date']) for it in on)} | "
          f"ongoing windows {sum(len(it['date']) for it in ong)}")

    X_on_t, y_on_t = gather_train(on, "on")
    ds_on = lgb.Dataset(X_on_t, label=y_on_t, feature_name=list(FT.FEATURES),
                        free_raw_data=True)
    ds_on.construct()
    del X_on_t, y_on_t
    gc.collect()
    X_ong_t, y_ong_t = gather_train(ong, "ong")
    ds_ong = lgb.Dataset(X_ong_t, label=y_ong_t, feature_name=list(FT.FEATURES),
                         free_raw_data=True)
    ds_ong.construct()
    del X_ong_t, y_ong_t
    gc.collect()
    print("training rows:", ds_on.num_data(), ds_ong.num_data())

    results, models = {}, {}
    for name, cfg in CONFIGS.items():
        print(f"\n### config {name}: leaves={cfg['params']['num_leaves']} "
              f"lr={cfg['params']['learning_rate']}")
        m_on = lgb.train(cfg["params"], ds_on, num_boost_round=cfg["on_rounds"])
        m_ong = lgb.train(cfg["params"], ds_ong, num_boost_round=cfg["ong_rounds"])
        probs_on = probs_by_window(on, m_on, "on")
        probs_ong = probs_by_window(ong, m_ong, "ong")
        on_rule, on_tune = tune_scores_onset(on, probs_on)
        ong_th, ong_tune = tune_scores_ongoing(ong, probs_ong)
        print(f"  tune: onset {on_rule} -> {on_tune:.4f} | ongoing th {ong_th} -> {ong_tune:.4f}")
        results[name] = {"on_rule": on_rule, "on_tune": on_tune, "ong_th": ong_th,
                         "ong_tune": ong_tune,
                         "combined": 0.5 * (on_tune + ong_tune)}
        models[name] = (m_on, m_ong)
        gc.collect()

    best = max(results, key=lambda k: results[k]["combined"])
    print(f"\nselected config: {best} (tune {results[best]['combined']:.4f})")
    on_rule = results[best]["on_rule"]
    ong_th = results[best]["ong_th"]
    m_on, m_ong = models[best]
    probs_on = probs_by_window(on, m_on, "on")
    probs_ong = probs_by_window(ong, m_ong, "ong")
    detail = feb_detail(on, ong, probs_on, probs_ong, on_rule, ong_th)
    detail.to_csv(os.path.join(C.OUT2, "gbm_cv.csv"), index=False)
    agg = C.aggregate_iou(detail)
    print("\n== FINAL held-out (Feb) pseudo-CV ==")
    print(f"overall {agg['overall']:.4f} | onset {agg['onset']:.4f} | ongoing {agg['ongoing']:.4f}")
    print("per panel:", {k: round(v, 3) for k, v in agg["per_panel"].items()})
    print("per family:", {k: round(v, 3) for k, v in agg["per_family"].items()})

    meta = {"config": best,
            "params": CONFIGS[best]["params"],
            "on_rounds": CONFIGS[best]["on_rounds"],
            "ong_rounds": CONFIGS[best]["ong_rounds"],
            "onset_rule": [on_rule[0], float(on_rule[1])],
            "ongoing_th": float(ong_th),
            "features": list(FT.FEATURES),
            "selection": results,
            "cv_overall": float(agg["overall"]), "cv_onset": float(agg["onset"]),
            "cv_ongoing": float(agg["ongoing"]),
            "cv_per_panel": {k: float(v) for k, v in agg["per_panel"].items()},
            "cv_per_family": {k: float(v) for k, v in agg["per_family"].items()}}
    with open(os.path.join(C.OUT2, "gbm_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    # ---- refit on all windows, incrementally per panel to stay memory-light
    print("refitting on all train windows (incremental per panel) ...")
    for name, items, key, rounds in (("gbm_onset", on, "on", CONFIGS[best]["on_rounds"]),
                                     ("gbm_ongoing", ong, "ong", CONFIGS[best]["ong_rounds"])):
        model = None
        per_panel = max(1, int(rounds / max(len(items), 1)))
        for it in items:
            per = it["L"] if key == "on" else 6 * it["L"]
            ds = lgb.Dataset(it["X"], label=it["y"], feature_name=list(FT.FEATURES),
                             free_raw_data=False, params=CONFIGS[best]["params"])
            ds.construct()
            model = lgb.train(CONFIGS[best]["params"], ds,
                              num_boost_round=max(per_panel, 20),
                              init_model=model)
            del ds
            gc.collect()
        model.save_model(os.path.join(C.MODEL2, f"{name}.txt"))
        print(f"saved {name}.txt")


if __name__ == "__main__":
    main()
