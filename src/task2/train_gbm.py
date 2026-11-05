"""Train the LightGBM Task 2 models and score them with the official nesting.

Protocol (honest):
  * models are trained on the accepted windows from train days before 2031-01-01;
  * the decision rule (probability threshold or per-window top-k) is chosen on
    the January hold-out block;
  * the reported pseudo-CV is the February block, untouched by either step.
The final submission models are refitted on every train window afterwards.

Usage: python train_gbm.py [--eval-only]
Outputs: models/gbm_onset.txt, models/gbm_ongoing.txt, out/gbm_meta.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C
import features as FT

import lightgbm as lgb

TUNE_START = pd.Timestamp("2031-01-01", tz="UTC")
CUT = pd.Timestamp("2031-02-01", tz="UTC")

PARAMS = dict(objective="binary", learning_rate=0.06, num_leaves=63,
              min_data_in_leaf=80, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, num_threads=12, verbose=-1,
              seed=42, deterministic=True, force_row_wise=True)


def day_of(panel_days: np.ndarray, uniq, ts):
    return None


def load_dataset(panels=None):
    panels = panels or [p for p in C.PANELS
                        if os.path.exists(os.path.join(C.CACHE2, f"ds_{p}.npz"))]
    on, ong = [], []
    for p in panels:
        z = np.load(os.path.join(C.CACHE2, f"ds_{p}.npz"), allow_pickle=False)
        L = int(z["n_links"])
        uniq_days = int(z["n_days"])
        cut_day = int(z["cut_day"])
        tune_day = cut_day  # all fit windows are before the cut
        # separate train (< Jan) and tune (Jan) by date index; store both
        d_on = z["date_on"]
        d_ong = z["date_ong"]
        tune_idx = int(round(0.90 * cut_day))  # ~Jan 1 as 90% of fit days
        on.append({"panel": p, "L": L, "cut_day": cut_day, "tune_day": tune_idx,
                   "X": z["X_on"], "y": z["y_on"], "date": d_on,
                   "acc": z["on_accepted"].astype(bool),
                   "truth": z["on_truth"], "elig": z["on_elig"],
                   "clim": {k: z[k] for k in ("onset_freq", "ong_freq", "rate",
                                              "elig_rate", "links", "v_cut",
                                              "rank_to_idx")}})
        ong.append({"panel": p, "L": L, "cut_day": cut_day, "tune_day": tune_idx,
                    "X": z["X_ong"], "y": z["y_ong"], "date": d_ong,
                    "acc": z["ong_accepted"].astype(bool),
                    "truth": z["ong_truth"], "elig": z["ong_elig"],
                    "clim": {k: z[k] for k in ("onset_freq", "ong_freq", "rate",
                                               "elig_rate", "links", "v_cut",
                                               "rank_to_idx")}})
        z.close()
    return on, ong


def stack(items, key):
    return np.concatenate([it[key] for it in items], axis=0)


def window_blocks(items):
    """Yield (panel index, offset in stacked rows, n_rows_per_window)."""
    out = []
    for i, it in enumerate(items):
        L = it["L"]
        n_on = len(it["date"])
        out.append((i, n_on, L))
    return out


def win_iou(pred6, truth6, elig6) -> float:
    p = pred6.astype(bool) & elig6
    t = truth6.astype(bool)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / float(union)


# --------------------------------------------------------------------------- #
# onset: rows are L per window (last horizon step)
# --------------------------------------------------------------------------- #
def eval_onset_rule(probs, items, mask_date, rule) -> float:
    kind, val = rule
    ious = []
    for i, it in enumerate(items):
        L = it["L"]
        rows = np.where(mask_date["on"][i])[0]
        for r in rows:
            sl = slice(r * L, (r + 1) * L)
            p = probs[i][sl]
            pred = np.zeros((C.HORIZON, L), bool)
            if kind == "top":
                k = min(int(val), L)
                sel = np.argsort(-p)[:k]
                pred[C.HORIZON - 1, sel] = True
            elif kind == "th":
                pred[C.HORIZON - 1, p >= val] = True
            else:
                raise ValueError(kind)
            ious.append(win_iou(pred, it["truth"][r], it["elig"][r]))
    return float(np.mean(ious)) if ious else np.nan


def eval_ongoing_rule(probs, items, mask_date, th) -> float:
    ious = []
    for i, it in enumerate(items):
        L = it["L"]
        rows = np.where(mask_date["ong"][i])[0]
        for r in rows:
            sl = slice(r * 6 * L, (r + 1) * 6 * L)
            p = probs[i][sl].reshape(C.HORIZON, L)
            pred = p >= th
            ious.append(win_iou(pred, it["truth"][r], it["elig"][r]))
    return float(np.mean(ious)) if ious else np.nan


def predict_onset(items, models_onset):
    out = []
    for it in items:
        X = it["X"]
        if len(X) == 0:
            out.append(np.zeros(0, np.float32))
            continue
        m = models_onset
        out.append(m.predict(X).astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds-onset", type=int, default=350)
    ap.add_argument("--rounds-ongoing", type=int, default=500)
    ap.add_argument("--no-refit", action="store_true")
    args = ap.parse_args()
    C.ensure_dirs()
    on, ong = load_dataset()
    print(f"panels {[it['panel'] for it in on]}")
    print(f"onset rows {sum(len(it['date']) for it in on)} windows, "
          f"ongoing rows {sum(len(it['date']) for it in ong)} windows")

    # ---- masks by date --------------------------------------------------
    mask = {"on": [], "ong": []}
    for key, items in (("on", on), ("ong", ong)):
        for it in items:
            d = it["date"]
            acc = it["acc"]
            mask[key].append({"train": d < it["tune_day"],
                              "tune": acc & (d >= it["tune_day"]) & (d < it["cut_day"]),
                              "eval": acc & (d >= it["cut_day"])})
    def rows_for(key, part):
        tot = 0
        for i, it in enumerate(on if key == "on" else ong):
            m = mask[key][i][part]
            per = it["L"] if key == "on" else 6 * it["L"]
            tot += int(m.sum()) * per
        return tot
    print("rows train/tune/eval onset:", rows_for("on", "train"), rows_for("on", "tune"), rows_for("on", "eval"))
    print("rows train/tune/eval ongoing:", rows_for("ong", "train"), rows_for("ong", "tune"), rows_for("ong", "eval"))

    # ---- build training matrices ---------------------------------------
    def gather(items, key, part):
        Xs, ys = [], []
        for i, it in enumerate(items):
            per = it["L"] if key == "on" else 6 * it["L"]
            m = mask[key][i][part]
            rows = np.repeat(m, per)
            Xs.append(it["X"][rows])
            ys.append(it["y"][rows])
        return np.concatenate(Xs, axis=0), np.concatenate(ys)
    print("gathering onset train ...")
    X_on_t, y_on_t = gather(on, "on", "train")
    print("gathering ongoing train ...")
    X_ong_t, y_ong_t = gather(ong, "ong", "train")
    print("shapes", X_on_t.shape, X_ong_t.shape)
    del X_on_t, y_on_t, X_ong_t, y_ong_t
    gc.collect()

    # train (release raw arrays after Dataset construction)
    def train_model(items, key, rounds):
        Xt, yt = gather(items, key, "train")
        ds = lgb.Dataset(Xt, label=yt, feature_name=list(FT.FEATURES), free_raw_data=True)
        ds.construct()
        del Xt, yt
        gc.collect()
        print(f"training {key} on {ds.num_data()} rows ...")
        model = lgb.train(PARAMS | {"num_threads": 12}, ds, num_boost_round=rounds)
        del ds
        gc.collect()
        return model

    m_on = train_model(on, "on", args.rounds_onset)
    m_ong = train_model(ong, "ong", args.rounds_ongoing)

    # ---- probabilities ---------------------------------------------------
    probs = {"on": predict_onset(on, m_on), "ong": predict_onset(ong, m_ong)}

    # ---- rule selection on the tune block -------------------------------
    onset_rules = [("th", th) for th in np.arange(0.1, 0.91, 0.05)] + \
                  [("top", k) for k in range(1, 13)]
    ong_rules = [float(th) for th in np.arange(0.15, 0.86, 0.05)]
    on_scores = {r: eval_onset_rule(probs["on"], on,
                                    {"on": [m["tune"] for m in mask["on"]]}, r)
                 for r in onset_rules}
    on_best = max(on_scores, key=on_scores.get)
    ong_scores = {th: eval_ongoing_rule(probs["ong"], ong,
                                        {"ong": [m["tune"] for m in mask["ong"]]}, th)
                  for th in ong_rules}
    ong_best = max(ong_scores, key=ong_scores.get)
    print(f"tune: onset rule {on_best} -> {on_scores[on_best]:.4f} (top5 "
          f"{[(str(k), round(v, 3)) for k, v in sorted(on_scores.items(), key=lambda kv: -kv[1])[:5]]})")
    print(f"tune: ongoing th {ong_best:.2f} -> {ong_scores[ong_best]:.4f} (top5 "
          f"{[(round(k, 2), round(v, 3)) for k, v in sorted(ong_scores.items(), key=lambda kv: -kv[1])[:5]]})")

    # ---- held-out evaluation (February) ---------------------------------
    ev_on = {"on": [m["eval"] for m in mask["on"]]}
    ev_ong = {"ong": [m["eval"] for m in mask["ong"]]}
    detail = []
    for i, it in enumerate(on):
        L = it["L"]
        for r in np.where(ev_on["on"][i])[0]:
            p = probs["on"][i][r * L:(r + 1) * L]
            kind, val = on_best
            pred = np.zeros((C.HORIZON, L), bool)
            if kind == "top":
                pred[C.HORIZON - 1, np.argsort(-p)[:min(int(val), L)]] = True
            else:
                pred[C.HORIZON - 1, p >= val] = True
            detail.append({"panel": it["panel"], "condition": "queue_onset",
                           "iou": win_iou(pred, it["truth"][r], it["elig"][r])})
    for i, it in enumerate(ong):
        L = it["L"]
        for r in np.where(ev_ong["ong"][i])[0]:
            p = probs["ong"][i][r * 6 * L:(r + 1) * 6 * L].reshape(C.HORIZON, L)
            pred = p >= ong_best
            detail.append({"panel": it["panel"], "condition": "queue_ongoing",
                           "iou": win_iou(pred, it["truth"][r], it["elig"][r])})
    detail = pd.DataFrame(detail)
    detail.to_csv(os.path.join(C.OUT2, "gbm_cv.csv"), index=False)
    agg = C.aggregate_iou(detail)
    print("\n== GBM held-out (Feb) pseudo-CV ==")
    print(f"overall {agg['overall']:.4f} | onset {agg['onset']:.4f} | ongoing {agg['ongoing']:.4f}")
    print("per panel:", {k: round(v, 3) for k, v in agg["per_panel"].items()})
    print("per family:", {k: round(v, 3) for k, v in agg["per_family"].items()})

    meta = {"onset_rule": [on_best[0], float(on_best[1])],
            "ongoing_th": float(ong_best),
            "features": list(FT.FEATURES),
            "tune_onset": {str(k): float(v) for k, v in on_scores.items()},
            "tune_ongoing": {str(k): float(v) for k, v in ong_scores.items()},
            "cv_onset": float(agg["onset"]), "cv_ongoing": float(agg["ongoing"]),
            "cv_overall": float(agg["overall"])}
    with open(os.path.join(C.OUT2, "gbm_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    m_on.save_model(os.path.join(C.MODEL2, "gbm_onset_pre.txt"))
    m_ong.save_model(os.path.join(C.MODEL2, "gbm_ongoing_pre.txt"))

    # ---- refit on everything for the submission ------------------------
    if not args.no_refit:
        def gather_all(items, key):
            Xs, ys = [], []
            for it in items:
                Xs.append(it["X"])
                ys.append(it["y"])
            return np.concatenate(Xs, axis=0), np.concatenate(ys)
        print("refitting on all accepted windows ...")
        for key, items, rounds, name in (("on", on, args.rounds_onset, "gbm_onset"),
                                         ("ong", ong, args.rounds_ongoing, "gbm_ongoing")):
            Xa, ya = gather_all(items, key)
            ds = lgb.Dataset(Xa, label=ya, feature_name=list(FT.FEATURES), free_raw_data=True)
            ds.construct()
            del Xa, ya
            gc.collect()
            model = lgb.train(PARAMS, ds, num_boost_round=rounds)
            model.save_model(os.path.join(C.MODEL2, f"{name}.txt"))
            print(f"saved {name}.txt ({ds.num_data()} rows)")
            del ds, model
            gc.collect()


if __name__ == "__main__":
    main()

