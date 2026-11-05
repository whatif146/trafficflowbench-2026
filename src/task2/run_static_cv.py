"""Pseudo-CV sweep for the static Task 2 rules; reports and saves the selection.

Selection is honest: the knobs (onset link count rule, ongoing persistence
variant) are chosen by 4-block cross-validation *inside* the fit period (train
days before ``--cut``), then scored once on the held-out block (February 2031).
The candidate rule set is deliberately small so selection stays stable with the
few windows per block.

Usage: python run_static_cv.py [--panels ...] [--cut 2031-02-01]
"""
from __future__ import annotations

import argparse
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
import static_model as SM


def iou(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> float:
    p = pred.astype(bool) & elig
    t = truth.astype(bool)
    union = float(np.logical_or(p, t).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / union


def compact_hist(d: dict, ti: int) -> dict:
    """Only what the static rules need: robust current queue flags."""
    s = d["speed"][ti - C.HIST:ti]
    vc = d["v_cut"]
    qnow = np.isfinite(s[-1]) & (s[-1] <= vc)
    med3 = np.nanmedian(s[-3:], axis=0)
    med6 = np.nanmedian(s[-6:], axis=0)
    tt = d["time_dt"][ti + 1:ti + 1 + C.HORIZON]
    return {"qnow": qnow,
            "qmed3": np.isfinite(med3) & (med3 <= vc),
            "qmed6": np.isfinite(med6) & (med6 <= vc),
            "target_slots": ((tt.hour * 60 + tt.minute) // 5).to_numpy()}


def onset_freq_from(d: dict, wins: pd.DataFrame) -> np.ndarray:
    step = C.HORIZON - 1
    f = np.zeros(d["n_links"], np.float64)
    for r in wins.itertuples(index=False):
        f += d["Qobs"][r.ti + 1 + step]
    return f / max(len(wins), 1)


def rate_from(d: dict, wins: pd.DataFrame) -> np.ndarray:
    L = d["n_links"]
    if len(wins) == 0:
        return np.zeros((288, L), np.float64)
    keep = set(wins.time_dt.dt.date)
    days = pd.Series(d["time_dt"].date).isin(keep).to_numpy()
    tod = (d["time_dt"].hour * 60 + d["time_dt"].minute).to_numpy(np.int32)
    slots = tod // 5
    s = np.zeros((288, L))
    np.add.at(s, slots[days], d["Q"][days].astype(np.float64))
    cnt = np.bincount(slots[days], minlength=288).astype(np.float64)
    return s / np.maximum(cnt, 1.0)[:, None]


ONSET_RULES = (
    [("top", k) for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)]
    + [("ge", t) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
)

ONGOING_RULES = []
for mode in ("persist", "robust", "robust6"):
    for dil in (0, 1, 2):
        for rate_th in (None, 0.2, 0.35, 0.5):
            for shift in (0, -1, 1):
                ONGOING_RULES.append((mode, dil, rate_th, shift))


def onset_pred(freq: np.ndarray, rule) -> np.ndarray:
    kind, val = rule
    pred = np.zeros((C.HORIZON, len(freq)), bool)
    if kind == "top":
        sel = np.argsort(-freq)[:int(val)]
    else:
        sel = np.where(freq >= val)[0]
        if len(sel) == 0:
            sel = np.argsort(-freq)[:2]
    pred[C.HORIZON - 1, sel] = True
    return pred


def ongoing_pred(hist: dict, params: dict, rule) -> np.ndarray:
    """hist is a compact_hist dict."""
    mode, dil, rate_th, shift = rule
    if mode == "persist":
        base = hist["qnow"]
    elif mode == "robust":
        base = hist["qmed3"]
    else:
        base = hist["qmed6"]
    pred = np.repeat(base[None, :], C.HORIZON, axis=0)
    r2i = params["rank_to_idx"]
    if shift:
        for k in range(C.HORIZON):
            pred[k] = SM._shift_rank(pred[k][None, :], r2i, shift * (k + 1))[0]
    if dil:
        pred = (SM._shift_rank(pred, r2i, dil) | SM._shift_rank(pred, r2i, -dil) | pred)
    if rate_th is not None:
        for k in range(C.HORIZON):
            pred[k] |= params["rate"][hist["target_slots"][k]] >= rate_th
    return pred


def prep_windows(d: dict, wins: pd.DataFrame):
    out = []
    for r in wins.itertuples(index=False):
        ti = int(r.ti)
        out.append((compact_hist(d, ti),
                    d["Q"][ti + 1:ti + 1 + C.HORIZON].copy(),
                    d["elig"][ti + 1:ti + 1 + C.HORIZON].copy()))
    return out


def eval_panel(panel: str, cut: pd.Timestamp):
    d = C.load_train_panel(panel)
    cand = C.candidate_table(d)
    acc = C.greedy_select(cand).sort_values("time_dt").reset_index(drop=True)
    is_fit = (acc.time_dt < cut).to_numpy()
    fit = acc[is_fit].reset_index(drop=True)
    ev = acc[~is_fit].reset_index(drop=True)
    L = d["n_links"]
    step = C.HORIZON - 1

    fit_items = prep_windows(d, fit)
    ev_items = prep_windows(d, ev)
    fit_cond = fit.condition.to_numpy()
    ev_cond = ev.condition.to_numpy()

    # ---------- 4-block CV inside the fit period ---------------------------
    blocks = np.array_split(np.arange(len(fit)), 4)
    onset_scores = {rule: [] for rule in ONSET_RULES}
    ongoing_scores = {rule: [] for rule in ONGOING_RULES}
    for b in blocks:
        keep = np.ones(len(fit), bool)
        keep[b] = False
        freq = onset_freq_from(d, fit[(fit_cond == "queue_onset") & keep])
        rate = rate_from(d, fit[keep])
        params = {"n_links": L, "rank_to_idx": d["rank_to_idx"], "rate": rate}
        on_val = [j for j in b if fit_cond[j] == "queue_onset"]
        ong_val = [j for j in b if fit_cond[j] == "queue_ongoing"]
        for rule in ONSET_RULES:
            if on_val:
                onset_scores[rule].append(float(np.mean(
                    [iou(onset_pred(freq, rule), fit_items[j][1], fit_items[j][2])
                     for j in on_val])))
        for rule in ONGOING_RULES:
            if ong_val:
                ongoing_scores[rule].append(float(np.mean(
                    [iou(ongoing_pred(fit_items[j][0], params, rule),
                         fit_items[j][1], fit_items[j][2]) for j in ong_val])))

    def best(scores):
        m = {r: (float(np.mean(v)) if v else -np.inf) for r, v in scores.items()}
        r = max(m, key=m.get)
        return r, m[r], sorted(m.items(), key=lambda kv: -kv[1])[:5]

    on_rule, on_cv, on_top = best(onset_scores)
    ong_rule, ong_cv, ong_top = best(ongoing_scores)

    # ---------- final fit on all fit windows, one held-out score ----------
    freq = onset_freq_from(d, fit[fit_cond == "queue_onset"])
    rate = rate_from(d, fit)
    params = {"panel": panel, "links": d["links"], "n_links": L,
              "rank_to_idx": d["rank_to_idx"], "rate": rate, "onset_freq": freq}
    rows = []
    for j in range(len(ev)):
        h, truth, eligf = ev_items[j]
        if ev_cond[j] == "queue_onset":
            val = iou(onset_pred(freq, on_rule), truth, eligf)
            rows.append({"panel": panel, "condition": "queue_onset", "iou": val})
        else:
            val = iou(ongoing_pred(h, params, ong_rule), truth, eligf)
            rows.append({"panel": panel, "condition": "queue_ongoing", "iou": val})
    detail = pd.DataFrame(rows)

    # persistence reference on the held-out ongoing windows
    per_vals = []
    for j in range(len(ev)):
        if ev_cond[j] != "queue_ongoing":
            continue
        h, truth, eligf = ev_items[j]
        pred = np.repeat(h["qnow"][None, :], C.HORIZON, axis=0)
        per_vals.append(iou(pred, truth, eligf))

    info = {
        "panel": panel, "n_fit_onset": int((fit_cond == "queue_onset").sum()),
        "n_fit_ongoing": int((fit_cond == "queue_ongoing").sum()),
        "n_ev_onset": int((ev_cond == "queue_onset").sum()),
        "n_ev_ongoing": int((ev_cond == "queue_ongoing").sum()),
        "onset_rule": str(on_rule), "onset_cv": on_cv,
        "onset_heldout": float(detail[detail.condition == "queue_onset"].iou.mean()),
        "ongoing_rule": str(ong_rule), "ongoing_cv": ong_cv,
        "ongoing_heldout": float(detail[detail.condition == "queue_ongoing"].iou.mean()),
        "persistence_heldout": float(np.mean(per_vals)) if per_vals else np.nan,
        "onset_alt": [(str(r), round(f, 4)) for r, f in on_top],
        "ongoing_alt": [(str(r), round(f, 4)) for r, f in ong_top],
    }
    # online rule for other splits: onset links = rule applied to full-train freq;
    # save the selected links and ongoing rule so predict.py can reuse them.
    sel = np.where(onset_pred(freq, on_rule)[step])[0]
    save = {"panel": panel, "onset_links": d["links"][sel].tolist(),
            "onset_rule": str(on_rule), "ongoing_rule": str(ong_rule)}
    del d
    gc.collect()
    return detail, info, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--cut", default="2031-02-01")
    ap.add_argument("--tag", default="static")
    args = ap.parse_args()
    C.ensure_dirs()
    cut = pd.Timestamp(args.cut, tz="UTC")
    panels = args.panels or C.PANELS
    details, infos, saves = [], [], []
    for p in panels:
        if not os.path.exists(C.traffic.split_cache_path(p, "train")):
            print(f"{p}: no cache, skip")
            continue
        detail, info, save = eval_panel(p, cut)
        details.append(detail)
        infos.append(info)
        saves.append(save)
        print(f"{p}: onset {info['onset_rule']} cv={info['onset_cv']:.3f} "
              f"held={info['onset_heldout']:.3f} | ongoing {info['ongoing_rule']} "
              f"cv={info['ongoing_cv']:.3f} held={info['ongoing_heldout']:.3f} "
              f"| persist={info['persistence_heldout']:.3f}")
        gc.collect()
    detail = pd.concat(details, ignore_index=True)
    detail.to_csv(os.path.join(C.OUT2, f"{args.tag}_cv.csv"), index=False)
    with open(os.path.join(C.OUT2, f"{args.tag}_cv_info.json"), "w") as fh:
        json.dump({"infos": infos, "saves": saves}, fh, indent=1)
    agg = C.aggregate_iou(detail)
    print("\n== held-out pseudo-CV, official nesting ==")
    print(f"overall {agg['overall']:.4f} | onset {agg['onset']:.4f} | ongoing {agg['ongoing']:.4f}")
    print("per panel:", {k: round(v, 3) for k, v in agg["per_panel"].items()})
    print("per family:", {k: round(v, 3) for k, v in agg["per_family"].items()})
    rows = []
    for r in infos:
        rows.append({"panel": r["panel"], "condition": "queue_ongoing", "iou": r["persistence_heldout"]})
        rows.append({"panel": r["panel"], "condition": "queue_onset", "iou": 0.0})
    pers = C.aggregate_iou(pd.DataFrame(rows))
    print(f"persistence-only reference: overall {pers['overall']:.4f}")


if __name__ == "__main__":
    main()
