"""Task 2 submission builder.

Writes the binary queue predictions for the released windows of one split.

    python predict.py --split validation --output out/queue_validation.csv
    python predict.py --split private    --output out/queue_private.csv

Two model paths:
  --model gbm     per-link LightGBM models (train_gbm.py); default when the
                  model files exist
  --model static  rule-based fallback (fit_static.py): recurrent onset hotspot
                  links at the last horizon step + robust persistence for
                  ongoing windows

Only history stamped <= T is used (the released 12-step window history).
"""
from __future__ import annotations

import argparse
import ast
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C
import static_model as SM


# --------------------------------------------------------------------------- #
# static path
# --------------------------------------------------------------------------- #
def load_static_params(panel: str) -> dict:
    path = os.path.join(C.MODEL2, f"static_{panel}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}; run fit_static.py first")
    z = np.load(path, allow_pickle=False)
    p = {k: z[k] for k in z.files}
    z.close()
    p["panel"] = panel
    return p


def history_arrays_from_frame(h: pd.DataFrame, links: np.ndarray, v_cut: np.ndarray) -> dict:
    l_index = {str(l): i for i, l in enumerate(links)}
    times = np.sort(h.timestamp.unique())
    nT, L = len(times), len(links)
    speed = np.full((nT, L), np.nan, np.float32)
    flow = np.full((nT, L), np.nan, np.float32)
    elig = np.zeros((nT, L), bool)
    t_index = {t: i for i, t in enumerate(times)}
    ti = h.timestamp.map(t_index).to_numpy()
    li = h.link_id.map(l_index)
    keep = li.notna().to_numpy()
    sub = h[keep]
    ti = ti[keep]
    li = li[keep].astype(int).to_numpy()
    speed[ti, li] = pd.to_numeric(sub.speed_kmh, errors="coerce").to_numpy(np.float32)
    flow[ti, li] = pd.to_numeric(sub.flow_vph, errors="coerce").to_numpy(np.float32)
    elig[ti, li] = sub.is_score_eligible.astype(bool).to_numpy()
    return SM.history_arrays(speed, flow, elig, v_cut)


def static_ongoing_pred(arr: dict, params: dict, p: dict) -> np.ndarray:
    rule = ast.literal_eval(str(p["ongoing_rule"]))
    mode, dil, rate_th, shift = rule
    base = {"persist": arr["qnow"], "robust": arr["qmed3"], "robust6": arr["qmed6"]}[mode]
    pred = np.repeat(base[None, :], C.HORIZON, axis=0)
    r2i = np.asarray(params["rank_to_idx"], dtype=np.int64)
    if shift:
        for k in range(C.HORIZON):
            pred[k] = SM._shift_rank(pred[k][None, :], r2i, shift * (k + 1))[0]
    if dil:
        pred = (SM._shift_rank(pred, r2i, dil) | SM._shift_rank(pred, r2i, -dil) | pred)
    if rate_th is not None:
        for k in range(C.HORIZON):
            pred[k] |= params["rate"][arr["target_slots"][k]] >= rate_th
    return pred


# --------------------------------------------------------------------------- #
def predict_split(split: str, model: str) -> pd.DataFrame:
    windows = C.released_windows()
    windows = windows[windows.split == split].copy()
    template = C.traffic.load_queue_template()
    template["window_id"] = template.window_id.astype(str)
    template["link_id"] = template.link_id.astype(str)
    template["timestamp"] = pd.to_datetime(template.timestamp, utc=True)
    template = template[template.window_id.isin(set(windows.window_id))].copy()
    history = C.released_history()
    history = history[history.window_id.isin(set(windows.window_id))]

    gbm = None
    if model in ("gbm", "gbmv3"):
        import gbm_predict as GP
        version = None if model == "gbm" else "v3"
        meta = GP.load_meta(version)
        m_on, m_ong = GP.load_models(version)
        gbm = (GP, meta, m_on, m_ong)

    out_rows = []
    for panel, wp in windows.groupby("panel", sort=True):
        if model in ("gbm", "gbmv3"):
            GP, meta, m_on, m_ong = gbm
            clim = GP.attach_ramps(GP.load_clim(panel), split)
            links = np.asarray(clim["links"], dtype=str)
        else:
            p = load_static_params(panel)
            links = np.asarray(p["links"], dtype=str)
        hp = history[history.window_id.isin(set(wp.window_id))]
        tp = template[template.window_id.isin(set(wp.window_id))]
        for r in wp.itertuples(index=False):
            wid = str(r.window_id)
            h = hp[hp.window_id == wid]
            tpl = tp[tp.window_id == wid]
            step_times = pd.DatetimeIndex(np.sort(tpl.timestamp.unique()))
            if model in ("gbm", "gbmv3"):
                if len(step_times) != C.HORIZON:
                    raise RuntimeError(f"{wid}: expected 6 forecast steps, got {len(step_times)}")
                pred = GP.predict_window(h, clim, meta, m_on, m_ong, step_times)
            else:
                arr = history_arrays_from_frame(h, links, p["v_cut"])
                arr["target_slots"] = np.array(
                    [(pd.Timestamp(t).hour * 60 + pd.Timestamp(t).minute) // 5 for t in step_times])
                if r.condition == "queue_onset":
                    pred = np.zeros((len(step_times), len(links)), bool)
                    sel = np.asarray(p["onset_sel"], dtype=np.int64)
                    pred[len(step_times) - 1, sel] = True
                    if not pred.any():
                        pred[len(step_times) - 1, np.argsort(-p["onset_freq"])[:2]] = True
                else:
                    pred = static_ongoing_pred(arr, p, p)
            li = tpl.link_id.map({l: i for i, l in enumerate(links)})
            ti = tpl.timestamp.map({t: i for i, t in enumerate(step_times)})
            vals = np.zeros(len(tpl), np.int8)
            ok = li.notna().to_numpy() & ti.notna().to_numpy()
            vals[ok] = pred[ti[ok].astype(int).to_numpy(),
                            li[ok].astype(int).to_numpy()].astype(np.int8)
            sub = tpl[["window_id", "timestamp", "link_id"]].copy()
            sub["queue_pred"] = vals
            out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["validation", "private", "both"], default="validation")
    ap.add_argument("--output", default=None)
    ap.add_argument("--model", choices=["gbm", "gbmv3", "static"], default="gbm")
    args = ap.parse_args()
    C.ensure_dirs()
    if args.model in ("gbm", "gbmv3"):
        adir = os.path.join(C.MODEL2, "v3") if args.model == "gbmv3" else C.MODEL2
        need = [os.path.join(adir, f) for f in ("gbm_onset.txt", "gbm_ongoing.txt")]
        if not all(os.path.exists(p) for p in need):
            print("GBM artefacts missing, falling back to the static model")
            args.model = "static"
    splits = ["validation", "private"] if args.split == "both" else [args.split]
    for split in splits:
        out = args.output or os.path.join(C.OUT2, f"queue_{args.model}_{split}.csv")
        if len(splits) > 1 and args.output:
            out = f"{args.output.rsplit('.csv', 1)[0]}_{split}.csv"
        df = predict_split(split, args.model)
        df["timestamp"] = df.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df = df.sort_values(["window_id", "timestamp", "link_id"]).reset_index(drop=True)
        df.to_csv(out, index=False)
        print(f"{split} [{args.model}]: {len(df):,} rows -> {out} "
              f"(positives {int(df.queue_pred.sum()):,}, windows {df.window_id.nunique()})")


if __name__ == "__main__":
    main()
