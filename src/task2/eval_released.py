"""Score the current model path on the released *train* windows.

The released train windows are the real selector output and the train split
ships the unmasked observation layer, so the future horizon speeds are known:
the smoothed queue indicator there is the best local stand-in for the hidden
queue truth.  This is the second (and most release-like) proxy score; the
February pseudo-CV is the first.

    python eval_released.py --model gbm
"""
from __future__ import annotations

import argparse
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C


def iou(pred, truth, elig):
    p = pred.astype(bool) & elig
    t = truth.astype(bool)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / float(union)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gbm", "static"], default="gbm")
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()
    windows = C.released_windows()
    windows = windows[windows.split == "train"].copy()
    hist = C.released_history()
    rows = []
    if args.model == "gbm":
        import gbm_predict as GP
        meta = GP.load_meta()
        m_on, m_ong = GP.load_models()
    for panel, wp in windows.groupby("panel", sort=True):
        d = C.load_train_panel(panel, smooth=args.smooth)
        time_index = pd.Series(np.arange(len(d["times"])), index=d["time_dt"])
        if args.model == "gbm":
            clim = GP.attach_ramps(GP.load_clim(panel), "train")
            links = np.asarray(clim["links"], dtype=str)
        else:
            from predict import load_static_params, history_arrays_from_frame, static_ongoing_pred
            p = load_static_params(panel)
            links = np.asarray(p["links"], dtype=str)
        link_pos = {l: i for i, l in enumerate(links)}
        hp = hist[hist.window_id.isin(set(wp.window_id))]
        for r in wp.itertuples(index=False):
            wid = str(r.window_id)
            h = hp[hp.window_id == wid]
            tgt = pd.DatetimeIndex([r.forecast_start + pd.Timedelta(minutes=5 * k)
                                    for k in range(C.HORIZON)])
            # truth from the plain layer (cache link order) -> reorder to model order
            ti0 = int(time_index.loc[r.forecast_origin])
            truth_cache = d["Q"][ti0 + 1:ti0 + 1 + C.HORIZON]
            elig_cache = d["elig"][ti0 + 1:ti0 + 1 + C.HORIZON]
            order = np.array([link_pos[l] for l in d["links"]], dtype=np.int64)
            L = len(links)
            truth = np.zeros((C.HORIZON, L), np.int8)
            elig = np.zeros((C.HORIZON, L), bool)
            truth[:, order] = truth_cache
            elig[:, order] = elig_cache
            if args.model == "gbm":
                pred = GP.predict_window(h, clim, meta, m_on, m_ong, tgt)
            else:
                arr = history_arrays_from_frame(h, links, p["v_cut"])
                arr["target_slots"] = np.array([(t.hour * 60 + t.minute) // 5 for t in tgt])
                if r.condition == "queue_onset":
                    pred = np.zeros((C.HORIZON, L), bool)
                    pred[C.HORIZON - 1, np.asarray(p["onset_sel"], dtype=np.int64)] = True
                else:
                    pred = static_ongoing_pred(arr, p, p)
            rows.append({"panel": panel, "condition": r.condition,
                         "window_id": wid, "iou": iou(pred, truth, elig),
                         "truth_n": int(truth.sum())})
        del d
    detail = pd.DataFrame(rows)
    detail.to_csv(os.path.join(C.OUT2, f"released_train_{args.model}.csv"), index=False)
    agg = C.aggregate_iou(detail)
    print(f"== released train windows, {args.model} ==")
    print(f"overall {agg['overall']:.4f} | onset {agg['onset']:.4f} | ongoing {agg['ongoing']:.4f}")
    print("per panel:", {k: round(v, 3) for k, v in agg["per_panel"].items()})
    print("per family:", {k: round(v, 3) for k, v in agg["per_family"].items()})
    print(detail.groupby("condition").iou.agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
