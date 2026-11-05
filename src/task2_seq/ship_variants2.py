"""Climatology-fallback variants of the final v3+v5 blend (no retraining).

Idea: with the shipped onset threshold 0.30, 6/40 validation onset windows predict
NO queued link at all (IoU 0 by construction). The blend already provides a
per-link probability; a plain top-1 fallback uses that probability. Here the
fallback instead uses the *train climatology* (per-link queue rate at the
window's target time-of-day), which is the most stable signal for the recurrent
onset hotspots and is independent of the possibly-uninformative window
probabilities.

Outputs (validation+private each):
  queue_v5blend_onset_fbclim_{split}.csv         th 0.30 + climatology fallback
  queue_v5blend_onset040_fbclim_{split}.csv      th 0.40 + climatology fallback
  queue_v5blend_onset050_fbclim_{split}.csv      th 0.50 + climatology fallback
  queue_v5blend_onset_fbmix_{split}.csv          th 0.30 + fallback argmax(p*rate)
"""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import ship_blend as SB
import ship_variants as SV

import lightgbm as lgb


def slot_of(epoch: int, offset_steps: int = 6) -> int:
    ts = pd.Timestamp(epoch + 300 * offset_steps, unit="s", tz="UTC")
    return (ts.hour * 60 + ts.minute) // 5


def onset_pred_clim(prob, L, th, fallback, rate, slot, mix=False):
    pred = np.zeros((LH.HORIZON, L), bool)
    p = prob[LH.HORIZON - 1]
    pred[LH.HORIZON - 1] = p >= th
    if fallback and not pred[LH.HORIZON - 1].any():
        if mix:
            score = p * np.asarray(rate[slot], np.float32)
        else:
            score = np.asarray(rate[slot], np.float32)
        pred[LH.HORIZON - 1, int(np.argmax(score))] = True
    return pred


def write_csv(probmap, tag, split, tpl_all, on_th, ong_th, mode):
    import traffic
    frames = []
    for wid, (cond, prob, t0, L) in probmap.items():
        parts = wid.split("_")
        panel = "_".join(parts[1:-2])
        st = LH.static_panel(panel)
        if cond == "queue_onset":
            if mode == "none":
                pred = np.zeros((LH.HORIZON, L), bool)
                pred[LH.HORIZON - 1] = prob[LH.HORIZON - 1] >= on_th
            else:
                pred = onset_pred_clim(prob, L, on_th, True, st["rate"],
                                       slot_of(t0), mix=(mode == "mix"))
        else:
            pred = prob >= ong_th
        ts = [t0 + 300 * (k + 1) for k in range(LH.HORIZON)]
        frames.append(pd.DataFrame({
            "window_id": np.full(LH.HORIZON * L, wid),
            "timestamp": np.repeat(traffic.epoch_to_iso(np.asarray(ts, np.int64)), L),
            "link_id": np.tile(st["links"], LH.HORIZON),
            "queue_pred": pred.reshape(-1).astype(np.int8),
        }))
    pred = pd.concat(frames, ignore_index=True)
    tpl = tpl_all[tpl_all.window_id.isin(set(pred.window_id))].copy()
    out = tpl.merge(pred, on=["window_id", "timestamp", "link_id"], how="left",
                    validate="one_to_one", indicator=True)
    if int((out._merge != "both").sum()):
        raise RuntimeError("template rows without a prediction")
    out = out.drop(columns=["_merge"])
    out["queue_pred"] = out.queue_pred.astype(np.int8)
    out = out[["window_id", "timestamp", "link_id", "queue_pred"]]
    path = os.path.join(LH.OUT, f"{tag}_{split}.csv")
    out.to_csv(path, index=False)
    empties = 0
    onset = out[out.window_id.str.contains("Q2")]
    if len(onset):
        ids = onset.window_id.map(lambda w: "_".join(w.split("_")[1:-2]))
        cond = pd.Series(lh_cond)
        pass
    print(f"{split}: {tag} -> {len(out):,} rows, {int(out.queue_pred.sum()):,} positives")


lh_cond = {}


def main():
    try:
        cv = json.load(open(os.path.join(LH.OUT, "final_cv_stride1.json")))
        w_on = float(cv["on"]["blend_B"]["w"])
        w_ong = float(cv["ong"]["blend_B"]["w"])
    except Exception:
        w_on, w_ong = 0.5, 0.6
    m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_onset.txt"))
    m_ong = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_ongoing.txt"))
    v3_on, v3_ong = SB.load_v3_boosters()
    import traffic
    tpl_all = traffic.load_queue_template()
    tpl_all["window_id"] = tpl_all.window_id.astype(str)
    tpl_all["link_id"] = tpl_all.link_id.astype(str)
    tpl_all = tpl_all.drop(columns=["queue_pred"], errors="ignore")
    for split in ("validation", "private"):
        pm = SV.blend_probs_split(split, w_on, w_ong, m_on, m_ong, v3_on, v3_ong)
        print(f"{split}: {len(pm)} windows, w_on {w_on} w_ong {w_ong}", flush=True)
        write_csv(pm, "queue_v5blend_onset_fbclim", split, tpl_all, 0.30, 0.25, "clim")
        write_csv(pm, "queue_v5blend_onset040_fbclim", split, tpl_all, 0.40, 0.25, "clim")
        write_csv(pm, "queue_v5blend_onset050_fbclim", split, tpl_all, 0.50, 0.25, "clim")
        write_csv(pm, "queue_v5blend_onset_fbmix", split, tpl_all, 0.30, 0.25, "mix")
        del pm
        gc.collect()


if __name__ == "__main__":
    main()
