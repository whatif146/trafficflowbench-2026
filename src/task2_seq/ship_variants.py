"""Threshold/rule variants of the FINAL v3+v5 blend (no retraining, frozen artefacts).

Writes small variant files for the Lead to probe on the LB with everything else
held fixed:

  queue_v5blend_ong015_{validation,private}.csv   ongoing th 0.15
  queue_v5blend_ong035_{...}.csv                  ongoing th 0.35
  queue_v5blend_on020_{...}.csv                   onset th 0.20 (last step only)
  queue_v5blend_on040_{...}.csv                   onset th 0.40
  queue_v5blend_onfb1_{...}.csv                   onset th 0.30 union top-1 fallback
  queue_v5blend_ong015_onfb1_{...}.csv            both of the above combined

The base candidate out/queue_v5_{validation,private}.csv is unchanged.
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
import run_cv as RC
import ship_blend as SB

import lightgbm as lgb


def blend_probs_split(split: str, w_on: float, w_ong: float, m_on, m_ong, v3_on, v3_ong):
    """dict window_id -> (condition, 6xL probability matrix) for one split."""
    out = {}
    for p in LH.PANELS:
        st = LH.static_panel(p)
        L = st["n_links"]
        z = RC.load_rel(p)
        sel = np.where(z["split"] == split)[0]
        if len(sel) == 0:
            z.close()
            continue
        X3 = z["v3"]
        X5 = RC.rel_X("on", "v5", z)      # 65 v3 + 80 long-history = 145
        P = LH.HORIZON * L
        cond = z["condition"]
        for j, wj in enumerate(sel):
            rr = np.arange(wj * P, (wj + 1) * P)
            if cond[wj] == "queue_onset":
                a = v3_on.predict(np.ascontiguousarray(X3[rr]))
                b = m_on.predict(np.ascontiguousarray(X5[rr]))
                prob = (w_on * a + (1 - w_on) * b).reshape(LH.HORIZON, L)
            else:
                a = v3_ong.predict(np.ascontiguousarray(X3[rr]))
                b = m_ong.predict(np.ascontiguousarray(X5[rr]))
                prob = (w_ong * a + (1 - w_ong) * b).reshape(LH.HORIZON, L)
            out[str(z["window_ids"][wj])] = (str(cond[wj]), prob, int(z["time"][wj]), L)
        del X3, X5
        z.close()
        gc.collect()
    return out


def onset_pred(prob, L, th, fallback):
    pred = np.zeros((LH.HORIZON, L), bool)
    p = prob[LH.HORIZON - 1]
    pred[LH.HORIZON - 1] = p >= th
    if fallback and not pred[LH.HORIZON - 1].any():
        k = int(fallback) if isinstance(fallback, (int, np.integer)) else 1
        k = max(1, min(k, L))
        pred[LH.HORIZON - 1, np.argsort(-p)[:k]] = True
    return pred


def write_csv(probmap, tag, split, tpl_all, on_th, on_fb, ong_th):
    import traffic
    frames = []
    for wid, (cond, prob, t0, L) in probmap.items():
        parts = wid.split("_")
        panel = "_".join(parts[1:-2])      # Q2_<panel>_<split>_<NNN>
        st = LH.static_panel(panel)
        if cond == "queue_onset":
            pred = onset_pred(prob, L, on_th, on_fb)
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
    print(f"{split}: {tag} -> {len(out):,} rows, {int(out.queue_pred.sum()):,} positives "
          f"-> {os.path.basename(path)}")


def main():
    cv = json.load(open(os.path.join(LH.OUT, "final_cv_stride1.json")))
    w_on = float(cv["on"]["blend_B"]["w"])
    w_ong = float(cv["ong"]["blend_B"]["w"])
    m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_onset.txt"))
    m_ong = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_ongoing.txt"))
    v3_on, v3_ong = SB.load_v3_boosters()
    import traffic
    tpl_all = traffic.load_queue_template()
    tpl_all["window_id"] = tpl_all.window_id.astype(str)
    tpl_all["link_id"] = tpl_all.link_id.astype(str)
    tpl_all = tpl_all.drop(columns=["queue_pred"], errors="ignore")
    variants = [
        ("ongoing_0.15", dict(on_th=0.30, on_fb=False, ong_th=0.15)),
        ("ongoing_0.35", dict(on_th=0.30, on_fb=False, ong_th=0.35)),
        ("onset_0.20", dict(on_th=0.20, on_fb=False, ong_th=0.25)),
        ("onset_0.25", dict(on_th=0.25, on_fb=False, ong_th=0.25)),
        ("onset020_fb1", dict(on_th=0.20, on_fb=True, ong_th=0.25)),
        ("onset025_fb1", dict(on_th=0.25, on_fb=True, ong_th=0.25)),
        ("onset_0.40", dict(on_th=0.40, on_fb=False, ong_th=0.25)),
        ("onset_0.45", dict(on_th=0.45, on_fb=False, ong_th=0.25)),
        ("onset_0.50", dict(on_th=0.50, on_fb=False, ong_th=0.25)),
        ("onset_0.55", dict(on_th=0.55, on_fb=False, ong_th=0.25)),
        ("onset_0.60", dict(on_th=0.60, on_fb=False, ong_th=0.25)),
        ("onset_0.65", dict(on_th=0.65, on_fb=False, ong_th=0.25)),
        ("onset060_fb1", dict(on_th=0.60, on_fb=True, ong_th=0.25)),
        ("onset065_fb1", dict(on_th=0.65, on_fb=True, ong_th=0.25)),
        ("onset_fb1", dict(on_th=0.30, on_fb=True, ong_th=0.25)),
        ("onset_fb2", dict(on_th=0.30, on_fb=2, ong_th=0.25)),
        ("onset040_fb1", dict(on_th=0.40, on_fb=True, ong_th=0.25)),
        ("onset050_fb1", dict(on_th=0.50, on_fb=True, ong_th=0.25)),
        ("onset055_fb1", dict(on_th=0.55, on_fb=True, ong_th=0.25)),
        ("onset_0.575", dict(on_th=0.575, on_fb=False, ong_th=0.25)),
        ("onset_0.625", dict(on_th=0.625, on_fb=False, ong_th=0.25)),
        ("onset0575_fb1", dict(on_th=0.575, on_fb=True, ong_th=0.25)),
        ("onset0625_fb1", dict(on_th=0.625, on_fb=True, ong_th=0.25)),
        ("onset055_fb2", dict(on_th=0.55, on_fb=2, ong_th=0.25)),
        ("onset060_fb2", dict(on_th=0.60, on_fb=2, ong_th=0.25)),
        ("ong015_onfb1", dict(on_th=0.30, on_fb=True, ong_th=0.15)),
    ]
    for split in ("validation", "private"):
        pm = blend_probs_split(split, w_on, w_ong, m_on, m_ong, v3_on, v3_ong)
        print(f"{split}: {len(pm)} windows, w_on {w_on} w_ong {w_ong}", flush=True)
        for name, kw in variants:
            write_csv(pm, f"queue_v5blend_{name}", split, tpl_all, **kw)
        del pm
        gc.collect()


if __name__ == "__main__":
    main()
