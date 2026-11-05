"""Ship queue variants with different v3/v5 blend weights (honest-CV driven).

Two-protocol honest CV (models refit with every June origin excluded; B = untouched
Feb pseudo block, A = released June windows) gave:
  onset weight  w_on=0.3  beats shipped 0.5  (mean onset IoU 0.7926 vs 0.7857)
  ongoing weight w_ong=0.4 with th 0.20 beats w 0.6 / th 0.25 (0.8558 vs 0.8523, stride 2)

Outputs (validation+private each), no refit:
  queue_w30_*                  w_on 0.3, onset th 0.30; ongoing w 0.6 / th 0.25
  queue_w30_on025fb1_*         w_on 0.3, onset th 0.25 + top-1 fallback
  queue_w30_on030fb1_*         w_on 0.3, onset th 0.30 + top-1 fallback
  queue_w30ong40_*             w_on 0.3, onset th 0.30; ongoing w 0.4 / th 0.20
  queue_w30ong40_on025fb1_*    w_on 0.3, onset th 0.25 + fallback; ongoing w 0.4 / th 0.20
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
import ship_blend as SB

import lightgbm as lgb

W_ON = 0.3
CONFIGS = {
    "w30": dict(w_on=0.3, w_ong=0.6, on_th=0.30, fb=0, ong_th=0.25),
    "w30_on025fb1": dict(w_on=0.3, w_ong=0.6, on_th=0.25, fb=1, ong_th=0.25),
    "w30_on030fb1": dict(w_on=0.3, w_ong=0.6, on_th=0.30, fb=1, ong_th=0.25),
    "w30ong40": dict(w_on=0.3, w_ong=0.4, on_th=0.30, fb=0, ong_th=0.20),
    "w30ong40fb1": dict(w_on=0.3, w_ong=0.4, on_th=0.30, fb=1, ong_th=0.20),
    # refinements of the LB-winning fallback family (ongoing calibration on top)
    "w30ong40fb1_ong25": dict(w_on=0.3, w_ong=0.4, on_th=0.30, fb=1, ong_th=0.25, fb_ong=1),
    "w30ong30fb1": dict(w_on=0.3, w_ong=0.3, on_th=0.30, fb=1, ong_th=0.20, fb_ong=1),
    "w40ong40fb1": dict(w_on=0.4, w_ong=0.4, on_th=0.30, fb=1, ong_th=0.20, fb_ong=1),
    "w30ong40_th35fb1": dict(w_on=0.3, w_ong=0.4, on_th=0.35, fb=1, ong_th=0.20),
    "w30ong40_th40fb1": dict(w_on=0.3, w_ong=0.4, on_th=0.40, fb=1, ong_th=0.20),
    "w30ong40_th45fb1": dict(w_on=0.3, w_ong=0.4, on_th=0.45, fb=1, ong_th=0.20),
    "w30ong40_on025fb1": dict(w_on=0.3, w_ong=0.4, on_th=0.25, fb=1, ong_th=0.20),
    # top-2 fallback (every onset window contains a queue; the 0.87 public
    # notebook also falls back to the two most likely links)
    "w30fb2": dict(w_on=0.3, w_ong=0.6, on_th=0.30, fb=2, ong_th=0.25),
    "w30ong40fb2": dict(w_on=0.3, w_ong=0.4, on_th=0.30, fb=2, ong_th=0.20),
    # combined best of the multi-month simulation (onset th .25 + ongoing w .4/th .20)
    "w30ong40_on025": dict(w_on=0.3, w_ong=0.4, on_th=0.25, fb=0, ong_th=0.20),
    "on025ong40": dict(w_on=0.5, w_ong=0.4, on_th=0.25, fb=0, ong_th=0.20),
    # refinement grid around the LB-best q_w30ong40 (0.82984)
    "w30ong30": dict(w_on=0.3, w_ong=0.3, on_th=0.30, fb=0, ong_th=0.20),
    "w40ong40": dict(w_on=0.4, w_ong=0.4, on_th=0.30, fb=0, ong_th=0.20),
    "w30ong40_th25": dict(w_on=0.3, w_ong=0.4, on_th=0.30, fb=0, ong_th=0.25),
    # "minimum predicted set" variants: every onset window provably contains a queue,
    # so ensure at least min_k links are predicted (top-up with the highest probabilities)
    "min2": dict(w_on=0.5, w_ong=0.6, on_th=0.30, fb=0, ong_th=0.25, min_k=2),
    "min3": dict(w_on=0.5, w_ong=0.6, on_th=0.30, fb=0, ong_th=0.25, min_k=3),
    "w30ong40_min2": dict(w_on=0.3, w_ong=0.4, on_th=0.30, fb=0, ong_th=0.20, min_k=2),
    "w30ong40_min3": dict(w_on=0.3, w_ong=0.4, on_th=0.30, fb=0, ong_th=0.20, min_k=3),
    # window-level fallback for ongoing windows too (spec: every ongoing window
    # contains a queue somewhere in the horizon -> an empty mask scores 0)
    "w30ong40fb1_all": dict(w_on=0.3, w_ong=0.4, on_th=0.30, fb=1, ong_th=0.20, fb_ong=1),
    "onfb1_all": dict(w_on=0.5, w_ong=0.6, on_th=0.30, fb=1, ong_th=0.25, fb_ong=1),
}
ONLY = set(sys.argv[1:]) if len(sys.argv) > 1 else None


def predict_onset(prob, th, fb, min_k=0):
    pred = np.zeros((LH.HORIZON, prob.shape[1]), bool)
    p = prob[LH.HORIZON - 1]
    pred[LH.HORIZON - 1] = p >= th
    if fb and not pred[LH.HORIZON - 1].any():
        k = max(1, min(int(fb), len(p)))
        pred[LH.HORIZON - 1, np.argsort(-p)[:k]] = True
    if min_k and pred[LH.HORIZON - 1].sum() < min_k:
        k = max(1, min(int(min_k), len(p)))
        pred[LH.HORIZON - 1, np.argsort(-p)[:k]] = True
    return pred


def main():
    m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_onset.txt"))
    m_ong = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_ongoing.txt"))
    v3_on, v3_ong = SB.load_v3_boosters()
    import traffic
    tpl_all = traffic.load_queue_template()
    tpl_all["window_id"] = tpl_all.window_id.astype(str)
    tpl_all["link_id"] = tpl_all.link_id.astype(str)
    tpl_all = tpl_all.drop(columns=["queue_pred"], errors="ignore")
    for split in ("validation", "private"):
        rows = {}
        for p in LH.PANELS:
            st = LH.static_panel(p)
            links = st["links"]
            L = st["n_links"]
            z = RC.load_rel(p)
            sel = np.where(z["split"] == split)[0]
            if len(sel) == 0:
                z.close()
                continue
            X3 = z["v3"]
            X5 = RC.rel_X("on", "v5", z)
            P = LH.HORIZON * L
            cond = z["condition"]
            on_sel = sel[cond[sel] == "queue_onset"]
            ong_sel = sel[cond[sel] == "queue_ongoing"]
            prob_on = {w: np.zeros((len(sel), LH.HORIZON, L), np.float32) for w in (0.3, 0.4, 0.5)}
            prob_ong = {w: np.zeros((len(sel), LH.HORIZON, L), np.float32) for w in (0.3, 0.4, 0.6)}
            if len(on_sel):
                rr = np.concatenate([np.arange(j * P, (j + 1) * P) for j in on_sel])
                a = v3_on.predict(np.ascontiguousarray(X3[rr])).reshape(len(on_sel), LH.HORIZON, L)
                b = m_on.predict(np.ascontiguousarray(X5[rr])).reshape(len(on_sel), LH.HORIZON, L)
                for w in prob_on:
                    prob_on[w][np.isin(sel, on_sel)] = w * a + (1 - w) * b
            if len(ong_sel):
                rr = np.concatenate([np.arange(j * P, (j + 1) * P) for j in ong_sel])
                a = v3_ong.predict(np.ascontiguousarray(X3[rr])).reshape(len(ong_sel), LH.HORIZON, L)
                b = m_ong.predict(np.ascontiguousarray(X5[rr])).reshape(len(ong_sel), LH.HORIZON, L)
                for w in prob_ong:
                    prob_ong[w][np.isin(sel, ong_sel)] = w * a + (1 - w) * b
            del X3, X5
            for key, cfg in CONFIGS.items():
                if ONLY and key not in ONLY:
                    continue
                for j, wj in enumerate(sel):
                    wid = str(z["window_ids"][wj])
                    t0 = int(z["time"][wj])
                    ts = [t0 + 300 * (k + 1) for k in range(LH.HORIZON)]
                    if cond[wj] == "queue_onset":
                        pred = predict_onset(prob_on[cfg["w_on"]][j], cfg["on_th"],
                                             cfg["fb"], cfg.get("min_k", 0))
                    else:
                        pred = prob_ong[cfg["w_ong"]][j] >= cfg["ong_th"]
                        if cfg.get("fb_ong") and not pred.any():
                            # spec: every ongoing window also contains a queue somewhere
                            # in the horizon, so an empty prediction is a guaranteed 0;
                            # marking the single most likely cell weakly dominates.
                            flat = int(np.argmax(prob_ong[cfg["w_ong"]][j].reshape(-1)))
                            pred.reshape(-1)[flat] = True
                    rows.setdefault(key, []).append(pd.DataFrame({
                        "window_id": np.full(P, wid),
                        "timestamp": np.repeat(traffic.epoch_to_iso(np.asarray(ts, np.int64)), L),
                        "link_id": np.tile(links, LH.HORIZON),
                        "queue_pred": pred.reshape(-1).astype(np.int8),
                    }))
            z.close()
            gc.collect()
        for key, frames in rows.items():
            pred = pd.concat(frames, ignore_index=True)
            tpl = tpl_all[tpl_all.window_id.isin(set(pred.window_id))].copy()
            out = tpl.merge(pred, on=["window_id", "timestamp", "link_id"], how="left",
                            validate="one_to_one", indicator=True)
            if int((out._merge != "both").sum()):
                raise RuntimeError(f"{split} {key}: template rows without prediction")
            out = out.drop(columns=["_merge"])
            out["queue_pred"] = out.queue_pred.astype(np.int8)
            path = os.path.join(LH.OUT, f"queue_{key}_{split}.csv")
            out.to_csv(path, index=False)
            print(f"{split} {key}: {len(out):,} rows, {int(out.queue_pred.sum()):,} positives -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
