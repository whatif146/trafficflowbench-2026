"""Write the ramp-candidate Task 2 CSVs (only after a positive honest ramp CV).

Onset half: shipped v5 blend (w_on from the January selection, th 0.30).
Ongoing half: w * p_v3 + (1-w) * p_v5r with v5r = v3 + 80 long-history + 8 ramp
jam-sensor features; w and th are the January-selected values in
out/ramp_cv.json (blend_v3_v5r).
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


def main():
    cv = json.load(open(os.path.join(LH.OUT, "final_cv_stride1.json")))
    ramp = json.load(open(os.path.join(LH.OUT, "ramp_cv.json")))
    w_on = float(cv["on"]["blend_B"]["w"])
    on_th = float(cv["on"]["blend_B"]["th"])
    w_ong = float(ramp["B"]["blend_v5r"]["w"])
    ong_th = float(ramp["B"]["blend_v5r"]["th"])
    print(f"onset: w {w_on} th {on_th} | ongoing ramp blend: w {w_ong} th {ong_th}")
    m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_onset.txt"))
    m_ongr = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5r_ongoing.txt"))
    v3_on, v3_ong = SB.load_v3_boosters()
    import traffic
    tpl_all = traffic.load_queue_template()
    tpl_all["window_id"] = tpl_all.window_id.astype(str)
    tpl_all["link_id"] = tpl_all.link_id.astype(str)
    tpl_all = tpl_all.drop(columns=["queue_pred"], errors="ignore")
    names_ong = list(RC.feature_names("v5")) + list(LH.RAMP_FEATURES)
    for split in ("validation", "private"):
        frames = []
        for p in LH.PANELS:
            st = LH.static_panel(p)
            L = st["n_links"]
            z = RC.load_rel(p)
            sel = np.where(z["split"] == split)[0]
            if len(sel) == 0:
                z.close()
                continue
            X3 = z["v3"]
            X5 = RC.rel_X("on", "v5", z)                 # 145 (onset half)
            lh6 = z["lhr"]
            nf = lh6.shape[2]
            lh6 = np.broadcast_to(lh6[:, None, :, :],
                                  (lh6.shape[0], LH.HORIZON, lh6.shape[1], nf)).reshape(-1, nf)
            X5r = np.concatenate([X5, lh6], axis=1)      # 153 (ongoing half)
            P = LH.HORIZON * L
            cond = z["condition"]
            for j, wj in enumerate(sel):
                rr = np.arange(wj * P, (wj + 1) * P)
                wid = str(z["window_ids"][wj])
                t0 = int(z["time"][wj])
                ts = [t0 + 300 * (k + 1) for k in range(LH.HORIZON)]
                if cond[wj] == "queue_onset":
                    a = v3_on.predict(np.ascontiguousarray(X3[rr]))
                    b = m_on.predict(np.ascontiguousarray(X5[rr]))
                    prob = (w_on * a + (1 - w_on) * b).reshape(LH.HORIZON, L)
                    pred = np.zeros((LH.HORIZON, L), bool)
                    pred[LH.HORIZON - 1] = prob[LH.HORIZON - 1] >= on_th
                else:
                    a = v3_ong.predict(np.ascontiguousarray(X3[rr]))
                    b = m_ongr.predict(np.ascontiguousarray(X5r[rr]))
                    prob = (w_ong * a + (1 - w_ong) * b).reshape(LH.HORIZON, L)
                    pred = prob >= ong_th
                frames.append(pd.DataFrame({
                    "window_id": np.full(P, wid),
                    "timestamp": np.repeat(traffic.epoch_to_iso(np.asarray(ts, np.int64)), L),
                    "link_id": np.tile(st["links"], LH.HORIZON),
                    "queue_pred": pred.reshape(-1).astype(np.int8),
                }))
            del X3, X5, X5r, lh6
            z.close()
            gc.collect()
        pred = pd.concat(frames, ignore_index=True)
        tpl = tpl_all[tpl_all.window_id.isin(set(pred.window_id))].copy()
        out = tpl.merge(pred, on=["window_id", "timestamp", "link_id"], how="left",
                        validate="one_to_one", indicator=True)
        if int((out._merge != "both").sum()):
            raise RuntimeError("template rows without a prediction")
        out = out.drop(columns=["_merge"])
        out["queue_pred"] = out.queue_pred.astype(np.int8)
        out = out[["window_id", "timestamp", "link_id", "queue_pred"]]
        path = os.path.join(LH.OUT, f"queue_v5ramp_{split}.csv")
        out.to_csv(path, index=False)
        npos = int(out.queue_pred.sum())
        print(f"{split}: {len(out):,} rows, {npos:,} positives "
              f"({npos / len(out):.3%}) -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
