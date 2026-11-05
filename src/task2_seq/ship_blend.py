"""Ship candidate: v3+v5 probability blend for Task 2 validation/private.

1. Refits the long-history (v5) models on every released train window (all fit
   block origins before 2031-02-01; the deployment may legally use all train
   months); the frozen shipped v3 boosters are reused as the other half of the
   blend.
2. Predicts every released validation/private window as
   p = w*p_v3 + (1-w)*p_v5, then applies the January-selected thresholds.
3. Writes exactly the template rows: window_id,timestamp,link_id,queue_pred.

All blend parameters (w, thresholds) come from the honest January selection in
run_final_cv.py; February/June are never used to choose them.
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
import lhcore as LH
import run_cv as RC
import fit_final as FF

import lightgbm as lgb


def load_v3_boosters():
    d = os.path.join(LH.WORK, "task2", "models", "v3")
    return (lgb.Booster(model_file=os.path.join(d, "gbm_onset.txt")),
            lgb.Booster(model_file=os.path.join(d, "gbm_ongoing.txt")))


def split_rows(split: str, feat: str, w_on: float, w_ong: float, on_rule, ong_th: float,
               m_on, m_ong, v3_on, v3_ong) -> pd.DataFrame:
    rows = []
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
        X5 = RC.rel_X("on", feat, z) if feat == "v5" else None
        P = LH.HORIZON * L
        cond = z["condition"]
        on_sel = sel[cond[sel] == "queue_onset"]
        ong_sel = sel[cond[sel] == "queue_ongoing"]
        prob = np.zeros((len(sel), LH.HORIZON, L), np.float32)
        if len(on_sel):
            rr = np.concatenate([np.arange(j * P, (j + 1) * P) for j in on_sel])
            a = v3_on.predict(np.ascontiguousarray(X3[rr])).reshape(len(on_sel), LH.HORIZON, L)
            b = m_on.predict(np.ascontiguousarray(X5[rr])).reshape(len(on_sel), LH.HORIZON, L)
            prob[np.isin(sel, on_sel)] = w_on * a + (1 - w_on) * b
        if len(ong_sel):
            rr = np.concatenate([np.arange(j * P, (j + 1) * P) for j in ong_sel])
            a = v3_ong.predict(np.ascontiguousarray(X3[rr])).reshape(len(ong_sel), LH.HORIZON, L)
            b = m_ong.predict(np.ascontiguousarray(X5[rr])).reshape(len(ong_sel), LH.HORIZON, L)
            prob[np.isin(sel, ong_sel)] = w_ong * a + (1 - w_ong) * b
        del X3, X5
        for j, wj in enumerate(sel):
            wid = str(z["window_ids"][wj])
            t0 = int(z["time"][wj])
            ts = [t0 + 300 * (k + 1) for k in range(LH.HORIZON)]
            if cond[wj] == "queue_onset":
                pred = np.zeros((LH.HORIZON, L), bool)
                kind, val = on_rule
                if kind == "th":
                    pred[LH.HORIZON - 1] = prob[j][LH.HORIZON - 1] >= float(val)
                else:
                    k = min(int(val), L)
                    pred[LH.HORIZON - 1, np.argsort(-prob[j][LH.HORIZON - 1])[:k]] = True
            else:
                pred = prob[j] >= ong_th
            rows.append(pd.DataFrame({
                "window_id": np.full(P, wid),
                "timestamp": np.repeat(LH.traffic.epoch_to_iso(np.asarray(ts, np.int64)), L),
                "link_id": np.tile(links, LH.HORIZON),
                "queue_pred": pred.reshape(-1).astype(np.int8),
            }))
        z.close()
        gc.collect()
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default=os.path.join(LH.OUT, "final_cv_stride1.json"))
    ap.add_argument("--feat", default="v5")
    ap.add_argument("--tag", default="queue_v5")
    ap.add_argument("--refit", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    cv = json.load(open(args.cv))
    w_on = float(cv["on"]["blend_B"]["w"])
    w_ong = float(cv["ong"]["blend_B"]["w"])
    on_rule = ("th", float(cv["on"]["blend_B"]["th"]))
    ong_th = float(cv["ong"]["blend_B"]["th"])
    print(f"blend w_on={w_on} w_ong={w_ong} onset rule={on_rule} ongoing th={ong_th}")
    if args.refit:
        m_on = FF.fit_condition("on", args.feat, RC.ON_ROUNDS)
        m_on.save_model(os.path.join(LH.MODELS, f"{args.feat}_onset.txt"))
        del m_on
        gc.collect()
        m_ong = FF.fit_condition("ong", args.feat, RC.ONG_ROUNDS)
        m_ong.save_model(os.path.join(LH.MODELS, f"{args.feat}_ongoing.txt"))
        del m_ong
        gc.collect()
    meta = {"blend_w_on": w_on, "blend_w_ong": w_ong,
            "onset_rule": list(on_rule), "ongoing_th": ong_th,
            "feat": args.feat, "v3_boosters": os.path.join(LH.WORK, "task2", "models", "v3"),
            "cv": os.path.basename(args.cv),
            "cv_on_B": cv["on"]["blend_B"], "cv_ong_B": cv["ong"]["blend_B"],
            "cv_on_A": cv["on"]["blend_A"], "cv_ong_A": cv["ong"]["blend_A"]}
    with open(os.path.join(LH.MODELS, f"{args.feat}_blend_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, f"{args.feat}_onset.txt"))
    m_ong = lgb.Booster(model_file=os.path.join(LH.MODELS, f"{args.feat}_ongoing.txt"))
    v3_on, v3_ong = load_v3_boosters()
    import traffic
    tpl_all = traffic.load_queue_template()
    tpl_all["window_id"] = tpl_all.window_id.astype(str)
    tpl_all["link_id"] = tpl_all.link_id.astype(str)
    tpl_all = tpl_all.drop(columns=["queue_pred"], errors="ignore")
    for split in ("validation", "private"):
        pred = split_rows(split, args.feat, w_on, w_ong, on_rule, ong_th,
                          m_on, m_ong, v3_on, v3_ong)
        win_ids = set(pred.window_id.unique())
        tpl = tpl_all[tpl_all.window_id.isin(win_ids)].copy()
        out = tpl.merge(pred, on=["window_id", "timestamp", "link_id"], how="left",
                        validate="one_to_one", indicator=True)
        if int((out._merge != "both").sum()):
            raise RuntimeError(f"{split}: template rows without a prediction")
        out = out.drop(columns=["_merge"])
        out["queue_pred"] = out.queue_pred.astype(np.int8)
        out = out[["window_id", "timestamp", "link_id", "queue_pred"]]
        assert not out.duplicated(["window_id", "timestamp", "link_id"]).any()
        path = os.path.join(LH.OUT, f"{args.tag}_{split}.csv")
        out.to_csv(path, index=False)
        npos = int(out.queue_pred.sum())
        print(f"{split}: {len(out):,} rows, {out.window_id.nunique()} windows, "
              f"{npos:,} positives ({npos / len(out):.3%}) -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
