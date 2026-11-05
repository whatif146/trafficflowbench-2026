"""Write the Task 2 queue submission for the long-history model.

    python predict_v5.py --feat v5 --tag queue_v5 --split both

Reads the frozen model artefacts (models/<feat>_onset.txt, <feat>_ongoing.txt,
<meta>.json with the decision rule) and the released-window feature caches
(cache/rel_<PANEL>.npz), predicts the six horizon steps for every released
window of the validation/private splits and writes exactly the template rows:

    window_id,timestamp,link_id,queue_pred   (binary 0/1)

Deterministic: no sampling, fixed LightGBM seed, feature caches on disk.
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

import lightgbm as lgb


def load_artifacts(feat: str):
    meta = json.load(open(os.path.join(LH.MODELS, f"{feat}_meta.json")))
    m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, f"{feat}_onset.txt"))
    m_ong = lgb.Booster(model_file=os.path.join(LH.MODELS, f"{feat}_ongoing.txt"))
    return meta, m_on, m_ong


def predict_split(split: str, feat: str, meta: dict, m_on, m_ong) -> pd.DataFrame:
    on_rule = meta["onset_rule"]
    ong_th = float(meta["ongoing_th"])
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
        X = RC.rel_X("on", feat, z)          # rel arrays are condition-agnostic
        P = LH.HORIZON * L
        prob = np.zeros((len(sel), LH.HORIZON, L), np.float32)
        cond = z["condition"]
        on_rows = np.where(cond[sel] == "queue_onset")[0]
        ong_rows = np.where(cond[sel] == "queue_ongoing")[0]
        if len(on_rows):
            rr = np.concatenate([np.arange(sel[j] * P, (sel[j] + 1) * P) for j in on_rows])
            prob[on_rows] = m_on.predict(np.ascontiguousarray(X[rr])).reshape(len(on_rows), LH.HORIZON, L)
        if len(ong_rows):
            rr = np.concatenate([np.arange(sel[j] * P, (sel[j] + 1) * P) for j in ong_rows])
            prob[ong_rows] = m_ong.predict(np.ascontiguousarray(X[rr])).reshape(len(ong_rows), LH.HORIZON, L)
        del X
        for j, wj in enumerate(sel):
            wid = str(z["window_ids"][wj])
            t0 = int(z["time"][wj])
            ts = [int(t0) + 300 * (k + 1) for k in range(LH.HORIZON)]
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
                "window_id": np.full(LH.HORIZON * L, wid),
                "timestamp": np.repeat(traffic_iso(ts), L),
                "link_id": np.tile(links, LH.HORIZON),
                "queue_pred": pred.reshape(-1).astype(np.int8),
            }))
        z.close()
    gc.collect()
    return pd.concat(rows, ignore_index=True)


def traffic_iso(epochs) -> np.ndarray:
    import traffic
    return traffic.epoch_to_iso(np.asarray(epochs, np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="v5")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--split", choices=["validation", "private", "both"], default="both")
    args = ap.parse_args()
    tag = args.tag or args.feat
    meta, m_on, m_ong = load_artifacts(args.feat)
    print(f"artefacts {args.feat}: onset rule {meta['onset_rule']} ongoing th {meta['ongoing_th']}")
    import traffic
    tpl_all = traffic.load_queue_template()
    tpl_all["window_id"] = tpl_all.window_id.astype(str)
    tpl_all["link_id"] = tpl_all.link_id.astype(str)
    tpl_all = tpl_all.drop(columns=["queue_pred"], errors="ignore")
    splits = ["validation", "private"] if args.split == "both" else [args.split]
    for split in splits:
        pred = predict_split(split, args.feat, meta, m_on, m_ong)
        win_ids = set(pred.window_id.unique())
        tpl = tpl_all[tpl_all.window_id.isin(win_ids)].copy()
        out = tpl.merge(pred, on=["window_id", "timestamp", "link_id"], how="left",
                        validate="one_to_one", indicator=True)
        miss = int((out._merge != "both").sum())
        if miss:
            raise RuntimeError(f"{split}: {miss} template rows without a prediction")
        out = out.drop(columns=["_merge"])
        out["queue_pred"] = out.queue_pred.astype(np.int8)
        out = out[["window_id", "timestamp", "link_id", "queue_pred"]]
        assert out.queue_pred.isin([0, 1]).all()
        assert not out.duplicated(["window_id", "timestamp", "link_id"]).any()
        path = os.path.join(LH.OUT, f"{tag}_{split}.csv")
        out.to_csv(path, index=False)
        npos = int(out.queue_pred.sum())
        print(f"{split}: {len(out):,} rows, {out.window_id.nunique()} windows, "
              f"{npos:,} positives ({npos / len(out):.3%}) -> {os.path.basename(path)}")
    print("done")


if __name__ == "__main__":
    main()
