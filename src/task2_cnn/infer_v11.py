"""Build the Task 2 v11 candidate (onset rows from the CNN, ongoing rows copied).

Kept for the record: the honest two-protocol CV concluded NO-SHIP (the CNN and
every blend of it lose to the shipped onset model), so this script was only run
in --dry-run mode to report the changed-cell footprint.

For each split:
  * onset rows: prediction = (w_cnn * p_cnn + (1-w_cnn) * p_ref) with the shipped
    rule (last horizon step only, threshold `th`, optional top-1 fallback);
    p_ref is the shipped v3+v5 blend probability (w_on = 0.3) from the frozen
    boosters in work/task2/models/v3 + work/task2_seq/models/v5_onset.txt;
  * every other onset cell is 0;
  * ongoing rows are copied verbatim from
    work/task2_seq/out/queue_w30ong40fb1_all_<split>.csv.

    python infer_v11.py --dry-run
    python infer_v11.py --write --w-cnn 0.25 --th 0.375 --fb 0
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\lenovo\Desktop\kaggle"
WORK = os.path.join(ROOT, "work")
SEQ = os.path.join(WORK, "task2_seq")
CNNDIR = os.path.join(WORK, "task2_cnn")
for _p in (os.path.join(WORK, "lib"), os.path.join(WORK, "task2"), SEQ, CNNDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lhcore as LH  # noqa: E402
import run_cv as RC  # noqa: E402
import train_cnn as T  # noqa: E402
import traffic  # noqa: E402

import lightgbm as lgb
import torch

BASE = os.path.join(SEQ, "out")
OUT = os.path.join(CNNDIR, "out")
os.makedirs(OUT, exist_ok=True)


def shipped_onset_probs(split: str):
    """Shipped v3+v5 blend probability for released onset windows of `split`."""
    v3_on, _ = (lgb.Booster(model_file=os.path.join(WORK, "task2", "models", "v3", "gbm_onset.txt")),
                None)
    v5_on = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_onset.txt"))
    out = {}
    for p in LH.PANELS:
        z = RC.load_rel(p)
        sel = np.where((z["split"] == split) & (z["condition"] == "queue_onset"))[0]
        if len(sel) == 0:
            z.close()
            continue
        L = int(z["n_links"])
        per = LH.HORIZON * L
        x3 = RC.rel_X("on", "v3", z)
        x5 = RC.rel_X("on", "v5", z)
        rows = (sel[:, None] * per + np.arange(per)[None, :]).reshape(-1)
        a = v3_on.predict(np.ascontiguousarray(x3[rows])).astype(np.float32).reshape(len(sel), LH.HORIZON, L)
        b = v5_on.predict(np.ascontiguousarray(x5[rows])).astype(np.float32).reshape(len(sel), LH.HORIZON, L)
        for j, r in enumerate(sel):
            out[(p, int(r))] = (0.3 * a[j] + 0.7 * b[j])[LH.HORIZON - 1]
        del x3, x5, a, b
        z.close()
        gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CNNDIR, "models", "cnn_aug1_iou.pt"))
    ap.add_argument("--arch", default="small")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--w-cnn", type=float, default=0.0,
                    help="CNN blend weight (0 = shipped reference behaviour)")
    ap.add_argument("--th", type=float, default=0.30)
    ap.add_argument("--fb", type=int, default=1)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rel = T.load_released()
    c_in = rel[LH.PANELS[0]]["X"].shape[1]
    model = T.OnsetCNN(c_in=c_in, drop=args.drop, arch=args.arch).cuda()
    model.load_state_dict(torch.load(args.ckpt))
    tpl = traffic.load_queue_template()
    tpl["window_id"] = tpl.window_id.astype(str)
    tpl["link_id"] = tpl.link_id.astype(str)
    summary = []
    for split in ("validation", "private"):
        refp = shipped_onset_probs(split)
        base = pd.read_csv(os.path.join(BASE, f"queue_w30ong40fb1_all_{split}.csv"))
        base["window_id"] = base.window_id.astype(str)
        base["link_id"] = base.link_id.astype(str)
        new = base.copy()
        w = args.w_cnn
        cnnp = T.predict_released(model, rel, split, "queue_onset")
        # onset prediction per window (last horizon step only)
        onset_pred_map = {}
        for p in LH.PANELS:
            sel = np.where((rel[p]["split"] == split) & (rel[p]["condition"] == "queue_onset"))[0]
            for i in sel:
                k = (p, int(i))
                prob = w * cnnp[k] + (1 - w) * refp[k]
                pred = prob >= args.th
                if args.fb and not pred.any():
                    pred[int(np.argmax(prob))] = True
                onset_pred_map[(p, int(i))] = pred
        # simpler: iterate windows via rel cache condition
        changed = 0
        for p in LH.PANELS:
            sel = np.where((rel[p]["split"] == split) & (rel[p]["condition"] == "queue_onset"))[0]
            for i in sel:
                wid = str(rel[p]["window_ids"][i])
                sub_idx = base.index[base.window_id == wid]
                if len(sub_idx) == 0:
                    continue
                sub = base.loc[sub_idx]
                ts_sorted = np.sort(sub.timestamp.unique())
                last = ts_sorted[-1]
                link_order = np.asarray(LH.static_panel(p)["links"], dtype=str)
                pred = onset_pred_map[(p, int(i))]
                pmap = {l: int(pred[j]) for j, l in enumerate(link_order)}
                for ii in sub_idx:
                    if base.at[ii, "timestamp"] != last:
                        continue
                    want = pmap.get(str(base.at[ii, "link_id"]).split(".")[0], 0)
                    if int(base.at[ii, "queue_pred"]) != want:
                        changed += 1
                    new.at[ii, "queue_pred"] = want
        if args.write:
            path = os.path.join(OUT, f"queue_v11_{split}.csv")
            new.to_csv(path, index=False)
            print(f"{split}: wrote {path} ({len(new):,} rows, changed cells {changed:,})")
        else:
            print(f"{split}: dry-run, would change {changed:,} of {len(new):,} cells "
                  f"(w_cnn={w}, th={args.th}, fb={args.fb})")
        summary.append({"split": split, "changed": int(changed), "rows": len(new)})
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "v11_footprint.csv"), index=False)


if __name__ == "__main__":
    main()
