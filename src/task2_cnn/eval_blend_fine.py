"""Fine blend sweep for the onset CNN + shipped reference probability.

Loads a saved CNN checkpoint and the reference (v3+v5, w_on=0.3) probabilities
from ref_probs.npz, then sweeps the CNN blend weight on the January block,
reports the two honest protocols (February pseudo block / released June windows)
for the January-selected rule.

    python eval_blend_fine.py --ckpt models/cnn_xxx.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_cnn as T  # noqa: E402
import lhcore as LH  # noqa: E402

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", default="small")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panels = T.load_images()
    rel = T.load_released()
    ref = np.load(os.path.join(T.CACHE, "ref_probs.npz"), allow_pickle=False)
    tune_keys = T.block_keys(panels, "tune")
    feb_keys = T.block_keys(panels, "eval_feb")
    ref_tune, ref_feb = {}, {}
    for p in LH.PANELS:
        for block, store in (("tune", ref_tune), ("eval_feb", ref_feb)):
            idx = ref[f"{block}_{p}_idx"]
            p3, p5 = ref[f"{block}_{p}_prob3"], ref[f"{block}_{p}_prob5"]
            for j, r in enumerate(idx):
                store[(p, int(r))] = 0.3 * p3[j] + 0.7 * p5[j]
    rel_keys, ref_rel = [], []
    for p in LH.PANELS:
        sel = np.where((rel[p]["split"] == "train") & (rel[p]["condition"] == "queue_onset"))[0]
        p3, p5 = ref[f"rel_train_prob3_{p}"], ref[f"rel_train_prob5_{p}"]
        for j in range(len(sel)):
            rel_keys.append((p, int(sel[j])))
            ref_rel.append((0.3 * p3[j] + 0.7 * p5[j])[LH.HORIZON - 1])
    rel_truth, rel_elig = {}, {}
    for p in LH.PANELS:
        sel = np.where((rel[p]["split"] == "train") & (rel[p]["condition"] == "queue_onset"))[0]
        tr = ref[f"rel_train_truth_{p}"]
        el = ref[f"rel_train_elig_{p}"]
        for j in range(len(sel)):
            rel_truth[(p, int(sel[j]))] = tr[j]
            rel_elig[(p, int(sel[j]))] = el[j]

    c_in = panels[LH.PANELS[0]]["X"].shape[1]
    model = T.OnsetCNN(c_in=c_in, drop=args.drop, arch=args.arch).cuda()
    model.load_state_dict(torch.load(args.ckpt))
    tune_probs = T.predict_keys(model, panels, tune_keys)
    feb_probs = T.predict_keys(model, panels, feb_keys)
    rel_probs = T.predict_released(model, rel, "train", "queue_onset")
    feb_truth = {k: panels[k[0]]["truth"][k[1]] for k in feb_keys}
    feb_elig = {k: panels[k[0]]["elig"][k[1]] for k in feb_keys}
    tune_truth = {k: panels[k[0]]["truth"][k[1]] for k in tune_keys}
    tune_elig = {k: panels[k[0]]["elig"][k[1]] for k in tune_keys}

    # shipped reference at its own rule
    ref_B = float(np.mean([LH.iou(T.onset_pred(ref_feb[k], 0.30, 1), feb_truth[k], feb_elig[k])
                           for k in feb_keys]))
    ref_A = float(np.mean([LH.iou(T.onset_pred(ref_rel[j], 0.30, 1), rel_truth[k], rel_elig[k])
                           for j, k in enumerate(rel_keys)]))
    print(f"reference: B_feb {ref_B:.4f} | A_june {ref_A:.4f}")

    # fine blend sweep selected on January
    ths = np.round(np.arange(0.10, 0.86, 0.025), 3)
    rows = []
    for w in np.round(np.arange(0.0, 0.61, 0.05), 2):
        best_jan = None
        for th in ths:
            for fb in (0, 1):
                vals = [LH.iou(T.onset_pred(w * tune_probs[k] + (1 - w) * ref_tune[k], th, fb),
                               tune_truth[k], tune_elig[k]) for k in tune_keys]
                v = float(np.mean(vals))
                if best_jan is None or v > best_jan[0]:
                    best_jan = (v, float(th), int(fb))
        jan, th, fb = best_jan
        vb = float(np.mean([LH.iou(T.onset_pred(w * feb_probs[k] + (1 - w) * ref_feb[k], th, fb),
                                   feb_truth[k], feb_elig[k]) for k in feb_keys]))
        va = float(np.mean([LH.iou(T.onset_pred(w * rel_probs[k] + (1 - w) * ref_rel[j], th, fb),
                                   rel_truth[k], rel_elig[k]) for j, k in enumerate(rel_keys)]))
        rows.append({"w_cnn": w, "th": th, "fb": fb, "jan": jan,
                     "B_feb": vb, "A_june": va,
                     "B_delta": vb - ref_B, "A_delta": va - ref_A})
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    out = args.out or (os.path.splitext(os.path.basename(args.ckpt))[0] + "_blendfine.csv")
    df.to_csv(os.path.join(T.OUT, out), index=False)
    sel = df.iloc[df.jan.idxmax()]
    print("\nselected on January:", dict(sel.round(4)))
    print(f"ship bar: B_delta >= +0.01 and A_delta >= +0.01 -> "
          f"{'PASS' if (sel.B_delta >= 0.01 and sel.A_delta >= 0.01) else 'FAIL'}")


if __name__ == "__main__":
    main()
