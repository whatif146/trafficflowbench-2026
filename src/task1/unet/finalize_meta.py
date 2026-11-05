"""Add the evidence tables + honesty notes to deploy_meta.json (post-hoc enrichment).

    python finalize_meta.py --splitmeta deploy_meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\work\task1\unet"


def table(path, cols=("S_state", "S_physics", "est_total")):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    out = {}
    for variant, g in d.groupby("variant"):
        out[variant] = {c: round(float(g[c].mean()), 4) for c in cols if c in g}
        out[variant]["per_panel"] = {
            r.panel: {c: round(float(getattr(r, c)), 4) for c in cols if c in g}
            for r in g.itertuples()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="deploy_meta.json")
    args = ap.parse_args()
    p = os.path.join(HERE, args.out)
    meta = json.load(open(p, encoding="utf-8"))
    meta["gbm_source"] = ("work/task1/gbm/state_gbm_<split>.csv - the SHIPPED GBM state "
                          "predictions; no GBM was refit for the blend")
    meta["blend"] = ("state = w * GBM_shipped + (1 - w) * U-Net, w = %s, one global "
                     "weight (no per-panel tuning)" % meta.get("w_gbm"))
    meta["deploy_note"] = (
        "Deployed U-Net models are refit on all 273 train days with 60 epochs x 35 "
        "steps/epoch (~2100 optimizer steps, the same budget as the evaluated "
        "80 x 27). They are NOT the exact checkpoints behind the holdout table: "
        "that table comes from the separate protocol that fits days<213 only and "
        "predicts the last 60 days. The refit only adds train data.")
    meta["holdout_protocol"] = (
        "fit on train days < 213, predict the published template rows of days "
        "213-272 from that split's masked view only; GBM side = shipped holdout "
        "CSVs matched row-for-row (documented S_state reproduced to +-0.0000); "
        "score = t1lib.score (official Task-1 formula); Task-3 proxy = "
        "work/task3/local_eval.py + adjust.py imported read-only, projected flux, "
        "window 1, jitter 0.06; est_total = 0.35*S_state + 0.15*S_physics")
    meta["holdout_single_seed_42"] = table(os.path.join(HERE, "holdout_physics_u80.csv"))
    meta["holdout_seed_averaged"] = table(os.path.join(HERE, "holdout_physics_u80ens.csv"))
    meta["w_edge_check_single_seed"] = {
        "note": "S_state mean over the 4 holdout panels, single seed 42",
        "S_state": {"0.50": 0.9411, "0.55": 0.9410, "0.60": 0.9408, "0.65": 0.9406,
                    "0.70": 0.9403, "0.80": 0.9393},
        "decision": ("kept w=0.6: w=0.50/0.55 gain only +0.0003/+0.0002 S_state "
                     "(~+0.0001/+0.00007 est_total, below the +0.0002 switch rule), "
                     "and the physics term is flat or slightly better toward 0.7")}
    meta["seed_coverage"] = ("D12_I5_N holdout seeds 43/44 were lost to a machine "
                             "suspend; its holdout evidence is single-seed only")
    json.dump(meta, open(p, "w", encoding="utf-8"), indent=2)
    print("updated", p)
    print(json.dumps({k: v for k, v in meta.items()
                      if k in ("w_gbm", "seeds_requested", "epochs", "blend")}, indent=1))


if __name__ == "__main__":
    main()
