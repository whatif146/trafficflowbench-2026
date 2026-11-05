"""Enrich deploy_meta_unet3.json with seeds, footprint-vs-2-seed and verification."""
from __future__ import annotations

import json
import os

HERE = r"C:\Users\lenovo\Desktop\kaggle\work\task1\unet"


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


meta = load("deploy_meta_unet3.json")
meta["seeds_requested"] = [45]
meta["seeds_used_actual"] = "seeds 43, 44, 45 (43/44 reused from their refit checkpoints, 45 trained now)"
meta["pred_suffix"] = "_u3"
meta["blend"] = ("state = 0.6 * GBM_shipped + 0.4 * mean(U-Net seeds 43, 44, 45); "
                 "GBM side = work/task1/gbm/state_gbm_<split>.csv shipped predictions, "
                 "no GBM refit")
meta["fit"] = "all 273 train days, 60 epochs x 35 steps/epoch (~2100 steps, same budget as seeds 43/44)"
meta["footprint_vs_2seed"] = {
    "validation": load("footprint_u3w06_val.json"),
    "private": load("footprint_u3w06_priv.json"),
    "note": ("mean_abs_dq_vph is raw vph summed over lanes; per-lane = /lanes "
             "(the deploy script reports 4.14 vph/lane for validation, 4.11 for private)"),
}
meta["footprint_w05_vs_2seed_w05"] = {
    "validation": load("footprint_u3w05_val.json"),
    "private": load("footprint_u3w05_priv.json"),
}
meta["verification"] = load("verify_unet3.json")
meta["files"] = {
    "w06": ["state_gbm_unet3_validation.csv", "state_gbm_unet3_private.csv"],
    "w05": ["state_gbm_unet3_w05_validation.csv", "state_gbm_unet3_w05_private.csv"],
    "untouched_fallback_2seed": ["state_gbm_unet_validation.csv", "state_gbm_unet_private.csv",
                                 "state_gbm_unet_w05_validation.csv",
                                 "state_gbm_unet_w05_private.csv",
                                 "state_gbm_unet_validation_s43.csv",
                                 "state_gbm_unet_private_s43.csv"],
}
meta["holdout_evidence"] = (
    "The 3rd seed is variance reduction only; the honest holdout evidence is the "
    "seed-averaged (43+44) table in deploy_meta.json: S_state 0.9423 vs GBM 0.9367, "
    "est_total +0.0028 (w=0.6) / +0.0031 (w=0.5), no panel regressing. The seed-43 "
    "w=0.6 variant probed on the LB at 0.83996 (+0.00262 vs 0.83734).")
json.dump(meta, open(os.path.join(HERE, "deploy_meta_unet3.json"), "w", encoding="utf-8"),
          indent=2)
print("meta keys:", list(meta.keys()))
print("rows:", {k: v["rows"] for k, v in meta["verification"].items()})
print("footprint val (3-seed vs 2-seed):", meta["footprint_vs_2seed"]["validation"]["mean"])
print("footprint priv (3-seed vs 2-seed):", meta["footprint_vs_2seed"]["private"]["mean"])
