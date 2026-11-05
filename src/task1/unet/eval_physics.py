"""Task-3 physics proxy + combined objective for the U-Net blend.

Reuses the saved holdout predictions (`pred_unet_<panel>_<tag>.npz`) so the
proxy can be run/rerun without retraining.  The proxy is the task3 team's
frozen holdout proxy imported read-only from work/task3 (projected flux,
window 1, jitter 0.06), exactly as work/task1/gbm/run_density_eval.py does.

    python eval_physics.py --panels D7_I210_W D7_I405_S D7_I10_E D12_I5_N --tag u80
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "unet"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "gbm"))
sys.path.insert(0, os.path.join(ROOT, "work", "task3"))

import unetlib as U  # noqa: E402
from t1lib import score  # noqa: E402
from run_holdout import load_gbm_rows, physics, WEIGHTS, GBM_DIR  # noqa: E402

VARIANTS = ["gbm", "unet"] + [f"w{int(round(w * 100))}" for w in WEIGHTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*",
                    default=["D7_I210_W", "D7_I405_S", "D7_I10_E", "D12_I5_N"])
    ap.add_argument("--tag", default="u80")
    ap.add_argument("--jitter", type=float, default=0.06)
    ap.add_argument("--w-set", default="0.6,0.7,0.8,0.9")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    weights = [float(x) for x in args.w_set.split(",")]
    variants_all = ["gbm", "unet"] + [f"w{int(round(w * 100))}" for w in weights]
    rows = []
    for panel in args.panels:
        c = U.load_cache(panel, "train_hold60")
        pz = np.load(os.path.join(U.HERE, f"pred_unet_{panel}_{args.tag}.npz"))
        hold_rows = pz["rows"]
        v_un_all, q_un_all = pz["v"], pz["q"]
        df, tpl_pos, ps_gbm, pq_gbm = load_gbm_rows(
            panel, c, os.path.join(GBM_DIR, f"holdout_{panel}.csv"))
        tv, tq, lanes, reg = U.truth_rows(c, tpl_pos)
        ix = np.searchsorted(hold_rows, tpl_pos)
        v_un, q_un = v_un_all[ix], q_un_all[ix]
        variants = {"gbm": (ps_gbm, pq_gbm), "unet": (v_un, q_un)}
        for w in weights:
            variants[f"w{int(round(w * 100))}"] = (w * ps_gbm + (1 - w) * v_un,
                                                   w * pq_gbm + (1 - w) * q_un)
        ph = physics(df, panel, variants, args.jitter)
        for tag in variants_all:
            vs, vq = variants[tag]
            res = score(tv, tq, vs, vq, lanes, reg)
            ds = np.abs(np.asarray(vs) - np.asarray(ps_gbm))
            dq_rel = np.abs(np.asarray(vq) - np.asarray(pq_gbm)) / np.maximum(
                np.abs(np.asarray(pq_gbm)), 1.0)
            rows.append({
                "panel": panel, "variant": tag, "S_state": res["macro"],
                **{f"R{r}": res[r]["S_state"] for r in (1, 2, 3)},
                "rmse_speed": float(np.mean([res[r]["rmse_speed"] for r in (1, 2, 3)])),
                "rmse_flow_lane": float(np.mean([res[r]["rmse_flow"] for r in (1, 2, 3)])),
                **ph.get(tag, {}),
                "mean_abs_dv": float(np.mean(ds)),
                "p99_abs_dv": float(np.percentile(ds, 99)),
                "frac_dq_gt1pct": float(np.mean(dq_rel > 0.01)),
                "frac_dq_gt5pct": float(np.mean(dq_rel > 0.05)),
                "mean_abs_dq": float(np.mean(np.abs(np.asarray(vq) - np.asarray(pq_gbm)))),
            })
        del c
    d = pd.DataFrame(rows)
    d["est_total"] = 0.35 * d["S_state"] + 0.15 * d["S_physics"]
    out = args.out or os.path.join(U.HERE, f"holdout_physics_{args.tag}.csv")
    d.to_csv(out, index=False)
    print("\n=== S_state ===")
    print(d.pivot(index="panel", columns="variant", values="S_state").to_string(
        float_format=lambda v: f"{v:.4f}"))
    print("\n=== S_physics ===")
    print(d.pivot(index="panel", columns="variant", values="S_physics").to_string(
        float_format=lambda v: f"{v:.4f}"))
    print("\n=== est_total = 0.35*S_state + 0.15*S_physics ===")
    piv = d.pivot(index="panel", columns="variant", values="est_total")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n=== deltas vs shipped GBM ===")
    for tag in variants_all[1:]:
        de = (piv[tag] - piv["gbm"])
        dss = (d.pivot(index="panel", columns="variant", values="S_state")[tag]
               - d.pivot(index="panel", columns="variant", values="S_state")["gbm"])
        print(f"{tag:5s} mean d_est_total={de.mean():+.4f}  "
              f"per-panel={de.round(4).to_dict()}  "
              f"mean dS_state={dss.mean():+.4f} max_regress={max(0.0, -dss.min()):.4f}")
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
