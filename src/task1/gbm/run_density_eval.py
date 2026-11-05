"""Density-target variant of the Task 1 GBM (Task-3-aware experiment).

The official Task 3 conservation term is an L1 penalty on density/accumulation
error N = (q/v)*L at the Task-1 cells.  The shipped model predicts flow q with an
L2 residual; this experiment adds a third LightGBM that predicts density
k = q/v directly (same features/config as the shipped model) and reconstructs
flow as q_new = clip(k_hat * v_hat, 0, q_max), keeping the predicted speed.

Variants per panel (holdout protocol, same 400k rows / 700 trees as the v1
holdout numbers):
  a = shipped baseline      : q = q_flow,          v = v_hat
  b = density variant       : q = k_hat * v_hat,   v = v_hat
  c = 0.5 blend (untuned)   : q = 0.5 q_flow + 0.5 k_hat * v_hat

Scored with the official Task 1 formula (holdout rows) and the task3 team's
frozen holdout physics proxy (`work/task3/local_eval.py` + `adjust.py` imported,
not modified): observation-derived projected flux, window 1, jitter 0.06.

  python run_density_eval.py [--panels D7_I10_E ...] [--train-cap 400000]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task3")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import traffic  # noqa: E402

from t1lib import score, fmt  # noqa: E402
from t1gbm import LGB_PARAMS, clip_predictions  # noqa: E402
from run_eval import prepare, build_union, mem_gb  # noqa: E402
import lightgbm as lgb  # noqa: E402

import local_eval as le  # noqa: E402  (task3 frozen proxy, read-only import)
import adjust as ad  # noqa: E402


def per_link_kmax(panel, b, day_lo):
    """p99.9 observed density per link from the plain layer of fit days only."""
    plain = traffic._load_or_build(panel, "train", "plain")
    times = plain["times"]
    days = ((times - times[0]) // 86400).astype(np.int64)
    fit = days < day_lo
    with np.errstate(invalid="ignore", divide="ignore"):
        k = plain["flow"] / np.maximum(plain["speed"], 1.0)
    k = k[fit]
    with np.errstate(all="ignore"):
        kmax = np.nanpercentile(k, 99.9, axis=0).astype(np.float64)
    del plain
    gc.collect()
    return kmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*",
                    default=["D7_I10_E", "D12_I5_N", "D7_I405_S", "D7_I210_W"])
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--train-cap", type=int, default=400000)
    ap.add_argument("--eval-cap", type=int, default=800000)
    ap.add_argument("--n-estimators", type=int, default=700)
    ap.add_argument("--jitter", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows_out = []
    for panel in args.panels:
        t0 = time.time()
        b = prepare(panel, args.holdout_days, args.train_cap, args.eval_cap, args.seed)
        n_days = int(b["masked"]["times"].size) // 288
        day_lo = n_days - args.holdout_days
        kmax = per_link_kmax(panel, b, day_lo)
        F = build_union(b)
        nf = len(b["fit_idx"])
        ss_h = b["ss_all"][b["hold_idx"]]
        reg_h = b["reg_all"][b["hold_idx"]]
        lanes = b["masked"]["lanes"][ss_h].astype(np.float64)
        free_speed = b["masked"]["free_speed"][ss_h].astype(np.float64)
        k_base_fit = F["base_q_fit"] / np.maximum(F["base_s_fit"], 1e-3)
        k_base_hold = F["base_q_hold"] / np.maximum(F["base_s_hold"], 1e-3)
        with np.errstate(invalid="ignore", divide="ignore"):
            k_true_fit = b["true_q_fit"] / np.maximum(b["true_s_fit"], 1.0)
        ok = np.isfinite(k_true_fit) & np.isfinite(k_base_fit) & (b["true_s_fit"] > 0)
        params = dict(LGB_PARAMS)
        params.update({"n_estimators": args.n_estimators, "seed": args.seed})
        ms = lgb.LGBMRegressor(**params)
        ms.fit(F["X_fit"], b["true_s_fit"] - F["base_s_fit"])
        mq = lgb.LGBMRegressor(**params)
        mq.fit(F["X_fit"], b["true_q_fit"] - F["base_q_fit"])
        mk = lgb.LGBMRegressor(**params)
        mk.fit(F["X_fit"][ok], (k_true_fit - k_base_fit)[ok])
        print(f"  [{panel}] 3 models fit {time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)

        v_raw = F["base_s_hold"] + ms.predict(F["X_hold"])
        q_flow = F["base_q_hold"] + mq.predict(F["X_hold"])
        k_hat = k_base_hold + mk.predict(F["X_hold"])
        k_hat = np.clip(k_hat, 0.0, np.maximum(kmax[ss_h], 1e-6))
        # speed is unchanged from the shipped model: clip it with the shipped guard
        v_hat, _ = clip_predictions(v_raw, np.zeros_like(v_raw), free_speed,
                                    b["lo_s"][ss_h], b["hi_s"][ss_h], np.full_like(ss_h, 1e9, dtype=np.float64))
        hi_q = b["hi_q"][ss_h].astype(np.float64)
        q_dense = np.clip(k_hat * v_hat, 0.0, np.where(np.isfinite(hi_q), hi_q, 1e9))
        q_blend = np.clip(0.5 * q_flow + 0.5 * q_dense, 0.0,
                          np.where(np.isfinite(hi_q), hi_q, 1e9))
        q_flow = np.clip(q_flow, 0.0, np.where(np.isfinite(hi_q), hi_q, 1e9))

        variants = {"a": (v_hat, q_flow), "b": (v_hat, q_dense), "c": (v_hat, q_blend)}
        res = {}
        for tag, (ps, pq) in variants.items():
            res[tag] = score(b["true_s_hold"], b["true_q_hold"], ps, pq, lanes, reg_h)
            print("  [%s %s] %s" % (panel, tag, fmt(res[tag])), flush=True)
        # changed-cell footprint vs baseline (on the holdout cells)
        dq_abs = np.abs(q_dense - q_flow) / lanes
        dq_rel = np.abs(q_dense - q_flow) / np.maximum(np.abs(q_flow), 1.0)
        foot = {"frac_dq_rel_gt1pct": float(np.mean(dq_rel > 0.01)),
                "frac_dq_rel_gt5pct": float(np.mean(dq_rel > 0.05)),
                "mean_abs_dq_per_lane": float(np.mean(dq_abs))}
        print("  footprint b vs a: %.3f cells >1%% flow change, %.3f >5%%, mean|dq|/lane %.2f" % (
            foot["frac_dq_rel_gt1pct"], foot["frac_dq_rel_gt5pct"], foot["mean_abs_dq_per_lane"]),
            flush=True)

        # write holdout CSVs for the physics proxy (same rows as holdout_<panel>.csv)
        base_csv = os.path.join(HERE, f"holdout_{panel}.csv")
        df = pd.read_csv(base_csv)
        if len(df) != len(v_hat):
            raise SystemExit(f"{panel}: holdout csv rows {len(df)} != {len(v_hat)}")
        for tag, (ps, pq) in variants.items():
            out = df.copy()
            out["speed_kmh"] = np.round(ps, 3)
            out["flow_vph"] = np.round(pq, 2)
            out.to_csv(os.path.join(HERE, f"holdout_dense_{tag}_{panel}.csv"), index=False)
            del out
        del df, F
        gc.collect()
        print(f"  [{panel}] csvs written {time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)

        # ---- frozen task3 physics proxy on the holdout cells ----
        try:
            pdata = le.load_panel(panel, "train")
            flux = le.build_frozen_flux(pdata, "projected", 1, flux_jitter=args.jitter)
            for tag in ("a", "b", "c"):
                csv = os.path.join(HERE, f"holdout_dense_{tag}_{panel}.csv")
                srows = le._read_submission_rows(csv, [panel], "train")
                cs, cf = le._rows_to_arrays(srows, pdata)
                r = le.evaluate({panel: pdata}, lambda pd_: (cs, cf), {panel: flux})
                r = r[r.level == "panel"].iloc[0]
                res[tag]["S_FD"] = float(r.S_FD)
                res[tag]["S_LWR"] = float(r.S_LWR)
                res[tag]["S_physics"] = float(r.S_physics)
                res[tag]["est_total"] = 0.35 * res[tag]["macro"] + 0.15 * float(r.S_physics)
                print("  [%s %s] S_state=%.4f S_FD=%.4f S_LWR=%.4f S_physics=%.4f est_total=%.4f" % (
                    panel, tag, res[tag]["macro"], r.S_FD, r.S_LWR, r.S_physics, res[tag]["est_total"]),
                    flush=True)
            del pdata, flux
            gc.collect()
        except Exception as exc:  # proxy optional: report S_state alone if it fails
            print(f"  [{panel}] physics proxy failed: {exc!r}", flush=True)
            for tag in ("a", "b", "c"):
                res[tag]["S_FD"] = res[tag]["S_LWR"] = res[tag]["S_physics"] = float("nan")
                res[tag]["est_total"] = float("nan")

        for tag in ("a", "b", "c"):
            rows_out.append({"panel": panel, "variant": tag,
                             "S_state": res[tag]["macro"],
                             **{f"R{r}": res[tag][r]["S_state"] for r in (1, 2, 3)},
                             "rmse_speed": float(np.mean([res[tag][r]["rmse_speed"] for r in (1, 2, 3)])),
                             "rmse_flow_lane": float(np.mean([res[tag][r]["rmse_flow"] for r in (1, 2, 3)])),
                             "S_FD": res[tag].get("S_FD"), "S_LWR": res[tag].get("S_LWR"),
                             "S_physics": res[tag].get("S_physics"),
                             "est_total": res[tag].get("est_total"),
                             **foot})
        del b
        gc.collect()

    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(HERE, "density_variant_holdout.csv"), index=False)
    print("\n=== summary ===")
    piv = df.pivot(index="panel", columns="variant", values="est_total")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    if piv.shape[1] >= 2:
        d = (piv["b"] - piv["a"])
        print("\nS_state  :", df.pivot(index="panel", columns="variant", values="S_state").to_string(
            float_format=lambda v: f"{v:.4f}"))
        print("S_physics:", df.pivot(index="panel", columns="variant", values="S_physics").to_string(
            float_format=lambda v: f"{v:.4f}"))
        print("est_total delta (b-a):", d.round(4).to_dict(), " mean=%.4f" % d.mean())
        if "c" in piv:
            dc = (piv["c"] - piv["a"])
            print("est_total delta (c-a):", dc.round(4).to_dict(), " mean=%.4f" % dc.mean())
        ds = df.pivot(index="panel", columns="variant", values="S_state")
        print("S_state delta (b-a):", (ds["b"] - ds["a"]).round(4).to_dict(),
              " max regression=%.4f" % max(0.0, -(ds["b"] - ds["a"]).min()))
    print("wrote", os.path.join(HERE, "density_variant_holdout.csv"))


if __name__ == "__main__":
    main()
