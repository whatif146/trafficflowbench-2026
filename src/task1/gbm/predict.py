"""Task 1 deliverable: template-exact state predictions for train / validation / private.

Model: interp/smoother base + LightGBM residual (see t1gbm.py and NOTES.md).

    python predict.py --split validation --output work/task1/gbm/state_gbm_validation.csv
    python predict.py --split private    --output work/task1/gbm/state_gbm_private.csv
    python predict.py --split train      # per-panel files state_gbm_train_<PANEL>.csv

Deterministic (fixed seeds), memory-lean (one panel at a time, float32, plain
layer freed before the masked layer is loaded; nothing larger than ~1 GB).

Deployment model for validation/private: fitted on every published train target
row (subsample capped for memory).  The train-split output uses 2-fold
cross-fitting on contiguous day halves so every train row is predicted by a
model that never saw its day's truth (needed to calibrate Task 3 post-processing
honestly on train).
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import traffic  # noqa: E402

from t1lib import day_index, tod_index, dow_index, link_perm  # noqa: E402
from t1gbm import (build_profiles, per_day_station_means, per_link_clip_bounds,  # noqa: E402
                   FeatureBuilder, clip_predictions, LGB_PARAMS)
import lightgbm as lgb  # noqa: E402


def mem_gb():
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = M()
        st.dwLength = ctypes.sizeof(M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        return st.ullAvailPhys / 1e9
    except Exception:
        return float("nan")


def target_rows(panel: str, split: str):
    tg = traffic.load_targets(panel, split)
    return (tg["t_idx"].astype(np.int64), tg["s_idx"].astype(np.int64),
            tg["regime"].astype(np.int8) + 1)


def fit_channel_models(X, ys, yq, n_estimators, seed):
    params = dict(LGB_PARAMS)
    params["n_estimators"] = n_estimators
    params["seed"] = seed
    ms = lgb.LGBMRegressor(**params)
    ms.fit(X, ys)
    mq = lgb.LGBMRegressor(**params)
    mq.fit(X, yq)
    return ms, mq


def split_assets(panel: str, split: str, profiles, order_p, inv, attrs, n_estimators,
                 seed, model_pack=None):
    """Load a split's masked view, build features for its template rows, predict."""
    from t1gbm import FeatureBuilder as FB

    masked = traffic._load_or_build(panel, split, "masked")
    masked.pop("elig", None)
    masked.pop("regime", None)
    times = masked["times"]
    days = day_index(times)
    tod = tod_index(times)
    dow = dow_index(times)
    n_days = int(days.max()) + 1
    day_means = per_day_station_means(masked, days, n_days)
    ts, ss, reg = target_rows(panel, split)
    fb = FB(masked, profiles, day_means, attrs, order_p, inv, n_days, dow, tod)

    if model_pack is None:
        raise ValueError("model_pack required")
    ms, mq, lo_s, hi_s, hi_q = model_pack
    out_s = np.empty(len(ts), np.float64)
    out_q = np.empty(len(ts), np.float64)
    CH = 400000
    for lo in range(0, len(ts), CH):
        hi = min(len(ts), lo + CH)
        D = fb.build(ts[lo:hi], ss[lo:hi], reg[lo:hi])
        out_s[lo:hi] = D["base_s"] + ms.predict(D["X"])
        out_q[lo:hi] = D["base_q"] + mq.predict(D["X"])
        del D
        gc.collect()
    lanes = masked["lanes"][ss].astype(np.float64)
    free_speed = masked["free_speed"][ss].astype(np.float64)
    out_s, out_q = clip_predictions(out_s, out_q, free_speed, lo_s[ss], hi_s[ss], hi_q[ss])
    frame = pd.DataFrame({
        "panel": panel,
        "timestamp": traffic.epoch_to_iso(times[ts]),
        "station_id": masked["stations"][ss].astype(str),
        "link_id": masked["links"][ss].astype(str),
        "mask_regime": np.array(traffic.REGIME_NAMES)[reg - 1],
        "speed_kmh": out_s,
        "flow_vph": out_q,
    })
    del masked, day_means, fb
    gc.collect()
    return frame


def prepare_fit(panel: str, train_cap: int, seed: int, day_lo=None, model_params=None):
    """Fit the deployment model on train target rows (optionally only days < day_lo)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    plain = traffic._load_or_build(panel, "train", "plain")
    times = plain["times"]
    days = day_index(times)
    tod = tod_index(times)
    dow = dow_index(times)
    n_days = int(days.max()) + 1
    fit_day_t = np.ones(len(times), dtype=bool) if day_lo is None else days < day_lo
    tg = traffic.load_targets(panel, "train")
    ts_all = tg["t_idx"].astype(np.int64)
    ss_all = tg["s_idx"].astype(np.int64)
    reg_all = tg["regime"].astype(np.int8) + 1
    day_r = days[ts_all]
    fin = np.isfinite(plain["speed"][ts_all, ss_all]) & np.isfinite(plain["flow"][ts_all, ss_all])
    is_fit = (day_r < (n_days if day_lo is None else day_lo)) & fin
    profiles = build_profiles(plain, fit_day_t, dow, tod)
    lo_s, hi_s, hi_q = per_link_clip_bounds(plain, fit_day_t)
    idx = np.where(is_fit)[0]
    if len(idx) > train_cap:
        idx = np.sort(rng.choice(idx, size=train_cap, replace=False))
    true_s = plain["speed"][ts_all[idx], ss_all[idx]].astype(np.float64)
    true_q = plain["flow"][ts_all[idx], ss_all[idx]].astype(np.float64)
    del plain
    gc.collect()

    masked = traffic._load_or_build(panel, "train", "masked")
    masked.pop("elig", None)
    masked.pop("regime", None)
    order_p, inv = link_perm(masked)
    day_means = per_day_station_means(masked, days, n_days)
    attrs = {k: masked[k] for k in ("lanes", "free_speed", "capacity", "k_jam", "milepost", "length_km")}
    fb = FeatureBuilder(masked, profiles, day_means, attrs, order_p, inv, n_days, dow, tod)
    D = fb.build(ts_all[idx], ss_all[idx], reg_all[idx])
    params = dict(LGB_PARAMS)
    if model_params:
        params.update(model_params)
    params["seed"] = seed
    ms = lgb.LGBMRegressor(**params)
    ms.fit(D["X"], true_s - D["base_s"])
    mq = lgb.LGBMRegressor(**params)
    mq.fit(D["X"], true_q - D["base_q"])
    print(f"  [{panel}] fit rows={len(idx)} feat+fit {time.time()-t0:.1f}s "
          f"avail={mem_gb():.2f}GB", flush=True)
    pack = (ms, mq, lo_s, hi_s, hi_q)
    del D, fb, masked, day_means, true_s, true_q
    gc.collect()
    return {"pack": pack, "profiles": profiles, "order_p": order_p, "inv": inv,
            "attrs": attrs, "t0": t0}


def predict_split_with_pack(panel: str, split: str, assets: dict, n_estimators, seed):
    return split_assets(panel, split, assets["profiles"], assets["order_p"], assets["inv"],
                        assets["attrs"], n_estimators, seed, model_pack=assets["pack"])


def predict_split(panel: str, split: str, train_cap: int, seed: int, n_estimators: int):
    assets = prepare_fit(panel, train_cap, seed)
    frame = predict_split_with_pack(panel, split, assets, n_estimators, seed)
    del assets
    gc.collect()
    return frame


def train_crossfit_panel(panel: str, train_cap: int, seed: int, n_estimators: int,
                         out_path: str, model_params=None) -> int:
    """2-fold cross-fitted predictions for every train template row.

    Day halves [0, mid) and [mid, n_days); each half's rows are predicted by a
    model fitted only on the other half (profiles/clip bounds/labels from that
    half only), so no row is predicted in-sample.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    plain = traffic._load_or_build(panel, "train", "plain")
    times = plain["times"]
    days = day_index(times)
    tod = tod_index(times)
    dow = dow_index(times)
    n_days = int(days.max()) + 1
    mid = n_days // 2
    tg = traffic.load_targets(panel, "train")
    ts_all = tg["t_idx"].astype(np.int64)
    ss_all = tg["s_idx"].astype(np.int64)
    reg_all = tg["regime"].astype(np.int8) + 1
    day_r = days[ts_all]
    fin = np.isfinite(plain["speed"][ts_all, ss_all]) & np.isfinite(plain["flow"][ts_all, ss_all])
    masked = traffic._load_or_build(panel, "train", "masked")
    masked.pop("elig", None)
    masked.pop("regime", None)
    order_p, inv = link_perm(masked)
    attrs = {k: masked[k] for k in ("lanes", "free_speed", "capacity", "k_jam", "milepost", "length_km")}
    day_means = per_day_station_means(masked, days, n_days)
    out_s = np.full(len(ts_all), np.nan)
    out_q = np.full(len(ts_all), np.nan)
    for fold in (0, 1):
        held = (day_r >= mid) if fold == 0 else (day_r < mid)
        fit_rows = np.where(fin & ~held)[0]
        pred_rows = np.where(held)[0]
        if len(fit_rows) > train_cap:
            fit_rows = np.sort(rng.choice(fit_rows, size=train_cap, replace=False))
        fit_day_t = (days < mid) if fold == 0 else (days >= mid)
        profiles = build_profiles(plain, fit_day_t, dow, tod)
        lo_s, hi_s, hi_q = per_link_clip_bounds(plain, fit_day_t)
        true_s = plain["speed"][ts_all[fit_rows], ss_all[fit_rows]].astype(np.float64)
        true_q = plain["flow"][ts_all[fit_rows], ss_all[fit_rows]].astype(np.float64)
        fb = FeatureBuilder(masked, profiles, day_means, attrs, order_p, inv, n_days, dow, tod)
        D = fb.build(ts_all[fit_rows], ss_all[fit_rows], reg_all[fit_rows])
        params = dict(LGB_PARAMS)
        if model_params:
            params.update(model_params)
        params["seed"] = seed
        ms = lgb.LGBMRegressor(**params)
        ms.fit(D["X"], true_s - D["base_s"])
        mq = lgb.LGBMRegressor(**params)
        mq.fit(D["X"], true_q - D["base_q"])
        del D, true_s, true_q
        gc.collect()
        CH = 400000
        for lo in range(0, len(pred_rows), CH):
            rr = pred_rows[lo:lo + CH]
            D = fb.build(ts_all[rr], ss_all[rr], reg_all[rr])
            out_s[rr] = D["base_s"] + ms.predict(D["X"])
            out_q[rr] = D["base_q"] + mq.predict(D["X"])
            del D
            gc.collect()
        lanes = masked["lanes"][ss_all[pred_rows]].astype(np.float64)
        free_speed = masked["free_speed"][ss_all[pred_rows]].astype(np.float64)
        out_s[pred_rows], out_q[pred_rows] = clip_predictions(
            out_s[pred_rows], out_q[pred_rows], free_speed,
            lo_s[ss_all[pred_rows]], hi_s[ss_all[pred_rows]], hi_q[ss_all[pred_rows]])
        print(f"  [{panel}] crossfit fold {fold}: fit={len(fit_rows)} pred={len(pred_rows)} "
              f"{time.time()-t0:.1f}s avail={mem_gb():.2f}GB", flush=True)
        del fb, profiles, lo_s, hi_s, hi_q, ms, mq
        gc.collect()
    frame = pd.DataFrame({
        "panel": panel,
        "timestamp": traffic.epoch_to_iso(times[ts_all]),
        "station_id": masked["stations"][ss_all].astype(str),
        "link_id": masked["links"][ss_all].astype(str),
        "mask_regime": np.array(traffic.REGIME_NAMES)[reg_all - 1],
        "speed_kmh": out_s,
        "flow_vph": out_q,
    })
    frame.to_csv(out_path, index=False)
    print(f"  [{panel}] wrote {out_path} rows={len(frame)} total {time.time()-t0:.1f}s", flush=True)
    del plain, masked, day_means, frame
    gc.collect()
    return 0


def verify_frame(frame: pd.DataFrame, panel: str, split: str) -> None:
    """Compare keys against the published template (order and values)."""
    rel = os.path.join(traffic.REL, "task1", panel, split, "sample_submission_state.csv")
    tpl = pd.read_csv(rel, dtype={"timestamp": str, "station_id": str, "link_id": str,
                                  "panel": str, "mask_regime": str})
    if len(tpl) != len(frame):
        raise SystemExit(f"{panel} {split}: row count {len(frame)} != template {len(tpl)}")
    same = (
        (tpl["panel"].to_numpy() == frame["panel"].to_numpy()).all()
        and (tpl["timestamp"].to_numpy() == frame["timestamp"].to_numpy()).all()
        and (tpl["station_id"].to_numpy() == frame["station_id"].to_numpy()).all()
        and (tpl["link_id"].to_numpy() == frame["link_id"].to_numpy()).all()
        and (tpl["mask_regime"].to_numpy() == frame["mask_regime"].to_numpy()).all()
    )
    if not same:
        raise SystemExit(f"{panel} {split}: keys do not match the template order/values")
    print(f"  [{panel} {split}] template-exact ({len(frame)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["validation"],
                    choices=["train", "validation", "private"])
    ap.add_argument("--output", default=None,
                    help="output CSV for a single scored split (validation/private)")
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--train-cap", type=int, default=500000)
    ap.add_argument("--n-estimators", type=int, default=LGB_PARAMS["n_estimators"])
    ap.add_argument("--num-leaves", type=int, default=LGB_PARAMS["num_leaves"])
    ap.add_argument("--learning-rate", type=float, default=LGB_PARAMS["learning_rate"])
    ap.add_argument("--outprefix", default="state_gbm")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    panels = args.panels or traffic.PANELS
    t_all = time.time()
    model_params = {"n_estimators": args.n_estimators, "num_leaves": args.num_leaves,
                    "learning_rate": args.learning_rate}

    if "train" in args.split:
        for panel in panels:
            try:
                traffic._load_or_build(panel, "train", "masked")
            except FileNotFoundError:
                print(f"{panel}: no cache, skipped", flush=True)
                continue
            out = os.path.join(args.outdir, f"{args.outprefix}_train_{panel}.csv")
            train_crossfit_panel(panel, args.train_cap, args.seed, args.n_estimators, out,
                                 model_params=model_params)

    scored = [s for s in args.split if s in ("validation", "private")]
    if scored:
        if len(scored) == 1:
            out_paths = {scored[0]: args.output or os.path.join(args.outdir, f"{args.outprefix}_{scored[0]}.csv")}
        else:
            out_paths = {s: os.path.join(args.outdir, f"{args.outprefix}_{s}.csv") for s in scored}
        handles = {s: open(out_paths[s], "w", newline="", encoding="utf-8") for s in scored}
        written = {s: 0 for s in scored}
        try:
            for panel in panels:
                try:
                    traffic._load_or_build(panel, "train", "masked")
                except FileNotFoundError:
                    print(f"{panel}: no cache, skipped", flush=True)
                    continue
                assets = prepare_fit(panel, args.train_cap, args.seed, model_params=model_params)
                for s in scored:
                    frame = predict_split_with_pack(panel, s, assets, args.n_estimators, args.seed)
                    verify_frame(frame, panel, s)
                    # compact but lossless-enough output (speed 1e-3, flow 1e-2)
                    frame["speed_kmh"] = np.round(frame["speed_kmh"].to_numpy(), 3)
                    frame["flow_vph"] = np.round(frame["flow_vph"].to_numpy(), 2)
                    frame.to_csv(handles[s], header=(written[s] == 0), index=False)
                    handles[s].flush()
                    written[s] += len(frame)
                    del frame
                    gc.collect()
                del assets
                gc.collect()
        finally:
            for s in scored:
                handles[s].close()
        for s in scored:
            print(f"wrote {out_paths[s]} rows={written[s]}", flush=True)
    print(f"ALL DONE {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
