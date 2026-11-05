"""Public-notebook density post-processing for Task 3 (LB 0.87141 recipe) —
replicated exactly and evaluated honestly with the calibrated local proxy.

Recipe (work/research/lb-0-87141/cells.txt, cells 3/7/10):
  ok    = eligible & finite speed & flow          (released/masked observation layer)
  N     = (q/v)*L
  prev  = last ok step at or before t, nxt = next ok step at or after t (same link)
  N_i   = linear interpolation of N between (prev, nxt);  run = nxt-prev-1
  k_i   = N_i / L,  k_hat = q_hat / v_hat
  d     = w * (log k_i - log k_hat)                (0 where run > max_run)
  v_new = clip(v_hat * exp(-(1-alpha)*d), 1, 180)
  q_new = clip(q_hat * exp(alpha*d), 0, 150000)

Evaluation (honest, train holdout, shipped GBM `holdout_<PANEL>.csv`):
  dS_state   official Task-1 formula against the unmasked truth
  dS_FD/S_LWR/S_physics  calibrated local proxy (flux projected, window 1, jitter .06)
  est_LB = 0.35*dS_state + 0.15*(dS_FD/3 + 2*dS_LWR/3)      (notebook selection metric)

USAGE
-----
  python density_adjust.py --mode grid          # full grid on the two fast panels + confirmation
  python density_adjust.py --mode validate      # deployment check on submission_v4 state rows
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
for _p in (os.path.join(ROOT, "libs"), os.path.join(ROOT, "work", "lib"),
           os.path.join(ROOT, "work", "task3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import local_eval as le  # noqa: E402
import adjust as ad        # noqa: E402

GBM_DIR = os.path.join(ROOT, "work", "task1", "gbm")
WORK3 = os.path.join(ROOT, "work", "task3")
SUB = os.path.join(ROOT, "work", "submissions")

GRID_W = (0.25, 0.5, 0.75, 1.0)
GRID_ALPHA = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_RUN = (1, 2, 4, 9999)
JITTER = 0.06


# --------------------------------------------------------------------------- #
# notebook recipe
# --------------------------------------------------------------------------- #
def interp_density(pdata: dict, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`interp_N` from the notebook: linear N between nearest observed anchors.

    Anchors are eligible & finite speed/flow cells of the released (masked) layer.
    Returns N_i at (rows, cols) and the hidden-run length (nxt - prev - 1).
    """
    T = pdata["T"]
    v_obs = pdata["masked_speed"]
    q_obs = pdata["masked_flow"]
    ok = pdata["masked_elig"] & np.isfinite(v_obs) & np.isfinite(q_obs)
    L = pdata["p_arr"]["length_km"]
    N = np.where(ok, q_obs / np.maximum(v_obs, 1.0) * L[None, :], np.nan).astype(np.float64)
    pos = np.arange(T, dtype=np.int64)[:, None]
    prev = np.maximum.accumulate(np.where(ok, pos, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(ok, pos, T)[::-1], axis=0)[::-1]
    p = prev[rows, cols]
    n = nxt[rows, cols]
    has = (p >= 0) & (n < T)
    pp = np.maximum(p, 0)
    nn = np.minimum(n, T - 1)
    t = (rows - pp) / np.maximum(nn - pp, 1)
    Ni = N[pp, cols] * (1.0 - t) + N[nn, cols] * t
    return np.where(has, Ni, np.nan), np.where(has, nn - pp - 1, 10 ** 6)


def apply_density_correction(pdata: dict, cs: np.ndarray, cf: np.ndarray, cfg: dict):
    """Return corrected (speed, flow) arrays and per-config stats."""
    rows, cols = np.nonzero(np.isfinite(cs) & np.isfinite(cf))
    L = pdata["p_arr"]["length_km"].astype(np.float64)
    Ni, run = interp_density(pdata, rows, cols)
    v_hat = cs[rows, cols].astype(np.float64)
    q_hat = cf[rows, cols].astype(np.float64)
    k_hat = q_hat / np.maximum(v_hat, 1.0)
    k_i = Ni / L[cols]
    ok = np.isfinite(k_i) & (k_i > 0) & (k_hat > 0) & (run <= cfg["max_run"])
    d = np.where(ok, np.log(np.maximum(k_i, 1e-9)) - np.log(np.maximum(k_hat, 1e-9)), 0.0) * cfg["w"]
    vn = np.clip(v_hat * np.exp(-(1.0 - cfg["alpha"]) * d), 1, 180)
    qn = np.clip(q_hat * np.exp(cfg["alpha"] * d), 0, 150000)
    cs2 = cs.copy()
    cf2 = cf.copy()
    cs2[rows, cols] = vn.astype(np.float32)
    cf2[rows, cols] = qn.astype(np.float32)
    changed = int((np.abs(vn - v_hat) > 1e-6).sum() + (np.abs(qn - q_hat) > 1e-6).sum())
    stats = {"anchored_frac": float(np.isfinite(k_i).mean()) if len(k_i) else 0.0,
             "applied_frac": float(ok.mean()) if len(ok) else 0.0,
             "changed": changed, "cells": int(len(rows))}
    return cs2, cf2, stats


# --------------------------------------------------------------------------- #
# scoring helpers
# --------------------------------------------------------------------------- #
def load_holdout_panel(panel: str) -> tuple[dict, np.ndarray, np.ndarray]:
    pdata = le.load_panel(panel, "train")
    rows = le._read_submission_rows(os.path.join(GBM_DIR, f"holdout_{panel}.csv"), [panel], "train")
    cs, cf = le._rows_to_arrays(rows, pdata)
    return pdata, cs, cf


def metrics(pdata: dict, flux: dict, cs: np.ndarray, cf: np.ndarray) -> dict:
    st = ad.task1_state_scores(pdata, cs, cf)
    r = le.evaluate({pdata["panel"]: pdata}, lambda pd_: (cs, cf), {pdata["panel"]: flux})
    r = r[r.level == "panel"].iloc[0]
    return {"S_state": st["macro"], "S_FD": r.S_FD, "S_LWR": r.S_LWR,
            "S_physics": r.S_physics}


def evaluate_panel_grid(panel: str, cfgs: list[dict], verbose: bool = False) -> list[dict]:
    t0 = time.time()
    pdata, cs, cf = load_holdout_panel(panel)
    flux = le.build_frozen_flux(pdata, "projected", 1, flux_jitter=JITTER)
    base = metrics(pdata, flux, cs, cf)
    print(f"[{panel}] T={pdata['T']} L={pdata['L']} holdout cells={int(np.isfinite(cs).sum()):,} "
          f"base S_state={base['S_state']:.4f} S_LWR={base['S_LWR']:.4f} S_FD={base['S_FD']:.4f} "
          f"S_phys={base['S_physics']:.4f}", flush=True)
    out = []
    for cfg in cfgs:
        cs2, cf2, stats = apply_density_correction(pdata, cs, cf, cfg)
        m = metrics(pdata, flux, cs2, cf2)
        d = {k: m[k] - base[k] for k in base}
        est = 0.35 * d["S_state"] + 0.15 * (d["S_FD"] / 3.0 + 2.0 * d["S_LWR"] / 3.0)
        out.append({"panel": panel, **cfg,
                    **{f"S_{k}": m[f"S_{k}"] for k in ("state", "FD", "LWR", "physics")},
                    **{f"dS_{k}": d[f"S_{k}"] for k in ("state", "FD", "LWR", "physics")},
                    "est_LB": est, **stats})
    print(f"[{panel}] {len(cfgs)} configs in {time.time() - t0:.0f}s", flush=True)
    del pdata, cs, cf, flux
    return out


def grid_configs():
    return [{"w": w, "alpha": a, "max_run": int(r)}
            for w in GRID_W for a in GRID_ALPHA for r in GRID_RUN]


def summarise(rows: list[dict], label: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = (df.groupby(["w", "alpha", "max_run"], as_index=False)
             [["dS_state", "dS_FD", "dS_LWR", "dS_physics", "est_LB", "changed", "applied_frac"]]
             .mean())
    agg = agg.sort_values("est_LB", ascending=False).reset_index(drop=True)
    print(f"\n=== {label}: top 15 configs by notebook est_LB (macro over panels) ===")
    print(agg.head(15).round(5).to_string(index=False))
    return agg


# --------------------------------------------------------------------------- #
# deployment check on the scored splits (shipped state == submission_v4 rows,
# verified: submission_v4 state values match state_gbm_{validation,private}.csv
# to CSV rounding, so those files are used directly to avoid parsing 613 MB of key)
# --------------------------------------------------------------------------- #
def _load_shipped_state(split: str) -> dict:
    """Per-panel dense candidate arrays from the shipped GBM state file."""
    path = os.path.join(GBM_DIR, f"state_gbm_{split}.csv")
    meta = {p: le._panel_meta(p, split) for p in le._panel_order()}
    t_index = {}
    s_index = {}
    for p, m in meta.items():
        t_index[p] = pd.Series(np.arange(len(m["times"]), dtype=np.int64),
                               index=pd.DatetimeIndex(pd.to_datetime(
                                   m["times"].astype("int64"), unit="s", utc=True)))
        s_index[p] = pd.Index(m["stations"].astype(str))
    got = {p: {"ti": [], "si": [], "v": [], "q": []} for p in meta}
    n = 0
    for chunk in pd.read_csv(path, usecols=["panel", "timestamp", "station_id", "speed_kmh", "flow_vph"],
                             dtype={"panel": str, "timestamp": str, "station_id": str},
                             chunksize=1_000_000):
        for p, g in chunk.groupby("panel", sort=False):
            if p not in got:
                continue
            ti = pd.to_datetime(g["timestamp"], utc=True, errors="coerce").map(t_index[p]).to_numpy()
            si = s_index[p].get_indexer(g["station_id"].astype(str))
            ok = np.isfinite(ti) & (si >= 0)
            got[p]["ti"].append(ti[ok].astype(np.int64))
            got[p]["si"].append(si[ok].astype(np.int64))
            got[p]["v"].append(g["speed_kmh"].to_numpy()[ok].astype(np.float32))
            got[p]["q"].append(g["flow_vph"].to_numpy()[ok].astype(np.float32))
            n += int(ok.sum())
    print(f"[{split}] shipped state rows mapped: {n:,}", flush=True)
    out = {}
    for p, m in meta.items():
        T, L = len(m["times"]), len(m["stations"])
        cs = np.full((T, L), np.nan, np.float32)
        cf = np.full((T, L), np.nan, np.float32)
        b = got[p]
        if b["ti"]:
            ti = np.concatenate(b["ti"])
            si = np.concatenate(b["si"])
            cs[ti, si] = np.concatenate(b["v"])
            cf[ti, si] = np.concatenate(b["q"])
        out[p] = (cs, cf)
    return out


def _interp_flux(pdata: dict) -> dict:
    """Flux whose rhs equals the interp-filled observed conservation balance.

    On a scored split the hidden-cell observations are unavailable, so the only
    available rhs proxy is dN of the masked layer with its gaps linearly filled
    (exactly the notebook's interp_N density).  This is self-consistent with the
    correction, hence optimistic -- it is reported only as a deployment sanity
    check, never as evidence (train holdout is the evidence).
    """
    T, L = pdata["T"], pdata["L"]
    rhs_est = ad.estimate_rhs(pdata, "projected", 1)  # (T-1, L)
    on = pdata["ramps"]["on_flow"].astype(np.float32)
    off = pdata["ramps"]["off_flow"].astype(np.float32)
    inflow = np.zeros((T, L), np.float32)
    outflow = np.zeros((T, L), np.float32)
    dt = ad.DT_H
    inflow[:-1] = (rhs_est / dt + off[:-1] - on[:-1]).astype(np.float32)
    return {"kind": "projected", "inflow": inflow, "outflow": outflow}


def deployment_check(cfg: dict | None):
    rows = []
    for split in ("validation", "private"):
        shipped = _load_shipped_state(split)
        products = [(p, shipped[p][0], shipped[p][1]) for p in shipped]
        del shipped
        for panel, cs, cf in products:
            pdata = le.load_panel(panel, split)
            flux = _interp_flux(pdata)
            base = metrics(pdata, flux, cs, cf)
            out = {"split": split, "panel": panel, "cells": int(np.isfinite(cs).sum()),
                   "base_S_FD": base["S_FD"], "base_S_LWR_interp_proxy": base["S_LWR"],
                   "base_S_phys_interp_proxy": base["S_physics"]}
            if cfg is not None:
                cs2, cf2, stats = apply_density_correction(pdata, cs, cf, cfg)
                new = metrics(pdata, flux, cs2, cf2)
                out.update({"new_S_FD": new["S_FD"], "new_S_LWR_interp_proxy": new["S_LWR"],
                            "new_S_phys_interp_proxy": new["S_physics"],
                            "dS_FD": new["S_FD"] - base["S_FD"],
                            "dS_LWR_interp_proxy": new["S_LWR"] - base["S_LWR"],
                            "dS_phys_interp_proxy": new["S_physics"] - base["S_physics"],
                            "changed_frac": stats["changed"] / max(stats["cells"], 1),
                            "applied_frac": stats["applied_frac"]})
            rows.append(out)
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
            del pdata, cs, cf, flux
    df = pd.DataFrame(rows)
    path = os.path.join(WORK3, "reports", "density_deployment_check.csv")
    df.to_csv(path, index=False)
    if len(df):
        num = df.select_dtypes("number")
        print("\nmean over panels/splits:\n", num.mean().round(6).to_string())
    print(f"wrote {path}")
    return df


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="grid", choices=["grid", "validate", "apply"])
    ap.add_argument("--panels", default="D7_I10_E,D7_I210_W",
                    help="panels for the full grid (comma list)")
    ap.add_argument("--confirm-panels", default="D7_I405_S,D12_I5_N")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--cfg", default=None, help="forced config w:alpha:max_run for validate/apply")
    ap.add_argument("--min-est", type=float, default=0.001)
    args = ap.parse_args(argv)

    if args.mode == "grid":
        panels = [p for p in args.panels.split(",") if p.strip()]
        cfgs = grid_configs()
        rows = []
        for p in panels:
            rows += evaluate_panel_grid(p, cfgs)
        agg = summarise(rows, f"full grid on {panels}")
        agg.to_csv(os.path.join(WORK3, "reports", "density_grid.csv"), index=False)
        top = agg.head(args.top)
        conf_panels = [p for p in args.confirm_panels.split(",") if p.strip()]
        top_cfgs = [{"w": float(r.w), "alpha": float(r.alpha), "max_run": int(r.max_run)}
                    for r in top.itertuples()]
        for p in conf_panels:
            rows += evaluate_panel_grid(p, top_cfgs)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(WORK3, "reports", "density_grid_all.csv"), index=False)
        # overall ranking restricted to configs evaluated on every panel
        allp = df.groupby(["w", "alpha", "max_run"]).panel.nunique()
        full = set(allp[allp == len(panels) + len(conf_panels)].index)
        sub = df[df.set_index(["w", "alpha", "max_run"]).index.isin(full)]
        best = (sub.groupby(["w", "alpha", "max_run"], as_index=False)
                   [["dS_state", "dS_FD", "dS_LWR", "dS_physics", "est_LB", "changed", "applied_frac"]].mean()
                   .sort_values("est_LB", ascending=False))
        print("\n=== best configs evaluated on ALL panels ===")
        print(best.round(5).to_string(index=False))
        best.to_csv(os.path.join(WORK3, "reports", "density_best_all_panels.csv"), index=False)
        b = best.iloc[0]
        print(f"\nBEST: w={b.w} alpha={b.alpha} max_run={int(b.max_run)} est_LB={b.est_LB:+.5f} "
              f"(dS_state={b.dS_state:+.4f}, dS_physics={b.dS_physics:+.4f}, "
              f"changed per cell={b.changed:.0f}, applied_frac={b.applied_frac:.3f})")
        print(f"SHIP: {'YES' if b.est_LB >= args.min_est else 'NO'} (bar {args.min_est:+.4f})")
        return 0

    cfg = None
    if args.cfg:
        w, a, mr = args.cfg.split(":")
        cfg = {"w": float(w), "alpha": float(a), "max_run": int(mr)}
    deployment_check(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
