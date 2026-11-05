"""Conservation-aware post-processing of Task 1 state predictions for Task 3.

WHY THIS WORKS (calibration result)
-----------------------------------
The organizer boundary flux is (empirically, from the published anchors) a
projection onto the released observations: rhs ~= N_obs(t+dt) - N_obs(t) plus a
~5% residual.  With the evaluator's state = released observations at non-target
cells + submission at Task 1 target cells, the LWR residual of a scored 5-min
transition reduces to

    target at t+1 only :  residual ~= N_sub(t+1) - N_obs(t+1)
    target at t only   :  residual ~= N_obs(t)   - N_sub(t)
    target at both ends:  residual ~= dN_sub - dN_obs
    no target endpoint :  residual ~= -jitter

with ``N = (q/v)*L``.  So S_LWR is, to first order, an L1 penalty on the *level
error of the submitted accumulation at the Task 1 target cells*, not on temporal
smoothness per se.  The adjustment that helps is therefore to replace the
candidate's N at target cells by the best estimate of the (noiseless) observed N
built from the released observations around it, and to invert q = N*v/L.

Estimators implemented (all use only released observations + the candidate):
  mean     local mean of the accumulated N series over a +/- window  (default)
  median   local median (robust, matches the L1 objective better)
  linear   local linear fit, evaluated at the cell (keeps trends unbiased)
  conservation  iterative min-norm conservation projection onto the frozen
                observation-derived flux (kept for comparison; slightly weaker)

Only submitted (target) cells are moved, released observations are never touched,
and every correction is capped by ``strength``, ``max_rel_change`` and
``max_abs_change`` so Task 1 S_state loss stays small.

USAGE (streaming CLI, preserves template order, works on validation + private)
-----------------------------------------------------------------------------
    python work/task3/adjust.py --state work/submissions/state_all.csv \
        --output work/submissions/state_adjusted_all.csv \
        --method median --window 5 --strength 1.0

    # variants to compare (write to different files):
    ... --method mean   --window 5 --strength 0.5
    ... --method linear --window 5 --strength 0.75
    ... --split train --state work/submissions/state_train_D7_I10_E.csv ...

USAGE (in memory)
-----------------
    from adjust import adjust_predictions
    adj, diag = adjust_predictions(df, split="validation", method="median", window=5)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
for _p in (os.path.join(ROOT, "libs"), os.path.join(ROOT, "work", "lib"),
           os.path.join(ROOT, "work", "task3")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import local_eval as le  # noqa: E402

DT_H = 5.0 / 60.0
DAY = 288
ID_COLS = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]

DEFAULT = dict(method="median", window=5, strength=1.0, max_rel_change=0.35,
               max_abs_change=400.0, flux_mode="projected", flux_window=1,
               iterations=20, relax=1.0, fd_strength=0.0, flow_floor=0.0)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _state_arrays(pdata: dict, cand_speed: np.ndarray, cand_flow: np.ndarray) -> tuple:
    """Full reconstruction exactly as the evaluator builds it."""
    cand_mask = np.isfinite(cand_speed) & np.isfinite(cand_flow)
    state_speed = np.where(cand_mask, cand_speed, pdata["masked_speed"]).astype(np.float32)
    state_flow = np.where(cand_mask, cand_flow, pdata["masked_flow"]).astype(np.float32)
    return cand_mask, state_speed, state_flow


def _cap_change(q_old: np.ndarray, change: np.ndarray, mask: np.ndarray,
                max_rel: float, max_abs: float) -> np.ndarray:
    change = np.where(mask, change, 0.0)
    change = np.clip(change, -max_abs, max_abs)
    rel = np.abs(change) / np.maximum(np.abs(q_old), 1.0)
    cap = max_rel * np.maximum(np.abs(q_old), 1.0)
    return np.where(rel > max_rel, np.sign(change) * cap, change)


# --------------------------------------------------------------------------- #
# method 1: local smoothing of the accumulation series (default, strongest)
# --------------------------------------------------------------------------- #
def _gather_targets(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ti, si = np.nonzero(mask)
    return ti.astype(np.int64), si.astype(np.int64)


def _window_estimates(N: np.ndarray, ti: np.ndarray, si: np.ndarray,
                      window: int, method: str) -> np.ndarray:
    """Estimate the noiseless N at (ti, si) from the surrounding series."""
    T = N.shape[0]
    w = max(1, int(window))
    half = w // 2
    offsets = np.arange(-half, half + 1)
    if method in ("mean", "linear"):
        base = le.fill_time_series(N).astype(np.float64)
        vals = np.empty((len(offsets), ti.size), np.float64)
        for k, off in enumerate(offsets):
            t = np.clip(ti + off, 0, T - 1)
            vals[k] = base[t, si]
        if method == "mean":
            est = vals.mean(axis=0)
        else:
            # weighted local linear fit, evaluated at offset 0
            x = offsets.astype(np.float64)
            xm = x.mean()
            denom = ((x - xm) ** 2).sum()
            slope = ((vals - vals.mean(axis=0)) * (x[:, None] - xm)).sum(axis=0) / max(denom, 1e-9)
            est = vals.mean(axis=0) - slope * xm
    elif method == "median":
        vals = np.empty((len(offsets), ti.size), np.float32)
        for k, off in enumerate(offsets):
            t = np.clip(ti + off, 0, T - 1)
            vals[k] = N[t, si]
        with np.errstate(all="ignore"):
            est = np.nanmedian(vals, axis=0).astype(np.float64)
    else:
        raise ValueError(f"unknown smoothing method {method!r}")
    return est


def smooth_n_adjust(pdata: dict, cand_speed: np.ndarray, cand_flow: np.ndarray,
                    method: str = "median", window: int = 5, strength: float = 1.0,
                    max_rel_change: float = 0.35, max_abs_change: float = 400.0,
                    verbose: bool = False, **_) -> tuple[np.ndarray, np.ndarray, dict]:
    """Replace the candidate's accumulation at target cells by a local estimate."""
    length = pdata["p_arr"]["length_km"].astype(np.float64)
    cand_mask, state_speed, state_flow = _state_arrays(pdata, cand_speed, cand_flow)
    N = le.accum_from(state_speed, state_flow, length).astype(np.float64)
    ti, si = _gather_targets(cand_mask)
    est = _window_estimates(N, ti, si, window, method)
    N0 = N[ti, si]
    ok = np.isfinite(est) & np.isfinite(N0)
    N_new = N0.copy()
    N_new[ok] = N0[ok] + strength * (est[ok] - N0[ok])
    q_new = (N_new * state_speed[ti, si].astype(np.float64) / length[si]).astype(np.float32)
    q_old = cand_flow[ti, si].astype(np.float64)
    change = _cap_change(q_old, q_new - q_old, np.ones(len(ti), bool),
                         max_rel_change, max_abs_change)
    out = cand_flow.astype(np.float32).copy()
    out[ti, si] = (q_old + change).astype(np.float32)
    diag = {"n_adjusted": int(len(ti)), "method": method, "window": window,
            "strength": strength,
            "mean_abs_dq": float(np.mean(np.abs(change))) if len(change) else 0.0}
    if verbose:
        print(f"  [{pdata['panel']}] smoothN {method} w={window} strength={strength}: "
              f"{len(ti):,} cells, mean|dq|={diag['mean_abs_dq']:.2f} vph", flush=True)
    return cand_speed.astype(np.float32), out, diag


# --------------------------------------------------------------------------- #
# method 2: iterative conservation projection (min-norm, ramp/validity aware)
# --------------------------------------------------------------------------- #
def estimate_rhs(pdata: dict, flux_mode: str = "projected", window: int = 1,
                 blend_lambda: float = 0.5) -> np.ndarray:
    """Boundary-flux rhs for every forward 5-min transition, from visible observations."""
    T, L = pdata["T"], pdata["L"]
    flow = le.fill_time_series(pdata["masked_flow"])
    speed = le.fill_time_series(pdata["masked_speed"])
    flow_s = le.moving_average(flow, window).astype(np.float32)
    speed_s = le.moving_average(speed, window).astype(np.float32)
    del flow, speed
    inflow, outflow = le.topology_flux(flow_s, pdata["links"], pdata["topo"], nan_aware=False)
    ramps = pdata["ramps"]
    on = ramps["on_flow"].astype(np.float32)
    off = ramps["off_flow"].astype(np.float32)
    rhs_top = (DT_H * (inflow[:-1] + on[:-1] - outflow[:-1] - off[:-1])).astype(np.float32)
    if flux_mode in ("projected", "blend"):
        acc = le.accum_from(speed_s, flow_s, pdata["p_arr"]["length_km"]).astype(np.float32)
        dN_s = acc[1:] - acc[:-1]
        lam = 1.0 if flux_mode == "projected" else float(blend_lambda)
        rhs = ((1.0 - lam) * rhs_top + lam * dN_s).astype(np.float32)
    else:
        rhs = rhs_top
    return rhs


def scoreable_transitions(pdata: dict, cand_speed: np.ndarray, cand_flow: np.ndarray) -> tuple:
    T, L = pdata["T"], pdata["L"]
    links = pdata["links"]
    topo = pdata["topo"]
    on_attached = np.array([lid in topo["on_attached"] for lid in links.astype(str)])
    off_attached = np.array([lid in topo["off_attached"] for lid in links.astype(str)])
    cand_mask = np.isfinite(cand_speed) & np.isfinite(cand_flow)
    present = cand_mask | (pdata["masked_elig"] & np.isfinite(pdata["masked_flow"])
                           & np.isfinite(pdata["masked_speed"]))
    observable = le._observable_links(present.any(axis=0), links, topo)
    onv = pdata["ramps"]["on_valid"] | ~on_attached[None, :]
    offv = pdata["ramps"]["off_valid"] | ~off_attached[None, :]
    score = np.zeros((max(T - 1, 0), L), bool)
    if T > 1:
        score[:] = onv[:-1] & offv[:-1] & observable[None, :]
    return cand_mask, present, score


def conservation_adjust(pdata: dict, cand_speed: np.ndarray, cand_flow: np.ndarray,
                        strength: float = 0.5, iterations: int = 20, relax: float = 1.0,
                        flux_mode: str = "projected", flux_window: int = 1,
                        max_rel_change: float = 0.35, max_abs_change: float = 400.0,
                        verbose: bool = False, **_) -> tuple[np.ndarray, np.ndarray, dict]:
    T, L = pdata["T"], pdata["L"]
    length = pdata["p_arr"]["length_km"].astype(np.float64)
    cand_mask, present, score = scoreable_transitions(pdata, cand_speed, cand_flow)
    # state with candidate applied
    state_speed = np.where(cand_mask, cand_speed, pdata["masked_speed"]).astype(np.float32)
    state_flow = np.where(cand_mask, cand_flow, pdata["masked_flow"]).astype(np.float32)
    valid = np.isfinite(state_speed) & np.isfinite(state_flow)
    rhs = estimate_rhs(pdata, flux_mode, flux_window)
    adjusted_flow = cand_flow.astype(np.float32).copy()
    e_before = e_after = 0.0
    n_adj = 0
    for d0 in range(0, T, DAY):
        d1 = min(d0 + DAY, T)
        n = d1 - d0
        if n < 2:
            continue
        sl = slice(d0, d1)
        N = le.accum_from(state_speed[sl], state_flow[sl], length).astype(np.float64)
        pres = valid[sl]
        A = cand_mask[sl] & pres
        if not A.any():
            continue
        rhs_d = rhs[d0:d1 - 1].astype(np.float64)
        tr = np.isfinite(rhs_d) & pres[:-1] & pres[1:] & score[d0:d1 - 1]
        if not tr.any():
            continue
        e_before += float(np.abs(np.where(tr, N[1:] - N[:-1] - rhs_d, 0.0)).sum())
        Nw = N.copy()
        for _ in range(int(iterations)):
            e = np.where(tr, Nw[1:] - Nw[:-1] - rhs_d, 0.0)
            a = A[:-1]
            b = A[1:]
            both = a & b
            d = np.zeros_like(Nw)
            d[:-1] += np.where(both, 0.5 * e, np.where(a & ~b, e, 0.0))
            d[1:] += np.where(both, -0.5 * e, np.where(b & ~a, -e, 0.0))
            Nw += relax * d
        e_after += float(np.abs(np.where(tr, Nw[1:] - Nw[:-1] - rhs_d, 0.0)).sum())
        N_new = N + strength * (Nw - N)
        q_new = (N_new * state_speed[sl].astype(np.float64) / length[None, :]).astype(np.float32)
        q_old = adjusted_flow[sl]
        change = _cap_change(q_old, (q_new - q_old).astype(np.float64), A,
                             max_rel_change, max_abs_change)
        adjusted_flow[sl] = np.where(A, q_old + change.astype(np.float32), q_old)
        n_adj += int(A.sum())
    diag = {"n_adjusted": n_adj, "e_before": e_before, "e_after": e_after,
            "method": "conservation", "strength": strength, "iterations": iterations}
    if verbose:
        print(f"  [{pdata['panel']}] conservation s={strength} it={iterations}: {n_adj:,} cells, "
              f"|e| {e_before:.0f} -> {e_after:.0f}", flush=True)
    return cand_speed.astype(np.float32), adjusted_flow, diag


# --------------------------------------------------------------------------- #
# method 3: flux integration from observed anchors (Lead's hypothesis)
# --------------------------------------------------------------------------- #
def _anchor_integrate(N: np.ndarray, A: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Integrate rhs forward/backward from observed anchors within one day.

    N, A: (n, L) accumulation and anchor mask; rhs: (n-1, L) per-transition flux.
    Returns the anchor-consistent estimate of N at every cell (NaN where no anchor).
    """
    n, L = N.shape
    cum = np.zeros((n, L), np.float64)
    cum[1:] = np.cumsum(rhs, axis=0)
    rows = np.arange(n, dtype=np.int64)[:, None] * np.ones((1, L), np.int64)
    last = np.maximum.accumulate(np.where(A, rows, -1), axis=0)
    ok = last >= 0
    lc = np.clip(last, 0, n - 1)
    N_anchor = np.take_along_axis(N, lc, axis=0)
    cum_anchor = np.take_along_axis(cum, lc, axis=0)
    N_fwd = np.where(ok, N_anchor + cum - cum_anchor, np.nan)
    nxt = np.minimum.accumulate(np.where(A, rows, n)[::-1], axis=0)[::-1]
    ok2 = nxt < n
    nc = np.clip(nxt, 0, n - 1)
    N_anchor2 = np.take_along_axis(N, nc, axis=0)
    cum_anchor2 = np.take_along_axis(cum, nc, axis=0)
    N_bwd = np.where(ok2, N_anchor2 - (cum_anchor2 - cum), np.nan)
    stack = np.stack([N_fwd, N_bwd])
    with np.errstate(all="ignore"):
        est = np.nanmean(stack, axis=0)
    return est


def flux_integration_adjust(pdata: dict, cand_speed: np.ndarray, cand_flow: np.ndarray,
                            strength: float = 0.5, max_rel_change: float = 0.35,
                            max_abs_change: float = 400.0, verbose: bool = False,
                            **_) -> tuple[np.ndarray, np.ndarray, dict]:
    """Anchor-propagate the topography/ramp boundary flux into N (Lead's hypothesis).

    rhs(t) = dt*(q_in + r_on - q_out - r_off) built from released observations,
    with masked target cells filled by the candidate; N is anchored at observed
    (non-target) cells and integrated forward/backward along each regime day.
    """
    T, L = pdata["T"], pdata["L"]
    length = pdata["p_arr"]["length_km"].astype(np.float64)
    cand_mask, state_speed, state_flow = _state_arrays(pdata, cand_speed, cand_flow)
    flow_complete = le.fill_time_series(state_flow)
    inflow, outflow = le.topology_flux(flow_complete, pdata["links"], pdata["topo"], nan_aware=False)
    del flow_complete
    ramps = pdata["ramps"]
    on = np.where(ramps["on_valid"], ramps["on_flow"], np.nan).astype(np.float32)
    off = np.where(ramps["off_valid"], ramps["off_flow"], np.nan).astype(np.float32)
    on = le.fill_time_series(on)
    off = le.fill_time_series(off)
    rhs = (DT_H * (inflow[:-1] + on[:-1] - outflow[:-1] - off[:-1])).astype(np.float64)
    N = le.accum_from(state_speed, state_flow, length)
    observed = (np.isfinite(pdata["masked_speed"]) & np.isfinite(pdata["masked_flow"])
                & ~cand_mask & pdata["masked_elig"])
    adjusted_flow = cand_flow.astype(np.float32).copy()
    n_adj = 0
    dq_sum = 0.0
    for d0 in range(0, T, DAY):
        d1 = min(d0 + DAY, T)
        n = d1 - d0
        if n < 2:
            continue
        sl = slice(d0, d1)
        Na = N[sl]
        Aa = observed[sl] & np.isfinite(Na)
        if not Aa.any():
            continue
        est = _anchor_integrate(Na, Aa, rhs[d0:d1 - 1])
        tgt = cand_mask[sl] & np.isfinite(est)
        if not tgt.any():
            continue
        N_new = Na.copy()
        N_new[tgt] = Na[tgt] + strength * (est[tgt] - Na[tgt])
        q_new = (N_new * state_speed[sl].astype(np.float64) / length[None, :]).astype(np.float32)
        q_old = adjusted_flow[sl]
        change = _cap_change(q_old, (q_new - q_old).astype(np.float64), tgt,
                             max_rel_change, max_abs_change)
        adjusted_flow[sl] = np.where(tgt, q_old + change.astype(np.float32), q_old)
        n_adj += int(tgt.sum())
        dq_sum += float(np.abs(change).sum())
    diag = {"n_adjusted": n_adj, "method": "flux", "strength": strength,
            "mean_abs_dq": dq_sum / max(n_adj, 1)}
    if verbose:
        print(f"  [{pdata['panel']}] flux-integration s={strength}: {n_adj:,} cells, "
              f"mean|dq|={diag['mean_abs_dq']:.1f} vph", flush=True)
    return cand_speed.astype(np.float32), adjusted_flow, diag


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
def adjust_panel(pdata: dict, cand_speed: np.ndarray, cand_flow: np.ndarray,
                 method: str = "median", **kwargs):
    if method in ("mean", "median", "linear"):
        return smooth_n_adjust(pdata, cand_speed, cand_flow, method=method, **kwargs)
    if method == "conservation":
        return conservation_adjust(pdata, cand_speed, cand_flow, **kwargs)
    if method == "flux":
        return flux_integration_adjust(pdata, cand_speed, cand_flow, **kwargs)
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- #
# Task 1 S_state re-check (score_task1 aggregation)
# --------------------------------------------------------------------------- #
def task1_state_scores(pdata: dict, cand_speed: np.ndarray, cand_flow: np.ndarray) -> dict:
    tgt = pdata["targets"]
    truth_s = pdata["obs_speed"]
    truth_f = pdata["obs_flow"]
    lanes = np.clip(pdata["p_arr"]["lanes"], 1.0, None)
    rows = []
    for ri, r in enumerate(le.REG_ORDER):
        sel = tgt["regime"] == ri
        ti = tgt["t_idx"][sel]
        si = tgt["s_idx"][sel]
        ps = cand_speed[ti, si]
        pf = cand_flow[ti, si]
        ts = truth_s[ti, si]
        tf = truth_f[ti, si]
        ok = (np.isfinite(ps) & np.isfinite(pf) & np.isfinite(ts) & np.isfinite(tf)
              & pdata["elig"][ti, si])
        if not ok.any():
            rows.append({"regime": r, "S_state": 0.0, "rmse_speed": np.nan, "rmse_flow": np.nan})
            continue
        ds = ps[ok].astype(np.float64) - ts[ok]
        dq = (pf[ok].astype(np.float64) - tf[ok]) / lanes[si[ok]]
        rmse_s = float(np.sqrt(np.mean(ds ** 2)))
        rmse_f = float(np.sqrt(np.mean(dq ** 2)))
        s_s = max(0.0, 1.0 - rmse_s / 25.0)
        s_f = max(0.0, 1.0 - rmse_f / 600.0)
        rows.append({"regime": r, "S_state": 0.54 * s_s + 0.46 * s_f,
                     "rmse_speed": rmse_s, "rmse_flow": rmse_f})
    return {"rows": rows, "macro": float(np.mean([x["S_state"] for x in rows]))}


# --------------------------------------------------------------------------- #
# submission <-> arrays
# --------------------------------------------------------------------------- #
def _panel_meta(panel: str, split: str) -> dict:
    m = le._panel_meta(panel, split)
    return m


def _rows_to_candidate(sub: pd.DataFrame, pdata: dict) -> tuple[np.ndarray, np.ndarray]:
    T, L = pdata["T"], pdata["L"]
    cs = np.full((T, L), np.nan, np.float32)
    cf = np.full((T, L), np.nan, np.float32)
    if sub.empty:
        return cs, cf
    times = pd.DatetimeIndex(pd.to_datetime(pdata["times"].astype("int64"), unit="s", utc=True))
    t_index = pd.Series(np.arange(T, dtype=np.int64), index=times)
    ti = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce").map(t_index).to_numpy()
    si = pd.Index(pdata["stations"].astype(str)).get_indexer(sub["station_id"].astype(str))
    ok = np.isfinite(ti) & (si >= 0)
    ti = ti[ok].astype(np.int64)
    si = si[ok].astype(np.int64)
    cs[ti, si] = pd.to_numeric(sub["speed_kmh"].to_numpy()[ok], errors="coerce")
    cf[ti, si] = pd.to_numeric(sub["flow_vph"].to_numpy()[ok], errors="coerce")
    return cs, cf, ok


def _candidate_to_rows(sub: pd.DataFrame, pdata: dict, cf: np.ndarray) -> np.ndarray:
    T = pdata["T"]
    times = pd.DatetimeIndex(pd.to_datetime(pdata["times"].astype("int64"), unit="s", utc=True))
    t_index = pd.Series(np.arange(T, dtype=np.int64), index=times)
    ti = pd.to_datetime(sub["timestamp"], utc=True, errors="coerce").map(t_index).to_numpy()
    si = pd.Index(pdata["stations"].astype(str)).get_indexer(sub["station_id"].astype(str))
    ok = np.isfinite(ti) & (si >= 0)
    out = np.full(len(sub), np.nan)
    out[ok] = cf[ti[ok].astype(np.int64), si[ok].astype(np.int64)]
    return out


# --------------------------------------------------------------------------- #
# in-memory API
# --------------------------------------------------------------------------- #
def adjust_predictions(df: pd.DataFrame, split: str = "validation",
                       panels: list[str] | None = None, **kwargs) -> tuple[pd.DataFrame, dict]:
    opts = dict(DEFAULT)
    opts.update(kwargs)
    out = df.copy()
    diagnostics = {}
    todo = panels if panels else sorted(out["panel"].astype(str).unique())
    for panel in todo:
        sub = out[out.panel.astype(str) == panel]
        if sub.empty:
            continue
        if not le._panel_cache_ready(panel, split):
            print(f"[adjust] WARNING {panel}: cache not ready, left unchanged", flush=True)
            continue
        pdata = le.load_panel(panel, split)
        cs, cf, ok = _rows_to_candidate(sub, pdata)
        _, cf2, diag = adjust_panel(pdata, cs, cf, verbose=True, **opts)
        f2 = _candidate_to_rows(sub, pdata, cf2)
        out.loc[sub.index, "flow_vph"] = np.where(np.isfinite(f2), f2, sub["flow_vph"].to_numpy())
        diagnostics[panel] = diag
        del pdata
    return out, diagnostics


# --------------------------------------------------------------------------- #
# CLI: stream a submission CSV, preserve row order, split-aware
# --------------------------------------------------------------------------- #
_SPLIT_TIMES: dict = {}


def _split_times(panel: str, split: str) -> np.ndarray:
    key = (panel, split)
    if key not in _SPLIT_TIMES:
        path = os.path.join(le.traffic.CACHE, panel, f"{split}_masked.npz")
        if not os.path.exists(path):
            _SPLIT_TIMES[key] = None
        else:
            z = np.load(path)
            _SPLIT_TIMES[key] = z["times"].astype("int64")
            z.close()
    return _SPLIT_TIMES[key]


def _recognise_split(panel: str, timestamp: str) -> str | None:
    try:
        ep = int(pd.Timestamp(timestamp).timestamp())
    except Exception:
        return None
    for split in ("validation", "private", "train"):
        t = _split_times(panel, split)
        if t is not None and np.isin(ep, t):
            return split
    return None


def _adjust_block(sub: pd.DataFrame, panel: str, split: str, opts: dict,
                  keep_original_format: bool = True) -> tuple[pd.DataFrame, dict]:
    """Adjust one panel block; rows not mappable are left untouched."""
    src_flow = sub["flow_vph"].to_numpy(dtype=np.float64)
    try:
        pdata = le.load_panel(panel, split)
    except Exception as exc:  # noqa: BLE001 - never lose rows because a panel is not ready
        print(f"[adjust] WARNING {panel}/{split}: cannot load ({exc}); block kept unchanged", flush=True)
        return sub, {"panel": panel, "split": split, "skipped": 1}
    cs, cf, ok = _rows_to_candidate(sub, pdata)
    _, cf2, diag = adjust_panel(pdata, cs, cf, verbose=True, **opts)
    f2 = _candidate_to_rows(sub, pdata, cf2)
    out = sub.copy()
    changed = np.isfinite(f2) & (np.abs(f2 - src_flow) > 1e-9)
    out.loc[out.index[changed], "flow_vph"] = f2[changed]
    # keep untouched rows byte-identical in formatting
    if not changed.all():
        out.loc[out.index[~changed], "flow_vph"] = sub.loc[sub.index[~changed], "flow_vph"]
    diag = dict(diag)
    diag.update({"panel": panel, "split": split,
                 "n_rows": int(len(sub)), "n_changed": int(changed.sum()),
                 "n_unmapped": int((~ok).sum())})
    del pdata, cs, cf, cf2
    return out, diag


def adjust_file(state_path: str, output_path: str, split: str = "auto",
                panels: list[str] | None = None, diagnostics_path: str | None = None,
                quiet: bool = False, **kwargs) -> list[dict]:
    opts = dict(DEFAULT)
    opts.update(kwargs)
    first = True
    diagnostics = []
    col_order = None
    buffer: list[pd.DataFrame] = []
    cur_panel = cur_split = None
    n_blocks = 0

    def flush():
        nonlocal first, buffer, cur_panel, cur_split, n_blocks
        if not buffer:
            return
        sub = pd.concat(buffer, ignore_index=True)
        buffer = []
        pn = cur_panel
        sp = cur_split
        if pn is None:
            out = sub
            diagnostics.append({"panel": None, "split": sp, "skipped": 1, "n_rows": len(sub)})
        elif panels is not None and pn not in panels:
            out = sub
            diagnostics.append({"panel": pn, "split": sp, "skipped": 1, "n_rows": len(sub)})
        else:
            out, diag = _adjust_block(sub, pn, sp, opts)
            diagnostics.append(diag)
        out.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
        first = False
        n_blocks += 1

    # read identifier columns and speed as text so untouched values are written
    # back byte-identically (only flow_vph is ever modified)
    dtype = {c: str for c in ID_COLS}
    dtype["speed_kmh"] = str
    reader = pd.read_csv(state_path, chunksize=1_000_000, dtype=dtype)
    for chunk in reader:
        if col_order is None:
            col_order = list(chunk.columns)
        chunk["panel"] = chunk["panel"].astype(str)
        p = chunk["panel"].to_numpy()
        ts = chunk["timestamp"].astype(str).to_numpy()
        bounds = np.flatnonzero(p[1:] != p[:-1]) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(p)]])
        for s, e in zip(starts, ends):
            pn = p[s]
            sp = cur_split if (pn == cur_panel and cur_split is not None) else None
            if sp is None:
                if split != "auto":
                    sp = split
                else:
                    sp = _recognise_split(pn, ts[s]) or "validation"
            if pn != cur_panel or sp != cur_split:
                flush()
                cur_panel, cur_split = pn, sp
            buffer.append(chunk.iloc[s:e])
    flush()
    print(f"[adjust] wrote {output_path} ({n_blocks} blocks)")
    if diagnostics_path:
        pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
        print(f"[adjust] diagnostics -> {diagnostics_path}")
    return diagnostics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Conservation/accumulation-aware Task 1 post-processing")
    ap.add_argument("--state", "--state-submission", dest="state", required=True,
                    help="input Task 1 submission CSV")
    ap.add_argument("--output", required=True, help="adjusted output CSV (same columns/order)")
    ap.add_argument("--split", default="auto", choices=["auto", "train", "validation", "private"])
    ap.add_argument("--panel", action="append", default=None)
    ap.add_argument("--method", default=DEFAULT["method"],
                    choices=["median", "mean", "linear", "conservation", "flux"])
    ap.add_argument("--window", type=int, default=DEFAULT["window"],
                    help="local window (5-min steps) for the accumulation estimate")
    ap.add_argument("--strength", type=float, default=DEFAULT["strength"],
                    help="shrink factor on the correction (0 = no change)")
    ap.add_argument("--max-rel-change", type=float, default=DEFAULT["max_rel_change"])
    ap.add_argument("--max-abs-change", type=float, default=DEFAULT["max_abs_change"])
    ap.add_argument("--iterations", type=int, default=DEFAULT["iterations"],
                    help="conservation method iterations")
    ap.add_argument("--flux-mode", default=DEFAULT["flux_mode"],
                    choices=["projected", "smooth", "blend", "topology"])
    ap.add_argument("--flux-window", type=int, default=DEFAULT["flux_window"])
    ap.add_argument("--diagnostics", default=None, help="optional diagnostics CSV path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    kw = dict(method=args.method, window=args.window, strength=args.strength,
              max_rel_change=args.max_rel_change, max_abs_change=args.max_abs_change,
              iterations=args.iterations, flux_mode=args.flux_mode,
              flux_window=args.flux_window)
    adjust_file(args.state, args.output, args.split, args.panel, args.diagnostics,
                args.quiet, **kw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
