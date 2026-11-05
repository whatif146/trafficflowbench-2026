"""Local proxy evaluator for Task 3 (physics consistency) on the train split.

WHY A PROXY IS NEEDED
---------------------
The official ``score_task3.py`` derives the conservation frame from organizer-held
boundary flows which are never published.  With the topology fallback it prints a
number that is not a score: on validation a *perfect* answer scores 0.3301
locally against 0.9598 with organizer flows (docs/RUN_LOCAL.md).  ``S_LWR`` floors
at 0 for everybody, so the local number cannot rank Task 1 methods.

WHAT THIS FILE DOES
-------------------
``train`` ships the unmasked observation layer, i.e. the Task 1 truth.  That is
enough to mimic the organizer frame:

* state(p, r, t, link) = the full reconstructed state the evaluator would build:
  unmasked observations for every non-target cell, candidate values for the
  published Task 1 target cells of that regime, eligible cells only;
* ramp terms  = the released ramp observations with the official validity rules;
* boundary flux = a *frozen* proxy built from the unmasked observed flow field:
  the released topology applied to the observed field, optionally temporally
  denoised, optionally projected onto the observed conservation balance.  It is
  frozen, i.e. built once from observations and never re-derived from the
  candidate, which is what makes this a physics check rather than a
  self-fulfilling identity.

Everything else (FD per lane, empty-road guard, observable-transition filter,
ramp-valid transition filter, coverage mode, dN/dt forward interval) replicates
``src/task3/score_task3.py`` field for field.

The ``--flux fallback`` mode reproduces the official topology fallback exactly,
which is how the replication is validated against the organizer script on a
staging slice of the release.

USAGE
-----
    python local_eval.py --candidate observed --flux smooth --flux-window 9 \
        --panel D7_I10_E --panel D7_I10_W --output reports/calib.csv
    python local_eval.py --state-submission task1_train.csv --flux blend \
        --blend-lambda 0.5 --panel D7_I10_E

Built-in candidates: ``observed`` (truth replica), ``historical`` (weekday x
time-of-day mean of the masked layer), ``noisy:SD_SPEED:REL_FLOW``,
``lagged:N``, ``smooth:W``, ``constant``, ``linkmean``.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
if os.path.join(ROOT, "libs") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "libs"))
if os.path.join(ROOT, "work", "lib") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
import traffic  # noqa: E402  (shared loaders / cache)

REL = os.path.join(ROOT, "kaggle_public")
REPO = os.path.join(ROOT, "repo", "trafficflowbench-public-main")
WORK3 = os.path.join(ROOT, "work", "task3")
CACHE3 = os.path.join(WORK3, "cache")

REG_ORDER = ("R1", "R2", "R3")
SLOTS_PER_DAY = 288
DT_MAX_H = 5.1 / 60.0          # official `dt <= 5.1/60` in hours
EMPTY_FLOOR_VPH = 50.0          # official empty-road guard
EMPTY_SHARE = 0.20
MIN_VALID_CELLS = 100_000       # official mode thresholds (all corridors are A anyway)
MODE_A_COVERAGE = 0.75


# --------------------------------------------------------------------------- #
# network assets (replicated from src/task3/score_task3.py, which is normative)
# --------------------------------------------------------------------------- #
def network_parameters(panel: str) -> pd.DataFrame:
    """Per-link triangular FD parameters.  Byte-for-byte copy of the official
    ``network_parameters``; fd_parameters.csv is authoritative wherever present."""
    panel_dir = os.path.join(REL, "corridors", panel)
    n = pd.read_csv(os.path.join(panel_dir, "network", "links.csv"))
    n.link_id = n.link_id.astype(str)
    n = n.drop_duplicates("link_id")
    keep = ["link_id", "length_km", "free_speed_kmh", "capacity_vph", "lanes"]
    out = n[keep].copy()
    for c, default in (("length_km", 1.0), ("free_speed_kmh", 100.0)):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(default)
    out["lanes"] = pd.to_numeric(out["lanes"], errors="coerce").fillna(1.0).clip(lower=1.0)
    out.capacity_vph = pd.to_numeric(out.capacity_vph, errors="coerce")
    lanes = pd.to_numeric(n.get("lanes", pd.Series(4.0, index=n.index)), errors="coerce").fillna(4.0)
    lane_fallback = pd.Series(1800.0 * lanes.to_numpy(), index=n.index)
    lane_fallback.index = out.index
    out.capacity_vph = out.capacity_vph.fillna(lane_fallback)

    fd = os.path.join(panel_dir, "network", "fd_parameters.csv")
    if os.path.exists(fd):
        f = pd.read_csv(fd)
        f.link_id = f.link_id.astype(str)
        have = [c for c in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if c in f.columns]
        m = f.groupby("link_id", as_index=False)[have].mean()
        out = out.merge(m, on="link_id", how="left", suffixes=("", "_fd"))
        for c in ("free_speed_kmh", "capacity_vph", "lanes"):
            if f"{c}_fd" in out.columns:
                out[c] = out[f"{c}_fd"].fillna(out[c])
                out = out.drop(columns=[f"{c}_fd"])
        if "k_jam" not in out.columns:
            out["k_jam"] = np.nan
        out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
    else:
        out["critical_density"] = out.capacity_vph / out.free_speed_kmh.clip(lower=1)
        out["k_jam"] = np.nan
    out["k_jam"] = out.k_jam.fillna(100.0 * out.lanes.clip(lower=1.0))
    out["k_jam"] = np.maximum(out.k_jam, out.critical_density * 1.05)
    return out


def topology(panel: str) -> dict:
    """Released LWR topology: neighbour lists, ramp attachment, coverage mode."""
    path = os.path.join(REL, "corridors", panel, "network", "lwr_mainline_topology.csv")
    t = pd.read_csv(path, dtype=str)
    t["link_id"] = t.link_id.astype(str)
    for c in ("incoming_link_ids", "outgoing_link_ids", "on_ramp_link_ids", "off_ramp_link_ids"):
        if c in t.columns:
            t[c] = t[c].fillna("").astype(str)
        else:
            t[c] = ""
    incoming, outgoing = {}, {}
    for lid, inc, out in zip(t.link_id, t.incoming_link_ids, t.outgoing_link_ids):
        incoming[lid] = [x.strip() for x in inc.split(";") if x.strip()]
        outgoing[lid] = [x.strip() for x in out.split(";") if x.strip()]
    on_attached = set(t.loc[t.on_ramp_link_ids.str.len() > 0, "link_id"])
    off_attached = set(t.loc[t.off_ramp_link_ids.str.len() > 0, "link_id"])
    return {"frame": t, "ids": set(t.link_id), "incoming": incoming,
            "outgoing": outgoing, "on_attached": on_attached, "off_attached": off_attached}


# --------------------------------------------------------------------------- #
# ramp observations (released cells -> attached mainline link, official rules)
# --------------------------------------------------------------------------- #
def _ramp_npz_path(panel: str, split: str) -> str:
    return os.path.join(CACHE3, f"ramps_{panel}_{split}.npz")


def _official_ramp_arrays(panel: str, split: str, times_epoch: np.ndarray,
                          links: np.ndarray) -> dict:
    """Exact numpy re-implementation of score_task3.load_ramp_flows, aligned to
    the mainline (time x link) grid.  Used to build/verify the cache file."""
    panel_dir = os.path.join(REL, "corridors", panel)
    amap_path = os.path.join(panel_dir, "network", "ramp_attachment_map.csv")
    ramp_root = os.path.join(panel_dir, split, "ramp_states")
    T, L = len(times_epoch), len(links)
    on_flow = np.zeros((T, L), np.float32)
    off_flow = np.zeros((T, L), np.float32)
    on_valid = np.zeros((T, L), bool)
    off_valid = np.zeros((T, L), bool)
    if not args_missing(amap_path, ramp_root):
        return {"on_flow": on_flow, "off_flow": off_flow,
                "on_valid": on_valid, "off_valid": off_valid}
    amap = pd.read_csv(amap_path, dtype=str).rename(columns={"nearest_mainline_link_id": "link_id"})
    amap["ramp_link_id"] = amap.ramp_link_id.astype(str)
    amap["link_id"] = amap.link_id.astype(str)
    pieces = []
    for path in sorted(glob.glob(os.path.join(ramp_root, "**", "*.parquet"), recursive=True)):
        f = pd.read_parquet(path, columns=["timestamp", "ramp_link_id", "ramp_type", "flow_vph",
                                           "pct_observed", "is_score_eligible"])
        f["ramp_link_id"] = f.ramp_link_id.astype(str)
        f["timestamp"] = pd.to_datetime(f.timestamp, utc=True)
        f["flow_vph"] = pd.to_numeric(f.flow_vph, errors="coerce")
        f["pct_observed"] = pd.to_numeric(f.pct_observed, errors="coerce")
        f = f[f.is_score_eligible.astype(bool) & (f.pct_observed >= 75) & f.flow_vph.notna()]
        if f.empty:
            continue
        f = f.merge(amap[["ramp_link_id", "link_id"]], on="ramp_link_id", how="inner")
        is_on = f.ramp_type.astype(str).str.upper().isin(["OR", "ON"])
        f["on_ramp_flow_vph"] = np.where(is_on, f.flow_vph, 0.0)
        f["off_ramp_flow_vph"] = np.where(is_on, 0.0, f.flow_vph)
        f["on_ramp_valid"] = is_on.astype(int)
        f["off_ramp_valid"] = (~is_on).astype(int)
        pieces.append(f[["timestamp", "link_id", "on_ramp_flow_vph", "off_ramp_flow_vph",
                         "on_ramp_valid", "off_ramp_valid"]])
    if not pieces:
        return {"on_flow": on_flow, "off_flow": off_flow,
                "on_valid": on_valid, "off_valid": off_valid}
    d = pd.concat(pieces, ignore_index=True)
    g = d.groupby(["timestamp", "link_id"], as_index=False).agg(
        on_flow=("on_ramp_flow_vph", "sum"), off_flow=("off_ramp_flow_vph", "sum"),
        on_valid=("on_ramp_valid", "max"), off_valid=("off_ramp_valid", "max"))
    t_index = pd.Series(np.arange(T, dtype=np.int64),
                        index=pd.to_datetime(times_epoch.astype("int64"), unit="s", utc=True))
    l_index = pd.Series(np.arange(L, dtype=np.int64), index=pd.Index(links.astype(str)))
    g["ti"] = g.timestamp.map(t_index)
    g["li"] = g.link_id.map(l_index)
    g = g[g.ti.notna() & g.li.notna()]
    ti = g.ti.to_numpy(np.int64)
    li = g.li.to_numpy(np.int64)
    on_flow[ti, li] = g.on_flow.to_numpy(np.float32)
    off_flow[ti, li] = g.off_flow.to_numpy(np.float32)
    on_valid[ti, li] = g.on_valid.to_numpy().astype(bool)
    off_valid[ti, li] = g.off_valid.to_numpy().astype(bool)
    return {"on_flow": on_flow, "off_flow": off_flow,
            "on_valid": on_valid, "off_valid": off_valid}


def args_missing(*paths: str) -> bool:
    return all(os.path.exists(p) for p in paths)


def load_ramps(panel: str, split: str, times_epoch: np.ndarray, links: np.ndarray) -> dict:
    """Released ramp observations on the mainline grid, cached under work/task3.

    Prefers the shared team cache (work/cache/<panel>/ramp_<split>.npz built by
    work/lib/traffic.py); falls back to the exact official re-implementation and
    caches the result locally.  ``on_valid``/``off_valid`` mark whether at least
    one attached ramp of that type was observed (official validity rules)."""
    local = _ramp_npz_path(panel, split)
    if os.path.exists(local):
        z = np.load(local)
        return {k: z[k] for k in z.files}
    shared = os.path.join(traffic.CACHE, panel, f"ramp_{split}.npz")
    if os.path.exists(shared):
        z = np.load(shared)
        out = {k: z[k] for k in z.files}
    else:
        out = _official_ramp_arrays(panel, split, times_epoch, links)
        os.makedirs(CACHE3, exist_ok=True)
        np.savez_compressed(local, **out)
    return out


# --------------------------------------------------------------------------- #
# regime assignment (frozen official hash) and panel loading
# --------------------------------------------------------------------------- #
def regime_labels(panel: str, times_epoch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Official ``regime_of_dates``: one regime per calendar day, hashed by panel."""
    dates = pd.to_datetime(times_epoch.astype("int64"), unit="s", utc=True).strftime("%Y-%m-%d").to_numpy()
    labels = np.fromiter(
        (REG_ORDER[int.from_bytes(hashlib.blake2b((panel + "|" + d).encode("utf-8"),
                                                  digest_size=8).digest(), "big") % 3]
         for d in dates),
        dtype=object, count=len(dates))
    return labels, dates


def _panel_cache_ready(panel: str, split: str) -> bool:
    if split == "train":
        return (os.path.exists(traffic.split_cache_path(panel, "train"))
                and os.path.exists(os.path.join(traffic.CACHE, panel, "train_masked.npz")))
    return os.path.exists(os.path.join(traffic.CACHE, panel, f"{split}_masked.npz"))


def load_panel(panel: str, split: str = "train") -> dict:
    """Everything the proxy needs for one panel, on the shared station x time grid.

    Keys: times, stations, links, T, L, obs_speed/obs_flow/elig (full observation
    layer for train; published masked layer otherwise), masked_* layer, params
    (aligned per link), topo, targets, ramps, regime/dates per time step.
    """
    mask_npz = os.path.join(traffic.CACHE, panel, f"{split}_masked.npz")
    z = np.load(mask_npz)
    masked = {k: z[k] for k in z.files}
    z.close()
    times = masked["times"].astype("int64")
    stations = masked["stations"].astype(str)
    links = masked["links"].astype(str)
    out = {"panel": panel, "split": split, "times": times, "stations": stations,
           "links": links, "T": len(times), "L": len(links),
           "masked_speed": masked["speed"].astype(np.float32),
           "masked_flow": masked["flow"].astype(np.float32),
           "masked_elig": masked["elig"].astype(bool)}
    if split == "train":
        z = np.load(traffic.split_cache_path(panel, "train"))
        plain = {k: z[k] for k in z.files}
        z.close()
        assert np.array_equal(times, plain["times"].astype("int64")), f"{panel}: train time grids differ"
        out["obs_speed"] = plain["speed"].astype(np.float32)
        out["obs_flow"] = plain["flow"].astype(np.float32)
        out["elig"] = plain["elig"].astype(bool)
        out["has_truth"] = True
    else:
        out["obs_speed"] = out["masked_speed"]
        out["obs_flow"] = out["masked_flow"]
        out["elig"] = out["masked_elig"]
        out["has_truth"] = False
    links_i = pd.Index(links)
    p = network_parameters(panel).set_index("link_id").reindex(links_i)
    out["params"] = p
    out["p_arr"] = {c: p[c].to_numpy(np.float64) for c in
                    ("lanes", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam", "length_km")}
    out["topo"] = topology(panel)
    out["targets"] = traffic.load_targets(panel, split)
    out["ramps"] = load_ramps(panel, split, times, links)
    labels, dates = regime_labels(panel, times)
    out["regime"] = labels
    out["dates"] = dates
    return out


# --------------------------------------------------------------------------- #
# frozen boundary-flux proxies
# --------------------------------------------------------------------------- #
def fill_time_series(x: np.ndarray) -> np.ndarray:
    """Fill NaN gaps along axis 0 by linear interpolation per column (edge hold)."""
    T, L = x.shape
    out = np.empty((T, L), np.float32)
    tgrid = np.arange(T)
    for j in range(L):
        y = x[:, j]
        good = np.isfinite(y)
        if not good.any():
            out[:, j] = 0.0
            continue
        idx = np.flatnonzero(good)
        out[:, j] = np.interp(tgrid, idx, y[idx])
    return out


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average of width w along axis 0 (edge-replicated)."""
    if w <= 1:
        return x
    T, L = x.shape
    half = w // 2
    cs = np.concatenate([np.zeros((1, L)), np.cumsum(x, axis=0)], axis=0)
    i0 = np.clip(np.arange(T) - half, 0, T)
    i1 = np.clip(np.arange(T) - half + w, 0, T)
    return (cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1)[:, None]


def topology_flux(field: np.ndarray, links: np.ndarray, topo: dict,
                  nan_aware: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Mainline boundary flows implied by the released topology applied to a field.

    Mirrors score_task3.topology_boundary_flows: sum of the observed flows of the
    incoming / outgoing mainline neighbours (external connector ids ignored); a
    link with no neighbour of that type uses its own flow.  ``nan_aware`` skips
    NaN neighbours the way the official pandas pivot does."""
    T, L = field.shape
    idx = {lid: i for i, lid in enumerate(links.astype(str))}
    inflow = np.empty_like(field)
    outflow = np.empty_like(field)
    for j, lid in enumerate(links.astype(str)):
        inc = [idx[x] for x in topo["incoming"].get(lid, []) if x in idx]
        out = [idx[x] for x in topo["outgoing"].get(lid, []) if x in idx]
        if inc:
            inflow[:, j] = np.nansum(field[:, inc], axis=1) if nan_aware else field[:, inc].sum(axis=1)
        else:
            inflow[:, j] = field[:, j]
        if out:
            outflow[:, j] = np.nansum(field[:, out], axis=1) if nan_aware else field[:, out].sum(axis=1)
        else:
            outflow[:, j] = field[:, j]
    return inflow, outflow


def accum_from(speed: np.ndarray, flow: np.ndarray, length_km: np.ndarray) -> np.ndarray:
    """N = k*L = (q/v)*L with the official speed floor of 1 km/h."""
    return (flow / np.clip(speed, 1.0, None)) * length_km[None, :]


def build_frozen_flux(pdata: dict, kind: str = "smooth", window: int = 9,
                      blend_lambda: float = 1.0, flux_jitter: float = 0.0,
                      seed: int = 7) -> dict:
    """Boundary flux frozen from the *observations* (never from the candidate).

    kinds
    -----
    raw        topology applied to the raw observed flow field (NaN skipped)
    smooth     topology applied to the gap-filled, w-step smoothed observed flow
    projected  min-norm symmetric correction making the observed accumulation
               satisfy conservation: rhs = dN of the (optionally smoothed)
               observed accumulation.  window=1 is the *exact* projection onto
               the released observations (rhs = observed dN).
    blend      rhs = (1-lam)*rhs_smooth + lam*dN_smoothed
    fallback   placeholder: computed per candidate inside score_panel exactly like
               the official topology fallback (used only for replication checks)

    The official organizer flux is hidden, but the two published anchors pin it
    down: a perfect answer scores S_physics 0.9636 while its S_FD is ~0.982
    (computable from observations), so the organizer-flux S_LWR of an exact
    answer is ~0.95 -- the organizer flux is a projection onto the released
    observations with a residual of order 5%, not the observations themselves.
    ``flux_jitter`` injects a deterministic residual of that relative size so the
    proxy reproduces the published perfect-answer anchor; it is independent of
    the candidate, so rankings are unaffected.
    """
    times = pdata["times"]
    T, L = pdata["T"], pdata["L"]
    links = pdata["links"]
    length = pdata["p_arr"]["length_km"]
    ramps = pdata["ramps"]
    on_flow = ramps["on_flow"].astype(np.float32)
    off_flow = ramps["off_flow"].astype(np.float32)
    dt = np.full(T - 1, 5.0 / 60.0)
    if kind == "fallback":
        return {"kind": kind, "inflow": None, "outflow": None}
    if kind == "raw":
        if not pdata["has_truth"]:
            raise ValueError("raw flux needs the unmasked observed flow field (train split)")
        inflow, outflow = topology_flux(pdata["obs_flow"], links, pdata["topo"], nan_aware=True)
        # A link with no neighbour of that type falls back to its own flow, and the
        # official evaluator then fills any remaining gap with the state's own flow.
        own = fill_time_series(pdata["obs_flow"])
        inflow = np.where(np.isfinite(inflow), inflow, own).astype(np.float32)
        outflow = np.where(np.isfinite(outflow), outflow, own).astype(np.float32)
        return {"kind": kind, "inflow": inflow, "outflow": outflow}
    if not pdata["has_truth"]:
        raise ValueError(f"flux kind {kind!r} needs the unmasked observed field (train split)")
    flow = fill_time_series(pdata["obs_flow"])
    speed = fill_time_series(pdata["obs_speed"])
    flow_s = moving_average(flow, window).astype(np.float32)
    speed_s = moving_average(speed, window).astype(np.float32)
    del flow, speed
    inflow, outflow = topology_flux(flow_s, links, pdata["topo"], nan_aware=False)
    if kind == "smooth":
        return {"kind": kind, "window": window,
                "inflow": inflow.astype(np.float32), "outflow": outflow.astype(np.float32)}
    acc = accum_from(speed_s, flow_s, length).astype(np.float32)
    del speed_s, flow_s
    dN = np.full((T, L), np.nan, np.float32)
    dN[:-1] = acc[1:] - acc[:-1]
    del acc
    rhs0 = dt[:, None] * (inflow[:-1, :] + on_flow[:-1, :] - outflow[:-1, :] - off_flow[:-1, :])
    resid = dN[:-1] - rhs0
    del dN, rhs0
    if kind == "projected":
        lam = 1.0
    elif kind == "blend":
        lam = float(blend_lambda)
    else:
        raise ValueError(f"unknown flux kind {kind!r}")
    corr = (lam * resid / (2.0 * dt[:, None])).astype(np.float32)
    del resid
    inflow = inflow.astype(np.float32)
    outflow = outflow.astype(np.float32)
    inflow[:-1, :] += corr
    outflow[:-1, :] -= corr
    if flux_jitter > 0:
        rng = np.random.default_rng(seed)
        rhs_cur = (dt[:, None].astype(np.float32)
                   * (inflow[:-1, :] + on_flow[:-1, :] - outflow[:-1, :] - off_flow[:-1, :]))
        eps = (flux_jitter * np.abs(rhs_cur) * rng.standard_normal(rhs_cur.shape)).astype(np.float32)
        corr2 = eps / (2.0 * dt[:, None]).astype(np.float32)
        inflow[:-1, :] += corr2
        outflow[:-1, :] -= corr2
    return {"kind": kind, "window": window, "blend_lambda": lam, "flux_jitter": flux_jitter,
            "inflow": inflow, "outflow": outflow}


def state_fallback_flux(field_flow: np.ndarray, links: np.ndarray, topo: dict) -> tuple[np.ndarray, np.ndarray]:
    """Official topology fallback applied to a candidate *state* (NaN-aware).

    score_task3: ``inflow = pivot[incoming].sum(axis=1)`` (skipna), fallback to
    the link's own flow when no incoming neighbour exists; then a ``fillna(flow)``
    for any row without a value."""
    inflow, outflow = topology_flux(field_flow, links, topo, nan_aware=True)
    own = field_flow
    bad = ~np.isfinite(inflow)
    inflow[bad] = own[bad]
    bad = ~np.isfinite(outflow)
    outflow[bad] = own[bad]
    return inflow, outflow


# --------------------------------------------------------------------------- #
# candidate predictors
# --------------------------------------------------------------------------- #
def candidate_observed(pdata: dict) -> tuple[np.ndarray, np.ndarray]:
    return pdata["obs_speed"].copy(), pdata["obs_flow"].copy()


def candidate_linkmean(pdata: dict) -> tuple[np.ndarray, np.ndarray]:
    sp = pdata["obs_speed"] if pdata["has_truth"] else pdata["masked_speed"]
    fl = pdata["obs_flow"] if pdata["has_truth"] else pdata["masked_flow"]
    el = pdata["elig"]
    ok = el & np.isfinite(sp) & np.isfinite(fl)
    n = np.maximum(ok.sum(axis=0), 1)
    ms = np.nansum(np.where(ok, sp, 0.0), axis=0) / n
    mf = np.nansum(np.where(ok, fl, 0.0), axis=0) / n
    T, L = sp.shape
    return np.tile(ms, (T, 1)), np.tile(mf, (T, 1))


def candidate_historical(pdata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Weekday x time-of-day historical mean, built from the masked observation
    layer (the official Task 1 baseline; the same construction as
    ``build_task1_baseline_submission.py`` but from released observations only)."""
    sp = pdata["masked_speed"]
    fl = pdata["masked_flow"]
    el = pdata["masked_elig"] if "masked_elig" in pdata else pdata["elig"]
    T, L = sp.shape
    ts = pd.DatetimeIndex(pd.to_datetime(pdata["times"].astype("int64"), unit="s", utc=True))
    weekday = np.asarray(ts.weekday, dtype=np.int16)
    tod = (np.asarray(ts.hour, dtype=np.int16) * 12 + np.asarray(ts.minute, dtype=np.int16) // 5).astype(np.int16)
    slot = weekday * SLOTS_PER_DAY + tod
    n_slots = 7 * SLOTS_PER_DAY
    valid = el & np.isfinite(sp) & np.isfinite(fl)
    sum_sp = np.zeros((n_slots, L))
    cnt_sp = np.zeros((n_slots, L))
    sum_fl = np.zeros((n_slots, L))
    cnt_fl = np.zeros((n_slots, L))
    vs = valid & np.isfinite(sp)
    vf = valid & np.isfinite(fl)
    np.add.at(sum_sp, slot[vs.any(axis=1)], np.where(vs[vs.any(axis=1)], sp[vs.any(axis=1)], 0.0))
    np.add.at(cnt_sp, slot[vs.any(axis=1)], vs[vs.any(axis=1)].astype(float))
    np.add.at(sum_fl, slot[vf.any(axis=1)], np.where(vf[vf.any(axis=1)], fl[vf.any(axis=1)], 0.0))
    np.add.at(cnt_fl, slot[vf.any(axis=1)], vf[vf.any(axis=1)].astype(float))
    mean_sp = sum_sp / np.maximum(cnt_sp, 1)
    mean_fl = sum_fl / np.maximum(cnt_fl, 1)
    link_sp = np.nansum(np.where(vs, sp, 0.0), axis=0) / np.maximum(vs.sum(axis=0), 1)
    link_fl = np.nansum(np.where(vf, fl, 0.0), axis=0) / np.maximum(vf.sum(axis=0), 1)
    no_sp = cnt_sp == 0
    no_fl = cnt_fl == 0
    mean_sp = np.where(no_sp, link_sp[None, :], mean_sp)
    mean_fl = np.where(no_fl, link_fl[None, :], mean_fl)
    return mean_sp[slot], mean_fl[slot]


def candidate_noisy(pdata: dict, sd_speed: float, rel_flow: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    sp, fl = candidate_observed(pdata)
    rng = np.random.default_rng(seed)
    sp = sp + rng.normal(0.0, sd_speed, size=sp.shape)
    fl = fl * np.exp(rng.normal(0.0, rel_flow, size=fl.shape))
    return sp, fl


def candidate_lagged(pdata: dict, lag: int) -> tuple[np.ndarray, np.ndarray]:
    sp, fl = candidate_observed(pdata)
    out_s = np.full_like(sp, np.nan)
    out_f = np.full_like(fl, np.nan)
    if lag == 0:
        return sp, fl
    if lag > 0:                 # values from `lag` steps earlier
        out_s[lag:] = sp[:-lag]
        out_f[lag:] = fl[:-lag]
    else:
        out_s[:lag] = sp[-lag:]
        out_f[:lag] = fl[-lag:]
    return out_s, out_f


def candidate_smoothed(pdata: dict, window: int) -> tuple[np.ndarray, np.ndarray]:
    sp, fl = candidate_observed(pdata)
    return (moving_average(fill_time_series(sp), window).astype(np.float32),
            moving_average(fill_time_series(fl), window).astype(np.float32))


def candidate_constant(pdata: dict) -> tuple[np.ndarray, np.ndarray]:
    sp = pdata["obs_speed"] if pdata["has_truth"] else pdata["masked_speed"]
    fl = pdata["obs_flow"] if pdata["has_truth"] else pdata["masked_flow"]
    return np.full_like(sp, 100.0), np.full_like(fl, 6000.0)


def candidate_from_csv(path: str, panels: list[str], pdata_by_panel: dict,
                       chunk: int = 2_000_000) -> dict:
    """Read a Task 1 submission CSV in chunks, returning per-panel (speed, flow)
    arrays with candidate values at every row present in the file (NaN elsewhere)."""
    rows = _read_submission_rows(path, panels, "train", pdata_by_panel=pdata_by_panel)
    return {p: _rows_to_arrays(rows, pdata_by_panel[p]) for p in rows}


def _panel_meta(panel: str, split: str) -> dict:
    path = os.path.join(traffic.CACHE, panel, f"{split}_masked.npz")
    z = np.load(path)
    meta = {"times": z["times"].astype("int64"), "stations": z["stations"].astype(str)}
    z.close()
    return meta


def _read_submission_rows(path: str, panels: list[str], split: str,
                          pdata_by_panel: dict | None = None) -> dict:
    """Stream a Task 1 submission CSV into per-panel index arrays (memory friendly)."""
    need = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]
    meta = {}
    for p in panels:
        if pdata_by_panel is not None:
            meta[p] = {"times": pdata_by_panel[p]["times"],
                       "stations": pdata_by_panel[p]["stations"]}
        else:
            meta[p] = _panel_meta(p, split)
    t_index = {}
    s_index = {}
    for p in panels:
        t_index[p] = pd.Series(np.arange(len(meta[p]["times"]), dtype=np.int64),
                               index=pd.DatetimeIndex(pd.to_datetime(
                                   meta[p]["times"].astype("int64"), unit="s", utc=True)))
        s_index[p] = pd.Index(meta[p]["stations"].astype(str))
    buckets = {p: {"ti": [], "si": [], "sp": [], "fl": []} for p in panels}
    seen = 0
    reader = pd.read_csv(path, usecols=lambda c: c in need, chunksize=2_000_000)
    for chunk_df in reader:
        miss = set(need) - set(chunk_df.columns)
        if miss:
            raise ValueError(f"state submission is missing required columns: {sorted(miss)}")
        chunk_df["panel"] = chunk_df.panel.astype(str)
        ts = pd.to_datetime(chunk_df.timestamp, utc=True, errors="coerce")
        for panel, g in chunk_df.groupby("panel", sort=False):
            if panel not in buckets:
                continue
            ti = ts.loc[g.index].map(t_index[panel]).to_numpy()
            si = s_index[panel].get_indexer(g.station_id.astype(str))
            ok = np.isfinite(ti) & (si >= 0)
            if not ok.any():
                continue
            b = buckets[panel]
            b["ti"].append(ti[ok].astype(np.int32))
            b["si"].append(si[ok].astype(np.int32))
            b["sp"].append(pd.to_numeric(g.speed_kmh.to_numpy()[ok], errors="coerce").astype(np.float32))
            b["fl"].append(pd.to_numeric(g.flow_vph.to_numpy()[ok], errors="coerce").astype(np.float32))
            seen += int(ok.sum())
        print(f"  [csv] mapped {seen:,} submission rows", flush=True)
    out = {}
    for p in panels:
        b = buckets[p]
        out[p] = ({k: (np.concatenate(v) if v else np.empty(0, dtype=np.int64))
                   for k, v in b.items()})
    return out


def _rows_to_arrays(rows_by_panel: dict, pdata: dict) -> tuple[np.ndarray, np.ndarray]:
    """Dense (T x S) candidate arrays from streamed submission index arrays."""
    panel = pdata["panel"]
    T, S = pdata["T"], pdata["L"]
    cs = np.full((T, S), np.nan, np.float32)
    cf = np.full((T, S), np.nan, np.float32)
    b = rows_by_panel.get(panel)
    if b is not None and len(b["ti"]):
        cs[b["ti"], b["si"]] = b["sp"]
        cf[b["ti"], b["si"]] = b["fl"]
    return cs, cf


def build_candidate(pdata: dict, spec: str) -> tuple[np.ndarray, np.ndarray]:
    parts = str(spec).split(":")
    kind = parts[0]
    if kind == "observed":
        return candidate_observed(pdata)
    if kind == "historical":
        return candidate_historical(pdata)
    if kind == "linkmean":
        return candidate_linkmean(pdata)
    if kind == "constant":
        return candidate_constant(pdata)
    if kind == "noisy":
        sd = float(parts[1]) if len(parts) > 1 else 3.0
        rel = float(parts[2]) if len(parts) > 2 else 0.05
        return candidate_noisy(pdata, sd, rel)
    if kind == "lagged":
        return candidate_lagged(pdata, int(parts[1]) if len(parts) > 1 else 1)
    if kind == "smooth":
        return candidate_smoothed(pdata, int(parts[1]) if len(parts) > 1 else 3)
    raise ValueError(f"unknown candidate spec {spec!r}")


# --------------------------------------------------------------------------- #
# scoring (numpy replication of score_task3.score_panel)
# --------------------------------------------------------------------------- #
def _mode_table() -> dict:
    path = os.path.join(REPO, "config", "task3_lwr_modes.json")
    return json.loads(open(path, encoding="utf-8").read())


def _panel_order() -> list[str]:
    manifest = json.loads(open(os.path.join(REPO, "config", "corridors.json"), encoding="utf-8").read())
    return [p["corridor_id"] for p in manifest["panels"]]


def _observable_links(present_links: np.ndarray, links: np.ndarray, topo: dict) -> np.ndarray:
    """score_task3's observable-transition filter, per link."""
    own = set(links[present_links].astype(str))
    topo_ids = topo["ids"]
    obs = np.zeros(len(links), bool)
    for j, lid in enumerate(links.astype(str)):
        if lid not in topo_ids:
            continue
        inc = {x for x in topo["incoming"].get(lid, []) if x in topo_ids}
        out = {x for x in topo["outgoing"].get(lid, []) if x in topo_ids}
        obs[j] = inc.issubset(own) and out.issubset(own)
    return obs


def score_panel(pdata: dict, cand_speed: np.ndarray, cand_flow: np.ndarray,
                flux: dict, day_mask: np.ndarray | None = None) -> list[dict]:
    """Score one panel for all three regimes, mirroring score_task3.score_panel."""
    panel = pdata["panel"]
    mode = str(_mode_table()["panels"][panel]["mode"])
    T, L = pdata["T"], pdata["L"]
    links = pdata["links"]
    p = pdata["p_arr"]
    length = p["length_km"]
    lanes = np.clip(p["lanes"], 1.0, None)
    fs = p["free_speed_kmh"]
    cap = p["capacity_vph"]
    kcrit = p["critical_density"]
    kjam = p["k_jam"]
    topo = pdata["topo"]
    elems = np.arange(L)
    on_attached = np.array([lid in topo["on_attached"] for lid in links.astype(str)])
    off_attached = np.array([lid in topo["off_attached"] for lid in links.astype(str)])
    ramp = pdata["ramps"]
    tgt = pdata["targets"]
    times = pdata["times"]
    regime = pdata["regime"]
    global_epoch = times
    rows = []
    for ri, r in enumerate(REG_ORDER):
        t_sel = np.flatnonzero(regime == r)
        if day_mask is not None:
            t_sel = t_sel[day_mask[t_sel]]
        if t_sel.size == 0:
            rows.append({"level": "panel_regime", "panel": panel, "regime": r, "mode": mode,
                         "n_rows": 0, "n_trans": 0, "S_FD": 0.0, "S_LWR": 0.0, "E_LWR": 1.0,
                         "S_physics": 0.0})
            continue
        nr = t_sel.size
        # ---- state: observations everywhere, candidate values at this regime's targets
        speed = pdata["obs_speed"][t_sel].copy()
        flow = pdata["obs_flow"][t_sel].copy()
        present = pdata["elig"][t_sel].copy()
        ti = tgt["t_idx"][tgt["regime"] == ri]
        si = tgt["s_idx"][tgt["regime"] == ri]
        posmap = np.full(T, -1, np.int64)
        posmap[t_sel] = np.arange(nr)
        tp = posmap[ti]
        keep = tp >= 0
        tp = tp[keep]
        si = si[keep]
        target_mask = np.zeros((nr, L), bool)
        target_mask[tp, si] = True
        speed[tp, si] = cand_speed[ti[keep], si]
        flow[tp, si] = cand_flow[ti[keep], si]
        present &= np.isfinite(speed) & np.isfinite(flow)
        # ---- params (all links share one row; dropna rows without params)
        finite_p = np.isfinite(fs) & np.isfinite(cap) & np.isfinite(kcrit) & np.isfinite(kjam) & np.isfinite(length)
        valid = present & finite_p[None, :]
        if not valid.any():
            rows.append({"level": "panel_regime", "panel": panel, "regime": r, "mode": mode,
                         "n_rows": 0, "n_trans": 0, "S_FD": 0.0, "S_LWR": 0.0, "E_LWR": 1.0,
                         "S_physics": 0.0})
            continue
        # ---- FD branch (full state rows), float32 arithmetic, float64 sums
        f32 = np.float32
        v_speed = speed[valid].astype(f32)
        v_flow = flow[valid].astype(f32)
        v_link = np.broadcast_to(elems[None, :], (nr, L))[valid]
        lanes_v = lanes[v_link].astype(f32)
        q_lane = v_flow / lanes_v
        k_lane = (v_flow / np.clip(v_speed, f32(1.0), None)) / lanes_v
        cap_lane = (cap[v_link] / lanes[v_link]).astype(f32)
        kcrit_lane = (kcrit[v_link] / lanes[v_link]).astype(f32)
        kjam_lane = (kjam[v_link] / lanes[v_link]).astype(f32)
        q_fd = np.where(k_lane <= kcrit_lane, fs[v_link].astype(f32) * k_lane,
                        (cap_lane / np.maximum(kjam_lane - kcrit_lane, f32(1e-9)))
                        * np.maximum(kjam_lane - k_lane, f32(0.0)))
        del lanes_v, k_lane, cap_lane, kcrit_lane, kjam_lane, v_speed, v_link
        s_fd = float(max(0.0, 1.0 - np.abs(q_lane - q_fd).sum(dtype=np.float64)
                         / (np.abs(q_lane).sum(dtype=np.float64) + 1e-9)))
        del q_lane, q_fd
        submitted = target_mask[valid]
        if submitted.any() and float((v_flow[submitted] < EMPTY_FLOOR_VPH).mean()) > EMPTY_SHARE:
            s_fd = 0.0
        del v_flow
        # ---- observable transitions
        present_links = valid.any(axis=0)
        observable = _observable_links(present_links, links, topo)
        # ---- ramp validity with structural no-ramp rule
        onv = ramp["on_valid"][t_sel].copy()
        offv = ramp["off_valid"][t_sel].copy()
        onv |= ~on_attached[None, :]
        offv |= ~off_attached[None, :]
        on_flow = ramp["on_flow"][t_sel].astype(f32)
        off_flow = ramp["off_flow"][t_sel].astype(f32)
        # ---- forward interval N(t+dt)-N(t) on adjacent regime rows
        acc = np.where(valid, accum_from(speed, flow, length), np.nan).astype(f32)
        dN = np.full((nr, L), np.nan, f32)
        dN[:-1] = acc[1:] - acc[:-1]
        del acc
        dt_h = np.full(nr, np.nan, np.float64)
        dt_h[:-1] = np.diff(global_epoch[t_sel]) / 3600.0
        tr = (np.isfinite(dN) & (dt_h[:, None] > 0) & (dt_h[:, None] <= DT_MAX_H)
              & observable[None, :])
        if mode in ("A", "B"):
            ramp_ok = np.zeros((nr, L), bool)
            if nr > 1:
                ramp_ok[:-1] = onv[:-1] & offv[:-1]
            tr &= ramp_ok
        # ---- rhs from the frozen boundary flux
        if flux["kind"] == "fallback":
            # official semantics: flux derived from the candidate state, then fillna(state flow)
            inflow, outflow = state_fallback_flux(np.where(valid, flow, np.nan), links, topo)
            inflow = np.where(np.isfinite(inflow), inflow, flow).astype(f32)
            outflow = np.where(np.isfinite(outflow), outflow, flow).astype(f32)
        else:
            inflow = flux["inflow"][t_sel]
            outflow = flux["outflow"][t_sel]
        speed = flow = None
        # rhs is evaluated at the interval start row
        rhs = (dt_h[:, None].astype(f32) * (inflow + on_flow - outflow - off_flow)).astype(f32)
        rhs_sel = rhs[:-1][tr[:-1]]
        dN_sel = dN[:-1][tr[:-1]]
        if rhs_sel.size == 0:
            s_lwr, e_lwr = 0.0, 1.0
        else:
            resid = np.abs(dN_sel - rhs_sel)
            e_lwr = float(resid.sum(dtype=np.float64)
                          / (np.abs(rhs_sel).sum(dtype=np.float64) + 1e-9))
            s_lwr = max(0.0, 1.0 - min(1.0, e_lwr))
        s_phys = (s_fd + 2.0 * s_lwr) / 3.0
        n_tr = int(tr[:-1].sum()) if nr > 1 else 0
        rows.append({"level": "panel_regime", "panel": panel, "regime": r, "mode": mode,
                     "n_rows": n_tr, "n_trans": n_tr,
                     "S_FD": s_fd, "S_LWR": s_lwr, "E_LWR": e_lwr, "S_physics": s_phys})
    return rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    """panel mean over regimes, then equal-weight mean over panels (official shape)."""
    detail = pd.DataFrame(rows)
    panel_rows = []
    for panel, g in detail.groupby("panel", sort=False):
        panel_rows.append({"level": "panel", "panel": panel, "regime": "macro_mean",
                           "mode": g["mode"].iloc[0], "n_rows": int(g.n_rows.sum()),
                           "n_trans": int(g.n_trans.sum()),
                           "S_FD": float(g.S_FD.mean()), "S_LWR": float(g.S_LWR.mean()),
                           "E_LWR": float(g.E_LWR.mean()), "S_physics": float(g.S_physics.mean())})
    panel_df = pd.DataFrame(panel_rows)
    overall = float(panel_df.S_physics.mean()) if len(panel_df) else 0.0
    overall_row = pd.DataFrame([{"level": "overall", "panel": "ALL", "regime": "macro_mean",
                                 "mode": "", "n_rows": int(detail.n_rows.sum()),
                                 "n_trans": int(detail.n_trans.sum()),
                                 "S_FD": float(panel_df.S_FD.mean()) if len(panel_df) else 0.0,
                                 "S_LWR": float(panel_df.S_LWR.mean()) if len(panel_df) else 0.0,
                                 "E_LWR": float(panel_df.E_LWR.mean()) if len(panel_df) else 0.0,
                                 "S_physics": overall}])
    return pd.concat([detail, panel_df, overall_row], ignore_index=True)


def evaluate(pdata_by_panel: dict, candidate_fn, flux_by_panel: dict,
             day_filter: dict | None = None) -> pd.DataFrame:
    """Score all panels held in memory (small-panel use; the CLI streams instead)."""
    rows = []
    for panel in pdata_by_panel:
        pdata = pdata_by_panel[panel]
        cs, cf = candidate_fn(pdata)
        dm = None
        if day_filter and panel in day_filter:
            dm = np.isin(pdata["dates"], list(day_filter[panel]))
        rows += score_panel(pdata, cs, cf, flux_by_panel[panel], dm)
    return aggregate(rows)


def evaluate_streaming(panels: list[str], split: str, candidate_for_panel, flux_kind: str,
                       flux_window: int, blend_lambda: float, flux_jitter: float = 0.0,
                       day_filter: dict | None = None, verbose: bool = False) -> pd.DataFrame:
    """One panel at a time, freeing its arrays before the next (memory friendly)."""
    rows = []
    for panel in panels:
        pdata = load_panel(panel, split)
        if verbose:
            print(f"[local_eval] {panel}: T={pdata['T']} L={pdata['L']}", flush=True)
        flux = ({"kind": "fallback"} if flux_kind == "fallback"
                else build_frozen_flux(pdata, flux_kind, flux_window, blend_lambda, flux_jitter))
        cs, cf = candidate_for_panel(pdata)
        dm = None
        if day_filter and panel in day_filter:
            dm = np.isin(pdata["dates"], list(day_filter[panel]))
        rows += score_panel(pdata, cs, cf, flux, dm)
        if verbose:
            print(f"[local_eval] {panel}: done", flush=True)
        del pdata, flux, cs, cf
    return aggregate(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_spec_candidate(spec: str):
    """Return a callable pdata -> (speed, flow) for a built-in candidate spec."""
    def f(pdata):
        return build_candidate(pdata, spec)
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local train-split proxy for Task 3 physics scoring")
    ap.add_argument("--state-submission", type=str, default=None,
                    help="Task 1 submission CSV (panel,timestamp,station_id,link_id,mask_regime,speed_kmh,flow_vph)")
    ap.add_argument("--candidate", default=None,
                    help="built-in candidate: observed|historical|linkmean|constant|noisy:SD:REL|lagged:N|smooth:W "
                         "(used when --state-submission is not given; default observed)")
    ap.add_argument("--panel", action="append", default=None, help="panel id (repeatable); default all ready panels")
    ap.add_argument("--split", default="train", choices=["train"])
    ap.add_argument("--flux", default="smooth", choices=["raw", "smooth", "projected", "blend", "fallback"])
    ap.add_argument("--flux-window", type=int, default=9, help="temporal smoothing width for the flux proxy")
    ap.add_argument("--blend-lambda", type=float, default=1.0, help="blend weight towards the projected flux")
    ap.add_argument("--flux-jitter", type=float, default=0.0,
                    help="calibrated relative residual of the organizer flux projection "
                         "(0.06 reproduces the published perfect-answer anchor)")
    ap.add_argument("--dates", default=None,
                    help="comma-separated YYYY-MM-DD to restrict scoring (used for staging parity checks)")
    ap.add_argument("--output", default=os.path.join(WORK3, "reports", "local_eval.csv"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    panels = args.panel if args.panel else [p for p in _panel_order() if _panel_cache_ready(p, "train")]
    panels = [p for p in panels if _panel_cache_ready(p, "train")]
    if not panels:
        raise SystemExit("no cached train panels available; run build_cache first")
    if not args.quiet:
        print(f"[local_eval] panels={panels} candidate={args.candidate or args.state_submission} "
              f"flux={args.flux} window={args.flux_window} lambda={args.blend_lambda}", flush=True)
    day_filter = None
    if args.dates:
        keep = {d.strip() for d in args.dates.split(",") if d.strip()}
        day_filter = {p: keep for p in panels}
    if args.state_submission:
        # read the CSV once, keeping only (panel, t_idx, s_idx, speed, flow); the
        # per-panel dense arrays are built later, one panel at a time
        rows_by_panel = _read_submission_rows(args.state_submission, panels, "train")

        def candidate_for_panel(pdata):
            return _rows_to_arrays(rows_by_panel, pdata)
    else:
        spec = args.candidate or "observed"

        def candidate_for_panel(pdata):
            return build_candidate(pdata, spec)
    result = evaluate_streaming(panels, "train", candidate_for_panel, args.flux,
                                args.flux_window, args.blend_lambda, args.flux_jitter,
                                day_filter, verbose=not args.quiet)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result.to_csv(args.output, index=False)
    if not args.quiet:
        print(result.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ov = result[result.level == "overall"].iloc[0]
    print(f"Overall S_physics = {ov.S_physics:.6f}  (S_FD {ov.S_FD:.4f}, S_LWR {ov.S_LWR:.4f}, "
          f"E_LWR {ov.E_LWR:.4f})")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
