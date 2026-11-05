"""Shared data library for the TrafficFlowBench competition work.

Every script in ``work/`` should start with ``from traffic import *`` (run with
``work/lib`` on ``sys.path``) or add that directory to ``sys.path`` first.

Layout produced by ``work/build_cache.py``::

    work/cache/<PANEL>/network/            copies of the network CSVs
    work/cache/<PANEL>/<split>.npz         observation arrays for that split
    work/cache/task2_windows.parquet       all window_index rows
    work/cache/task2_history.parquet       all window_history rows
"""
from __future__ import annotations

import gc
import os
import re
import sys

LIBS = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\libs"
# pyarrow / psutil / the kaggle client are now pip-installed into the user site
# (see work/setup/SETUP.md); the old wheel-extraction directory is only a
# fallback if that install is missing.
try:  # noqa: SIM105
    import pyarrow  # noqa: F401
except ImportError:  # pragma: no cover
    if os.path.isdir(LIBS) and LIBS not in sys.path:
        sys.path.insert(0, LIBS)

import numpy as np
import pandas as pd

# Path layout: default to the workspace this library lives in (works both in the
# development tree and in the final package, where this file is code/lib/traffic.py).
# Organizers can point the pipeline at the released data with environment
# variables: TRAFFIC_REL (release root), TRAFFIC_WORK (cache/output root),
# TRAFFIC_ROOT (repository root).
_DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("TRAFFIC_ROOT", _DEFAULT_ROOT)
REL = os.environ.get("TRAFFIC_REL", os.path.join(ROOT, "kaggle_public"))
WORK = os.environ.get("TRAFFIC_WORK", os.path.join(ROOT, "work"))
CACHE = os.path.join(WORK, "cache")
REPO = os.path.join(ROOT, "repo", "trafficflowbench-public-main")
SUBS = os.path.join(WORK, "submissions")

PANELS = [
    "D7_I10_E", "D7_I10_W",
    "D7_I210_E", "D7_I210_W",
    "D7_I405_N", "D7_I405_S",
    "D12_I5_N", "D12_I5_S",
    "D12_I405_N", "D12_I405_S",
]
FAMILIES = {
    "D7_I10": ["D7_I10_E", "D7_I10_W"],
    "D7_I210": ["D7_I210_E", "D7_I210_W"],
    "D7_I405": ["D7_I405_N", "D7_I405_S"],
    "D12_I5": ["D12_I5_N", "D12_I5_S"],
    "D12_I405": ["D12_I405_N", "D12_I405_S"],
}
PANEL_FAMILY = {p: f for f, ps in FAMILIES.items() for p in ps}
REGIME_NAMES = ("R1", "R2", "R3")
REGIME_RATE = {"R1": 0.20, "R2": 0.30, "R3": 0.50}
N_TIMES = {"train": 273 * 288, "validation": 31 * 288, "private": 30 * 288}


# --------------------------------------------------------------------------- #
# cache path helpers
# --------------------------------------------------------------------------- #
def panel_cache(panel: str) -> str:
    return os.path.join(CACHE, panel)


def network_dir(panel: str) -> str:
    return os.path.join(panel_cache(panel), "network")


def split_cache_path(panel: str, split: str) -> str:
    return os.path.join(panel_cache(panel), f"{split}.npz")


def targets_cache_path(panel: str, split: str) -> str:
    return os.path.join(panel_cache(panel), f"targets_{split}.npz")


def ensure_dirs():
    for p in PANELS:
        os.makedirs(panel_cache(p), exist_ok=True)
    os.makedirs(SUBS, exist_ok=True)


# --------------------------------------------------------------------------- #
# network assets
# --------------------------------------------------------------------------- #
def _copy_network(panel: str) -> None:
    src = os.path.join(REL, "corridors", panel, "network")
    dst = network_dir(panel)
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.endswith(".csv") or name.endswith(".json"):
            target = os.path.join(dst, name)
            if not os.path.exists(target):
                with open(os.path.join(src, name), "rb") as fh_in, open(target, "wb") as fh_out:
                    fh_out.write(fh_in.read())


def load_network(panel: str) -> dict:
    """Links, FD parameters and topology for one panel, aligned by link id."""
    nd = network_dir(panel)
    if not os.path.exists(os.path.join(nd, "links.csv")):
        _copy_network(panel)
    links = pd.read_csv(os.path.join(nd, "links.csv"), dtype={"link_id": str})
    fd = pd.read_csv(os.path.join(nd, "fd_parameters.csv"), dtype={"link_id": str})
    # fd_parameters is measured; it is authoritative for lanes/free speed/capacity
    # (see the Task 3 evaluator notes).  links.csv only supplies length_km.
    m = fd.groupby("link_id", as_index=False).agg(
        lanes=("lanes", "mean"),
        free_speed=("free_speed_kmh", "mean"),
        capacity=("capacity_vph", "mean"),
        k_jam=("k_jam", "mean"),
        length_km=("length_km", "mean"),
    )
    m["critical_density"] = m["capacity"] / m["free_speed"].clip(lower=1.0)
    m["v_cut"] = 0.60 * m["free_speed"]
    out = links[["link_id", "length_km"]].merge(m, on="link_id", how="left", suffixes=("_links", ""))
    if "length_km_links" in out.columns:
        out["length_km"] = out["length_km"].fillna(out["length_km_links"])
        out = out.drop(columns=["length_km_links"])
    for col in ("lanes", "free_speed", "capacity"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1.0)
    out["lanes"] = out["lanes"].clip(lower=1.0)
    topo = pd.read_csv(os.path.join(nd, "lwr_mainline_topology.csv"), dtype=str)
    topo["mainline_link_id"] = topo["mainline_link_id"].astype(str)
    for col in ("incoming_link_ids", "outgoing_link_ids"):
        if col in topo.columns:
            topo[col] = topo[col].fillna("").astype(str)
    if "on_ramp_link_ids" in topo.columns:
        topo["on_ramp_link_ids"] = topo["on_ramp_link_ids"].fillna("").astype(str)
    if "off_ramp_link_ids" in topo.columns:
        topo["off_ramp_link_ids"] = topo["off_ramp_link_ids"].fillna("").astype(str)
    return {"links": out, "topo": topo}


def link_order(panel: str) -> np.ndarray:
    """Mainline links sorted upstream -> downstream using milepost/order_index."""
    net = load_network(panel)
    topo = net["topo"]
    col = "order_index" if "order_index" in topo.columns else None
    if col is not None:
        t = topo[["mainline_link_id", col]].copy()
        t[col] = pd.to_numeric(t[col], errors="coerce")
        t = t.sort_values(col)
        return t["mainline_link_id"].to_numpy(dtype=object)
    t = topo[["mainline_link_id", "hash"]].copy() if "hash" in topo.columns else topo
    return t["mainline_link_id"].to_numpy(dtype=object)


# --------------------------------------------------------------------------- #
# observation arrays
# --------------------------------------------------------------------------- #
def masked_files(panel: str, split: str):
    import glob

    root = os.path.join(REL, "corridors", panel, split, "mainline_states_masked")
    return sorted(glob.glob(os.path.join(root, "*", "*.parquet")))


def plain_files(panel: str, split: str):
    import glob

    root = os.path.join(REL, "corridors", panel, split, "mainline_states")
    return sorted(glob.glob(os.path.join(root, "*", "*.parquet")))


def ramp_files(panel: str, split: str):
    import glob

    root = os.path.join(REL, "corridors", panel, split, "ramp_states")
    return sorted(glob.glob(os.path.join(root, "*", "*.parquet")))


def _ts_to_epoch(ts: pd.Series) -> np.ndarray:
    """UTC text timestamps -> integer seconds since epoch (pandas 3 uses us)."""
    dt = pd.to_datetime(ts, utc=True).dt.as_unit("s")
    return dt.astype("int64").to_numpy()


def epoch_to_iso(epoch: np.ndarray) -> np.ndarray:
    dt = pd.to_datetime(np.asarray(epoch), unit="s", utc=True)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy()


def load_mainline(panel: str, split: str) -> dict:
    """Cached per-panel observation arrays.  NaN = missing/blanked.

    For ``train`` the cache holds the *unmasked* observation layer (the truth).
    For every split a ``<split>_masked`` cache holds the published masked view.
    """
    ensure_dirs()
    if split == "train":
        plain = _load_or_build(panel, "train", "plain")
        masked = _load_or_build(panel, "train", "masked")
        return {"plain": plain, "masked": masked}
    return {"masked": _load_or_build(panel, split, "masked")}


def _derive_network_view(panel: str, stations: np.ndarray, links: np.ndarray) -> dict:
    net = load_network(panel)["links"].set_index("link_id")
    order = link_order(panel)
    rank = {link: i for i, link in enumerate(order)}
    return {
        "stations": stations,
        "links": links,
        "milepost": np.array([rank.get(str(l), -1) for l in links]),
        "lanes": net.reindex(links)["lanes"].to_numpy(dtype=np.float32),
        "free_speed": net.reindex(links)["free_speed"].to_numpy(dtype=np.float32),
        "capacity": net.reindex(links)["capacity"].to_numpy(dtype=np.float32),
        "k_jam": net.reindex(links)["k_jam"].to_numpy(dtype=np.float32),
        "critical_density": net.reindex(links)["critical_density"].to_numpy(dtype=np.float32),
        "v_cut": net.reindex(links)["v_cut"].to_numpy(dtype=np.float32),
        "length_km": net.reindex(links)["length_km"].to_numpy(dtype=np.float32),
    }


def _load_or_build(panel: str, split: str, kind: str) -> dict:
    suffix = "" if kind == "plain" else "_masked"
    path = os.path.join(panel_cache(panel), f"{split}{suffix}.npz")
    if os.path.exists(path):
        return _read_npz(path)
    _build_split(panel, split, kind)
    return _read_npz(path)


def _read_npz(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    z.close()
    return out


def _write_npz(path: str, **arrays) -> None:
    # Write directly: the sandbox denies delete/rename of files inside the
    # workspace, so an atomic temp-file swap is not available here.
    np.savez_compressed(path, **arrays)


def _build_split(panel: str, split: str, kind: str) -> None:
    from concurrent.futures import ThreadPoolExecutor

    if kind == "plain":
        files = plain_files(panel, split)
    else:
        files = masked_files(panel, split)
    if not files:
        raise FileNotFoundError(f"no files for {panel} {split} {kind}")

    cols = ["timestamp", "station_id", "link_id", "speed_kmh", "flow_vph", "pct_observed", "is_score_eligible"]
    id_cols = ["timestamp", "station_id", "link_id"]
    BATCH = 8

    # Pass 1: the id grid only, batched so that only a few files are in memory.
    pair = {}
    time_set = set()

    def read_ids(path):
        d = pd.read_parquet(path, columns=id_cols)
        return (d["timestamp"].to_numpy(dtype=object),
                d["station_id"].to_numpy(dtype=object),
                d["link_id"].astype(str).to_numpy())

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i in range(0, len(files), BATCH):
            chunk = files[i:i + BATCH]
            for ts, sid, lid in ex.map(read_ids, chunk):
                time_set.update(ts)
                pair.update(zip(sid, lid))
    stations = np.array(sorted(pair), dtype="<U40")
    station_index = {s: i for i, s in enumerate(stations)}
    links = np.array([pair[s] for s in stations], dtype="<U40")
    times = np.array(sorted(time_set), dtype="<U30")
    t_epoch = _ts_to_epoch(pd.Series(times))
    t_index = {t: i for i, t in enumerate(times)}
    S, T = len(stations), len(times)

    speed = np.full((T, S), np.nan, dtype=np.float32)
    flow = np.full((T, S), np.nan, dtype=np.float32)
    pct = np.zeros((T, S), dtype=np.uint8)
    elig = np.zeros((T, S), dtype=bool)
    regime = np.zeros(T, dtype=np.int8)
    flat_speed = speed.reshape(-1)
    flat_flow = flow.reshape(-1)
    flat_pct = pct.reshape(-1)
    flat_elig = elig.reshape(-1)

    def read_full(path):
        return path, pd.read_parquet(path, columns=cols)

    # Pass 2: values, batched. Each batch is assigned then released.
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i in range(0, len(files), BATCH):
            chunk = files[i:i + BATCH]
            for path, d in ex.map(read_full, chunk):
                ts = d["timestamp"].to_numpy(dtype=object)
                ti = np.fromiter((t_index[t] for t in ts), dtype=np.int64, count=len(ts))
                si = np.fromiter((station_index[s] for s in d["station_id"].to_numpy(dtype=object)),
                                 dtype=np.int64, count=len(ts))
                flat = ti * S + si
                flat_speed[flat] = d["speed_kmh"].to_numpy(dtype=np.float32)
                flat_flow[flat] = d["flow_vph"].to_numpy(dtype=np.float32)
                flat_pct[flat] = d["pct_observed"].to_numpy(dtype=np.uint8)
                flat_elig[flat] = d["is_score_eligible"].to_numpy().astype(bool)
                if kind == "masked":
                    mreg = re.search(r"mask_regime=(R\d)", path)
                    if mreg is not None:
                        regime[ti] = int(REGIME_NAMES.index(mreg.group(1))) + 1
                del d
            gc.collect()

    view = _derive_network_view(panel, stations, links)
    out = {
        "times": t_epoch,
        "speed": speed,
        "flow": flow,
        "pct": pct,
        "elig": elig,
        "regime": regime,
        **view,
    }
    _write_npz(os.path.join(panel_cache(panel), f"{split}{'' if kind == 'plain' else '_masked'}.npz"), **out)


# --------------------------------------------------------------------------- #
# ramp arrays
# --------------------------------------------------------------------------- #
def load_ramps(panel: str, split: str) -> dict:
    """Ramp flows aggregated to attached mainline links on the mainline time grid.

    ``on_flow`` / ``off_flow`` are means over valid ramp cells on that link at
    that time (0 where no valid observation); ``on_valid`` / ``off_valid`` mark
    whether at least one attached ramp of that type was observed.
    """
    import glob

    path = os.path.join(panel_cache(panel), f"ramp_{split}.npz")
    if os.path.exists(path):
        return _read_npz(path)
    files = ramp_files(panel, split)
    if not files:
        raise FileNotFoundError(f"no ramp files {panel} {split}")
    cols = ["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "pct_observed", "is_score_eligible"]
    from concurrent.futures import ThreadPoolExecutor

    amap = pd.read_csv(os.path.join(network_dir(panel), "ramp_attachment_map.csv"), dtype=str)
    amap = amap[["ramp_link_id", "ramp_type", "nearest_mainline_link_id"]].drop_duplicates("ramp_link_id")
    nearest_map = dict(zip(amap["ramp_link_id"], amap["nearest_mainline_link_id"]))
    type_map = dict(zip(amap["ramp_link_id"], amap["ramp_type"]))
    main = _load_or_build(panel, split, "masked")
    iso = epoch_to_iso(main["times"])
    t_index = {s: i for i, s in enumerate(iso)}
    links = main["links"].astype(str)
    link_index = {l: i for i, l in enumerate(links)}
    T, L = len(iso), len(links)
    on_flow = np.zeros((T, L), dtype=np.float32)
    off_flow = np.zeros((T, L), dtype=np.float32)
    on_valid = np.zeros((T, L), dtype=bool)
    off_valid = np.zeros((T, L), dtype=bool)
    BATCH = 20

    def process(d):
        rid = d["ramp_link_id"].astype(str)
        rtype = d["ramp_type"].astype(str).where(d["ramp_type"].notna(), rid.map(type_map))
        nearest = rid.map(nearest_map)
        ti = d["timestamp"].map(t_index)
        li = nearest.map(link_index)
        valid = (
            d["is_score_eligible"].fillna(0).astype(bool)
            & (pd.to_numeric(d["pct_observed"], errors="coerce") >= 75)
            & pd.to_numeric(d["flow_vph"], errors="coerce").notna()
            & ti.notna() & li.notna()
        )
        if not valid.any():
            return
        fi = ti[valid].to_numpy(dtype=np.int64)
        li_ = li[valid].to_numpy(dtype=np.int64)
        fv = pd.to_numeric(d["flow_vph"], errors="coerce")[valid].to_numpy(dtype=np.float64)
        is_on = rtype[valid].astype(str).str.upper().isin(["OR", "ON"]).to_numpy()
        # Each file is a different day, so threads write disjoint time rows.
        np.add.at(on_flow, (fi[is_on], li_[is_on]), fv[is_on])
        np.add.at(off_flow, (fi[~is_on], li_[~is_on]), fv[~is_on])
        on_valid[fi[is_on], li_[is_on]] = True
        off_valid[fi[~is_on], li_[~is_on]] = True

    with ThreadPoolExecutor(max_workers=6) as ex:
        for i in range(0, len(files), BATCH):
            chunk = files[i:i + BATCH]
            list(ex.map(lambda f: process(pd.read_parquet(f, columns=cols)), chunk))
            gc.collect()

    out = {"on_flow": on_flow, "off_flow": off_flow, "on_valid": on_valid, "off_valid": off_valid}
    _write_npz(path, **out)
    return out


# --------------------------------------------------------------------------- #
# task 1 targets (the published row set)
# --------------------------------------------------------------------------- #
def load_targets(panel: str, split: str) -> dict:
    """The Task 1 rows the release asks for, as index arrays into the time grid."""
    path = targets_cache_path(panel, split)
    if os.path.exists(path):
        return _read_npz(path)
    tpl = pd.read_csv(
        os.path.join(REL, "task1", panel, split, "sample_submission_state.csv"),
        dtype={"timestamp": str, "station_id": str, "link_id": str, "mask_regime": str},
    )
    main = _load_or_build(panel, split, "masked")
    times = pd.to_datetime(main["times"], unit="s", utc=True)
    t_index = pd.Series(np.arange(len(times)), index=pd.DatetimeIndex(times))
    stations = pd.Index(main["stations"].astype(str))
    t_idx = pd.to_datetime(tpl["timestamp"], utc=True).map(t_index).to_numpy(dtype=np.int64)
    s_idx = stations.get_indexer(tpl["station_id"].astype(str))
    if (t_idx < 0).any() or (s_idx < 0).any():
        raise ValueError(f"{panel} {split}: template rows not found in cache")
    reg = np.array([REGIME_NAMES.index(r) for r in tpl["mask_regime"]], dtype=np.int8)
    out = {"t_idx": t_idx.astype(np.int32), "s_idx": s_idx.astype(np.int16), "regime": reg}
    _write_npz(path, **out)
    return out


# --------------------------------------------------------------------------- #
# task 2 assets
# --------------------------------------------------------------------------- #
def load_window_index() -> pd.DataFrame:
    path = os.path.join(CACHE, "task2_windows.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    parts = []
    import glob

    for f in sorted(glob.glob(os.path.join(REL, "task2", "*", "*", "window_index.csv"))):
        parts.append(pd.read_csv(f))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def load_window_history() -> pd.DataFrame:
    path = os.path.join(CACHE, "task2_history.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    parts = []
    import glob

    for f in sorted(glob.glob(os.path.join(REL, "task2", "*", "*", "window_history.parquet"))):
        parts.append(pd.read_parquet(f))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def load_queue_template() -> pd.DataFrame:
    parts = []
    import glob

    for f in sorted(glob.glob(os.path.join(REL, "task2", "*", "*", "sample_submission_queue.csv"))):
        parts.append(pd.read_csv(f, dtype={"window_id": str, "link_id": str}))
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# task 4 assets
# --------------------------------------------------------------------------- #
def task4_dir(panel: str, split: str) -> str:
    return os.path.join(REL, "task4", panel, split)


def load_task4(panel: str, split: str) -> dict:
    d = task4_dir(panel, split)
    prior = pd.read_csv(os.path.join(d, "synthetic_weak_prior.csv"), dtype={"path_id": str})
    counts = pd.read_csv(os.path.join(d, "synthetic_link_counts.csv"), dtype={"link_id": str})
    tpl = pd.read_csv(os.path.join(d, "sample_submission_path_flow.csv"), dtype={"path_id": str})
    nd = network_dir(panel)
    paths = pd.read_csv(os.path.join(nd, "path_set.csv"), dtype={"path_id": str})
    inc = pd.read_csv(os.path.join(nd, "path_link_incidence.csv"), dtype={"path_id": str, "link_id": str})
    return {"prior": prior, "counts": counts, "template": tpl, "paths": paths, "incidence": inc}


def build_incidence(panel: str, link_ids=None) -> tuple[np.ndarray, list, list]:
    t4 = load_task4(panel, "validation")
    paths = t4["paths"]
    inc = t4["incidence"]
    path_ids = paths["path_id"].tolist()
    path_index = {p: i for i, p in enumerate(path_ids)}
    if link_ids is None:
        link_ids = sorted(inc["link_id"].unique().tolist())
    link_index = {l: i for i, l in enumerate(link_ids)}
    A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
    rr = inc["link_id"].map(link_index)
    cc = inc["path_id"].map(path_index)
    m = rr.notna() & cc.notna()
    A[rr[m].to_numpy(dtype=np.int64), cc[m].to_numpy(dtype=np.int64)] = 1.0
    return A, path_ids, list(link_ids)


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
def state_score(pred_speed, true_speed, pred_flow, true_flow, lanes, regimes):
    """Reproduce score_task1 aggregation: regime -> panel -> family -> mean."""
    out = {}
    for r in np.unique(regimes):
        m = regimes == r
        ds = np.asarray(pred_speed)[m] - np.asarray(true_speed)[m]
        dq = (np.asarray(pred_flow)[m] - np.asarray(true_flow)[m]) / np.asarray(lanes)[m]
        rmse_s = float(np.sqrt(np.mean(ds * ds)))
        rmse_q = float(np.sqrt(np.mean(dq * dq)))
        s_s = max(0.0, 1.0 - rmse_s / 25.0)
        s_q = max(0.0, 1.0 - rmse_q / 600.0)
        out[int(r)] = {"rmse_speed": rmse_s, "rmse_flow": rmse_q, "S_speed": s_s, "S_flow": s_q,
                       "S_state": 0.54 * s_s + 0.46 * s_q, "n": int(m.sum())}
    out["macro"] = float(np.mean([v["S_state"] for k, v in out.items() if k != "macro"]))
    return out


if __name__ == "__main__":
    print("traffic.py", REL, "exists", os.path.isdir(REL))
    print("panels", len(PANELS))
