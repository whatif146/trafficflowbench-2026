"""Shared prediction helpers for the GBM path (used by predict.py).

Given the released window history of one window, build the feature matrix with
exactly the same code that produced the training data, then apply the fitted
decision rule.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import t2core as C
import features as FT

PANEL_ID = {p: i for i, p in enumerate(C.PANELS)}


def load_clim(panel: str) -> dict:
    path = os.path.join(C.MODEL2, f"static_{panel}.npz")
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in ("onset_freq", "ong_freq", "rate", "elig_rate",
                             "links", "v_cut", "free_speed", "capacity", "lanes",
                             "k_jam", "length_km", "rel_rank", "on_ramp",
                             "off_ramp", "rank_to_idx", "ratio_tod", "fcap_tod")}
    z.close()
    out["n_links"] = len(out["links"])
    out["critical_density"] = (out["capacity"] / np.maximum(
        np.asarray(out["v_cut"], np.float32) / 0.6, 1.0)).astype(np.float32)
    out["panel"] = panel
    return out


def artifact_dir(version: str | None = None) -> str:
    """Directory holding one model version's artefacts."""
    if version is None:
        return C.MODEL2
    return os.path.join(C.MODEL2, version)


def load_meta(version: str | None = None) -> dict:
    if version is None:
        path = os.path.join(C.OUT2, "gbm_meta.json")
    else:
        path = os.path.join(artifact_dir(version), "gbm_meta.json")
    with open(path) as fh:
        return json.load(fh)


def load_models(version: str | None = None):
    import lightgbm as lgb
    d = artifact_dir(version)
    m_on = lgb.Booster(model_file=os.path.join(d, "gbm_onset.txt"))
    m_ong = lgb.Booster(model_file=os.path.join(d, "gbm_ongoing.txt"))
    return m_on, m_ong


def attach_ramps(clim: dict, split: str) -> dict:
    """Attach the split's ramp arrays + mainline time grid to a clim bundle."""
    panel = clim["panel"]
    try:
        clim = dict(clim)
        clim["ramp"] = C.traffic.load_ramps(panel, split)
        clim["ramp_times"] = C.traffic.load_mainline(panel, split)["masked"]["times"]
    except FileNotFoundError:
        clim["ramp"] = None
    return clim


def _ramp_history(clim: dict, times) -> dict | None:
    if not clim.get("ramp"):
        return None
    grid = clim["ramp_times"]
    epoch = np.array([int(pd.Timestamp(t).timestamp()) for t in times], dtype=np.int64)
    pos = np.searchsorted(grid, epoch)
    if np.any(pos >= len(grid)) or np.any(grid[np.minimum(pos, len(grid) - 1)] != epoch):
        return None
    r = clim["ramp"]
    return {"on": r["on_flow"][pos], "off": r["off_flow"][pos],
            "on_valid": r["on_valid"][pos], "off_valid": r["off_valid"][pos]}


def window_frame_to_arrays(h: pd.DataFrame, clim: dict):
    """window_history frame -> (12, L) arrays aligned to clim['links']."""
    links = np.asarray(clim["links"], dtype=str)
    l_index = {str(l): i for i, l in enumerate(links)}
    times = np.sort(h.timestamp.unique())
    nT, L = len(times), len(links)
    speed = np.full((nT, L), np.nan, np.float32)
    flow = np.full((nT, L), np.nan, np.float32)
    elig = np.zeros((nT, L), bool)
    t_index = {t: i for i, t in enumerate(times)}
    ti = h.timestamp.map(t_index).to_numpy()
    li = h.link_id.map(l_index)
    keep = li.notna().to_numpy()
    sub = h[keep]
    ti = ti[keep]
    li = li[keep].astype(int).to_numpy()
    speed[ti, li] = pd.to_numeric(sub.speed_kmh, errors="coerce").to_numpy(np.float32)
    flow[ti, li] = pd.to_numeric(sub.flow_vph, errors="coerce").to_numpy(np.float32)
    elig[ti, li] = sub.is_score_eligible.astype(bool).to_numpy()
    return times, speed, flow, elig


def window_feature_block(h: pd.DataFrame, clim: dict, target_times):
    """Feature matrix (6L, F) for one released window.

    ``target_times`` are the six forecast timestamps (T+5..T+30).
    """
    times, speed, flow, elig = window_frame_to_arrays(h, clim)
    links = np.asarray(clim["links"], dtype=str)
    L = len(links)
    # a pseudo panel-data dict with the history in rows 0..11 and the forecast
    # timestamps in rows 13..18 (window_features indexes ti-12..ti for history,
    # ti+1..ti+6 for the target times, with ti = 12).
    times_full = list(pd.DatetimeIndex(times)) + [pd.Timestamp("1970-01-01", tz="UTC")] \
        + list(pd.DatetimeIndex(target_times)) + [pd.Timestamp("1970-01-01", tz="UTC")]
    d = {
        "n_links": L, "links": links,
        "speed": speed, "flow": flow, "elig": elig,
        "time_dt": pd.DatetimeIndex(times_full),
        "v_cut": clim["v_cut"], "free_speed": clim["free_speed"],
        "capacity": clim["capacity"],
        "lanes": clim["lanes"], "k_jam": clim["k_jam"],
        "critical_density": clim["critical_density"],
        "length_km": clim["length_km"], "rel_rank": clim["rel_rank"],
        "on_ramp": clim["on_ramp"], "off_ramp": clim["off_ramp"],
        "rank_to_idx": np.asarray(clim["rank_to_idx"], np.int64),
    }
    last = pd.Timestamp(times[-1])
    slot_last = (last.hour * 60 + last.minute) // 5
    qflags = (np.isfinite(speed) & (speed <= clim["v_cut"][None, :]))
    c = {"onset_freq": clim["onset_freq"], "ong_freq": clim["ong_freq"],
         "rate": clim["rate"], "elig_rate": clim["elig_rate"],
         "ratio_tod": clim["ratio_tod"], "fcap_tod": clim["fcap_tod"],
         "slot_last": slot_last, "nq_hist": float(qflags.sum()),
         "coverage": float(elig.mean())}
    nq_hist = float(qflags.sum())
    is_ongoing = 1 if (nq_hist >= 2 and qflags.sum(axis=0).max() >= 2) else 0
    ramp_hist = _ramp_history(clim, times)
    X = FT.window_features(d, 12, c, PANEL_ID[clim["panel"]], is_ongoing, ramp_hist)
    return X, is_ongoing


def select_features(X: np.ndarray, meta: dict):
    """Reorder/limit the feature matrix to the columns the model was trained on."""
    want = list(meta["features"])
    if want == list(FT.FEATURES):
        return X
    idx = [FT.FEATURES.index(name) for name in want]
    return X[:, idx]


def predict_window(h: pd.DataFrame, clim: dict, meta: dict, m_on, m_ong,
                   target_times) -> np.ndarray:
    """(6, L) binary prediction for one released window."""
    X, is_ongoing = window_feature_block(h, clim, target_times)
    X = select_features(X, meta)
    L = clim["n_links"]
    if not is_ongoing:
        p = m_on.predict(X).astype(np.float32)
        p = p[(C.HORIZON - 1) * L:]
        kind, val = meta["onset_rule"]
        pred = np.zeros((C.HORIZON, L), bool)
        if kind == "top":
            k = min(int(val), L)
            pred[C.HORIZON - 1, np.argsort(-p)[:k]] = True
        else:
            pred[C.HORIZON - 1, p >= float(val)] = True
        return pred
    p = m_ong.predict(X).astype(np.float32).reshape(C.HORIZON, L)
    return p >= float(meta["ongoing_th"])
