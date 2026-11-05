"""Feature construction for the Task 2 supervised models.

Every feature is computed from the 12 released history steps T-60..T-5 (and the
network assets) only, so the model stays online.  A window yields a
(6 * L, F) matrix: rows are (horizon step k=0..5, link) in that order, so row
k*L + l corresponds to the future cell (T + 5*(k+1), link l).
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HORIZON = 6
HIST = 12

FEATURES = [
    # current / recent speed level (all relative to the link queue threshold)
    "r_last", "r_med3", "r_med6", "r_min", "r_mean",
    "dr15", "dr30", "dr_all",
    # flow and density pressure
    "f_last_cap", "f_mean_cap", "f_trend", "d_last", "d_mean", "d_ratio",
    # queue indicators inside the history
    "qnow", "qmed3", "qmed6", "qmean3", "qmean6", "qcount", "n_nan",
    # spatial neighbours in rank (travel) order
    "r_last_up1", "r_last_dn1", "r_med3_up1", "r_med3_dn1",
    "qnow_up1", "qnow_dn1", "qnow_up2", "qnow_dn2", "qnow_up3", "qnow_dn3",
    "qnow_up5", "qnow_dn5", "qsum_up5", "qsum_dn5",
    "dist_up", "dist_dn",
    # climatology / anomalies against the time-of-day profile
    "rate_tgt", "rate_last", "onset_freq", "ong_freq", "elig_rate",
    "qnow_rate_tgt", "r_anom_last", "r_anom_med3", "fcap_anom_last",
    "fcap_anom_mean",
    # static link / panel descriptors
    "rel_rank", "lanes", "free_speed", "capacity", "k_jam", "length_km",
    "on_ramp", "off_ramp", "cap_per_lane",
    # window level context (broadcast to each row)
    "corr_r_last", "corr_f_last_cap", "nq_hist", "coverage",
    # attached ramp demand (on-ramp merge pressure / off-ramp relief)
    "on_last_cap", "on_mean3_cap", "on_trend_cap", "off_last_cap",
    "off_mean3_cap", "ramp_valid_rate",
    # horizon step descriptors
    "hour_sin", "hour_cos", "step_k", "is_ongoing", "panel_id",
]
N_FEATURES = len(FEATURES)


def _shift_rank(a: np.ndarray, r2i: np.ndarray, d: int, fill=np.nan) -> np.ndarray:
    """Shift a per-link array by d rank positions (positive = downstream)."""
    out = np.full_like(a, fill, dtype=np.float32)
    if d == 0:
        out[:] = a
        return out
    if d > 0:
        out[r2i[d:]] = a[r2i[:-d]]
        out[r2i[:d]] = fill
    else:
        k = -d
        out[r2i[:len(r2i) - k]] = a[r2i[k:]]
        out[r2i[len(r2i) - k:]] = fill
    return out


def _dist_to_queued(q: np.ndarray, r2i: np.ndarray, direction: int, cap: int = 20) -> np.ndarray:
    """Rank distance to the nearest queued link in one direction (cap if none)."""
    qr = (q[r2i] > 0).astype(np.int32)
    L = len(qr)
    out = np.full(L, cap, np.float32)
    best = np.full(L, cap, np.int32)
    # distance to the closest position j with qr[j]=1 in direction 'direction'
    if direction > 0:
        # look downstream: positions rank > i
        nxt = np.full(L, cap, np.int32)
        for i in range(L - 2, -1, -1):
            nxt[i] = 1 if qr[i + 1] else min(nxt[i + 1] + 1, cap)
    else:
        nxt = np.full(L, cap, np.int32)
        for i in range(1, L):
            nxt[i] = 1 if qr[i - 1] else min(nxt[i - 1] + 1, cap)
    out[:] = nxt
    return out


def window_features(d: dict, ti: int, clim: dict, panel_id: int,
                    condition_ongoing: int,
                    ramp_hist: dict | None = None) -> np.ndarray:
    """(6*L, N_FEATURES) float32 feature matrix for one origin."""
    L = d["n_links"]
    r2i = d["rank_to_idx"]
    vc = d["v_cut"]
    s = d["speed"][ti - HIST:ti].astype(np.float32)
    f = d["flow"][ti - HIST:ti].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = s / vc[None, :]
        dens = f / np.maximum(s, 1e-3)
    cap = np.maximum(d["capacity"], 1e-3)
    kjam = np.maximum(d["k_jam"], 1e-3)
    crit = np.maximum(d["critical_density"], 1e-3)

    r_last = r[-1]
    r_med3 = np.nanmedian(r[-3:], axis=0)
    r_med6 = np.nanmedian(r[-6:], axis=0)
    r_min = np.nanmin(r, axis=0)
    r_mean = np.nanmean(r, axis=0)
    dr15 = r_last - r[-4]
    dr30 = r_last - r[-7]
    dr_all = r_last - r[0]
    f_last = f[-1]
    f_mean = np.nanmean(f, axis=0)
    f_last_cap = f_last / cap
    f_mean_cap = f_mean / cap
    f_trend = (f_last - f_mean) / cap
    d_last = dens[-1] / kjam
    d_mean = np.nanmean(dens, axis=0) / kjam
    d_ratio = np.nanmean(dens, axis=0) / crit
    qflags = (np.isfinite(s) & (s <= vc[None, :])).astype(np.float32)
    qnow = qflags[-1]
    qmed3 = (np.isfinite(r_med3) & (r_med3 <= 1.0)).astype(np.float32)
    qmed6 = (np.isfinite(r_med6) & (r_med6 <= 1.0)).astype(np.float32)
    qmean3 = qflags[-3:].mean(axis=0)
    qmean6 = qflags[-6:].mean(axis=0)
    qcount = qflags.sum(axis=0)
    n_nan = np.isfinite(s).sum(axis=0).astype(np.float32) / HIST

    ratio_tod = clim["ratio_tod"]
    fcap_tod = clim["fcap_tod"]
    slot_last = clim["slot_last"]
    base = {
        "r_last": r_last, "r_med3": r_med3, "r_med6": r_med6, "r_min": r_min,
        "r_mean": r_mean, "dr15": dr15, "dr30": dr30, "dr_all": dr_all,
        "f_last_cap": f_last_cap, "f_mean_cap": f_mean_cap, "f_trend": f_trend,
        "d_last": d_last, "d_mean": d_mean, "d_ratio": d_ratio,
        "qnow": qnow, "qmed3": qmed3, "qmed6": qmed6, "qmean3": qmean3,
        "qmean6": qmean6, "qcount": qcount, "n_nan": n_nan,
        "r_last_up1": _shift_rank(r_last, r2i, -1),
        "r_last_dn1": _shift_rank(r_last, r2i, 1),
        "r_med3_up1": _shift_rank(r_med3, r2i, -1),
        "r_med3_dn1": _shift_rank(r_med3, r2i, 1),
        "qnow_up1": _shift_rank(qnow, r2i, -1, fill=0.0),
        "qnow_dn1": _shift_rank(qnow, r2i, 1, fill=0.0),
        "qnow_up2": _shift_rank(qnow, r2i, -2, fill=0.0),
        "qnow_dn2": _shift_rank(qnow, r2i, 2, fill=0.0),
        "qnow_up3": _shift_rank(qnow, r2i, -3, fill=0.0),
        "qnow_dn3": _shift_rank(qnow, r2i, 3, fill=0.0),
        "qnow_up5": _shift_rank(qnow, r2i, -5, fill=0.0),
        "qnow_dn5": _shift_rank(qnow, r2i, 5, fill=0.0),
        "qsum_up5": _shift_rank(qnow, r2i, -1, fill=0.0) + _shift_rank(qnow, r2i, -2, fill=0.0)
        + _shift_rank(qnow, r2i, -3, fill=0.0) + _shift_rank(qnow, r2i, -4, fill=0.0)
        + _shift_rank(qnow, r2i, -5, fill=0.0),
        "qsum_dn5": _shift_rank(qnow, r2i, 1, fill=0.0) + _shift_rank(qnow, r2i, 2, fill=0.0)
        + _shift_rank(qnow, r2i, 3, fill=0.0) + _shift_rank(qnow, r2i, 4, fill=0.0)
        + _shift_rank(qnow, r2i, 5, fill=0.0),
        "dist_up": _dist_to_queued(qnow, r2i, -1),
        "dist_dn": _dist_to_queued(qnow, r2i, 1),
        "r_anom_last": r_last - ratio_tod[slot_last],
        "r_anom_med3": r_med3 - ratio_tod[slot_last],
        "fcap_anom_last": f_last_cap - fcap_tod[slot_last],
        "fcap_anom_mean": f_mean_cap - fcap_tod[slot_last],
        "onset_freq": clim["onset_freq"],
        "ong_freq": clim["ong_freq"],
        "elig_rate": clim["elig_rate"],
        "rate_last": clim["rate"][clim["slot_last"], :],
        # static descriptors
        "rel_rank": d["rel_rank"],
        "lanes": d["lanes"],
        "free_speed": d["free_speed"],
        "capacity": d["capacity"],
        "k_jam": d["k_jam"],
        "length_km": d["length_km"],
        "on_ramp": d["on_ramp"],
        "off_ramp": d["off_ramp"],
        "cap_per_lane": d["capacity"] / np.maximum(d["lanes"], 1.0),
        # window level
        "corr_r_last": np.full(L, float(np.nanmean(r_last)), np.float32),
        "corr_f_last_cap": np.full(L, float(np.nanmean(f_last_cap)), np.float32),
        "nq_hist": np.full(L, clim["nq_hist"], np.float32),
        "coverage": np.full(L, clim["coverage"], np.float32),
        "is_ongoing": np.full(L, float(condition_ongoing), np.float32),
        "panel_id": np.full(L, float(panel_id), np.float32),
    }
    # ramp demand: on_flow/off_flow are 0 where the ramp cell is not observed,
    # so mask them with the validity flags before summarising.
    if ramp_hist:
        on = np.where(ramp_hist["on_valid"], ramp_hist["on"], np.nan).astype(np.float32)
        off = np.where(ramp_hist["off_valid"], ramp_hist["off"], np.nan).astype(np.float32)
        capv = np.maximum(d["capacity"], 1e-3)
        on_last = on[-1] / capv
        on_mean = np.nanmean(on, axis=0) / capv
        base["on_last_cap"] = on_last
        base["on_mean3_cap"] = np.nanmean(on[-3:], axis=0) / capv
        base["on_trend_cap"] = (on_last - on_mean)
        base["off_last_cap"] = off[-1] / capv
        base["off_mean3_cap"] = np.nanmean(off[-3:], axis=0) / capv
        base["ramp_valid_rate"] = (ramp_hist["on_valid"] | ramp_hist["off_valid"])[-6:].mean(axis=0).astype(np.float32)
    else:
        for name in ("on_last_cap", "on_mean3_cap", "on_trend_cap", "off_last_cap",
                     "off_mean3_cap", "ramp_valid_rate"):
            base[name] = np.zeros(L, np.float32)
    # per-step features
    tt = d["time_dt"][ti + 1:ti + 1 + HORIZON]
    tod_h = (tt.hour + tt.minute / 60.0).to_numpy(np.float32)
    slots = ((tt.hour * 60 + tt.minute) // 5).to_numpy(np.int32)
    rows = []
    for k in range(HORIZON):
        fk = dict(base)
        rate_tgt = clim["rate"][slots[k], :]
        fk["rate_tgt"] = rate_tgt
        fk["qnow_rate_tgt"] = qnow * rate_tgt
        ang = 2.0 * np.pi * tod_h[k] / 24.0
        fk["hour_sin"] = np.full(L, np.sin(ang), np.float32)
        fk["hour_cos"] = np.full(L, np.cos(ang), np.float32)
        fk["step_k"] = np.full(L, float(k), np.float32)
        rows.append(np.stack([fk[name] for name in FEATURES], axis=1).astype(np.float32))
    return np.concatenate(rows, axis=0)


def clim_from(d: dict, onset_freq: np.ndarray, ong_freq: np.ndarray,
              rate: np.ndarray, elig_rate: np.ndarray,
              ratio_tod: np.ndarray, fcap_tod: np.ndarray) -> dict:
    """Bundle the panel climatology arrays used by window_features."""
    return {"onset_freq": onset_freq, "ong_freq": ong_freq, "rate": rate,
            "elig_rate": elig_rate, "slot_last": None, "nq_hist": 0.0,
            "coverage": 0.0, "ratio_tod": ratio_tod, "fcap_tod": fcap_tod}


def set_window_clim(clim: dict, d: dict, ti: int) -> dict:
    """Fill the per-window scalars of a climatology bundle (in place copy)."""
    c = dict(clim)
    last = d["time_dt"][ti - 1]
    c["slot_last"] = (last.hour * 60 + last.minute) // 5
    s = d["speed"][ti - HIST:ti]
    c["nq_hist"] = float(np.nansum((np.isfinite(s) & (s <= d["v_cut"][None, :]))))
    e = d["elig"][ti - HIST:ti]
    c["coverage"] = float(e.mean())
    return c

