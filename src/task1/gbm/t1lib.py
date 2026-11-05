"""Shared Task 1 utilities: time helpers, row-level neighbours, chunked smoothers,
scoring.  Memory-lean by design (the box often has <1 GB free): full (T,S) fields
are only created one at a time, and neighbour features are computed row-wise.

Reads only the shared cache (work/cache via work/lib/traffic.py).
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import traffic  # noqa: E402

from scipy.ndimage import convolve1d, gaussian_filter1d  # noqa: E402

HO = 56  # default holdout length in days (8 weeks)


# --------------------------------------------------------------------------- #
# time helpers
# --------------------------------------------------------------------------- #
def day_index(times: np.ndarray) -> np.ndarray:
    return ((times - times[0]) // 86400).astype(np.int64)


def tod_index(times: np.ndarray) -> np.ndarray:
    return (((times - times[0]) % 86400) // 300).astype(np.int64)


def dow_index(times: np.ndarray) -> np.ndarray:
    return pd.to_datetime(times, unit="s", utc=True).dayofweek.to_numpy().astype(np.int8)


def regime_per_day(obs: dict, days: np.ndarray, n_days: int) -> np.ndarray:
    reg = np.zeros(n_days, dtype=np.int8)
    for d in range(n_days):
        r = obs["regime"][days == d]
        r = r[r > 0]
        if len(r):
            reg[d] = r[0]
    return reg


# --------------------------------------------------------------------------- #
# spatial ordering (milepost rank, never sorted link ids)
# --------------------------------------------------------------------------- #
def link_perm(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (order, inv): order[k] = original column index at spatial position k."""
    order = np.argsort(obs["milepost"], kind="stable").astype(np.int64)
    inv = np.empty(len(order), dtype=np.int64)
    inv[order] = np.arange(len(order), dtype=np.int64)
    return order, inv


# --------------------------------------------------------------------------- #
# row-level temporal neighbours (no full (T,S) temporaries)
# --------------------------------------------------------------------------- #
def temporal_neighbour_rows(arr: np.ndarray, ts: np.ndarray, ss: np.ndarray) -> dict:
    """Nearest observed value before/after in time, per requested row.

    Returns dict of 1-D arrays: before, after, dt_before, dt_after (999 = none).
    """
    T, S = arr.shape
    n = len(ts)
    before = np.full(n, np.nan, np.float32)
    after = np.full(n, np.nan, np.float32)
    dtb = np.full(n, 999.0, np.float32)
    dta = np.full(n, 999.0, np.float32)
    order = np.argsort(ss, kind="stable")
    ss_sorted = ss[order]
    us = np.unique(ss)
    bounds = np.searchsorted(ss_sorted, us, side="left")
    bounds_r = np.searchsorted(ss_sorted, us, side="right")
    idx = np.arange(T, dtype=np.int64)
    for i, s in enumerate(us):
        rows = order[bounds[i]:bounds_r[i]]
        if len(rows) == 0:
            continue
        x = arr[:, s]
        obs = np.isfinite(x)
        last = np.maximum.accumulate(np.where(obs, idx, -1))
        nxt = np.minimum.accumulate(np.where(obs, idx, T)[::-1])[::-1]
        t = ts[rows]
        li = last[t]
        ni = nxt[t]
        ok_b = li >= 0
        ok_a = ni < T
        before[rows[ok_b]] = x[li[ok_b]].astype(np.float32)
        after[rows[ok_a]] = x[ni[ok_a]].astype(np.float32)
        dtb[rows[ok_b]] = (t[ok_b] - li[ok_b]).astype(np.float32)
        dta[rows[ok_a]] = (ni[ok_a] - t[ok_a]).astype(np.float32)
    return {"before": before, "after": after, "dt_before": dtb, "dt_after": dta}


def interp_from_neighbours(nb: dict) -> np.ndarray:
    b, a = nb["before"], nb["after"]
    db, da = nb["dt_before"], nb["dt_after"]
    both = np.isfinite(b) & np.isfinite(a)
    out = np.where(both, (b * da + a * db) / np.maximum(db + da, 1.0),
                   np.where(np.isfinite(b), b, a)).astype(np.float32)
    return out


def temporal_interp(arr: np.ndarray) -> np.ndarray:
    """Full-field linear interpolation (used only by analysis scripts)."""
    T, S = arr.shape
    out = np.full((T, S), np.nan, np.float32)
    ts = np.arange(T, dtype=np.int64)
    for s in range(S):
        nb = temporal_neighbour_rows(arr, ts, np.full(T, s, np.int64))
        out[:, s] = interp_from_neighbours(nb)
    return out


# --------------------------------------------------------------------------- #
# row-level spatial neighbours
# --------------------------------------------------------------------------- #
def spatial_shift_rows(arr: np.ndarray, ts: np.ndarray, ss: np.ndarray, d: int,
                       order_p: np.ndarray, inv: np.ndarray) -> np.ndarray:
    S = len(order_p)
    pos = inv[ss] + d
    valid = (pos >= 0) & (pos < S)
    col = np.where(valid, order_p[np.clip(pos, 0, S - 1)], 0)
    out = arr[ts, col].astype(np.float32)
    out[~valid] = np.nan
    return out


def spatial_nb_mean_rows(arr: np.ndarray, ts: np.ndarray, ss: np.ndarray,
                         order_p: np.ndarray, inv: np.ndarray,
                         dists=(-2, -1, 1, 2)) -> np.ndarray:
    vals = [spatial_shift_rows(arr, ts, ss, d, order_p, inv) for d in dists]
    V = np.column_stack(vals)
    with np.errstate(all="ignore"):
        return np.nanmean(V, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# chunked Gaussian spatio-temporal smoother (memory-lean)
# --------------------------------------------------------------------------- #
def smooth_field(
    y: np.ndarray,
    sigma_t: float,
    sigma_s: float = 0.0,
    order: int = 0,
    order_p: np.ndarray | None = None,
    inv: np.ndarray | None = None,
    min_support: float = 1e-4,
    chunk: int = 4096,
    den: np.ndarray | None = None,
) -> np.ndarray:
    """Gaussian-weighted local mean (order=0) or local-linear-in-time intercept
    (order=1) of the observed cells of ``y`` (NaN = unobserved).

    With ``den`` given, smooths the ratio y/den (masked to finite, den>0) instead
    of y; this is how the density k = q/v is smoothed without materialising k.

    Weights exp(-dt^2/2sigma_t^2) * exp(-ds^2/2sigma_s^2), ds in link-order
    steps.  Returns a dense (T,S) float32 field in the original column order.
    Spatial smoothing along the link axis is done per time chunk; the time
    convolution uses a halo of 3*sigma_t rows so interior results are exact.
    """
    T, S = y.shape
    use_sp = sigma_s > 0 and order_p is not None and inv is not None
    coef = None
    K = 0
    if sigma_t > 0:
        K = int(np.ceil(3 * sigma_t))
        ks = np.arange(-K, K + 1, dtype=np.float64)
        w = np.exp(-0.5 * (ks / sigma_t) ** 2)
        coef = (w[::-1].copy(), (ks * w)[::-1].copy(), (ks * ks * w)[::-1].copy())

    out = np.full((T, S), np.nan, np.float32)
    CH = max(256, min(chunk, T))
    for lo in range(0, T, CH):
        hi = min(T, lo + CH)
        plo = max(0, lo - K)
        phi = min(T, hi + K)
        sub = y[plo:phi]
        if den is not None:
            dsub = den[plo:phi]
            valid = np.isfinite(sub) & np.isfinite(dsub) & (dsub > 1e-3)
            m = valid.astype(np.float32)
            yv = np.where(valid, sub / np.maximum(dsub, 1e-3), 0.0).astype(np.float32)
        else:
            m = np.isfinite(sub).astype(np.float32)
            yv = np.where(np.isfinite(sub), sub, 0.0).astype(np.float32)
        if use_sp:
            m = gaussian_filter1d(m, sigma_s, axis=1, mode="nearest")
            yv = gaussian_filter1d(yv, sigma_s, axis=1, mode="nearest")
            m = np.ascontiguousarray(m[:, order_p])
            yv = np.ascontiguousarray(yv[:, order_p])
        rows = slice(lo - plo, lo - plo + (hi - lo))
        if coef is None:
            with np.errstate(all="ignore"):
                vals = np.where(m > min_support, yv / np.maximum(m, 1e-6), np.nan)
            block = vals[rows]
        else:
            k0, k1, k2 = coef
            shp = m.shape
            S0 = convolve1d(m, k0, axis=0, mode="constant", cval=0.0,
                            output=np.empty(shp, np.float32))
            Sy = convolve1d(yv, k0, axis=0, mode="constant", cval=0.0,
                            output=np.empty(shp, np.float32))
            if order == 0:
                with np.errstate(all="ignore"):
                    vals = np.where(S0 > min_support, Sy / np.maximum(S0, 1e-6), np.nan)
                block = vals[rows]
            else:
                St = convolve1d(m, k1, axis=0, mode="constant", cval=0.0,
                                output=np.empty(shp, np.float32))
                Stt = convolve1d(m, k2, axis=0, mode="constant", cval=0.0,
                                 output=np.empty(shp, np.float32))
                Sty = convolve1d(yv, k1, axis=0, mode="constant", cval=0.0,
                                 output=np.empty(shp, np.float32))
                det = S0 * Stt - St * St
                num = Sy * Stt - Sty * St
                vals = np.full(shp, np.nan, np.float32)
                good = np.abs(det) > 1e-5
                vals[good] = (num[good] / det[good]).astype(np.float32)
                with np.errstate(all="ignore"):
                    fallback = np.where(S0 > min_support, Sy / np.maximum(S0, 1e-6), np.nan)
                vals = np.where(np.isfinite(vals), vals, fallback)
                block = vals[rows]
        if use_sp:
            block = np.ascontiguousarray(block[:, inv])
        out[lo:hi] = block
        del m, yv, sub
    return out


# --------------------------------------------------------------------------- #
# scoring (official Task 1 formula; regime 1..3)
# --------------------------------------------------------------------------- #
def score(true_s, true_q, pred_s, pred_q, lanes, regimes):
    pred_s = np.where(np.isfinite(pred_s), pred_s, 0.0)
    pred_q = np.where(np.isfinite(pred_q), pred_q, 0.0)
    out = {}
    for r in (1, 2, 3):
        m = regimes == r
        if not m.any():
            out[r] = dict(n=0, rmse_speed=float("nan"), rmse_flow=float("nan"),
                          S_speed=0.0, S_flow=0.0, S_state=0.0)
            continue
        ds = pred_s[m] - true_s[m]
        dq = (pred_q[m] - true_q[m]) / lanes[m]
        rmse_s = float(np.sqrt(np.mean(ds * ds)))
        rmse_q = float(np.sqrt(np.mean(dq * dq)))
        s_s = max(0.0, 1.0 - rmse_s / 25.0)
        s_q = max(0.0, 1.0 - rmse_q / 600.0)
        out[r] = dict(n=int(m.sum()), rmse_speed=rmse_s, rmse_flow=rmse_q, S_speed=s_s, S_flow=s_q,
                      S_state=0.54 * s_s + 0.46 * s_q)
    out["macro"] = float(np.mean([out[r]["S_state"] for r in (1, 2, 3)]))
    return out


def fmt(res: dict) -> str:
    s = "macro S_state %.4f" % res["macro"]
    for r in (1, 2, 3):
        s += " | R%d n=%d S=%.4f (s %.3f q %.3f rmse %.2f/%.0f)" % (
            r, res[r]["n"], res[r]["S_state"], res[r]["S_speed"], res[r]["S_flow"],
            res[r]["rmse_speed"], res[r]["rmse_flow"])
    return s
