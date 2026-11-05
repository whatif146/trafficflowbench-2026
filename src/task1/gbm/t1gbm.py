"""Task 1 statistical feature model: interp/smoother base + LightGBM residual.

Pipeline (per panel, memory-lean; one panel at a time):
  1. load the train plain layer only, extract labels at the published target
     rows, historical weekday x time-of-day profiles, per-link percentile clip
     bounds, then free it
  2. load the train masked layer (the published input view)
  3. features at the published target rows, computed from the masked view only;
     labels form the residual against the interp/smoother base
  4. LightGBM (speed and flow residuals)
  5. predict target rows of any split from its masked view, apply clip bounds

Features never touch the plain layer of the split being predicted.
"""
from __future__ import annotations

import gc
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task1\gbm")
import numpy as np  # noqa: E402

from t1lib import (smooth_field, temporal_neighbour_rows, interp_from_neighbours,  # noqa: E402
                   spatial_shift_rows, spatial_nb_mean_rows, day_index)

N_TOD = 288


# --------------------------------------------------------------------------- #
# historical assets
# --------------------------------------------------------------------------- #
def build_profiles(plain: dict, fit_day: np.ndarray, dow: np.ndarray,
                   tod: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(S, 7, 288) weekday x time-of-day means of the plain layer on fit days."""
    S = plain["speed"].shape[1]
    ps = np.full((S, 7, N_TOD), np.nan, dtype=np.float32)
    pq = np.full((S, 7, N_TOD), np.nan, dtype=np.float32)
    sp, fl = plain["speed"], plain["flow"]
    for w in range(7):
        wm = fit_day & (dow == w)
        for h in range(N_TOD):
            m = wm & (tod == h)
            if not m.any():
                continue
            ps[:, w, h] = np.nanmean(sp[m], axis=0)
            pq[:, w, h] = np.nanmean(fl[m], axis=0)
    return ps, pq


def per_link_clip_bounds(plain: dict, fit_day: np.ndarray):
    """Per-link p0.5 / p99.5 observed speed and p99.5 flow on fit days."""
    with np.errstate(all="ignore"):
        lo_s = np.nanpercentile(plain["speed"][fit_day], 0.5, axis=0).astype(np.float32)
        hi_s = np.nanpercentile(plain["speed"][fit_day], 99.5, axis=0).astype(np.float32)
        hi_q = np.nanpercentile(plain["flow"][fit_day], 99.5, axis=0).astype(np.float32)
    return lo_s, hi_s, hi_q


def per_day_station_means(masked: dict, days: np.ndarray, n_days: int):
    """(n_days, S) nan-mean observed speed/flow per day and station (masked view)."""
    S = masked["speed"].shape[1]
    ds = np.full((n_days, S), np.nan, dtype=np.float32)
    dq = np.full((n_days, S), np.nan, dtype=np.float32)
    sp, fl = masked["speed"], masked["flow"]
    order = np.argsort(days, kind="stable")
    bounds = np.searchsorted(days[order], np.arange(n_days + 1))
    for d in range(n_days):
        lo, hi = bounds[d], bounds[d + 1]
        if hi <= lo:
            continue
        ds[d] = np.nanmean(sp[order[lo:hi]], axis=0)
        dq[d] = np.nanmean(fl[order[lo:hi]], axis=0)
    return ds, dq


# --------------------------------------------------------------------------- #
# feature builder
# --------------------------------------------------------------------------- #
class FeatureBuilder:
    FEATURES: list[str] = []

    def __init__(self, masked: dict, profiles, day_means, attrs, order_p, inv,
                 n_days: int, dow: np.ndarray, tod: np.ndarray):
        self.m = masked
        self.ps, self.pq = profiles
        self.dsm, self.dqm = day_means
        self.order_p, self.inv = order_p, inv
        self.n_days = n_days
        self.dow, self.tod = dow, tod
        self.lanes = attrs["lanes"].astype(np.float32)
        self.free_speed = attrs["free_speed"].astype(np.float32)
        self.capacity = attrs["capacity"].astype(np.float32)
        self.k_jam = attrs["k_jam"].astype(np.float32)
        self.rank = np.empty(len(order_p), np.float32)
        self.rank[order_p] = np.arange(len(order_p), dtype=np.float32)
        self.length = attrs["length_km"].astype(np.float32)
        self.days = day_index(masked["times"])

    def _sp(self, arr, ts, ss, d):
        return spatial_shift_rows(arr, ts, ss, d, self.order_p, self.inv)

    def build(self, ts: np.ndarray, ss: np.ndarray, regime: np.ndarray) -> dict:
        ts = np.ascontiguousarray(ts, dtype=np.int64)
        ss = np.ascontiguousarray(ss, dtype=np.int64)
        n = len(ts)
        cols: dict[str, np.ndarray] = {}
        sp, fl = self.m["speed"], self.m["flow"]
        obs_m = np.isfinite(sp)
        dow_r = self.dow[ts]
        tod_r = self.tod[ts]
        day_r = self.days[ts]

        def add(name, v):
            cols[name] = np.asarray(v, dtype=np.float32)

        # ---- temporal interp + spatial neighbour mean base ----
        nbs = temporal_neighbour_rows(sp, ts, ss)
        nbq = temporal_neighbour_rows(fl, ts, ss)
        s_intr = interp_from_neighbours(nbs)
        q_intr = interp_from_neighbours(nbq)
        s_nbr = spatial_nb_mean_rows(sp, ts, ss, self.order_p, self.inv)
        q_nbr = spatial_nb_mean_rows(fl, ts, ss, self.order_p, self.inv)
        add("s_interp", s_intr)
        add("q_interp", q_intr)
        add("s_nb_mean", s_nbr)
        add("q_nb_mean", q_nbr)
        base_s_rows = np.where(np.isfinite(s_nbr), 0.9 * s_intr + 0.1 * s_nbr, s_intr)
        base_q_anchor = np.where(np.isfinite(q_nbr), 0.9 * q_intr + 0.1 * q_nbr, q_intr)
        add("base_s", base_s_rows)
        # ---- observed temporal neighbours ----
        for k in ("before", "after", "dt_before", "dt_after"):
            add("s_" + k, nbs[k])
        add("s_disagree", nbs["after"] - nbs["before"])
        add("s_gap", nbs["dt_before"] + nbs["dt_after"])
        del nbs
        for k in ("before", "after", "dt_before", "dt_after"):
            add("q_" + k, nbq[k])
        add("q_disagree", nbq["after"] - nbq["before"])
        del nbq
        gc.collect()

        # ---- smoothed speed fields ----
        f = smooth_field(sp, 1.5, 1.0, order=1, order_p=self.order_p, inv=self.inv)
        add("s_sm15", f[ts, ss])
        add("s_sm_up1", self._sp(f, ts, ss, 1))
        add("s_sm_dn1", self._sp(f, ts, ss, -1))
        s_sm15_rows = cols["s_sm15"]
        del f
        gc.collect()
        f = smooth_field(sp, 3.0, 0.0, order=1, order_p=self.order_p, inv=self.inv)
        add("s_sm3_nosp", f[ts, ss])
        del f
        f = smooth_field(sp, 6.0, 1.5, order=1, order_p=self.order_p, inv=self.inv)
        add("s_sm6", f[ts, ss])
        del f
        gc.collect()

        # ---- smoothed flow fields ----
        f = smooth_field(fl, 1.5, 1.0, order=1, order_p=self.order_p, inv=self.inv)
        q_sm_r = f[ts, ss].copy()
        add("q_sm15", q_sm_r)
        add("q_sm_up1", self._sp(f, ts, ss, 1))
        add("q_sm_dn1", self._sp(f, ts, ss, -1))
        del f
        gc.collect()
        base_q_rows = np.where(np.isfinite(q_sm_r), 0.5 * q_sm_r + 0.5 * base_q_anchor, base_q_anchor)
        add("base_q", base_q_rows)
        add("base_q_interp", base_q_anchor)
        f = smooth_field(fl, 6.0, 1.5, order=1, order_p=self.order_p, inv=self.inv)
        add("q_sm6", f[ts, ss])
        del f

        # ---- density path k = q / v (smoothed ratio, no full k field) ----
        f = smooth_field(fl, 3.0, 1.0, order=1, order_p=self.order_p, inv=self.inv, den=sp)
        k_sm_r = f[ts, ss]
        add("k_sm", k_sm_r)
        add("q_from_k", k_sm_r * np.asarray(s_sm15_rows, np.float32))
        del f
        gc.collect()

        # ---- raw spatial neighbours at same timestamp ----
        add("s_sp_up1", self._sp(sp, ts, ss, 1))
        add("s_sp_dn1", self._sp(sp, ts, ss, -1))
        add("s_sp_up2", self._sp(sp, ts, ss, 2))
        add("s_sp_dn2", self._sp(sp, ts, ss, -2))
        add("q_sp_up1", self._sp(fl, ts, ss, 1))
        add("q_sp_dn1", self._sp(fl, ts, ss, -1))
        add("s_ud_diff", cols["s_sp_up1"] - cols["s_sp_dn1"])

        # ---- historical profile ----
        add("prof_s", self.ps[ss, dow_r, tod_r])
        add("prof_q", self.pq[ss, dow_r, tod_r])
        add("s_minus_prof", cols["s_sm15"] - cols["prof_s"])
        add("q_minus_prof", cols["q_sm15"] - cols["prof_q"])

        # ---- day-level level features (masked view) ----
        prev = np.clip(day_r - 1, 0, self.n_days - 1)
        prev2 = np.clip(day_r - 2, 0, self.n_days - 1)
        v1 = day_r >= 1
        v2 = day_r >= 2
        add("prevday_s", np.where(v1, self.dsm[prev, ss], np.nan))
        add("prevday_q", np.where(v1, self.dqm[prev, ss], np.nan))
        add("prevday2_s", np.where(v2, self.dsm[prev2, ss], np.nan))
        add("station_day_dev_s", cols["prevday_s"] - cols["prof_s"])

        # ---- quality / missingness ----
        add("pct", self.m["pct"][ts, ss])
        f = smooth_field(obs_m.astype(np.float32), 3.0, 1.0, order=0,
                         order_p=self.order_p, inv=self.inv)
        add("obs_density", f[ts, ss])
        del f, obs_m
        gc.collect()

        # ---- attributes / static ----
        add("lanes", self.lanes[ss])
        add("free_speed", self.free_speed[ss])
        add("capacity", self.capacity[ss])
        add("k_jam", self.k_jam[ss])
        add("milepost_rank", self.rank[ss])
        add("length_km", self.length[ss])

        # ---- time ----
        add("tod", tod_r)
        add("dow", dow_r)
        add("regime", regime)
        add("is_weekend", (dow_r >= 5).astype(np.float32))

        # ---- derived physics ratios ----
        with np.errstate(invalid="ignore", divide="ignore"):
            add("s_over_free", cols["s_sm15"] / self.free_speed[ss])
            add("q_over_cap", cols["q_sm15"] / self.capacity[ss])
            add("s_fine_minus_coarse", cols["s_sm15"] - cols["s_sm6"])
            add("q_fine_minus_coarse", cols["q_sm15"] - cols["q_sm6"])

        names = list(cols.keys())
        X = np.empty((n, len(names)), np.float32)
        for i, nm in enumerate(names):
            X[:, i] = cols[nm]
            cols[nm] = None
        FeatureBuilder.FEATURES = names
        del cols
        gc.collect()
        return {"X": X, "names": names,
                "base_s": np.asarray(base_s_rows, np.float64),
                "base_q": np.asarray(base_q_rows, np.float64)}


# --------------------------------------------------------------------------- #
# model fit / predict
# --------------------------------------------------------------------------- #
LGB_PARAMS = dict(
    objective="l2",
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=1.0,
    n_estimators=600,
    num_threads=12,
    seed=42,
    deterministic=True,
    force_col_wise=True,
    verbosity=-1,
)


def fit_models(X: np.ndarray, ys: np.ndarray, yq: np.ndarray):
    import lightgbm as lgb

    params = dict(LGB_PARAMS)
    ms = lgb.LGBMRegressor(**params)
    ms.fit(X, ys)
    mq = lgb.LGBMRegressor(**params)
    mq.fit(X, yq)
    return ms, mq


def clip_predictions(speed, flow, free_speed, lo_s, hi_s, hi_q):
    """Only guard against implausible extremes: a low-speed floor of 1 km/h and
    per-link p99.5 upper bounds.  A per-link low percentile is NOT used because
    it clips genuine congestion."""
    speed = np.asarray(speed, np.float64)
    flow = np.asarray(flow, np.float64)
    lo = np.maximum(lo_s.astype(np.float64) * 0.0 + 1.0, 1.0)
    hi = np.where(np.isfinite(hi_s), hi_s, free_speed * 1.1).astype(np.float64)
    speed = np.clip(np.where(np.isfinite(speed), speed, np.nan), lo, hi)
    flow = np.clip(np.where(np.isfinite(flow), flow, np.nan), 0.0,
                   np.where(np.isfinite(hi_q), hi_q, 1e9).astype(np.float64))
    return speed, flow
