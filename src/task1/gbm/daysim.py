"""Day-similarity / month-tensor completion features for the Task 1 GBM (v3 test).

For each target cell (day d, time-of-day h, link s) the estimator looks at the
*same (h, s)* on the other days of the month's masked view and takes a
similarity-weighted average, with weekday and day-level-offset adjustment:

    sim_est(d,h,s)      = sum_{d'!=d} w[d,d'] * O[d',h,s] / sum_{d'!=d} w[d,d']
    sim_adj(d,h,s)      = sim_est + dmean(d,s) - weighted_mean_{d'}(dmean(d',s))

where O is the masked observation tensor [day, 288, link], dmean(d,s) is the
day's mean observed value at that link (available for the target day from its
non-blanked cells), and w[d,d'] is a soft similarity between the days' observed
time-of-day speed/flow profiles (correlation-based, weekday boost, optional
temporal decay).  Also provides same-tod values at d+-1, d+-7 and the
all-day same-tod mean.

Only the masked view of the split being predicted is used: no truth of the
predicted split is touched, so this is valid on validation/private.
"""
from __future__ import annotations

import numpy as np

N_TOD = 288


def _std_profile_sim(P: np.ndarray) -> np.ndarray:
    """Correlation-like similarity between rows of P (D, 288), NaN-tolerant."""
    mu = np.nanmean(P, axis=1, keepdims=True)
    sd = np.nanstd(P, axis=1, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    Z = (P - mu) / sd
    Z = np.where(np.isfinite(Z), Z, 0.0)
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    Z = Z / np.maximum(nrm, 1e-6)
    return (Z @ Z.T).astype(np.float32)


class DaySimFeatures:
    """Builds day-similarity completion features for arbitrary (t, s) rows."""

    def __init__(self, masked: dict, dow_day: np.ndarray, alpha_sim: float = 0.25,
                 dow_boost: float = 2.0, t_decay: float = 0.0):
        sp = masked["speed"]
        fl = masked["flow"]
        T, S = sp.shape
        assert T % N_TOD == 0, f"time grid {T} not a multiple of {N_TOD}"
        D = T // N_TOD
        self.D, self.S, self.T = D, S, T
        self.sp = sp.reshape(D, N_TOD, S)
        self.fl = fl.reshape(D, N_TOD, S)
        self.flat_s = self.sp.reshape(D, N_TOD * S)
        self.flat_q = self.fl.reshape(D, N_TOD * S)
        obs_s = np.isfinite(self.flat_s)
        obs_q = np.isfinite(self.flat_q)
        self.dmean_s = np.nanmean(self.sp, axis=1).astype(np.float32)  # (D,S)
        self.dmean_q = np.nanmean(self.fl, axis=1).astype(np.float32)
        self.wsum_s = None

        # ---- day similarity from observed time-of-day profiles ----
        prof_s = np.nanmean(self.sp, axis=2).astype(np.float32)  # (D,288)
        prof_q = np.nanmean(self.fl, axis=2).astype(np.float32)
        corr = 0.5 * (_std_profile_sim(prof_s) + _std_profile_sim(prof_q))
        corr = np.clip(corr, -1.0, 1.0)
        w = np.exp(-(1.0 - corr) / max(alpha_sim, 1e-3)).astype(np.float32)
        same_dow = (dow_day[:, None] == dow_day[None, :])
        w = w * np.where(same_dow, dow_boost, 1.0)
        if t_decay > 0:
            dd = np.abs(np.arange(D)[:, None] - np.arange(D)[None, :])
            w = w * np.exp(-dd / t_decay)
        np.fill_diagonal(w, 0.0)
        self.W = w
        self.corr_max = (corr.max(axis=1)).astype(np.float32)
        self.dow_day = dow_day

        # ---- similarity-weighted same-tod estimate (full tensor, computed lazily) ----
        wsum = w.sum(axis=1)  # (D,)
        self.wsum = wsum
        num_s = self._wmatmul(w, self.flat_s, obs_s)  # (D, HS) sums, NaN->0
        den_s = self._wmatmul(w, self.flat_s, obs_s, weights_only=True)
        num_q = self._wmatmul(w, self.flat_q, obs_q)
        den_q = self._wmatmul(w, self.flat_q, obs_q, weights_only=True)
        self.sim_s = np.where(den_s > 1e-4, num_s / np.maximum(den_s, 1e-6), np.nan).astype(np.float32)
        self.sim_q = np.where(den_q > 1e-4, num_q / np.maximum(den_q, 1e-6), np.nan).astype(np.float32)
        self.den_s = den_s.astype(np.float32)
        # weighted day-link mean for the offset adjustment (small matrices)
        wd_s = (w @ np.nan_to_num(self.dmean_s)) / np.maximum(w @ np.isfinite(self.dmean_s), 1e-6)
        wd_q = (w @ np.nan_to_num(self.dmean_q)) / np.maximum(w @ np.isfinite(self.dmean_q), 1e-6)
        self.wdmean_s = wd_s.astype(np.float32)  # (D,S)
        self.wdmean_q = wd_q.astype(np.float32)
        # all-day same-tod mean over observed cells
        cnt_s = obs_s.sum(axis=0)
        cnt_q = obs_q.sum(axis=0)
        self.allday_s = np.where(cnt_s > 0,
                                  np.nansum(self.flat_s, axis=0) / np.maximum(cnt_s, 1),
                                  np.nan).astype(np.float32)
        self.allday_q = np.where(cnt_q > 0,
                                  np.nansum(self.flat_q, axis=0) / np.maximum(cnt_q, 1),
                                  np.nan).astype(np.float32)
        del num_s, den_s, num_q, den_q, obs_s, obs_q
        self.names = [
            "ds_sim_s", "ds_sim_q", "ds_sim_adj_s", "ds_sim_adj_q", "ds_w",
            "ds_allday_s", "ds_allday_q",
            "ds_prev1_s", "ds_next1_s", "ds_prev7_s", "ds_next7_s",
            "ds_prev1_q", "ds_next1_q", "ds_corr_max",
        ]

    @staticmethod
    def _wmatmul(w, flat, obs, weights_only=False):
        src = obs.astype(np.float32) if weights_only else np.nan_to_num(flat, nan=0.0)
        return (w @ src).astype(np.float32)

    def build(self, ts: np.ndarray, ss: np.ndarray) -> dict:
        ts = np.ascontiguousarray(ts, dtype=np.int64)
        ss = np.ascontiguousarray(ss, dtype=np.int64)
        d = ts // N_TOD
        h = ts % N_TOD
        D, S = self.D, self.S
        col = h * S + ss
        row_col = d * (N_TOD * S) + col
        out = {}
        out["ds_sim_s"] = self.sim_s.reshape(-1)[row_col]
        out["ds_sim_q"] = self.sim_q.reshape(-1)[row_col]
        out["ds_sim_adj_s"] = out["ds_sim_s"] + self.dmean_s[d, ss] - self.wdmean_s[d, ss]
        out["ds_sim_adj_q"] = out["ds_sim_q"] + self.dmean_q[d, ss] - self.wdmean_q[d, ss]
        out["ds_w"] = self.den_s.reshape(-1)[row_col]
        out["ds_allday_s"] = self.allday_s[col]
        out["ds_allday_q"] = self.allday_q[col]
        out["ds_prev1_s"] = self._gather_day(self.sp, d - 1, h, ss)
        out["ds_next1_s"] = self._gather_day(self.sp, d + 1, h, ss)
        out["ds_prev7_s"] = self._gather_day(self.sp, d - 7, h, ss)
        out["ds_next7_s"] = self._gather_day(self.sp, d + 7, h, ss)
        out["ds_prev1_q"] = self._gather_day(self.fl, d - 1, h, ss)
        out["ds_next1_q"] = self._gather_day(self.fl, d + 1, h, ss)
        out["ds_corr_max"] = self.corr_max[d]
        return {k: np.asarray(v, np.float32) for k, v in out.items()}

    def _gather_day(self, tensor, dd, h, ss):
        ok = (dd >= 0) & (dd < self.D)
        vals = tensor[np.clip(dd, 0, self.D - 1), h, ss]
        vals = np.asarray(vals, np.float32)
        vals[~ok] = np.nan
        return vals

    def extra_names(self):
        return list(self.names)


def day_dow(dow_time: np.ndarray, days: np.ndarray, n_days: int) -> np.ndarray:
    """Day-of-week per day (from any time index of that day)."""
    out = np.zeros(n_days, dtype=np.int8)
    seen = np.zeros(n_days, dtype=bool)
    for i, dd in enumerate(days):
        if not seen[dd]:
            out[dd] = dow_time[i]
            seen[dd] = True
    return out


class LRTFeatures:
    """Low-rank soft-impute completion of the [day, time-of-day*link] matrix.

    M is (D, 288*S) with NaN at unobserved cells.  Iteration: rank-r SVD of the
    filled matrix, then reset observed entries to their observed values.  Uses a
    randomized SVD so an iteration costs O(D*HS*r) instead of O(D^2*HS).
    Returns the completed estimate at the requested rows.
    """

    def __init__(self, masked: dict, rank: int = 12, iters: int = 10, seed: int = 42,
                 channel_rel_tol: float = 0.0):
        sp = masked["speed"].reshape(masked["speed"].shape[0] // N_TOD, N_TOD * masked["speed"].shape[1])
        fl = masked["flow"].reshape(masked["flow"].shape[0] // N_TOD, N_TOD * masked["flow"].shape[1])
        self.S = masked["speed"].shape[1]
        self.D = sp.shape[0]
        self.tensor_shape = (self.D, N_TOD, self.S)
        self.est_s = self._soft_impute(sp, rank, iters, seed)
        self.est_q = self._soft_impute(fl, rank, iters, seed + 1)
        self.rank, self.iters = rank, iters
        self.names = ["lrt_s", "lrt_q"]

    @staticmethod
    def _soft_impute(M, rank, iters, seed):
        from sklearn.utils.extmath import randomized_svd

        mask = np.isfinite(M)
        colmean = np.where(mask.any(axis=0),
                           np.nansum(M, axis=0) / np.maximum(mask.sum(axis=0), 1), np.nan)
        global_mean = np.nanmean(colmean)
        colmean = np.where(np.isfinite(colmean), colmean, global_mean)
        filled = np.where(mask, M, colmean[None, :]).astype(np.float32)
        obs_vals = np.where(mask, M, 0.0).astype(np.float32)
        for it in range(iters):
            U, sv, Vt = randomized_svd(filled, n_components=min(rank, min(filled.shape) - 1),
                                       n_iter=4, random_state=seed + it)
            low = (U * sv) @ Vt
            filled = np.where(mask, obs_vals, low).astype(np.float32)
        return filled

    def build(self, ts, ss):
        d = np.asarray(ts, np.int64) // N_TOD
        h = np.asarray(ts, np.int64) % N_TOD
        col = h * self.S + np.asarray(ss, np.int64)
        flat = d * (N_TOD * self.S) + col
        return {"lrt_s": self.est_s.reshape(-1)[flat].astype(np.float32),
                "lrt_q": self.est_q.reshape(-1)[flat].astype(np.float32)}

    def extra_names(self):
        return list(self.names)

