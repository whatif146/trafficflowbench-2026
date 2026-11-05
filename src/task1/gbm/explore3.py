"""Spatio-temporal kernel smoothers for Task 1 on one panel (train holdout).

- measurement-noise estimates from consecutive observed pairs
- separable Gaussian Nadaraya-Watson / local-linear-in-time smoothers on the
  masked field, evaluated at holdout target cells
- density (k=q/v) consistency variant for flow
"""
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import traffic  # noqa: E402

from explore1 import day_index, tod_index, temporal_interp, score, fmt, HO  # noqa: E402

from scipy.ndimage import gaussian_filter1d  # noqa: E402


def link_perm(obs):
    """perm/back-perm to put stations in spatial (milepost) order."""
    order = np.argsort(obs["milepost"])
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    inv = np.argsort(rank)
    return order, inv


def spatial_smooth(a, sigma_s, inv):
    """Gaussian smooth along the link-order axis. a is (T,S)."""
    if sigma_s <= 0:
        return a
    b = gaussian_filter1d(a, sigma_s, axis=1, mode="nearest")
    # reorder columns back: currently columns are in spatial order
    out = np.empty_like(b)
    out[:, inv] = b
    return out


def smooth_field(y, sigma_t, sigma_s, order=0, inv=None):
    """Weighted local mean (order 0) or local-linear-in-time intercept (order 1).

    Uses observed cells of ``y`` (NaN = unobserved) only.  Returns (T,S) float32.
    """
    T, S = y.shape
    m = np.isfinite(y).astype(np.float32)
    yv = np.where(np.isfinite(y), y, 0.0).astype(np.float32)
    if inv is not None and sigma_s > 0:
        sp = np.empty_like(m)
        sp[:, inv] = m
        sp = spatial_smooth(sp, sigma_s, inv)
        yp = np.empty_like(yv)
        yp[:, inv] = yv
        yp = spatial_smooth(yp, sigma_s, inv)
    else:
        sp, yp = m, yv
    if sigma_t <= 0:
        return np.where(sp > 0, yp / np.maximum(sp, 1e-6), np.nan).astype(np.float32)
    K = int(np.ceil(3 * sigma_t))
    ks = np.arange(-K, K + 1)
    w = np.exp(-0.5 * (ks / sigma_t) ** 2).astype(np.float32)

    def pad(a):
        return np.pad(a, ((K, K), (0, 0)), mode="constant", constant_values=0.0)

    sp_p, yp_p = pad(sp), pad(yp)
    S0 = np.zeros((T, S), np.float32)
    Sy = np.zeros((T, S), np.float32)
    if order == 0:
        for i, k in enumerate(ks):
            sl = wp = w[i]
            S0 += wp * sp_p[K + k: K + k + T]
            Sy += wp * yp_p[K + k: K + k + T]
        out = np.where(S0 > 1e-4, Sy / np.maximum(S0, 1e-6), np.nan)
        return out.astype(np.float32)
    St = np.zeros((T, S), np.float32)
    Stt = np.zeros((T, S), np.float32)
    Sty = np.zeros((T, S), np.float32)
    for i, k in enumerate(ks):
        wp = w[i]
        a = sp_p[K + k: K + k + T]
        ax = a * np.float32(k)
        S0 += wp * a
        St += wp * ax
        Stt += wp * np.float32(k) * ax
        b = yp_p[K + k: K + k + T]
        Sy += wp * b
        Sty += wp * np.float32(k) * b
    det = S0 * Stt - St * St
    num = Sy * Stt - Sty * St
    good = np.abs(det) > 1e-5
    out = np.full((T, S), np.nan, np.float32)
    out[good] = (num[good] / det[good]).astype(np.float32)
    # fallback to local mean where the design is degenerate (single-sided tiny support)
    out = np.where(np.isfinite(out), out, np.where(S0 > 1e-4, Sy / np.maximum(S0, 1e-6), np.nan))
    return out.astype(np.float32)


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
    tr = traffic.load_mainline(panel, "train")
    plain, masked = tr["plain"], tr["masked"]
    tg = traffic.load_targets(panel, "train")
    times = plain["times"]
    days = day_index(times)
    tod = tod_index(times)
    n_days = days.max() + 1
    hold_day = days >= (n_days - HO)
    order_p, inv = link_perm(plain)

    t_idx, s_idx, treg = tg["t_idx"], tg["s_idx"], tg["regime"] + 1
    finite = np.isfinite(plain["speed"][t_idx, s_idx]) & np.isfinite(plain["flow"][t_idx, s_idx])
    h = hold_day[t_idx] & finite
    ts, ss, reg = t_idx[h], s_idx[h], treg[h]
    true_s = plain["speed"][ts, ss].astype(np.float64)
    true_q = plain["flow"][ts, ss].astype(np.float64)
    lanes = plain["lanes"][ss].astype(np.float64)

    # ---------- noise estimates from consecutive observed pairs ----------
    print("=== noise estimates (std of 5-min differences / sqrt2, by time-of-day) ===")
    for name, arr, lanes_flag in (("speed", masked["speed"], False), ("flow", masked["flow"], True),
                                  ("density", None, False)):
        if name == "density":
            with np.errstate(invalid="ignore", divide="ignore"):
                arr = (masked["flow"] / np.maximum(masked["speed"], 1e-3)) / plain["lanes"][None, :]
        ok = np.isfinite(arr[:-1]) & np.isfinite(arr[1:])
        d = (arr[1:] - arr[:-1])[ok]
        hh = np.broadcast_to(tod[:-1, None], ok.shape)[ok]
        for lo, hi in ((0, 60), (60, 120), (120, 168), (168, 240), (240, 288)):
            m = (hh >= lo) & (hh < hi)
            if m.sum() > 1000:
                print("  %-12s tod[%3d,%3d): n=%8d sigma~%.3f" % (name, lo, hi, m.sum(), d[m].std() / np.sqrt(2)))

    # ---------- smoother grid ----------
    print("\n=== Gaussian spatio-temporal smoother (order 1 in time) on holdout targets ===")
    best = {}
    for ch, target_true, mask_arr in (("speed", true_s, masked["speed"]), ("flow", true_q, masked["flow"])):
        for st in (1.0, 1.5, 2.0, 3.0, 4.0):
            for ssz in (0.0, 1.0, 2.0):
                f = smooth_field(mask_arr, st, ssz, order=1, inv=inv)
                pred = f[ts, ss].astype(np.float64)
                if ch == "speed":
                    res = score(true_s, true_q, pred, np.full_like(pred, np.nan), lanes, reg)
                else:
                    # evaluate flow only: reuse score with perfect speed
                    res = score(true_s, true_q, true_s.copy(), pred, lanes, reg)
                key = (ch, st, ssz)
                best[key] = res
                print("  %-5s sig_t=%.1f sig_s=%.0f  S_%s=%.4f  per-regime %s" % (
                    ch, st, ssz, "speed" if ch == "speed" else "flow",
                    np.mean([res[r]["S_speed" if ch == "speed" else "S_flow"] for r in (1, 2, 3)]),
                    ["%.4f" % res[r]["S_speed" if ch == "speed" else "S_flow"] for r in (1, 2, 3)]))

    # baseline interp for reference
    sp_i, _ = temporal_interp(masked["speed"])
    q_i, _ = temporal_interp(masked["flow"])
    res = score(true_s, true_q, sp_i[ts, ss], q_i[ts, ss], lanes, reg)
    print("\n[reference temporal interp]", fmt(res))

    # ---------- density-consistent flow ----------
    print("\n=== density q/v consistency for flow ===")
    with np.errstate(invalid="ignore", divide="ignore"):
        k_obs = masked["flow"] / np.maximum(masked["speed"], 1e-3)  # total density proxy (no lanes)
    for st, ssz in ((2.0, 1.0), (3.0, 1.0), (4.0, 1.0)):
        f_k = smooth_field(k_obs, st, ssz, order=1, inv=inv)
        f_v = smooth_field(masked["speed"], st, ssz, order=1, inv=inv)
        pred = f_k[ts, ss] * f_v[ts, ss]
        res = score(true_s, true_q, true_s.copy(), pred.astype(np.float64), lanes, reg)
        print("  q=ksmooth*vsmooth st=%.1f ss=%.0f  S_flow=%.4f per-regime %s" % (
            st, ssz, np.mean([res[r]["S_flow"] for r in (1, 2, 3)]),
            ["%.4f" % res[r]["S_flow"] for r in (1, 2, 3)]))


if __name__ == "__main__":
    main()

