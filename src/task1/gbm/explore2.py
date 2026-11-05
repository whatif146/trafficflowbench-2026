"""Error structure analysis of the temporal-interp baseline on train holdout.

1. Measurement-noise floor estimate (predict observed cells from neighbours,
   excluding the cell itself).
2. Error distribution at target cells: identify heavy tails / hard cells.
3. Try better local estimators: temporal local-linear, kernel smoothing, and
   spatial smoothing of the filled field.
"""
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import traffic  # noqa: E402

from explore1 import day_index, tod_index, temporal_neighbours, temporal_interp, score, fmt, HO  # noqa: E402


def fill_without_self(arr):
    """Temporal interp where each observed cell is predicted from its neighbours
    (self excluded), so the error tells us measurement noise + interpolation error."""
    T, S = arr.shape
    obs = np.isfinite(arr)
    idx = np.arange(T, dtype=np.int32)[:, None]
    # previous observed strictly before
    pos = np.where(obs, idx, -1)
    last = np.maximum.accumulate(pos, axis=0)
    # remove self: value at t-1 max
    last_prev = np.vstack([np.full((1, S), -1, dtype=np.int32), last[:-1]])
    nxt = np.minimum.accumulate(np.where(obs, idx, T)[::-1], axis=0)[::-1]
    nxt_next = np.vstack([nxt[1:], np.full((1, S), T, dtype=np.int32)])
    flat = arr.reshape(-1)
    cols = np.arange(S, dtype=np.int64)[None, :]

    def gather(p, valid):
        val = np.full((T, S), np.nan, dtype=np.float32)
        ii = np.where(valid, p, 0).astype(np.int64)
        val[valid] = flat[ii[valid] * S + np.broadcast_to(cols, (T, S))[valid]]
        return val

    b = gather(last_prev, last_prev >= 0)
    a = gather(nxt_next, nxt_next < T)
    dtb = np.where(last_prev >= 0, idx - last_prev, 999).astype(np.float32)
    dta = np.where(nxt_next < T, nxt_next - idx, 999).astype(np.float32)
    both = np.isfinite(b) & np.isfinite(a)
    out = np.where(both, (b * dta + a * dtb) / np.maximum(dtb + dta, 1),
                   np.where(np.isfinite(b), b, a)).astype(np.float32)
    return out, b, a, dtb, dta


def temporal_linreg(arr, half=2):
    """Local linear regression in time on observed values within +-half steps."""
    T, S = arr.shape
    obs = np.isfinite(arr)
    x = np.arange(-half, half + 1, dtype=np.float64)
    num0 = np.zeros((T, S), np.float64)
    num1 = np.zeros((T, S), np.float64)
    den0 = np.zeros((T, S), np.float64)  # sum w
    den1 = np.zeros((T, S), np.float64)  # sum w x
    den2 = np.zeros((T, S), np.float64)  # sum w x^2
    yv = np.where(obs, arr, 0.0).astype(np.float64)
    for k in range(-half, half + 1):
        # contribution of source time t+k to target rows t: shift
        if k >= 0:
            src0, src1 = 0, T - k
            dst0, dst1 = k, T
        else:
            src0, src1 = -k, T
            dst0, dst1 = 0, T + k
        o = obs[src0:src1]
        v = yv[src0:src1]
        w = x[k + half]
        num0[dst0:dst1] += o * v
        num1[dst0:dst1] += w * o * v
        den0[dst0:dst1] += o
        den1[dst0:dst1] += w * o
        den2[dst0:dst1] += w * w * o
    # intercept of weighted ls at x=0: (num0*den2 - num1*den1)/(den0*den2-den1^2)
    det = den0 * den2 - den1 * den1
    good = den0 >= 2
    out = np.full((T, S), np.nan, np.float64)
    num = num0 * den2 - num1 * den1
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = np.where(np.abs(det) > 1e-9, num / det, num0 / np.maximum(den0, 1))
    out[good] = vals[good]
    return out.astype(np.float32)


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
    tr = traffic.load_mainline(panel, "train")
    plain, masked = tr["plain"], tr["masked"]
    tg = traffic.load_targets(panel, "train")
    times = plain["times"]
    days = day_index(times)
    tod = tod_index(times)
    dow = pd.to_datetime(times, unit="s", utc=True).dayofweek.to_numpy().astype(np.int8)
    n_days = days.max() + 1
    hold_day = days >= (n_days - HO)

    t_idx, s_idx, treg = tg["t_idx"], tg["s_idx"], tg["regime"] + 1
    finite = np.isfinite(plain["speed"][t_idx, s_idx]) & np.isfinite(plain["flow"][t_idx, s_idx])
    h = hold_day[t_idx] & finite
    ts, ss, reg = t_idx[h], s_idx[h], treg[h]
    true_s = plain["speed"][ts, ss].astype(np.float64)
    true_q = plain["flow"][ts, ss].astype(np.float64)
    lanes = plain["lanes"][ss].astype(np.float64)

    # ---------- 1. noise floor ----------
    print("=== measurement-noise / self-excluded interp error on OBSERVED holdout cells ===")
    for name, arr in (("speed", masked["speed"]), ("flow", masked["flow"])):
        pred, b, a, dtb, dta = fill_without_self(arr)
        m = hold_day[:, None] & np.isfinite(arr)
        # restrict to a sample of observed cells (rows of template? no - all observed)
        ii = np.where(m)
        rng = np.random.default_rng(0)
        take = rng.choice(len(ii[0]), size=min(400000, len(ii[0])), replace=False)
        tt, uu = ii[0][take], ii[1][take]
        p = pred[tt, uu]
        ok = np.isfinite(p)
        t_ = arr[tt, uu].astype(np.float64)
        e = p[ok] - t_[ok]
        gap = (dtb[tt, uu] + dta[tt, uu])[ok]
        print("%s: n=%d rmse=%.4f mae=%.4f | gap=2: rmse=%.4f mae=%.4f | gap>2: rmse=%.4f" % (
            name, ok.sum(), np.sqrt(np.mean(e * e)), np.mean(np.abs(e)),
            np.sqrt(np.mean(e[gap == 2] ** 2)), np.mean(np.abs(e[gap == 2])),
            np.sqrt(np.mean(e[gap > 2] ** 2)) if (gap > 2).any() else float("nan")))

    # ---------- 2. error distribution of temporal interp at target cells ----------
    sp_i, nb_s = temporal_interp(masked["speed"])
    q_i, nb_q = temporal_interp(masked["flow"])
    pred_s = sp_i[ts, ss].astype(np.float64)
    pred_q = q_i[ts, ss].astype(np.float64)
    res = score(true_s, true_q, pred_s, pred_q, lanes, reg)
    print("\n[temporal interp]", fmt(res))
    es = pred_s - true_s
    eq = pred_q - true_q
    print("speed err percentiles (abs):", np.percentile(np.abs(es), [50, 75, 90, 95, 99, 99.9]))
    print("flow err percentiles (abs):", np.percentile(np.abs(eq), [50, 75, 90, 95, 99, 99.9]))
    sq = es * es
    print("speed SSE share: top1%% %.3f top5%% %.3f top10%% %.3f" % (
        np.sort(sq)[::-1][: int(len(sq) * 0.01)].sum() / sq.sum(),
        np.sort(sq)[::-1][: int(len(sq) * 0.05)].sum() / sq.sum(),
        np.sort(sq)[::-1][: int(len(sq) * 0.10)].sum() / sq.sum()))
    print("flow SSE share: top1%% %.3f top5%% %.3f top10%% %.3f" % (
        np.sort(eq ** 2)[::-1][: int(len(eq) * 0.01)].sum() / (eq ** 2).sum(),
        np.sort(eq ** 2)[::-1][: int(len(eq) * 0.05)].sum() / (eq ** 2).sum(),
        np.sort(eq ** 2)[::-1][: int(len(eq) * 0.10)].sum() / (eq ** 2).sum()))

    # hard cells: where neighbours disagree
    b, a = nb_s["before"][ts, ss], nb_s["after"][ts, ss]
    dis = np.abs(b - a)
    for lo, hi in [(0, 1), (1, 3), (3, 8), (8, 20), (20, 1e9)]:
        m = (dis >= lo) & (dis < hi)
        if m.any():
            print("  neighbour speed disagreement [%.0f,%.0f): n=%d rmse=%.2f" % (
                lo, hi, m.sum(), np.sqrt(np.mean(es[m] ** 2))))
    # free speed ratio
    fs = plain["free_speed"][ss]
    for lo, hi in [(0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        ratio = true_s / fs
        m = (ratio >= lo) & (ratio < hi)
        if m.any():
            print("  true speed/free_speed [%.1f,%.1f): n=%d rmse=%.2f bias=%.2f" % (
                lo, hi, m.sum(), np.sqrt(np.mean(es[m] ** 2)), es[m].mean()))

    # ---------- 3. alternative estimators ----------
    print("\n=== alternative temporal estimators ===")
    for half in (1, 2, 3):
        p = temporal_linreg(masked["speed"], half=half)[ts, ss]
        q = temporal_linreg(masked["flow"], half=half)[ts, ss]
        res = score(true_s, true_q, p, q, lanes, reg)
        print("linreg half=%d" % half, fmt(res))

    # spatial smoothing of the filled field (fill = observed where present else interp)
    filled_s = np.where(np.isfinite(masked["speed"]), masked["speed"], sp_i)
    filled_q = np.where(np.isfinite(masked["flow"]), masked["flow"], q_i)
    order = np.argsort(plain["milepost"])
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    inv = np.argsort(rank)
    for alpha in (0.9, 0.8, 0.7, 0.5):
        # neighbour mean in link order (up+down)
        nb_mean_s = np.zeros_like(filled_s)
        cnt = np.zeros(filled_s.shape, dtype=np.float32)
        for d in (-1, 1):
            shifted = np.full_like(filled_s, np.nan)
            if d == -1:
                shifted[:, inv[1:]] = filled_s[:, inv[:-1]]
            else:
                shifted[:, inv[:-1]] = filled_s[:, inv[1:]]
            ok = np.isfinite(shifted)
            nb_mean_s += np.where(ok, shifted, 0)
            cnt += ok
        nb_mean_s = np.where(cnt > 0, nb_mean_s / np.maximum(cnt, 1), filled_s)
        same = np.zeros_like(filled_q)
        cntq = np.zeros(filled_q.shape, dtype=np.float32)
        for d in (-1, 1):
            shifted = np.full_like(filled_q, np.nan)
            if d == -1:
                shifted[:, inv[1:]] = filled_q[:, inv[:-1]]
            else:
                shifted[:, inv[:-1]] = filled_q[:, inv[1:]]
            ok = np.isfinite(shifted)
            same += np.where(ok, shifted, 0)
            cntq += ok
        same = np.where(cntq > 0, same / np.maximum(cntq, 1), filled_q)
        p = alpha * filled_s[ts, ss] + (1 - alpha) * nb_mean_s[ts, ss]
        q = alpha * filled_q[ts, ss] + (1 - alpha) * same[ts, ss]
        res = score(true_s, true_q, p, q, lanes, reg)
        print("spatial smooth alpha=%.1f" % alpha, fmt(res))


if __name__ == "__main__":
    main()

