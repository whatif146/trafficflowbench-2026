"""Quick exploration of Task 1 data and cheap estimators on one panel (train).

Holdout: last HO days of train (default 56 = 8 weeks). Scores only the target
rows (published template) inside the holdout, with the official Task 1 formula.

Estimators:
  - historical weekday x time-of-day profile (shipped baseline style, fit on the
    remaining days' plain layer)
  - temporal interpolation from the masked view (nearest before/after)
  - spatial neighbour copy
  - simple blends
"""
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import traffic  # noqa: E402

HO = 56


def day_index(times):
    return ((times - times[0]) // 86400).astype(np.int64)


def tod_index(times):
    return (((times - times[0]) % 86400) // 300).astype(np.int64)


def temporal_neighbours(arr):
    """Nearest observed value before/after along time for each (t, s)."""
    T, S = arr.shape
    obs = np.isfinite(arr)
    idx = np.arange(T, dtype=np.int32)[:, None]
    last = np.maximum.accumulate(np.where(obs, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(obs, idx, T)[::-1], axis=0)[::-1]
    flat = arr.reshape(-1)
    cols = np.arange(S, dtype=np.int64)[None, :]

    def gather(pos, valid):
        val = np.full((T, S), np.nan, dtype=np.float32)
        ii = np.where(valid, pos, 0).astype(np.int64)
        val[valid] = flat[ii[valid] * S + np.broadcast_to(cols, (T, S))[valid]]
        return val

    before = gather(last, last >= 0)
    after = gather(nxt, nxt < T)
    dtb = np.where(last >= 0, idx - last, 999).astype(np.float32)
    dta = np.where(nxt < T, nxt - idx, 999).astype(np.float32)
    return {"before": before, "after": after, "dt_before": dtb, "dt_after": dta}


def temporal_interp(arr):
    nb = temporal_neighbours(arr)
    b, a = nb["before"], nb["after"]
    both = np.isfinite(b) & np.isfinite(a)
    out = np.where(both, (b * nb["dt_after"] + a * nb["dt_before"]) / np.maximum(nb["dt_before"] + nb["dt_after"], 1),
                   np.where(np.isfinite(b), b, a)).astype(np.float32)
    return out, nb


def score(true_s, true_q, pred_s, pred_q, lanes, regimes):
    """Official Task 1 formula; regimes are 1..3."""
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


def fmt(res):
    s = "macro S_state %.4f" % res["macro"]
    for r in (1, 2, 3):
        s += " | R%d n=%d S=%.4f (s %.3f q %.3f rmse %.2f/%.0f)" % (
            r, res[r]["n"], res[r]["S_state"], res[r]["S_speed"], res[r]["S_flow"],
            res[r]["rmse_speed"], res[r]["rmse_flow"])
    return s


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
    day_reg = np.zeros(n_days, dtype=np.int8)
    for d in range(n_days):
        r = masked["regime"][days == d]
        day_reg[d] = r[r > 0][0]
    print("panel", panel, "T", len(times), "S", len(plain["stations"]), "days", n_days,
          "plain nan %.4f masked nan %.4f elig %.4f" % (
              np.isnan(plain["speed"]).mean(), np.isnan(masked["speed"]).mean(), plain["elig"].mean()))
    print("day regimes:", np.bincount(day_reg[1:4]), "(R1,R2,R3)")

    t_idx, s_idx, treg = tg["t_idx"], tg["s_idx"], tg["regime"] + 1  # 1..3
    tday = days[t_idx]
    hold = tday >= (n_days - HO)
    print("target rows %d holdout rows %d" % (len(t_idx), hold.sum()))
    print("holdout day regimes:", np.bincount(day_reg[n_days - HO:]))

    finite = np.isfinite(plain["speed"][t_idx, s_idx]) & np.isfinite(plain["flow"][t_idx, s_idx])
    h = hold & finite
    ts, ss, reg = t_idx[h], s_idx[h], treg[h]
    true_s = plain["speed"][ts, ss].astype(np.float64)
    true_q = plain["flow"][ts, ss].astype(np.float64)
    lanes = plain["lanes"][ss].astype(np.float64)
    print("scored holdout cells", h.sum(), "per regime", [int((reg == r).sum()) for r in (1, 2, 3)])

    # ---- estimator 1: temporal interpolation from masked view ----
    sp_i, nb_s = temporal_interp(masked["speed"])
    q_i, nb_q = temporal_interp(masked["flow"])
    res = score(true_s, true_q, sp_i[ts, ss].astype(np.float64), q_i[ts, ss].astype(np.float64), lanes, reg)
    print("\n[temporal interp]", fmt(res))

    # ---- estimator 2: historical weekday x tod profile from fit days (plain) ----
    fit_day_mask = days < (n_days - HO)
    prof_s = np.full((7, 288, plain["speed"].shape[1]), np.nan, dtype=np.float64)
    prof_q = np.full_like(prof_s, np.nan)
    for w in range(7):
        for hh in range(288):
            m = fit_day_mask & (dow == w) & (tod == hh)
            if not m.any():
                continue
            prof_s[w, hh] = np.nanmean(plain["speed"][m], axis=0)
            prof_q[w, hh] = np.nanmean(plain["flow"][m], axis=0)
    pred_s = prof_s[dow[ts], tod[ts], ss]
    pred_q = prof_q[dow[ts], tod[ts], ss]
    res = score(true_s, true_q, pred_s, pred_q, lanes, reg)
    print("[historical mean]", fmt(res))

    # ---- estimator 3: spatial neighbour (same timestamp, adjacent in link order) ----
    order = np.argsort(plain["milepost"])
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    inv = np.argsort(rank)
    sp_obs, q_obs = masked["speed"], masked["flow"]
    up_s = np.full((len(times), len(rank)), np.nan, dtype=np.float32)
    up_q = np.full_like(up_s, np.nan)
    up_s[:, inv[1:]] = sp_obs[:, inv[:-1]]
    up_q[:, inv[1:]] = q_obs[:, inv[:-1]]
    res = score(true_s, true_q, up_s[ts, ss].astype(np.float64), up_q[ts, ss].astype(np.float64), lanes, reg)
    print("[spatial up copy]", fmt(res))

    # ---- blend interp + profile ----
    for w in (0.95, 0.9, 0.8, 0.7, 0.6):
        bs = w * sp_i[ts, ss] + (1 - w) * pred_s
        bq = w * q_i[ts, ss] + (1 - w) * pred_q
        res = score(true_s, true_q, bs, bq, lanes, reg)
        print("[blend %.2f interp]" % w, fmt(res))

    # gap structure at target cells
    dtb = nb_s["dt_before"][ts, ss]
    dta = nb_s["dt_after"][ts, ss]
    gap = dtb + dta
    print("\ngap(dtb+dta) pct [50,75,90,95,99]:", np.percentile(gap, [50, 75, 90, 95, 99]))
    print("dtb pct [50,75,90,95,99]:", np.percentile(dtb, [50, 75, 90, 95, 99]))
    err = np.abs(sp_i[ts, ss] - true_s)
    for g in [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 1000)]:
        m = (gap >= g[0]) & (gap < g[1])
        if m.any():
            print("  gap[%d,%d): n=%d mae_s=%.2f  bias=%.2f" % (
                g[0], g[1], m.sum(), err[m].mean(), (sp_i[ts, ss][m] - true_s[m]).mean()))


if __name__ == "__main__":
    main()
