"""Compare candidate base estimators (speed and flow) on holdout target cells.

Base candidates:
  - temporal interp
  - Gaussian spatio-temporal smoother at several bandwidths
  - two-stage: fill masked cells with interp, then smooth the filled field
  - interp + spatial-neighbour blend
Uses D7_I10_E train, last --holdout-days days, all target cells (no subsample).
"""
import sys
import time

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task1\gbm")
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import traffic  # noqa: E402

from t1lib import (day_index, tod_index, dow_index, link_perm, smooth_field,  # noqa: E402
                   temporal_interp, score, fmt)

HOLD = 60


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
    t0 = time.time()
    tr = traffic.load_mainline(panel, "train")
    plain, masked = tr["plain"], tr["masked"]
    tg = traffic.load_targets(panel, "train")
    times = masked["times"]
    days = day_index(times)
    n_days = int(days.max()) + 1
    ts = tg["t_idx"].astype(np.int64)
    ss = tg["s_idx"].astype(np.int64)
    reg = tg["regime"] + 1
    day_r = days[ts]
    fin = np.isfinite(plain["speed"][ts, ss]) & np.isfinite(plain["flow"][ts, ss])
    sel = (day_r >= n_days - HOLD) & fin
    ts, ss, reg = ts[sel], ss[sel], reg[sel]
    true_s = plain["speed"][ts, ss].astype(np.float64)
    true_q = plain["flow"][ts, ss].astype(np.float64)
    lanes = plain["lanes"][ss].astype(np.float64)
    print(panel, "holdout cells", len(ts), "avail", flush=True)
    order_p, inv = link_perm(masked)
    del plain
    import gc
    gc.collect()

    def ev(pred_s, pred_q, label):
        res = score(true_s, true_q, pred_s, pred_q, lanes, reg)
        print("%-34s %s  [%.1fs]" % (label, fmt(res), time.time() - t0), flush=True)

    # ---- 1. temporal interp ----
    sp_i = temporal_interp(masked["speed"])
    q_i = temporal_interp(masked["flow"])
    ev(sp_i[ts, ss], q_i[ts, ss], "interp")

    # ---- 2. Gaussian smoothers ----
    for st, ssz, order in ((1.0, 0.0, 1), (1.5, 0.0, 1), (2.0, 0.0, 1),
                           (1.0, 1.0, 1), (1.5, 1.0, 1), (2.0, 1.0, 1), (3.0, 1.0, 1),
                           (1.5, 1.5, 1), (2.0, 1.0, 0)):
        fs = smooth_field(masked["speed"], st, ssz, order=order, order_p=order_p, inv=inv)
        fq = smooth_field(masked["flow"], st, ssz, order=order, order_p=order_p, inv=inv)
        ev(fs[ts, ss], fq[ts, ss], "smooth_t%.1f_s%.1f_o%d" % (st, ssz, order))
        del fs, fq
        gc.collect()

    # ---- 3. fill interp then smooth filled field (local mean over observed+filled) ----
    filled_s = np.where(np.isfinite(masked["speed"]), masked["speed"], sp_i)
    filled_q = np.where(np.isfinite(masked["flow"]), masked["flow"], q_i)
    for st, ssz in ((1.0, 1.0), (1.5, 1.0), (2.0, 1.0)):
        fs = smooth_field(filled_s, st, ssz, order=0, order_p=order_p, inv=inv)
        fq = smooth_field(filled_q, st, ssz, order=0, order_p=order_p, inv=inv)
        ev(fs[ts, ss], fq[ts, ss], "fill+smooth_t%.1f_s%.1f" % (st, ssz))
        del fs, fq
        gc.collect()

    # ---- 4. interp + spatial neighbour blend (raw masked adjacent links) ----
    def spatial_nb_mean(arr):
        # mean of observed same-time neighbours at link ranks +-1, +-2 (NaN if none)
        acc = np.zeros_like(arr, dtype=np.float32)
        cnt = np.zeros(arr.shape, dtype=np.float32)
        for d in (-2, -1, 1, 2):
            pos = inv + d
            ok = (pos >= 0) & (pos < len(order_p))
            col = np.where(ok, order_p[np.clip(pos, 0, len(order_p) - 1)], 0)
            v = arr[:, col]
            good = ok[None, :] & np.isfinite(v)
            acc += np.where(good, v, 0.0)
            cnt += good
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)

    nb_s = spatial_nb_mean(masked["speed"])
    nb_q = spatial_nb_mean(masked["flow"])
    print("  spatial neighbour means built [%.1fs]" % (time.time() - t0), flush=True)
    for w in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        ps = np.where(np.isfinite(nb_s[ts, ss]), w * sp_i[ts, ss] + (1 - w) * nb_s[ts, ss], sp_i[ts, ss])
        pq = np.where(np.isfinite(nb_q[ts, ss]), w * q_i[ts, ss] + (1 - w) * nb_q[ts, ss], q_i[ts, ss])
        ev(ps, pq, "interp+spatialnb w=%.1f" % w)

    # ---- 5. smoother blended with interp ----
    fs = smooth_field(masked["speed"], 1.5, 1.0, order=1, order_p=order_p, inv=inv)
    fq = smooth_field(masked["flow"], 1.5, 1.0, order=1, order_p=order_p, inv=inv)
    for w in (0.7, 0.5, 0.3):
        ps = np.where(np.isfinite(fs[ts, ss]), w * sp_i[ts, ss] + (1 - w) * fs[ts, ss], sp_i[ts, ss])
        pq = np.where(np.isfinite(fq[ts, ss]), w * q_i[ts, ss] + (1 - w) * fq[ts, ss], q_i[ts, ss])
        ev(ps, pq, "interp+smooth w=%.1f" % w)


if __name__ == "__main__":
    main()
