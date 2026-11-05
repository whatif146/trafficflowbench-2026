"""Grid over spatio-temporal smoother bandwidths for speed and flow on holdout."""
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task1\gbm")
import numpy as np  # noqa: E402
from t1lib import (load_train_holdout, link_perm, smooth_field, temporal_interp,  # noqa: E402
                   score, fmt)


def main():
    panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
    d = load_train_holdout(panel)
    plain, masked = d["plain"], d["masked"]
    ts, ss, reg = d["t_idx"], d["s_idx"], d["regime"]
    true_s, true_q, lanes = d["true_speed"], d["true_flow"], d["lanes"]
    order_p, inv = link_perm(plain)
    print(panel, "holdout cells", len(ts))

    sp_i = temporal_interp(masked["speed"])
    q_i = temporal_interp(masked["flow"])
    res = score(true_s, true_q, sp_i[ts, ss], q_i[ts, ss], lanes, reg)
    print("[interp]", fmt(res))

    print("\n=== speed smoother grid (order=1) ===")
    best_s = (0, None)
    for st in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        for ssz in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            f = smooth_field(masked["speed"], st, ssz, order=1, order_p=order_p, inv=inv)
            pred = f[ts, ss]
            nanfrac = float(np.isnan(pred).mean())
            rr = score(true_s, true_q, pred.astype(np.float64), np.zeros_like(true_q), lanes, reg)
            sm = float(np.mean([rr[r]["S_speed"] for r in (1, 2, 3)]))
            if sm > best_s[0]:
                best_s = (sm, (st, ssz))
            print("  st=%.1f ss=%.1f S_speed=%.4f per-regime %s nan=%.4f" % (
                st, ssz, sm, ["%.3f" % rr[r]["S_speed"] for r in (1, 2, 3)], nanfrac))
    print("BEST speed", best_s)

    print("\n=== flow smoother grid (order=1) ===")
    best_q = (0, None)
    for st in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        for ssz in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
            f = smooth_field(masked["flow"], st, ssz, order=1, order_p=order_p, inv=inv)
            pred = f[ts, ss]
            rr = score(true_s, true_q, true_s.copy(), pred.astype(np.float64), lanes, reg)
            sm = float(np.mean([rr[r]["S_flow"] for r in (1, 2, 3)]))
            if sm > best_q[0]:
                best_q = (sm, (st, ssz))
            print("  st=%.1f ss=%.1f S_flow=%.4f per-regime %s" % (
                st, ssz, sm, ["%.3f" % rr[r]["S_flow"] for r in (1, 2, 3)]))
    print("BEST flow", best_q)

    # best combination
    ss_, (st_s, ssz_s) = best_s
    fs = smooth_field(masked["speed"], st_s, ssz_s, order=1, order_p=order_p, inv=inv)
    fq = smooth_field(masked["flow"], best_q[1][0], best_q[1][1], order=1, order_p=order_p, inv=inv)
    res = score(true_s, true_q, fs[ts, ss].astype(np.float64), fq[ts, ss].astype(np.float64), lanes, reg)
    print("\n[BEST combo]", fmt(res))


if __name__ == "__main__":
    main()
