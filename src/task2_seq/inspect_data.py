"""Inspect the released layers for the long-history Task 2 attempt."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import traffic

PANEL = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "validation"

print("== window index ==")
w = traffic.load_window_index()
print(w.columns.tolist())
print(w[w.panel == PANEL].groupby(["split", "condition"]).size())
wu = w[(w.panel == PANEL) & (w.split == SPLIT)].sort_values("forecast_origin")
print(wu[["window_id", "condition", "forecast_origin", "history_start", "forecast_start", "forecast_end", "history_coverage"]].to_string())

print("\n== split masked layer ==")
m = traffic.load_mainline(PANEL, SPLIT)["masked"]
print("keys:", sorted(m.keys()))
T, S = m["speed"].shape
print("shape", m["speed"].shape, "times", m["times"][0], m["times"][-1])
nan_frac = float(np.isnan(m["speed"]).mean())
print("speed nan frac", round(nan_frac, 4), "elig frac", round(float(m["elig"].mean()), 4),
      "mean pct", round(float(m["pct"].mean()), 1))
# nan fraction by time of day
dt = pd.to_datetime(m["times"], unit="s", utc=True)
tod = (dt.hour * 60 + dt.minute) // 5
nf = np.isnan(m["speed"]).mean(axis=1)
print("nan frac by tod quartiles:", np.round(np.percentile(nf, [0, 5, 25, 50, 75, 95, 100]), 3))
# row at origin T: check observed
times = m["times"]
for r in wu.itertuples(index=False):
    t0 = int(pd.Timestamp(r.forecast_origin).timestamp())
    i = int(np.searchsorted(times, t0))
    assert times[i] == t0, (r.window_id, times[i], t0)
    row = m["speed"][i]
    # fractions of observed cells in [i-36, i]
    seg = m["speed"][max(0, i - 35):i + 1]
    print(f"{r.window_id} {r.condition} origin_i={i} nan@T={np.isnan(row).mean():.2f} "
          f"nan_hist36={np.isnan(seg).mean():.2f} elig_hist36={m['elig'][max(0,i-35):i+1].mean():.2f}")

print("\n== window_history vs masked layer ==")
h = traffic.load_window_history()
hp = h[h.window_id.isin(set(wu.window_id)) & (h.get("split", SPLIT) == SPLIT)] if "split" in h.columns else h[h.window_id.isin(set(wu.window_id))]
print("history cols", h.columns.tolist(), "rows", len(hp))
r = wu.iloc[0]
hh = hp[hp.window_id == r.window_id]
print("history times", hh.timestamp.min(), hh.timestamp.max())
t0 = int(pd.Timestamp(r.forecast_origin).timestamp())
i = int(np.searchsorted(times, t0))
# map history rows to masked layer
hh_t = np.array([int(pd.Timestamp(t).timestamp()) for t in hh.timestamp])
print("history offsets(min):", np.unique((hh_t - t0) // 60))
links = m["links"].astype(str)
li = {l: k for k, l in enumerate(links)}
sel = hh[hh.link_id.isin(li)]
ti = np.array([int(np.searchsorted(times, int(pd.Timestamp(t).timestamp()))) for t in sel.timestamp])
si = np.array([li[l] for l in sel.link_id.astype(str)])
mv = m["speed"][ti, si]
hv = pd.to_numeric(sel.speed_kmh, errors="coerce").to_numpy(float)
both = np.isfinite(mv) & np.isfinite(hv)
print("hist cells", len(hv), "masked observed frac", np.isfinite(mv).mean(),
      "corr(equal?)", np.corrcoef(mv[both], hv[both])[0, 1] if both.sum() > 10 else None,
      "max abs diff", float(np.nanmax(np.abs(mv[both] - hv[both]))) if both.sum() else None)

print("\n== train plain vs train masked ==")
tr = traffic.load_mainline(PANEL, "train")
print("plain nan", round(float(np.isnan(tr["plain"]["speed"]).mean()), 4),
      "masked nan", round(float(np.isnan(tr["masked"]["speed"]).mean()), 4))
# same timestamp grid?
print("same times:", np.array_equal(tr["plain"]["times"], tr["masked"]["times"]))

print("\n== dense_gbm ==")
for dname in ("dense_gbm", "dense_smooth"):
    p = os.path.join(r"C:\Users\lenovo\Desktop\kaggle\work\submissions", dname,
                     f"{PANEL}_{SPLIT}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        print(dname, {k: (z[k].shape, str(z[k].dtype)) for k in z.files})
        for k in z.files:
            a = z[k]
            if a.ndim == 2 and a.dtype.kind == "f":
                print("  ", k, "nan frac", round(float(np.isnan(a).mean()), 4))
        z.close()
