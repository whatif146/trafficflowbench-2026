"""Check released window months, horizon blanking, mask rates, link order."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import traffic

w = traffic.load_window_index().copy()
w["forecast_origin"] = pd.to_datetime(w.forecast_origin, utc=True)
w["month"] = w.forecast_origin.dt.strftime("%Y-%m")
print("== released windows per split/month ==")
print(w.groupby(["split", "month"]).size().unstack(0).fillna(0).astype(int).to_string())
print("\n== train windows per panel/month ==")
print(w[w.split == "train"].groupby(["panel", "month"]).size().unstack().fillna(0).astype(int).to_string())
print("\n== event dates per condition/split (first 3 panels) ==")
print(w[w.split == "train"].head(6)[["panel", "condition", "forecast_origin"]].to_string())

print("\n== horizon blanking check (validation) ==")
P = "D7_I10_E"
m = traffic.load_mainline(P, "validation")["masked"]
times = m["times"]
links = m["links"].astype(str)
print("masked links order == plain links order (train):",
      np.array_equal(links, traffic.load_mainline(P, "train")["plain"]["links"].astype(str)))
for r in w[(w.panel == P) & (w.split == "validation")].itertuples(index=False):
    t0 = int(pd.Timestamp(r.forecast_origin).timestamp())
    i = int(np.searchsorted(times, t0))
    fut = m["speed"][i + 1:i + 7]
    print(r.window_id, r.condition, "future nan frac", round(float(np.isnan(fut).mean()), 3),
          "rowT nan", round(float(np.isnan(m["speed"][i]).mean()), 3),
          "hist12 nan", round(float(np.isnan(m["speed"][i - 12:i]).mean()), 3))

print("\n== mask rates by day (train masked vs val masked) ==")
for split in ("train", "validation", "private"):
    mm = traffic.load_mainline(P, split)["masked"]
    dt = pd.to_datetime(mm["times"], unit="s", utc=True)
    day = dt.strftime("%Y-%m-%d")
    nf = pd.Series(np.isnan(mm["speed"]).mean(axis=1), index=day).groupby(level=0).mean()
    reg = pd.Series(mm["regime"], index=day).groupby(level=0).max()
    print(split, "day nan-frac quantiles", np.round(np.percentile(nf, [0, 10, 50, 90, 100]), 3),
          "regime counts", reg.value_counts().to_dict())

print("\n== train plain genuine missingness by hour ==")
tr = traffic.load_mainline(P, "train")["plain"]
print("nan frac", float(np.isnan(tr["speed"]).mean()), "elig frac", float(tr["elig"].mean()))
print("\n== window_history train row count ==")
h = traffic.load_window_history()
print(len(h), h.window_id.nunique(), h.split.value_counts().to_dict() if "split" in h.columns else "no split col")
