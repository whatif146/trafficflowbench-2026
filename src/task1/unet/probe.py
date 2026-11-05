import sys, time
import numpy as np

ROOT = r"C:\Users\lenovo\Desktop\kaggle"
sys.path.insert(0, ROOT + r"\work\lib")
sys.path.insert(0, ROOT + r"\work\task1\gbm")
import traffic
from t1lib import day_index, link_perm, temporal_interp

panel = "D7_I10_E"
t0 = time.time()
plain = traffic._load_or_build(panel, "train", "plain")
masked = traffic._load_or_build(panel, "train", "masked")
tg = traffic.load_targets(panel, "train")
T, S = masked["speed"].shape
print("T,S", T, S, "load %.1fs" % (time.time() - t0))
elig = masked["elig"]
print("elig", elig.sum(), "template rows", len(tg["t_idx"]))
f = np.isfinite(masked["speed"])
ts = np.isfinite(plain["speed"])
ti, si = tg["t_idx"].astype(np.int64), tg["s_idx"].astype(np.int64)
print("template all eligible:", elig[ti, si].mean())
print("truth finite at template:", (ts[ti, si] & np.isfinite(plain["flow"][ti, si])).mean())
print("observed in masked at template:", f[ti, si].mean())
order_p, inv = link_perm(masked)
t1 = time.time()
s_intr = temporal_interp(masked["speed"])
print("temporal_interp %.1fs" % (time.time() - t1))
rem = ts[ti, si] & ~f[ti, si]
bs = s_intr[ti[rem], si[rem]]
print("removed target cells", int(rem.sum()), "base NaN frac %.4f" % np.isnan(bs).mean())
obs = f[ti, si]
d = np.abs(s_intr[ti[obs], si[obs]] - masked["speed"][ti[obs], si[obs]])
print("interp exact at observed cells: max diff %.4f" % np.nanmax(d))
