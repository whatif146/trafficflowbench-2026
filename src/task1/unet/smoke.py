"""End-to-end smoke test: build a cache, train 2 epochs, predict + score."""
import sys, time
import numpy as np

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task1\unet")
import unetlib as U
from t1lib import score, fmt

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I210_W"
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 2
t0 = time.time()
U.build_cache(panel, "train", holdout_days=60)
c = U.load_cache(panel, "train_hold60")
print("cache loaded %.1fs" % (time.time() - t0), {k: v.dtype.name for k, v in list(c.items())[:5]})
n_days = c["times"].size // U.N_TOD
days = np.arange(n_days)
fit = days[:n_days - 60 - 20][0:n_days - 60 - 20]
val = np.arange(n_days - 60 - 20, n_days - 60)
hold = np.arange(n_days - 60, n_days)
print("fit days", fit[0], fit[-1], "val", val[0], val[-1], "hold", hold[0], hold[-1])
model, info = U.train_panel(c, panel, fit, val, epochs=epochs, log_every=1)
print(info["hist"])
rows = np.where(U.row_day(c, np.arange(len(c["tpl_t"]))) >= n_days - 60)[0]
t1 = time.time()
v, q = U.predict_rows(model, c, rows, info["W"], info["device"])
print("predict %d rows %.1fs" % (len(rows), time.time() - t1))
tv, tq, lanes, reg = U.truth_rows(c, rows)
print("UNet", fmt(score(tv, tq, v, q, lanes, reg)))
