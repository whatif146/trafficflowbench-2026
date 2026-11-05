"""Single-panel training trial: holdout S_state curve + base reference."""
import os, sys, time
import numpy as np

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task1\unet")
import unetlib as U
from t1lib import score, fmt

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I210_W"
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 60
tag = sys.argv[3] if len(sys.argv) > 3 else "trial"
t0 = time.time()
U.build_cache(panel, "train", holdout_days=60)
c = U.load_cache(panel, "train_hold60")
n_days = c["times"].size // U.N_TOD
days = np.arange(n_days)
fit = days[:n_days - 60 - 20]
val = days[n_days - 60 - 20:n_days - 60]
hold = days[n_days - 60:n_days]
rows = np.where(U.row_day(c, np.arange(len(c["tpl_t"]))) >= n_days - 60)[0]
tv, tq, lanes, reg = U.truth_rows(c, rows)

# base-only reference
ss = c["tpl_s"][rows]
bs = c["base_s"][c["tpl_t"][rows], ss].astype(np.float64)
bq = c["base_q"][c["tpl_t"][rows], ss].astype(np.float64)
print("[base]", fmt(score(tv, tq, bs, bq, lanes, reg)), flush=True)

model, info = U.train_panel(c, panel, fit, val, epochs=epochs, log_every=5)
v, q = U.predict_rows(model, c, rows, info["W"], info["device"])
print("[unet]", fmt(score(tv, tq, v, q, lanes, reg)), flush=True)
np.savez(os.path.join(U.HERE, f"pred_{panel}_{tag}.npz"),
         rows=rows, v=v, q=q, true_v=tv, true_q=tq, lanes=lanes, reg=reg,
         W=info["W"], hist=np.array([(h["epoch"], h["train_loss"], h["val_loss"]) for h in info["hist"]]))
import torch
torch.save({"state": model.state_dict(), "W": info["W"], "widths": info["widths"],
            "panel": panel, "tag": tag, "best_epoch": info["best_epoch"]},
           os.path.join(U.HERE, f"model_{panel}_{tag}.pt"))
print("total %.1fs" % (time.time() - t0), flush=True)
