"""Seed ensemble: train extra seeds (43, 44) with the identical recipe."""
import os, sys, gc, time
import numpy as np

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "unet"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
import torch
import unetlib as U
from t1lib import score, fmt

PANELS = sys.argv[1].split(",") if len(sys.argv) > 1 else \
    ["D7_I210_W", "D7_I405_S", "D7_I10_E", "D12_I5_N"]
SEEDS = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["43", "44"])]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 80
TAG = "u80"
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0), flush=True)

for panel in PANELS:
    t0 = time.time()
    c = U.load_cache(panel, "train_hold60")
    n_days = c["times"].size // U.N_TOD
    days = np.arange(n_days)
    fit = days[:n_days - 60 - 20]
    val = days[n_days - 60 - 20:n_days - 60]
    hold_rows = np.where(U.row_day(c, np.arange(len(c["tpl_t"]))) >= n_days - 60)[0]
    for seed in SEEDS:
        model, info = U.train_panel(c, panel, fit, val, epochs=EPOCHS, seed=seed,
                                    device=dev, log_every=20)
        v, q = U.predict_rows(model, c, hold_rows, info["W"], dev)
        np.savez(os.path.join(U.HERE, f"pred_unet_{panel}_{TAG}s{seed}.npz"),
                 rows=hold_rows, v=v, q=q, W=info["W"], seed=seed)
        torch.save({"state": model.state_dict(), "W": info["W"], "widths": info["widths"],
                    "panel": panel, "tag": f"{TAG}s{seed}", "epochs": EPOCHS},
                   os.path.join(U.HERE, f"model_{panel}_{TAG}s{seed}.pt"))
        tv, tq, lanes, reg = U.truth_rows(c, hold_rows)
        print(f"  [{panel} seed{seed}] {fmt(score(tv, tq, v, q, lanes, reg))} "
              f"{time.time() - t0:.0f}s", flush=True)
        del model
        gc.collect()
    del c
    gc.collect()
print("done", flush=True)
