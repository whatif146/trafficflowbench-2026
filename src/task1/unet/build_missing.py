import sys, time, os
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task1\unet")
import unetlib as U
missing = []
for p in ["D7_I10_E","D7_I10_W","D7_I210_E","D7_I210_W","D7_I405_N","D7_I405_S",
          "D12_I5_N","D12_I5_S","D12_I405_N","D12_I405_S"]:
    if not os.path.exists(U.cache_path(p, "train_hold60")):
        missing.append(p)
print("missing:", missing, flush=True)
for p in missing:
    t = time.time(); U.build_cache(p, "train", holdout_days=60)
    print(p, "%.1fs" % (time.time()-t), flush=True)
print("DONE", flush=True)
