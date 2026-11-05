import sys, time
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task1\unet")
import unetlib as U
for p in ["D7_I405_S", "D7_I10_E", "D12_I5_N"]:
    t=time.time(); U.build_cache(p, "train", holdout_days=60); print(p, "%.1fs"%(time.time()-t), flush=True)
