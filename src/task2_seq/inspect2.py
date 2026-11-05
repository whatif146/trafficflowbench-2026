"""Inspect the v3 dataset cache + static clim artefacts + CV detail."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task2")
import traffic
import t2core as C

print("== ds cache ==")
for p in ["D7_I10_E", "D7_I210_E"]:
    z = np.load(os.path.join(C.CACHE2, f"ds_{p}.npz"), allow_pickle=False)
    print(p, {k: (z[k].shape, str(z[k].dtype)) for k in z.files if z[k].ndim <= 2})
    z.close()

print("\n== models dir ==")
for d in sorted(os.listdir(C.MODEL2)):
    print(os.path.join(d, ) if False else d, end="  ")
print()
print("\n== meta v3 ==")
m = json.load(open(os.path.join(C.MODEL2, "v3", "gbm_meta.json")))
for k, v in m.items():
    if k == "features":
        print("features", len(v))
    else:
        print(k, ":", v)
print("\n== gbm_cv.csv ==")
cv = pd.read_csv(os.path.join(C.OUT2, "gbm_cv.csv"))
print(cv.columns.tolist())
print(cv.head(8).to_string())
print("n rows", len(cv), "conditions", cv.groupby("condition").size().to_dict() if "condition" in cv else None)

print("\n== static npz keys ==")
z = np.load(os.path.join(C.MODEL2, "static_D7_I10_E.npz"), allow_pickle=False)
print({k: (z[k].shape, str(z[k].dtype)) for k in z.files})
z.close()

print("\n== reproduce candidate counts ==")
for p in ["D7_I10_E", "D12_I5_N"]:
    d = C.load_train_panel(p)
    cand = C.candidate_table(d)
    acc = C.greedy_select(cand)
    print(p, "origins", len(cand), "onset cand ok", int(((~cand.ongoing) & cand.ok).sum()),
          "ongoing cand ok", int((cand.ongoing & cand.ok).sum()),
          "accepted", acc.groupby("condition").size().to_dict())
    del d, cand, acc
