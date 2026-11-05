"""Unit test of smooth_field on synthetic data with known smooth field."""
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task1\gbm")
import numpy as np
from explore3 import smooth_field

rng = np.random.default_rng(0)
T, S = 2000, 30
t = np.arange(T)[:, None]
s = np.arange(S)[None, :]
f = 60 + 20 * np.sin(2 * np.pi * t / 288.0) + 5 * np.cos(2 * np.pi * s / 40.0) + 0.02 * s
f = f + 10 * np.sin(2 * np.pi * (t + 3 * s) / 288.0)
noise = rng.normal(0, 1.5, size=(T, S))
y = f + noise
mask = rng.random((T, S)) < 0.7  # 30% observed
ym = np.where(mask, y, np.nan)

inv = np.arange(S)
for order in (0, 1):
    for st in (0.7, 1.0, 1.5, 3.0):
        for ssz in (0.0, 1.0):
            est = smooth_field(ym, st, ssz, order=order, inv=inv)
            e = est - f
            print("order=%d st=%.1f ss=%.0f  rmse_vs_true_f=%.3f  rmse_vs_noisy_y=%.3f  nan=%.4f" % (
                order, st, ssz, np.sqrt(np.nanmean(e * e)),
                np.sqrt(np.nanmean((est - y) ** 2)), np.isnan(est).mean()))
