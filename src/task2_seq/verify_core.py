"""Verify that lhcore reproduces the v3 window list / truth arrays exactly."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH

PANEL = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
CUT = "2031-02-01"

z = np.load(os.path.join(LH.WORK, "task2", "cache", f"ds_{PANEL}.npz"), allow_pickle=False)
n_on = len(z["date_on"])
n_ong = len(z["date_ong"])
L = int(z["n_links"])
print(f"cache: n_on={n_on} n_ong={n_ong} L={L} X_on={z['X_on'].shape} X_ong={z['X_ong'].shape}")

idx = LH.train_index_arrays(PANEL)
print(f"mine : n_on={len(idx['on_ti'])} n_ong={len(idx['ong_ti'])}")
assert len(idx["on_ti"]) == n_on, "onset window count mismatch"
assert len(idx["ong_ti"]) == n_ong, "ongoing window count mismatch"
assert len(idx["on_ti"]) * L == z["X_on"].shape[0]
assert len(idx["ong_ti"]) * 6 * L == z["X_ong"].shape[0]

# truth + elig + accepted flags + day index must match row-for-row
on_truth = np.stack([LH.window_labels(PANEL, int(ti))[0] for ti in idx["on_ti"]])
on_elig = np.stack([LH.window_labels(PANEL, int(ti))[1] for ti in idx["on_ti"]])
ong_truth = np.stack([LH.window_labels(PANEL, int(ti))[0] for ti in idx["ong_ti"]])
ong_elig = np.stack([LH.window_labels(PANEL, int(ti))[1] for ti in idx["ong_ti"]])
print("on truth equal:", np.array_equal(on_truth, z["on_truth"]))
print("on elig  equal:", np.array_equal(on_elig, z["on_elig"]))
print("ong truth equal:", np.array_equal(ong_truth, z["ong_truth"]))
print("ong elig  equal:", np.array_equal(ong_elig, z["ong_elig"]))
print("on accepted equal:", np.array_equal(idx["on_acc"], z["on_accepted"]))
print("ong accepted equal:", np.array_equal(idx["ong_acc"], z["ong_accepted"]))

# day index check
days = pd.DatetimeIndex(idx["dt"]).normalize()
uniq = np.sort(np.asarray(days.unique(), dtype="datetime64[ns]"))
pos = np.searchsorted(uniq, np.asarray(days, dtype="datetime64[ns]"))
print("on date equal:", np.array_equal(pos[idx["on_ti"]], z["date_on"]))
print("ong date equal:", np.array_equal(pos[idx["ong_ti"]], z["date_ong"]))

# recompute the v3 feature block for the first two windows and compare to cache
import features as FT
from t2core import load_train_panel

d = load_train_panel(PANEL)
ramp = LH.traffic.load_ramps(PANEL, "train")
clim = {k: z[k] for k in ("onset_freq", "ong_freq", "rate", "elig_rate",
                          "ratio_tod", "fcap_tod")}
cut_day = int(z["cut_day"])
from build_dataset import onset_freq_all, ong_freq_all, rate_all, tod_profiles  # noqa: E402
# recompute clim exactly as build_dataset did (cheap for one panel)
cand = LH.C.candidate_table(d)
cand["day_idx"] = pos[cand.ti.to_numpy()]
acc = LH.C.greedy_select(cand)
onset = acc[acc.condition == "queue_onset"]
ongoing = acc[acc.condition == "queue_ongoing"]
ratio_tod, fcap_tod = tod_profiles(d)
clim = FT.clim_from(d, onset_freq_all(d, onset), ong_freq_all(d, ongoing), rate_all(d),
                    d["elig"].mean(axis=0).astype(np.float32), ratio_tod, fcap_tod)
for k, ti in enumerate(idx["on_ti"][:2]):
    c = FT.set_window_clim(clim, d, int(ti))
    X = FT.window_features(d, int(ti), c, 0, 0,
                           {"on": ramp["on_flow"][ti - 12:ti], "off": ramp["off_flow"][ti - 12:ti],
                            "on_valid": ramp["on_valid"][ti - 12:ti], "off_valid": ramp["off_valid"][ti - 12:ti]})
    X = X[(LH.HORIZON - 1) * L:]
    cached = z["X_on"][k * L:(k + 1) * L]
    print(f"window {k} ti={ti}: v3 feature max abs diff = {np.nanmax(np.abs(X - cached)):.3g}")

# long-history features for the same windows
wh = LH.window_histories(PANEL)
plain = LH.split_data(PANEL, "train")["plain"]
for k, ti in enumerate(idx["on_ti"][:2]):
    ti = int(ti)
    S, F = LH.make_history(PANEL, "train", ti,
                           plain_hist=(plain["speed"][ti - 12:ti], plain["flow"][ti - 12:ti]))
    lh = LH.lh_features(PANEL, pd.Timestamp(idx["dt"][ti]), S, F, ramp, ti)
    print(f"window {k}: LH shape {lh.shape} finite {np.isfinite(lh).mean():.3f}",
          "mean r_T", np.nanmean(lh[:, LH.LH_INDEX['r_T']]),
          "qfrac_l", np.nanmean(lh[:, LH.LH_INDEX['qfrac_l']]),
          "nan frac per col max", float(np.isnan(lh).mean(axis=0).max()))
z.close()
