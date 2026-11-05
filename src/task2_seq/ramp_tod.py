"""Per-link on/off-ramp time-of-day profiles from the train split (legal: past data).

cache/ramp_tod_<PANEL>.npz holds (288, L) mean ramp flow per 5-min time-of-day
slot, used to turn the observed ramp flow into a jam-sensor ratio
(observed / its own TOD expectation) in the long-history features.
"""
import gc
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH


def profile(arr: np.ndarray, valid: np.ndarray, slots: np.ndarray, L: int) -> np.ndarray:
    T = arr.shape[0]
    a = np.asarray(arr, np.float64).ravel()
    v = (valid & np.isfinite(arr)).ravel()
    flat_idx = np.repeat(slots, L) * L + np.tile(np.arange(L), T)
    s = np.zeros(288 * L, np.float64)
    c = np.zeros(288 * L, np.float64)
    np.add.at(s, flat_idx[v], a[v])
    np.add.at(c, flat_idx[v], 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = s.reshape(288, L) / c.reshape(288, L)
    link_mean = np.nanmean(np.where(valid & np.isfinite(arr), arr, np.nan), axis=0)
    bad = ~np.isfinite(prof)
    prof = np.where(bad, link_mean[None, :], prof)
    prof = np.where(np.isfinite(prof) & (prof > 1e-6), prof, np.nan)
    return prof.astype(np.float32)


def main():
    for p in LH.PANELS:
        st = LH.static_panel(p)
        L = st["n_links"]
        times_p = os.path.join(LH.WORK, "cache", p, "train_masked.npz")
        t = np.load(times_p, allow_pickle=False)["times"]
        dt = np.asarray(t, dtype="datetime64[s]").astype("datetime64[s]")
        import pandas as pd
        dti = pd.to_datetime(t, unit="s", utc=True)
        slots = ((dti.hour * 60 + dti.minute) // 5).to_numpy(np.int64)
        r = LH.traffic.load_ramps(p, "train")
        po = profile(r["on_flow"], r["on_valid"], slots, L)
        pf = profile(r["off_flow"], r["off_valid"], slots, L)
        np.savez_compressed(os.path.join(LH.CACHE, f"ramp_tod_{p}.npz"), on=po, off=pf)
        print(f"{p}: profile on nan {np.isnan(po).mean():.3f} -> ramp_tod_{p}.npz", flush=True)
        del r, po, pf
        gc.collect()


if __name__ == "__main__":
    main()
