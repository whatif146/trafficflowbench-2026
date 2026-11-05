"""v7: denoised (latent-state) queue labels and truth.

The scoring spec says the official queue truth comes from the *underlying state*,
not the noisy observation, so thresholding observed speed at v_cut injects
measurement flips.  Here the latent speed field is estimated from the unmasked
train observation layer with a NaN-aware temporal Gaussian smoother
(sigma = 2.5 steps = 12.5 min, reflect boundaries) and thresholded at
v_cut = 0.60*free_speed, masked to is_score_eligible:

    latent(t) = sum_t' K(t-t') m(t') s(t') / sum_t' K(t-t') m(t')

(the mask-normalised Gaussian is NaN-aware and does not shrink at gaps).

Saved per panel (cache/v7_<PANEL>.npz), row-aligned with the v3 dataset cache:
  on_y_dn      (n_on*L,)        onset last-step labels
  on_truth_dn  (n_on,6,L)       onset horizon truth
  ong_y_dn     (n_ong*6L,)      ongoing labels
  ong_truth_dn (n_ong,6,L)      ongoing horizon truth
  rel_truth_dn (n_rel,6,L)      released-window truth (train split only)
Plus flip statistics (noisy-vs-denoised) in out/v7_flips.json.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH

from scipy.ndimage import gaussian_filter1d

SIGMA_T = 2.5
FLIP_STATS = []


def denoise_speed(panel: str) -> np.ndarray:
    d = LH.split_data(panel, "train")
    sp = d["plain"]["speed"]
    obs = np.isfinite(sp)
    fill = np.where(obs, sp, 0.0).astype(np.float32)
    num = gaussian_filter1d(fill, SIGMA_T, axis=0, mode="reflect")
    den = gaussian_filter1d(obs.astype(np.float32), SIGMA_T, axis=0, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)
    return lat.astype(np.float32)


def build_panel(panel: str) -> str:
    t0 = time.time()
    idx = LH.train_index_arrays(panel)
    d = LH.split_data(panel, "train")
    st = LH.static_panel(panel)
    L = st["n_links"]
    vc = st["v_cut"][None, :]
    elig = d["plain"]["elig"].astype(bool)
    sp = d["plain"]["speed"]
    lat = denoise_speed(panel)

    n_on, n_ong = len(idx["on_ti"]), len(idx["ong_ti"])
    on_truth = np.empty((n_on, LH.HORIZON, L), np.int8)
    ong_truth = np.empty((n_ong, LH.HORIZON, L), np.int8)
    on_y = np.empty((n_on, L), np.int8)
    ong_y = np.empty((n_ong * LH.HORIZON, L), np.int8)
    for k, ti in enumerate(idx["on_ti"]):
        ti = int(ti)
        tr = ((lat[ti + 1:ti + 1 + LH.HORIZON] <= vc) & elig[ti + 1:ti + 1 + LH.HORIZON]).astype(np.int8)
        on_truth[k] = tr
        on_y[k] = tr[LH.HORIZON - 1]
    for k, ti in enumerate(idx["ong_ti"]):
        ti = int(ti)
        tr = ((lat[ti + 1:ti + 1 + LH.HORIZON] <= vc) & elig[ti + 1:ti + 1 + LH.HORIZON]).astype(np.int8)
        ong_truth[k] = tr
        ong_y[k * LH.HORIZON:(k + 1) * LH.HORIZON] = tr
    # released windows (only the train split has truth)
    rel_ids, rel_truth = [], []
    for split in ("train", "validation", "private"):
        w = LH.released_windows_panel(panel, split)
        tgrid = LH.split_data(panel, split)["times"]
        for r in w.itertuples(index=False):
            ti = LH.origin_index(tgrid, r.forecast_origin)
            if split == "train":
                tr = ((lat[ti + 1:ti + 1 + LH.HORIZON] <= vc) & elig[ti + 1:ti + 1 + LH.HORIZON]).astype(np.int8)
            else:
                tr = np.zeros((LH.HORIZON, L), np.int8)
            rel_truth.append(tr)
            rel_ids.append(str(r.window_id))
        LH.free_split(panel, split)
        gc.collect()
    rel_truth = np.stack(rel_truth)

    # ---- flip statistics vs the noisy (3-step median) truth -----------------
    z = np.load(os.path.join(LH.CACHE, f"tr_{panel}.npz"), allow_pickle=False)
    noisy_on = z["on_truth"]
    noisy_ong = z["ong_truth"]
    z.close()
    z = np.load(os.path.join(LH.CACHE, f"rel_{panel}.npz"), allow_pickle=False)
    rel_elig = z["elig"]
    noisy_rel = z["truth"]
    z.close()
    obs_near = np.isfinite(sp) & (np.abs(sp - vc) <= 5.0)
    stats = {"panel": panel}
    for name, dn, ny in (("onset", on_truth, noisy_on),
                         ("ongoing", ong_truth, noisy_ong),
                         ("released", rel_truth, noisy_rel)):
        if name == "released":
            ti_list = [LH.origin_index(LH.split_data(panel, "train")["times"], r.forecast_origin)
                       for r in LH.released_windows_panel(panel, "train").itertuples(index=False)]
            near = np.stack([obs_near[ti + 1:ti + 1 + LH.HORIZON] for ti in ti_list])
            flip = (dn[:10] != ny[:10])
            stats[name] = {
                "cells": int(dn.size), "flips": int(flip.sum()),
                "flip_frac": float(flip.sum() / dn.size),
                "flip_frac_near_cut": float(flip[near].mean()) if near.any() else None,
                "truth_pos_noisy": int(ny.sum()), "truth_pos_dn": int(dn.sum()),
            }
        else:
            ti_all = idx["on_ti"] if name == "onset" else idx["ong_ti"]
            near = np.stack([obs_near[int(ti) + 1:int(ti) + 1 + LH.HORIZON] for ti in ti_all])
            flip = (dn != ny)
            stats[name] = {
                "cells": int(dn.size), "flips": int(flip.sum()),
                "flip_frac": float(flip.mean()),
                "flip_frac_near_cut": float(flip[near].mean()) if near.any() else None,
                "truth_pos_noisy": int(ny.sum()), "truth_pos_dn": int(dn.sum()),
            }
    FLIP_STATS.append(stats)

    out = {"on_y_dn": on_y.reshape(-1), "on_truth_dn": on_truth,
           "ong_y_dn": ong_y.reshape(-1), "ong_truth_dn": ong_truth,
           "rel_truth_dn": rel_truth,
           "rel_ids": np.array(rel_ids, dtype="<U64"), "sigma_t": np.array(SIGMA_T)}
    path = os.path.join(LH.CACHE, f"v7_{panel}.npz")
    np.savez_compressed(path, **out)
    print(f"{panel}: on {n_on} ong {n_ong} L={L} | flips onset {stats['onset']['flip_frac']:.4f} "
          f"ongoing {stats['ongoing']['flip_frac']:.4f} released {stats['released']['flip_frac']:.4f} "
          f"| {time.time() - t0:.1f}s", flush=True)
    del on_truth, ong_truth, on_y, ong_y, rel_truth, lat, d
    LH.free_split(panel, "train")
    gc.collect()
    return path


def main():
    for p in LH.PANELS:
        build_panel(p)
    with open(os.path.join(LH.OUT, "v7_flips.json"), "w") as fh:
        json.dump(FLIP_STATS, fh, indent=1)


if __name__ == "__main__":
    main()
