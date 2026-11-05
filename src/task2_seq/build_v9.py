"""v9 prerequisites: speed observation-noise estimate, soft Bayesian labels and a
mild-denoised hard proxy truth.

1. sigma_v from the second difference of the plain speed at consecutive 5-min
   steps: if v_t = latent_t + eps_t with iid eps (var sigma^2), then
   d_t = v_{t+1} - 2 v_t + v_{t-1} has var 6*sigma^2 (linear trends cancel).
   Robust pooled + per-panel estimates via MAD.
2. Soft label y = P(latent <= v_cut | v_obs) = Phi((v_cut - v_obs)/sigma_v),
   eligibility-masked (0 for ineligible, same convention as v3).
3. Mild-denoised hard hard truth: NaN-aware temporal Gaussian, sigma=1 step.
   (proxy (ii); proxy (i) is the existing 3-step-median noisy truth.)

cache/v9_<PANEL>.npz:
  on_y_soft (n_on*L,), ong_y_soft (n_ong*6L,)       soft training labels
  on_truth_s1 (n_on,6,L), ong_truth_s1 (n_ong,6,L)  mild hard truth (pseudo)
  rel_truth_s1 (n_rel,6,L)                          mild hard truth (released)
  sigma_v scalar
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
from scipy.stats import norm

SIGMA_MILD = 1.0
SIGMA_STATS = []


def estimate_sigma_v(sp: np.ndarray) -> dict:
    """Robust observation-noise sigma from second differences of observed speed."""
    d = sp[2:] - 2.0 * sp[1:-1] + sp[:-2]
    v = d[np.isfinite(d)]
    if len(v) < 1000:
        return {"sigma": None, "n": int(len(v))}
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma_d = 1.4826 * mad
    # trimmed std as a cross-check
    lo, hi = np.percentile(v, [0.5, 99.5])
    w = v[(v >= lo) & (v <= hi)]
    return {"sigma": float(sigma_d / np.sqrt(6.0)),
            "sigma_trimstd": float(np.std(w) / np.sqrt(6.0)),
            "n": int(len(v)), "mad_sigma_d": sigma_d}


def build_panel(panel: str) -> str:
    t0 = time.time()
    idx = LH.train_index_arrays(panel)
    d = LH.split_data(panel, "train")
    st = LH.static_panel(panel)
    L = st["n_links"]
    vc = st["v_cut"][None, :]
    elig = d["plain"]["elig"].astype(bool)
    sp = d["plain"]["speed"]
    sig = estimate_sigma_v(sp)
    sigma = sig["sigma"] or 3.0
    sig["panel"] = panel
    SIGMA_STATS.append(sig)

    # mild hard truth: sigma=1 Gaussian, NaN-aware
    obs = np.isfinite(sp)
    num = gaussian_filter1d(np.where(obs, sp, 0.0).astype(np.float32), SIGMA_MILD, axis=0, mode="reflect")
    den = gaussian_filter1d(obs.astype(np.float32), SIGMA_MILD, axis=0, mode="reflect")
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)

    n_on, n_ong = len(idx["on_ti"]), len(idx["ong_ti"])
    on_truth = np.empty((n_on, LH.HORIZON, L), np.int8)
    ong_truth = np.empty((n_ong, LH.HORIZON, L), np.int8)
    on_y_soft = np.empty((n_on, LH.HORIZON, L), np.float32)
    ong_y_soft = np.empty((n_ong, LH.HORIZON, L), np.float32)
    for k, ti in enumerate(idx["on_ti"]):
        ti = int(ti)
        sl = slice(ti + 1, ti + 1 + LH.HORIZON)
        on_truth[k] = ((lat[sl] <= vc) & elig[sl]).astype(np.int8)
        on_y_soft[k] = (np.where(np.isfinite(sp[sl]), norm.cdf((vc - sp[sl]) / sigma), 0.0) * elig[sl]).astype(np.float32)
    for k, ti in enumerate(idx["ong_ti"]):
        ti = int(ti)
        sl = slice(ti + 1, ti + 1 + LH.HORIZON)
        ong_truth[k] = ((lat[sl] <= vc) & elig[sl]).astype(np.int8)
        ong_y_soft[k] = (np.where(np.isfinite(sp[sl]), norm.cdf((vc - sp[sl]) / sigma), 0.0) * elig[sl]).astype(np.float32)

    rel_truth, rel_ids = [], []
    for split in ("train", "validation", "private"):
        w = LH.released_windows_panel(panel, split)
        tgrid = LH.split_data(panel, split)["times"]
        for r in w.itertuples(index=False):
            ti = LH.origin_index(tgrid, r.forecast_origin)
            if split == "train":
                sl = slice(ti + 1, ti + 1 + LH.HORIZON)
                rel_truth.append(((lat[sl] <= vc) & elig[sl]).astype(np.int8))
            else:
                rel_truth.append(np.zeros((LH.HORIZON, L), np.int8))
            rel_ids.append(str(r.window_id))
        LH.free_split(panel, split)
        gc.collect()
    rel_truth = np.stack(rel_truth)

    # onset soft labels are needed for the last horizon step only (v3 convention)
    on_y_soft_last = on_y_soft[:, LH.HORIZON - 1, :]
    # soft-label summary
    flips = (on_truth != np.load(os.path.join(LH.CACHE, f"tr_{panel}.npz"), allow_pickle=False)["on_truth"])
    out = {"on_y_soft": on_y_soft_last.reshape(-1), "ong_y_soft": ong_y_soft.reshape(-1),
           "on_truth_s1": on_truth, "ong_truth_s1": ong_truth,
           "rel_truth_s1": rel_truth, "rel_ids": np.array(rel_ids, dtype="<U64"),
           "sigma_v": np.array(sigma)}
    np.savez_compressed(os.path.join(LH.CACHE, f"v9_{panel}.npz"), **out)
    print(f"{panel}: sigma_v {sigma:.2f} km/h | mild-truth flips vs noisy "
          f"{flips.mean():.4f} | soft mean {float(np.nanmean(on_y_soft)):.4f} "
          f"nan {int(np.isnan(on_y_soft).sum())} | {time.time() - t0:.1f}s",
          flush=True)
    del out, on_truth, ong_truth, on_y_soft, on_y_soft_last, ong_y_soft, rel_truth, lat, d
    LH.free_split(panel, "train")
    gc.collect()
    return panel


def main():
    for p in LH.PANELS:
        build_panel(p)
    with open(os.path.join(LH.OUT, "v9_sigma.json"), "w") as fh:
        json.dump(SIGMA_STATS, fh, indent=1)


if __name__ == "__main__":
    main()
