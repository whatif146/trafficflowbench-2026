"""Build features for the 80+80+80 released windows (10 per panel per split).

For each released window:
  * recent 12 history steps come from the released window_history (observed layer);
  * history rows 0..23 (-180..-65 min) and row T come from the split's masked
    layer, NaN-aware (causal);
  * the 65 v3 features are rebuilt with the frozen v3 feature function
    (work/task2/gbm_predict.window_feature_block) so the model input is exactly
    the same function of the same data as during v3 fitting;
  * train-split windows also get their horizon truth from the train plain layer.

Writes work/task2_seq/cache/rel_<PANEL>.npz.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH

import gbm_predict as GP  # noqa: E402  (work/task2, read-only)

V3C = LH.v3_col_index()


def build_panel(panel: str) -> str:
    t0 = time.time()
    st = LH.static_panel(panel)
    L = st["n_links"]
    clim = GP.load_clim(panel)
    wh = LH.window_histories(panel)
    hframe = LH.window_history_frame()
    hframe = hframe[hframe.window_id.isin(wh.keys())]
    n = len(wh)
    win_ids, splits, conds, tis, times_out = [], [], [], [], []
    V3 = np.zeros((n * LH.HORIZON * L, len(V3C)), np.float32)
    LHf = np.zeros((n, L, LH.N_LH), np.float32)
    RRf = np.zeros((n, L, LH.N_RAMP), np.float32)
    truth = np.zeros((n, LH.HORIZON, L), np.int8)
    elig = np.zeros((n, LH.HORIZON, L), bool)
    prof = LH.ramp_profile(panel)
    r2i = st["rank_to_idx"]
    k = 0
    for split in ("train", "validation", "private"):
        d = LH.split_data(panel, split)
        ramp = LH.traffic.load_ramps(panel, split)
        w = LH.released_windows_panel(panel, split)
        tgrid = d["times"]
        for r in w.itertuples(index=False):
            wid = str(r.window_id)
            ti = LH.origin_index(tgrid, r.forecast_origin)
            tgt = [pd.Timestamp(int(tgrid[ti + 1 + j]), unit="s", tz="UTC")
                   for j in range(LH.HORIZON)]
            h = hframe[hframe.window_id == wid]
            X, _ = GP.window_feature_block(h, clim, tgt)
            V3[k * LH.HORIZON * L:(k + 1) * LH.HORIZON * L] = X[:, V3C]
            S, F = LH.make_history(panel, split, ti, win_hist=wh[wid])
            LHf[k] = LH.lh_features(panel, r.forecast_origin, S, F, ramp, ti)
            RRf[k] = LH.ramp_ratio_features(tgrid, ti, ramp, prof, r2i)
            if split == "train":
                tr, el = LH.window_labels(panel, ti)
                truth[k] = tr
                elig[k] = el
            win_ids.append(wid)
            splits.append(split)
            conds.append(str(r.condition))
            tis.append(ti)
            times_out.append(int(tgrid[ti]))
            k += 1
    out = {
        "panel": np.array(panel), "n_links": np.array(L, np.int32),
        "window_ids": np.array(win_ids, dtype="<U64"), "split": np.array(splits, dtype="<U16"),
        "condition": np.array(conds, dtype="<U16"), "ti": np.array(tis, np.int32),
        "time": np.array(times_out, np.int64),
        "v3": V3, "lh": LHf, "lhr": RRf, "truth": truth, "elig": elig,
    }
    path = os.path.join(LH.CACHE, f"rel_{panel}.npz")
    np.savez_compressed(path, **out)
    print(f"{panel}: {k} released windows | LH finite {np.isfinite(LHf).mean():.3f} "
          f"ramp ratios finite {np.isfinite(RRf).mean():.3f} "
          f"| {time.time() - t0:.1f}s -> {os.path.basename(path)}")
    del out, V3, LHf, RRf, d, ramp
    for split in ("train", "validation", "private"):
        LH.free_split(panel, split)
    gc.collect()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=None)
    args = ap.parse_args()
    for p in (args.panels or LH.PANELS):
        build_panel(p)


if __name__ == "__main__":
    main()
