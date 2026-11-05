"""Build the long-history feature cache for the selector-based pseudo windows.

One npz per panel in work/task2_seq/cache/tr_<PANEL>.npz.  The window list and
the v3 feature columns are read row-for-row from the frozen v3 dataset cache
(work/task2/cache/ds_<PANEL>.npz), which was verified to be reproduced exactly
by lhcore.train_index_arrays / the window truth arrays.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH

V3C = LH.v3_col_index()


def build_panel(panel: str, panel_id: int) -> str:
    t0 = time.time()
    z = np.load(os.path.join(LH.WORK, "task2", "cache", f"ds_{panel}.npz"), allow_pickle=False)
    L = int(z["n_links"])
    X_on_v3 = np.ascontiguousarray(z["X_on"][:, V3C])
    X_ong_v3 = np.ascontiguousarray(z["X_ong"][:, V3C])
    on_truth, on_elig = z["on_truth"], z["on_elig"].astype(bool)
    ong_truth, ong_elig = z["ong_truth"], z["ong_elig"].astype(bool)
    on_y, ong_y = z["y_on"], z["y_ong"]
    on_acc, ong_acc = z["on_accepted"].astype(bool), z["ong_accepted"].astype(bool)
    cut_day = int(z["cut_day"])
    z.close()

    idx = LH.train_index_arrays(panel)
    n_on, n_ong = len(idx["on_ti"]), len(idx["ong_ti"])
    assert on_y.shape[0] == n_on * L, (panel, on_y.shape, n_on * L)
    assert ong_y.shape[0] == n_ong * 6 * L
    # exact truth correspondence check (strong alignment test)
    tr0 = np.stack([LH.window_labels(panel, int(ti))[0] for ti in idx["on_ti"]])
    assert np.array_equal(tr0, on_truth), f"{panel}: onset truth misaligned"
    tr1 = np.stack([LH.window_labels(panel, int(ti))[0] for ti in idx["ong_ti"]])
    assert np.array_equal(tr1, ong_truth), f"{panel}: ongoing truth misaligned"
    del tr0, tr1
    gc.collect()

    d = LH.split_data(panel, "train")
    plain = d["plain"]
    dt = idx["dt"]
    ramp = LH.traffic.load_ramps(panel, "train")
    prof = LH.ramp_profile(panel)
    r2i = LH.static_panel(panel)["rank_to_idx"]
    ps, pf = plain["speed"], plain["flow"]

    lh_on = np.zeros((n_on * L, LH.N_LH), np.float32)
    rr_on = np.zeros((n_on * L, LH.N_RAMP), np.float32)
    for k, ti in enumerate(idx["on_ti"]):
        ti = int(ti)
        S, F = LH.make_history(panel, "train", ti,
                               plain_hist=(ps[ti - LH.HIST:ti], pf[ti - LH.HIST:ti]))
        lh_on[k * L:(k + 1) * L] = LH.lh_features(panel, dt[ti], S, F, ramp, ti)
        rr_on[k * L:(k + 1) * L] = LH.ramp_ratio_features(idx["times"], ti, ramp, prof, r2i)
    lh_ong = np.zeros((n_ong, L, LH.N_LH), np.float32)
    rr_ong = np.zeros((n_ong, L, LH.N_RAMP), np.float32)
    for k, ti in enumerate(idx["ong_ti"]):
        ti = int(ti)
        S, F = LH.make_history(panel, "train", ti,
                               plain_hist=(ps[ti - LH.HIST:ti], pf[ti - LH.HIST:ti]))
        lh_ong[k] = LH.lh_features(panel, dt[ti], S, F, ramp, ti)
        rr_ong[k] = LH.ramp_ratio_features(idx["times"], ti, ramp, prof, r2i)

    out = {
        "panel": np.array(panel), "panel_id": np.array(panel_id, np.int32), "n_links": np.array(L, np.int32),
        "cut_day": np.array(cut_day, np.int32),
        "on_ti": idx["on_ti"], "on_acc": on_acc, "on_time": idx["times"][idx["on_ti"]],
        "on_v3": X_on_v3, "on_lh": lh_on, "on_lhr": rr_on, "on_y": on_y.astype(np.int8),
        "on_truth": on_truth, "on_elig": on_elig,
        "ong_ti": idx["ong_ti"], "ong_acc": ong_acc, "ong_time": idx["times"][idx["ong_ti"]],
        "ong_v3": X_ong_v3, "ong_lh": lh_ong, "ong_lhr": rr_ong, "ong_y": ong_y.astype(np.int8),
        "ong_truth": ong_truth, "ong_elig": ong_elig,
    }
    path = os.path.join(LH.CACHE, f"tr_{panel}.npz")
    np.savez_compressed(path, **out)
    print(f"{panel}: on {n_on} ong {n_ong} L={L} LH props finite "
          f"{np.isfinite(lh_on).mean():.3f} ramp ratios finite {np.isfinite(rr_on).mean():.3f} "
          f"| {time.time() - t0:.1f}s -> {os.path.basename(path)}")
    del out, lh_on, lh_ong, rr_on, rr_ong, X_on_v3, X_ong_v3, d, ramp
    LH.free_split(panel, "train")
    gc.collect()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=None)
    args = ap.parse_args()
    panels = args.panels or LH.PANELS
    for i, p in enumerate(LH.PANELS):
        if p in panels:
            build_panel(p, i)


if __name__ == "__main__":
    main()
