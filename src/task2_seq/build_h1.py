"""H1 truth: raw single-step observation labels (speed <= v_cut on eligible cells).

The current v3 cache labels/truth are already H2 (NaN-aware 3-step temporal median
then threshold, verified bit-exact), so this caches the *other* hypothesis for the
label-regime comparison: labels and truth from the raw observed speed.

cache/h1_<PANEL>.npz: on_y_h1, ong_y_h1, on_truth_h1, ong_truth_h1, rel_truth_h1
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH


def build_panel(panel: str) -> str:
    t0 = time.time()
    idx = LH.train_index_arrays(panel)
    d = LH.split_data(panel, "train")
    st = LH.static_panel(panel)
    L = st["n_links"]
    vc = st["v_cut"][None, :]
    elig = d["plain"]["elig"].astype(bool)
    sp = d["plain"]["speed"]

    def lab(ti):
        sl = slice(ti + 1, ti + 1 + LH.HORIZON)
        return ((sp[sl] <= vc) & elig[sl]).astype(np.int8)

    n_on, n_ong = len(idx["on_ti"]), len(idx["ong_ti"])
    on_truth = np.stack([lab(int(ti)) for ti in idx["on_ti"]])
    ong_truth = np.stack([lab(int(ti)) for ti in idx["ong_ti"]])
    rel_truth, rel_ids = [], []
    for split in ("train", "validation", "private"):
        w = LH.released_windows_panel(panel, split)
        tgrid = LH.split_data(panel, split)["times"]
        for r in w.itertuples(index=False):
            ti = LH.origin_index(tgrid, r.forecast_origin)
            rel_truth.append(lab(ti) if split == "train" else np.zeros((LH.HORIZON, L), np.int8))
            rel_ids.append(str(r.window_id))
        LH.free_split(panel, split)
        gc.collect()
    rel_truth = np.stack(rel_truth)
    np.savez_compressed(os.path.join(LH.CACHE, f"h1_{panel}.npz"),
                        on_y_h1=on_truth[:, LH.HORIZON - 1, :].reshape(-1),
                        ong_y_h1=ong_truth.reshape(-1),
                        on_truth_h1=on_truth, ong_truth_h1=ong_truth,
                        rel_truth_h1=rel_truth,
                        rel_ids=np.array(rel_ids, dtype="<U64"))
    print(f"{panel}: on {n_on} ong {n_ong} | H1 positives on(last) "
          f"{int(on_truth[:, -1].sum())} ong {int(ong_truth.sum())} | {time.time() - t0:.1f}s",
          flush=True)
    del on_truth, ong_truth, rel_truth, d
    LH.free_split(panel, "train")
    gc.collect()
    return panel


def main():
    for p in LH.PANELS:
        build_panel(p)


if __name__ == "__main__":
    main()
