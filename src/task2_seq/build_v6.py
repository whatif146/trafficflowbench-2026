"""Build the v6 feature caches (one panel at a time, memory-lean).

cache/v6_<PANEL>.npz:
  on_v6   (n_on*L, N_V6)   pseudo-window onset features (row-major window->link)
  ong_v6  (n_ong, L, N_V6) pseudo-window ongoing features (broadcast over steps)
  rel_v6  (n_rel, L, N_V6) released windows, in build_released.py order
  rel_ids (n_rel,) window ids for verification
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import v6core as V6


def build_panel(panel: str) -> str:
    t0 = time.time()
    st = LH.static_panel(panel)
    L = st["n_links"]
    prof = V6.profile_24h(panel)
    idx = LH.train_index_arrays(panel)
    n_on, n_ong = len(idx["on_ti"]), len(idx["ong_ti"])
    ser = V6.panel_series(panel, "train")
    on_v6 = np.zeros((n_on * L, V6.N_V6), np.float32)
    for k, ti in enumerate(idx["on_ti"]):
        on_v6[k * L:(k + 1) * L] = V6.v6_features(ser, int(ti), prof)
    ong_v6 = np.zeros((n_ong, L, V6.N_V6), np.float32)
    for k, ti in enumerate(idx["ong_ti"]):
        ong_v6[k] = V6.v6_features(ser, int(ti), prof)
    del ser
    LH.free_split(panel, "train")
    gc.collect()

    rel_ids, rel_list = [], []
    for split in ("train", "validation", "private"):
        ser = V6.panel_series(panel, split)
        w = LH.released_windows_panel(panel, split)
        tgrid = ser["times"]
        for r in w.itertuples(index=False):
            ti = LH.origin_index(tgrid, r.forecast_origin)
            rel_list.append(V6.v6_features(ser, ti, prof))
            rel_ids.append(str(r.window_id))
        del ser
        LH.free_split(panel, split)
        gc.collect()
    rel_v6 = np.stack(rel_list) if rel_list else np.zeros((0, L, V6.N_V6), np.float32)

    path = os.path.join(LH.CACHE, f"v6_{panel}.npz")
    np.savez_compressed(path, on_v6=on_v6, ong_v6=ong_v6, rel_v6=rel_v6,
                        rel_ids=np.array(rel_ids, dtype="<U64"))
    print(f"{panel}: on {n_on} ong {n_ong} rel {len(rel_ids)} L={L} "
          f"finite {np.isfinite(on_v6).mean():.3f} | {time.time() - t0:.1f}s", flush=True)
    del on_v6, ong_v6, rel_v6, prof
    gc.collect()
    return path


def main():
    for p in LH.PANELS:
        build_panel(p)


if __name__ == "__main__":
    main()
