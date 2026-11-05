"""Reference onset probabilities for the identical-rows comparison.

Trains the same v3 (65 feats) and v5 (145 feats) onset LightGBM models that the
shipped blend uses, under the team's honest protocol: fit on selector-valid
pseudo onset windows with origin before 2031-01-01 and outside June 2030 (so the
released-June protocol A stays honest).  Then stores the per-window probabilities
for (a) the January tuning block, (b) the February evaluation block, and (c) the
40 released train onset windows, so the CNN can be compared / blended row for row.

Read-only reuse of work/task2_seq/run_cv.py and lhcore.py.

Output: work/task2_cnn/cache/ref_probs.npz
  tr_<panel>_idx, tr_<panel>_time, tr_<panel>_prob3, tr_<panel>_prob5  (Jan+Feb blocks)
  rel_train_idx, rel_train_prob3, rel_train_prob5, rel_train_panel,
  rel_train_window, rel_train_truth, rel_train_elig
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\lenovo\Desktop\kaggle"
WORK = os.path.join(ROOT, "work")
SEQ = os.path.join(WORK, "task2_seq")
for _p in (os.path.join(WORK, "lib"), os.path.join(WORK, "task2"), SEQ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lhcore as LH  # noqa: E402
import run_cv as RC  # noqa: E402

CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)


def main():
    t0 = time.time()
    import lightgbm as lgb
    m5_path = os.path.join(CACHE, "ref_v5_onset.txt")
    m3_path = os.path.join(CACHE, "ref_v3_onset.txt")
    if os.path.exists(m5_path) and os.path.exists(m3_path):
        m5 = lgb.Booster(model_file=m5_path)
        m3 = lgb.Booster(model_file=m3_path)
        print("loaded cached reference boosters", flush=True)
        out = {}
        return store_all(out, m3, m5, t0)
    print("fitting v5 onset reference ...", flush=True)
    X, y = RC.gather_train("on", "v5", exclude_june=True, stride=1)
    m5 = RC.train_model("on", "v5", X, y, RC.ON_ROUNDS)
    m5.save_model(m5_path)
    del X, y
    gc.collect()
    print(f"  v5 fitted ({time.time()-t0:.0f}s)", flush=True)
    print("fitting v3 onset reference ...", flush=True)
    X, y = RC.gather_train("on", "v3", exclude_june=True, stride=1)
    m3 = RC.train_model("on", "v3", X, y, RC.ON_ROUNDS)
    m3.save_model(m3_path)
    del X, y
    gc.collect()
    print(f"  v3 fitted ({time.time()-t0:.0f}s)", flush=True)
    return store_all({}, m3, m5, t0)


def store_all(out, m3, m5, t0):
    for block in ("tune", "eval_feb"):
        p3 = RC.pseudo_probs("on", "v3", m3, block)
        p5 = RC.pseudo_probs("on", "v5", m5, block)
        for panel in p3:
            assert np.array_equal(p3[panel]["idx"], p5[panel]["idx"])
            key = f"{block}_{panel}"
            out[f"{key}_idx"] = p3[panel]["idx"].astype(np.int32)
            out[f"{key}_time"] = p3[panel]["time"].astype(np.int64)
            out[f"{key}_prob3"] = p3[panel]["prob"].astype(np.float32)
            out[f"{key}_prob5"] = p5[panel]["prob"].astype(np.float32)
            out[f"{key}_L"] = np.array(p3[panel]["L"], np.int32)
        print(f"  {block} probs stored ({time.time()-t0:.0f}s)", flush=True)

    panels = []
    prob3_by_panel, prob5_by_panel = {}, {}
    per_truth, per_elig = {}, {}
    for panel in LH.PANELS:
        z = RC.load_rel(panel)
        sel = np.where((z["split"] == "train") & (z["condition"] == "queue_onset"))[0]
        L = int(z["n_links"])
        per = LH.HORIZON * L
        x3 = RC.rel_X("on", "v3", z)
        x5 = RC.rel_X("on", "v5", z)
        rows = (sel[:, None] * per + np.arange(per)[None, :]).reshape(-1)
        q3 = m3.predict(np.ascontiguousarray(x3[rows])).astype(np.float32).reshape(len(sel), LH.HORIZON, L)
        q5 = m5.predict(np.ascontiguousarray(x5[rows])).astype(np.float32).reshape(len(sel), LH.HORIZON, L)
        panels += [panel] * len(sel)
        prob3_by_panel[panel] = q3
        prob5_by_panel[panel] = q5
        if panel == LH.PANELS[0]:
            wids = np.asarray(z["window_ids"])[sel]
        else:
            wids = np.concatenate([wids, np.asarray(z["window_ids"])[sel]])
        per_truth[panel] = z["truth"][sel].astype(np.int8)
        per_elig[panel] = z["elig"][sel].astype(bool)
        del x3, x5, z, q3, q5
        gc.collect()
    out["rel_train_panel"] = np.array(panels)
    out["rel_train_window"] = np.array(wids, dtype="<U64")
    for panel in LH.PANELS:
        out[f"rel_train_truth_{panel}"] = per_truth[panel]
        out[f"rel_train_elig_{panel}"] = per_elig[panel]
        out[f"rel_train_prob3_{panel}"] = prob3_by_panel[panel]
        out[f"rel_train_prob5_{panel}"] = prob5_by_panel[panel]
    path = os.path.join(CACHE, "ref_probs.npz")
    np.savez_compressed(path, **out)
    print(f"saved {path} ({time.time()-t0:.0f}s) windows: "
          f"{len(out['rel_train_window'])} released onset", flush=True)
    return out


if __name__ == "__main__":
    main()
