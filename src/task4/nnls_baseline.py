"""Official-baseline NNLS ODME solver for validation/private (fallback).

min ||A f - c||^2 + lam ||f - b||^2, f >= 0, on the counted links only, exactly
like src/task4/build_task4_odme_artifacts.py but written for our cache and for
both scored splits in one pass.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
import numpy as np
import pandas as pd
from scipy.optimize import nnls

import traffic

LAM = 0.05
COLS = ["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]


def solve_panel(panel, split):
    t4 = traffic.load_task4(panel, split)
    paths = t4["paths"]
    path_ids = paths["path_id"].tolist()
    inc = t4["incidence"]
    link_ids = sorted(inc["link_id"].unique().tolist())
    pidx = {p: i for i, p in enumerate(path_ids)}
    lidx = {l: i for i, l in enumerate(link_ids)}
    rr = inc["link_id"].map(lidx)
    cc = inc["path_id"].map(pidx)
    A = np.zeros((len(link_ids), len(path_ids)))
    m = rr.notna() & cc.notna()
    A[rr[m].to_numpy(int), cc[m].to_numpy(int)] = 1.0
    counts = t4["counts"].set_index("link_id")
    measured = np.array([l in counts.index for l in link_ids])
    c = counts.reindex(np.array(link_ids)[measured])["count"].to_numpy(float)
    b = (t4["prior"].set_index("path_id").reindex(path_ids)["path_flow"]
         .fillna(0.0).to_numpy(float))
    A_meas = A[measured]
    keep = A_meas.sum(axis=1) > 0
    A_meas, c = A_meas[keep], c[keep]
    n = A_meas.shape[1]
    aa = np.vstack([A_meas, np.sqrt(LAM) * np.eye(n)])
    bb = np.concatenate([c, np.sqrt(LAM) * b])
    f, _ = nnls(aa, bb)
    dep = str(t4["prior"].departure_time.iloc[0])
    out = paths[["path_id", "origin_zone", "destination_zone"]].copy()
    out["panel"] = panel
    out["departure_time"] = dep
    out["path_flow"] = f
    loaded = A_meas @ f
    s_link = max(0.0, 1 - np.abs(loaded - c).sum() / max(c.sum(), 1e-9))
    return out[COLS], float(s_link)


def main():
    base = os.path.join(traffic.WORK, "task4", "out")
    os.makedirs(base, exist_ok=True)
    for split in ("validation", "private"):
        frames, scores = [], []
        for panel in traffic.PANELS:
            out, s = solve_panel(panel, split)
            frames.append(out)
            scores.append(s)
            print(f"{split} {panel:12s} S_link={s:.4f}", flush=True)
        df = pd.concat(frames, ignore_index=True)
        path = os.path.join(base, f"path_flow_{split}.csv")
        df.to_csv(path, index=False)
        print(f"{split}: {len(df):,} rows, mean S_link={np.mean(scores):.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
