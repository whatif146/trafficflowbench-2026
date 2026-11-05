"""Maximum-entropy (minimum-KL) projection of the weak prior onto the cut counts.

For a linear corridor, a path (i,j) crosses exactly segments i..j-1, so a
multiplicative correction constant per segment has the potential form
    f_ij = b_ij * exp(L_j - L_i),  L_0 = 0,
with one potential per zone boundary (nzones-1 free parameters = number of
independent cut constraints).  Solving A f = c for L is the exact minimum-KL
projection of b onto the released counts: the canonical ODME estimator.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import odme
import cand


def cut_profile(f, oi, di, M):
    """C_k = sum_{i<=k<j} f_ij, k = 0..M-1."""
    F = np.zeros((M + 1, M + 1))
    np.add.at(F, (oi, di), f)
    rowpref = np.cumsum(F.sum(axis=1))
    colpref = np.cumsum(F.sum(axis=0))
    k = np.arange(M)
    return rowpref[k] - colpref[k]


def segment_of_rows(A_full, oi, measured):
    """Segment index k of every measured link row (k = max origin zone of paths using it)."""
    Ared = A_full[measured]
    seg = np.zeros(Ared.shape[0], dtype=int)
    for r in range(Ared.shape[0]):
        mask = Ared[r] > 0
        seg[r] = int(oi[mask].max())
    return seg


def entropy_project(b, oi, di, A_meas, c, seg_rows=None, max_iter=100, verbose=False):
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    M = int(max(oi.max(), di.max()))          # zones 0..M, segments 0..M-1
    level = cut_profile(b, oi, di, M)
    level = np.where(np.abs(level) > 1e-9, level, 1.0)
    if seg_rows is None:
        seg_rows = np.arange(A_meas.shape[0]) % M
    level_row = level[seg_rows]
    r_scale = float(np.sqrt(np.mean(((A_meas @ b - c) / level_row) ** 2))) + 1e-12

    def predict(Lfull):
        return b * np.exp(Lfull[di] - Lfull[oi])

    def resid_jac(theta):
        Lfull = np.concatenate([[0.0], theta])
        f = predict(Lfull)
        r = (A_meas @ f - c) / level_row / r_scale
        J = np.zeros((A_meas.shape[0], M))
        for p in range(1, M + 1):
            col = f * ((di == p).astype(float) - (oi == p).astype(float))
            J[:, p - 1] = (A_meas @ col) / level_row / r_scale
        return r, J

    theta = np.zeros(M)
    r, J = resid_jac(theta)
    obj = float(r @ r)
    lam = 1e-2
    it = 0
    for it in range(max_iter):
        g = J.T @ r
        H = J.T @ J
        sc = float(np.mean(np.diag(H))) + 1e-12
        step = np.linalg.solve(H + lam * sc * np.eye(M), -g)
        new = theta + step
        r_new, J_new = resid_jac(new)
        obj_new = float(r_new @ r_new)
        if obj_new < obj:
            theta, r, J, obj = new, r_new, J_new, obj_new
            lam = max(lam * 0.3, 1e-13)
            if float(np.max(np.abs(step))) < 1e-10:
                break
        else:
            lam *= 10.0
            if lam > 1e10:
                break
    Lfull = np.concatenate([[0.0], theta])
    f = predict(Lfull)
    if verbose:
        print(f"   entropy iters={it} obj={obj:.3e} Lrange=({Lfull.min():.2f},{Lfull.max():.2f})", flush=True)
    return np.where(np.isfinite(f) & (f > 0), f, 0.0), Lfull


def get_prior(panel, split, mode):
    sd = odme.split_arrays(panel, split)
    if mode == "own":
        return sd["b"]
    pooled = np.mean([odme.split_arrays(panel, s)["b"] for s in odme.SPLITS], axis=0)
    if mode == "pool":
        return pooled
    if mode.startswith("shrink:"):
        k = float(mode.split(":", 1)[1])
        return pooled + k * (sd["b"] - pooled)
    raise ValueError(mode)


def make_entropy(panel, split, prior="own", verbose=False):
    sd = odme.split_arrays(panel, split)
    oi, di = cand.path_meta(panel)
    b = get_prior(panel, split, prior)
    f, Lp = entropy_project(b, oi, di, sd["A_meas"], sd["counts"],
                            seg_rows=segment_of_rows(sd["A"], oi, sd["measured"]),
                            verbose=verbose)
    return f, sd


if __name__ == "__main__":
    import time
    for p in ["D12_I405_N", "D7_I10_W"]:
        for mode in ("own", "pool"):
            t = time.time()
            f, sd = make_entropy(p, "validation", mode, verbose=True)
            print(p, mode, "%.2fs" % (time.time() - t), "S_link", round(odme.s_link_panel(f, sd), 8),
                  "sum", round(f.sum()), "sum_b", round(sd["b"].sum()), flush=True)
