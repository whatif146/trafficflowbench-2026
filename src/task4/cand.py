"""Candidate Task-4 estimators.

All estimators produce f >= 0 over the panel's path set (path == OD pair, one path
per ordered zone pair). Geometry: linear corridor, counted mainline links give the
cut flows C_k = sum of OD flows whose interval crosses segment k.

Methods
-------
baseline          min ||A f - c||^2 + lam||f - b||^2, f>=0  (official baseline)
scale:<s>         same, but with prior s*b (tests the total-scale hypothesis)
pool              same, but with the 3-split mean prior (denoise split noise)
shrink:<k>        prior = pooled + k*(own - pooled)
margcorr          multiplicative marginal correction: f = b*exp(a_i + d_j),
                  a,d fitted so that A f = c (keeps prior's per-OD interaction,
                  fixes the systematic marginal trend error), then nnls touch-up
denoise           gravity-denoised prior (log-linear in origin/dest/distance),
                  level-matched to own prior, then nnls
denoise_margcorr  denoise, then margcorr
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

import odme

CACHE = {}


def path_meta(panel):
    """origin/dest integer index per path (aligned to odme path order) and segcount."""
    paths = odme.load_paths(panel)
    oi = paths.origin_zone.str[1:].astype(int).to_numpy()
    di = paths.destination_zone.str[1:].astype(int).to_numpy()
    return oi, di


def get_prior(panel, split, mode="own"):
    key = (panel, split, mode)
    if key in CACHE:
        return CACHE[key]
    sd = odme.split_arrays(panel, split)
    own = sd["b"]
    if mode == "own":
        b = own
    else:
        parts = [odme.split_arrays(panel, s)["b"] for s in odme.SPLITS]
        pooled = np.mean(parts, axis=0)
        if mode == "pool":
            b = pooled
        elif mode.startswith("shrink:"):
            k = float(mode.split(":", 1)[1])
            b = pooled + k * (own - pooled)
        else:
            raise ValueError(mode)
    CACHE[key] = b
    return b


def solve_ridge(A, c, b, lam=0.05, maxiter=2000):
    A = np.asarray(A, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def fun(f):
        r = A @ f - c
        return 0.5 * float(r @ r) + 0.5 * lam * float(np.sum((f - b) ** 2))

    def jac(f):
        return A.T @ (A @ f - c) + lam * (f - b)

    res = minimize(fun, np.clip(b, 1e-9, None), jac=jac, method="L-BFGS-B",
                   bounds=[(0.0, None)] * len(b),
                   options=dict(maxiter=maxiter, maxfun=3 * maxiter + 10,
                                ftol=1e-15, gtol=1e-11, maxcor=40))
    return np.clip(res.x, 0.0, None)


def denoise_prior(b, oi, di):
    """Project log b on intercept + origin + destination + segment-count effects."""
    y = np.log(np.clip(b, 1e-9, None))
    n = len(y)
    uo = {z: i for i, z in enumerate(np.unique(oi))}
    ud = {z: i for i, z in enumerate(np.unique(di))}
    uo = {z: i for i, z in enumerate(sorted(uo))}
    ud = {z: i for i, z in enumerate(sorted(ud))}
    cols = [np.ones(n), (di - oi).astype(float)]
    X = np.zeros((n, 2 + len(uo) + len(ud)))
    X[:, 0] = 1.0
    X[:, 1] = (di - oi).astype(float)
    X[np.arange(n), 2 + np.array([uo[z] for z in oi])] = 1.0
    X[np.arange(n), 2 + len(uo) + np.array([ud[z] for z in di])] = 1.0
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    m = np.exp(X @ coef)
    m *= float(b.sum() / max(m.sum(), 1e-12))
    return m


def margcorr(A, c, b, oi, di, w_reg=1e-6, bounds=3.0, lam_touch=0.05, touch=True,
             max_iter=60, verbose=False):
    """f = b*exp(a_oi + d_di) with (a,d) fitted so A f ~= c (damped Gauss-Newton).

    Uses the relative cut residual (each segment row divided by its mean level) so
    the marginal corrections are not dominated by the largest cuts.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    m = A.shape[0]
    level = A @ (b / max(b.sum(), 1e-12))
    level = np.where(level > 0, level, 1.0)
    uo = {z: i for i, z in enumerate(sorted(set(oi.tolist())))}
    ud = {z: i for i, z in enumerate(sorted(set(di.tolist())))}
    no, nd = len(uo), len(ud)
    oi_i = np.array([uo[z] for z in oi])
    di_i = np.array([ud[z] for z in di])
    r_scale = float(np.sqrt(np.mean(((A @ b - c) / level) ** 2))) + 1e-12

    def predict(theta):
        return b * np.exp(theta[:no][oi_i] + theta[no:][di_i])

    def res_jac(theta):
        f = predict(theta)
        r = (A @ f - c) / level / r_scale
        J = np.zeros((m + no + nd, no + nd))
        for k in range(no):
            J[:m, k] = (A @ (f * (oi_i == k))) / level / r_scale
        for k in range(nd):
            J[:m, no + k] = (A @ (f * (di_i == k))) / level / r_scale
        J[m:, :] = np.sqrt(w_reg) * np.eye(no + nd)
        return np.concatenate([r, np.sqrt(w_reg) * theta]), J

    theta = np.zeros(no + nd)
    r, J = res_jac(theta)
    obj = float(r @ r)
    lam = 1e-2
    for it in range(max_iter):
        # Levenberg-Marquardt step on the (regularised) nonlinear least squares
        g = J.T @ r
        H = J.T @ J
        scale = float(np.mean(np.diag(H))) + 1e-12
        step = np.linalg.solve(H + lam * scale * np.eye(no + nd), -g)
        cand_theta = np.clip(theta + step, -bounds, bounds)
        r_new, J_new = res_jac(cand_theta)
        obj_new = float(r_new @ r_new)
        if obj_new < obj:
            theta, r, J, obj = cand_theta, r_new, J_new, obj_new
            lam = max(lam * 0.3, 1e-12)
            if float(np.max(np.abs(step))) < 1e-9:
                break
        else:
            lam *= 10.0
            if lam > 1e8:
                break
    if verbose:
        print(f"   margcorr iters={it} obj={obj:.3e} theta_max={np.abs(theta).max():.3f}")
    f = predict(theta)
    f = np.where(np.isfinite(f) & (f > 0), f, 0.0)
    if touch:
        f = solve_ridge(A, c, f, lam=lam_touch)
    return f, theta


def make(panel, split, method):
    """Return f for one panel."""
    sd = odme.split_arrays(panel, split)
    A, c, bown = sd["A_meas"], sd["counts"], sd["b"]
    oi, di = path_meta(panel)
    oi = oi[sd["paths"].path_id.isin(sd["path_ids"]).to_numpy()] if False else oi
    # path order in split_arrays is odme.build_A order == path_set order
    method = method.strip()
    if method == "baseline":
        return solve_ridge(A, c, bown, 0.05), sd
    if method.startswith("scale:"):
        s = float(method.split(":", 1)[1])
        return solve_ridge(A, c, s * bown, 0.05), sd
    if method == "autoscale":
        # global prior scale chosen so the implied cut LEVEL matches the counts
        Ab = A @ bown
        s = float((c @ Ab) / max(Ab @ Ab, 1e-12))
        return solve_ridge(A, c, s * bown, 0.05), sd
    if method == "pool" or method.startswith("shrink:"):
        b = get_prior(panel, split, method if method != "pool" else "pool")
        return solve_ridge(A, c, b, 0.05), sd
    if method == "denoise":
        b = denoise_prior(bown, oi, di)
        return solve_ridge(A, c, b, 0.05), sd
    if method.startswith("denoise_margcorr"):
        b = denoise_prior(bown, oi, di)
        f, _ = margcorr(A, c, b, oi, di)
        return f, sd
    if method == "margcorr":
        f, _ = margcorr(A, c, bown, oi, di)
        return f, sd
    if method.startswith("margcorr_pool"):
        b = get_prior(panel, split, "pool")
        f, _ = margcorr(A, c, b, oi, di)
        return f, sd
    raise ValueError(method)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--panels", nargs="*")
    args = ap.parse_args()
    panels = args.panels if args.panels else odme.PANELS
    frames = {}
    for p in panels:
        f, sd = make(p, args.split, args.method)
        sl = odme.s_link_panel(f, sd)
        dhat = float(np.sum(np.abs(f - sd["b"])))
        print(f"[{args.method}] {p:12s} sum={f.sum():10.1f} S_link={sl:.5f} "
              f"Dhat={dhat:10.1f} Dhat/sum_b={dhat/sd['b'].sum():.3f} "
              f"nnz={int((f>0).sum())}", flush=True)
        tpl = odme.load_template(p, args.split)
        out = tpl[["panel", "departure_time", "path_id", "origin_zone",
                   "destination_zone"]].copy()
        out["path_flow"] = f
        frames[p] = out
    fams = []
    for fam, ps in odme.FAMILIES.items():
        vals = []
        for p in ps:
            if p in frames:
                sd = odme.split_arrays(p, args.split)
                f = frames[p].path_flow.to_numpy()
                vals.append(odme.s_link_panel(f, sd))
        if vals:
            fams.append(float(np.mean(vals)))
            print(f"[{args.method}] family {fam} S_link={np.mean(vals):.5f}")
    if fams:
        print(f"[{args.method}] OVERALL S_link = {np.mean(fams):.6f}")
    sub = pd.concat(frames.values(), ignore_index=True)
    assert np.isfinite(sub.path_flow.to_numpy()).all() and (sub.path_flow.to_numpy() >= 0).all()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f"[{args.method}] wrote {len(sub)} rows -> {args.output}")


if __name__ == "__main__":
    main()
