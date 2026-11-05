"""Task 4 path-flow submission builder.

Usage:
    python predict.py --split validation --output out/path_flow_validation.csv
    python predict.py --split private    --output out/path_flow_private.csv
    python predict.py --split train      --output out/path_flow_train.csv

Methods (--method):
    baseline          official regularised NNLS, lambda = 0.05  (safe default)
    ridge:<lam>       same estimator with a different prior weight
    prior             submit the released weak prior unchanged
    prior_scale       prior rescaled by the least-squares cut-profile factor
    blend:<a>         a*prior_scale + (1-a)*ridge_solution  (after rescaling)
    share             OD-flow = prior share x cut-implied totals

Rows are emitted in submission-template order, one row per template row, with
finite non-negative path_flow. Panels are processed one at a time (low memory).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import odme


def solve_ridge_nnls(A, c, b, lam=0.05, x0=None, maxiter=2000):
    """min ||A f - c||^2 + lam ||f - b||^2 s.t. f >= 0  (L-BFGS-B, exact)."""
    A = np.asarray(A, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    def fun(f):
        r = A @ f - c
        return 0.5 * float(r @ r) + 0.5 * lam * float(np.sum((f - b) ** 2))

    def jac(f):
        return A.T @ (A @ f - c) + lam * (f - b)

    start = np.clip(b, 1e-9, None) if x0 is None else np.clip(x0, 0.0, None)
    res = minimize(fun, start, jac=jac, method="L-BFGS-B",
                   bounds=[(0.0, None)] * len(b),
                   options=dict(maxiter=maxiter, maxfun=3 * maxiter + 10,
                                ftol=1e-15, gtol=1e-11, maxcor=40))
    return np.clip(res.x, 0.0, None), res


def build_panel(panel: str, split: str, method: str, lam: float):
    sd = odme.split_arrays(panel, split)
    A, c, b = sd["A_meas"], sd["counts"], sd["b"]
    if method == "prior":
        f = b.copy()
    elif method == "prior_scale":
        # least-squares scalar on the cut profile, then non-negative
        s = float((c @ (A @ b)) / max(float(np.sum((A @ b) ** 2)), 1e-12))
        f = np.clip(s * b, 0.0, None)
    elif method.startswith("entropy"):
        # max-entropy / minimum-KL projection of the prior onto the cuts
        #   entropy          prior = own split prior
        #   entropy_pool     prior = mean of the three splits' priors
        #   entropy_shrink:<k>  prior = pooled + k*(own - pooled)
        import entropy as entropy_mod
        prior_mode = "own"
        if method == "entropy_pool":
            prior_mode = "pool"
        elif method.startswith("entropy_shrink:"):
            prior_mode = "shrink:" + method.split(":", 1)[1]
        f, sd = entropy_mod.make_entropy(panel, split, prior_mode)
        return f, sd
    else:
        f, res = solve_ridge_nnls(A, c, b, lam=lam)
        if method.startswith("blend:"):
            alpha = float(method.split(":", 1)[1])
            s = float((c @ (A @ b)) / max(float(np.sum((A @ b) ** 2)), 1e-12))
            f = np.clip(alpha * (s * b) + (1.0 - alpha) * f, 0.0, None)
    return f, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation", "private"], required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--method", default="baseline",
                    help="baseline | ridge:<lam> | prior | prior_scale | blend:<alpha> | entropy | entropy_pool | entropy_shrink:<k>")
    ap.add_argument("--lam", type=float, default=None, help="override prior weight")
    ap.add_argument("--panels", nargs="*", default=None)
    args = ap.parse_args()

    method = args.method
    lam = args.lam
    if lam is None:
        lam = 0.05
        if method.startswith("ridge:"):
            lam = float(method.split(":", 1)[1])

    panels = args.panels if args.panels else odme.PANELS
    frames = []
    per_panel = {}
    t0 = time.time()
    for p in panels:
        t = time.time()
        f, sd = build_panel(p, args.split, method, lam)
        per_panel[p] = f
        sl = odme.s_link_panel(f, sd)
        tpl = odme.load_template(p, args.split)
        assert len(tpl) == len(f), (p, len(tpl), len(f))
        out = tpl[["panel", "departure_time", "path_id", "origin_zone",
                   "destination_zone"]].copy()
        out["path_flow"] = f
        frames.append(out)
        print(f"[task4] {p:12s} n={len(f):5d} sum={f.sum():10.1f} "
              f"S_link={sl:.5f} nnz={int((f > 0).sum()):5d} {time.time()-t:.1f}s",
              flush=True)
    ##############################################################
    fam = {}
    for fam_id, ps in odme.FAMILIES.items():
        if all(p in per_panel for p in ps):
            vals = []
            for p in ps:
                sd = odme.split_arrays(p, args.split)
                vals.append(odme.s_link_panel(per_panel[p], sd))
            fam[fam_id] = float(np.mean(vals))
    if fam:
        print(f"[task4] family S_link: " +
              " ".join(f"{k}={v:.5f}" for k, v in fam.items()), flush=True)
        print(f"[task4] family-mean S_link = {np.mean(list(fam.values())):.6f}", flush=True)

    sub = pd.concat(frames, ignore_index=True)
    assert np.isfinite(sub.path_flow.to_numpy()).all()
    assert (sub.path_flow.to_numpy() >= 0).all()
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f"[task4] wrote {len(sub):,} rows to {args.output} in {time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()

