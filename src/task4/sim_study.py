"""Simulation study: which estimator would win under plausible data-generating processes.

All simulated truths satisfy A f* = c exactly (the identified part is the same as
the real task); they differ in the null-space allocation and the total:

  dgp 'smooth'  latent OD = gravity-smoothed pooled prior   (per-OD noise removed)
  dgp 'raw'     latent OD = pooled prior (keeps per-OD structure)
  dgp 'own'     latent OD = this split's own prior
  each with optional independent lognormal per-OD noise sigma, then projected
  onto the cut constraints with the entropy (min-KL) projection.

Estimators: baseline NNLS, pooled-prior NNLS, autoscale, marginal correction,
gravity denoised + marginal correction, entropy projection (own/pooled).
Metrics follow S_ODME exactly: 0.45 S_od + 0.25 S_link + 0.15 S_dev + 0.15 S_attr.
"""
import os
import sys
import time

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import cand
import entropy
import odme

pd.set_option("display.width", 250)
REP = odme.REPORTS
PANELS = ["D12_I405_N", "D7_I210_E", "D7_I10_E", "D12_I5_N"]
SPLIT = "validation"
DGPS = [("raw", 0.0), ("raw", 0.4), ("smooth", 0.0), ("smooth", 0.4), ("own", 0.3)]
ESTIMATORS = ["baseline", "pool", "autoscale", "margcorr", "denoise_margcorr",
              "ent_own", "ent_pool"]


def metrics(f, fstar, sd, b):
    A, c = sd["A_meas"], sd["counts"]
    sl = max(0.0, 1.0 - float(np.abs(A @ f - c).sum()) / max(c.sum(), 1e-9))
    sod = max(0.0, 1.0 - float(np.abs(f - fstar).sum()) / max(fstar.sum(), 1e-9))
    dstar = float(np.abs(fstar - b).sum())
    dhat = float(np.abs(f - b).sum())
    sdev = float(np.exp(-abs(dhat / max(dstar, 1e-9) - 1.0)))
    oi, di = cand.path_meta(cur_panel)
    def dest_share(x):
        X = np.bincount(di, weights=x, minlength=int(di.max()) + 1).astype(float)
        return X / max(X.sum(), 1e-12)
    a1, a2 = dest_share(f), dest_share(fstar)
    sattr = max(0.0, 1.0 - 0.5 * float(np.abs(a1 - a2).sum()))
    s = 0.45 * sod + 0.25 * sl + 0.15 * sdev + 0.15 * sattr
    return dict(S_link=sl, S_od=sod, S_dev=sdev, S_attr=sattr, S_ODME=s)


def truth(panel, split, kind, sigma, rng):
    sd = odme.split_arrays(panel, split)
    oi, di = cand.path_meta(panel)
    bpool = np.mean([odme.split_arrays(panel, s)["b"] for s in odme.SPLITS], axis=0)
    if kind == "smooth":
        latent = cand.denoise_prior(bpool, oi, di)
    elif kind == "raw":
        latent = bpool
    elif kind == "own":
        latent = sd["b"]
    else:
        raise ValueError(kind)
    if sigma > 0:
        latent = latent * np.exp(rng.normal(0.0, sigma, len(latent)))
    fstar, _ = entropy.entropy_project(
        latent, oi, di, sd["A_meas"], sd["counts"],
        seg_rows=entropy.segment_of_rows(sd["A"], oi, sd["measured"]))
    return sd, fstar


def main():
    global cur_panel
    rows = []
    rng = np.random.default_rng(20261101)
    for panel in PANELS:
        cur_panel = panel
        for kind, sigma in DGPS:
            sd, fstar = truth(panel, SPLIT, kind, sigma, rng)
            for est in ESTIMATORS:
                if est.startswith("ent_"):
                    f, sd2 = entropy.make_entropy(panel, SPLIT, est.split("_", 1)[1])
                else:
                    f, sd2 = cand.make(panel, SPLIT, est)
                m = metrics(f, fstar, sd2, sd2["b"])
                m.update(panel=panel, dgp=f"{kind}_s{sigma}", estimator=est,
                         truth_sum=float(fstar.sum()), est_sum=float(f.sum()))
                rows.append(m)
            print(f"[sim] {panel} {kind} sigma={sigma} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(REP, "simulation_study.csv"), index=False)
    print("\n=== mean S_ODME by estimator x DGP ===")
    print(df.pivot_table(index="estimator", columns="dgp", values="S_ODME").round(4).to_string())
    print("\n=== mean S_od ===")
    print(df.pivot_table(index="estimator", columns="dgp", values="S_od").round(4).to_string())
    print("\n=== mean S_dev ===")
    print(df.pivot_table(index="estimator", columns="dgp", values="S_dev").round(4).to_string())
    print("\n=== mean S_attr ===")
    print(df.pivot_table(index="estimator", columns="dgp", values="S_attr").round(4).to_string())


if __name__ == "__main__":
    main()
