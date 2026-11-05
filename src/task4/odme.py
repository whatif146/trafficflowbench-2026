"""Task 4 (ODME) shared helpers: data loading, incidence, solvers, local scoring.

Reads the release directly from traffic.REL (read-only) so it does not depend on
the Lead's cache being built. Equivalent to traffic.load_task4 + traffic.build_incidence,
except the link ordering is sorted for determinism.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np
import pandas as pd

import traffic

REL = traffic.REL
PANELS = traffic.PANELS
FAMILIES = traffic.FAMILIES
WORK = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\work\task4"
REPORTS = os.path.join(WORK, "reports")
CAND = os.path.join(WORK, "candidates")
SPLITS = ("train", "validation", "private")


def net_dir(panel: str) -> str:
    return os.path.join(REL, "corridors", panel, "network")


def t4_dir(panel: str, split: str) -> str:
    return os.path.join(REL, "task4", panel, split)


def load_paths(panel: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(net_dir(panel), "path_set.csv"), dtype={"path_id": str})


def load_incidence(panel: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(net_dir(panel), "path_link_incidence.csv"),
                       dtype={"path_id": str, "link_id": str})


def link_order(panel: str) -> list:
    """Mainline + connector links ordered along the corridor (for structure work)."""
    links = pd.read_csv(os.path.join(net_dir(panel), "links.csv"), dtype={"link_id": str})
    topo = pd.read_csv(os.path.join(net_dir(panel), "lwr_mainline_topology.csv"), dtype=str)
    col = "order_index" if "order_index" in topo.columns else None
    if col is None:
        return sorted(links.link_id.tolist())
    t = topo[["mainline_link_id", col]].copy()
    t[col] = pd.to_numeric(t[col], errors="coerce")
    t = t.sort_values(col)
    return t["mainline_link_id"].tolist()


def build_A(panel: str):
    """A (n_links x n_paths), path_ids, link_ids.  link_ids sorted for determinism."""
    paths = load_paths(panel)
    inc = load_incidence(panel)
    path_ids = paths.path_id.tolist()
    link_ids = sorted(inc.link_id.unique().tolist())
    pidx = {p: i for i, p in enumerate(path_ids)}
    lidx = {l: i for i, l in enumerate(link_ids)}
    rr = inc.link_id.map(lidx)
    cc = inc.path_id.map(pidx)
    m = rr.notna() & cc.notna()
    A = np.zeros((len(link_ids), len(path_ids)), dtype=np.float64)
    A[rr[m].to_numpy(dtype=np.int64), cc[m].to_numpy(dtype=np.int64)] = 1.0
    return A, path_ids, link_ids, paths


def load_counts(panel: str, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(t4_dir(panel, split), "synthetic_link_counts.csv"),
                       dtype={"link_id": str})


def load_prior(panel: str, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(t4_dir(panel, split), "synthetic_weak_prior.csv"),
                       dtype={"path_id": str})


def load_template(panel: str, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(t4_dir(panel, split), "sample_submission_path_flow.csv"),
                       dtype={"path_id": str})


def split_arrays(panel: str, split: str):
    """Return dict with A (full), measured link mask, counts on measured links,
    prior b aligned to path_ids, and departure_time token."""
    A, path_ids, link_ids, paths = build_A(panel)
    cdf = load_counts(panel, split)
    measured = np.isin(np.array(link_ids), cdf.link_id.values)
    cnt = (cdf.set_index("link_id").reindex(np.array(link_ids)[measured])["count"]
           .fillna(0.0).to_numpy(dtype=float))
    pdf = load_prior(panel, split)
    b = (pdf.set_index("path_id").reindex(path_ids)["path_flow"]
         .fillna(0.0).to_numpy(dtype=float))
    dep = str(pdf.departure_time.iloc[0]) if len(pdf) else ""
    A_meas = A[measured]
    keep = A_meas.sum(axis=1) > 0
    return dict(A=A, A_meas=A_meas[keep], counts=cnt[keep], link_ids=link_ids,
                path_ids=path_ids, paths=paths, measured=measured,
                measured_link_ids=np.array(link_ids)[measured][keep],
                b=b, departure_time=dep)


# --------------------------------------------------------------------------- #
# solvers
# --------------------------------------------------------------------------- #
def nnls_prior(A_meas, counts, b, lam=0.05):
    """Official baseline: min ||A f - c||^2 + lam ||f - b||^2, f >= 0."""
    from scipy.optimize import nnls
    n = A_meas.shape[1]
    aa = np.vstack([A_meas, np.sqrt(lam) * np.eye(n)])
    bb = np.concatenate([counts, np.sqrt(lam) * b])
    f, _ = nnls(aa, bb)
    return f


def ridge_prior(A_meas, counts, b, lam, w=None):
    """Weighted ridge with warm-start-free closed form via nnls on augmented rows.

    w: optional per-path prior weight multiplier (uncertainty weighting).
    """
    from scipy.optimize import nnls
    n = A_meas.shape[1]
    sw = np.ones(n) if w is None else np.asarray(w, float)
    aa = np.vstack([A_meas, np.sqrt(lam) * np.diag(np.sqrt(sw))])
    bb = np.concatenate([counts, np.sqrt(lam) * sw ** 0.5 * b])
    f, _ = nnls(aa, bb)
    return f


def nnls_cd(A_meas, counts, b, lam=0.05, max_sweeps=400, tol=1e-9, verbose=False):
    """Exact-ish non-negative ridge by cyclic coordinate descent.

    min_f  ||A f - c||^2 + lam*||f - b||^2,  f >= 0
    A is (m x n) 0/1 with m << n (m = counted links, n = paths), so the
    Hessian is A'A + lam*I: one coordinate update is O(m).
    """
    A = np.asfortranarray(np.asarray(A_meas, dtype=np.float64))
    c = np.asarray(counts, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m, n = A.shape
    lam = float(lam)
    diag = np.einsum("ij,ij->j", A, A) + lam   # path length + lam
    # init: dual ridge solution clipped at 0 (cheap and close)
    r0 = c - A @ bb
    try:
        x = np.linalg.solve(A @ A.T + lam * np.eye(m), r0)
        f = np.clip(bb + A.T @ x, 0.0, None)
    except np.linalg.LinAlgError:
        f = np.clip(bb, 0.0, None)
    r = c - A @ f                              # current count residual
    # gradient-based rhs at current f:  A'(c - A f) + lam*(b - f)
    prev = None
    for sweep in range(max_sweeps):
        maxdelta = 0.0
        for i in range(n):
            col = A[:, i]
            rhs = col @ r + lam * bb[i]
            fi = rhs / diag[i]
            if fi < 0.0:
                fi = 0.0
            d = fi - f[i]
            if d != 0.0:
                f[i] = fi
                r -= d * col
                if abs(d) > maxdelta:
                    maxdelta = abs(d)
        if verbose and (sweep % 20 == 0):
            print(f"   sweep {sweep} maxdelta {maxdelta:.3e}", flush=True)
        if maxdelta < tol * max(1.0, float(np.max(f))):
            break
        if prev is not None and abs(prev - maxdelta) < 1e-12:
            break
        prev = maxdelta
    return f


def nnls_dual(A_meas, counts, b, lam=0.05):
    """Unconstrained ridge closed form (may be negative), clipped at 0."""
    r0 = np.asarray(counts, float) - A_meas @ np.asarray(b, float)
    m = A_meas.shape[0]
    x = np.linalg.solve(A_meas @ A_meas.T + lam * np.eye(m), r0)
    return np.clip(np.asarray(b, float) + A_meas.T @ x, 0.0, None)


# --------------------------------------------------------------------------- #
# local scoring (mirrors src/task4/score_task4.py)
# --------------------------------------------------------------------------- #
def s_link_panel(f, split_data) -> float:
    loaded = split_data["A_meas"] @ np.asarray(f, float)
    counts = split_data["counts"]
    denom = max(float(np.sum(counts)), 1e-9)
    return max(0.0, 1.0 - float(np.sum(np.abs(loaded - counts))) / denom)


def s_link_overall(f_by_panel, split: str) -> dict:
    per = {}
    for p in PANELS:
        sd = split_arrays(p, split)
        per[p] = s_link_panel(f_by_panel[p], sd)
    fam = {fam: float(np.mean([per[p] for p in ps])) for fam, ps in FAMILIES.items()}
    overall = float(np.mean(list(fam.values())))
    return {"panel": per, "family": fam, "overall": overall}


# --------------------------------------------------------------------------- #
# submission writer
# --------------------------------------------------------------------------- #
def write_submission(f_by_panel, split: str, out_csv: str):
    frames = []
    for p in PANELS:
        tpl = load_template(p, split)
        f = np.asarray(f_by_panel[p], float)
        f = np.where(np.isfinite(f) & (f >= 0), f, 0.0)
        tpl = tpl.copy()
        tpl["path_flow"] = f
        frames.append(tpl[["panel", "departure_time", "path_id", "origin_zone",
                           "destination_zone", "path_flow"]])
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out
