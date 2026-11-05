"""Prior structure analysis for Task 4.

Questions:
  A. What structural model generates the weak prior? (gravity: row/col effects + distance decay)
  B. Is the shared (cross-split) part of the prior closer to the truth's cut profile
     than each split's own prior? (pooling test, using the released counts as truth proxy)
  C. Does the split-specific part of the prior track the split-specific part of truth?
     (regression of cut-profile split deviations; decides how much to pool)
  D. How does the prior's implied cut profile relate to the counts after scaling?
"""
import sys, os
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, pandas as pd
import odme

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
REP = odme.REPORTS


def geom(panel):
    """Return path -> (i,j,segcount,length), segment index per mainline link."""
    paths = odme.load_paths(panel)
    paths["oi"] = paths.origin_zone.str[1:].astype(int)
    paths["di"] = paths.destination_zone.str[1:].astype(int)
    fits = pd.read_csv(os.path.join(odme.net_dir(panel), "links.csv"), dtype={"link_id": str})
    km = dict(zip(fits.link_id, fits.length_km))
    fz = paths.oi.min()
    cand = paths[paths.oi == fz].copy()
    cand["n"] = cand.link_seq.str.count(";") + 1
    full = cand.sort_values("n").iloc[-1]
    chain = full.link_seq.split(";")
    pref = {r.di: r.link_seq.split(";") for r in cand.itertuples()}
    seglen, prev = {}, 0
    for z in sorted(pref):
        seglen[z - 1] = len(pref[z]) - prev
        prev = len(pref[z])
    seg_of, cum = {}, 0
    for k in sorted(seglen):
        for l in chain[cum:cum + seglen[k]]:
            seg_of[l] = k
        cum += seglen[k]
    lg = {p: sum(km.get(l, 0.0) for l in s.split(";")) for p, s in
          zip(paths.path_id, paths.link_seq)}
    return paths, seg_of, lg, len(seglen)


rows = []
cutrows = []
for panel in odme.PANELS:
    paths, seg_of, lg, nseg = geom(panel)
    b = {}
    Cs = {}
    for split in ("train", "validation", "private"):
        sd = odme.split_arrays(panel, split)
        b[split] = sd["b"]
        # cut profile (per counted segment) of counts and of prior
        segs = np.array([seg_of[l] for l in sd["measured_link_ids"]])
        prof_b = pd.Series(sd["A_meas"] @ sd["b"]).groupby(segs).mean()
        prof_c = pd.Series(sd["counts"]).groupby(segs).mean()
        idx = sorted(set(prof_b.index) & set(prof_c.index))
        Cs[split] = dict(segs=np.array(idx), c=prof_c.loc[idx].to_numpy(float),
                         Ab=prof_b.loc[idx].to_numpy(float))
    idx = Cs["validation"]["segs"]
    c = Cs["validation"]["c"]
    # ---- A: gravity fit on pooled prior ----
    m = np.mean([b[s] for s in b], axis=0)
    oi = paths.set_index("path_id").reindex(sd["path_ids"]).oi.to_numpy()
    di = paths.set_index("path_id").reindex(sd["path_ids"]).di.to_numpy()
    L = np.array([lg[p] for p in sd["path_ids"]])
    segc = (di - oi).astype(float)
    y = np.log(np.clip(m, 1e-9, None))
    # design: intercept + origin one-hot + dest one-hot + distance
    n = len(y)
    X = np.zeros((n, 1 + len(np.unique(oi)) + len(np.unique(di)) + 2))
    X[:, 0] = 1.0
    uo = {z: i for i, z in enumerate(sorted(set(oi)))}
    ud = {z: i for i, z in enumerate(sorted(set(di)))}
    X[np.arange(n), 1 + np.array([uo[z] for z in oi])] = 1.0
    X[np.arange(n), 1 + len(uo) + np.array([ud[z] for z in di])] = 1.0
    X[:, -2] = segc
    X[:, -1] = L
    coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    rows.append(dict(panel=panel, nzones=int(max(oi.max(), di.max()) - oi.min() + 1),
                     npaths=n, rank=y.size - (X.shape[1] - 1),
                     grav_R2=1 - ss_res / ss_tot, grav_resid_sd=float(np.std(y - pred)),
                     beta_seg=float(coef[-2]), beta_len=float(coef[-1]),
                     sd_logb=float(np.std(y)),
                     sumb_va=float(b["validation"].sum()),
                     sumAb_va=float(Cs["validation"]["Ab"].sum()),
                     sumc_va=float(c.sum()),
                     ratio_va=float(Cs["validation"]["Ab"].sum() / c.sum())))
    # ---- B: cut-profile match of each split's prior and of pooled prior ----
    for split in ("train", "validation", "private"):
        cb = Cs[split]
        # truth = counts of THAT split; prior = that split's prior / pooled prior
        for name, bb in (("own", b[split]), ("pool", m)):
            sd = odme.split_arrays(panel, split)
            prof = pd.Series(sd["A_meas"] @ bb).groupby(
                np.array([seg_of[l] for l in sd["measured_link_ids"]])).mean()
            x = cb["c"]; yy = prof.reindex(cb["segs"]).to_numpy(float)
            s = float((x @ yy) / max(yy @ yy, 1e-12))
            cutrows.append(dict(panel=panel, split=split, prior=name,
                                corr=float(np.corrcoef(x, yy)[0, 1]),
                                relL1_scaled=float(np.abs(x - s * yy).sum() / x.sum()),
                                scale=s))
    # ---- C: split-deviation regression ----
    # y_dev = c_s - mean_s c ; x_dev = Ab_s - mean_s Ab
    ylist, xlist = [], []
    for split in ("train", "validation", "private"):
        sd = odme.split_arrays(panel, split)
        segs = np.array([seg_of[l] for l in sd["measured_link_ids"]])
        prof_b = pd.Series(sd["A_meas"] @ b[split]).groupby(segs).mean()
        prof_c = pd.Series(sd["counts"]).groupby(segs).mean()
        ii = sorted(set(prof_b.index) & set(prof_c.index))
        ylist.append(prof_c.loc[ii].to_numpy(float)); xlist.append(prof_b.loc[ii].to_numpy(float))
    Y = np.array(ylist); Xb = np.array(xlist)
    yd = (Y - Y.mean(axis=0)); xd = (Xb - Xb.mean(axis=0))
    # normalize each segment by its mean level to compare relative deviations
    lvl = Y.mean(axis=0)
    ydn = yd / lvl; xdn = xd / lvl
    rows[-1]["split_dev_corr"] = float(np.corrcoef(ydn.ravel(), xdn.ravel())[0, 1])
    rows[-1]["split_dev_ratio"] = float(np.std(ydn) / max(np.std(xdn), 1e-12))
    rows[-1]["n_seg"] = len(cb["segs"])

df = pd.DataFrame(rows)
print("=== A/C: gravity structure of pooled prior + split-deviation tracking ===")
print(df.round(4).to_string(index=False))
cd = pd.DataFrame(cutrows)
print("\n=== B: cut-profile match (truth=released counts of the same split) ===")
print(cd.pivot_table(index=["panel", "split"], columns="prior",
                     values=["corr", "relL1_scaled"]).round(4).to_string())
print("\nmean by prior type:")
print(cd.groupby("prior")[["corr", "relL1_scaled"]].mean().round(4).to_string())
df.to_csv(os.path.join(REP, "prior_gravity_diag.csv"), index=False)
cd.to_csv(os.path.join(REP, "prior_cut_match.csv"), index=False)
