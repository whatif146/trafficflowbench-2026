"""Cut-increment structure: does the prior's zone-level net demand track the truth's?

dC_k = C_k - C_{k-1} = E_k - X_k  (origin minus destination flow at zone k)
Truth: dC from released counts (exact, since counts are A f*).
Prior: dC from A b (pooled over 3 splits, or per split).
If they correlate strongly with an amplitude ratio != 1, the prior's marginals are
informative but mis-scaled, and we can correct the prior's marginals to match.
"""
import sys, os
sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np, pandas as pd
import odme

pd.set_option("display.width", 250)
REP = odme.REPORTS


def geom(panel):
    paths = odme.load_paths(panel)
    paths["oi"] = paths.origin_zone.str[1:].astype(int)
    paths["di"] = paths.destination_zone.str[1:].astype(int)
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
    return paths, seg_of, seglen


rows = []
for panel in odme.PANELS:
    paths, seg_of, seglen = geom(panel)
    bs = {}
    cprof = {}
    bprof = {}
    for split in ("train", "validation", "private"):
        sd = odme.split_arrays(panel, split)
        bs[split] = sd["b"]
        segs = np.array([seg_of[l] for l in sd["measured_link_ids"]])
        cprof[split] = pd.Series(sd["counts"]).groupby(segs).mean()
        bprof[split] = pd.Series(sd["A_meas"] @ sd["b"]).groupby(segs).mean()
    bp = np.mean([bs[s] for s in bs], axis=0)
    sd = odme.split_arrays(panel, "validation")
    segs = np.array([seg_of[l] for l in sd["measured_link_ids"]])
    bprof["pool"] = pd.Series(sd["A_meas"] @ bp).groupby(segs).mean()
    # truth increments from validation counts
    idx = sorted(set(cprof["validation"].index) & set(seg_of.values()))
    C = cprof["validation"].reindex(idx).to_numpy(float)
    dC = np.diff(C)
    # segments are consecutive integers 0..M-1 (idx array)
    out = dict(panel=panel, nseg=len(idx), seg0=idx[0], seg_last=idx[-1],
               C_mean=float(C.mean()), C_std_rel=float(C.std() / C.mean()),
               min_total=float(C[0] + np.clip(dC, 0, None).sum()))
    for name in ["validation", "pool"]:
        B = bprof[name].reindex(idx).to_numpy(float)
        dB = np.diff(B)
        # best multiplicative scale on the cut LEVEL profile
        s = float((C @ B) / max(B @ B, 1e-12))
        # regression of dC on dB (through origin and with intercept)
        sl = float((dC @ dB) / max(dB @ dB, 1e-12))
        cc = float(np.corrcoef(dC, dB)[0, 1]) if np.std(dB) > 0 else np.nan
        # ratio of amplitudes
        amp = float(np.std(dC) / max(np.std(dB), 1e-12))
        out[f"corr_d_{name}"] = cc
        out[f"slope_d_{name}"] = sl
        out[f"amp_ratio_{name}"] = amp
        out[f"scale_{name}"] = s
        out[f"level_relL1_{name}"] = float(np.abs(C - s * B).sum() / C.sum())
        out[f"sumB_over_C_{name}"] = float(B.sum() / C.sum())
    # also: raw prior row/col sums relative structure
    oi = paths.set_index("path_id").reindex(sd["path_ids"]).oi.to_numpy()
    di = paths.set_index("path_id").reindex(sd["path_ids"]).di.to_numpy()
    E = np.bincount(oi, weights=bp); X = np.bincount(di, weights=bp)
    out["sum_b_pool"] = float(bp.sum())
    Es = pd.Series(E, index=np.arange(len(E))); Xs = pd.Series(X, index=np.arange(len(X)))
    j = Es.index.intersection(Xs.index)
    out["E_over_X_corr"] = float(np.corrcoef(Es.loc[j], Xs.loc[j])[0, 1])
    rows.append(out)

df = pd.DataFrame(rows)
print(df.round(4).to_string(index=False))
print("\nMean by prior type:")
for name in ["validation", "pool"]:
    print(name, " corr_d=%.3f slope=%.3f amp=%.3f level_relL1=%.3f" % (
        df[f"corr_d_{name}"].mean(), df[f"slope_d_{name}"].mean(),
        df[f"amp_ratio_{name}"].mean(), df[f"level_relL1_{name}"].mean()))
print("\nmin feasible total / prior total (pool):")
print(pd.DataFrame({"panel": df.panel, "min_total": df.min_total.round(0),
                    "sum_b_pool": df.sum_b_pool.round(0),
                    "ratio": (df.sum_b_pool / df.min_total).round(2)}).to_string(index=False))
df.to_csv(os.path.join(REP, "cut_increment_diag.csv"), index=False)
