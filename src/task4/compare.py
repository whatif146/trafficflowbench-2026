"""Compare the candidate allocations to each other (validation) and their
destination-attraction distributions, so an LB delta can be interpreted."""
import os
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import cand
import odme

REP = odme.REPORTS
TAGS = {
    "baseline": "candidates/path_flow_baseline_validation.csv",
    "pool": "candidates/path_flow_pool_validation.csv",
    "autoscale": "candidates/path_flow_autoscale_validation.csv",
    "margcorr": "candidates/path_flow_margcorr_validation.csv",
    "denoise_margcorr": "candidates/path_flow_denoise_margcorr_validation.csv",
    "scale0p5": "candidates/path_flow_scale0p5_validation.csv",
    "ent_own": "candidates/path_flow_ent_own_validation.csv",
    "ent_pool": "candidates/path_flow_ent_pool_validation.csv",
    "ent_shrink0.5": "candidates/path_flow_ent_shrink0.5_validation.csv",
}

subs = {}
for tag, rel in TAGS.items():
    d = pd.read_csv(os.path.join(odme.WORK, rel), dtype={"path_id": str, "panel": str})
    subs[tag] = d

rows = []
for panel in odme.PANELS:
    sd = odme.split_arrays(panel, "validation")
    oi, di = cand.path_meta(panel)
    F = {}
    for tag, d in subs.items():
        g = d[d.panel == panel].set_index("path_id").reindex(sd["path_ids"])
        F[tag] = np.nan_to_num(g.path_flow.to_numpy(float))
    base = F["baseline"]
    for tag, f in F.items():
        X = np.bincount(di, weights=f, minlength=int(di.max()) + 1).astype(float)
        a = X / max(X.sum(), 1e-12)
        Xb = np.bincount(di, weights=base, minlength=int(di.max()) + 1).astype(float)
        ab = Xb / max(Xb.sum(), 1e-12)
        rows.append(dict(panel=panel, tag=tag,
                         sum_f=float(f.sum()),
                         relL1_to_baseline=float(np.abs(f - base).sum() / max(base.sum(), 1e-9)),
                         relL1_to_b=float(np.abs(f - sd["b"]).sum() / max(sd["b"].sum(), 1e-9)),
                         attr_L1_vs_baseline=float(np.abs(a - ab).sum()),
                         s_link=odme.s_link_panel(f, sd)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(REP, "candidate_allocation_compare.csv"), index=False)
piv = df.pivot_table(index="tag", columns="panel",
                     values=["relL1_to_baseline", "attr_L1_vs_baseline"], aggfunc="mean")
print(piv.round(4).to_string())
print("\nmean by tag:")
print(df.groupby("tag")[["sum_f", "relL1_to_baseline", "relL1_to_b", "attr_L1_vs_baseline",
                         "s_link"]].mean().round(4).to_string())
