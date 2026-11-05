"""Build a batch of Task-4 candidate submissions + a diagnostics table.

For every panel and split, run each method, save one CSV per (method, split)
plus the combined validation+private file, and record S_link / sum / Dhat.
"""
import os
import sys
import time

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import cand
import odme

OUT = os.path.join(odme.WORK, "candidates")
REP = odme.REPORTS
os.makedirs(OUT, exist_ok=True)

METHODS = ["baseline", "autoscale", "margcorr", "denoise_margcorr", "pool", "scale:0.5"]
SPLITS = ["validation", "private"]


def save_split(frames, split, method, tag):
    sub = pd.concat(frames, ignore_index=True)
    assert np.isfinite(sub.path_flow.to_numpy()).all()
    assert (sub.path_flow.to_numpy() >= 0).all()
    path = os.path.join(OUT, f"path_flow_{tag}_{split}.csv")
    sub.to_csv(path, index=False)
    return path, sub


def main():
    rows = []
    made = {}
    for method in METHODS:
        tag = method.replace(":", "").replace(".", "p")
        for split in SPLITS:
            frames = []
            t0 = time.time()
            for p in odme.PANELS:
                f, sd = cand.make(p, split, method)
                sl = odme.s_link_panel(f, sd)
                dhat = float(np.sum(np.abs(f - sd["b"])))
                denom = float(sd["b"].sum())
                rows.append(dict(method=tag, split=split, panel=p,
                                 n=len(f), sum_f=float(f.sum()), s_link=sl,
                                 dhat=dhat, dhat_over_sumb=dhat / max(denom, 1e-9),
                                 sumb=denom, nnz=int((f > 0).sum())))
                tpl = odme.load_template(p, split)
                out = tpl[["panel", "departure_time", "path_id", "origin_zone",
                           "destination_zone"]].copy()
                out["path_flow"] = np.clip(f, 0.0, None)
                frames.append(out)
            made[(method, split)] = frames
            print(f"[batch] {tag:16s} {split:10s} done in {time.time()-t0:.1f}s", flush=True)
    # overall S_link per method/split via family means
    df = pd.DataFrame(rows)
    for (method, split), g in df.groupby(["method", "split"]):
        fam = []
        for fam_id, ps in odme.FAMILIES.items():
            fam.append(float(g.set_index("panel").loc[ps, "s_link"].mean()))
        df.loc[(df.method == method) & (df.split == split), "fam_mean_s_link"] = np.mean(fam)
    df.to_csv(os.path.join(REP, "candidate_diagnostics.csv"), index=False)
    # write files and combined pairs
    for method in METHODS:
        tag = method.replace(":", "").replace(".", "p")
        frames_all = []
        for split in SPLITS:
            path, sub = save_split(made[(method, split)], split, method, tag)
            frames_all.append(sub)
            print(f"[batch] wrote {path} ({len(sub)} rows)", flush=True)
        comb = pd.concat(frames_all, ignore_index=True)
        cpath = os.path.join(OUT, f"path_flow_{tag}_validation_plus_private.csv")
        comb.to_csv(cpath, index=False)
        print(f"[batch] wrote {cpath} ({len(comb)} rows)", flush=True)
    print("\n=== summary ===")
    piv = df.pivot_table(index="method", columns="split",
                         values=["fam_mean_s_link", "sum_f", "dhat_over_sumb"],
                         aggfunc="mean")
    print(piv.round(5).to_string())


if __name__ == "__main__":
    main()
