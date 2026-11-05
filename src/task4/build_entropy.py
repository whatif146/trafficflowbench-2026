"""Build maximum-entropy projection candidates (own / pooled / shrunk prior)."""
import os
import sys
import time

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import entropy
import odme

CAND = os.path.join(odme.WORK, "candidates")
os.makedirs(CAND, exist_ok=True)

MODES = {"ent_own": "own", "ent_pool": "pool", "ent_shrink0.5": "shrink:0.5"}
SPLITS = ["validation", "private"]


def main():
    rows = []
    for tag, mode in MODES.items():
        frames = []
        for split in SPLITS:
            t0 = time.time()
            for p in odme.PANELS:
                f, sd = entropy.make_entropy(p, split, mode)
                sl = odme.s_link_panel(f, sd)
                rows.append(dict(tag=tag, split=split, panel=p, n=len(f),
                                 sum_f=float(f.sum()), sum_b=float(sd["b"].sum()),
                                 s_link=sl,
                                 dhat=float(np.abs(f - sd["b"]).sum()),
                                 nnz=int((f > 0).sum())))
                tpl = odme.load_template(p, split)
                out = tpl[["panel", "departure_time", "path_id", "origin_zone",
                           "destination_zone"]].copy()
                out["path_flow"] = np.clip(f, 0.0, None)
                frames.append(out)
            print(f"[{tag}] {split} {time.time()-t0:.1f}s", flush=True)
        sub = pd.concat(frames, ignore_index=True)
        assert np.isfinite(sub.path_flow.to_numpy()).all() and (sub.path_flow.to_numpy() >= 0).all()
        for split in SPLITS:
            g = sub[sub.departure_time == ("SYN_PM_" + split.upper())]
            g.to_csv(os.path.join(CAND, f"path_flow_{tag}_{split}.csv"), index=False)
        sub.to_csv(os.path.join(CAND, f"path_flow_{tag}_validation_plus_private.csv"), index=False)
        print(f"[{tag}] wrote {len(sub)} rows", flush=True)
    df = pd.DataFrame(rows)
    df["dhat_over_sumb"] = df.dhat / df.sum_b
    fams = {}
    for tag, g in df.groupby("tag"):
        per = g[g.split == "validation"].set_index("panel").s_link
        fams[tag] = float(np.mean([per.loc[ps].mean() for ps in odme.FAMILIES.values()]))
    print(df.groupby(["tag", "split"])[["sum_f", "sum_b", "dhat_over_sumb", "s_link"]].mean().round(4).to_string())
    print("family-mean S_link (validation):", {k: round(v, 6) for k, v in fams.items()})
    df.to_csv(os.path.join(odme.REPORTS, "entropy_diagnostics.csv"), index=False)


if __name__ == "__main__":
    main()
