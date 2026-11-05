"""Spare candidates: entropy projection of a gravity-blended prior.

prior_k = b^(1-k) * m^k, m = gravity log-linear fit of the pooled prior (denoised),
rescaled to sum(b). k=0 is the raw own prior (cand_b), k=1 is the fully smoothed
prior. k=0.35 / 0.7 bracket the "how much of the prior's per-OD structure is real"
question without changing the total-flow scale.
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

CAND = os.path.join(odme.WORK, "candidates")
OUT = os.path.join(odme.WORK, "out")
KS = {"ent_blend35": 0.35, "ent_blend70": 0.70}


def main():
    rows = []
    for tag, k in KS.items():
        frames = []
        for split in ("validation", "private"):
            t0 = time.time()
            for p in odme.PANELS:
                sd = odme.split_arrays(p, split)
                oi, di = cand.path_meta(p)
                bpool = np.mean([odme.split_arrays(p, s)["b"] for s in odme.SPLITS], axis=0)
                m = cand.denoise_prior(bpool, oi, di)
                prior = np.exp((1 - k) * np.log(np.clip(sd["b"], 1e-9, None)) +
                               k * np.log(np.clip(m, 1e-9, None)))
                prior *= float(sd["b"].sum() / prior.sum())
                f, _ = entropy.entropy_project(
                    prior, oi, di, sd["A_meas"], sd["counts"],
                    seg_rows=entropy.segment_of_rows(sd["A"], oi, sd["measured"]))
                rows.append(dict(tag=tag, split=split, panel=p, sum_f=float(f.sum()),
                                 sum_b=float(sd["b"].sum()), s_link=odme.s_link_panel(f, sd),
                                 dhat=float(np.abs(f - sd["b"]).sum())))
                tpl = odme.load_template(p, split)
                out = tpl[["panel", "departure_time", "path_id", "origin_zone",
                           "destination_zone"]].copy()
                out["path_flow"] = np.clip(f, 0.0, None)
                frames.append(out)
            print(f"[{tag}] {split} {time.time()-t0:.1f}s", flush=True)
        sub = pd.concat(frames, ignore_index=True)
        assert np.isfinite(sub.path_flow.to_numpy()).all() and (sub.path_flow.to_numpy() >= 0).all()
        sub.to_csv(os.path.join(CAND, f"path_flow_{tag}_validation_plus_private.csv"), index=False)
        for split in ("validation", "private"):
            g = sub[sub.departure_time == "SYN_PM_" + split.upper()]
            g.to_csv(os.path.join(OUT, f"cand_x_{tag}_{split}.csv"), index=False)
        comb = pd.concat([sub[sub.departure_time == "SYN_PM_VALIDATION"],
                          sub[sub.departure_time == "SYN_PM_PRIVATE"]], ignore_index=True)
        comb.to_csv(os.path.join(OUT, f"cand_x_{tag}_validation_plus_private.csv"), index=False)
        print(f"[{tag}] wrote {len(comb)} rows", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(["tag", "split"])[["sum_f", "sum_b", "dhat", "s_link"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
