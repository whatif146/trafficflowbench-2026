"""Finalise the three Task-4 LB-probe candidates with the Lead's naming, verify
template coverage and S_link with the official scorer, and build a diagnostics table.
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\task4")
import numpy as np
import pandas as pd

import odme

OUT = os.path.join(odme.WORK, "out")
CAND = os.path.join(odme.WORK, "candidates")
REP = odme.REPORTS
os.makedirs(OUT, exist_ok=True)

# candidate letter -> method tag in candidates/
MAPPING = {
    "A": "pool",              # pooled-prior denoising inside NNLS (low risk)
    "B": "denoise_margcorr",  # gravity-structured recompose + cut projection (aggressive)
    "C": "autoscale",         # prior rescaled to the cut level (scale axis)
}

KEY = os.path.join(odme.REL, "submission_key.csv")


def key_odme_rows():
    key = pd.read_csv(KEY, dtype=str)
    k = key[key.task == "odme"][["panel", "departure_time", "path_id"]]
    return k.reset_index(drop=True)


def main():
    kr = key_odme_rows()
    print("key odme rows:", len(kr))
    rows = []
    for letter, tag in MAPPING.items():
        src = os.path.join(CAND, f"path_flow_{tag}_validation_plus_private.csv")
        dst = os.path.join(OUT, f"cand_{letter}_validation_plus_private.csv")
        sub = pd.read_csv(src, dtype={"panel": str, "path_id": str, "departure_time": str})
        # exact key coverage check
        m = sub.merge(kr, on=["panel", "departure_time", "path_id"], how="right", indicator=True)
        miss = int((m._merge == "right_only").sum())
        extra = int((m._merge == "left_only").sum())
        f = sub.path_flow.to_numpy(float)
        print(f"cand_{letter} ({tag}): rows={len(sub)} missing_key={miss} extra_key={extra} "
              f"finite={np.isfinite(f).all()} min={f.min():.3f} dups="
              f"{int(sub.duplicated(['panel','departure_time','path_id']).sum())}")
        shutil.copyfile(src, dst)
        for split in ("validation", "private"):
            src_s = os.path.join(CAND, f"path_flow_{tag}_{split}.csv")
            dst_s = os.path.join(OUT, f"cand_{letter}_{split}.csv")
            shutil.copyfile(src_s, dst_s)
    # diagnostics per panel for each candidate
    for letter, tag in MAPPING.items():
        sub = pd.read_csv(os.path.join(CAND, f"path_flow_{tag}_validation.csv"),
                          dtype={"path_id": str, "panel": str})
        for p, g in sub.groupby("panel"):
            sd = odme.split_arrays(p, "validation")
            f = g.set_index("path_id").reindex(sd["path_ids"]).path_flow.to_numpy(float)
            rows.append(dict(candidate=letter, tag=tag, panel=p,
                             sum_f=float(f.sum()), sum_b=float(sd["b"].sum()),
                             s_link=odme.s_link_panel(f, sd),
                             dhat=float(np.abs(f - sd["b"]).sum())))
    df = pd.DataFrame(rows)
    df["dhat_over_sumb"] = df.dhat / df.sum_b
    df["sumf_over_sumb"] = df.sum_f / df.sum_b
    df.to_csv(os.path.join(REP, "final_candidates_diagnostics.csv"), index=False)
    piv = df.pivot_table(index="panel", columns="candidate",
                         values=["sum_f", "dhat_over_sumb"], aggfunc="mean")
    print("\nper-panel sum_f / dhat_over_sumb (validation):")
    print(piv.round(3).to_string())
    print("\nfamily-mean S_link by candidate:")
    for letter in MAPPING:
        g = df[df.candidate == letter].set_index("panel")
        fam = [float(g.loc[ps, "s_link"].mean()) for ps in odme.FAMILIES.values()]
        print(f"  cand_{letter} ({MAPPING[letter]}): {np.mean(fam):.6f}")
    print("\nmean sum_f / mean sum_b by candidate:",
          {letter: round(df[df.candidate == letter].sum_f.mean(), 0) for letter in MAPPING})


if __name__ == "__main__":
    main()
