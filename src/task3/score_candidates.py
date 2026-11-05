"""Score several Task 1 candidates on one train panel with the calibrated proxy.

Example:
    python score_candidates.py --panel D7_I10_E \
        --state-submission work/submissions/state_train_D7_I10_E.csv \
        --candidate observed,historical,noisy:5:0.10 \
        --flux projected --flux-window 1 --flux-jitter 0.06
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "libs"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task3"))
import local_eval as le
import adjust as ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="D7_I10_E")
    ap.add_argument("--split", default="train", choices=["train"])
    ap.add_argument("--state-submission", default=None, help="Task 1 candidate CSV to score")
    ap.add_argument("--candidate", default="observed,historical,noisy:5:0.10",
                    help="comma list of built-in candidates")
    ap.add_argument("--flux", default="projected")
    ap.add_argument("--flux-window", type=int, default=1)
    ap.add_argument("--flux-jitter", type=float, default=0.06)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    panel = args.panel
    pdata = le.load_panel(panel, "train")
    print(f"[{panel}] T={pdata['T']} L={pdata['L']}", flush=True)
    flux = le.build_frozen_flux(pdata, args.flux, args.flux_window, flux_jitter=args.flux_jitter)
    specs = {}
    for c in [x.strip() for x in args.candidate.split(",") if x.strip()]:
        specs[c] = (le.build_candidate(pdata, c), c)
    if args.state_submission:
        rows = le._read_submission_rows(args.state_submission, [panel], "train")
        specs[os.path.basename(args.state_submission)] = (le._rows_to_arrays(rows, pdata),
                                                          os.path.basename(args.state_submission))
    rows_out = []
    for name, ((cs, cf), label) in specs.items():
        res = le.evaluate({panel: pdata}, lambda pd_, cs=cs, cf=cf: (cs, cf), {panel: flux})
        r = res[res.level == "panel"].iloc[0]
        st = ad.task1_state_scores(pdata, cs, cf) if args.split == "train" else {"macro": np.nan}
        n_missing = int((~np.isfinite(cs)).sum())
        rows_out.append({"panel": panel, "candidate": label, "S_FD": r.S_FD, "S_LWR": r.S_LWR,
                         "S_physics": r.S_physics, "S_state": st["macro"],
                         "n_target_missing": n_missing})
        print(f"  {label:28s} S_FD={r.S_FD:.4f} S_LWR={r.S_LWR:.4f} "
              f"S_physics={r.S_physics:.4f}  S_state={st['macro']:.4f}", flush=True)
    import pandas as pd
    out = args.output or os.path.join(le.WORK3, "reports", f"candidates_{panel}.csv")
    pd.DataFrame(rows_out).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
