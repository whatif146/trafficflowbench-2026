"""Measure the Task 3 post-processing trade-off on the train holdout.

For a real candidate Task 1 train submission (or a synthetic corruption of the
observed field) this reports, on one panel:

  * Task 1 S_state before/after (the exact score_task1 regime aggregation);
  * proxy S_FD / S_LWR / S_physics before/after under the calibrated frozen-flux
    proxy (rhs = projection onto released observations) at several calibration
    residual levels, so a gain that only exists at one calibration is visible;
  * the competition-weighted trade-off 0.35*dS_state + 0.15*dS_physics.

One panel at a time (RAM is tight).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "libs"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task3"))
import local_eval as le
import adjust as ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="D7_I10_E")
    ap.add_argument("--state-submission", default=None,
                    help="candidate Task 1 train submission CSV (else a synthetic corruption)")
    ap.add_argument("--candidate", default="noisy:6:0.15",
                    help="built-in corruption when --state-submission is absent")
    ap.add_argument("--eval-windows", default="1", help="flux smoothing windows for evaluation")
    ap.add_argument("--eval-jitters", default="0.0,0.06,0.12",
                    help="calibration residual levels (0.06 reproduces the official anchor)")
    ap.add_argument("--adjust-flux", default="projected")
    ap.add_argument("--adjust-window", type=int, default=1)
    ap.add_argument("--grid", default="0:0,0.5:20,1:40,1:100,1:200",
                    help="comma list of strength:iterations operating points")
    ap.add_argument("--fd-strength", type=float, default=0.0)
    ap.add_argument("--max-rel-change", type=float, default=0.35)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    panel = args.panel
    if not le._panel_cache_ready(panel, "train"):
        raise SystemExit(f"{panel}: train cache not ready")
    pdata = le.load_panel(panel, "train")
    print(f"[{panel}] T={pdata['T']} L={pdata['L']}", flush=True)
    if args.state_submission:
        rows = le._read_submission_rows(args.state_submission, [panel], "train")
        cs, cf = le._rows_to_arrays(rows, pdata)
        cand_name = os.path.basename(args.state_submission)
    else:
        cs, cf = le.build_candidate(pdata, args.candidate)
        cand_name = args.candidate
    base_scores = ad.task1_state_scores(pdata, cs, cf)
    windows = [int(w) for w in args.eval_windows.split(",") if w.strip()]
    jitters = [float(j) for j in args.eval_jitters.split(",") if j.strip()]
    variants = {}
    for w in windows:
        for j in jitters:
            name = f"proj{w}j{j:g}"
            variants[name] = le.build_frozen_flux(pdata, "projected", w, flux_jitter=j)
    print(f"[{panel}] candidate={cand_name} base S_state={base_scores['macro']:.4f}", flush=True)
    print("  base rows:", [(r["regime"], round(r["S_state"], 4), round(r["rmse_speed"], 2),
                            round(r["rmse_flow"], 1)) for r in base_scores["rows"]], flush=True)

    def score(cs_, cf_, name):
        res = le.evaluate({panel: pdata}, lambda pd_: (cs_, cf_), {panel: variants[name]})
        return res[res.level == "panel"].iloc[0]

    base_phys = {name: score(cs, cf, name) for name in variants}
    for name, r in base_phys.items():
        print(f"  base {name:14s} S_FD={r.S_FD:.4f} S_LWR={r.S_LWR:.4f} S_phys={r.S_physics:.4f}",
              flush=True)
    grid = []
    for g in args.grid.split(","):
        s, it = g.split(":")
        grid.append((float(s), int(it)))
    rows = []
    for strength, iters in grid:
        _, cf2, diag = ad.adjust_panel(
            pdata, cs, cf, strength=strength, iterations=iters,
            flux_mode=args.adjust_flux, flux_window=args.adjust_window,
            fd_strength=args.fd_strength, max_rel_change=args.max_rel_change,
            verbose=True)
        st = ad.task1_state_scores(pdata, cs, cf2)
        row = {"panel": panel, "candidate": cand_name, "strength": strength,
               "iterations": iters, "S_state": st["macro"],
               "dS_state": st["macro"] - base_scores["macro"],
               "e_before": diag["e_before"], "e_after": diag["e_after"]}
        for name in variants:
            r = score(cs, cf2, name)
            row[f"S_FD_{name}"] = r.S_FD
            row[f"S_LWR_{name}"] = r.S_LWR
            row[f"S_phys_{name}"] = r.S_physics
            row[f"dS_phys_{name}"] = r.S_physics - base_phys[name].S_physics
        cal = [n for n in variants if n.endswith("j0.06")]
        cal_name = cal[0] if cal else list(variants)[0]
        trade = 0.35 * row["dS_state"] + 0.15 * row[f"dS_phys_{cal_name}"]
        row["tradeoff_calibrated"] = trade
        rows.append(row)
        print(f"  strength={strength} iters={iters}: S_state={row['S_state']:.4f} "
              f"(dS={row['dS_state']:+.4f}) " +
              " ".join(f"{n}:LWR{row[f'S_LWR_{n}']:.3f}(dSphys{row[f'dS_phys_{n}']:+.4f})"
                       for n in variants) +
              f" tradeoff={trade:+.5f}", flush=True)
    df = pd.DataFrame(rows)
    out = args.output or os.path.join(le.WORK3, "reports", f"adjust_tradeoff_{panel}.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
