"""Post-probe finalization: pick the best candidate, refresh docs and package.

Run after the 2026-11-05 probe batch:
  python work/pipeline/finalize_after_probes.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = r"C:\Users\lenovo\Desktop\kaggle"
WORK = os.path.join(ROOT, "work")
SUBS = os.path.join(WORK, "submissions")
RESULTS = os.path.join(SUBS, "probe_results.json")
HEDGE = [("q_w30ong40fb1", 57947511), ("q_w30ong40fb1_all", 57967779)]
EXCLUDE = {"cal_zeroqueue", "cal_zerostate", "cal_allones", "cal_statezero",
           "cal_zeroodme", "cal_zerostate", "q_on040fb1"}
STRUCTURAL = ("fb2", "min2", "_all", "ong25")


def main():
    res = json.load(open(RESULTS, encoding="utf-8"))
    scores = {t: v.get("public_score") for t, v in res.items()
              if t not in EXCLUDE and v.get("public_score") is not None}
    if not scores:
        raise SystemExit("no probe scores yet")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    print("ranking:")
    for t, sc in ranked:
        print(f"  {t:20s} {sc:.5f}")
    best, best_sc = ranked[0]
    # if a structural/2-seed candidate is within 0.0005, prefer it
    near = [t for t, sc in ranked if best_sc - sc <= 5e-4 and any(k in t for k in STRUCTURAL)]
    if near and near[0] != best:
        print(f"near-tie preference: {near[0]} over {best}")
        best = near[0]
        best_sc = scores[best]
    print(f"BEST = {best} ({best_sc:.5f}) ref {res[best]['ref']}")

    # FINAL_CHOICE update
    fc = os.path.join(SUBS, "FINAL_CHOICE.md")
    c = open(fc, encoding="utf-8").read()
    lines = [f"| {t} | {sc:.5f} | {res[t]['ref']} |" for t, sc in ranked]
    c = c.rstrip() + "\n\n## 2026-11-05 probe results\n\n| candidate | public LB | ref |\n|---|---:|---:|\n" + \
        "\n".join(lines) + f"\n\n**Selected primary: {best} (ref {res[best]['ref']}, {best_sc:.5f}).**\n" + \
        f"Backup (GBM-only hedge): q_w30ong40fb1 ref 57947511 (0.83734).\n"
    open(fc, "w", encoding="utf-8").write(c)
    print("FINAL_CHOICE updated")

    # package priority
    mp = os.path.join(WORK, "pipeline", "make_package.py")
    s = open(mp, encoding="utf-8").read()
    s2 = re.sub(r'ZIP_PRIORITY = \("[^"]+",', f'ZIP_PRIORITY = ("submission_{best}.zip",', s, count=1)
    if s2 == s:
        raise SystemExit("could not update ZIP_PRIORITY")
    open(mp, "w", encoding="utf-8").write(s2)
    print(f"make_package priority -> submission_{best}.zip")

    # rebuild package and report
    subprocess.run([sys.executable, os.path.join(WORK, "pipeline", "make_package.py")], check=True)
    subprocess.run([sys.executable, os.path.join(WORK, "report", "make_pdf.py"),
                    "--out", os.path.join(ROOT, "final", "report", "TrafficFlowBench_Report.pdf")], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "final", "work", "pipeline", "verify_final.py")], check=True)
    print("\nSELECT on Kaggle:")
    print(f"  primary  ref {res[best]['ref']}  ({best}, {best_sc:.5f})")
    print(f"  backup   ref {HEDGE[0][1]}  (q_w30ong40fb1, 0.83734)")
    print("\nBoth must be selected before 2026-11-07 06:55 UTC.")


if __name__ == "__main__":
    main()
