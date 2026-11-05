"""Assemble the final submission package (code + report + predictions).

Creates `final/` with a self-contained copy of the code that produces the
submitted predictions, the experiment log, task NOTES, the report, the final
submission archive and the frozen per-row components used by verify_final.py.

Usage: python work/pipeline/make_package.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from copy_code import copy_code, copy_full_dirs

ROOT = r"C:\Users\lenovo\Desktop\kaggle"
WORK = os.path.join(ROOT, "work")
FINAL = os.path.join(ROOT, "final")
# final submission archive (current best public-LB candidate, in priority order)
ZIP_PRIORITY = ("submission_q_unet2_w05.zip", "submission_q_unet2_w06.zip",
                "submission_q_w30ong40fb1_all.zip", "submission_q_w30ong40fb1.zip",
                "submission_q_onfb1.zip", "submission_p4.zip", "submission_v4.zip")
PACKAGE_DATA = ("state_gbm_unet_validation.csv.gz", "state_gbm_unet_private.csv.gz",
                "queue_w30ong40fb1_all_validation.csv", "queue_w30ong40fb1_all_private.csv")


def main():
    if os.path.isdir(FINAL):
        shutil.rmtree(FINAL)
    for d in ("work", "report", "submissions"):
        os.makedirs(os.path.join(FINAL, d), exist_ok=True)
    n_code = copy_code(WORK, FINAL)
    n_full = copy_full_dirs(WORK, FINAL)
    print(f"copied {n_code} code files + {n_full} model/prediction files")
    # docs at the package root
    for rel, dstname in [("EXPERIMENTS.md", "EXPERIMENTS.md"),
                         ("submissions/FINAL_CHOICE.md", "FINAL_CHOICE.md"),
                         ("submissions/FINAL_STATE.md", "FINAL_STATE.md"),
                         ("submissions/EMAIL_DRAFT.txt", "EMAIL_DRAFT.txt"),
                         ("report/MANIFEST.md", "MANIFEST.md"),
                         ("report/FINAL_SUMMARY_ZH.md", "FINAL_SUMMARY_ZH.md"),
                         ("report/REPORT_DRAFT.md", "report/REPORT_DRAFT.md")]:
        src = os.path.join(WORK, rel)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(FINAL, dstname))
    # final submission archive + frozen components
    for name in ZIP_PRIORITY:
        src = os.path.join(WORK, "submissions", name)
        if not os.path.exists(src):
            continue
        shutil.copy2(src, os.path.join(FINAL, "submissions", "submission_final.zip"))
        print("final submission archive:", name)
        final_csv = os.path.join(WORK, "submissions", name.replace(".zip", ".csv"))
        pkgdata = os.path.join(FINAL, "submissions", "package_data")
        os.makedirs(pkgdata, exist_ok=True)
        for fn in PACKAGE_DATA:
            s = os.path.join(WORK, "submissions", "package_data", fn)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(pkgdata, fn))
        if os.path.exists(final_csv):
            subprocess.run([sys.executable,
                            os.path.join(WORK, "pipeline", "make_components_npz.py"),
                            "--submission", final_csv,
                            "--out", os.path.join(pkgdata, "components.npz")], check=True)
        print("package_data:", sorted(os.listdir(pkgdata)))
        break
    # top-level README
    readme = os.path.join(WORK, "report", "PACKAGE_README.md")
    if os.path.exists(readme):
        shutil.copy2(readme, os.path.join(FINAL, "README.md"))
    # typeset report + figures
    for name, dest in [("TrafficFlowBench_Report.pdf", "report/TrafficFlowBench_Report.pdf"),
                       ("lb_progress.png", "report/lb_progress.png"),
                       ("threshold_curve.png", "report/threshold_curve.png")]:
        src = os.path.join(WORK, "report", name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(FINAL, dest))
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(FINAL) for f in fs)
    print(f"package assembled: {FINAL} ({size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
