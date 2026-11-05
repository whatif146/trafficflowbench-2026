"""Build Task-2 variant submissions from the P4 base for public-LB probing.

Usage:
  python work/pipeline/build_queue_variants.py
Builds work/submissions/submission_<tag>.csv/.zip for each variant in VARIANTS,
patching only the queue task into the v4 base (Task 1 = LightGBM, Task 4 = NNLS).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
WORK = os.path.join(ROOT, "work")
BASE = os.path.join(WORK, "submissions", "submission_v4.csv")
OUT = os.path.join(WORK, "task2_seq", "out")
SUBS = os.path.join(WORK, "submissions")

VARIANTS = {
    "q_w30ong40fb1_all": ("queue_w30ong40fb1_all_validation.csv", "queue_w30ong40fb1_all_private.csv"),
    "q_onfb1_all": ("queue_onfb1_all_validation.csv", "queue_onfb1_all_private.csv"),
}


def combine(paths, out):
    with open(out, "wb") as w:
        first = True
        for p in paths:
            with open(p, "rb") as r:
                if first:
                    shutil.copyfileobj(r, w, 1 << 22)
                    first = False
                else:
                    r.readline()
                    shutil.copyfileobj(r, w, 1 << 22)


def main():
    for tag, (v, p) in VARIANTS.items():
        allcsv = os.path.join(OUT, f"{tag}_all.csv")
        if not (os.path.exists(os.path.join(OUT, v)) and os.path.exists(os.path.join(OUT, p))):
            print("missing variant", tag)
            continue
        combine([os.path.join(OUT, v), os.path.join(OUT, p)], allcsv)
        outcsv = os.path.join(SUBS, f"submission_{tag}.csv")
        subprocess.run([sys.executable, os.path.join(WORK, "pipeline", "patch_task.py"),
                        "--base", BASE, "--task", "queue", "--source", allcsv,
                        "--output", outcsv], check=True)
        zpath = os.path.join(SUBS, f"submission_{tag}.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            z.write(outcsv, arcname="submission.csv")
        print(f"{tag}: {os.path.getsize(zpath)/1e6:.1f} MB -> {zpath}", flush=True)


if __name__ == "__main__":
    main()
