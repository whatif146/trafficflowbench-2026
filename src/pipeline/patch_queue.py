"""Generic builder: patch an arbitrary Task-2 queue layer into the P4 base upload.

Usage:
  python work/pipeline/patch_queue.py --val work/task2_seq/out/queue_v9_validation.csv \
      --priv work/task2_seq/out/queue_v9_private.csv --tag q_v9

Concatenates the validation+private queue files, patches the queue rows of
work/submissions/submission_v4.csv (Task-1 LightGBM + Task-2 v3 + Task-4 NNLS
base) by submission_key, and writes work/submissions/submission_<tag>.{csv,zip}.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
WORK = os.path.join(ROOT, "work")
SUBS = os.path.join(WORK, "submissions")
BASE = os.path.join(SUBS, "submission_v4.csv")


def combine(paths, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--priv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default=BASE,
                    help="base merged upload to patch (default: submission_v4.csv)")
    args = ap.parse_args()
    allcsv = os.path.join(WORK, "task2_seq", "out", f"{args.tag}_all.csv")
    combine([args.val, args.priv], allcsv)
    outcsv = os.path.join(SUBS, f"submission_{args.tag}.csv")
    subprocess.run([sys.executable, os.path.join(WORK, "pipeline", "patch_task.py"),
                    "--base", args.base, "--task", "queue", "--source", allcsv,
                    "--output", outcsv], check=True)
    zpath = os.path.join(SUBS, f"submission_{args.tag}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(outcsv, arcname="submission.csv")
    print(f"{args.tag}: {os.path.getsize(zpath)/1e6:.1f} MB -> {zpath}")


if __name__ == "__main__":
    main()
