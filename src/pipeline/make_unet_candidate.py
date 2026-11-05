"""Build a Task-1 candidate submission from two state CSVs and patch it into the
current best upload (queue + ODME kept at the current best layer).

Usage:
  python work/pipeline/make_unet_candidate.py \
      --val work/task1/unet/state_unet_validation.csv \
      --priv work/task1/unet/state_unet_private.csv --tag q_unet_blend

Steps: pack the state CSVs into dense per-panel arrays, patch only the Task-1
values of work/submissions/submission_q_w30ong40fb1_all.csv, validate the 6.98M
row upload and zip it as work/submissions/submission_<tag>.zip.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
WORK = os.path.join(ROOT, "work")
SUBS = os.path.join(WORK, "submissions")
BASE = os.path.join(SUBS, "submission_q_w30ong40fb1_all.csv")


def run(cmd):
    print(">>>", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit(f"step failed ({rc}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--priv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default=BASE,
                    help="base merged upload whose non-state layers are kept")
    args = ap.parse_args()
    py = sys.executable
    dense = f"dense_{args.tag}"
    run([py, os.path.join(WORK, "pipeline", "pack_state.py"), args.val, args.priv,
         "--out-name", dense])
    outcsv = os.path.join(SUBS, f"submission_{args.tag}.csv")
    run([py, os.path.join(WORK, "pipeline", "patch_state.py"),
         "--base", args.base, "--state-dir", os.path.join(SUBS, dense), "--output", outcsv])
    run([py, os.path.join(WORK, "pipeline", "check_lean.py"), outcsv])
    zpath = os.path.join(SUBS, f"submission_{args.tag}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(outcsv, arcname="submission.csv")
    print(f"{args.tag}: {os.path.getsize(zpath)/1e6:.1f} MB -> {zpath}")


if __name__ == "__main__":
    main()
