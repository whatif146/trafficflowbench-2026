"""Single entry script for the final package (organizer requirement).

Reproduces the selected final submission end-to-end from the released data:

  python code/pipeline/reproduce_final.py --release-root /path/to/kaggle_public

Steps (all deterministic; CPU only):
  1. pipeline/run_all.py           build caches, Task-1 LightGBM state predictions,
                                   Task-4 NNLS path flows -> submission_base.csv
                                   (Task 2 is skipped; its queue layer is shipped
                                   as a frozen component in submissions/package_data)
  2. pipeline/reproduce_unet_state.py
                                   blend 0.6*GBM + 0.4*U-Net from the shipped
                                   deploy npz predictions (seeds 43+44, refit on all
                                   273 train days) -> state_gbm_unet_final_*.csv
  3. pipeline/pack_state.py        pack the blended state into dense per-panel arrays
  4. pipeline/patch_state.py       patch the state rows into the base upload
  5. pipeline/patch_queue.py       patch the shipped Task-2 queue component
  6. pipeline/check_lean.py        validate the 6,980,503-row upload

The package's `work/` directory is created next to the code; nothing outside it is
written. Set TRAFFIC_* env vars if you want a different cache location.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(BASE, "work")
WORK_DIR = os.path.join(BASE, "work")
PKG_DATA = os.path.join(BASE, "submissions", "package_data")


def run(cmd, env):
    print(">>>", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=BASE, env=env).returncode
    if rc != 0:
        raise SystemExit(f"step failed ({rc}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True,
                    help="directory that contains config/, corridors/, task1/, task2/, task4/")
    ap.add_argument("--skip-cache", action="store_true")
    args = ap.parse_args()
    env = os.environ.copy()
    env["TFB_ROOT"] = BASE
    env["TFB_REL"] = os.path.abspath(args.release_root)
    env["TRAFFIC_ROOT"] = BASE
    env["TRAFFIC_REL"] = os.path.abspath(args.release_root)
    env["TRAFFIC_WORK"] = WORK_DIR
    py = sys.executable
    pipe = os.path.join(CODE, "pipeline")
    subs = os.path.join(WORK_DIR, "submissions")

    cmd = [py, os.path.join(pipe, "run_all.py"),
           "--release-root", args.release_root,
           "--out", os.path.join(subs, "submission_base.csv")]
    if args.skip_cache:
        cmd.append("--skip-cache")
    run(cmd, env)

    run([py, os.path.join(pipe, "reproduce_unet_state.py"), "--w", "0.6",
         "--gbm-dir", os.path.join(WORK_DIR, "task1", "gbm"),
         "--npz-dir", os.path.join(CODE, "task1", "unet", "deploy_npz"),
         "--out-dir", os.path.join(WORK_DIR, "task1", "unet")], env)

    st_val = os.path.join(WORK_DIR, "task1", "unet", "state_gbm_unet_final_validation.csv")
    st_priv = os.path.join(WORK_DIR, "task1", "unet", "state_gbm_unet_final_private.csv")
    run([py, os.path.join(pipe, "pack_state.py"), st_val, st_priv,
         "--out-name", "dense_unet_final"], env)
    run([py, os.path.join(pipe, "patch_state.py"),
         "--base", os.path.join(subs, "submission_base.csv"),
         "--state-dir", os.path.join(subs, "dense_unet_final"),
         "--output", os.path.join(subs, "submission_unet_state.csv")], env)

    q_val = os.path.join(PKG_DATA, "queue_w30ong40fb1_all_validation.csv")
    q_priv = os.path.join(PKG_DATA, "queue_w30ong40fb1_all_private.csv")
    if not (os.path.exists(q_val) and os.path.exists(q_priv)):
        raise SystemExit(f"missing shipped queue component in {PKG_DATA}")
    run([py, os.path.join(pipe, "patch_queue.py"),
         "--val", q_val, "--priv", q_priv, "--tag", "final",
         "--base", os.path.join(subs, "submission_unet_state.csv")], env)

    final = os.path.join(subs, "submission_final.csv")
    run([py, os.path.join(pipe, "check_lean.py"), final], env)
    print("FINAL PACKAGE REPRODUCED:", final)


if __name__ == "__main__":
    main()
