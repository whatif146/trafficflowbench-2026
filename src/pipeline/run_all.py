"""End-to-end reproduction of the submitted TrafficFlowBench solution.

Produces the Kaggle upload file from the released data in one command:

    python work/pipeline/run_all.py --release-root C:\\Users\\lenovo\\Desktop\\kaggle\\kaggle_public \
        --out work/submissions/submission_repro.csv

Steps (all deterministic):
  1. build/refresh the per-panel cache                   work/build_cache.py
  2. Task 1 state predictions (validation + private)     work/task1/gbm/predict.py
  3. Task 2 queue predictions (validation + private)     work/task2/predict.py --model gbmv3
  4. Task 4 path flows      (validation + private)       work/task4/predict.py
  5. pack T1 predictions into dense arrays               work/pipeline/pack_state.py
  6. stream-merge the three task files into the upload    work/pipeline/split_merge.py
  7. validate the upload format                          work/pipeline/check_lean.py

Task 3 needs no file: it is scored on the Task 1 numbers.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
PY = sys.executable
WORK = os.path.join(ROOT, "work")


def run(cmd, **kw):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default=os.path.join(ROOT, "kaggle_public"))
    ap.add_argument("--out", default=os.path.join(WORK, "submissions", "submission_repro.csv"))
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--skip-t1", action="store_true")
    ap.add_argument("--skip-t2", action="store_true")
    ap.add_argument("--skip-t4", action="store_true")
    ap.add_argument("--t1-outdir", default=None,
                    help="where the Task-1 CSVs are written (default work/task1/gbm)")
    ap.add_argument("--t2-outdir", default=None,
                    help="where the Task-2 CSVs are written (default work/task2/out)")
    ap.add_argument("--t4-outdir", default=None,
                    help="where the Task-4 CSVs are written (default work/task4/out)")
    ap.add_argument("--dense-name", default="dense_gbm")
    args = ap.parse_args()
    subs = os.path.join(WORK, "submissions")
    t1dir = args.t1_outdir or os.path.join(WORK, "task1", "gbm")
    t2dir = args.t2_outdir or os.path.join(WORK, "task2", "out")
    t4dir = args.t4_outdir or os.path.join(WORK, "task4", "out")
    os.makedirs(subs, exist_ok=True)
    os.makedirs(t1dir, exist_ok=True)
    os.makedirs(t2dir, exist_ok=True)
    os.makedirs(t4dir, exist_ok=True)

    if not args.skip_cache:
        run([PY, os.path.join(WORK, "build_cache.py")])

    if not args.skip_t1:
        run([PY, os.path.join(WORK, "task1", "gbm", "predict.py"),
             "--split", "validation", "private", "--outdir", t1dir])
        run([PY, os.path.join(WORK, "pipeline", "pack_state.py"),
             os.path.join(t1dir, "state_gbm_validation.csv"),
             os.path.join(t1dir, "state_gbm_private.csv"), "--out-name", args.dense_name])
    else:
        run([PY, os.path.join(WORK, "pipeline", "pack_state.py"),
             os.path.join(subs, "state_gbm_all.csv"), "--out-name", args.dense_name])

    if not args.skip_t2:
        run([PY, os.path.join(WORK, "task2", "predict.py"), "--model", "gbmv3",
             "--split", "both", "--output", os.path.join(t2dir, "queue_gbmv3.csv")])
        # split both -> validation/private files; combine for the merge
        combine([os.path.join(t2dir, "queue_gbmv3_validation.csv"),
                 os.path.join(t2dir, "queue_gbmv3_private.csv")],
                os.path.join(t2dir, "queue_gbmv3_all.csv"))

    if not args.skip_t4:
        for split in ("validation", "private"):
            run([PY, os.path.join(WORK, "task4", "predict.py"), "--split", split,
                 "--method", "baseline",
                 "--output", os.path.join(t4dir, f"path_flow_{split}.csv")])
        combine([os.path.join(t4dir, "path_flow_validation.csv"),
                 os.path.join(t4dir, "path_flow_private.csv")],
                os.path.join(t4dir, "path_flow_validation_plus_private.csv"))

    run([PY, os.path.join(WORK, "pipeline", "fast_merge.py"),
         "--state-dir", os.path.join(subs, args.dense_name),
         "--queue", os.path.join(t2dir, "queue_gbmv3_all.csv"),
         "--odme", os.path.join(t4dir, "path_flow_validation_plus_private.csv"),
         "--output", args.out, "--check"])
    run([PY, os.path.join(WORK, "pipeline", "check_lean.py"), args.out])
    print("\nREPRODUCTION COMPLETE:", args.out)


def combine(paths, out):
    import shutil
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
    print("combined ->", out, os.path.getsize(out), flush=True)


if __name__ == "__main__":
    main()
