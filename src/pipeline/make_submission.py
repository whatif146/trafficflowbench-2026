"""Build the single upload file and validate it against the sample submission.

Usage:
  python work/pipeline/make_submission.py \
      --state work/submissions/state_validation.csv \
      --queue work/task2/out/queue_validation.csv \
      --odme  work/task4/out/path_flow_validation.csv \
      --out   work/submissions/submission.csv

The merge itself is the official src/merge_submissions.py (streamed over the
642 MB submission_key.csv).  The checker then verifies row count, ids, task tags,
blank cells and per-task value domains against sample_submission.csv.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
REL = os.path.join(ROOT, "kaggle_public")
REPO = os.path.join(ROOT, "repo", "trafficflowbench-public-main")
LIBS = os.path.join(ROOT, "libs")
COLS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def merge(state, queue, odme, out):
    cmd = [sys.executable, os.path.join(REPO, "src", "merge_submissions.py"),
           "--state", state, "--queue", queue, "--odme", odme,
           "--key", os.path.join(REL, "submission_key.csv"), "--output", out]
    env = dict(os.environ)
    env["PYTHONPATH"] = LIBS + os.pathsep + env.get("PYTHONPATH", "")
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def check(out):
    print("checking", out, flush=True)
    sample = pd.read_csv(os.path.join(REL, "sample_submission.csv"),
                         usecols=["submission_id", "task"])
    sub = pd.read_csv(out)
    assert list(sub.columns) == COLS, f"columns {list(sub.columns)}"
    assert len(sub) == len(sample), f"rows {len(sub)} != {len(sample)}"
    assert (sub.submission_id.to_numpy() == sample.submission_id.to_numpy()).all(), "ids/order mismatch"
    assert (sub.task.astype(str).to_numpy() == sample.task.astype(str).to_numpy()).all(), "task tags mismatch"
    blanks = sub.isna().sum()
    assert blanks.sum() == 0, f"blank cells: {blanks[blanks > 0].to_dict()}"
    print(f"rows ok: {len(sub):,}")
    print("task counts:", sub.task.value_counts().to_dict())
    for task in sub.task.unique():
        d = sub[sub.task == task]
        print(f"\n[{task}] rows={len(d):,}")
        for c in COLS[2:]:
            v = d[c].to_numpy(dtype=float)
            if np.all(v == 0):
                continue
            print(f"  {c}: min={np.min(v):.4g} max={np.max(v):.4g} mean={np.mean(v):.4g} "
                  f"nonzero={np.mean(v != 0):.4f} finite={np.isfinite(v).mean():.6f}")
    st = sub[sub.task == "state"]
    assert (st.speed_kmh >= 0).all() and (st.flow_vph >= 0).all(), "negative state values"
    q = sub[sub.task == "queue"]
    assert q.queue_pred.isin([0, 1]).all(), "non-binary queue_pred"
    o = sub[sub.task == "odme"]
    assert (o.path_flow >= 0).all() and np.isfinite(o.path_flow).all(), "bad path_flow"
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--odme", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not args.check_only:
        merge(os.path.abspath(args.state), os.path.abspath(args.queue),
              os.path.abspath(args.odme), out)
    check(out)
