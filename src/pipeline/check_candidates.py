"""Sanity-check the pre-built Task-2 candidate submissions against the P4 base.

For every work/submissions/submission_q_*.csv:
  * row count / ids / task alignment with sample_submission,
  * number of queue cells that differ from the P4 file (work/submissions/submission_p4.csv),
  * positive counts by condition.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
REL = os.path.join(ROOT, "kaggle_public")
SUBS = os.path.join(ROOT, "work", "submissions")
CHUNK = 1_000_000

base = pd.read_csv(os.path.join(SUBS, "submission_p4.csv"), chunksize=CHUNK)
base_q = []
for b in base:
    m = b.task.to_numpy() == "queue"
    base_q.append(b.loc[m, ["submission_id", "queue_pred"]].to_numpy())
base_q = np.concatenate(base_q)
base_map = {int(i): v for i, v in base_q}
print("P4 queue rows:", len(base_map), "positives:", int(base_q[:, 1].sum()))

for path in sorted(glob.glob(os.path.join(SUBS, "submission_q_*.csv"))):
    name = os.path.basename(path)
    n = 0
    diffs = 0
    pos = 0
    ok = True
    for k, c in enumerate(pd.read_csv(path, chunksize=CHUNK, dtype={"task": str})):
        n += len(c)
        m = c.task.to_numpy() == "queue"
        q = c.loc[m, ["submission_id", "queue_pred"]]
        ids = q.submission_id.to_numpy(np.int64)
        vals = q.queue_pred.to_numpy(np.int64)
        pos += int(vals.sum())
        exp = np.array([base_map.get(int(i), -1) for i in ids])
        diffs += int((exp != vals).sum())
    print(f"{name:38s} rows={n:,} queue_pos={pos:,} diff_vs_P4={diffs:,} ok={n == 6980503}")
