"""Build the calibration submissions for inverse analysis of the hidden scoring.

Both are derived from the final candidate (submission_q_w30ong40fb1_all.csv,
Task-1 LightGBM + Task-2 best queue + Task-4 NNLS), changing ONLY one task so the
total-score difference attributes exactly to that task:

  submission_cal_zeroqueue  queue_pred := 0 for every queue row
      -> the official Task-2 evaluator gives IoU 0 for every window (every scored
         window contains a queue by spec), so S_queue = 0 exactly and
         Q_best = (total_best - total_zeroqueue) / 0.30.
  submission_cal_zeroodme   path_flow := 0 for every odme row
      -> S_link = 0 exactly (zero flows); S_od/S_dev/S_attr are expected 0 for
         zero flows (verify against src/task4/score_task4.py guards), so this
         estimates S_ODME and hence T4 = (total_best - total_zeroodme) / 0.20.

Streams the ~400 MB CSV in chunks to stay memory-lean; validates row count, ids,
task tags and the format guards before zipping.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
WORK = os.path.join(ROOT, "work")
SUBS = os.path.join(WORK, "submissions")
BASE = os.path.join(SUBS, "submission_q_w30ong40fb1_all.csv")
TEMPLATE = os.path.join(ROOT, "kaggle_public", "sample_submission.csv")
CHUNK = 1_000_000


def build(tag: str, task: str):
    outcsv = os.path.join(SUBS, f"submission_{tag}.csv")
    first = True
    changed = 0
    for chunk in pd.read_csv(BASE, chunksize=CHUNK):
        m = chunk.task.to_numpy() == task
        if m.any():
            if task == "queue":
                chunk.loc[m, "queue_pred"] = 0
            elif task == "odme":
                chunk.loc[m, "path_flow"] = 0.0
            changed += int(m.sum())
        chunk.to_csv(outcsv, mode="w" if first else "a", header=first, index=False)
        first = False
    zpath = os.path.join(SUBS, f"submission_{tag}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(outcsv, arcname="submission.csv")
    print(f"{tag}: {changed:,} {task} rows zeroed -> {outcsv} "
          f"({os.path.getsize(outcsv)/1e6:.1f} MB), zip {os.path.getsize(zpath)/1e6:.1f} MB")
    return outcsv


def validate(csv_path: str):
    subprocess.run([sys.executable, os.path.join(WORK, "pipeline", "check_lean.py"), csv_path],
                   check=True)


def main():
    q = build("cal_zeroqueue", "queue")
    validate(q)
    o = build("cal_zeroodme", "odme")
    validate(o)


if __name__ == "__main__":
    main()
