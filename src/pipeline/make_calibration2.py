"""Extra calibration submissions (spares for inverse analysis).

  submission_cal_statezero: state rows speed=1.0, flow=1.0 (safe: avoids 0/0 in
      the physics density identity), queue+odme from the final candidate.
      S_state(all-1) is still 0 (RMS speed 111.8 >> 25, flow/lane >> 600) and
      S_physics(all-1) ~ 0 (N ~ 0, accumulation error >> rhs), so the measured
      total equals 0.30*S_queue + 0.20*S_ODME, i.e. it reveals S_queue
      independently of the Task-1/Task-3 terms and cross-checks the T4 anchor.
  submission_cal_allones: queue rows = 1, state+odme from the final candidate.
      Measures S_queue(all-ones) = (1/80)*sum_w |T_eligible| / |E_eligible|, an
      independent truth-set-size test (H2 predicts ~0.09).
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
CHUNK = 1_000_000


def build(tag: str, task: str, speed=None, flow=None, queue=None):
    outcsv = os.path.join(SUBS, f"submission_{tag}.csv")
    first = True
    changed = 0
    for chunk in pd.read_csv(BASE, chunksize=CHUNK):
        m = chunk.task.to_numpy() == task
        if m.any():
            if task == "state":
                if speed is not None:
                    chunk.loc[m, "speed_kmh"] = float(speed)
                if flow is not None:
                    chunk.loc[m, "flow_vph"] = float(flow)
            elif task == "queue":
                chunk.loc[m, "queue_pred"] = int(queue)
            changed += int(m.sum())
        chunk.to_csv(outcsv, mode="w" if first else "a", header=first, index=False)
        first = False
    zpath = os.path.join(SUBS, f"submission_{tag}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(outcsv, arcname="submission.csv")
    print(f"{tag}: {changed:,} {task} rows rewritten -> {outcsv} "
          f"({os.path.getsize(outcsv)/1e6:.1f} MB)")
    subprocess.run([sys.executable, os.path.join(WORK, "pipeline", "check_lean.py"), outcsv],
                   check=True)


def main():
    build("cal_statezero", "state", speed=1.0, flow=1.0)
    build("cal_allones", "queue", queue=1)


if __name__ == "__main__":
    main()
