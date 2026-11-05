"""Memory-lean validation of the merged upload file against sample_submission.

Streams both CSVs in chunks so it works with a few hundred MB of RAM.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle")
REL = os.environ.get("TFB_REL", ROOT + r"\kaggle_public")
SUB = sys.argv[1] if len(sys.argv) > 1 else ROOT + r"\work\submissions\submission_v1.csv"
CHUNK = 750_000

stats = {}


def add(task, col, v):
    v = np.asarray(v, dtype=np.float64)
    s = stats.setdefault((task, col), {"n": 0, "min": np.inf, "max": -np.inf, "sum": 0.0,
                                       "nonzero": 0, "nan": 0})
    s["n"] += len(v)
    s["nan"] += int(np.isnan(v).sum())
    vv = v[np.isfinite(v)]
    if len(vv):
        s["min"] = min(s["min"], float(vv.min()))
        s["max"] = max(s["max"], float(vv.max()))
        s["sum"] += float(vv.sum())
        s["nonzero"] += int((vv != 0).sum())


sample = pd.read_csv(REL + r"\sample_submission.csv", usecols=["submission_id", "task"],
                     chunksize=CHUNK)
sub = pd.read_csv(SUB, chunksize=CHUNK)
sid, stask = [], []
rows = 0
for a, b in zip(sample, sub):
    rows += len(b)
    if len(a) != len(b):
        raise SystemExit(f"chunk length mismatch {len(a)} vs {len(b)}")
    if not (a.submission_id.to_numpy() == b.submission_id.to_numpy()).all():
        raise SystemExit("submission_id order mismatch")
    if not (a.task.astype(str).to_numpy() == b.task.astype(str).to_numpy()).all():
        raise SystemExit("task tag mismatch")
    for task in ("state", "queue", "odme"):
        m = b.task.to_numpy() == task
        if m.any():
            d = b.loc[m]
            if task == "state":
                add(task, "speed_kmh", d.speed_kmh.to_numpy())
                add(task, "flow_vph", d.flow_vph.to_numpy())
            elif task == "queue":
                add(task, "queue_pred", d.queue_pred.to_numpy())
            else:
                add(task, "path_flow", d.path_flow.to_numpy())
print(f"rows={rows:,} ids/tasks/chunks OK")
for (task, col), s in sorted(stats.items()):
    print(f"[{task}] {col}: n={s['n']:,} nan={s['nan']} nonzero_share={s['nonzero']/max(s['n'],1):.4f} "
          f"min={s['min']:.4g} max={s['max']:.4g} mean={s['sum']/max(s['n'],1):.4g}")
    if s["nan"]:
        raise SystemExit(f"NaN in {task}.{col}")
q = stats.get(("queue", "queue_pred"))
if q and not (q["min"] >= 0 and q["max"] <= 1):
    raise SystemExit("queue_pred out of {0,1}")
o = stats.get(("odme", "path_flow"))
if o and o["min"] < 0:
    raise SystemExit("negative path_flow")
print("LEAN CHECK PASSED")
