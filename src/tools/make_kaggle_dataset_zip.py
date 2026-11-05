"""Create the final-package Kaggle Dataset from one archive (fast, robust).

The per-file upload path was far too slow (hundreds of small files + big CSVs), so
this builds a single `submission_package.zip` and lets the Kaggle CLI upload one
file. Run after the final package is frozen:
  python work/tools/make_kaggle_dataset_zip.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile

ROOT = r"C:\Users\lenovo\Desktop\kaggle"
FINAL = os.path.join(ROOT, "final")
STAGING = os.path.join(ROOT, "work", "dataset_staging")
USERNAME = "hsuyooyoo"
DATASET_ID = f"{USERNAME}/trafficflowbench-campus-sunny-final-package"
METADATA = {
    "title": "TrafficFlowBench CAMPUS SUNNY final package",
    "id": DATASET_ID,
    "subtitle": "2026 IEEE BigData Cup Challenge 06 Track 2 - code, report and predictions",
    "description": (
        "Final package for the 2026 IEEE BigData Cup Challenge 06, Track 2 "
        "(TrafficFlowBench). The archive contains the full source code, the "
        "single-entry CPU reproduction script, frozen U-Net deploy predictions and "
        "weights, the selected submission file, the experiment ledger, and the PDF "
        "report. Reproduction: python work/pipeline/reproduce_final.py "
        "--release-root /path/to/kaggle_public."
    ),
    "licenses": [{"name": "CC0-1.0"}],
}


def main():
    os.makedirs(STAGING, exist_ok=True)
    with open(os.path.join(STAGING, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(METADATA, f, indent=2)
    zpath = os.path.join(STAGING, "submission_package.zip")
    print("zipping final/ ->", zpath)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dp, dns, fns in os.walk(FINAL):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for fn in fns:
                if fn == "dataset-metadata.json":
                    continue
                src = os.path.join(dp, fn)
                rel = os.path.relpath(src, FINAL)
                z.write(src, arcname=rel)
    print(f"archive {os.path.getsize(zpath)/1e6:.0f} MB")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "create", "-p", STAGING]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"dataset URL: https://www.kaggle.com/datasets/{DATASET_ID}")


if __name__ == "__main__":
    main()
