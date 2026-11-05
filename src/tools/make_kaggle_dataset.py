"""Upload final/ as a public Kaggle Dataset so the e-mail can carry a link.

Usage (after the final package is rebuilt):
  python work/tools/make_kaggle_dataset.py
Requires the package directory final/ (579 MB) and the kaggle API token.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = r"C:\Users\lenovo\Desktop\kaggle"
FINAL = os.path.join(ROOT, "final")
USERNAME = "hsuyooyoo"
METADATA = {
    "title": "TrafficFlowBench CAMPUS SUNNY final package",
    "id": f"{USERNAME}/trafficflowbench-campus-sunny-final-package",
    "subtitle": "2026 IEEE BigData Cup Challenge 06 Track 2 - code, report and predictions",
    "description": (
        "Final package for the 2026 IEEE BigData Cup Challenge 06, Track 2 "
        "(TrafficFlowBench). Contents: full source code, the single-entry "
        "reproduction script (CPU only), the frozen U-Net deploy predictions and "
        "weights, the selected submission file, the experiment ledger, and the "
        "PDF report. Reproduction: python work/pipeline/reproduce_final.py "
        "--release-root /path/to/kaggle_public."
    ),
    "licenses": [{"name": "CC0-1.0"}],
}


def main():
    meta_path = os.path.join(FINAL, "dataset-metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(METADATA, f, indent=2)
    print("wrote", meta_path)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "create", "-p", FINAL,
           "--dir-mode", "zip"]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"dataset URL: https://www.kaggle.com/datasets/{METADATA['id']}")


if __name__ == "__main__":
    main()
