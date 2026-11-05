"""Download and mine public Kaggle notebooks for the competition (review-only).

Usage:
  python work/tools/kaggle_pull.py --list
  python work/tools/kaggle_pull.py --pull-all
  python work/tools/kaggle_pull.py --grep "leak" --grep "submission"
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

COMP = "2026-ieee-big-data-traffic-flow-bench"
OUT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\work\research"

KEY_KERNELS = [
    "anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517",
    "sujanmajhisuzan/trafficflowbench-top-rank-pipeline-0-90-target",
    "robinamirbahar/trafficflowbench-physics-guided-state-odme",
    "plarmuseau/trafficflowbench-the-physics-of-a-synthetic-jam",
    "plarmuseau/trafficflowbench-eleven-anomalies-of-the-data",
    "sachith7/trafficflowbench-when-does-temporal-repair-fail",
    "yangkuangou/traffic-task-2-forecast-time-availability-audit",
    "avikdas567/hydrodynamic-traffic-reconstruction-queue-odme",
    "yanamiiannaa/trafficflow-directional-features-safe-profiles",
    "cacinie/tfb-v5orig",
    "cacinie/tfb-report",
    "paragpu/trafficflow-v1",
    "prvsiyan/traffic-flow-trying-some-stuff",
    "chaitanyajamble/2026-ieee-big-data-traffic-flow-bench-2",
    "chaitanyajamble/2026-ieee-big-data-traffic-flow-bench-3",
    "faresabbasai2022/trafficflowbench-2026-74",
]


def list_kernels(api):
    ks = api.kernels_list(competition=COMP, page=1, page_size=50) or []
    for k in ks:
        ref = getattr(k, "ref", "")
        title = getattr(k, "title", "")
        votes = getattr(k, "total_votes", getattr(k, "totalVotes", "?"))
        print(f"{ref} | {title} | votes={votes}")


def pull(api, ref):
    name = ref.split("/")[-1]
    dest = os.path.join(OUT, name)
    os.makedirs(dest, exist_ok=True)
    try:
        api.kernels_pull(ref, dest, metadata=True, quiet=True)
    except Exception as e:
        print(f"  pull failed: {e}")
        return None
    # find notebook file
    nb = None
    for f in os.listdir(dest):
        if f.endswith(".ipynb"):
            nb = os.path.join(dest, f)
    if nb:
        try:
            data = json.load(open(nb, encoding="utf-8"))
            cells = data.get("cells", [])
            print(f"  {name}: {len(cells)} cells")
            return nb
        except Exception as e:
            print(f"  {name}: bad notebook {e}")
    return None


def summarize(nb_path):
    """Print code-cell source snippets 1-6 lines, and flag interesting keywords."""
    data = json.load(open(nb_path, encoding="utf-8"))
    keywords = ["leak", "target", "truth", "submission", "score", "0.9", "hack",
                "submission_key", "window", "queue_true", "private", "validation"]
    for i, c in enumerate(data.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        hits = [k for k in keywords if k.lower() in src.lower()]
        print(f"\n--- cell {i} ({len(src)} chars) hits={hits}")
        print(src[:1200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pull-all", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    api = KaggleApi()
    api.authenticate()
    os.makedirs(OUT, exist_ok=True)
    if args.list:
        list_kernels(api)
    if args.pull_all:
        for ref in KEY_KERNELS:
            print("====", ref)
            nb = pull(api, ref)
            if nb and args.summarize:
                summarize(nb)


if __name__ == "__main__":
    main()
