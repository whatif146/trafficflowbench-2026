"""Batch-submit the pre-built Task-2 probe candidates and collect their LB scores.

Reads work/submissions/probe_plan.txt (one "tag zip_name" per line, comments with #),
submits each candidate through work/tools/kaggle_submit.py, waits for the score, and
appends the result to work/submissions/probe_results.json.  Re-running skips tags
that already have a result and skips candidates that are already COMPLETE on Kaggle
(matched by description), so the remaining quota is never wasted on a duplicate.

Usage:
  python work/tools/probe_batch.py                # submit everything pending in the plan
  python work/tools/probe_batch.py --dry-run      # only print what would be submitted
  python work/tools/probe_batch.py --only q_on025fb1,q_on025
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
WORK = os.path.join(ROOT, "work")
SUBS = os.path.join(WORK, "submissions")
PLAN = os.path.join(SUBS, "probe_plan.txt")
RESULTS = os.path.join(SUBS, "probe_results.json")
SUBMIT = os.path.join(WORK, "tools", "kaggle_submit.py")
COMP = "2026-ieee-big-data-traffic-flow-bench"


def load_results() -> dict:
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    return {}


def existing_submissions(api) -> dict:
    """description -> (ref, status, public_score) for our submissions."""
    out = {}
    try:
        subs = api.competition_submissions(COMP, page_size=50) or []
    except Exception as exc:  # pragma: no cover
        print("could not list submissions:", exc)
        return out
    for s in subs:
        desc = getattr(s, "description", "") or ""
        out[desc] = (getattr(s, "ref", None),
                     str(getattr(s, "status", "")).split(".")[-1],
                     getattr(s, "public_score", None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--wait", type=int, default=1500, help="seconds to wait per submission score")
    args = ap.parse_args()
    only = set(x.strip() for x in args.only.split(",")) if args.only else None

    plan = []
    for line in open(PLAN, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        tag, zipname = line.split()
        plan.append((tag, zipname))

    results = load_results()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    existing = existing_submissions(api)

    for tag, zipname in plan:
        if only is not None and tag not in only:
            continue
        if tag in results:
            print(f"skip {tag}: already recorded {results[tag].get('public_score')}")
            continue
        if tag in existing:
            ref, status, score = existing[tag]
            print(f"skip {tag}: already on Kaggle (ref {ref} {status} score {score})")
            if status == "COMPLETE" and score is not None:
                results[tag] = {"ref": ref, "public_score": float(score)}
                json.dump(results, open(RESULTS, "w"), indent=1)
            continue
        path = os.path.join(SUBS, zipname)
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        print(f"=== submitting {tag} ({zipname})", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run([sys.executable, SUBMIT, "--file", path, "--message", tag,
                             "--wait", "900"], cwd=ROOT)
        if rc.returncode != 0:
            print(f"submission command failed for {tag} (rc={rc.returncode})")
            continue
        # poll the score
        deadline = time.time() + args.wait
        score = None
        ref = None
        while time.time() < deadline:
            time.sleep(30)
            try:
                subs = api.competition_submissions(COMP, page_size=10) or []
            except Exception as exc:
                print("poll failed:", exc)
                continue
            for s in subs:
                if (getattr(s, "description", "") or "") == tag:
                    ref = getattr(s, "ref", None)
                    status = str(getattr(s, "status", "")).split(".")[-1]
                    if status == "COMPLETE" and getattr(s, "public_score", None) is not None:
                        score = float(s.public_score)
                    break
            if score is not None:
                break
        if score is not None:
            results[tag] = {"ref": ref, "public_score": score}
            json.dump(results, open(RESULTS, "w"), indent=1)
            print(f"=== {tag}: public {score} (ref {ref})", flush=True)
        else:
            print(f"=== {tag}: score not available yet; leaving unrecorded", flush=True)


if __name__ == "__main__":
    main()
