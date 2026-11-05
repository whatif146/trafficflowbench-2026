"""Submit a file to the TrafficFlowBench Kaggle competition and report the score.

Usage:
  python work/tools/submit.py --file work/submissions/submission_v1.zip --message "v1 baseline"
  python work/tools/submit.py --list
"""
from __future__ import annotations

import argparse
import os
import sys
import time

LIBS = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\libs"
if LIBS not in sys.path:
    sys.path.insert(0, LIBS)

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

COMP = "2026-ieee-big-data-traffic-flow-bench"


def show(api):
    subs = api.competition_submissions(COMP, page_size=10) or []
    for s in subs:
        attrs = {a: getattr(s, a) for a in dir(s) if not a.startswith("_") and
                 a in ("ref", "date", "description", "fileName", "status", "score",
                       "publicScore", "privateScore", "errorDescription", "submissionDate")}
        print(attrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--message", default="candidate")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--wait", type=int, default=600)
    args = ap.parse_args()
    api = KaggleApi()
    api.authenticate()
    if args.list:
        show(api)
        return
    path = os.path.abspath(args.file)
    print(f"submitting {path} ({os.path.getsize(path)/1e6:.1f} MB): {args.message}", flush=True)
    resp = api.competition_submit(path, args.message, COMP, quiet=False)
    print("submit response:", resp, flush=True)
    ref = None
    for attr in ("ref", "submissionRef", "id"):
        if hasattr(resp, attr):
            ref = getattr(resp, attr)
            break
    print("ref:", ref, flush=True)
    deadline = time.time() + args.wait
    while time.time() < deadline:
        time.sleep(15)
        subs = api.competition_submissions(COMP, page_size=10) or []
        if not subs:
            print("no submissions listed yet", flush=True)
            continue
        s = subs[0]
        status = str(getattr(s, "status", "")).split(".")[-1]
        score = getattr(s, "score", None)
        print(f"{time.strftime('%H:%M:%S')} status={status} score={score}", flush=True)
        if status.upper() in ("COMPLETE", "ERROR", "FAILED"):
            print("FINAL:", {a: getattr(s, a) for a in dir(s)
                             if not a.startswith("_") and not callable(getattr(s, a))}, flush=True)
            return
    print("timed out waiting; check --list", flush=True)


if __name__ == "__main__":
    main()
