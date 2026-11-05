"""Collect public competition material (discussion topics, notebooks) for review.

Usage:
  python work/tools/kaggle_research.py --topics
  python work/tools/kaggle_research.py --kernels
  python work/tools/kaggle_research.py --pull <kernel_ref> --dest work/research/<name>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

COMP = "2026-ieee-big-data-traffic-flow-bench"
OUT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\work\research"


def topics(api):
    os.makedirs(OUT, exist_ok=True)
    page = 1
    all_topics = []
    while page <= 5:
        try:
            ts = api.competition_list_topics(COMP, page=page)
        except Exception as e:
            print("topics page", page, "error", e)
            break
        if not ts:
            break
        for t in ts:
            attrs = {a: getattr(t, a) for a in dir(t) if not a.startswith("_")
                     and not callable(getattr(t, a))}
            all_topics.append(attrs)
        print(f"page {page}: {len(ts)} topics")
        page += 1
    print(f"total topics {len(all_topics)}")
    for t in all_topics:
        print("\n====", json.dumps(t, default=str)[:800])
    json.dump(all_topics, open(os.path.join(OUT, "topics.json"), "w"), default=str, indent=1)
    # messages of each topic
    for t in all_topics:
        tid = t.get("id") or t.get("topic_id")
        if not tid:
            continue
        try:
            ms = api.competition_list_topic_messages(COMP, int(tid), page_size=50)
        except Exception as e:
            print("messages", tid, "error", e)
            continue
        for m in ms:
            body = getattr(m, "message", "") or getattr(m, "raw_markdown", "") or ""
            print(f"\n----- topic {tid} by {getattr(m,'poster_name','?')} at {getattr(m,'post_date','?')}\n{body[:4000]}")


def kernels(api):
    os.makedirs(OUT, exist_ok=True)
    all_k = []
    for page in range(1, 6):
        try:
            ks = api.kernels_list(competition=COMP, page=page, page_size=50)
        except Exception as e:
            print("kernels page", page, "error", e)
            break
        if not ks:
            break
        for k in ks:
            attrs = {a: getattr(k, a) for a in dir(k) if not a.startswith("_")
                     and not callable(getattr(k, a))}
            all_k.append(attrs)
        print(f"page {page}: {len(ks)} kernels")
    print(f"total kernels {len(all_k)}")
    for k in all_k:
        print("====", {kk: k.get(kk) for kk in ("ref", "title", "author", "lastRunTime",
                                                 "totalVotes", "language") if kk in k})
    json.dump(all_k, open(os.path.join(OUT, "kernels.json"), "w"), default=str, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", action="store_true")
    ap.add_argument("--kernels", action="store_true")
    args = ap.parse_args()
    api = KaggleApi()
    api.authenticate()
    if args.topics:
        topics(api)
    if args.kernels:
        kernels(api)


if __name__ == "__main__":
    main()
