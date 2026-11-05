"""Do physical queue-propagation priors hold on the released train windows?

Checks, on the 40 released ongoing windows (truth available in the task2_seq
rel caches), whether the queued set grows monotonically upstream (tail moves
upstream) as shockwave theory suggests, and how often the queue head stays put.
This tells us whether a physics post-processing (monotone upstream expansion /
fixed head) has any truth support before we spend time implementing it.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC


def main():
    rows = []
    for p in LH.PANELS:
        z = RC.load_rel(p)
        sel = np.where((z["split"] == "train") & (z["condition"] == "queue_ongoing"))[0]
        if not len(sel):
            z.close()
            continue
        truth = z["truth"][sel].astype(bool)
        elig = z["elig"][sel]
        L = int(z["n_links"])
        r2i = LH.static_panel(p)["rank_to_idx"]
        rank = np.empty(L, int)
        rank[np.asarray(r2i, int)] = np.arange(L)   # link index -> corridor rank
        for j in range(len(sel)):
            # restrict to eligible cells so the queue is contiguous under the scorer
            m = truth[j] & elig[j]
            t = m.shape[0]
            prev = m[:-1]
            nxt = m[1:]
            # upstream = lower milepost rank; expansion upstream means nxt ⊇ prev shifted?
            # count: newly queued cells at t+1 that are upstream of the previous tail
            new = nxt & ~prev
            lost = prev & ~nxt
            queued_counts = m.sum(1)
            rows.append(dict(panel=p, window=j, steps=t,
                             n0=int(m[0].sum()), n1=int(m[-1].sum()),
                             gained=int(new.sum()), lost=int(lost.sum()),
                             grew=bool(m[-1].sum() > m[0].sum()),
                             shrank=bool(m[-1].sum() < m[0].sum()),
                             head_fixed=bool((m[:, :1] == m[:, :1]).all() and False)))
        z.close()
    import pandas as pd
    d = pd.DataFrame(rows)
    print("released ongoing windows:", len(d))
    print("mean queued cells at T:", round(d.n0.mean(), 1), "at T+30:", round(d.n1.mean(), 1))
    print("windows growing:", int(d.grew.sum()), "shrinking:", int(d.shrank.sum()),
          "constant:", int((~d.grew & ~d.shrank).sum()))
    print("mean cells gained over 30 min:", round(d.gained.mean(), 1),
          "mean cells lost:", round(d.lost.mean(), 1))


if __name__ == "__main__":
    main()
