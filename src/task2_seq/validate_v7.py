"""Validate the v7 candidate files and compare their footprint with queue_w30ong40fb1."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import traffic

wi = traffic.load_window_index()
cond = dict(zip(wi.window_id.astype(str), wi.condition))
tpl = traffic.load_queue_template()
tpl["window_id"] = tpl.window_id.astype(str)
tpl["link_id"] = tpl.link_id.astype(str)

for split in ("validation", "private"):
    cand = pd.read_csv(os.path.join(LH.OUT, f"queue_v7_{split}.csv"),
                       dtype={"window_id": str, "link_id": str})
    best = pd.read_csv(os.path.join(LH.OUT, f"queue_w30ong40fb1_{split}.csv"),
                       dtype={"window_id": str, "link_id": str})
    best = best[best.window_id.isin(cand.window_id)]
    print(f"\n===== {split} =====")
    m = tpl[tpl.window_id.isin(cand.window_id)].merge(
        cand, on=["window_id", "timestamp", "link_id"], how="left",
        validate="one_to_one", indicator=True)
    print("template coverage:", int((m._merge == "both").sum()), "/", len(m),
          "| binary:", bool(cand.queue_pred.isin([0, 1]).all()),
          "| dup keys:", int(cand.duplicated(["window_id", "timestamp", "link_id"]).sum()))
    cand = cand.copy()
    cand["condition"] = cand.window_id.map(cond)
    best = best.copy()
    best["condition"] = best.window_id.map(cond)
    for c in ("queue_onset", "queue_ongoing"):
        a = cand[cand.condition == c]
        b = best[best.condition == c]
        print(f"  {c}: v7 {len(a):,} cells / {int(a.queue_pred.sum()):,} pos "
              f"({a.queue_pred.mean():.3%}) | best {int(b.queue_pred.sum()):,} pos "
              f"({b.queue_pred.mean():.3%})")
        if c == "queue_onset":
            ga = a.groupby("window_id").queue_pred.sum()
            gb = b.groupby("window_id").queue_pred.sum()
            print(f"    onset links/window: v7 mean {ga.mean():.2f} (min {int(ga.min())}, "
                  f"max {int(ga.max())}) | best mean {gb.mean():.2f} "
                  f"(min {int(gb.min())}, max {int(gb.max())})")
    mm = best.merge(cand, on=["window_id", "timestamp", "link_id"], suffixes=("_best", "_v7"))
    diff = mm[mm.queue_pred_best != mm.queue_pred_v7]
    print("  differing cells:", len(diff), "| best-only:", int(((mm.queue_pred_best == 1) & (mm.queue_pred_v7 == 0)).sum()),
          "| v7-only:", int(((mm.queue_pred_best == 0) & (mm.queue_pred_v7 == 1)).sum()))
