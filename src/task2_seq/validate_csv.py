"""Validate the candidate queue CSVs against the template and the shipped v3."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import traffic


def load(path):
    d = pd.read_csv(path, dtype={"window_id": str, "link_id": str})
    return d


tpl = traffic.load_queue_template()
tpl["window_id"] = tpl.window_id.astype(str)
tpl["link_id"] = tpl.link_id.astype(str)
v3_dir = os.path.join(LH.WORK, "task2", "out")

for split in ("validation", "private"):
    print(f"\n===== {split} =====")
    cand = load(os.path.join(LH.OUT, f"queue_v5_{split}.csv"))
    v3 = load(os.path.join(v3_dir, f"queue_gbmv3_{split}.csv"))
    wids = set(v3.window_id)
    t = tpl[tpl.window_id.isin(wids)].copy()
    print("template rows", len(t), "| candidate rows", len(cand), "| v3 rows", len(v3))
    m = t.merge(cand, on=["window_id", "timestamp", "link_id"], how="left",
                suffixes=("", "_cand"), validate="one_to_one")
    print("template rows covered:", int(m.queue_pred_cand.notna().sum()),
          "| binary:", bool(pd.to_numeric(m.queue_pred_cand, errors="coerce").isin([0, 1]).all()),
          "| dup keys:", int(cand.duplicated(["window_id", "timestamp", "link_id"]).sum()))
    # per-condition diagnostics
    wi = traffic.load_window_index()
    wi = wi[(wi.split == split)]
    cond = dict(zip(wi.window_id.astype(str), wi.condition))
    for d, name in ((cand, "candidate"), (v3, "v3")):
        d = d.copy()
        d["condition"] = d.window_id.map(cond)
        d["step"] = d.groupby("window_id").cumcount() // (len(t) // (d.window_id.nunique() * 6))
        n_on = d[d.condition == "queue_onset"]
        n_ong = d[d.condition == "queue_ongoing"]
        print(f"{name:>9}: positives total {d.queue_pred.mean():.3%} | onset {n_on.queue_pred.mean():.3%} "
              f"| ongoing {n_ong.queue_pred.mean():.3%}")
    both = v3.merge(cand, on=["window_id", "timestamp", "link_id"], suffixes=("_v3", "_cand"))
    print("cells differing from v3:", int((both.queue_pred_v3 != both.queue_pred_cand).sum()),
          f"({(both.queue_pred_v3 != both.queue_pred_cand).mean():.2%})",
          "| v3 pos & cand 0:", int(((both.queue_pred_v3 == 1) & (both.queue_pred_cand == 0)).sum()),
          "| v3 0 & cand pos:", int(((both.queue_pred_v3 == 0) & (both.queue_pred_cand == 1)).sum()))

# raw prediction positive rate from the current v5 models alone (sanity, not tuning)
import run_cv as RC
import lightgbm as lgb
m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_onset.txt"))
m_ong = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_ongoing.txt"))
print("\nv5 deployment models: trees", m_on.num_trees(), m_ong.num_trees())
