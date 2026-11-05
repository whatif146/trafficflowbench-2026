"""Sanity check: does the shipped GBM holdout CSV reproduce the documented S_state?"""
import os, sys
import numpy as np

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "unet"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "gbm"))

import unetlib as U
from t1lib import score, fmt
from run_holdout import load_gbm_rows

DOC = {"D7_I210_W": 0.9291, "D7_I405_S": 0.9323, "D7_I10_E": 0.9391, "D12_I5_N": 0.9461}
for panel, doc in DOC.items():
    c = U.load_cache(panel, "train_hold60")
    df, tpl_pos, ps, pq = load_gbm_rows(panel, c, os.path.join(ROOT, "work", "task1", "gbm",
                                                               f"holdout_{panel}.csv"))
    tv, tq, lanes, reg = U.truth_rows(c, tpl_pos)
    res = score(tv, tq, ps, pq, lanes, reg)
    print(f"{panel:11s} csv_rows={len(df):8d} S_state={res['macro']:.4f} (doc {doc:.4f}) "
          f"diff={res['macro']-doc:+.4f}", flush=True)
    ss = c["tpl_s"][tpl_pos]
    bs = c["base_s"][c["tpl_t"][tpl_pos], ss].astype(np.float64)
    bq = c["base_q"][c["tpl_t"][tpl_pos], ss].astype(np.float64)
    print("   base-only", fmt(score(tv, tq, bs, bq, lanes, reg)), flush=True)
