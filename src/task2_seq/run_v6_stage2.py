"""Stage 2: v3+v6 blend vs the recorded v3+v5 blend baseline.

Trains only the 65-feature v3 models (the v6 models were saved by
run_v6_cv.py --stage 1), then:
  * protocol B: (w, th) chosen on January for the v3+v6 blend, scored on Feb;
  * protocol A: same January parameters, scored on the released June windows.
Baseline = the v3+v5 blend numbers in out/final_cv_stride1.json (same rows).
"""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import run_cv as RC
import run_final_cv as FC
import run_v6_cv as V6C
import run_exp as EX

import lightgbm as lgb


def main():
    base = json.load(open(os.path.join(LH.OUT, "final_cv_stride1.json")))
    out = {"v5_blend": {"B": {k: base["on"]["blend_B"][k] for k in ("overall", "onset", "ongoing")},
                        "A": {k: base["on"]["blend_A"][k] for k in ("overall", "onset", "ongoing")}}}
    for cond, rounds, tag in (("on", RC.ON_ROUNDS, "onset"), ("ong", RC.ONG_ROUNDS, "ongoing")):
        # ---------- protocol B: v3 model + v6 probs ----------
        X, y = FC.gather_lean(cond, "v3", exclude_june=False)[:2]
        m3 = FC.train_model(cond, "v3", X, y, rounds)
        del X, y
        gc.collect()
        p3j = EX.block_probs(cond, FC.COLS["v3"], m3, "tune")
        p3f = EX.block_probs(cond, FC.COLS["v3"], m3, "feb")
        del m3
        gc.collect()
        m6 = lgb.Booster(model_file=os.path.join(LH.MODELS, f"v6_{tag}_B.txt"))
        p6j = V6C.probs_pseudo(cond, "v6", m6, "tune")
        p6f = V6C.probs_pseudo(cond, "v6", m6, "feb")
        del m6
        gc.collect()
        w, th, jan = V6C.tune_blend(cond, p3j, p6j)
        feb = V6C.score_pseudo(cond, V6C.blend(p3f, p6f, w), th)
        feb.to_csv(os.path.join(LH.OUT, f"v6blend_{cond}_feb.csv"), index=False)
        del p3j, p3f, p6j, p6f
        gc.collect()
        # ---------- protocol A: June-excluded v3 + v6 probs ----------
        X, y = FC.gather_lean(cond, "v3", exclude_june=True)[:2]
        m3 = FC.train_model(cond, "v3", X, y, rounds)
        del X, y
        gc.collect()
        p3jr = EX.june_probs(cond, FC.COLS["v3"], m3)
        del m3
        gc.collect()
        m6 = lgb.Booster(model_file=os.path.join(LH.MODELS, f"v6_{tag}_A.txt"))
        p6jr = V6C.probs_rel(cond, "v6", m6)
        del m6
        gc.collect()
        june = V6C.score_rel(cond, V6C.blend(p3jr, p6jr, w), th)
        june.to_csv(os.path.join(LH.OUT, f"v6blend_{cond}_june.csv"), index=False)
        out[cond] = {"B": {"w": w, "th": th, "jan": jan, "detail": feb},
                     "A": {"w": w, "th": th, "detail": june}}
        del p3jr, p6jr
        gc.collect()
    summ = {"v5_blend": out["v5_blend"], "v6_blend": {}}
    for proto in ("B", "A"):
        d = pd.concat([out["on"][proto]["detail"], out["ong"][proto]["detail"]], ignore_index=True)
        agg = LH.aggregate(d)
        summ["v6_blend"][proto] = {"overall": agg["overall"], "onset": agg["onset"],
                                   "ongoing": agg["ongoing"], "per_panel": agg["per_panel"],
                                   "w_on": out["on"][proto]["w"], "w_ong": out["ong"][proto]["w"],
                                   "th_on": out["on"][proto]["th"], "th_ong": out["ong"][proto]["th"]}
        b = summ["v5_blend"][proto]
        print(f"== v3+v6 blend protocol {proto}: overall {agg['overall']:.4f} | "
              f"onset {agg['onset']:.4f} | ongoing {agg['ongoing']:.4f} "
              f"(v5 baseline {b['overall']:.4f} / {b['onset']:.4f} / {b['ongoing']:.4f})", flush=True)
    with open(os.path.join(LH.OUT, "v6_stage2.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
