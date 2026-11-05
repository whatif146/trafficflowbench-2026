"""Extra onset-threshold variants of the final blend (0.45 / 0.50 / 0.55).

Same frozen artefacts as out/queue_v5_*.csv; only the onset decision threshold on
the last horizon step changes (ongoing stays at the shipped 0.25).
"""
from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lhcore as LH
import ship_variants as SV
import ship_blend as SB

import lightgbm as lgb


def main():
    cv = json.load(open(os.path.join(LH.OUT, "final_cv_stride1.json")))
    w_on = float(cv["on"]["blend_B"]["w"])
    w_ong = float(cv["ong"]["blend_B"]["w"])
    m_on = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_onset.txt"))
    m_ong = lgb.Booster(model_file=os.path.join(LH.MODELS, "v5_ongoing.txt"))
    v3_on, v3_ong = SB.load_v3_boosters()
    import traffic
    tpl_all = traffic.load_queue_template()
    tpl_all["window_id"] = tpl_all.window_id.astype(str)
    tpl_all["link_id"] = tpl_all.link_id.astype(str)
    tpl_all = tpl_all.drop(columns=["queue_pred"], errors="ignore")
    for split in ("validation", "private"):
        pm = SV.blend_probs_split(split, w_on, w_ong, m_on, m_ong, v3_on, v3_ong)
        for name, th in (("on045", 0.45), ("on050", 0.50), ("on055", 0.55)):
            SV.write_csv(pm, f"queue_v5blend_{name}", split, tpl_all,
                         on_th=th, on_fb=False, ong_th=0.25)
        del pm
        gc.collect()


if __name__ == "__main__":
    main()
