"""Fit the Task 2 static parameters on the full train split and save them.

Reads the rule choice produced by run_static_cv.py (out/static_cv_info.json) and
re-fits the frequency / time-of-day climatology on *all* train days, which is
what a submission for validation/private may use (the splits are later months).

Output: models/static_<PANEL>.npz
  links, rank_to_idx, rate (288,L) f32, onset_freq (L), ong_freq (L),
  elig_rate (L), onset_sel (link indices), onset_rule, ongoing_rule
"""
from __future__ import annotations

import gc
import json
import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C
import static_model as SM

DEFAULT_RULES = {
    "onset_rule": "('top', 3)",
    "ongoing_rule": "('robust', 0, 0.35, 0)",
}


def load_rules() -> dict:
    path = os.path.join(C.OUT2, "static_cv_info.json")
    rules = {}
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        for s in data.get("saves", []):
            rules[s["panel"]] = {"onset_rule": s["onset_rule"],
                                 "ongoing_rule": s["ongoing_rule"]}
    return rules


def onset_freq_full(d: dict, onset: pd.DataFrame) -> np.ndarray:
    step = C.HORIZON - 1
    f = np.zeros(d["n_links"], np.float64)
    for r in onset.itertuples(index=False):
        f += d["Qobs"][r.ti + 1 + step]
    return (f / max(len(onset), 1)).astype(np.float32)


def ong_freq_full(d: dict, ongoing: pd.DataFrame) -> np.ndarray:
    f = np.zeros(d["n_links"], np.float64)
    for r in ongoing.itertuples(index=False):
        f += d["Qobs"][r.ti - 1]
    return (f / max(len(ongoing), 1)).astype(np.float32)


def rate_full(d: dict) -> np.ndarray:
    L = d["n_links"]
    tod = (d["time_dt"].hour * 60 + d["time_dt"].minute).to_numpy(np.int32)
    slots = tod // 5
    s = np.zeros((288, L))
    np.add.at(s, slots, d["Q"].astype(np.float64))
    cnt = np.bincount(slots, minlength=288).astype(np.float64)
    return (s / np.maximum(cnt, 1.0)[:, None]).astype(np.float32)


def fit_panel_save(panel: str, onset_rule: str, ongoing_rule: str):
    d = C.load_train_panel(panel)
    cand = C.candidate_table(d)
    acc = C.greedy_select(cand).sort_values("time_dt").reset_index(drop=True)
    onset = acc[acc.condition == "queue_onset"]
    ongoing = acc[acc.condition == "queue_ongoing"]
    freq = onset_freq_full(d, onset)
    ong_freq = ong_freq_full(d, ongoing)
    rate = rate_full(d)
    elig_rate = d["elig"].mean(axis=0).astype(np.float32)
    import build_dataset as BD
    ratio_tod, fcap_tod = BD.tod_profiles(d)
    # apply the selected onset rule to the full-train frequency
    kind, val = _parse_rule(onset_rule)
    if kind == "top":
        sel = np.argsort(-freq)[:int(val)]
    else:
        sel = np.where(freq >= val)[0]
        if len(sel) == 0:
            sel = np.argsort(-freq)[:2]
    path = os.path.join(C.MODEL2, f"static_{panel}.npz")
    np.savez_compressed(
        path,
        links=d["links"].astype(str), rank_to_idx=d["rank_to_idx"].astype(np.int32),
        rate=rate, onset_freq=freq, ong_freq=ong_freq, elig_rate=elig_rate,
        onset_sel=sel.astype(np.int32), v_cut=d["v_cut"], free_speed=d["free_speed"],
        capacity=d["capacity"], lanes=d["lanes"], k_jam=d["k_jam"],
        length_km=d["length_km"], rank=d["rank"], rel_rank=d["rel_rank"],
        on_ramp=d["on_ramp"], off_ramp=d["off_ramp"],
        onset_rule=np.array(onset_rule), ongoing_rule=np.array(ongoing_rule),
        ratio_tod=ratio_tod, fcap_tod=fcap_tod,
    )
    print(f"{panel}: saved {path} onset_sel={len(sel)} ({', '.join(d['links'][sel][:8])})")
    del d
    gc.collect()


def _parse_rule(text: str):
    import ast
    return ast.literal_eval(text)


def main():
    C.ensure_dirs()
    rules = load_rules()
    for panel in C.PANELS:
        if not os.path.exists(C.traffic.split_cache_path(panel, "train")):
            print(f"{panel}: cache missing, skip")
            continue
        r = rules.get(panel, DEFAULT_RULES)
        fit_panel_save(panel, r["onset_rule"], r["ongoing_rule"])


if __name__ == "__main__":
    main()

