"""Build the supervised Task 2 dataset from the train split.

For every panel:
  * reproduce the released window selector (greedy chronological acceptance with
    the official condition definition, horizon-queue requirement and 360-min
    spacing) and take all accepted origins as supervised windows;
  * queue_onset windows contribute one row per link (the last horizon step only,
    which is where the selector puts the onset queue) with the full horizon
    truth kept aside for evaluation;
  * queue_ongoing windows contribute 6*L rows (horizon steps x links); the fit
    part is subsampled with ``--ongoing-stride`` to keep the dataset small;
  * features come from features.window_features (history only);
  * labels are the smoothed-speed queue indicator (eligibility-masked).

Saved per panel in work/task2/cache/ds_<PANEL>.npz.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2core as C
import features as FT


def onset_freq_all(d: dict, onset: pd.DataFrame) -> np.ndarray:
    step = C.HORIZON - 1
    f = np.zeros(d["n_links"], np.float64)
    for r in onset.itertuples(index=False):
        f += d["Qobs"][r.ti + 1 + step]
    return (f / max(len(onset), 1)).astype(np.float32)


def ong_freq_all(d: dict, ongoing: pd.DataFrame) -> np.ndarray:
    f = np.zeros(d["n_links"], np.float64)
    for r in ongoing.itertuples(index=False):
        f += d["Qobs"][r.ti - 1]
    return (f / max(len(ongoing), 1)).astype(np.float32)


def rate_all(d: dict) -> np.ndarray:
    L = d["n_links"]
    tod = (d["time_dt"].hour * 60 + d["time_dt"].minute).to_numpy(np.int32)
    slots = tod // 5
    s = np.zeros((288, L))
    np.add.at(s, slots, d["Q"].astype(np.float64))
    cnt = np.bincount(slots, minlength=288).astype(np.float64)
    return (s / np.maximum(cnt, 1.0)[:, None]).astype(np.float32)


def tod_profiles(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """(288, L) NaN-aware time-of-day means of speed/v_cut and flow/capacity."""
    L = d["n_links"]
    tod = (d["time_dt"].hour * 60 + d["time_dt"].minute).to_numpy(np.int32) // 5
    with np.errstate(invalid="ignore", divide="ignore"):
        r = d["speed"] / d["v_cut"][None, :]
        fc = d["flow"] / np.maximum(d["capacity"], 1e-3)[None, :]
    order = np.argsort(tod, kind="stable")
    bounds = np.searchsorted(tod[order], np.arange(289))
    out_r = np.zeros((288, L), np.float32)
    out_f = np.zeros((288, L), np.float32)
    rv = r[order]
    fv = fc[order]
    for s in range(288):
        a, b = bounds[s], bounds[s + 1]
        if b > a:
            out_r[s] = np.nanmean(rv[a:b], axis=0)
            out_f[s] = np.nanmean(fv[a:b], axis=0)
    return out_r, out_f


def day_index(d: dict):
    dates = pd.DatetimeIndex(d["time_dt"]).normalize()
    uniq = np.sort(np.asarray(dates.unique(), dtype="datetime64[ns]"))
    pos = np.searchsorted(uniq, np.asarray(dates, dtype="datetime64[ns]"))
    return pos.astype(np.int32), uniq


def build_panel(panel: str, panel_id: int, cut: pd.Timestamp,
                ongoing_stride: int = 4, onset_stride: int = 4) -> str:
    d = C.load_train_panel(panel)
    cand = C.candidate_table(d)
    days, uniq = day_index(d)
    cand["day_idx"] = days[cand.ti.to_numpy()]
    acc = C.greedy_select(cand)
    acc["day_idx"] = days[acc.ti.to_numpy()]
    acc_ti = {int(r.ti): r.condition for r in acc.itertuples(index=False)}
    cut_day = int(np.searchsorted(uniq, np.datetime64(cut.tz_localize(None))))
    onset = acc[acc.condition == "queue_onset"]
    ongoing = acc[acc.condition == "queue_ongoing"]
    ratio_tod, fcap_tod = tod_profiles(d)
    clim = FT.clim_from(d, onset_freq_all(d, onset), ong_freq_all(d, ongoing),
                        rate_all(d), d["elig"].mean(axis=0).astype(np.float32),
                        ratio_tod, fcap_tod)
    L = d["n_links"]
    ramp = C.traffic.load_ramps(panel, "train")
    for k in ("on_flow", "off_flow", "on_valid", "off_valid"):
        d["ramp_" + k] = ramp[k]

    def ramp_slice(ti):
        sl = slice(ti - C.HIST, ti)
        return {"on": d["ramp_on_flow"][sl], "off": d["ramp_off_flow"][sl],
                "on_valid": d["ramp_on_valid"][sl], "off_valid": d["ramp_off_valid"][sl]}

    # ---- onset windows: every valid selector candidate (subsampled) plus the
    # greedy-accepted ones (kept even if the stride would skip them) ----------
    pool = cand[(~cand.ongoing) & cand.ok].sort_values("ti")
    fit_pool = pool[pool.day_idx < cut_day]
    sel_ti = list(fit_pool.ti.to_numpy()[::max(onset_stride, 1)])
    chosen = set(sel_ti)
    for r in onset[onset.day_idx < cut_day].itertuples(index=False):
        if int(r.ti) not in chosen:
            chosen.add(int(r.ti))
            sel_ti.append(int(r.ti))
    sel_ti = sorted(chosen)
    X_on, y_on, date_on, on_truth, on_elig, on_acc = [], [], [], [], [], []
    for ti in sel_ti:
        c = FT.set_window_clim(clim, d, ti)
        X = FT.window_features(d, ti, c, panel_id, 0, ramp_slice(ti))
        X_on.append(X[(C.HORIZON - 1) * L:])
        y_on.append(d["Q"][ti + C.HORIZON].copy())
        date_on.append(int(days[ti]))
        on_truth.append(d["Q"][ti + 1:ti + 1 + C.HORIZON].copy())
        on_elig.append(d["elig"][ti + 1:ti + 1 + C.HORIZON].copy())
        on_acc.append(int(ti) in acc_ti)
    # eval = greedy-accepted onset windows from the held-out block
    for r in onset[onset.day_idx >= cut_day].itertuples(index=False):
        ti = int(r.ti)
        c = FT.set_window_clim(clim, d, ti)
        X = FT.window_features(d, ti, c, panel_id, 0, ramp_slice(ti))
        X_on.append(X[(C.HORIZON - 1) * L:])
        y_on.append(d["Q"][ti + C.HORIZON].copy())
        date_on.append(int(days[ti]))
        on_truth.append(d["Q"][ti + 1:ti + 1 + C.HORIZON].copy())
        on_elig.append(d["elig"][ti + 1:ti + 1 + C.HORIZON].copy())
        on_acc.append(1)

    # ---- ongoing windows: greedy accepted, fit part strided -------------
    X_ong, y_ong, date_ong, ong_truth, ong_elig, ong_acc = [], [], [], [], [], []
    n_fit_ong = 0
    for r in ongoing.itertuples(index=False):
        ti = int(r.ti)
        fit = int(r.day_idx) < cut_day
        if fit:
            n_fit_ong += 1
            if n_fit_ong % ongoing_stride:
                continue
        c = FT.set_window_clim(clim, d, ti)
        X_ong.append(FT.window_features(d, ti, c, panel_id, 1, ramp_slice(ti)))
        y_ong.append(d["Q"][ti + 1:ti + 1 + C.HORIZON].reshape(-1).copy())
        date_ong.append(int(r.day_idx))
        ong_truth.append(d["Q"][ti + 1:ti + 1 + C.HORIZON].copy())
        ong_elig.append(d["elig"][ti + 1:ti + 1 + C.HORIZON].copy())
        ong_acc.append(1)

    out = {
        "panel": np.array(panel), "n_links": np.array(L, np.int32),
        "X_on": (np.concatenate(X_on, axis=0) if X_on else np.zeros((0, FT.N_FEATURES), np.float32)),
        "y_on": (np.concatenate(y_on).astype(np.int8) if y_on else np.zeros(0, np.int8)),
        "date_on": np.array(date_on, np.int32),
        "on_accepted": np.array(on_acc, bool),
        "on_truth": np.array(on_truth, np.int8), "on_elig": np.array(on_elig, bool),
        "X_ong": (np.concatenate(X_ong, axis=0) if X_ong else np.zeros((0, FT.N_FEATURES), np.float32)),
        "y_ong": (np.concatenate(y_ong).astype(np.int8) if y_ong else np.zeros(0, np.int8)),
        "date_ong": np.array(date_ong, np.int32),
        "ong_accepted": np.array(ong_acc, bool),
        "ong_truth": np.array(ong_truth, np.int8), "ong_elig": np.array(ong_elig, bool),
        "cut_day": np.array(cut_day, np.int32), "n_days": np.array(len(uniq), np.int32),
        "onset_freq": clim["onset_freq"], "ong_freq": clim["ong_freq"],
        "rate": clim["rate"], "elig_rate": clim["elig_rate"],
        "ratio_tod": ratio_tod, "fcap_tod": fcap_tod,
        "links": d["links"].astype(str), "rank_to_idx": d["rank_to_idx"].astype(np.int32),
        "v_cut": d["v_cut"], "panel_id": np.array(panel_id, np.int32),
    }
    path = os.path.join(C.CACHE2, f"ds_{panel}.npz")
    np.savez_compressed(path, **out)
    print(f"{panel}: onset win {len(date_on)} ({len(y_on):,} rows) | "
          f"ongoing win {len(date_ong)} ({len(y_ong):,} rows) -> {os.path.basename(path)}")
    del d, out
    gc.collect()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--cut", default="2031-02-01")
    ap.add_argument("--ongoing-stride", type=int, default=4)
    ap.add_argument("--onset-stride", type=int, default=4)
    args = ap.parse_args()
    C.ensure_dirs()
    cut = pd.Timestamp(args.cut, tz="UTC")
    panels = args.panels or C.PANELS
    for i, p in enumerate(panels):
        if not os.path.exists(C.traffic.split_cache_path(p, "train")):
            print(f"{p}: cache missing, skip")
            continue
        build_panel(p, i, cut, args.ongoing_stride, args.onset_stride)


if __name__ == "__main__":
    main()


