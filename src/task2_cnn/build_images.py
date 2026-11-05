"""Build corridor-image caches for the onset CNN (Task 2, work/task2_cnn).

Input per window: a [C, 13, L] float16 corridor image with
  13 time steps = the released 60-min history (T-60 .. T-5, observed layer) plus
  the origin row T from the split's masked layer, and channels
  (speed/v_cut, speed-observed mask, flow/capacity, density/k_jam,
   queue flag, flow-observed mask, TOD sin, TOD cos, relative link rank,
   relative time, capacity/10k, lanes/6).

Sources are read-only: work/task2_seq/lhcore.py (history assembly), the frozen
v3 dataset caches (window lists / labels / truth) and the released-window caches.

Outputs (work/task2_cnn/cache/):
  tr_<PANEL>.npz   pseudo onset windows used for fitting (all pre-2031-02-01)
  rel_<PANEL>.npz  all 30 released windows of every split (for protocol A and
                   for building a submission)

Nothing here reads a timestamp > T.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\lenovo\Desktop\kaggle"
WORK = os.path.join(ROOT, "work")
SEQ = os.path.join(WORK, "task2_seq")
for _p in (os.path.join(WORK, "lib"), os.path.join(WORK, "task2"), SEQ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lhcore as LH  # noqa: E402  (read-only)
import traffic  # noqa: E402

CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

N_STEPS = 13          # T-60 .. T-5 observed + row T
CHANNELS = ["r", "obs", "fc", "dens", "q", "fobs", "tod_sin", "tod_cos",
            "rel_rank", "rel_time", "cap", "lanes",
            # per-link queue climatology (the v3 model's strongest onset prior)
            "onset_freq", "ong_freq", "rate_tgt", "rate_last", "elig_rate",
            # causal long-history summaries (rows -180..-65 of the masked layer)
            "qfrac_pre", "qage_T", "r_mean_pre", "dist_up_T", "dist_dn_T",
            "nq_panel_T"]
C = len(CHANNELS)


def _dist_queued(q: np.ndarray, r2i: np.ndarray, direction: int, cap: int = 20) -> np.ndarray:
    qr = q[r2i].astype(np.int32)
    L = len(qr)
    nxt = np.full(L, cap, np.int32)
    if direction > 0:
        for i in range(L - 2, -1, -1):
            nxt[i] = 1 if qr[i + 1] else min(nxt[i + 1] + 1, cap)
    else:
        for i in range(1, L):
            nxt[i] = 1 if qr[i - 1] else min(nxt[i - 1] + 1, cap)
    out = np.empty(L, np.float32)
    out[r2i] = nxt.astype(np.float32)
    return out


def _queue_age(rows: np.ndarray, fin: np.ndarray) -> np.ndarray:
    """Consecutive queued rows ending at the origin (unknown rows skipped)."""
    n, L = rows.shape
    age = np.zeros(L, np.float32)
    alive = np.ones(L, bool)
    for k in range(n - 1, -1, -1):
        known = fin[k]
        isq = rows[k] > 0.5
        age[alive & known & isq] += 1.0
        alive &= ~(known & ~isq)
    return age


def image_from_history(st: dict, S: np.ndarray, F: np.ndarray,
                       times: np.ndarray, target_slot: int | None = None,
                       last_slot: int | None = None) -> np.ndarray:
    """(C, N_STEPS, L) float16 image from a (37, L) causal history."""
    L = st["n_links"]
    S13 = S[LH.H_PRE - LH.HIST:LH.H_PRE + 1]      # rows 24..36 -> 13 steps
    F13 = F[LH.H_PRE - LH.HIST:LH.H_PRE + 1]
    tt = times[LH.H_PRE - LH.HIST:LH.H_PRE + 1]
    vc = st["v_cut"][None, :]
    cap = st["capacity"][None, :]
    kjam = st["k_jam"][None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = S13 / vc
        fc = F13 / cap
        dens = (F13 / np.maximum(S13, 1e-3)) / kjam
    obs = np.isfinite(S13)
    fobs = np.isfinite(F13)
    q = (r <= 1.0) & obs
    r = np.where(obs, r, 0.0)
    fc = np.where(fobs, fc, 0.0)
    dens = np.where(obs & fobs, dens, 0.0)
    r = np.clip(r, 0.0, 2.0)
    fc = np.clip(fc, 0.0, 2.0)
    dens = np.clip(dens, 0.0, 1.5)
    dti = pd.to_datetime(tt, unit="s", utc=True)
    tod = (dti.hour + dti.minute / 60.0).to_numpy(np.float64)
    ang = 2 * np.pi * tod / 24.0
    tod_sin = np.repeat(np.sin(ang)[:, None], L, axis=1)
    tod_cos = np.repeat(np.cos(ang)[:, None], L, axis=1)
    rel_rank = np.repeat(np.asarray(st["rel_rank"], np.float32)[None, :], N_STEPS, axis=0)
    rel_time = np.repeat(np.linspace(0.0, 1.0, N_STEPS, dtype=np.float32)[:, None], L, axis=1)
    cap_ch = np.repeat((np.asarray(st["capacity"], np.float32) / 10000.0).clip(0, 2)[None, :],
                       N_STEPS, axis=0)
    lane_ch = np.repeat((np.asarray(st["lanes"], np.float32) / 6.0).clip(0, 1.5)[None, :],
                        N_STEPS, axis=0)
    if target_slot is None:
        target_slot = int((dti[-2].hour * 60 + dti[-2].minute) // 5)
    if last_slot is None:
        last_slot = int((dti[-2].hour * 60 + dti[-2].minute) // 5)
    onset_freq = np.repeat(np.asarray(st["onset_freq"], np.float32)[None, :], N_STEPS, axis=0)
    ong_freq = np.repeat(np.asarray(st["ong_freq"], np.float32)[None, :], N_STEPS, axis=0)
    rate_tgt = np.repeat(np.asarray(st["rate"][target_slot % 288], np.float32)[None, :],
                         N_STEPS, axis=0)
    rate_last = np.repeat(np.asarray(st["rate"][last_slot % 288], np.float32)[None, :],
                          N_STEPS, axis=0)
    elig_rate = np.repeat(np.asarray(st["elig_rate"], np.float32)[None, :], N_STEPS, axis=0)
    # long-history summaries from rows 0..23 (-180..-65 min, masked layer)
    r2i = np.asarray(st["rank_to_idx"], np.int64)
    S_pre = S[:LH.H_PRE - LH.HIST]
    fin_pre = np.isfinite(S_pre)
    q_pre = (S_pre / vc) <= 1.0
    q_pre = q_pre & fin_pre
    qfrac_pre = q_pre.sum(axis=0) / np.maximum(fin_pre.sum(axis=0), 1)
    with np.errstate(invalid="ignore"):
        r_mean_pre = np.nanmean(S_pre / vc, axis=0)
    r_mean_pre = np.where(np.isfinite(r_mean_pre), r_mean_pre, 1.0)
    q_all = ((S / vc) <= 1.0) & np.isfinite(S)
    qage_T = _queue_age(q_all.astype(np.float32), np.isfinite(S))
    q_T_row = np.isfinite(S[LH.H_PRE]) & (S[LH.H_PRE] / vc.reshape(-1) <= 1.0)
    dist_up = _dist_queued(q_T_row.astype(np.float32), r2i, -1)
    dist_dn = _dist_queued(q_T_row.astype(np.float32), r2i, 1)
    nq_panel = np.full(L, float(q_T_row.sum()), np.float32)

    def bcast(a):
        return np.repeat(np.asarray(a, np.float32)[None, :], N_STEPS, axis=0)

    X = np.stack([r, obs.astype(np.float32), fc, dens.astype(np.float32),
                  q.astype(np.float32), fobs.astype(np.float32),
                  tod_sin.astype(np.float32), tod_cos.astype(np.float32),
                  rel_rank, rel_time, cap_ch, lane_ch,
                  onset_freq, ong_freq, rate_tgt, rate_last, elig_rate,
                  bcast(qfrac_pre), bcast(qage_T / 10.0), bcast(r_mean_pre),
                  bcast(dist_up / 20.0), bcast(dist_dn / 20.0), bcast(nq_panel / 20.0)],
                 axis=0)
    return X.astype(np.float16)


def build_train_panel(panel: str) -> str:
    t0 = time.time()
    st = LH.static_panel(panel)
    L = st["n_links"]
    z = np.load(os.path.join(LH.CACHE, f"tr_{panel}.npz"), allow_pickle=False)
    on_ti = np.asarray(z["on_ti"], np.int64)
    on_time = np.asarray(z["on_time"], np.int64)
    on_acc = np.asarray(z["on_acc"], bool)
    on_y = np.asarray(z["on_y"], np.int8).reshape(len(on_ti), L)
    on_truth = np.asarray(z["on_truth"], np.int8)
    on_elig = np.asarray(z["on_elig"], bool)
    z.close()
    assert np.array_equal(on_y, on_truth[:, LH.HORIZON - 1].astype(np.int8)), \
        f"{panel}: on_y is not the T+30 truth"
    d = LH.split_data(panel, "train")
    ps, pf = d["plain"]["speed"], d["plain"]["flow"]
    times = d["times"]
    tdt = pd.to_datetime(times, unit="s", utc=True)
    slots = ((tdt.hour * 60 + tdt.minute) // 5).to_numpy(np.int64)
    X = np.zeros((len(on_ti), C, N_STEPS, L), np.float16)
    for k, ti in enumerate(on_ti):
        ti = int(ti)
        S, F = LH.make_history(panel, "train", ti,
                               plain_hist=(ps[ti - LH.HIST:ti], pf[ti - LH.HIST:ti]))
        X[k] = image_from_history(st, S, F, times,
                                  target_slot=int(slots[min(ti + LH.HORIZON, len(slots) - 1)]),
                                  last_slot=int(slots[ti - 1]))
    out = {"panel": np.array(panel), "n_links": np.array(L, np.int32),
           "X": X, "y": on_y.astype(np.int8), "time": on_time,
           "acc": on_acc, "truth": on_truth, "elig": on_elig,
           "ti": on_ti.astype(np.int32)}
    path = os.path.join(CACHE, f"tr_{panel}.npz")
    np.savez_compressed(path, **out)
    print(f"{panel}: {len(on_ti)} onset windows L={L} -> {os.path.basename(path)} "
          f"| {X.nbytes/1e6:.0f}MB | {time.time()-t0:.1f}s", flush=True)
    del X, out
    LH.free_split(panel, "train")
    gc.collect()
    return path


def build_released_panel(panel: str) -> str:
    t0 = time.time()
    st = LH.static_panel(panel)
    L = st["n_links"]
    z = np.load(os.path.join(LH.CACHE, f"rel_{panel}.npz"), allow_pickle=False)
    win_ids = np.asarray(z["window_ids"], dtype=str)
    splits = np.asarray(z["split"], dtype=str)
    conds = np.asarray(z["condition"], dtype=str)
    tis = np.asarray(z["ti"], np.int64)
    z.close()
    wh = LH.window_histories(panel)
    X = np.zeros((len(win_ids), C, N_STEPS, L), np.float16)
    for k in range(len(win_ids)):
        split = str(splits[k])
        d = LH.split_data(panel, split)
        ti = int(tis[k])
        tdt = pd.to_datetime(d["times"], unit="s", utc=True)
        slots = ((tdt.hour * 60 + tdt.minute) // 5).to_numpy(np.int64)
        S, F = LH.make_history(panel, split, ti, win_hist=wh[str(win_ids[k])])
        X[k] = image_from_history(st, S, F, d["times"],
                                  target_slot=int(slots[min(ti + LH.HORIZON, len(slots) - 1)]),
                                  last_slot=int(slots[ti - 1]))
        LH.free_split(panel, split)
    out = {"panel": np.array(panel), "n_links": np.array(L, np.int32),
           "X": X, "window_ids": win_ids, "split": splits, "condition": conds}
    path = os.path.join(CACHE, f"rel_{panel}.npz")
    np.savez_compressed(path, **out)
    print(f"{panel}: {len(win_ids)} released windows L={L} -> {os.path.basename(path)} "
          f"| {X.nbytes/1e6:.0f}MB | {time.time()-t0:.1f}s", flush=True)
    del X, out
    gc.collect()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=None)
    ap.add_argument("--only", choices=["train", "rel", "both"], default="both")
    args = ap.parse_args()
    panels = args.panels or LH.PANELS
    for p in panels:
        if args.only in ("train", "both"):
            build_train_panel(p)
        if args.only in ("rel", "both"):
            build_released_panel(p)


if __name__ == "__main__":
    main()
