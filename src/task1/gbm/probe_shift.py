"""Probe the extra Task-2 horizon blanks in the released masked INPUT layer.

For each panel/split: compare the released masked view (NaN speed/flow) against
the Task-1 template targets, restricted to cells with pct_observed >= 75 (i.e.
cells that would be scoreable if not blanked).  Cells that are blanked, not
Task-1 targets and pct>=75 are the "extra" horizon blanks that exist on the
scored splits (and are claimed not to exist on train).  Reports the run shape
relative to the released Task-2 window origins.
"""
from __future__ import annotations

import sys

sys.path.insert(0, r"C:\Users\lenovo\Desktop\kaggle\work\lib")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import traffic  # noqa: E402


def to_epoch(s):
    d = pd.to_datetime(pd.Series([s]), utc=True).dt.as_unit("s")
    return int(d.astype("int64").iloc[0])


def probe(panel: str, split: str, verbose_windows: int = 3):
    masked = traffic._load_or_build(panel, split, "masked")
    times = masked["times"].astype(np.int64)
    sp, fl = masked["speed"], masked["flow"]
    T, S = sp.shape
    tg = traffic.load_targets(panel, split)
    target = np.zeros((T, S), dtype=bool)
    target[tg["t_idx"], tg["s_idx"]] = True
    blank = ~np.isfinite(sp) | ~np.isfinite(fl)
    pct = masked["pct"]
    elig = pct >= 75
    n_targets = int(target.sum())
    n_blank = int(blank.sum())
    extra = blank & ~target & elig
    inel = blank & ~target & ~elig
    print(f"\n== {panel} {split}: T={T} S={S} days={T//288} n_blank={n_blank} "
          f"targets={n_targets} blank_not_target_pct>=75={int(extra.sum())} "
          f"blank_not_target_pct<75={int(inel.sum())} "
          f"({extra.sum()/max(n_targets,1)*100:.2f}% of template)", flush=True)
    if extra.sum() == 0:
        return
    # structure over time
    per_t = extra.sum(axis=1)
    day_of_t = np.arange(T) // 288
    extra_days = np.unique(day_of_t[extra.any(axis=1)])
    print(f"   extra days: {extra_days.tolist()[:20]} ({len(extra_days)} days)", flush=True)
    # runs of consecutive time steps with extras, record link sets
    nz = np.where(per_t > 0)[0]
    runs = []
    if len(nz):
        start = nz[0]
        prev = nz[0]
        for t in nz[1:]:
            if t != prev + 1:
                runs.append((start, prev))
                start = t
            prev = t
        runs.append((start, prev))
    print(f"   runs of steps with extras: {len(runs)}; lengths: "
          f"{sorted(set(b - a + 1 for a, b in runs))[:10]}", flush=True)
    # offsets relative to window origins
    wi = traffic.load_window_index()
    w = wi[(wi.panel == panel) & (wi.split == split)]
    t_lookup = pd.Series(np.arange(T), index=pd.DatetimeIndex(pd.to_datetime(times, unit="s", utc=True)))
    offsets = []
    for _, row in w.iterrows():
        try:
            oi = int(t_lookup.loc[pd.to_datetime(row["forecast_origin"], utc=True)])
        except KeyError:
            continue
        cnt = [int(per_t[oi + k]) if (oi + k) < T else 0 for k in range(0, 22)]
        offsets.append(cnt)
        if len(offsets) <= verbose_windows:
            links = np.where(extra[oi + 1: oi + 19].any(axis=0))[0] if oi + 19 <= T else np.array([], int)
            print(f"   window {row['window_id']} origin_idx={oi} day={row['date']} "
                  f"per_step(+0..+21)={cnt} n_links_with_extra={len(links)}", flush=True)
    print(f"   n windows probed: {len(offsets)}", flush=True)
    # how many links are affected per run, and whether the link set is a prefix/contiguous
    for a, b in runs[:4]:
        links = np.where(extra[a: b + 1].any(axis=0))[0]
        cnt_per_link = extra[a: b + 1].sum(axis=0)
        print(f"   run steps {a}-{b} (len {b-a+1}): links={len(links)} "
              f"steps_per_link(min/max)={cnt_per_link[links].min()}/{cnt_per_link[links].max()} "
              f"link_idx_range={links.min() if len(links) else -1}..{links.max() if len(links) else -1}",
              flush=True)
    del masked, target, blank, extra
    import gc
    gc.collect()


if __name__ == "__main__":
    panels = sys.argv[1:] or ["D7_I10_E", "D12_I5_N"]
    for panel in panels:
        for split in ("train", "validation", "private"):
            try:
                probe(panel, split)
            except Exception as exc:
                print(f"{panel} {split}: failed {exc!r}")
