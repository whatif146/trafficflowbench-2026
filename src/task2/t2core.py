"""Task 2 core utilities: panel data, queue labels, window selection, IoU.

Facts established from the release (see NOTES.md):
  * History = 12 steps T-60..T-5, horizon = 6 steps T+5..T+30 (5-min grid).
  * The released windows were selected by a chronological greedy scan of
    candidate origins: accept the earliest origin whose visible history has no
    *established* queue (>=2 queued observations with >=2 on one link => ongoing,
    else onset) and whose horizon holds a queue, with >=360 min spacing inside
    each (panel, split, condition) group.  This reproduces every released
    queue_onset window exactly on the train split.
  * Consequence: queue_onset windows from that selector have their queue at the
    LAST horizon step only (T+30); steps 1-5 are empty for ~90% of the windows.
  * release facts: history_coverage = mean(is_score_eligible) over the 12xL
    history cells and must be >= 0.7.
"""
from __future__ import annotations

import os
import sys

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import pandas as pd

WORKLIB = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r"\work\lib"
if WORKLIB not in sys.path:
    sys.path.insert(0, WORKLIB)

import traffic  # noqa: E402

HIST = 12          # T-60 .. T-5
HORIZON = 6        # T+5 .. T+30
STEP_MIN = 5
MIN_SPACING_MIN = 360
MIN_HIST_COVERAGE = 0.7

PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W",
          "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S"]
FAMILY = {
    "D7_I10_E": "D7_I10", "D7_I10_W": "D7_I10",
    "D7_I210_E": "D7_I210", "D7_I210_W": "D7_I210",
    "D7_I405_N": "D7_I405", "D7_I405_S": "D7_I405",
    "D12_I5_N": "D12_I5", "D12_I5_S": "D12_I5",
}
FAMILIES = ["D7_I10", "D7_I210", "D7_I405", "D12_I5"]

WORK2 = os.path.dirname(os.path.abspath(__file__))
CACHE2 = os.path.join(WORK2, "cache")
MODEL2 = os.path.join(WORK2, "models")
OUT2 = os.path.join(WORK2, "out")


def ensure_dirs():
    for d in (CACHE2, MODEL2, OUT2):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def rolling_median_time(x: np.ndarray, w: int = 3) -> np.ndarray:
    """NaN-aware centred rolling median down the time axis (columns = links)."""
    return (pd.DataFrame(x).rolling(w, center=True, min_periods=1)
            .median().to_numpy(np.float32))


def queue_labels(speed: np.ndarray, v_cut: np.ndarray, elig: np.ndarray,
                 smooth: int = 3) -> np.ndarray:
    """Pseudo queue truth: smoothed speed <= v_cut on observed cells."""
    if smooth and smooth > 1:
        s = rolling_median_time(speed, smooth)
    else:
        s = speed
    return ((s <= v_cut[None, :]) & elig).astype(np.int8)


# --------------------------------------------------------------------------- #
# panel data
# --------------------------------------------------------------------------- #
def _ramp_counts(panel: str, links: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    path = os.path.join(traffic.network_dir(panel), "ramp_attachment_map.csv")
    on = np.zeros(len(links), np.float32)
    off = np.zeros(len(links), np.float32)
    if not os.path.exists(path):
        return on, off
    ramp = pd.read_csv(path, dtype=str)
    ramp["nearest_mainline_link_id"] = ramp["nearest_mainline_link_id"].astype(str)
    idx = {str(l): i for i, l in enumerate(links)}
    typ = ramp["ramp_type"].astype(str).str.upper()
    for lid, t in zip(ramp["nearest_mainline_link_id"], typ):
        i = idx.get(lid)
        if i is None:
            continue
        if t in ("OR", "ON"):
            on[i] += 1
        else:
            off[i] += 1
    return on, off


def panel_meta(panel: str, links: np.ndarray) -> dict:
    """Network arrays aligned to the cache link order of a panel."""
    net = traffic.load_network(panel)["links"]
    net = net.set_index("link_id")
    order = traffic.link_order(panel)
    rank_of = {str(l): i for i, l in enumerate(order)}
    n_links = len(links)
    rank = np.array([rank_of.get(str(l), -1) for l in links], dtype=np.int32)
    missing = int((rank < 0).sum())
    if missing:
        # fall back: append unknown links after the known order
        nxt = int(rank.max()) + 1
        for i in np.where(rank < 0)[0]:
            rank[i] = nxt
            nxt += 1
    rank_to_idx = np.argsort(rank, kind="stable").astype(np.int32)
    on_ramp, off_ramp = _ramp_counts(panel, links)
    free = net.reindex(links)["free_speed"].to_numpy(np.float32)
    return {
        "links": links.astype(str),
        "n_links": n_links,
        "rank": rank,                 # order_index (0 = upstream end)
        "rank_to_idx": rank_to_idx,   # map rank position -> link index
        "v_cut": net.reindex(links)["v_cut"].to_numpy(np.float32),
        "free_speed": free,
        "capacity": net.reindex(links)["capacity"].to_numpy(np.float32),
        "lanes": net.reindex(links)["lanes"].to_numpy(np.float32),
        "k_jam": net.reindex(links)["k_jam"].to_numpy(np.float32),
        "critical_density": net.reindex(links)["critical_density"].to_numpy(np.float32),
        "length_km": net.reindex(links)["length_km"].to_numpy(np.float32),
        "on_ramp": on_ramp,
        "off_ramp": off_ramp,
        "rel_rank": (rank / max(n_links - 1, 1)).astype(np.float32),
    }


def load_train_panel(panel: str, smooth: int = 3) -> dict:
    """Train-split plain observations + labels for one panel.

    Arrays are (n_times, n_links).  ``speed``/``flow``/``elig`` are the released
    unmasked observation layer; ``Q`` is the smoothed queue label.
    """
    plain = traffic.load_mainline(panel, "train")["plain"]
    meta = panel_meta(panel, plain["links"].astype(str))
    S = plain["speed"]
    F = plain["flow"]
    E = plain["elig"].astype(bool)
    Q = queue_labels(S, meta["v_cut"], E, smooth=smooth)
    return {
        "panel": panel,
        "times": plain["times"],          # epoch seconds (s), int64
        "time_dt": pd.to_datetime(plain["times"], unit="s", utc=True),
        "speed": S, "flow": F, "elig": E,
        "pct": plain["pct"],
        "Q": Q,                            # smoothed label
        "Qobs": ((S <= meta["v_cut"][None, :]) & E).astype(np.int8),
        **meta,
    }


def load_masked_panel(panel: str, split: str) -> dict:
    """Masked observations for validation/private (full split time grid)."""
    m = traffic.load_mainline(panel, split)["masked"]
    meta = panel_meta(panel, m["links"].astype(str))
    return {
        "panel": panel, "split": split,
        "times": m["times"],
        "time_dt": pd.to_datetime(m["times"], unit="s", utc=True),
        "speed": m["speed"], "flow": m["flow"], "elig": m["elig"].astype(bool),
        "pct": m["pct"], **meta,
    }


# --------------------------------------------------------------------------- #
# window candidates
# --------------------------------------------------------------------------- #
def candidate_table(d: dict) -> pd.DataFrame:
    """Per-origin candidate statistics for one panel (train split)."""
    T = len(d["times"])
    L = d["n_links"]
    QO = d["Qobs"]
    E = d["elig"].astype(np.int32)
    cs = np.zeros((T + 1, L), dtype=np.int32)
    cs[1:] = np.cumsum(QO, axis=0)
    ce = np.zeros((T + 1, L), dtype=np.int32)
    ce[1:] = np.cumsum(E, axis=0)
    ti = np.arange(HIST + 1, T - HORIZON - 1)
    hc = cs[ti] - cs[ti - HIST]
    nq = hc.sum(axis=1)
    maxl = hc.max(axis=1)
    cov = (ce[ti] - ce[ti - HIST]).sum(axis=1) / float(HIST * L)
    nfut = (cs[ti + HORIZON + 1] - cs[ti + 1]).sum(axis=1)
    ongoing = (nq >= 2) & (maxl >= 2)
    ok = (cov >= MIN_HIST_COVERAGE) & (nfut >= 1)
    return pd.DataFrame({
        "ti": ti,
        "time_dt": d["time_dt"][ti],
        "nq_hist": nq, "maxlink_hist": maxl, "coverage": cov,
        "nfut_obs": nfut,
        "ongoing": ongoing, "ok": ok,
    })


def greedy_select(cand: pd.DataFrame, cap: int | None = None,
                  spacing_min: int = MIN_SPACING_MIN) -> pd.DataFrame:
    """Chronological greedy acceptance inside each condition group."""
    out = []
    for cond, name in ((False, "queue_onset"), (True, "queue_ongoing")):
        sub = cand[cand.ok & (cand.ongoing == cond)]
        last = None
        rows = []
        for r in sub.itertuples(index=False):
            if last is not None and (r.time_dt - last) < pd.Timedelta(minutes=spacing_min):
                continue
            rows.append(r)
            last = r.time_dt
            if cap is not None and len(rows) >= cap:
                break
        if rows:
            df = pd.DataFrame(rows)
            df["condition"] = name
            out.append(df)
        else:
            out.append(pd.DataFrame(columns=list(sub.columns) + ["condition"]))
    return pd.concat(out, ignore_index=True)


def window_ti_from_time(d: dict, time_dt) -> int:
    t = pd.Timestamp(time_dt)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    epoch = int(t.timestamp())
    times = d["times"]
    idx = int(np.searchsorted(times, epoch))
    if idx >= len(times) or int(times[idx]) != epoch:
        raise KeyError(f"origin {time_dt} not on the panel grid")
    return idx


def released_windows() -> pd.DataFrame:
    w = traffic.load_window_index().copy()
    w["forecast_origin"] = pd.to_datetime(w.forecast_origin, utc=True)
    w["history_start"] = pd.to_datetime(w.history_start, utc=True)
    w["forecast_start"] = pd.to_datetime(w.forecast_start, utc=True)
    w["forecast_end"] = pd.to_datetime(w.forecast_end, utc=True)
    return w


def released_history() -> pd.DataFrame:
    h = traffic.load_window_history().copy()
    h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
    h["link_id"] = h.link_id.astype(str)
    return h


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def iou_window(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    inter = int(np.logical_and(pred, truth).sum())
    union = int(np.logical_or(pred, truth).sum())
    return 1.0 if union == 0 else inter / union


def aggregate_iou(detail: pd.DataFrame) -> dict:
    """detail needs columns panel, condition, iou (one row per window).

    Mirrors score_task2: window -> (panel, condition) -> panel -> family -> mean.
    """
    pc = detail.groupby(["panel", "condition"]).iou.mean()
    pm = pc.groupby(level=0).mean()
    fam = pm.groupby([FAMILY[p] for p in pm.index]).mean()
    return {
        "overall": float(fam.mean()),
        "onset": float(detail[detail.condition == "queue_onset"].iou.mean()),
        "ongoing": float(detail[detail.condition == "queue_ongoing"].iou.mean()),
        "per_panel": pm.to_dict(),
        "per_panel_condition": pc.to_dict(),
        "per_family": fam.to_dict(),
    }


def write_submission(rows: list[pd.DataFrame], path: str) -> str:
    out = pd.concat(rows, ignore_index=True)
    out = out[["window_id", "timestamp", "link_id", "queue_pred"]]
    out["queue_pred"] = out["queue_pred"].astype(np.int8)
    out.to_csv(path, index=False)
    return path
