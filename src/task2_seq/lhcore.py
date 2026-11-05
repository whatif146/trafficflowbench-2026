"""Long-history Task 2 core: causal history assembly + feature extraction.

Signal exploited here (the part v3 does not use): the *whole released
mainline_states_masked layer of the split* contains observations with
timestamp <= T, so every window can look back 3 h instead of the 60 min of
window_history, and the row at the origin T itself is allowed too.

Design rules (strict causality / consistency):
  * history rows 24..35 (offsets -60..-5 min) come from the fully observed layer:
    window_history for released windows, the train plain layer for arbitrary
    train origins (both are the same observed layer, so the feature function is
    identical for fitting and for the scored splits).
  * history rows 0..23 (offsets -180..-65 min) and row 36 (the origin T) come
    from the split's masked layer, aggregated NaN-aware (no imputation model is
    used, so nothing computed from a timestamp > T can leak in).  The masked
    layer keeps the 12 window_history steps unmasked, but those are overlaid
    from the observed layer anyway.
  * no cell with timestamp > T is ever read (the horizon cells are blanked in
    the release; row T is the last row read).
"""
from __future__ import annotations

import gc
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
WORK = os.path.join(ROOT, "work")
for _p in (os.path.join(WORK, "lib"), os.path.join(WORK, "task2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import traffic  # noqa: E402
import t2core as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "out")
MODELS = os.path.join(HERE, "models")
for _d in (CACHE, OUT, MODELS):
    os.makedirs(_d, exist_ok=True)

PANELS = C.PANELS
FAMILY = C.FAMILY
FAMILIES = C.FAMILIES
HORIZON = C.HORIZON          # 6
HIST = C.HIST               # 12 released history steps (offsets -60..-5)
H_PRE = 36                  # extra masked steps before the released history
N_HIST = H_PRE + 1          # rows 0..36, row 36 = origin T (offset 0)

TUNE_START = pd.Timestamp("2031-01-01", tz="UTC")
CUT = pd.Timestamp("2031-02-01", tz="UTC")
JUNE_END = pd.Timestamp("2030-07-01", tz="UTC")

STATIC_DIR = os.path.join(WORK, "task2", "models")


# --------------------------------------------------------------------------- #
# static network / climatology
# --------------------------------------------------------------------------- #
_STATIC: dict[str, dict] = {}


def static_panel(panel: str) -> dict:
    """Frozen per-panel network + climatology arrays (from the v3 artefacts)."""
    if panel in _STATIC:
        return _STATIC[panel]
    z = np.load(os.path.join(STATIC_DIR, f"static_{panel}.npz"), allow_pickle=False)
    s = {k: z[k].copy() for k in z.files}
    z.close()
    s["links"] = np.asarray(s["links"], dtype=str)
    s["n_links"] = int(len(s["links"]))
    s["rank_to_idx"] = np.asarray(s["rank_to_idx"], np.int64)
    s["v_cut"] = np.asarray(s["v_cut"], np.float32)
    s["capacity"] = np.maximum(np.asarray(s["capacity"], np.float32), 1e-3)
    s["k_jam"] = np.maximum(np.asarray(s["k_jam"], np.float32), 1e-3)
    s["lanes"] = np.maximum(np.asarray(s["lanes"], np.float32), 1.0)
    s["free_speed"] = np.asarray(s["free_speed"], np.float32)
    s["critical_density"] = (s["capacity"] / np.maximum(s["v_cut"] / 0.6, 1.0)).astype(np.float32)
    _STATIC[panel] = s
    return s


# --------------------------------------------------------------------------- #
# split observation layers
# --------------------------------------------------------------------------- #
_SPLIT: dict[tuple[str, str], dict] = {}


def split_data(panel: str, split: str, need_plain: bool = True) -> dict:
    """Load one panel/split's arrays (cached in-process, one panel at a time)."""
    key = (panel, split)
    if key in _SPLIT:
        return _SPLIT[key]
    st = static_panel(panel)
    if split == "train":
        both = traffic.load_mainline(panel, "train")
        plain, masked = both["plain"], both["masked"]
    else:
        plain, masked = None, traffic.load_mainline(panel, split)["masked"]
    assert np.array_equal(np.asarray(masked["links"], dtype=str), st["links"]), \
        f"{panel} link order mismatch between split layer and static artefacts"
    d = {"panel": panel, "split": split, "st": st, "masked": masked, "plain": plain,
         "times": np.asarray(masked["times"], np.int64)}
    if plain is not None and not need_plain:
        d["plain"] = plain
    _SPLIT[key] = d
    return d


def free_split(panel: str, split: str) -> None:
    _SPLIT.pop((panel, split), None)
    gc.collect()


# --------------------------------------------------------------------------- #
# released window history (all splits)
# --------------------------------------------------------------------------- #
_WH: pd.DataFrame | None = None


def window_history_frame() -> pd.DataFrame:
    global _WH
    if _WH is None:
        h = traffic.load_window_history().copy()
        h["timestamp"] = pd.to_datetime(h.timestamp, utc=True)
        h["link_id"] = h.link_id.astype(str)
        _WH = h
    return _WH


def window_histories(panel: str) -> dict[str, dict]:
    """window_id -> {'speed': (12, L), 'flow': (12, L)} aligned to static links.

    The released window_history is the observed layer at offsets -60..-5, the
    same function of the data for every split.
    """
    st = static_panel(panel)
    links = st["links"]
    l_index = {str(l): i for i, l in enumerate(links)}
    h = window_history_frame()
    h = h[h.link_id.isin(l_index)].copy()
    h["li"] = h.link_id.map(l_index).astype(np.int64)
    L = len(links)
    out = {}
    for wid, g in h.groupby("window_id", sort=False):
        times = np.sort(g.timestamp.unique())
        if len(times) != HIST:
            raise ValueError(f"{wid}: {len(times)} history steps")
        t_index = {t: i for i, t in enumerate(times)}
        ti = g.timestamp.map(t_index).to_numpy(np.int64)
        li = g.li.to_numpy(np.int64)
        speed = np.full((HIST, L), np.nan, np.float32)
        flow = np.full((HIST, L), np.nan, np.float32)
        speed[ti, li] = pd.to_numeric(g.speed_kmh, errors="coerce").to_numpy(np.float32)
        flow[ti, li] = pd.to_numeric(g.flow_vph, errors="coerce").to_numpy(np.float32)
        out[str(wid)] = {"speed": speed, "flow": flow,
                         "times": np.asarray([int(pd.Timestamp(t).timestamp()) for t in times],
                                             np.int64)}
    return out


# --------------------------------------------------------------------------- #
# rank-space helpers (link order is travel order: 0 = upstream)
# --------------------------------------------------------------------------- #
def shift_rank(a: np.ndarray, r2i: np.ndarray, d: int, fill=np.nan) -> np.ndarray:
    out = np.full(a.shape, fill, dtype=np.float32)
    if d == 0:
        out[:] = a
        return out
    if d > 0:
        out[r2i[d:]] = a[r2i[:-d]]
    else:
        k = -d
        out[r2i[:len(r2i) - k]] = a[r2i[k:]]
    return out


def _dist_to_queued(q: np.ndarray, r2i: np.ndarray, direction: int, cap: int = 20) -> np.ndarray:
    """Rank distance (in links) to the nearest queued link in one direction."""
    qr = (np.nan_to_num(q, nan=0.0)[r2i] > 0.5).astype(np.int32)
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


def spatial_impute(row: np.ndarray, r2i: np.ndarray, max_d: int = 5) -> np.ndarray:
    """Fill NaN cells of one same-timestamp row from the nearest observed rank
    neighbours (only that same timestamp is used -> causal)."""
    L = len(row)
    ar = row[r2i].astype(np.float64)
    obs = np.isfinite(ar)
    if obs.all():
        return row.astype(np.float32)
    pos = np.arange(L)
    fwd_idx = np.where(obs, pos, -10 ** 6)
    fwd = np.maximum.accumulate(fwd_idx)
    bwd_idx = np.where(obs, pos, 10 ** 6)
    bwd = np.minimum.accumulate(bwd_idx[::-1])[::-1]
    df = pos - fwd
    db = bwd - pos
    val = np.full(L, np.nan)
    okf = (df <= max_d)
    okb = (db <= max_d)
    val[okf] = ar[fwd[okf]]
    both = okf & okb
    val[both] = 0.5 * (ar[fwd[both]] + ar[bwd[both]])
    out_rank = val
    out = np.empty(L, np.float32)
    out[r2i] = out_rank.astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# history assembly
# --------------------------------------------------------------------------- #
def make_history(panel: str, split: str, ti: int,
                 win_hist: dict | None = None,
                 plain_hist: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(N_HIST, L) masked+observed causal history ending at the origin row ti.

    rows 0..23   offsets -180..-65 min: split masked layer, NaN where masked
    rows 24..35  offsets -60..-5 min:   observed layer (window_history or plain)
    row  36      offset 0 (T):          split masked layer (partially masked)
    """
    d = split_data(panel, split)
    st = d["st"]
    L = st["n_links"]
    m = d["masked"]
    mspeed, mflow = m["speed"], m["flow"]
    S = np.full((N_HIST, L), np.nan, np.float32)
    F = np.full((N_HIST, L), np.nan, np.float32)
    lo = ti - H_PRE
    if lo >= 0:
        S[:H_PRE] = mspeed[lo:ti]
        F[:H_PRE] = mflow[lo:ti]
    else:
        pad = -lo
        S[pad:] = mspeed[0:ti]
        F[pad:] = mflow[0:ti]
    S[H_PRE] = mspeed[ti]
    F[H_PRE] = mflow[ti]
    # overlay the fully observed recent hour
    if win_hist is not None:
        S[H_PRE - HIST:H_PRE] = win_hist["speed"]
        F[H_PRE - HIST:H_PRE] = win_hist["flow"]
    elif plain_hist is not None:
        S[H_PRE - HIST:H_PRE] = plain_hist[0]
        F[H_PRE - HIST:H_PRE] = plain_hist[1]
    return S, F


# --------------------------------------------------------------------------- #
# long-history features
# --------------------------------------------------------------------------- #
LH_FEATURES = [
    # --- current state (row T, masked layer; NaN-aware / spatially filled) ---
    "r_T", "r_T_spat", "rT_obs", "q_T", "q_T_spat", "fc_T", "fc_T_spat",
    # --- 3 h temporal levels and trends ---
    "r_med_l65", "r_med_p60", "r_med_o60", "r_min_o125", "r_med_all", "r_mean_all",
    "rT_minus_medl", "dr_lp", "dr_lo",
    "qfrac_all", "qfrac_o", "qfrac_p", "qfrac_l",
    "q_age", "t_since_q", "max_run", "n_trans",
    "fc_med_l", "fc_med_o", "fc_slope", "dens_ratio_l", "dens_ratio_o",
    "cov_all", "cov_T",
    # --- spatial state in travel order ---
    "dist_up_T", "dist_dn_T", "dist_up_30", "dist_dn_30",
    "ddist_up", "ddist_dn",
    "qsum_up5_T", "qsum_dn5_T", "qsum_up10_T", "qsum_dn10_T",
    # --- neighbourhood (upstream/downstream in rank order) ---
    "qfl_up1", "qfl_dn1", "qfl_up2", "qfl_dn2", "qfl_up3", "qfl_dn3",
    "qfl_up5", "qfl_dn5",
    "rml_up1", "rml_dn1", "rml_up2", "rml_dn2", "rml_up3", "rml_dn3",
    "qfa_up1", "qfa_dn1", "qfa_up2", "qfa_dn2", "qfa_up3", "qfa_dn3",
    "rTs_up1", "rTs_dn1", "rTs_up2", "rTs_dn2",
    # --- panel-level context ---
    "nq_T_panel", "nq_l_panel", "qfa_panel_l", "d_qfa_panel", "fc_panel_l",
    "qage_panel",
    # --- clock ---
    "lh_hour_sin", "lh_hour_cos", "lh_dow_sin", "lh_dow_cos",
    # --- causal ramp demand over the long history ---
    "on_l_cap", "on_o_cap", "on_slope_cap", "off_l_cap", "ramp_cov",
]
N_LH = len(LH_FEATURES)
LH_INDEX = {n: i for i, n in enumerate(LH_FEATURES)}

# --- jam-sensor ramp ratios (extra block, kept separate so the 80-feature v5
# feature cache keeps its shape) ----------------------------------------------
RAMP_FEATURES = [
    "rr_l", "rr_o", "rr_min", "rr_trend", "off_rr_l", "rr_up1", "rr_dn1", "rr_drop",
]
N_RAMP = len(RAMP_FEATURES)


def ramp_profile(panel: str) -> dict:
    """Train-split time-of-day ramp-flow profile (288, L), see ramp_tod.py."""
    z = np.load(os.path.join(CACHE, f"ramp_tod_{panel}.npz"), allow_pickle=False)
    out = {"on": np.asarray(z["on"], np.float32), "off": np.asarray(z["off"], np.float32)}
    z.close()
    return out


def ramp_ratio_features(times: np.ndarray, ti: int, ramp: dict, prof: dict,
                        r2i: np.ndarray) -> np.ndarray:
    """(L, N_RAMP) jam-sensor features for one origin.

    The observed attached on/off-ramp flow is divided by the ramp's own train
    time-of-day expectation.  A sustained ratio far below 1 on an on-ramp is the
    "adjacent mainline link is jammed" signature (public write-up: on-ramps drop
    to ~0.2x when the adjacent mainline jams).  Only ramp rows with
    timestamp <= T are read (offsets -60..0).
    """
    L = len(r2i)
    lo = max(0, ti - HIST)
    on = np.asarray(ramp["on_flow"][lo:ti + 1], np.float32)
    onv = np.asarray(ramp["on_valid"][lo:ti + 1], bool)
    off = np.asarray(ramp["off_flow"][lo:ti + 1], np.float32)
    offv = np.asarray(ramp["off_valid"][lo:ti + 1], bool)
    n = on.shape[0]
    if n == 0:
        return np.zeros((L, N_RAMP), np.float32)
    t = np.asarray(times[lo:ti + 1], np.int64)
    dti = pd.to_datetime(t, unit="s", utc=True)
    slots = ((dti.hour * 60 + dti.minute) // 5).to_numpy(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        ron = np.where(onv, on / prof["on"][slots], np.nan)
        roff = np.where(offv, off / prof["off"][slots], np.nan)
    l = slice(max(0, n - 13), n)
    o = slice(0, min(13, n))
    rr_l = _nmean(ron[l]).astype(np.float32)
    rr_o = _nmean(ron[o]).astype(np.float32)
    rr_min = _nmin(ron[l]).astype(np.float32)
    off_l = _nmean(roff[l]).astype(np.float32)
    up1 = shift_rank(rr_l, r2i, -1, fill=np.nan)
    dn1 = shift_rank(rr_l, r2i, 1, fill=np.nan)
    drop = np.where(np.isfinite(rr_l) & (rr_l < 0.35), 1.0, 0.0).astype(np.float32)
    out = np.stack([rr_l, rr_o, rr_min, rr_l - rr_o, off_l, up1, dn1, drop], axis=0)
    return np.ascontiguousarray(out.T)


def _nmed(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(x, axis=axis)


def _nmean(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(x, axis=axis)


def _nmin(x, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(x, axis=axis)


def _run_stats(qf: np.ndarray):
    """Consecutive queued run ending at the origin, steps since last queue, max run."""
    n_rows, L = qf.shape
    age = np.zeros(L, np.float32)
    since = np.full(L, float(n_rows), np.float32)
    max_run = np.zeros(L, np.float32)
    cur = np.zeros(L, np.float32)
    active = np.ones(L, bool)
    found = np.zeros(L, bool)
    # queue age: walk back from the origin, skipping unknown rows
    for k in range(n_rows - 1, -1, -1):
        q = qf[k]
        known = np.isfinite(q)
        isq = np.nan_to_num(q, nan=0.0) > 0.5
        # age
        upd = active & known
        age[upd & isq] += 1.0
        active &= ~(known & ~isq)
        # time since the most recent queued observation
        newly = (known & isq) & ~found
        since[newly] = float(n_rows - 1 - k)
        found |= known & isq
        # max continuous run of queued observations (unknown rows are skipped)
        cur = np.where(known, np.where(isq, cur + 1.0, 0.0), cur)
        max_run = np.maximum(max_run, cur)
    return age, since, max_run


def lh_features(panel: str, origin: pd.Timestamp, S: np.ndarray, F: np.ndarray,
                ramp: dict | None = None, ti: int | None = None) -> np.ndarray:
    """(L, N_LH) float32 long-history feature matrix for one window."""
    st = static_panel(panel)
    L = st["n_links"]
    r2i = st["rank_to_idx"]
    vc = st["v_cut"][None, :]
    cap = st["capacity"][None, :]
    kjam = st["k_jam"][None, :]
    crit = st["critical_density"][None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = S / vc
        fc = F / cap
        dens = (F / np.maximum(S, 1e-3)) / kjam
    finite = np.isfinite(S)
    qf = np.where(finite, (S <= vc).astype(np.float32), np.nan)
    qf0 = np.nan_to_num(qf, nan=0.0)

    l65 = slice(24, N_HIST)      # offsets -60..0 (incl. T)
    p60 = slice(12, 24)          # offsets -120..-65
    o60 = slice(0, 12)           # offsets -180..-125
    o125 = slice(0, 25)

    r_med_l = _nmed(r[l65])
    r_med_p = _nmed(r[p60])
    r_med_o = _nmed(r[o60])
    r_med_all = _nmed(r)
    fc_med_l = _nmed(fc[l65])
    fc_med_o = _nmed(fc[o60])
    dr_med_l = _nmean(dens[l65]) / crit[0]
    dr_med_o = _nmean(dens[o60]) / crit[0]
    qfrac_l = _nmean(qf[l65])
    qfrac_p = _nmean(qf[p60])
    qfrac_o = _nmean(qf[o60])
    qfrac_all = _nmean(qf)
    age, since, max_run = _run_stats(qf)
    dif = np.abs(np.diff(qf0, axis=0))
    known_diff = np.isfinite(qf[1:]) & np.isfinite(qf[:-1])
    n_trans = np.nansum(np.where(known_diff, dif, np.nan), axis=0)

    r_T = r[N_HIST - 1]
    q_T = qf[N_HIST - 1]
    fc_T = fc[N_HIST - 1]
    r_T_spat = spatial_impute(r_T, r2i)
    q_T_spat = spatial_impute(q_T, r2i)
    fc_T_spat = spatial_impute(fc_T, r2i)
    # fall back to the 3 h temporal levels where a long masked run leaves no
    # observed rank neighbour at the origin row
    r_T_spat = np.where(np.isfinite(r_T_spat), r_T_spat, r_med_l).astype(np.float32)
    fc_T_spat = np.where(np.isfinite(fc_T_spat), fc_T_spat, fc_med_l).astype(np.float32)
    q_T_spat = np.where(np.isfinite(q_T_spat), q_T_spat, qfrac_l).astype(np.float32)
    rT_obs = np.isfinite(r_T).astype(np.float32)
    cov_all = finite.mean(axis=0).astype(np.float32)

    qT_bool = np.nan_to_num(q_T, nan=0.0)
    q30_bool = np.nan_to_num(qf[N_HIST - 1 - 6], nan=0.0)   # offset -30 min
    dist_up_T = _dist_to_queued(qT_bool, r2i, -1)
    dist_dn_T = _dist_to_queued(qT_bool, r2i, 1)
    dist_up_30 = _dist_to_queued(q30_bool, r2i, -1)
    dist_dn_30 = _dist_to_queued(q30_bool, r2i, 1)
    qsum_up5 = sum(shift_rank(qT_bool, r2i, -d, fill=0.0) for d in range(1, 6))
    qsum_dn5 = sum(shift_rank(qT_bool, r2i, d, fill=0.0) for d in range(1, 6))
    qsum_up10 = sum(shift_rank(qT_bool, r2i, -d, fill=0.0) for d in range(1, 11))
    qsum_dn10 = sum(shift_rank(qT_bool, r2i, d, fill=0.0) for d in range(1, 11))

    # panel-level context (broadcast)
    nq_T = float(np.nansum(qT_bool))
    nq_l = float(np.nanmean(qf0[l65].sum(axis=0)))
    qfa_l_panel = float(np.nanmean(qf0[l65]))
    qfa_p_panel = float(np.nanmean(qf0[p60]))
    fc_panel_l = float(np.nanmean(fc[l65]))
    qage_panel = float(np.nanmean(age))
    tt = origin
    hour = tt.hour + tt.minute / 60.0
    dow = tt.dayofweek

    base = np.stack([
        r_T, r_T_spat, rT_obs, q_T, q_T_spat, fc_T, fc_T_spat,
        r_med_l, r_med_p, r_med_o, _nmin(r[o125]), r_med_all, _nmean(r),
        r_T - r_med_l, r_med_l - r_med_p, r_med_l - r_med_o,
        qfrac_all, qfrac_o, qfrac_p, qfrac_l,
        age, since, max_run, n_trans,
        fc_med_l, fc_med_o, fc_med_l - fc_med_o, dr_med_l, dr_med_o,
        cov_all, rT_obs.copy(),
        dist_up_T, dist_dn_T, dist_up_30, dist_dn_30,
        dist_up_T - dist_up_30, dist_dn_T - dist_dn_30,
        qsum_up5, qsum_dn5, qsum_up10, qsum_dn10,
        shift_rank(qfrac_l, r2i, -1, fill=np.nan), shift_rank(qfrac_l, r2i, 1, fill=np.nan),
        shift_rank(qfrac_l, r2i, -2, fill=np.nan), shift_rank(qfrac_l, r2i, 2, fill=np.nan),
        shift_rank(qfrac_l, r2i, -3, fill=np.nan), shift_rank(qfrac_l, r2i, 3, fill=np.nan),
        shift_rank(qfrac_l, r2i, -5, fill=np.nan), shift_rank(qfrac_l, r2i, 5, fill=np.nan),
        shift_rank(r_med_l, r2i, -1, fill=np.nan), shift_rank(r_med_l, r2i, 1, fill=np.nan),
        shift_rank(r_med_l, r2i, -2, fill=np.nan), shift_rank(r_med_l, r2i, 2, fill=np.nan),
        shift_rank(r_med_l, r2i, -3, fill=np.nan), shift_rank(r_med_l, r2i, 3, fill=np.nan),
        shift_rank(qfrac_all, r2i, -1, fill=np.nan), shift_rank(qfrac_all, r2i, 1, fill=np.nan),
        shift_rank(qfrac_all, r2i, -2, fill=np.nan), shift_rank(qfrac_all, r2i, 2, fill=np.nan),
        shift_rank(qfrac_all, r2i, -3, fill=np.nan), shift_rank(qfrac_all, r2i, 3, fill=np.nan),
        shift_rank(r_T_spat, r2i, -1, fill=np.nan), shift_rank(r_T_spat, r2i, 1, fill=np.nan),
        shift_rank(r_T_spat, r2i, -2, fill=np.nan), shift_rank(r_T_spat, r2i, 2, fill=np.nan),
        np.full(L, nq_T, np.float32), np.full(L, nq_l, np.float32),
        np.full(L, qfa_l_panel, np.float32), np.full(L, qfa_l_panel - qfa_p_panel, np.float32),
        np.full(L, fc_panel_l, np.float32), np.full(L, qage_panel, np.float32),
        np.full(L, np.sin(2 * np.pi * hour / 24.0), np.float32),
        np.full(L, np.cos(2 * np.pi * hour / 24.0), np.float32),
        np.full(L, np.sin(2 * np.pi * dow / 7.0), np.float32),
        np.full(L, np.cos(2 * np.pi * dow / 7.0), np.float32),
    ], axis=0)

    ramp_feats = np.zeros((5, L), np.float32)
    if ramp is not None and ti is not None:
        on = np.asarray(ramp["on_flow"], np.float32)
        off = np.asarray(ramp["off_flow"], np.float32)
        onv = np.asarray(ramp["on_valid"], bool)
        offv = np.asarray(ramp["off_valid"], bool)
        lo = max(0, ti - H_PRE)
        ov = onv[lo:ti + 1]
        ofv = offv[lo:ti + 1]
        with np.errstate(invalid="ignore", divide="ignore"):
            onr = np.where(ov, on[lo:ti + 1], np.nan) / cap
            offr = np.where(ofv, off[lo:ti + 1], np.nan) / cap
        n_r = onr.shape[0]
        l_idx = slice(max(0, n_r - 13), n_r)
        o_idx = slice(0, min(12, n_r))
        ramp_feats[0] = _nmean(onr[l_idx]) if n_r else 0.0
        ramp_feats[1] = _nmean(onr[o_idx]) if n_r else 0.0
        ramp_feats[2] = ramp_feats[0] - ramp_feats[1]
        ramp_feats[3] = _nmean(offr[l_idx]) if n_r else 0.0
        ramp_feats[4] = float(np.mean(ov | ofv)) if n_r else 0.0
    base = np.vstack([base, ramp_feats]).astype(np.float32)
    assert base.shape[0] == N_LH, (base.shape, N_LH)
    return np.ascontiguousarray(base.T)


# --------------------------------------------------------------------------- #
# v3 feature alignment (read-only use of the frozen v3 dataset cache)
# --------------------------------------------------------------------------- #
V3_META = os.path.join(WORK, "task2", "models", "v3", "gbm_meta.json")
_V3F: list[str] | None = None


def v3_features() -> list[str]:
    global _V3F
    if _V3F is None:
        import json
        with open(V3_META) as fh:
            _V3F = list(json.load(fh)["features"])
    return _V3F


ALL71: list[str] | None = None


def all71_features() -> list[str]:
    global ALL71
    if ALL71 is None:
        import features as FT
        ALL71 = list(FT.FEATURES)
    return ALL71


def v3_col_index() -> np.ndarray:
    allf = all71_features()
    return np.array([allf.index(n) for n in v3_features()], dtype=np.int64)


# --------------------------------------------------------------------------- #
# pseudo-window (selector-based) reconstruction, train split
# --------------------------------------------------------------------------- #
def day_index(d: dict):
    dates = pd.DatetimeIndex(d["time_dt"]).normalize()
    uniq = np.sort(np.asarray(dates.unique(), dtype="datetime64[ns]"))
    pos = np.searchsorted(uniq, np.asarray(dates, dtype="datetime64[ns]"))
    return pos.astype(np.int32), uniq


def train_index_arrays(panel: str, onset_stride: int = 4, ongoing_stride: int = 4,
                       cut: str = "2031-02-01") -> dict:
    """Reconstruct exactly the window list of the v3 dataset cache.

    Mirrors work/task2/build_dataset.py: greedy-accepted windows plus a
    chronological 1-in-``stride`` subsample of the valid selector candidates
    before the cut.  Returns the origin row indices (``ti``) so that the
    long-history features line up row-for-row with the cached v3 features.
    """
    st = static_panel(panel)
    d = split_data(panel, "train")
    plain = d["plain"]
    times = d["times"]
    dt = pd.to_datetime(times, unit="s", utc=True)
    vc = st["v_cut"][None, :]
    speed = plain["speed"]
    elig = plain["elig"].astype(bool)
    L = st["n_links"]
    Qobs = ((speed <= vc) & elig).astype(np.int8)
    dd = {"times": times, "time_dt": dt, "Qobs": Qobs, "elig": elig, "n_links": L}
    cand = C.candidate_table(dd)
    days, uniq = day_index(dd)
    cut_day = int(np.searchsorted(uniq, np.datetime64(pd.Timestamp(cut, tz="UTC").tz_localize(None))))
    cand["day_idx"] = days[cand.ti.to_numpy()]
    acc = C.greedy_select(cand)
    acc["day_idx"] = days[acc.ti.to_numpy()]

    onset = acc[acc.condition == "queue_onset"]
    ongoing = acc[acc.condition == "queue_ongoing"]
    pool = cand[(~cand.ongoing) & cand.ok].sort_values("ti")
    fit_pool = pool[pool.day_idx < cut_day]
    sel_ti = list(fit_pool.ti.to_numpy()[::max(onset_stride, 1)])
    chosen = set(sel_ti)
    for r in onset[onset.day_idx < cut_day].itertuples(index=False):
        if int(r.ti) not in chosen:
            chosen.add(int(r.ti))
            sel_ti.append(int(r.ti))
    sel_ti = sorted(chosen)
    on_ti = list(sel_ti)
    acc_set = set(int(t) for t in acc.ti.to_numpy())
    on_acc = [int(t) in acc_set for t in on_ti]
    for r in onset[onset.day_idx >= cut_day].itertuples(index=False):
        on_ti.append(int(r.ti))
        on_acc.append(1)
    on_ti = np.asarray(on_ti, np.int64)
    on_acc = np.asarray(on_acc, bool)

    ons_ti, ons_acc = [], []
    n_fit_ong = 0
    for r in ongoing.itertuples(index=False):
        ti = int(r.ti)
        fit = int(r.day_idx) < cut_day
        if fit:
            n_fit_ong += 1
            if n_fit_ong % max(ongoing_stride, 1):
                continue
        ons_ti.append(ti)
        ons_acc.append(1)
    ons_ti = np.asarray(ons_ti, np.int64)
    ons_acc = np.asarray(ons_acc, bool)
    del dd, Qobs, cand, acc
    return {"on_ti": on_ti, "on_acc": on_acc, "ong_ti": ons_ti, "ong_acc": ons_acc,
            "times": times, "dt": dt, "L": L}


def window_labels(panel: str, ti: int) -> tuple[np.ndarray, np.ndarray]:
    """Horizon queue truth (smoothed 3-step median, observed only) + eligibility.

    Truth is the smoothed-speed indicator over the *plain* layer (release facts:
    the grader uses the underlying state, the plain layer is the released
    train observation of it and the first agent verified the smoothing
    convention reproduces the released-window structure).
    """
    d = split_data(panel, "train")
    st = d["st"]
    plain = d["plain"]
    speed = plain["speed"][ti:ti + 8]
    s = C.rolling_median_time(speed, 3)[1:7]
    truth = ((s <= st["v_cut"][None, :]) & plain["elig"][ti + 1:ti + 7].astype(bool)).astype(np.int8)
    elig = plain["elig"][ti + 1:ti + 7].astype(bool)
    return truth, elig


# --------------------------------------------------------------------------- #
# released windows (all splits)
# --------------------------------------------------------------------------- #
def released_table() -> pd.DataFrame:
    w = C.released_windows()
    return w


def released_windows_panel(panel: str, split: str) -> pd.DataFrame:
    w = released_table()
    w = w[(w.panel == panel) & (w.split == split)].sort_values(["condition", "forecast_origin"])
    return w.reset_index(drop=True)


def origin_index(times: np.ndarray, origin: pd.Timestamp) -> int:
    t0 = int(pd.Timestamp(origin).timestamp())
    i = int(np.searchsorted(times, t0))
    if i >= len(times) or int(times[i]) != t0:
        raise KeyError(f"origin {origin} not on the grid")
    return i


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def iou(pred: np.ndarray, truth: np.ndarray, elig: np.ndarray) -> float:
    p = pred.astype(bool) & elig
    t = truth.astype(bool)
    union = int(np.logical_or(p, t).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / float(union)


def aggregate(detail: pd.DataFrame) -> dict:
    return C.aggregate_iou(detail)
