"""Compact U-Net over daily corridor images for the Task-1 masked-state residuals.

Model class in the team solution: the shipped Task-1 deliverable is a LightGBM
residual model on an interp/smoother base (`work/task1/gbm`, holdout S_state
0.929-0.946 on four panels).  A known strong public recipe blends such a tree
model with a convolutional image model at w=0.7/0.3 (slightly worse local RMSE,
better public LB because the Task-3 density penalty prefers smoother fields).

Design of this module (everything under work/task1/unet, nothing else is touched):

  * one image per calendar day and panel: [C, 288 time-of-day, S links]
  * C channels are all built from the split's own MASKED view (the published
    input) plus the same base reconstruction the GBM uses (temporal linear
    interpolation blended with the spatial-neighbour mean);
  * the network predicts the residual against that base, in *score units*:
        r_s = (v - base_s) / 25            (25 km/h is the S_speed denominator)
        r_q = (q - base_q) / (600 * lanes) (600 vph/lane is the S_flow denom.)
    so the per-cell loss 0.54*r_s^2 + 0.46*r_q^2 IS 1 - S_state per cell, and
    the training objective is exactly the official Task-1 score;
  * labels only exist at the published template rows, which are exactly the
    cells the organizer blanked out (verified: 0% of template rows are observed
    in the masked view), so the loss is masked to those cells.

Caches (`cache_<PANEL>_<tag>.npz`, float16 where harmless) hold the two masked
fields, the base reconstruction, a smoothed observation-density channel and the
per-template-row label arrays.  One panel at a time, peak RSS ~0.5-0.9 GB.
"""
from __future__ import annotations

import gc
import os
import sys
import time

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
HERE = os.path.join(ROOT, "work", "task1", "unet")
for _p in (os.path.join(ROOT, "work", "lib"), os.path.join(ROOT, "work", "task1", "gbm"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import traffic  # noqa: E402
from t1lib import (day_index, dow_index, link_perm, smooth_field,  # noqa: E402
                   temporal_interp, tod_index)

N_TOD = 288
HOLDOUT_DAYS = 60


# --------------------------------------------------------------------------- #
# base reconstruction (same recipe as t1gbm.FeatureBuilder, full fields)
# --------------------------------------------------------------------------- #
def spatial_nb_field(arr: np.ndarray, order_p: np.ndarray, inv: np.ndarray,
                     dists=(-2, -1, 1, 2), chunk: int = 8192) -> np.ndarray:
    """Mean of the observed values at spatial ranks +-1, +-2 (milepost order)."""
    T, S = arr.shape
    out = np.full((T, S), np.nan, np.float32)
    for lo in range(0, T, chunk):
        hi = min(T, lo + chunk)
        X = np.ascontiguousarray(arr[lo:hi][:, order_p], dtype=np.float32)
        acc = np.zeros_like(X)
        cnt = np.zeros_like(X)
        for d in dists:
            Y = np.full_like(X, np.nan)
            if d > 0:
                Y[:, :-d] = X[:, d:]
            else:
                Y[:, -d:] = X[:, :d]
            m = np.isfinite(Y)
            acc += np.where(m, Y, 0.0)
            cnt += m
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
        out[lo:hi] = mean[:, inv]
        del X, acc, cnt
    return out


def day_mean_field(arr: np.ndarray) -> np.ndarray:
    """(n_days, S) nan-mean of a (T, S) field on the 288-step day grid."""
    T, S = arr.shape
    return np.nanmean(arr.reshape(T // N_TOD, N_TOD, S), axis=1).astype(np.float32)


def base_fields(ms: np.ndarray, mq: np.ndarray, order_p: np.ndarray,
                inv: np.ndarray, free_speed: np.ndarray,
                capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Speed/flow base reconstruction from the masked view (GBM base recipe)."""
    s_intr = temporal_interp(ms)
    q_intr = temporal_interp(mq)
    s_nb = spatial_nb_field(ms, order_p, inv)
    q_nb = spatial_nb_field(mq, order_p, inv)
    with np.errstate(invalid="ignore"):
        base_s = np.where(np.isfinite(s_nb), 0.9 * s_intr + 0.1 * s_nb, s_intr)
        anchor_q = np.where(np.isfinite(q_nb), 0.9 * q_intr + 0.1 * q_nb, q_intr)
    del s_intr, q_intr, s_nb, q_nb
    gc.collect()
    q_sm = smooth_field(mq, 1.5, 1.0, order=1, order_p=order_p, inv=inv)
    with np.errstate(invalid="ignore"):
        base_q = np.where(np.isfinite(q_sm), 0.5 * q_sm + 0.5 * anchor_q, anchor_q)
    del q_sm, anchor_q
    gc.collect()

    # fallbacks, all from the masked view: day mean -> link mean -> free-flow
    dsm = day_mean_field(ms)
    dqm = day_mean_field(mq)
    T, S = ms.shape
    n_days = T // N_TOD
    ds_flat = np.where(np.isfinite(dsm), dsm, np.nan)
    with np.errstate(all="ignore"):
        link_s = np.nanmean(ds_flat, axis=0)
        link_q = np.nanmean(dqm, axis=0)
    link_s = np.where(np.isfinite(link_s), link_s, 0.6 * free_speed)
    link_q = np.where(np.isfinite(link_q), link_q, 0.3 * capacity)
    dsf = np.where(np.isfinite(dsm), dsm, link_s[None, :])
    dqf = np.where(np.isfinite(dqm), dqm, link_q[None, :])
    bs = base_s.reshape(n_days, N_TOD, S)
    bq = base_q.reshape(n_days, N_TOD, S)
    for d in range(n_days):
        bs[d] = np.where(np.isfinite(bs[d]), bs[d], dsf[d][None, :])
        bq[d] = np.where(np.isfinite(bq[d]), bq[d], dqf[d][None, :])
    base_s = bs.reshape(T, S).astype(np.float32)
    base_q = bq.reshape(T, S).astype(np.float32)
    del bs, bq, dsm, dqm, dsf, dqf
    gc.collect()
    return base_s, base_q


def clip_bounds(plain: dict, fit_day_t: np.ndarray):
    """Per-link p0.5/p99.5 speed and p99.5 flow from the plain layer of fit days."""
    with np.errstate(all="ignore"):
        lo_s = np.nanpercentile(plain["speed"][fit_day_t], 0.5, axis=0).astype(np.float32)
        hi_s = np.nanpercentile(plain["speed"][fit_day_t], 99.5, axis=0).astype(np.float32)
        hi_q = np.nanpercentile(plain["flow"][fit_day_t], 99.5, axis=0).astype(np.float32)
    return lo_s, hi_s, hi_q


def clip_predictions(speed, flow, free_speed, lo_s, hi_s, hi_q):
    speed = np.asarray(speed, np.float64)
    flow = np.asarray(flow, np.float64)
    lo = np.maximum(lo_s.astype(np.float64) * 0.0 + 1.0, 1.0)
    hi = np.where(np.isfinite(hi_s), hi_s, free_speed * 1.1).astype(np.float64)
    speed = np.clip(np.where(np.isfinite(speed), speed, np.nan), lo, hi)
    flow = np.clip(np.where(np.isfinite(flow), flow, np.nan), 0.0,
                   np.where(np.isfinite(hi_q), hi_q, 1e9).astype(np.float64))
    return speed, flow


# --------------------------------------------------------------------------- #
# per-panel cache
# --------------------------------------------------------------------------- #
def cache_path(panel: str, tag: str) -> str:
    return os.path.join(HERE, f"cache_{panel}_{tag}.npz")


def _tpl_arrays(panel: str, split: str, S: int):
    tg = traffic.load_targets(panel, split)
    t_idx = tg["t_idx"].astype(np.int64)
    s_idx = tg["s_idx"].astype(np.int64)
    reg = tg["regime"].astype(np.int8) + 1
    return t_idx, s_idx, reg


def build_cache(panel: str, split: str = "train", holdout_days: int = HOLDOUT_DAYS,
                tag: str | None = None, bounds=None, force: bool = False) -> str:
    """Build (or reuse) the cache for one panel and split.

    split='train'  : cached masked view, base, per-template-row labels, clip bounds
                     computed from the plain layer of the days < n_days-holdout_days
                     (holdout=0 -> all days, deployment).
    split='validation'/'private': masked view + base from that split only, template
                     rows, no labels; clip bounds are passed in from the train cache.
    """
    if tag is None:
        tag = ("train_hold%d" % holdout_days) if split == "train" else split
    out = cache_path(panel, tag)
    if os.path.exists(out) and not force:
        return out
    t0 = time.time()
    masked = traffic._load_or_build(panel, split, "masked")
    times = masked["times"].astype(np.int64)
    ms = masked["speed"]
    mq = masked["flow"]
    T, S = ms.shape
    days = day_index(times)
    order_p, inv = link_perm(masked)
    static = {k: np.asarray(masked[k], np.float32) for k in
              ("lanes", "free_speed", "capacity", "k_jam", "milepost", "length_km")}
    rank = np.empty(S, np.float32)
    rank[order_p] = np.arange(S, dtype=np.float32)

    if split == "train":
        plain = traffic._load_or_build(panel, "train", "plain")
        n_days = int(days.max()) + 1
        day_lo = n_days - holdout_days if holdout_days > 0 else n_days
        fit_day_t = days < day_lo
        lo_s, hi_s, hi_q = clip_bounds(plain, fit_day_t)
        tpl_t, tpl_s, tpl_reg = _tpl_arrays(panel, "train", S)
        true_s = plain["speed"][tpl_t, tpl_s].astype(np.float32)
        true_q = plain["flow"][tpl_t, tpl_s].astype(np.float32)
        finite = np.isfinite(true_s) & np.isfinite(true_q)
        del plain
        gc.collect()
        tpl_t, tpl_s, tpl_reg = tpl_t[finite], tpl_s[finite], tpl_reg[finite]
        true_s, true_q = true_s[finite], true_q[finite]
        n_lab = len(tpl_t)
    else:
        if bounds is None:
            raise ValueError("bounds required for validation/private caches")
        lo_s, hi_s, hi_q = bounds
        tpl_t, tpl_s, tpl_reg = _tpl_arrays(panel, split, S)
        true_s = true_q = None
        n_lab = len(tpl_t)

    base_s, base_q = base_fields(ms, mq, order_p, inv,
                                 static["free_speed"], static["capacity"])
    obs = np.isfinite(ms)
    obsden = smooth_field(obs.astype(np.float32), 3.0, 1.0, order=0,
                          order_p=order_p, inv=inv)
    with np.errstate(all="ignore"):
        base_k = (base_q / np.maximum(base_s, 1.0) / np.maximum(static["k_jam"], 1e-3)).astype(np.float32)

    arrs = dict(
        times=times,
        ms=np.where(obs, ms, 0.0).astype(np.float16),
        obs_s=obs.astype(np.float16),
        mq=np.where(np.isfinite(mq), mq, 0.0).astype(np.float16),
        obs_q=np.isfinite(mq).astype(np.float16),
        pct=masked["pct"],
        base_s=np.clip(base_s, 0.0, 200.0).astype(np.float16),
        base_q=np.clip(base_q, 0.0, 1e5).astype(np.float16),
        base_k=np.clip(base_k, 0.0, 4.0).astype(np.float16),
        obsden=obsden.astype(np.float16),
        rank=rank,
        lanes=static["lanes"], free_speed=static["free_speed"],
        capacity=static["capacity"], k_jam=static["k_jam"],
        length_km=static["length_km"], milepost=static["milepost"],
        lo_s=lo_s, hi_s=hi_s, hi_q=hi_q,
        tpl_t=tpl_t.astype(np.int32), tpl_s=tpl_s.astype(np.int16),
        tpl_reg=tpl_reg.astype(np.int8),
    )
    if split == "train":
        rs = (true_s - base_s[tpl_t, tpl_s]) / 25.0
        rq = (true_q - base_q[tpl_t, tpl_s]) / (600.0 * static["lanes"][tpl_s])
        arrs["lab_rs"] = rs.astype(np.float32)
        arrs["lab_rq"] = rq.astype(np.float32)
        arrs["n_lab"] = np.array([n_lab], np.int32)
    del ms, mq, base_s, base_q, base_k, obsden, masked
    gc.collect()
    np.savez_compressed(out, **arrs)
    print(f"[cache] {panel} {tag}: T={T} S={S} rows={len(tpl_t)} "
          f"{time.time() - t0:.1f}s -> {os.path.basename(out)}", flush=True)
    return out


def load_cache(panel: str, tag: str) -> dict:
    z = np.load(cache_path(panel, tag), allow_pickle=False)
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------- #
# channels
# --------------------------------------------------------------------------- #
# (cache key, divisor key or None, extra scale)
CHANNELS = [
    ("ms", "free_speed"),      # observed speed / free speed
    ("obs_s", None),           # speed observed flag
    ("mq", "capacity"),        # observed flow / capacity
    ("obs_q", None),
    ("pct", None),             # coverage percent / 100
    ("base_s", "free_speed"),  # base speed / free speed
    ("base_q", "capacity"),    # base flow / capacity
    ("base_k", "k_jam"),       # base density / jam density
    ("obsden", None),          # local observed-cell density
]
STATIC_CHANNELS = ["free_speed/130", "capacity/2400", "lanes/6", "rank_norm", "length_km/2"]
N_CH = len(CHANNELS) + len(STATIC_CHANNELS) + 2  # + tod sin/cos


def day_image(c: dict, d: int, tod_sin: np.ndarray, tod_cos: np.ndarray) -> np.ndarray:
    """[C, 288, S] float32 image for one calendar day."""
    S = c["lanes"].shape[0]
    img = np.empty((N_CH, N_TOD, S), np.float32)
    sl = slice(d * N_TOD, (d + 1) * N_TOD)
    i = 0
    for key, den in CHANNELS:
        v = np.asarray(c[key][sl], np.float32)
        if key == "pct":
            v = v * (1.0 / 100.0)
        elif den is not None:
            v = v / np.maximum(np.asarray(c[den], np.float32), 1e-3)
        img[i] = v
        i += 1
    img[i] = np.asarray(c["free_speed"], np.float32) / 130.0; i += 1
    img[i] = np.asarray(c["capacity"], np.float32) / 2400.0; i += 1
    img[i] = np.asarray(c["lanes"], np.float32) / 6.0; i += 1
    img[i] = np.asarray(c["rank"], np.float32) / max(S - 1, 1); i += 1
    img[i] = np.asarray(c["length_km"], np.float32) / 2.0; i += 1
    img[i] = tod_sin[:, None]; i += 1
    img[i] = tod_cos[:, None]; i += 1
    return img


def make_batch(c: dict, days, tod_sin, tod_cos, W: int) -> np.ndarray:
    S = c["lanes"].shape[0]
    out = np.zeros((len(days), N_CH, N_TOD, W), np.float32)
    for b, d in enumerate(days):
        out[b, :, :, :S] = day_image(c, int(d), tod_sin, tod_cos)
    return out


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_model(widths=(24, 48, 96, 128), cin: int = N_CH, cout: int = 2,
                groups: int = 8):
    torch, nn = _torch()

    class Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1, bias=False),
                nn.GroupNorm(min(groups, b), b),
                nn.SiLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1, bias=False),
                nn.GroupNorm(min(groups, b), b),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            chs = list(widths)
            self.enc = nn.ModuleList()
            a = cin
            for b in chs:
                self.enc.append(Block(a, b))
                a = b
            self.pool = nn.MaxPool2d(2)
            self.mid = Block(chs[-1], chs[-1])
            self.up = nn.ModuleList()
            self.dec = nn.ModuleList()
            a = chs[-1]
            for b in reversed(chs[:-1]):
                self.up.append(nn.ConvTranspose2d(a, b, 2, stride=2))
                self.dec.append(Block(2 * b, b))
                a = b
            self.head = nn.Conv2d(chs[0], cout, 1)

        def forward(self, x):
            skips = []
            for i, blk in enumerate(self.enc):
                x = blk(x)
                if i < len(self.enc) - 1:
                    skips.append(x)
                    x = self.pool(x)
            x = self.mid(x)
            for up, dec, sk in zip(self.up, self.dec, reversed(skips)):
                x = up(x)
                x = dec(torch.cat([x, sk], dim=1))
            return self.head(x)

    return UNet()


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# training / inference
# --------------------------------------------------------------------------- #
def _gather_rows(c: dict, rows: np.ndarray):
    """(tod, s, rs, rq) numpy arrays for the given template row indices."""
    return (c["tpl_t"][rows] % N_TOD, c["tpl_s"][rows],
            c["lab_rs"][rows], c["lab_rq"][rows])


def row_day(c: dict, rows: np.ndarray) -> np.ndarray:
    return (c["tpl_t"][rows] // N_TOD).astype(np.int64)


def _flog(msg):
    print(msg, flush=True)


def train_panel(c: dict, panel: str, fit_days: np.ndarray, val_days: np.ndarray,
                epochs: int = 50, batch: int = 8, lr: float = 2e-3,
                weight_decay: float = 1e-4, seed: int = 42, widths=(24, 48, 96, 128),
                device: str | None = None, log=_flog, log_every: int = 5,
                val_rows_cap: int = 400000):
    """Train one U-Net for one panel on `fit_days`; early stop on `val_days` loss."""
    torch, nn = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)
    S = c["lanes"].shape[0]
    W = int(np.ceil(S / 8.0) * 8)
    tod = np.arange(N_TOD, dtype=np.float32) * (2.0 * np.pi / N_TOD)
    tod_sin, tod_cos = np.sin(tod), np.cos(tod)

    # per-day template row lists (one stable sort, not a scan per day)
    d_rows = row_day(c, np.arange(len(c["tpl_t"])))
    n_days = int(d_rows.max()) + 1
    order_d = np.argsort(d_rows, kind="stable")
    bounds = np.searchsorted(d_rows[order_d], np.arange(n_days + 1))
    rows_by_day = {int(d): order_d[bounds[d]:bounds[d + 1]] for d in range(n_days)
                   if bounds[d + 1] > bounds[d]}
    fit_days = [int(d) for d in fit_days if int(d) in rows_by_day]
    val_days = [int(d) for d in val_days if int(d) in rows_by_day]

    vd = val_days

    model = build_model(widths).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, int(np.ceil(len(fit_days) / batch)))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * steps_per_epoch),
        pct_start=0.15, div_factor=10.0, final_div_factor=100.0)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    w_s, w_q = 0.54, 0.46
    best = (1e9, -1, None)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(fit_days))
        tot, nb = 0.0, 0
        t_ep = time.time()
        for bi in range(0, len(order), batch):
            days = [fit_days[i] for i in order[bi:bi + batch]]
            x = torch.from_numpy(make_batch(c, days, tod_sin, tod_cos, W)).to(device)
            rows = np.concatenate([rows_by_day[d] for d in days])
            t_, s_, rs, rq = _gather_rows(c, rows)
            b_ix = np.repeat(np.arange(len(days)),
                             [len(rows_by_day[d]) for d in days]).astype(np.int64)
            t_i = torch.from_numpy(t_).long().to(device)
            s_i = torch.from_numpy(s_).long().to(device)
            b_i = torch.from_numpy(b_ix).to(device)
            y_rs = torch.from_numpy(rs).to(device)
            y_rq = torch.from_numpy(rq).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                p_s = out[b_i, 0, t_i, s_i]
                p_q = out[b_i, 1, t_i, s_i]
                loss = w_s * ((p_s - y_rs) ** 2).mean() + w_q * ((p_q - y_rq) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        tr = tot / max(nb, 1)

        vl = float("nan")
        if vd and ((ep + 1) % log_every == 0 or ep == epochs - 1):
            vl = eval_loss(model, c, vd, rows_by_day, tod_sin, tod_cos, W,
                           device, batch)
            if vl < best[0]:
                best = (vl, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        hist.append({"epoch": ep + 1, "train_loss": tr, "val_loss": vl,
                     "sec": time.time() - t_ep})
        if (ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1:
            log(f"    ep {ep + 1:3d}/{epochs} train {tr:.5f} val {vl:.5f} "
                f"({time.time() - t_ep:.1f}s)")
    if best[2] is not None:
        model.load_state_dict(best[2])
        log(f"    best val {best[0]:.5f} @ep {best[1] + 1}")
    return model, {"W": W, "S": S, "best_val": best[0], "best_epoch": best[1] + 1,
                   "hist": hist, "widths": list(widths), "device": device}


def eval_loss(model, c, days, rows_by_day, tod_sin, tod_cos, W, device, batch=8):
    torch, _ = _torch()
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for bi in range(0, len(days), batch):
            dd = days[bi:bi + batch]
            x = torch.from_numpy(make_batch(c, dd, tod_sin, tod_cos, W)).to(device)
            rows = np.concatenate([rows_by_day[d] for d in dd])
            t_, s_, rs, rq = _gather_rows(c, rows)
            b_ix = np.repeat(np.arange(len(dd)), [len(rows_by_day[d]) for d in dd]).astype(np.int64)
            out = model(x)
            p_s = out[torch.from_numpy(b_ix).to(device), 0,
                      torch.from_numpy(t_).long().to(device),
                      torch.from_numpy(s_).long().to(device)]
            p_q = out[torch.from_numpy(b_ix).to(device), 1,
                      torch.from_numpy(t_).long().to(device),
                      torch.from_numpy(s_).long().to(device)]
            e_s = (p_s - torch.from_numpy(rs).to(device)) ** 2
            e_q = (p_q - torch.from_numpy(rq).to(device)) ** 2
            tot += float(0.54 * e_s.mean() + 0.46 * e_q.mean()) * len(dd)
            n += len(dd)
    return tot / max(n, 1)


def predict_rows(model, c: dict, rows: np.ndarray, W: int, device: str,
                 batch: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Speed/flow predictions (clipped, no observed override needed: target rows
    are exactly the blanked cells) for the given template row indices."""
    torch, _ = _torch()
    model.eval()
    S = c["lanes"].shape[0]
    tod = np.arange(N_TOD, dtype=np.float32) * (2.0 * np.pi / N_TOD)
    tod_sin, tod_cos = np.sin(tod), np.cos(tod)
    rows = np.asarray(rows, dtype=np.int64)
    days_all = c["tpl_t"][rows] // N_TOD
    uniq = np.unique(days_all)
    pred = np.full(len(rows), np.nan, np.float64)
    predq = np.full(len(rows), np.nan, np.float64)
    with torch.no_grad():
        for bi in range(0, len(uniq), batch):
            dd = uniq[bi:bi + batch]
            x = torch.from_numpy(make_batch(c, dd, tod_sin, tod_cos, W)).to(device)
            out = model(x).float().cpu().numpy()
            for k, d in enumerate(dd):
                m = days_all == d
                t_i = (c["tpl_t"][rows[m]] % N_TOD).astype(np.int64)
                s_i = c["tpl_s"][rows[m]].astype(np.int64)
                pred[m] = out[k, 0, t_i, s_i]
                predq[m] = out[k, 1, t_i, s_i]
            del x, out
    bs = c["base_s"][c["tpl_t"][rows], c["tpl_s"][rows]].astype(np.float64)
    bq = c["base_q"][c["tpl_t"][rows], c["tpl_s"][rows]].astype(np.float64)
    lanes = c["lanes"][c["tpl_s"][rows]].astype(np.float64)
    v = bs + 25.0 * pred
    q = bq + 600.0 * lanes * predq
    ss = c["tpl_s"][rows]
    v, q = clip_predictions(v, q, c["free_speed"][ss].astype(np.float64),
                            c["lo_s"][ss], c["hi_s"][ss], c["hi_q"][ss])
    return v, q


def truth_rows(c: dict, rows: np.ndarray):
    """True speed/flow and (lanes, regime) for template rows (train cache only)."""
    t = c["tpl_t"][rows]
    s = c["tpl_s"][rows]
    bs = c["base_s"][t, s].astype(np.float64)
    bq = c["base_q"][t, s].astype(np.float64)
    lanes = c["lanes"][s].astype(np.float64)
    v = bs + 25.0 * c["lab_rs"][rows].astype(np.float64)
    q = bq + 600.0 * lanes * c["lab_rq"][rows].astype(np.float64)
    return v, q, lanes, c["tpl_reg"][rows].astype(np.int64)
