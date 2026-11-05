"""Month-adaptivity diagnostics for Task 2.

Part A (structural): does each split's own observed queue structure (per-link
queue frequency from its masked observation layer; window-gap cells are NaN and
therefore excluded automatically) match the train climatology that the v3 model
relies on?  Train month-to-month variation is measured as the reference scale.

Part B (leave-one-month-out on February): a causal month-to-date logit offset is
added to the frozen v3 probabilities (weights trained only on pre-January
windows) and the adaptation weight is swept.  The target month is unseen by the
model weights; the climatology comes only from observed data of that month
before each window's origin day.  This mirrors the validation-month setting.

    python month_adapt.py
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
import features as FT
import static_model as SM
import gbm_predict as GP

import lightgbm as lgb

CUT = pd.Timestamp("2031-02-01", tz="UTC")


# --------------------------------------------------------------------------- #
def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def link_queue_rate_from_frame(speed, elig, v_cut, mask=None):
    """Per-link observed queue frequency and observed count."""
    if mask is not None:
        speed = speed[mask]
        elig = elig[mask]
    q = np.isfinite(speed) & (speed <= v_cut[None, :])
    n_q = q.sum(axis=0).astype(np.float64)
    n_o = np.isfinite(speed).sum(axis=0).astype(np.float64)
    return n_q, n_o


def part_a():
    print("=== Part A: observed per-link queue structure per split/month ===")
    rows = []
    for panel in C.PANELS:
        dtr = C.load_train_panel(panel)
        months = np.asarray(pd.DatetimeIndex(dtr["time_dt"]).to_period("M").astype(str))
        uniq = sorted(set(months.tolist()))
        month_rate = {}
        for m in uniq:
            sel = months == m
            if sel.sum() < 500:
                continue
            n_q, n_o = link_queue_rate_from_frame(dtr["speed"][sel], dtr["elig"][sel],
                                                 dtr["v_cut"])
            month_rate[m] = n_q / np.maximum(n_o, 1.0)
        last = uniq[-4:]
        ref = np.mean([month_rate[m] for m in last if m in month_rate], axis=0)
        pairs = [spearman(month_rate[last[i]], month_rate[last[j]])
                 for i in range(len(last)) for j in range(i + 1, len(last))
                 if last[i] in month_rate and last[j] in month_rate]
        row = {"panel": panel,
               "train_month_spearman_min": float(np.min(pairs)),
               "train_month_spearman_mean": float(np.mean(pairs)),
               "train_month_spearman_max": float(np.max(pairs))}
        for split in ("validation", "private"):
            dm = C.load_masked_panel(panel, split)
            # exclude cells inside any released window's history+horizon span
            w = C.released_windows()
            w = w[(w.panel == panel) & (w.split == split)]
            keep = np.ones(len(dm["time_dt"]), bool)
            for r in w.itertuples(index=False):
                keep &= ~((dm["time_dt"] >= r.history_start - pd.Timedelta(minutes=60)) &
                          (dm["time_dt"] <= r.forecast_start + pd.Timedelta(minutes=180)))
            n_q, n_o = link_queue_rate_from_frame(dm["speed"], dm["elig"],
                                                 dm["v_cut"], keep)
            rate = n_q / np.maximum(n_o, 1.0)
            row[f"sp_{split}"] = spearman(ref, rate)
            top_tr = set(np.argsort(-ref)[:8].tolist())
            top_sp = set(np.argsort(-rate)[:8].tolist())
            row[f"top8_overlap_{split}"] = len(top_tr & top_sp) / 8.0
            row[f"obs_frac_{split}"] = float(np.isfinite(dm["speed"]).mean())
            del dm
            gc.collect()
        rows.append(row)
        print(f"{panel}: train-month spearman {row['train_month_spearman_min']:.3f}-"
              f"{row['train_month_spearman_max']:.3f} (mean {row['train_month_spearman_mean']:.3f}) | "
              f"val sp {row['sp_validation']:.3f} top8 {row['top8_overlap_validation']:.2f} | "
              f"priv sp {row['sp_private']:.3f} top8 {row['top8_overlap_private']:.2f}")
        del dtr
        gc.collect()
    df = pd.DataFrame(rows).set_index("panel")
    df.to_csv(os.path.join(C.OUT2, "month_structure.csv"))
    print("\nmean over panels:", {c: round(float(df[c].mean()), 3) for c in df.columns})
    return df


# --------------------------------------------------------------------------- #
def daily_aggregates(masked: dict):
    """Per (day, slot) queue counts and observation counts from the masked layer."""
    td = pd.DatetimeIndex(masked["time_dt"])
    days = td.normalize()
    uniq_days = np.sort(days.unique().to_numpy(dtype="datetime64[ns]"))
    day_idx = np.searchsorted(uniq_days, days.to_numpy(dtype="datetime64[ns]"))
    slots = ((td.hour * 60 + td.minute) // 5).to_numpy()
    s = masked["speed"]
    q = (np.isfinite(s) & (s <= masked["v_cut"][None, :])).astype(np.float32)
    obs = np.isfinite(s).astype(np.float32)
    L = s.shape[1]
    n_d, n_s = len(uniq_days), 288
    qsum = np.zeros((n_d, n_s, L), np.float32)
    osum = np.zeros((n_d, n_s, L), np.float32)
    np.add.at(qsum, (day_idx, slots), q)
    np.add.at(osum, (day_idx, slots), obs)
    return qsum, osum, day_idx, uniq_days, day_idx


def month_tod_rate(qsum, osum, target_day, target_slots, L):
    """Causal per-link queue rate from days strictly before ``target_day``.

    Returns (rate, avail): rate is Laplace-smoothed, avail marks links where at
    least 5 observed cells exist at that time-of-day slot.
    """
    rate = np.full((C.HORIZON, L), 0.5, np.float32)
    avail = np.zeros((C.HORIZON, L), bool)
    if target_day <= 0:
        return rate, avail
    qs = qsum[:target_day].sum(axis=0)  # (288, L)
    os_ = osum[:target_day].sum(axis=0)
    for k in range(C.HORIZON):
        sl = int(target_slots[k]) % 288
        r = (qs[sl] + 1.0) / (os_[sl] + 2.0)  # Laplace
        rate[k] = r
        avail[k] = os_[sl] >= 5
    return rate, avail


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def win_iou(pred6, truth6, elig6):
    p = pred6.astype(bool) & elig6
    t = truth6.astype(bool)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum()) / float(union)


def part_b(lam_grid=(0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2)):
    print("\n=== Part B: causal month-to-date logit offset, Feb LOMO ===")
    meta = GP.load_meta("v3")
    idx = [FT.FEATURES.index(n) for n in meta["features"]]
    m_on = lgb.Booster(model_file=os.path.join(C.MODEL2, "v3pre_onset.txt"))
    m_ong = lgb.Booster(model_file=os.path.join(C.MODEL2, "v3pre_ongoing.txt"))
    on_rule, ong_th = float(meta["onset_rule"][1]), float(meta["ongoing_th"])

    def onset_pred(p, L):
        pred = np.zeros((C.HORIZON, L), bool)
        pred[C.HORIZON - 1, p >= on_rule] = True
        return pred

    # per-lambda detail rows
    details = {lam: [] for lam in lam_grid}
    for panel in C.PANELS:
        d = C.load_train_panel(panel)
        cand = C.candidate_table(d)
        acc = C.greedy_select(cand).sort_values("time_dt").reset_index(drop=True)
        uniq_days_train = np.sort(
            pd.DatetimeIndex(d["time_dt"]).normalize().unique().to_numpy(dtype="datetime64[ns]"))
        all_days = pd.DatetimeIndex(d["time_dt"]).normalize().to_numpy(dtype="datetime64[ns]")
        acc["day_idx"] = np.searchsorted(uniq_days_train, all_days)[acc.ti.to_numpy()]
        cut_day = int(np.searchsorted(uniq_days_train, np.datetime64(CUT.tz_localize(None))))
        ev = acc[acc.day_idx >= cut_day]
        if len(ev) == 0:
            del d
            gc.collect()
            continue
        masked = C.load_masked_panel(panel, "train")
        qsum, osum, masked_day, uniq_days, _ = daily_aggregates(masked)
        z = np.load(os.path.join(C.MODEL2, f"static_{panel}.npz"), allow_pickle=False)
        train_rate = z["rate"].astype(np.float32)  # (288, L)
        z.close()
        L = d["n_links"]
        clim = GP.load_clim(panel)
        clim = {"onset_freq": clim["onset_freq"], "ong_freq": clim["ong_freq"],
                "rate": clim["rate"], "elig_rate": clim["elig_rate"],
                "ratio_tod": clim["ratio_tod"], "fcap_tod": clim["fcap_tod"]}
        ong_preds = []
        for r in ev.itertuples(index=False):
            ti = int(r.ti)
            tt = d["time_dt"][ti + 1:ti + 1 + C.HORIZON]
            target_slots = ((tt.hour * 60 + tt.minute) // 5).to_numpy()
            c = FT.set_window_clim(clim, d, ti)
            ong_flag = 1 if r.condition == "queue_ongoing" else 0
            X = FT.window_features(d, ti, c, C.PANELS.index(panel), ong_flag)
            X = X[:, idx]
            truth = d["Q"][ti + 1:ti + 1 + C.HORIZON]
            elig = d["elig"][ti + 1:ti + 1 + C.HORIZON]
            # month-to-date climatology for this window
            masked_td = pd.Timestamp(tt[0]).normalize()
            pos = int(np.searchsorted(uniq_days, np.datetime64(masked_td.tz_localize(None))))
            mr, avail = month_tod_rate(qsum, osum, pos, target_slots, L)  # (6,L)
            tr = np.stack([train_rate[int(s) % 288] for s in target_slots], axis=0)
            delta = np.where(avail, logit(mr) - logit(tr), 0.0).astype(np.float32)
            if ong_flag:
                p = m_ong.predict(X).astype(np.float32).reshape(C.HORIZON, L)
                for lam in lam_grid:
                    pp = sigmoid(logit(p) + lam * delta) if lam else p
                    details[lam].append({"panel": panel, "condition": "queue_ongoing",
                                         "iou": win_iou(pp >= ong_th, truth, elig)})
            else:
                p = m_on.predict(X).astype(np.float32)[(C.HORIZON - 1) * L:]
                for lam in lam_grid:
                    pp = sigmoid(logit(p) + lam * delta[C.HORIZON - 1]) if lam else p
                    details[lam].append({"panel": panel, "condition": "queue_onset",
                                         "iou": win_iou(onset_pred(pp, L), truth, elig)})
        del d, masked, qsum, osum
        gc.collect()

    results = {}
    for lam, rows in details.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        agg = C.aggregate_iou(df)
        results[lam] = agg
        print(f"lambda {lam:.2f}: overall {agg['overall']:.4f} | onset {agg['onset']:.4f} "
              f"| ongoing {agg['ongoing']:.4f}")
    with open(os.path.join(C.OUT2, "month_adapt.json"), "w") as fh:
        json.dump({str(k): {"overall": v["overall"], "onset": v["onset"],
                            "ongoing": v["ongoing"],
                            "per_panel": {p: float(x) for p, x in v["per_panel"].items()}}
                   for k, v in results.items()}, fh, indent=1)
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()
    C.ensure_dirs()
    if args.part in ("a", "both"):
        part_a()
    if args.part in ("b", "both"):
        part_b()


if __name__ == "__main__":
    main()
