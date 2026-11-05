"""Honest holdout evaluation of the U-Net and its blend with the shipped GBM.

Protocol (identical to work/task1/gbm/run_eval.py): fit on train days
< n_days-60, predict the published template rows of the last 60 days from the
masked view only; score with the official Task-1 formula (t1lib.score).  The
GBM side is the shipped holdout prediction CSV (work/task1/gbm/holdout_<P>.csv),
matched row-for-row so the comparison and the blend are on exactly the same
cells.

For each panel it reports
  (a) GBM                      (b) pure U-Net
  (c) blend w*GBM + (1-w)*UNet for w in {0.6, 0.7, 0.8, 0.9}
and, when the task3 proxy imports, S_FD / S_LWR / S_physics (projected flux,
window 1, jitter 0.06, `work/task3/local_eval.py` imported read-only).

    python run_holdout.py --panels D7_I210_W D7_I405_S D7_I10_E D12_I5_N \
        --epochs 45 --tag u45 [--skip-physics]
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "unet"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "gbm"))
sys.path.insert(0, os.path.join(ROOT, "work", "task3"))

import unetlib as U  # noqa: E402
from t1lib import score, fmt  # noqa: E402

GBM_DIR = os.path.join(ROOT, "work", "task1", "gbm")
WEIGHTS = (0.6, 0.7, 0.8, 0.9)


def epoch_iso(times: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(times, unit="s", utc=True)


def station_index(panel: str, split: str = "train") -> tuple[pd.Index, np.ndarray, np.ndarray]:
    import traffic
    masked = traffic._load_or_build(panel, split, "masked")
    return (pd.Index(masked["stations"].astype(str)), masked["links"].astype(str),
            masked["times"])


def load_gbm_rows(panel: str, c: dict, path: str, split: str = "train"):
    """Return (tpl_row_pos, pred_s, pred_q) for the GBM holdout CSV rows."""
    st_index, _, times = station_index(panel, split)
    df = pd.read_csv(path, dtype={"timestamp": str, "station_id": str, "link_id": str,
                                  "panel": str, "mask_regime": str})
    t_index = pd.Series(np.arange(len(times), dtype=np.int64), index=epoch_iso(times))
    ti = pd.to_datetime(df["timestamp"], utc=True).map(t_index).to_numpy(dtype=np.float64)
    si = st_index.get_indexer(df["station_id"].astype(str))
    bad = ~np.isfinite(ti) | (si < 0)
    if bad.any():
        raise SystemExit(f"{panel}: {int(bad.sum())} GBM rows could not be mapped")
    ti = ti.astype(np.int64)
    S = c["lanes"].shape[0]
    keys = ti * S + si
    tpl_keys = c["tpl_t"].astype(np.int64) * S + c["tpl_s"].astype(np.int64)
    order = np.argsort(tpl_keys, kind="stable")
    tpl_sorted = tpl_keys[order]
    pos = np.searchsorted(tpl_sorted, keys)
    pos = np.clip(pos, 0, len(tpl_sorted) - 1)
    ok = tpl_sorted[pos] == keys
    if not ok.all():
        print(f"  [{panel}] WARNING {int((~ok).sum())}/{len(keys)} GBM rows unmatched", flush=True)
        pos = pos[ok]
        df = df[ok].reset_index(drop=True)
    tpl_pos = order[pos]
    ps = pd.to_numeric(df["speed_kmh"], errors="coerce").to_numpy(np.float64)
    pq = pd.to_numeric(df["flow_vph"], errors="coerce").to_numpy(np.float64)
    return df, tpl_pos, ps, pq


def physics(df_base: pd.DataFrame, panel: str, variants: dict, jitter: float = 0.06):
    """Frozen task3 holdout proxy for each named variant; writes temp CSVs."""
    import local_eval as le
    pdata = le.load_panel(panel, "train")
    flux = le.build_frozen_flux(pdata, "projected", 1, flux_jitter=jitter)
    out = {}
    for tag, (vs, vq) in variants.items():
        path = os.path.join(U.HERE, f"_phys_{panel}_{tag}.csv")
        d = df_base.copy()
        d["speed_kmh"] = np.round(np.asarray(vs, np.float64), 3)
        d["flow_vph"] = np.round(np.asarray(vq, np.float64), 2)
        d.to_csv(path, index=False)
        srows = le._read_submission_rows(path, [panel], "train")
        cs, cf = le._rows_to_arrays(srows, pdata)
        r = le.evaluate({panel: pdata}, lambda pd_: (cs, cf), {panel: flux})
        r = r[r.level == "panel"].iloc[0]
        out[tag] = {"S_FD": float(r.S_FD), "S_LWR": float(r.S_LWR),
                    "S_physics": float(r.S_physics)}
        os.remove(path)
        print("  [%s %s] S_FD=%.4f S_LWR=%.4f S_physics=%.4f" % (
            panel, tag, out[tag]["S_FD"], out[tag]["S_LWR"], out[tag]["S_physics"]), flush=True)
    del pdata, flux
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*",
                    default=["D7_I210_W", "D7_I405_S", "D7_I10_E", "D12_I5_N"])
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tag", default="u45")
    ap.add_argument("--holdout-days", type=int, default=60)
    ap.add_argument("--inner-val", type=int, default=20)
    ap.add_argument("--skip-physics", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--jitter", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--widths", default="24,48,96,128",
                    help="comma separated U-Net widths, e.g. 32,64,128,192")
    args = ap.parse_args()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} device={dev}", flush=True)
    rows_out = []
    for panel in args.panels:
        t0 = time.time()
        U.build_cache(panel, "train", holdout_days=args.holdout_days)
        c = U.load_cache(panel, f"train_hold{args.holdout_days}")
        n_days = c["times"].size // U.N_TOD
        days = np.arange(n_days)
        fit = days[:n_days - args.holdout_days - args.inner_val]
        val = days[n_days - args.holdout_days - args.inner_val:n_days - args.holdout_days]
        hold_rows = np.where(U.row_day(c, np.arange(len(c["tpl_t"])))
                             >= n_days - args.holdout_days)[0]
        widths = tuple(int(x) for x in args.widths.split(","))
        model, info = U.train_panel(c, panel, fit, val, epochs=args.epochs,
                                    batch=args.batch, lr=args.lr, seed=args.seed,
                                    widths=widths, device=dev, log_every=5)
        print(f"  [{panel}] widths={widths} params={U.n_params(model)/1e6:.2f}M", flush=True)
        v_un, q_un = U.predict_rows(model, c, hold_rows, info["W"], dev)
        np.savez(os.path.join(U.HERE, f"pred_unet_{panel}_{args.tag}.npz"),
                 rows=hold_rows, v=v_un, q=q_un, W=info["W"],
                 best_epoch=info["best_epoch"], best_val=info["best_val"])
        torch.save({"state": model.state_dict(), "W": info["W"], "widths": info["widths"],
                    "panel": panel, "tag": args.tag, "epochs": args.epochs},
                   os.path.join(U.HERE, f"model_{panel}_{args.tag}.pt"))

        # ---- GBM side (shipped holdout CSV), aligned to the same cells ----
        gbm_csv = os.path.join(GBM_DIR, f"holdout_{panel}.csv")
        df, tpl_pos, ps_gbm, pq_gbm = load_gbm_rows(panel, c, gbm_csv)
        tv, tq, lanes, reg = U.truth_rows(c, tpl_pos)
        v_un_m = v_un[np.searchsorted(hold_rows, tpl_pos)]
        q_un_m = q_un[np.searchsorted(hold_rows, tpl_pos)]

        variants = {"gbm": (ps_gbm, pq_gbm), "unet": (v_un_m, q_un_m)}
        for w in WEIGHTS:
            variants[f"w{int(round(w * 100))}"] = (w * ps_gbm + (1 - w) * v_un_m,
                                                   w * pq_gbm + (1 - w) * q_un_m)
        for tag, (vs, vq) in variants.items():
            res = score(tv, tq, vs, vq, lanes, reg)
            row = {"panel": panel, "variant": tag, "S_state": res["macro"],
                   **{f"R{r}": res[r]["S_state"] for r in (1, 2, 3)},
                   "rmse_speed": float(np.mean([res[r]["rmse_speed"] for r in (1, 2, 3)])),
                   "rmse_flow_lane": float(np.mean([res[r]["rmse_flow"] for r in (1, 2, 3)])),
                   "epochs": args.epochs, "tag": args.tag}
            rows_out.append(row)
            print("  [%s %-4s] %s" % (panel, tag, fmt(res)), flush=True)

        if not args.skip_physics:
            try:
                ph = physics(df, panel, variants, args.jitter)
                for r in rows_out:
                    if r["panel"] == panel and r["variant"] in ph:
                        r.update(ph[r["variant"]])
                        r["est_total"] = 0.35 * r["S_state"] + 0.15 * r["S_physics"]
            except Exception as exc:  # proxy optional
                print(f"  [{panel}] physics proxy failed: {exc!r}", flush=True)
        # keep the UNet holdout prediction CSV for the record
        d = df.copy()
        d["speed_kmh"] = np.round(v_un_m, 3)
        d["flow_vph"] = np.round(q_un_m, 2)
        d.to_csv(os.path.join(U.HERE, f"holdout_unet_{panel}_{args.tag}.csv"), index=False)
        print(f"  [{panel}] done {time.time() - t0:.0f}s", flush=True)
        del c, model
        gc.collect()

    df = pd.DataFrame(rows_out)
    out = os.path.join(U.HERE, f"holdout_table_{args.tag}.csv")
    df.to_csv(out, index=False)
    print("\n=== S_state ===")
    print(df.pivot(index="panel", columns="variant", values="S_state").to_string(
        float_format=lambda v: f"{v:.4f}"))
    if "est_total" in df and df["est_total"].notna().any():
        print("\n=== est_total = 0.35*S_state + 0.15*S_physics ===")
        print(df.pivot(index="panel", columns="variant", values="est_total").to_string(
            float_format=lambda v: f"{v:.4f}"))
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()
