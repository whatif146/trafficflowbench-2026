"""FINAL deployment: refit every panel on all 273 train days, seed-average, blend.

For each panel:
  1. clip bounds from the plain layer of ALL train days (per-link p0.5/p99.5
     speed, p99.5 flow) - the same guard the shipped GBM uses;
  2. cache for validation/private (masked view of that split only);
  3. one U-Net per seed (same architecture/config/epochs as the evaluated
     recipe, but fit on all train days instead of days<213), predicting the
     validation and private template rows of that split;
  4. average the seeds, then blend with the shipped GBM state file:
        state = w * GBM_shipped + (1 - w) * UNet_seedmean

Output: state_gbm_unet_validation.csv / state_gbm_unet_private.csv in the
shipped state-file format (identical columns, row order and rounding), plus
deploy_meta.json.

    python deploy_refit.py --seeds 43 44 --w 0.6 --epochs 80
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.environ.get("TFB_ROOT", r"C:\Users\lenovo\Desktop\kaggle") + r""
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "unet"))
sys.path.insert(0, os.path.join(ROOT, "work", "lib"))
sys.path.insert(0, os.path.join(ROOT, "work", "task1", "gbm"))

import traffic  # noqa: E402
import unetlib as U  # noqa: E402

GBM_DIR = os.path.join(ROOT, "work", "task1", "gbm")
COLS = ["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N",
          "D7_I405_S", "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]


def full_bounds(panel: str):
    """Clip bounds from the plain layer of ALL train days."""
    plain = traffic._load_or_build(panel, "train", "plain")
    allt = np.ones(plain["times"].size, dtype=bool)
    out = U.clip_bounds(plain, allt)
    del plain
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*", default=PANELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[43, 44])
    ap.add_argument("--w", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--splits", nargs="*", default=["validation", "private"])
    ap.add_argument("--outprefix", default="state_gbm_unet")
    ap.add_argument("--meta-name", default="deploy_meta.json")
    ap.add_argument("--pred-suffix", default="",
                    help="suffix for the seed-mean npz, so a new seed count does not "
                         "overwrite the previous seed-mean predictions")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--blend-only", action="store_true")
    args = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} device={dev}", flush=True)
    t_start = time.time()

    for panel in (args.panels if not args.blend_only else []):
        t0 = time.time()
        bounds = full_bounds(panel)
        for split in args.splits:
            U.build_cache(panel, split, tag=split, bounds=bounds)
        c = U.load_cache(panel, "train_hold60")
        n_days = c["times"].size // U.N_TOD
        days = np.arange(n_days)
        print(f"[{panel}] refit days 0..{n_days - 1} ({n_days} days) "
              f"rows={len(c['tpl_t'])}", flush=True)
        seed_preds = {s: {} for s in args.splits}
        for seed in args.seeds:
            mpath = os.path.join(U.HERE, f"model_{panel}_refit_s{seed}.pt")
            if os.path.exists(mpath) and not args.force_train:
                ck = torch.load(mpath, map_location="cpu", weights_only=False)
                model = U.build_model(tuple(ck["widths"])).to(dev)
                model.load_state_dict(ck["state"])
                W = ck["W"]
                print(f"  [{panel} s{seed}] loaded {os.path.basename(mpath)}", flush=True)
            else:
                model, info = U.train_panel(c, panel, days, [], epochs=args.epochs,
                                            batch=args.batch, lr=args.lr, seed=seed,
                                            device=dev, log_every=20)
                W = info["W"]
                torch.save({"state": model.state_dict(), "W": W,
                            "widths": info["widths"], "panel": panel, "seed": seed,
                            "epochs": args.epochs, "fit_days": int(n_days),
                            "lr": args.lr, "batch": args.batch},
                           mpath)
            model.eval()
            for split in args.splits:
                outf = os.path.join(U.HERE, f"pred_unet_refit_{panel}_{split}_s{seed}.npz")
                if os.path.exists(outf) and not args.force_train:
                    print(f"  [{panel} s{seed} {split}] preds cached", flush=True)
                    continue
                cc = U.load_cache(panel, split)
                rows = np.arange(len(cc["tpl_t"]))
                v, q = U.predict_rows(model, cc, rows, W, dev)
                np.savez(outf, v=v, q=q, seed=seed, W=W)
                del cc
                gc.collect()
            del model
            gc.collect()
            print(f"  [{panel} s{seed}] predicted {args.splits} {time.time() - t0:.0f}s",
                  flush=True)
        for split in args.splits:
            fs = sorted(glob.glob(os.path.join(U.HERE,
                                               f"pred_unet_refit_{panel}_{split}_s*.npz")))
            seeds_here = [int(np.load(f)["seed"]) for f in fs]
            vs = np.mean([np.load(f)["v"] for f in fs], axis=0)
            qs = np.mean([np.load(f)["q"] for f in fs], axis=0)
            np.savez(os.path.join(U.HERE, f"pred_unet_refit_{panel}_{split}{args.pred_suffix}.npz"),
                     v=vs, q=qs, seeds=np.array(seeds_here))
            print(f"  [{panel} {split}] seed-mean over {seeds_here}", flush=True)
        del c, seed_preds
        gc.collect()
        print(f"  [{panel}] done {time.time() - t0:.0f}s", flush=True)

    # ---------------- blend + write deliverables ----------------
    seeds_present = {}
    for split in args.splits:
        for panel in args.panels:
            fs = sorted(glob.glob(os.path.join(U.HERE, f"pred_unet_refit_{panel}_{split}_s*.npz")))
            seeds_present.setdefault(split, {})[panel] = [int(np.load(f)["seed"]) for f in fs]
    meta = {"w_gbm": args.w, "seeds_requested": args.seeds, "seeds_used": seeds_present,
            "epochs": args.epochs,
            "architecture": "compact 2D U-Net, 3 pools (4 levels), widths "
                            "[24,48,96,128], GroupNorm, SiLU, 1x1 head, 16 input channels, "
                            "2 output channels (speed/flow residual in score units)",
            "loss": "0.54*r_s^2 + 0.46*r_q^2 on the published template rows "
                    "(this IS 1 - S_state per cell)",
            "fit": "all 273 train days (refit); holdout evidence from the separate "
                   "protocol that fits on days<213 only",
            "splits": args.splits, "footprint": {}}
    for split in args.splits:
        gpath = os.path.join(GBM_DIR, f"state_gbm_{split}.csv")
        g = pd.read_csv(gpath, dtype={"timestamp": str, "station_id": str,
                                      "link_id": str, "panel": str, "mask_regime": str})
        off = 0
        foot = {}
        dst = os.path.join(U.HERE, f"{args.outprefix}_{split}.csv")
        for panel in args.panels:
            p = U.load_cache(panel, split)
            n = len(p["tpl_t"])
            sl = g.iloc[off:off + n]
            if len(sl) != n or (sl["panel"].to_numpy() != panel).any():
                raise SystemExit(f"{panel} {split}: GBM rows misaligned off={off} n={n}")
            z = np.load(os.path.join(U.HERE, f"pred_unet_refit_{panel}_{split}{args.pred_suffix}.npz"))
            v_un, q_un = z["v"], z["q"]
            if not (np.isfinite(v_un).all() and np.isfinite(q_un).all()):
                raise SystemExit(f"{panel} {split}: non-finite U-Net prediction")
            gs = sl["speed_kmh"].to_numpy(np.float64)
            gq = sl["flow_vph"].to_numpy(np.float64)
            vs = args.w * gs + (1 - args.w) * v_un
            vq = args.w * gq + (1 - args.w) * q_un
            f = sl.copy()
            f["speed_kmh"] = np.round(vs, 3)
            f["flow_vph"] = np.round(vq, 2)
            if f["speed_kmh"].isna().any() or f["flow_vph"].isna().any():
                raise SystemExit(f"{panel} {split}: NaN in the output")
            # stream panel by panel: never hold all 3.5M rows twice
            dst = os.path.join(U.HERE, f"{args.outprefix}_{split}.csv")
            f[COLS].to_csv(dst, mode=("w" if off == 0 else "a"),
                           header=(off == 0), index=False)
            del f
            lanes = p["lanes"][p["tpl_s"]].astype(np.float64)
            dq = np.abs(vq - gq) / np.maximum(np.abs(gq), 1.0)
            foot[panel] = {
                "mean_abs_dv_kmh": float(np.mean(np.abs(vs - gs))),
                "mean_abs_dq_per_lane": float(np.mean(np.abs(vq - gq) / lanes)),
                "p99_abs_dv_kmh": float(np.percentile(np.abs(vs - gs), 99)),
                "frac_dq_gt1pct": float(np.mean(dq > 0.01)),
                "frac_dq_gt5pct": float(np.mean(dq > 0.05)),
                "rows": int(n),
            }
            off += n
            del p, z
            gc.collect()
        if off != len(g):
            raise SystemExit(f"{split}: consumed {off} of {len(g)} GBM rows")
        print(f"wrote {dst} rows={off}", flush=True)
        agg = {k: float(np.mean([foot[p][k] for p in args.panels])) for k in
               ("mean_abs_dv_kmh", "mean_abs_dq_per_lane", "p99_abs_dv_kmh",
                "frac_dq_gt1pct", "frac_dq_gt5pct")}
        print(f"  [{split}] footprint vs shipped GBM: mean|dv|={agg['mean_abs_dv_kmh']:.3f} "
              f"km/h, mean|dq|/lane={agg['mean_abs_dq_per_lane']:.2f} vph, "
              f"frac dq>1%={agg['frac_dq_gt1pct']:.3f}, frac dq>5%={agg['frac_dq_gt5pct']:.3f}",
              flush=True)
        meta["footprint"][split] = {"per_panel": foot, "mean": agg}
        del g
        gc.collect()
    meta["elapsed_sec"] = round(time.time() - t_start, 1)
    with open(os.path.join(U.HERE, args.meta_name), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print("wrote", args.meta_name, flush=True)


if __name__ == "__main__":
    main()
