"""Onset-only corridor CNN for Task 2 (experiment, work/task2_cnn).

Input : [C, 13, L] corridor image (60-min observed history + the origin row T of
        the split's masked layer, plus per-link queue climatology channels), see
        build_images.py.
Output: per-link probability of "queue at T+30" (only the last horizon step is
        ever predicted: the released selector puts the onset queue there).

Honest two-protocol evaluation on identical rows/rules as the shipped onset model
(v3+v5 blend, w_on=0.3, threshold 0.30 on the last step + top-1 fallback):
  protocol B = February pseudo-window block;
  protocol A = the 40 released train onset windows (June 2030, never fitted);
  June-origin windows are excluded from the fit block exactly like the reference
  blend models, so protocol A stays honest.  Threshold / fallback / blend weight
  are selected on the January block only.
  Panel-wise batching: no link padding, so BatchNorm statistics are not polluted
  by padded zeros.

Usage:
  python train_cnn.py --epochs 80 --arch dilated
"""
from __future__ import annotations

import argparse
import gc
import json
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

import lhcore as LH  # noqa: E402

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "out")
MODELS = os.path.join(HERE, "models")
for _d in (OUT, MODELS):
    os.makedirs(_d, exist_ok=True)

TUNE_EPOCH = int(LH.TUNE_START.timestamp())
CUT_EPOCH = int(LH.CUT.timestamp())
MAR_EPOCH = int(pd.Timestamp("2031-03-01", tz="UTC").timestamp())
JUNE0 = int(pd.Timestamp("2030-06-01", tz="UTC").timestamp())
JUNE1 = int(pd.Timestamp("2030-07-01", tz="UTC").timestamp())
N_STEPS = 13


# --------------------------------------------------------------------------- #
def load_images():
    panels = {}
    for p in LH.PANELS:
        z = np.load(os.path.join(CACHE, f"tr_{p}.npz"), allow_pickle=False)
        panels[p] = {"X": z["X"], "y": z["y"], "time": z["time"], "acc": z["acc"],
                     "truth": z["truth"], "elig": z["elig"], "L": int(z["n_links"])}
        z.close()
    return panels


def load_released():
    rel = {}
    for p in LH.PANELS:
        z = np.load(os.path.join(CACHE, f"rel_{p}.npz"), allow_pickle=False)
        rel[p] = {"X": z["X"], "window_ids": np.asarray(z["window_ids"], dtype=str),
                  "split": np.asarray(z["split"], dtype=str),
                  "condition": np.asarray(z["condition"], dtype=str),
                  "L": int(z["n_links"])}
        z.close()
    return rel


def augment_batch(x, y, p_roll=0.4, p_tdrop=0.4, noise=0.02):
    """Cheap training augmentation: rank roll, time-step dropout, gaussian noise."""
    B, C, T, L = x.shape
    if torch.rand(1, device=x.device).item() < p_roll:
        shift = int(torch.randint(-max(1, L // 4), max(1, L // 4) + 1, (1,)).item())
        if shift:
            x = torch.roll(x, shift, dims=3)
            y = torch.roll(y, shift, dims=1)
    if torch.rand(1, device=x.device).item() < p_tdrop:
        n = int(torch.randint(1, 3, (1,)).item())
        idxs = torch.randint(0, T, (n,), device=x.device)
        x[:, :6, idxs, :] = 0.0          # drop the observed channels at those steps
    if noise > 0:
        x = x.clone()
        x[:, [0, 2, 3]] = x[:, [0, 2, 3]] + noise * torch.randn_like(x[:, [0, 2, 3]])
    return x, y


def batch_from(panels, p, idx):
    x = torch.from_numpy(panels[p]["X"][idx].astype(np.float32))
    y = torch.from_numpy(panels[p]["y"][idx].astype(np.float32))
    return x, y


# --------------------------------------------------------------------------- #
class OnsetCNN(nn.Module):
    def __init__(self, c_in=17, drop=0.1, arch="dilated"):
        super().__init__()
        self.arch = arch
        if arch == "small":
            body = [
                nn.Conv2d(c_in, 32, (3, 5), padding=(1, 2)), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 48, (3, 5), padding=(1, 2)), nn.BatchNorm2d(48), nn.ReLU(True),
                nn.Dropout2d(drop),
                nn.Conv2d(48, 64, (3, 3), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(True),
            ]
        else:
            # dilated spatial kernels see ~20 links of queue front
            body = [
                nn.Conv2d(c_in, 32, (3, 5), padding=(1, 2)), nn.BatchNorm2d(32), nn.ReLU(True),
                nn.Conv2d(32, 48, (3, 5), padding=(1, 4), dilation=(1, 2)),
                nn.BatchNorm2d(48), nn.ReLU(True), nn.Dropout2d(drop),
                nn.Conv2d(48, 64, (3, 3), padding=(1, 4), dilation=(1, 4)),
                nn.BatchNorm2d(64), nn.ReLU(True),
                nn.Conv2d(64, 64, (3, 3), padding=(1, 8), dilation=(1, 8)),
                nn.BatchNorm2d(64), nn.ReLU(True),
            ]
        self.body = nn.Sequential(*body)
        self.head = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Dropout(drop), nn.Conv1d(64, 1, 1),
        )

    def forward(self, x):
        h = self.body(x)                       # (B, 64, 13, L)
        h = torch.cat([h.mean(dim=2), h[:, :, -1, :]], dim=1)
        return self.head(h).squeeze(1)         # (B, L)


# --------------------------------------------------------------------------- #
# shipped onset rule + metrics
# --------------------------------------------------------------------------- #
def onset_pred(prob, th, fb):
    pred = np.zeros((LH.HORIZON, len(prob)), bool)
    pred[LH.HORIZON - 1] = prob >= th
    if fb and not pred[LH.HORIZON - 1].any():
        pred[LH.HORIZON - 1, int(np.argmax(prob))] = True
    return pred


def block_keys(panels, block, fit_mode="june_excl"):
    keys = []
    for p in LH.PANELS:
        t = panels[p]["time"]
        acc = panels[p]["acc"]
        if block == "fit":
            m = t < TUNE_EPOCH
            if fit_mode == "june_excl":
                m &= ~((t >= JUNE0) & (t < JUNE1))
        elif block == "tune":
            m = acc & (t >= TUNE_EPOCH) & (t < CUT_EPOCH)
        elif block == "eval_feb":
            m = acc & (t >= CUT_EPOCH) & (t < MAR_EPOCH)
        else:
            raise ValueError(block)
        for r in np.where(m)[0]:
            keys.append((p, int(r)))
    return keys


def predict_keys(model, panels, keys, batch=32):
    model.eval()
    out = {}
    by_panel = {}
    for k in keys:
        by_panel.setdefault(k[0], []).append(k[1])
    with torch.no_grad():
        for p, rows in by_panel.items():
            rows = np.asarray(rows)
            for i in range(0, len(rows), batch):
                idx = rows[i:i + batch]
                x, _ = batch_from(panels, p, idx)
                with autocast():
                    logits = model(x.cuda())
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                for j, r in enumerate(idx):
                    out[(p, int(r))] = probs[j].astype(np.float32)
    return out


def predict_released(model, rel, split="train", condition="queue_onset", batch=16):
    model.eval()
    out = {}
    for p in LH.PANELS:
        sel = np.where((rel[p]["split"] == split) & (rel[p]["condition"] == condition))[0]
        if len(sel) == 0:
            continue
        X = rel[p]["X"]
        with torch.no_grad():
            for i in range(0, len(sel), batch):
                idx = sel[i:i + batch]
                x = torch.from_numpy(X[idx].astype(np.float32))
                with autocast():
                    logits = model(x.cuda())
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                for j, r in enumerate(idx):
                    out[(p, int(r))] = probs[j].astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fit-mode", choices=["june_excl", "std", "all"], default="june_excl")
    ap.add_argument("--arch", choices=["small", "dilated"], default="dilated")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--aug", type=int, default=1, help="1 = roll/time-dropout/noise augmentation")
    ap.add_argument("--loss", choices=["bce", "bce_iou"], default="bce")
    ap.add_argument("--iou-w", type=float, default=0.5)
    args = ap.parse_args()
    tag = args.tag or (f"{args.arch}_{args.fit_mode}_e{args.epochs}_b{args.batch}"
                       f"_lr{args.lr}_pw{args.pos_weight}_s{args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    panels = load_images()
    rel = load_released()
    if args.fit_mode == "all":
        fit_keys = []
        for p in LH.PANELS:
            for r in range(len(panels[p]["time"])):
                fit_keys.append((p, int(r)))
    else:
        fit_keys = block_keys(panels, "fit", args.fit_mode)
    tune_keys = block_keys(panels, "tune")
    feb_keys = block_keys(panels, "eval_feb")
    n_rel_on = sum(int(((rel[p]["split"] == "train") & (rel[p]["condition"] == "queue_onset")).sum())
                   for p in LH.PANELS)
    print(f"windows: fit {len(fit_keys)} | tune {len(tune_keys)} | feb {len(feb_keys)} "
          f"| released onset {n_rel_on}", flush=True)

    ref = np.load(os.path.join(CACHE, "ref_probs.npz"), allow_pickle=False)
    ref_tune, ref_feb = {}, {}
    for p in LH.PANELS:
        for block, store in (("tune", ref_tune), ("eval_feb", ref_feb)):
            idx = ref[f"{block}_{p}_idx"]
            p3, p5 = ref[f"{block}_{p}_prob3"], ref[f"{block}_{p}_prob5"]
            for j, r in enumerate(idx):
                store[(p, int(r))] = 0.3 * p3[j] + 0.7 * p5[j]
    rel_keys, ref_rel = [], []
    for p in LH.PANELS:
        sel = np.where((rel[p]["split"] == "train") & (rel[p]["condition"] == "queue_onset"))[0]
        p3 = ref[f"rel_train_prob3_{p}"]
        p5 = ref[f"rel_train_prob5_{p}"]
        for j in range(len(sel)):
            rel_keys.append((p, int(sel[j])))
            ref_rel.append((0.3 * p3[j] + 0.7 * p5[j])[LH.HORIZON - 1])
    ref_truth = {p: ref[f"rel_train_truth_{p}"] for p in LH.PANELS}
    ref_elig = {p: ref[f"rel_train_elig_{p}"] for p in LH.PANELS}
    rel_truth, rel_elig = {}, {}
    for p in LH.PANELS:
        sel = np.where((rel[p]["split"] == "train") & (rel[p]["condition"] == "queue_onset"))[0]
        for j in range(len(sel)):
            rel_truth[(p, int(sel[j]))] = ref_truth[p][j]
            rel_elig[(p, int(sel[j]))] = ref_elig[p][j]

    c_in = panels[LH.PANELS[0]]["X"].shape[1]
    model = OnsetCNN(c_in=c_in, drop=args.drop, arch=args.arch).cuda()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model {args.arch} params {nparam/1e3:.1f}k c_in {c_in}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss(reduction="mean",
                                 pos_weight=torch.tensor(args.pos_weight))
    scaler = GradScaler()
    ths = np.round(np.arange(0.10, 0.86, 0.025), 3)
    fbs = (0, 1)

    fit_by_panel = {}
    for p, r in fit_keys:
        fit_by_panel.setdefault(p, []).append(r)
    fit_by_panel = {p: np.asarray(v) for p, v in fit_by_panel.items()}

    def eval_tune():
        tp = predict_keys(model, panels, tune_keys)
        bj = None
        for fb in fbs:
            for th in ths:
                vals = [LH.iou(onset_pred(tp[k], th, fb),
                               panels[k[0]]["truth"][k[1]],
                               panels[k[0]]["elig"][k[1]]) for k in tune_keys]
                v = float(np.mean(vals))
                if bj is None or v > bj[0]:
                    bj = (v, float(th), int(fb))
        return bj

    best = {"jan": -1.0}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(LH.PANELS))
        ep_loss, nb = 0.0, 0
        te = time.time()
        for pi in order:
            p = LH.PANELS[pi]
            rows = fit_by_panel.get(p)
            if rows is None or len(rows) == 0:
                continue
            perm = np.random.permutation(rows)
            for i in range(0, len(perm), args.batch):
                idx = perm[i:i + args.batch]
                if len(idx) < 2:
                    continue
                x, y = batch_from(panels, p, idx)
                x, y = x.cuda(), y.cuda()
                if args.aug:
                    x, y = augment_batch(x, y)
                opt.zero_grad(set_to_none=True)
                with autocast():
                    logits = model(x)
                loss = lossf(logits.float(), y)
                if args.loss == "bce_iou":
                    pr = torch.sigmoid(logits.float())
                    inter = (pr * y).sum(dim=1)
                    union = pr.sum(dim=1) + y.sum(dim=1) - inter
                    soft_iou = inter / union.clamp(min=1e-6)
                    loss = loss + args.iou_w * (1.0 - soft_iou).mean()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                ep_loss += float(loss.detach())
                nb += 1
        sched.step()
        history.append({"epoch": epoch, "loss": ep_loss / max(nb, 1),
                        "sec": round(time.time() - te, 2)})
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            bj = eval_tune()
            torch.cuda.empty_cache()
            print(f"epoch {epoch:3d} loss {ep_loss/max(nb,1):.4f} "
                  f"({time.time()-te:.1f}s) jan IoU {bj[0]:.4f} th {bj[1]} fb {bj[2]}",
                  flush=True)
            if bj[0] > best["jan"]:
                best = {"jan": bj[0], "th": bj[1], "fb": bj[2], "epoch": epoch}
                torch.save(model.state_dict(), os.path.join(MODELS, f"cnn_{tag}.pt"))
            history[-1].update({"jan_iou": bj[0], "jan_th": bj[1], "jan_fb": bj[2]})
        del x, y
    print(f"best Jan: {best}", flush=True)

    # ---------------- untouched-block evaluation -------------------------
    model.load_state_dict(torch.load(os.path.join(MODELS, f"cnn_{tag}.pt")))
    tune_probs = predict_keys(model, panels, tune_keys)
    feb_probs = predict_keys(model, panels, feb_keys)
    rel_probs_cnn = predict_released(model, rel, "train", "queue_onset")
    feb_truth = {k: panels[k[0]]["truth"][k[1]] for k in feb_keys}
    feb_elig = {k: panels[k[0]]["elig"][k[1]] for k in feb_keys}
    tune_truth = {k: panels[k[0]]["truth"][k[1]] for k in tune_keys}
    tune_elig = {k: panels[k[0]]["elig"][k[1]] for k in tune_keys}

    blend_grid = [(w, th, fb) for w in (0.0, 0.25, 0.5, 0.75, 1.0)
                  for th in ths for fb in fbs]
    best_blend = None
    for w, th, fb in blend_grid:
        vals = [LH.iou(onset_pred(w * tune_probs[k] + (1 - w) * ref_tune[k], th, fb),
                       tune_truth[k], tune_elig[k]) for k in tune_keys]
        v = float(np.mean(vals))
        if best_blend is None or v > best_blend[0]:
            best_blend = (v, w, float(th), int(fb))
    cnn_sel = None
    for w, th, fb in blend_grid:
        if w != 1.0:
            continue
        vals = [LH.iou(onset_pred(tune_probs[k], th, fb), tune_truth[k], tune_elig[k])
                for k in tune_keys]
        v = float(np.mean(vals))
        if cnn_sel is None or v > cnn_sel[0]:
            cnn_sel = (v, th, int(fb))
    print(f"Jan selection: CNN-only {cnn_sel} | blend {best_blend}", flush=True)

    def report(name, w, th, fb):
        vb = float(np.mean([LH.iou(onset_pred(w * feb_probs[k] + (1 - w) * ref_feb[k], th, fb),
                                   feb_truth[k], feb_elig[k]) for k in feb_keys]))
        va = float(np.mean([LH.iou(onset_pred(w * rel_probs_cnn[k] + (1 - w) * ref_rel[j], th, fb),
                                   rel_truth[k], rel_elig[k]) for j, k in enumerate(rel_keys)]))
        return {"name": name, "w_cnn": w, "th": th, "fb": fb,
                "B_feb_onset": vb, "A_june_onset": va}

    rows = [report("reference (shipped w_on .3, th .30, fb1)", 0.0, 0.30, 1),
            report("CNN-only (Jan-selected)", 1.0, cnn_sel[1], cnn_sel[2]),
            report("blend CNN+ref (Jan-selected)", best_blend[1], best_blend[2], best_blend[3])]
    df = pd.DataFrame(rows)
    print("\n" + df.round(4).to_string(index=False), flush=True)
    df.to_csv(os.path.join(OUT, f"protocols_{tag}.csv"), index=False)
    det = []
    for row in rows:
        w, th, fb = row["w_cnn"], row["th"], row["fb"]
        for k in feb_keys:
            prob = w * feb_probs[k] + (1 - w) * ref_feb[k]
            det.append({"model": row["name"], "panel": k[0], "protocol": "B_feb",
                        "iou": LH.iou(onset_pred(prob, th, fb), feb_truth[k], feb_elig[k])})
        for j, k in enumerate(rel_keys):
            prob = w * rel_probs_cnn[k] + (1 - w) * ref_rel[j]
            det.append({"model": row["name"], "panel": k[0], "protocol": "A_june",
                        "iou": LH.iou(onset_pred(prob, th, fb), rel_truth[k], rel_elig[k])})
    pd.DataFrame(det).to_csv(os.path.join(OUT, f"detail_{tag}.csv"), index=False)
    res = {"tag": tag, "fit_mode": args.fit_mode, "arch": args.arch,
           "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
           "pos_weight": args.pos_weight, "drop": args.drop, "seed": args.seed,
           "params_k": round(nparam / 1e3, 1), "best_jan_checkpoint": best,
           "cnn_jan_rule": cnn_sel, "blend_jan_rule": best_blend,
           "train_time_s": round(time.time() - t0, 1),
           "n_fit": len(fit_keys), "n_tune": len(tune_keys), "n_feb": len(feb_keys),
           "n_rel_onset": len(rel_keys), "protocols": rows, "history": history}
    with open(os.path.join(OUT, f"res_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"saved out/res_{tag}.json")


if __name__ == "__main__":
    main()
