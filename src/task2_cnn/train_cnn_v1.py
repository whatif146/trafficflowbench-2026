"""Onset-only corridor CNN for Task 2 (experiment, work/task2_cnn).

Input : [C, 13, L] corridor image (60-min observed history + the origin row T of
        the split's masked layer), see build_images.py.
Output: per-link probability of "queue at T+30" (only the last horizon step is
        ever predicted: the released selector puts the onset queue exactly there).

Honest two-protocol evaluation on identical rows/rules as the shipped onset model
(v3+v5 blend, w_on=0.3, threshold 0.30 on the last step + top-1 fallback):
  protocol B = February pseudo-window block;
  protocol A = the 40 released train onset windows (June 2030, never fitted);
  June-origin windows are excluded from the fit block exactly like the reference
  blend models, so protocol A stays honest.  Threshold / fallback / blend weight
  are selected on the January block only.

Usage:
  python train_cnn.py --epochs 40
  python train_cnn.py --epochs 40 --fit-mode std        # diagnostic, +June in fit
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
LMAX = 352
N_STEPS = 13
C_IN = 17        # reset from the caches at run time (kept for the padded batch)


# --------------------------------------------------------------------------- #
# data
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


def pad_batch(panels, keys):
    c_in = panels[keys[0][0]]["X"].shape[1]
    B = len(keys)
    x = np.zeros((B, c_in, N_STEPS, LMAX), np.float32)
    mask = np.zeros((B, LMAX), bool)
    for i, (p, r) in enumerate(keys):
        L = panels[p]["L"]
        x[i, :, :, :L] = panels[p]["X"][r].astype(np.float32)
        mask[i, :L] = True
    return torch.from_numpy(x), torch.from_numpy(mask)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class OnsetCNN(nn.Module):
    def __init__(self, c_in=C_IN, drop=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c_in, 32, (3, 5), padding=(1, 2)), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 48, (3, 5), padding=(1, 2)), nn.BatchNorm2d(48), nn.ReLU(True),
            nn.Dropout2d(drop),
            nn.Conv2d(48, 64, (3, 3), padding=(1, 1)), nn.BatchNorm2d(64), nn.ReLU(True),
        )
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
    with torch.no_grad():
        for i in range(0, len(keys), batch):
            ks = keys[i:i + batch]
            x, _ = pad_batch(panels, ks)
            with autocast():
                logits = model(x.cuda())
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            for j, k in enumerate(ks):
                out[k] = probs[j, :panels[k[0]]["L"]].astype(np.float32)
    return out


def predict_released(model, rel, split="train", condition="queue_onset", batch=16):
    model.eval()
    out = {}
    for p in LH.PANELS:
        sel = np.where((rel[p]["split"] == split) & (rel[p]["condition"] == condition))[0]
        if len(sel) == 0:
            continue
        L = rel[p]["L"]
        X = rel[p]["X"][sel]
        c_in = X.shape[1]
        with torch.no_grad():
            for i in range(0, len(sel), batch):
                n = min(batch, len(sel) - i)
                x = np.zeros((n, c_in, N_STEPS, LMAX), np.float32)
                for j in range(n):
                    x[j, :, :, :L] = X[i + j].astype(np.float32)
                with autocast():
                    logits = model(torch.from_numpy(x).cuda())
                probs = torch.sigmoid(logits.float()).cpu().numpy()[:, :L]
                for j in range(n):
                    out[(p, int(sel[i + j]))] = probs[j]
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--pos-weight", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fit-mode", choices=["june_excl", "std"], default="june_excl")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=2)
    args = ap.parse_args()
    tag = args.tag or (f"{args.fit_mode}_e{args.epochs}_b{args.batch}_lr{args.lr}"
                       f"_pw{args.pos_weight}_d{args.drop}_s{args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    panels = load_images()
    rel = load_released()
    fit_keys = block_keys(panels, "fit", args.fit_mode)
    tune_keys = block_keys(panels, "tune")
    feb_keys = block_keys(panels, "eval_feb")
    n_rel_on = sum(int(((rel[p]["split"] == "train") & (rel[p]["condition"] == "queue_onset")).sum())
                   for p in LH.PANELS)
    print(f"windows: fit {len(fit_keys)} | tune {len(tune_keys)} | feb {len(feb_keys)} "
          f"| released onset {n_rel_on}", flush=True)

    ref = np.load(os.path.join(CACHE, "ref_probs.npz"), allow_pickle=False)
    # reference (shipped blend) probabilities on the same rows
    ref_tune, ref_feb = {}, {}
    cnt = {}
    for p in LH.PANELS:
        for block, store in (("tune", ref_tune), ("eval_feb", ref_feb)):
            idx = ref[f"{block}_{p}_idx"]
            p3, p5 = ref[f"{block}_{p}_prob3"], ref[f"{block}_{p}_prob5"]
            for j, r in enumerate(idx):
                store[(p, int(r))] = 0.3 * p3[j] + 0.7 * p5[j]
    rel_keys = []
    for p in LH.PANELS:
        sel = np.where((rel[p]["split"] == "train") & (rel[p]["condition"] == "queue_onset"))[0]
        rel_keys += [(p, int(i)) for i in sel]
    rp = ref["rel_train_panel"]
    ref_rel = []
    for p in LH.PANELS:
        sel = np.where((rel[p]["split"] == "train") & (rel[p]["condition"] == "queue_onset"))[0]
        p3 = ref[f"rel_train_prob3_{p}"]
        p5 = ref[f"rel_train_prob5_{p}"]
        for j in range(len(sel)):
            ref_rel.append((0.3 * p3[j] + 0.7 * p5[j])[LH.HORIZON - 1])
    assert len(rel_keys) == len(rp) == len(ref_rel), (len(rel_keys), len(rp), len(ref_rel))
    ref_truth = {p: ref[f"rel_train_truth_{p}"] for p in LH.PANELS}
    ref_elig = {p: ref[f"rel_train_elig_{p}"] for p in LH.PANELS}

    c_in = panels[LH.PANELS[0]]["X"].shape[1]
    model = OnsetCNN(c_in=c_in, drop=args.drop).cuda()
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params {nparam/1e3:.1f}k", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.BCEWithLogitsLoss(reduction="none",
                                 pos_weight=torch.tensor(args.pos_weight))
    scaler = GradScaler()
    ths = np.round(np.arange(0.10, 0.81, 0.025), 3)
    fbs = (0, 1)

    best = {"jan": -1.0}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(fit_keys))
        ep_loss, nb = 0.0, 0
        te = time.time()
        for i in range(0, len(perm), args.batch):
            ks = [fit_keys[j] for j in perm[i:i + args.batch]]
            x, mask = pad_batch(panels, ks)
            y = torch.zeros(x.shape[0], LMAX)
            for j, (p, r) in enumerate(ks):
                L = panels[p]["L"]
                y[j, :L] = torch.from_numpy(panels[p]["y"][r].astype(np.float32))
            x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            opt.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
            loss = lossf(logits.float(), y)
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss.detach())
            nb += 1
        sched.step()
        history.append({"epoch": epoch, "loss": ep_loss / max(nb, 1),
                        "sec": round(time.time() - te, 2)})
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            tune_probs = predict_keys(model, panels, tune_keys)
            bj = None
            for fb in fbs:
                for th in ths:
                    vals = [LH.iou(onset_pred(tune_probs[k], th, fb),
                                   panels[k[0]]["truth"][k[1]],
                                   panels[k[0]]["elig"][k[1]]) for k in tune_keys]
                    v = float(np.mean(vals))
                    if bj is None or v > bj[0]:
                        bj = (v, float(th), int(fb))
            torch.cuda.empty_cache()
            print(f"epoch {epoch:3d} loss {ep_loss/max(nb,1):.4f} "
                  f"({time.time()-te:.1f}s) jan IoU {bj[0]:.4f} th {bj[1]} fb {bj[2]}",
                  flush=True)
            if bj[0] > best["jan"]:
                best = {"jan": bj[0], "th": bj[1], "fb": bj[2], "epoch": epoch}
                torch.save(model.state_dict(), os.path.join(MODELS, f"cnn_{tag}.pt"))
            history[-1].update({"jan_iou": bj[0], "jan_th": bj[1], "jan_fb": bj[2]})
        del x, mask, y
    print(f"best Jan: {best}", flush=True)

    # ---------------- untouched-block evaluation -------------------------
    model.load_state_dict(torch.load(os.path.join(MODELS, f"cnn_{tag}.pt")))
    tune_probs = predict_keys(model, panels, tune_keys)
    feb_probs = predict_keys(model, panels, feb_keys)
    rel_probs_cnn = predict_released(model, rel, "train", "queue_onset")

    def iou_list(keys, prob_map, truth_map, elig_map, th, fb):
        return [LH.iou(onset_pred(prob_map[k], th, fb), truth_map[k], elig_map[k])
                for k in keys]

    feb_truth = {k: panels[k[0]]["truth"][k[1]] for k in feb_keys}
    feb_elig = {k: panels[k[0]]["elig"][k[1]] for k in feb_keys}
    tune_truth = {k: panels[k[0]]["truth"][k[1]] for k in tune_keys}
    tune_elig = {k: panels[k[0]]["elig"][k[1]] for k in tune_keys}
    rel_truth, rel_elig = {}, {}
    pos = 0
    for p in LH.PANELS:
        sel = np.where((rel[p]["split"] == "train") & (rel[p]["condition"] == "queue_onset"))[0]
        for j in range(len(sel)):
            rel_truth[(p, int(sel[j]))] = ref_truth[p][j]
            rel_elig[(p, int(sel[j]))] = ref_elig[p][j]
            pos += 1

    # select (w_cnn, th, fb) on January for the blend, and separately for pure CNN
    blend_grid = [(w, th, fb) for w in (0.0, 0.25, 0.5, 0.75, 1.0)
                  for th in ths for fb in fbs]
    best_blend = None
    for w, th, fb in blend_grid:
        vals = []
        for k in tune_keys:
            prob = w * tune_probs[k] + (1 - w) * ref_tune[k]
            vals.append(LH.iou(onset_pred(prob, th, fb), tune_truth[k], tune_elig[k]))
        v = float(np.mean(vals))
        if best_blend is None or v > best_blend[0]:
            best_blend = (v, w, float(th), int(fb))
    # pure CNN rule (w=1) already inside the grid; report its January-selected optimum
    cnn_sel = None
    for w, th, fb in blend_grid:
        if w != 1.0:
            continue
        vals = []
        for k in tune_keys:
            vals.append(LH.iou(onset_pred(tune_probs[k], th, fb), tune_truth[k], tune_elig[k]))
        v = float(np.mean(vals))
        if cnn_sel is None or v > cnn_sel[0]:
            cnn_sel = (v, th, int(fb))
    print(f"Jan selection: CNN-only {cnn_sel} | blend {best_blend}", flush=True)

    def report(name, w, th, fb):
        val_b = float(np.mean([LH.iou(onset_pred(w * feb_probs[k] + (1 - w) * ref_feb[k], th, fb),
                                      feb_truth[k], feb_elig[k]) for k in feb_keys]))
        val_a = float(np.mean([LH.iou(onset_pred(w * rel_probs_cnn[rel_keys[j]] + (1 - w) * ref_rel[j], th, fb),
                                      rel_truth[rel_keys[j]], rel_elig[rel_keys[j]])
                               for j in range(len(rel_keys))]))
        return {"name": name, "w_cnn": w, "th": th, "fb": fb,
                "B_feb_onset": val_b, "A_june_onset": val_a}

    rows = [report("reference (shipped w_on .3, th .30, fb1)", 0.0, 0.30, 1),
            report("CNN-only (Jan-selected)", 1.0, cnn_sel[1], cnn_sel[2]),
            report("blend CNN+ref (Jan-selected)", best_blend[1], best_blend[2], best_blend[3])]
    df = pd.DataFrame(rows)
    print("\n" + df.round(4).to_string(index=False), flush=True)
    df.to_csv(os.path.join(OUT, f"protocols_{tag}.csv"), index=False)

    # per-panel detail for the selected rows
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

    res = {"tag": tag, "fit_mode": args.fit_mode, "epochs": args.epochs,
           "batch": args.batch, "lr": args.lr, "pos_weight": args.pos_weight,
           "drop": args.drop, "seed": args.seed, "params_k": round(nparam / 1e3, 1),
           "best_jan_checkpoint": best, "cnn_jan_rule": cnn_sel,
           "blend_jan_rule": best_blend, "train_time_s": round(time.time() - t0, 1),
           "n_fit": len(fit_keys), "n_tune": len(tune_keys), "n_feb": len(feb_keys),
           "n_rel_onset": len(rel_keys),
           "protocols": rows, "history": history}
    with open(os.path.join(OUT, f"res_{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"saved out/res_{tag}.json")


if __name__ == "__main__":
    main()
