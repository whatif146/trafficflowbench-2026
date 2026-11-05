# Task 2 — onset-only corridor CNN (bounded experiment, `work/task2_cnn/`)

**Verdict: NO-SHIP.** The CNN, and every blend of it with the shipped onset
model, loses to the shipped onset model on the honest two-protocol evaluation.
No submission file was written; every shipped queue artefact is untouched. The
work is kept reproducible below.

Timebox: one focused attempt (~1.2 h of the 3.5 h budget used). Hardware: RTX
5060 Laptop 8.55 GB, torch 2.9.1+cu128, CUDA available.

## What was built

Input per released/pseudo onset window: corridor image **`[C, 13, L]`**
(13 steps = the released 60-min observed history `T-60..T-5` overlaid from the
observed layer + the origin row `T` from the split's masked layer), channels:

* speed / `v_cut`, speed-observed mask, flow / capacity, density / `k_jam`,
  queue flag, flow-observed mask;
* time-of-day sin/cos, relative link rank, relative time, capacity/10k, lanes/6;
* per-link queue climatology (the v3 model's strongest onset prior):
  `onset_freq`, `ong_freq`, static time-of-day queue rate at the target slot and
  at the last history slot, per-link eligibility rate;
* causal long-history summaries from the masked layer rows `-180..-65`:
  pre-history queue fraction, queue age ending at `T`, mean speed ratio over the
  pre-history, rank distance to the nearest queued link upstream/downstream at
  `T`, corridor queued-link count at `T`.

23 channels total, float16, NaN-aware (missing = 0 + mask channels). Sources are
used read-only: `work/task2_seq/lhcore.py` history assembly (same causal
construction as the v5 model: observed hour overlaid, masked layer elsewhere, no
timestamp > T ever read) + the frozen v3 dataset caches for the window list,
labels and truth.

Data built (`build_images.py`): **5,858 onset windows** (the full v3/v5 onset
window list, one row per window), of which **4,278** are the honest fit block
(pre-2031-01-01, June-2030 origins removed so protocol A stays honest);
246 January tuning windows and 218 February evaluation windows; 40 released
train onset windows. Caches: `cache/tr_<PANEL>.npz`, `cache/rel_<PANEL>.npz`
(~465 MB total), plus `cache/ref_probs.npz` with the reference probabilities.

## Model / training

`OnsetCNN` (`train_cnn.py`): 2-D CNN over `[links x 13 time steps]`,
conv 32/48/64 with BatchNorm+ReLU+Dropout, temporal mean+last-step pooling, 1-D
spatial head, per-link logit. **103k parameters.**
Panel-wise batching (no link padding, so BatchNorm statistics are not polluted
by padded zeros). AMP + GradScaler, AdamW lr 1e-3 cosine, weight decay 1e-4,
batch 16, BCE with `pos_weight=4` (a soft-IoU term and cheap augmentation —
random rank roll, time-step dropout, gaussian noise — were also tried).
**~1.8-2.2 s/epoch**; 40-100 epochs in 1.5-4 min; peak VRAM < 0.3 GB (well under
the 5 GB cap). Output layer predicts only the last horizon step (`T+30`): the
released selector puts the onset queue there and earlier steps are guaranteed FP.

Decision rule = exactly the shipped one: probability threshold on the last step +
optional top-1 fallback when a window would predict nothing. Thresholds/fallback/
blend weights are selected on the January block only.

## Honest evaluation (identical rows, team protocol reused read-only)

`work/task2_seq/run_cv.py` / `lhcore.py` semantics: protocol **B** = February
pseudo-window block; protocol **A** = the 40 released train onset windows (all
June 2030, never in the fit block). Reference = the shipped onset probability
(v3+v5 blend, `w_on=0.3`) with the shipped rule (threshold 0.30 on the last step
+ top-1 fallback, i.e. config `w30ong40fb1_all`). Reference probabilities were
recomputed from scratch inside this folder: **B = 0.7656, A = 0.8162**, which
reproduces the team's `blend_weight_cv.csv` (+ `queue_w30ong40fb1_all`) exactly,
so the rows and rules are identical.

### Two-protocol table (mean per-window onset IoU)

| model | Jan tune | B (Feb pseudo) | A (released June) | Jan-selected blend |
|---|---:|---:|---:|---:|
| **reference (shipped v3+v5, w_on .3, th .30, fb1)** | **0.7835** | **0.7656** | **0.8162** | — |
| CNN small, 17ch (no long-history block), padded batches, BCE, 60ep | 0.7081 | 0.6980 | 0.7237 | w=0 (no gain) |
| CNN dilated, 100ep (overfits after ~15 epochs) | 0.6412 | 0.6084 | 0.6405 | w=0 |
| CNN small, 23ch, aug + soft-IoU loss, 80ep | 0.6899 | 0.6555 | 0.7073 | w=0 |
| CNN small, 23ch, aug, BCE, 40ep | 0.6749 | 0.6683 | 0.6914 | w=0 |

Every configuration is 0.07-0.15 below the reference on both protocols, and the
January-optimal blend weight is always **0** (pure reference).

### Fine blend sweep (best CNN, selected on January)

| w_cnn | Jan tune | B_feb | A_june | ΔB | ΔA |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.7835 | 0.7690 | 0.8162 | +0.0034 | 0.0000 |
| 0.05 | 0.7821 | 0.7704 | 0.8037 | +0.0049 | −0.0125 |
| 0.10 | 0.7805 | 0.7673 | 0.8091 | +0.0017 | −0.0071 |
| 0.25 | 0.7792 | 0.7592 | 0.7956 | −0.0063 | −0.0206 |
| 0.50 | 0.7587 | 0.7251 | 0.7893 | −0.0404 | −0.0269 |

No weight improves both protocols; the ship bar (+0.01 on **both**) fails
decisively. (The w=0 row differs from the reference row only because the
fine-sweep also re-selects the threshold/fallback on January; it is the same
reference probability.)

### Per-panel (best CNN, Feb / June, mean-IoU)

The CNN only matches the reference on 2/8 panels (B: D7_I10_E .820 vs .813,
D7_I405_N .739 vs .743) and collapses on others (B: D7_I210_W .375 vs .699,
D7_I405_S .454 vs .794; A: D7_I210_W .597 vs .883, D12_I5_S .533 vs .583).
Files: `out/detail_<tag>.csv`, `out/protocols_<tag>.csv`, `out/blendfine_aug1.csv`.

## Why it loses (diagnosis)

1. The reference is a **145-feature GBM ensemble** (v3 65 + v5 80 long-history
   features) with per-link climatology, queue-front geometry and explicit
   trends. The CNN sees the same raw window but has to learn those aggregates
   from 4,278 windows; the inductive bias of small convs is not enough.
2. Severe **overfitting**: January IoU peaks at epoch 15-24 and then declines
   while train loss keeps falling; augmentation/soft-IoU/weight decay reduce but
   do not close the gap.
3. The signal that separates onset windows is largely *static link identity +
   demand level*, which trees exploit directly; a conv that must infer it from
   channels is strictly less efficient.
4. Adding the long-history summary channels (17→23) gave the largest single
   improvement (A 0.72 vs 0.71 baseline, B +0.03 vs the 13-channel version) but
   only partial.

## Files / reproduction

```
work/task2_cnn/
  build_images.py     corridor-image caches (cache/tr_*.npz, cache/rel_*.npz)
  ref_probs.py        fits the reference v3/v5 onset GBMs (honest fit block) and
                      caches their probabilities (cache/ref_probs.npz)
  train_cnn.py        model + honest two-protocol training/evaluation
  eval_blend_fine.py  fine blend-weight sweep for a saved checkpoint
  infer_v11.py        would build out/queue_v11_{validation,private}.csv; only
                      run in --dry-run (verified: w=0 reproduces the shipped
                      baseline with 0 changed cells; w=0.25/th=0.375 would change
                      16 (validation) / 13 (private) cells of 87,000)
  models/cnn_*.pt     checkpoints (best = models/cnn_aug1_iou.pt)
  out/                protocol tables, per-window details, logs
```

```bash
python work/task2_cnn/ref_probs.py
python work/task2_cnn/build_images.py
python work/task2_cnn/train_cnn.py --epochs 80 --eval-every 4 --arch small --aug 1 --loss bce_iou
python work/task2_cnn/eval_blend_fine.py --ckpt work/task2_cnn/models/cnn_aug1_iou.pt --arch small
```

Deterministic: fixed torch/numpy seeds, deterministic cuDNN benchmark off,
fixed data order per epoch.

## Causality

Only timestamps `<= T` are read: the observed layer for `T-60..T-5`, the split's
masked layer for `-180..-65` and for row `T`, plus static network/climatology
assets. No horizon cell is ever touched; the truth for fitting/evaluation comes
from the released train plain layer (as in the v3/v5 protocol).

## Recommendation

Keep the shipped queue layer (`queue_w30ong40fb1_all_*`, current LB 0.83996
total). Do not spend more time on an image CNN for onset; if more Task-2 effort
is available it should go to better features/labels for the existing GBM
ensemble rather than to a from-scratch sequence model with 4k training windows.
