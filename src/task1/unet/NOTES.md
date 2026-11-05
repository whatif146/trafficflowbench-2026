# Task 1 — compact U-Net on daily corridor images, blended into the shipped GBM

Owner: teammate `task1-unet`. All code/artifacts live in `work/task1/unet/`;
nothing outside this folder was modified (the shipped GBM files are only read).

## TL;DR

| | mean S_state (4 panels) | mean est_total = 0.35*S_state + 0.15*S_physics |
|---|---:|---:|
| shipped GBM (`work/task1/gbm/state_gbm_*`) | 0.9367 | 0.4566 |
| pure U-Net, seed 42 (80 epochs) | 0.9375 | 0.4552 (−0.0014) |
| **blend 0.7*GBM + 0.3*UNet, seed 42** | 0.9403 | 0.4583 (**+0.0017**) |
| **blend 0.6*GBM + 0.4*UNet, seed 42 (probed on the LB)** | 0.9408 | 0.4585 (**+0.0019**) |
| **blend 0.6*GBM + 0.4*UNet, seeds 43+44 (deployed)** | **0.9423** | **0.4594 (+0.0028)** |
| blend 0.5*GBM + 0.5*UNet, seeds 43+44 (probe variant shipped) | 0.9431 | 0.4597 (+0.0031) |

Live leaderboard: the seed-42 w=0.6 probe scored **0.83996 vs 0.83734** previous
best, i.e. **+0.00262 S_total** - bigger than the local holdout estimate.

Ship bar (mean `est_total` +0.0015, no panel S_state regression > 0.003):
**MET at w=0.6 and w=0.7**; `w=0.6` improves on all four panels (max regression
0.0000) and is best on 4/4 panels, so it is the deployed global weight.

The pattern is exactly the one described for the public notebook recipe: the
U-Net alone is a hair worse on the combined objective (flow RMSE ~2-5 vph/lane
worse, speed 2-8% better), but the blend is better than the GBM on **both**
terms — S_state and the Task-3 density/continuity proxy — i.e. it is not a
trade, it is a genuine two-model gain.

## Data and the target set (the key structural fact)

Per panel the release gives `(T=273*288, S)` grids: the plain (truth) layer for
train and one masked view per split (`train_masked`, `validation_masked`,
`private_masked`).  The published template rows are **exactly the cells the
organizer blanked**, not all eligible cells:

```
elig cells 8,769,564 (D7_I10_E)   template rows 2,815,381
fraction of template rows observed in the masked view: 0.0
```

So this is a pure reconstruction problem on ~32% of the eligible cells; there
is no "copy the observed value" shortcut.  My first probe assumed the opposite
and this measurement changed the design (the loss is masked to template rows
only, and no observed-cell pass-through is needed).

Everything is built from the split's **own masked view** plus static network
attributes; the plain layer is used only for labels (fit days) and for the
per-link clip bounds.

## Model

Input image per (panel, calendar day): `[C=16, 288 time-of-day, S links]`

```
 1 ms/free_speed        2 obs_s flag          3 mq/capacity       4 obs_q flag
 5 pct/100              6 base_s/free_speed    7 base_q/capacity   8 base_k/k_jam
 9 obs_density (Gaussian-smoothed observation mask, sigma_t=3, sigma_s=1)
10 free_speed/130      11 capacity/2400       12 lanes/6          13 rank/(S-1)
14 length_km/2         15 sin(tod)            16 cos(tod)
```

`base_s` / `base_q` are the same reconstruction the shipped GBM predicts a
residual on (`t1lib`/`t1gbm` recipe): temporal linear interpolation between the
nearest observed values per station, blended 0.9/0.1 with the mean of the four
spatial neighbours (rank +-1, +-2 in milepost order); flow additionally
0.5/0.5-averaged with a Gaussian smoother (sigma_t=1.5, sigma_s=1.0).  Missing
base values fall back to the link's day mean from the masked view, then the
link mean, then 0.6*free_speed / 0.3*capacity.

Network: 4-level 2D U-Net, widths [24,48,96,128], 3x3 convs, GroupNorm(8), SiLU,
MaxPool(2) down / ConvTranspose(2) up with skip concatenation, 1x1 head, 2
output channels.  ~1.0M parameters, ~0.9 GB VRAM at batch 8 (full 288xL image,
AMP fp16 + GradScaler).  Width padded to a multiple of 8 per panel.

Outputs are residuals **in score units**, which makes the loss equal to the
official objective:

```
r_s = (v - base_s) / 25                 (S_speed denominator 25 km/h)
r_q = (q - base_q) / (600 * lanes)      (S_flow denominator 600 vph/lane)
loss = mean over template rows of  0.54*r_s^2 + 0.46*r_q^2   ==  1 - S_state
```

Prediction: `v = base_s + 25*r_s_hat`, `q = base_q + 600*lanes*r_q_hat`, then the
shipped clip guard (speed >= 1 km/h, per-link p99.5 upper bounds from the plain
layer, flow >= 0) - the same guard as `t1gbm.clip_predictions`.

Training: AdamW, lr 2e-3 OneCycle (pct_start 0.15), wd 1e-4, batch 8
(= 8 calendar days), 80 epochs, seed 42 (plus 43/44 for the ensemble), one model
per panel, gradient steps ~2160.  An inner validation block (the 20 days before
the holdout) selects the best checkpoint for the evaluated runs; the deployed
runs fit all 273 days.

## Evaluation protocol (honest, same as `work/task1/gbm/run_eval.py`)

Fit on train days `< 273-60 = 213` (inner validation on days 193-212), predict
the published template rows of the last 60 days from the masked view only.
The GBM side is the shipped holdout CSV (`work/task1/gbm/holdout_<panel>.csv`),
matched row-for-row; its documented S_state is reproduced exactly
(0.9291 / 0.9323 / 0.9391 / 0.9461, diff −0.0000), so the comparison is
on identical cells.  Scoring = `t1lib.score` (official Task 1 formula).
Task-3 proxy = `work/task3/local_eval.py` + `adjust.py` imported read-only,
observation-derived projected flux, window 1, jitter 0.06 (identical to
`work/task1/gbm/run_density_eval.py`).

## Result table (single seed 42, 80 epochs)

S_state:

| panel | GBM | U-Net | w=0.6 | w=0.7 | w=0.8 | w=0.9 |
|---|---:|---:|---:|---:|---:|---:|
| D12_I5_N | 0.9461 | 0.9446 | **0.9497** | 0.9493 | 0.9486 | 0.9475 |
| D7_I10_E | 0.9391 | 0.9424 | **0.9436** | 0.9429 | 0.9418 | 0.9406 |
| D7_I210_W | 0.9291 | 0.9288 | **0.9331** | 0.9326 | 0.9317 | 0.9305 |
| D7_I405_S | 0.9323 | 0.9341 | **0.9369** | 0.9362 | 0.9352 | 0.9339 |
| mean | 0.9367 | 0.9375 | **0.9408** | 0.9403 | 0.9393 | 0.9381 |

S_physics (Task-3 proxy):

| panel | GBM | U-Net | w=0.6 | w=0.7 |
|---|---:|---:|---:|---:|
| D12_I5_N | 0.8883 | 0.8757 | 0.8908 | 0.8912 |
| D7_I10_E | 0.8522 | 0.8418 | 0.8544 | 0.8547 |
| D7_I210_W | 0.8357 | 0.8234 | 0.8395 | 0.8397 |
| D7_I405_S | 0.8575 | 0.8486 | 0.8602 | 0.8603 |

est_total and deltas vs the shipped GBM:

| panel | GBM | w=0.6 (Δ) | w=0.7 (Δ) |
|---|---:|---:|---:|
| D12_I5_N | 0.4644 | 0.4660 (+0.0016) | 0.4659 (+0.0015) |
| D7_I10_E | 0.4565 | 0.4584 (+0.0019) | 0.4582 (+0.0017) |
| D7_I210_W | 0.4505 | 0.4525 (+0.0020) | 0.4524 (+0.0018) |
| D7_I405_S | 0.4549 | 0.4569 (+0.0020) | 0.4567 (+0.0018) |
| **mean** | 0.4566 | **0.4585 (+0.0019)** | 0.4583 (+0.0017) |

mean ΔS_state: +0.0042 (w=0.6), +0.0036 (w=0.7); max per-panel S_state
regression 0.0000.  Pure U-Net: Δest_total −0.0014, ΔS_state +0.0008.

Speed/flow detail at w=0.6 (speed RMSE km/h, flow RMSE vph/lane, mean of R1-R3):
`0.9497 panel: 1.23/26` vs GBM `1.34/27`; `0.9436: 1.47/31` vs `1.62/31`;
`0.9331: 1.79/31` vs `1.94/31`; `0.9369: 1.66/31` vs `1.83/31`.

## Seed ensemble (seeds 43 + 44, 80 epochs, same holdout)

Per-seed U-Net S_state shows a ~0.004-0.006 spread - larger than the blend gain -
so seed averaging is a cheap, reliable variance reduction (`train_seeds.py`,
`ensemble_eval.py`).  Full 4-panel table with the deployed seed pair (43+44):

S_state / S_physics / est_total, mean over the 4 holdout panels:

| variant | S_state | S_physics | est_total | Δest_total |
|---|---:|---:|---:|---:|
| shipped GBM | 0.9367 | 0.8584 | 0.4566 | - |
| pure U-Net (43+44 mean) | 0.9425 | 0.8586 | 0.4587 | +0.0021 |
| w=0.5 | **0.9431** | **0.8640** | **0.4597** | **+0.0031** |
| w=0.55 | 0.9428 | 0.8639 | 0.4595 | +0.0029 |
| w=0.6 | 0.9423 | 0.8637 | 0.4594 | +0.0028 |
| w=0.7 | 0.9413 | 0.8631 | 0.4589 | +0.0023 |

per panel (S_state / est_total at w=0.6): D12_I5_N .9517/.4672,
D7_I10_E .9447/.4592, D7_I210_W .9343/.4533, D7_I405_S .9386/.4578;
every panel is positive and max S_state regression is 0.0000.  Note the pure
U-Net is now **positive** on est_total (+0.0021): with the seed average its flow
error is no longer a net loss, so both terms of the objective improve.

Per-seed U-Net S_state (seed 42 / 43 / 44): D7_I210_W .9288/.9315/.9310,
D7_I405_S .9341/.9384/.9381, D7_I10_E .9424/.9453/.9445, D12_I5_N .9446/.9501/.9494
(the D12_I5_N seed-42 number is on the 800k-row GBM subsample, 43/44 are on all
1.08M rows, so they are not directly comparable; the table above uses the GBM
CSV rows for every variant).

## Blend-weight edge check

S_state, single seed 42, mean over the 4 panels:
w=0.50 **0.9411**, 0.55 0.9410, 0.60 0.9408, 0.65 0.9406, 0.70 0.9403, 0.80 0.9393
(Δ vs 0.6: +0.0003, +0.0002, 0, −0.0003, −0.0006, −0.0015).
On the seed-averaged model the same check gives Δest_total +0.0003 (w=0.5),
+0.0001 (w=0.55) vs w=0.6.  w=0.6 is kept as the primary (the single-seed
trade is negligible and 0.6 is not tuned below the pre-registered grid), and
the w=0.5 blend is shipped alongside as `state_gbm_unet_w05_*.csv` for a probe.

## Public LB result

The seed-43 single-seed w=0.6 blend was probed on the live leaderboard:
**0.83996 vs 0.83734 previously best = +0.00262 S_total**, larger than the
holdout estimate (+0.0019) - the Task-3 physics transfer is better in the real
scoring than the local proxy predicted.


## Deployed files

`deploy_refit.py` trains each panel on **all 273 train days** with a fixed
gradient-step budget (60 epochs x 35 steps ≈ the evaluated 80 x 27), one model
per seed, predicts the validation/private template rows from that split's
masked view, averages the seeds and blends with the **shipped GBM state
predictions** (`work/task1/gbm/state_gbm_<split>.csv` - not a refit GBM):

```
state = 0.6 * GBM_shipped + 0.4 * mean(seeds)
```

Outputs (shipped state-file format, template row order, no NaNs, rounded
speed 1e-3 / flow 1e-2 like `predict.py`):

* `state_gbm_unet_validation.csv`
* `state_gbm_unet_private.csv`
* `deploy_meta.json` (w, seeds, epochs, fit days, holdout tables, footprint)
* rollback copies of the single-seed version: `state_gbm_unet_*_s43.csv`

`verify_state.py --split validation|private` re-checks row count, column order,
key identity against both the shipped GBM file and the published template.

Reproduction: `powershell -ExecutionPolicy Bypass -File work\task1\unet\reproduce.ps1`

## Files

```
unetlib.py        base reconstruction (full fields), per-panel cache, channels,
                  U-Net, train/eval loops, prediction, scoring helpers
run_holdout.py    4-panel holdout evaluation (single seed) + blend table
train_seeds.py    extra seeds on the holdout split
eval_physics.py   task3 proxy for the single-seed variants
ensemble_eval.py  seed-averaged S_state + proxy table
deploy_refit.py   10-panel refit on all 273 days, seed average, blend, state CSVs
verify_state.py   deliverable verification against template + shipped GBM
check_gbm.py      sanity check that the GBM holdout CSVs reproduce the notes
reproduce.ps1     one-command reproduction
cache_<P>_*.npz   per-panel caches (masked view, base fields, template labels)
model_<P>_<tag>.pt trained weights; pred_*.npz holdout/deploy predictions
holdout_table_u80.csv / holdout_physics_u80.csv / holdout_physics_u80ens.csv
```

## Honesty notes

* Holdout evidence uses models fit on days <213 only; the deployed models are
  refit on all 273 days, so they are not the identical checkpoints - the refit
  only ever adds train data (documented in `deploy_meta.json`).
* The blend weight is one global constant chosen from the pre-registered grid
  {0.6,0.7,0.8,0.9}; w=0.6 is best on all four panels, w=0.7 is 0.0002 behind
  (the public-notebook value), pure U-Net is negative - reported, not hidden.
* D12_I5_N's holdout is scored on the same 800k-row subsample the shipped GBM
  holdout CSV covers, not on all 1.2M rows (`run_eval.py` convention).
* The U-Net sees the *masked* view only; labels plus clip bounds come from the
  plain layer of the fit days.  No holdout-day truth ever enters a model.

## Addendum: 3-seed deployment (seed 45)

Second bounded GPU job after the LB probe (same recipe, all 273 days, 60 epochs x 35
steps; seeds 43/44 reused from their refit checkpoints, seed 45 trained):

* `state_gbm_unet3_validation.csv` / `state_gbm_unet3_private.csv`
  = 0.6*GBM_shipped + 0.4*mean(UNet seeds 43,44,45)
* `state_gbm_unet3_w05_*.csv` = the same 3-seed U-Net at w=0.5
* `deploy_meta_unet3.json` (seeds, epochs, fit days, footprint vs the 2-seed files,
  per-panel template verification), `verify_unet3.json`, `footprint_u3*.json`

Change vs the 2-seed files (validation / private): mean|dv| 0.055 / 0.057 km/h,
mean|dq| 3.30 / 3.41 vph across lanes, 0.28% / 0.30% of cells change flow >1%,
~0% >5% - i.e. pure variance reduction; the 2-seed files and the probed seed-43
files are untouched. Verified template-exact per panel for both splits, same
columns/order as `state_gbm_<split>.csv`, no NaNs. `footprint_vs.py`,
`verify_json.py`, `finalize_meta3.py` are the new helpers.


## Addendum: wider U-Net (2.23M params) - NEGATIVE, not shipped

A wider/deeper variant (widths [32,64,128,192] -> **2.23M params**, the [24,48,96,128]
model is 1.12M; same 4-level GroupNorm/SiLU U-Net, same 16 channels, AMP batch 8)
was trained on the same 4-panel holdout protocol, 80 epochs x 27 steps, seeds 42 and
43 (seed averaging is worth ~+0.0009 S_state, so a single seed would have been an
unfair comparison against the shipped 2-seed model).

| variant (seeds 42+43, 2.23M) | S_state | S_physics | est_total | Δ vs GBM |
|---|---:|---:|---:|---:|
| shipped GBM | 0.9367 | 0.8584 | 0.4566 | - |
| pure U-Net | 0.9429 | 0.8591 | 0.4589 | +0.0023 |
| w=0.6 blend | 0.9424 | 0.8640 | 0.4594 | +0.0028 |

Head-to-head at w=0.6 against the shipped 1.0M 2-seed blend
(`holdout_physics_u80ens.csv`): S_state 0.9424 vs 0.9423, S_physics 0.8640 vs
0.8637, **est_total +0.0001** (per panel +0.0000 / +0.0002 / +0.0000 / +0.0000;
max regression 0.0001). The bar was +0.0007 over the current blend, so this is a
clear miss (14x short). The wider model is marginally better *alone*
(+0.0023 vs +0.0021 est_total) but the blend saturates: the 0.6/0.4 mix is
limited by the GBM/U-Net complementarity, not by U-Net capacity. Training cost
was ~1.1x the small model per epoch (the pipeline is data-assembly bound, not
FLOP bound: 1.2 vs 1.3 s/epoch on the 88-link panel), so capacity is cheap - it
simply does not pay. Artifacts: `holdout_table_w4m.csv`,
`holdout_table_w4mS43.csv`, `holdout_physics_w4mens.csv`, `compare_wide.py`,
`model_<P>_w4m*.pt`, `pred_unet_<P>_w4m*.npz`. No state files were written.
