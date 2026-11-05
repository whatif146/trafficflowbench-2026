# Task 1 — statistical feature model (LightGBM residual on a spatio-temporal base)

Owner: teammate `task1-gbm`. All code in `work/task1/gbm/`. Reads only the shared
cache (`work/cache` via `work/lib/traffic.py`). Nothing under `work/lib`,
`work/build_cache.py`, `work/submissions/` or the official repo is modified.

## TL;DR

Holdout protocol = `work/task1/eval.py --holdout-days 60` (fit on train days
`< n_days-60`, predict the published target rows of the last 60 days using only
the masked view, official Task 1 formula). On D7_I10_E:

| model | S_state | R1 | R2 | R3 | speed RMSE (km/h) | flow RMSE (vph/lane) |
|---|---:|---:|---:|---:|---:|---:|
| shipped historical-mean baseline (reference) | ~0.69 | | | | | |
| Lead's profile + residual smoother | 0.9092 | | | | ~3.0 | ~33 |
| my base only (interp + spatial-neighbour blend) | 0.9133 | 0.9158 | 0.9180 | 0.9060 | 2.64/2.55/3.03 | 36/35/37 |
| **my GBM residual model** | **0.9391** | **0.9414** | **0.9427** | **0.9333** | **1.62/1.60/1.97** | **31/30/31** |

Pre-clip 0.9393; the posted number is after the upper-only clip guard (0.9391).

Diagnostic for Task 3 (same holdout): target cells adjacent to a long blank run
(`dt_before >= 8` or `dt_after >= 8` steps, i.e. next to the 90-minute Task-2
window gaps): n=16,206, GBM speed RMSE **2.05** vs base 3.24. All other target
cells: n=622,583, GBM **1.77** vs base 2.78. The model halves the error exactly
where the Task-3 conservation residual lives.

## Method

### Base estimator (the GBM predicts a residual on top of it)

Per station s and time t, from the split's **masked** view only:

- `s_interp` = linear interpolation between the nearest observed speed before and
  after t (same station); nearest value if only one side exists; NaN if neither.
- `s_nb_mean` = mean of the observed speed at the four spatially adjacent links
  (rank ±1, ±2 in milepost order) at the same timestamp.
- `base_s = 0.9 * s_interp + 0.1 * s_nb_mean` (fallback `s_interp`).
- `base_q = 0.5 * q_smoother(1.5 time, 1.0 link) + 0.5 * (0.9*q_interp + 0.1*q_nb_mean)`.
  Flow benefits from more smoothing than speed.

Alone this base scores S_state 0.9133 on the D7_I10_E holdout.

### Features (60, all computed from the masked view of the split being predicted)

- temporal neighbours per station: nearest-before / nearest-after values, step
  distances, their disagreement and gap length (`s_before/after/dt_before/dt_after/
  s_disagree/s_gap` and the flow analogues);
- spatio-temporal Gaussian smoother outputs with bandwidths (time, links)
  (1.5,1), (3,0), (6,1.5) for speed and (1.5,1), (6,1.5) for flow, plus the
  smoothed value at spatial rank ±1;
- `k_sm` = smoothed density q/max(v,1) (smoothed ratio, no full k field) and
  `q_from_k = k_sm * s_sm15` (physical coupling);
- raw same-timestamp spatial neighbours (speed rank ±1,±2; flow ±1,±2),
  upstream-minus-downstream difference;
- historical weekday x time-of-day mean of speed and flow per station and link
  attributes (from the **plain layer of fit days only**), deviation of the
  smoothed value from that profile;
- day-level level: previous-day and day-2 mean observed speed/flow per station
  (from the masked view), deviation from the profile;
- quality: `pct_observed` at the cell, local observed-cell density;
- static link attributes: lanes, free_speed, capacity, k_jam, milepost rank,
  length;
- time: time-of-day index, day-of-week, mask regime, weekend flag;
- derived: smoothed speed / free_speed, smoothed flow / capacity, fine-minus-
  coarse smoother differences.

### Model

Two LightGBM regressors (speed residual, flow residual) with
`objective=l2, learning_rate=0.05, num_leaves=63, min_child_samples=100,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, lambda_l2=1.0,
n_estimators=600-700, num_threads=12, seed=42, deterministic=True,
force_col_wise=True`. Training rows: the published train target cells of the fit
days (labels from the plain layer), subsampled deterministically (400k for CV,
500k for deployment) by `np.random.default_rng(42)`.

Post-processing: speed clipped to `[1, per-link p99.5]`, flow to
`[0, per-link p99.5]` (both percentiles from the plain layer of fit days only).
No lower-quantile speed clip: congestion is real signal (a p0.5 lower clip cost
-0.010 S_state on the holdout).

### Determinism / reproducibility

Fixed seeds everywhere (`default_rng(42)`, LightGBM `seed=42`,
`deterministic=True`). One panel at a time; plain layer is freed before the
masked layer is loaded; float32 throughout; full (T,S) fields are created one at
a time inside a time-chunked smoother, so peak RSS stays under ~1 GB even on the
352-station corridors.

## Leakage guarantees

- Labels and historical profiles come only from the **plain layer of the fit
  days**. The holdout days' plain layer is touched only to *score*, never to
  build features, labels, profiles or clip bounds.
- Every feature of a predicted cell comes from the **masked view** of the split
  being predicted (the input we are given) plus the train plain layer (released
  data), plus static network attributes.
- The two-fold cross-fitting used for the `train` deliverable predicts each
  half of the train days with a model fitted on the other half only (profiles
  and clip bounds also restricted to that half), so no train row is predicted
  in-sample.

## Files

```
t1lib.py        time helpers, row-level temporal/spatial neighbours, chunked
                Gaussian spatio-temporal smoother, official scoring formula
t1gbm.py        profiles, clip bounds, day means, FeatureBuilder (60 features),
                LightGBM params, clip guard
run_eval.py     holdout evaluation (fit excludes last N days) + gap diagnostic
base_tune.py    comparison of candidate base estimators on the holdout
explore*.py     exploratory analyses (temporal interp, error structure, grids)
predict.py      DELIVERABLE: train / validation / private, template-exact CSVs
```

## Deliverables / commands

```powershell
$env:PYTHONPATH='C:\Users\lenovo\Desktop\kaggle\libs'
$py = 'C:\Users\lenovo\AppData\Local\Programs\Python\Python313\python.exe'

# scored splits, all 10 panels, one panel at a time (~3-4 min/panel)
& $py work\task1\gbm\predict.py --split validation private
# -> work/task1/gbm/state_gbm_validation.csv, state_gbm_private.csv

# train-split predictions (2-fold cross-fitted, full template per panel)
& $py work\task1\gbm\predict.py --split train --panels D7_I10_E D12_I5_N D7_I210_W
# -> work/task1/gbm/state_gbm_train_<PANEL>.csv

# holdout comparison with the Lead's evaluator
& $py work\task1\gbm\run_eval.py --panel D7_I10_E --holdout-days 60 --train-cap 400000 \
      --eval-cap 800000 --n-estimators 700 --csv work\task1\gbm\holdout_D7_I10_E.csv
& $py work\task1\eval.py --submission work\task1\gbm\holdout_D7_I10_E.csv \
      --panels D7_I10_E --holdout-days 60
```

## Multi-panel holdout table

Same protocol for every row: `work/task1/eval.py --holdout-days 60` (fit on train
days `< n_days-60`; predict the published target rows of the last 60 days from
the masked view; official Task 1 formula). Numbers are eval.py's own output
(holdout-only prediction CSVs `holdout_<PANEL>.csv` are kept for ensembling).

| panel | stations | S_state | R1 | R2 | R3 | speed RMSE R1/R2/R3 (km/h) | flow RMSE/lane R1/R2/R3 |
|---|---:|---:|---:|---:|---:|---|---|
| D12_I5_N | 258 | **0.9461** | 0.9506 | 0.9479 | 0.9398 | 1.34 / 1.45 / 1.76 | 27 / 27 / 29 |
| D7_I10_E | 146 | **0.9391** | 0.9414 | 0.9427 | 0.9333 | 1.62 / 1.60 / 1.97 | 31 / 30 / 31 |
| D7_I405_S | 109 | **0.9323** | 0.9363 | 0.9336 | 0.9271 | 1.83 / 1.96 / 2.22 | 31 / 31 / 33 |
| D7_I210_W | 81 | **0.9291** | 0.9342 | 0.9332 | 0.9198 | 1.94 / 2.01 / 2.46 | 31 / 30 / 32 |

(D12_I5_N evaluation capped at 800k random holdout rows, seeded; the panel has
>1.2M holdout target rows. The other panels are scored on all holdout rows.)

### Where the headroom is: cells next to long blanked runs (Task-2 window gaps)

Speed RMSE on holdout target cells split by `dt_before >= 8 or dt_after >= 8`
steps (a 40+ minute blank on one side, i.e. adjacent to the 90-minute Task-2
window removals). This is the subset whose temporal difference drives the
Task-3 conservation residual.

| panel | gap-adjacent n | GBM speed RMSE | base speed RMSE | other n | GBM | base |
|---|---:|---:|---:|---:|---:|---:|
| D7_I10_E | 16,206 | **2.05** | 3.24 | 622,583 | 1.77 | 2.78 |
| D12_I5_N | 20,376 | **2.00** | 3.59 | 779,624 | 1.56 | 2.69 |
| D7_I405_S | 11,944 | **2.44** | 3.90 | 463,103 | 2.04 | 3.24 |
| D7_I210_W | 10,286 | **3.16** | 4.67 | 364,572 | 2.23 | 3.52 |

The GBM roughly halves the error on the gap-adjacent subset; that is what lifts
the Task-3 physics proxy from 0.593 to 0.852.

## Train-split deliverable (2-fold cross-fitted)

`state_gbm_train_<PANEL>.csv` (full template, one file per panel) for D7_I10_E,
D12_I5_N, D7_I210_W, D12_I405_S. Train days are split into two contiguous
halves; the rows of each half are predicted by a model fitted only on the other
half, with historical profiles and clip bounds also restricted to that half.
Consequence: **no train row is predicted in-sample**, so the files are suitable
for calibrating Task-3 post-processing (a single in-sample model would make the
train physics look far better than deployment and bias any calibrated adjuster).
Note the training set for each fold is half of the data, so these predictions
are slightly *worse* than the deployment model used for validation/private; they
are an honest approximation of deployment behaviour, not the exact deployed
predictions.

## Experiment log (holdout, eval.py --holdout-days 60)

### v5: distribution shift — Task-2 horizon blanks in the scored-split input — NEGATIVE, no ship

Measured with `probe_shift.py` on the released masked layers vs the Task-1
templates (extra = cell blanked in speed+flow, pct_observed>=75, not a Task-1
target):

| split | D7_I10_E extra blanks (% of template) | D12_I5_N extra blanks (% of template) |
|---|---:|---:|
| train | **0** | **0** |
| validation | 11,675 (3.46%) on 6 days | 25,295 (4.30%) on 8 days |
| private | 12,653 (4.17%) on 6 days | 25,655 (4.72%) on 7 days |

Shape: exactly **10 runs per split-month** (= the 10 Task-2 windows), each **18
consecutive 5-min steps** starting at `forecast_origin+1 .. +18`, with the
entire step blanked (all links, both channels). Offsets +0 and >=+19 have zero
extra blanks. Train's mask simulation blanks only the R1/R2/R3 target cells, so
the scored splits genuinely have a different missingness pattern.

Holdout with validation-like blocks (`run_eval_shift.py`: 19 sampled origins over
the 60 holdout days = same 10-per-31-days rate, 18 full steps each; this puts
1.8-2.3% of holdout target rows inside blocks, matching the real splits):

| panel | A plain R-mask | A val-like mask | shift cost | B retrained with blocks | recovery |
|---|---:|---:|---:|---:|---:|
| D7_I10_E | 0.9391 | 0.9342 | -0.0049 | 0.9353 | +0.0011 |
| D12_I5_N | 0.9461 | 0.9381 | -0.0080 | 0.9407 | +0.0026 |
| D7_I405_S | 0.9323 | 0.9277 | -0.0047 | 0.9284 | +0.0008 |
| D7_I210_W | 0.9291 | 0.9242 | -0.0049 | 0.9262 | +0.0020 |
| **mean** | **0.9367** | **0.9311** | **-0.0056** | **0.9327** | **+0.0016** |

Bar was a mean recovery >= +0.002 under validation-like blanks: **+0.0016, not
met** (no panel regresses; the retrain is a small consistent win, but too small).
A stronger variant (3x augmentation rate on the fit days + the 14 day-similarity
features of the v3 experiment) does *not* help: recovery +0.0010/+0.0011 on
D7_I10_E/D12_I5_N (2-panel mean +0.0011) and it starts to regress the plain-mask
case (-0.0020 on D12_I5_N). Therefore **no v4 files were produced**; the shipped
`state_gbm_{validation,private}.csv` are untouched.

Robustness statement: the shipped model is **not fully robust** to this shift —
under the validation-like input it loses ~0.0056 S_state on the holdout. But
almost all of that is irreducible information loss: the blocks blank 18
consecutive steps at *every* link, so at those steps both the temporal and the
spatial neighbours are unavailable and only historical/profile/day-neighbour
information remains. Retraining at the matched rate recovers only ~30% of the
loss (+0.0016 of 0.0056), and extra augmentation/features do not improve that.
Practical consequence for expectations: the true LB `S_state` on the 6-8 window
days per month should be ~0.005 below the plain-mask holdout number, since the
scored splits ship exactly this pattern.

Artifacts: `probe_shift.py`, `run_eval_shift.py`, `shift_holdout.csv`,
`shift_holdout_strong.csv`.

### v4: density-target model (Task-3-aware, q reconstructed as k_hat*v_hat) — NEGATIVE (-0.0029)

Rationale (from work/task3/NOTES.md §1): the Task-3 conservation term is an L1
penalty on N=(q/v)L, so predicting density directly might beat predicting flow.
`run_density_eval.py` trains a third LightGBM on `k = q/v` with the same feature
matrix/config as the shipped model (L2 residual on k vs the base density
`base_q/base_s`, 400k rows, 700 trees, seed 42), keeps the shipped speed model,
and reconstructs `q = clip(k_hat * v_hat, 0, per-link p99.5 flow)`. k_hat is
clipped to the per-link p99.9 observed k of the fit days. Variants:
a = shipped (q_flow), b = density (k_hat*v_hat), c = 0.5/0.5 blend (single,
untuned alpha). Holdout rows, 4 panels; S_state is the official formula; the
physics proxy is the task3 team's frozen holdout proxy
(`local_eval.py` + `adjust.py`, imported not modified: observation-derived
projected flux, window 1, jitter 0.06).

| panel | variant | S_state | flow RMSE/lane | S_FD | S_LWR | S_physics | est_total |
|---|---|---:|---:|---:|---:|---:|---:|
| D7_I10_E | a | **0.9391** | 30.7 | 0.9820 | **0.7873** | **0.8522** | **0.4565** |
| | b | 0.9359 | 35.0 | 0.9819 | 0.7852 | 0.8508 | 0.4552 (-0.0014) |
| | c | 0.9383 | 31.8 | 0.9820 | 0.7887 | 0.8531 | 0.4564 (-0.0002) |
| D12_I5_N | a | **0.9461** | 27.6 | 0.9904 | **0.8372** | **0.8883** | **0.4644** |
| | b | 0.9370 | 39.5 | 0.9904 | 0.8301 | 0.8835 | 0.4605 (-0.0039) |
| | c | 0.9437 | 30.8 | 0.9904 | 0.8363 | 0.8877 | 0.4635 (-0.0009) |
| D7_I405_S | a | **0.9323** | 31.8 | 0.9850 | **0.7938** | **0.8575** | **0.4549** |
| | b | 0.9276 | 37.9 | 0.9850 | 0.7892 | 0.8545 | 0.4528 (-0.0021) |
| | c | 0.9311 | 33.5 | 0.9850 | 0.7939 | 0.8576 | 0.4545 (-0.0004) |
| D7_I210_W | a | **0.9291** | 32.3 | 0.9833 | **0.7620** | **0.8357** | **0.4505** |
| | b | 0.9205 | 43.4 | 0.9832 | 0.7500 | 0.8277 | 0.4463 (-0.0042) |
| | c | 0.9270 | 35.0 | 0.9833 | 0.7600 | 0.8344 | 0.4496 (-0.0009) |
| **mean est_total** | a/b/c | | | | | | **0.4566 / 0.4537 / 0.4560** |

Deltas (b-a) mean over panels: est_total **-0.0029**, S_state **-0.0064**
(max regression -0.0091 > the 0.003 allowed), S_physics -0.0043 and S_LWR
-0.0064; S_FD is flat to 1e-4. The 0.5 blend halves the damage (-0.0006) but
still loses. Bar (+0.001 est_total) is not met: **negative, no v3 files**.

Task-3-independent sanity check: the variant does not even trade flow accuracy
for physics — flow RMSE/lane rises 14-57% (e.g. D12_I5_N 27.6 → 39.5) *and*
S_LWR falls on all four panels, so there is no "helps where Task-3 penalises
less" gaming effect to report; it is a strict loss. Why: derived density of the
baseline is q_flow/v_hat, and cell-level L1 density error of the density model
is not better (holdout k MAE a vs b: 1.067/1.096, 1.190/1.278, 1.384/1.405,
1.199/1.316), while q = k_hat*v_hat re-injects the speed model's ~2% error into
the flow. The identity q = v*k means jointly-predicted channels cannot beat a
good direct flow model once flow noise is already at the 2.3% multiplicative
floor.

Changed-cell footprint (if it had been shipped, holdout cells): 26-33% of cells
change flow by >1%, 1-2% by >5%, mean |dq|/lane 7.5-9.4 — large enough that an
LB probe would resolve it, but the direction is negative. Shipped v1 state files
were not touched. Artifacts: `run_density_eval.py`,
`density_variant_holdout.csv`, `holdout_dense_{a,b,c}_<PANEL>.csv`.

### v3: day-similarity / low-rank tensor completion stack — NEGATIVE (+0.0006)

`daysim.py` implements two month-tensor estimators built only from the masked
view of the split being predicted (`run_eval_ds.py`):

1. **Day-similarity completion** (`DaySimFeatures`): template [day, 288, link];
   for a target cell (d, h, s) take the similarity-weighted average of the same
   (h, s) over the other days, weights from the correlation of the days' observed
   speed/flow time-of-day profiles (weekday boost 2x, diagonal removed), with a
   day-level offset adjustment `+ dmean(d,s) - weighted_mean_d' dmean(d',s)`.
   Plus same-tod values at d±1, d±7 and the all-day same-tod mean: 14 features.
2. **Low-rank soft-impute completion** (`LRTFeatures`): randomized-SVD rank-12
   soft-impute of the [day, 288*link] matrix (10 iterations), 2 features.

Results (same 4 holdout panels, control = current GBM 400k rows / 700 trees):

| panel | v1 GBM | +day-sim | **+day-sim +LRT (16 feats)** | delta |
|---|---:|---:|---:|---:|
| D7_I10_E | 0.9391 | 0.9390 | **0.9401** | +0.0010 |
| D12_I5_N | 0.9461 | – | **0.9465** | +0.0004 |
| D7_I405_S | 0.9323 | – | **0.9325** | +0.0002 |
| D7_I210_W | 0.9291 | – | **0.9299** | +0.0008 |
| **mean** | **0.9367** | | **0.9373** | **+0.0006** |

Below the +0.003 shipping bar (no panel regresses, all deltas are tiny positive),
so no `state_gbm_v3_*.csv` was produced.
Stand-alone estimates are far behind the GBM: day-sim `S_state` 0.772,
LRT rank-12 0.887 on D7_I10_E; blending any of them with the GBM always selected
w = 1.0 (pure GBM). LRT rank 20/30 with more iterations did not improve its
stand-alone quality (0.877-0.881), so the failure is structural, not a tuning
issue: the within-day temporal/spatial neighbours the GBM already uses carry
almost all of the information, and cross-day same-time-of-day structure is
already captured by the weekday x TOD profile features. A further practical
argument against shipping it: day-similarity weights in training see 273 donor
days while validation/private only offer 30-31 month days, so the holdout
estimate would be optimistic anyway.

### Ensemble alpha with the base estimator — NEGATIVE

`pred = alpha*GBM + (1-alpha)*base`, mean S_state over the four holdout panels
(D12_I5_N, D7_I10_E, D7_I405_S, D7_I210_W), computed by `blend_eval.py`:

| alpha | 1.00 | 0.95 | 0.90 | 0.80 | 0.70 | 0.60 |
|---|---:|---:|---:|---:|---:|---:|
| mean S_state | **0.9367** | 0.9364 | 0.9360 | 0.9347 | 0.9328 | 0.9302 |

Monotone: any shrink toward the base hurts, so pure GBM is used everywhere.
The per-panel base-only CSVs are saved as `holdout_base_<PANEL>.csv`.

### Bigger model (v2) — +0.0024, below the 0.003 ship bar

train_cap 900k (vs 500k), n_estimators 900 (vs 600-700), num_leaves 95 (vs 63),
learning_rate 0.045, same seeds; evaluated with `run_eval.py`:

| panel | v1 S_state | v2 S_state | delta |
|---|---:|---:|---:|
| D7_I10_E | 0.9391 | 0.9415 | +0.0024 |
| D12_I5_N | 0.9461 | 0.9485 | +0.0024 |
| D7_I405_S | 0.9323 | 0.9344 | +0.0021 |
| D7_I210_W | 0.9291 | 0.9316 | +0.0025 |
| **mean** | **0.9367** | **0.9390** | **+0.0024** |

Consistently positive on all four panels but below the agreed 0.003 shipping
threshold, so the shipped `state_gbm_{validation,private}.csv` stay on the v1
configuration (500k rows, 600 trees). v2 can be reproduced with
`predict.py --train-cap 900000 --n-estimators 900 --num-leaves 95
--learning-rate 0.045 --outprefix state_gbm_v2`.

### Physics blend (Task-3-aware) — NEGATIVE, pure GBM wins

`physics_blend.py` builds `alpha*GBM_crossfit + (1-alpha)*base` over the full
D7_I10_E train template (2,815,381 rows) from the 2-fold cross-fitted predictions
and writes one CSV per alpha under `blend_train/`. S_state is the official
full-train score of that file; S_physics is
`work/task3/local_eval.py --panel D7_I10_E --flux projected --flux-window 1
--flux-jitter 0.06` (the task3 team's calibrated proxy; `--flux smooth` is
useless — it floors S_LWR at 0 for everyone).

| alpha | S_state | S_FD | S_LWR | S_physics | 0.35*S_state + 0.15*S_physics |
|---:|---:|---:|---:|---:|---:|
| **1.0 (GBM)** | **0.9190** | 0.9827 | **0.5640** | **0.7036** | **0.4272** |
| 0.9 | 0.9186 | 0.9827 | 0.5441 | 0.6903 | 0.4250 |
| 0.8 | 0.9178 | 0.9827 | 0.5393 | 0.6871 | 0.4243 |
| 0.7 | 0.9165 | 0.9826 | 0.5320 | 0.6822 | 0.4231 |
| 0.5 | 0.9129 | 0.9823 | 0.5123 | 0.6690 | 0.4199 |

Blending toward the base lowers both S_state and the physics proxy, so
alpha = 1.0 (the shipped GBM) is optimal for the combined objective as well.
(Full-train S_state here, 0.9190, is lower than the 0.9391 holdout number because
the cross-fitted files are half-data models — honest, but slightly worse than
deployment.)

## FINAL: shipped files, exact reproduction, determinism

### Shipped scored-split files (do not regenerate/overwrite)

| file | rows | panels | status |
|---|---:|---:|---|
| `state_gbm_validation.csv` | 3,451,376 | 10/10 | template-exact, verified by `verify_outputs.py` |
| `state_gbm_private.csv` | 3,284,419 | 10/10 | template-exact, verified by `verify_outputs.py` |

Both were written by **all-default arguments** of `predict.py`, i.e.:
`--train-cap 500000 --n-estimators 600 --num-leaves 63 --learning-rate 0.05
--seed 42` (the remaining LightGBM params are `LGB_PARAMS` in `t1gbm.py`:
`objective=l2, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
lambda_l2=1.0, min_child_samples=100, num_threads=12, deterministic=True,
force_col_wise=True`), trained on all 273 train days' published target cells.
They contain full-precision floats and were flushed only at process exit (this
is why a reader that sampled the file mid-write saw a truncated tail; the files
on disk now are complete and verified).

Exact reproduction command (current code; same models, same rows, output values
rounded to speed 1e-3 / flow 1e-2 and flushed per panel):

```powershell
$env:PYTHONPATH='C:\Users\lenovo\Desktop\kaggle\libs'
python work\task1\gbm\predict.py --split validation private --outdir work\task1\gbm
```

Determinism was checked without touching the shipped files: `fix/state_gbm_*.csv`
is a default-argument regeneration of the D12_I405_S panel (made after the
rounding/flush patch), compared against the shipped full-precision files by
`verify_repro.py`:

```
validation: 93,718/93,718 rows, keys match, max|d_speed|=0.0005, max|d_flow|=0.005
private:   107,416/107,416 rows, keys match, max|d_speed|=0.0005, max|d_flow|=0.005
```

i.e. identical apart from the documented rounding half-units ⇒ same model
config, deterministic.

### Outcome

These files were used in the team's live submission (public LB 0.82448);
the Lead's attribution is that the Task-1 GBM added about +0.027 to S_total on
the LB and +0.12..0.26 to S_physics under the calibrated proxy. Task 1 is at the
measurement-noise-dominated ceiling (speed RMSE 1.3-2.0 km/h against a
measurement sigma of roughly 1.3-2.1 km/h), so v2/ensemble/physics blends were
evaluated and rejected (negative or < 0.003 ship bar) rather than shipped.

### File index

```
t1lib.py            time helpers, row-level temporal/spatial neighbours, chunked
                    Gaussian spatio-temporal smoother, official scoring formula
t1gbm.py            profiles, clip bounds, day means, FeatureBuilder (60 features),
                    LightGBM params, clip guard
predict.py          DELIVERABLE: train / validation / private CSVs, deterministic,
                    memory-lean, per-panel flush, template verification before write
run_eval.py         holdout evaluation (fit excludes last N days) + Task-2-gap
                    diagnostic (gap-adjacent vs other target cells)
base_tune.py        candidate base estimators compared on the holdout
blend_eval.py       GBM/base ensemble alpha test; also writes holdout_base_<PANEL>.csv
daysim.py           v3 experiment: day-similarity completion + low-rank soft-impute
run_eval_ds.py      v3 experiment driver: GBM + day-sim/LRT stack on the holdout
run_density_eval.py v4 experiment: density-target model + holdout physics proxy
density_variant_holdout.csv / holdout_dense_{a,b,c}_<PANEL>.csv  v4 artifacts
physics_blend.py    Task-3-aware alpha blend over the full D7_I10_E train template
verify_outputs.py   streaming per-panel row-count check of a state CSV vs templates
verify_repro.py     determinism check of default-arg regeneration vs shipped files
explore*.py         exploratory analyses (interp baseline, error structure, grids)
state_gbm_{validation,private}.csv          SHIPPED scored-split predictions
state_gbm_train_<PANEL>.csv                 train-split cross-fitted predictions
holdout_<PANEL>.csv / holdout_base_<PANEL>.csv   holdout predictions (GBM / base)
fix/                                        default-arg regeneration of D12_I405_S
blend_train/                                physics-blend experiment files
```




