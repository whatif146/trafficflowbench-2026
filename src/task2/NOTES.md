# Task 2 — online queue-propagation forecasting

Owner: **task2-queue**. All deliverables live in `work/task2/`.

Causal guarantee: every feature is computed from the released 12-step history
`T-60 … T-5` of the window plus the static network assets (`links.csv`,
`fd_parameters.csv`, ramp attachment map). No data stamped after `T-5` is used,
which is stricter than the rules require (the row at `T` would be allowed). The
released window history is exactly the observable layer, so the training features
built from the train `mainline_states` layer and the submission features built
from `window_history` are the same function of the same data.

## What is predicted

Binary queue status for every mainline link x the six future steps `T+5 … T+30`
(5-min grid), for each released window. Queue = `speed <= v_cut` with
`v_cut = 0.60*free_speed_kmh` per link from `fd_parameters.csv`
(`traffic.load_network(panel)["links"]`). Score per window = space-time IoU over
the score-eligible truth cells; equal mean over conditions, then panels, then the
four scored corridor families (D12_I405_N/S are not part of Task 2).

## Findings that drive the design (all verified on the released data)

1. **The released window selector is reproducible.** A chronological greedy scan
   over candidate origins, accepting the earliest origin that (a) satisfies the
   condition definition, (b) has a queue somewhere in the horizon, (c) sits at
   least 360 min after the previous accepted window of the same condition,
   reproduces **all five** `queue_onset` windows of the train split for every
   panel exactly (5/5 panels checked bit-exact before the cache was complete,
   plus every panel's window set later). Condition = "the visible history holds
   >=2 queued observations with >=2 on one link -> ongoing, else onset". The
   coverage gate is `>=0.7` and equals `window_index.history_coverage`.
2. **Onset windows are exactly 30 minutes before the queue starts.** The scan is
   greedy-earliest, so an onset window's horizon queue lies only at the last step
   `T+30`; steps `T+5 … T+25` are empty. Verified on all 40 released train onset
   windows (8 panels). The onset model therefore predicts only the last horizon
   step. Predicting earlier steps can only add to the IoU union.
3. **The onset queue set is a small recurrent hotspot set.** Per-link frequency
   of being queued at `T+30` across accepted onset windows is concentrated on
   1-7 links and is stable over the year (halves correlation 0.98-0.999): e.g.
   D7_I10_W is essentially always `L10W-212`; D7_I210_E is 5 links; D7_I405_N is
   `L405N-041/145`. A static hotspot predictor already reaches onset IoU ~0.61-0.75
   while the persistence baseline scores exactly 0 on onset.
4. **Predicting a superset of the queue set is only free through ineligibility.**
   The scorer drops predictions on cells that are not in the truth file
   (ineligible future cells cost nothing), while an eligible non-queued cell adds
   to the union. Decision rules were therefore calibrated directly on pseudo-IoU
   rather than on log-loss.
5. **The train plain layer has no window gaps.** `window_history` equals the
   train `mainline_states` observations at the released windows exactly (speed
   MAE ~2e-6, identical missingness), and the future horizon speeds are present
   there, which is what makes the pseudo-labels possible. Only the
   validation/private observation layers have the horizon cells blanked.

## Supervised data

Per panel, from the train split (2030-06-01 … 2031-02-28, 273 days):

* **Labels**: per-link NaN-aware centred rolling median (3 steps) of observed
  speed, thresholded at `v_cut`, masked to `is_score_eligible` (about 17% of
  cells are unobserved / imputed and are excluded; the scorer only sees eligible
  cells).
* **queue_onset windows**: every selector-valid candidate origin in the training
  block (subsampled 1-in-4 chronologically) plus all greedy-accepted windows ->
  5,858 windows, one supervised row per link (last horizon step only).
* **queue_ongoing windows**: greedy-accepted windows, fit part subsampled
  1-in-4 -> 1,284 windows, six rows per link (horizon steps x links).
* Training rows: 940,058 onset / 951,432 ongoing (fit block, i.e. dates before
  the tuning block). February windows are never used for fitting or rule choice.

65 features (`features.py`), all from history:

```
speed level/trend:   r_last r_med3 r_med6 r_min r_mean dr15 dr30 dr_all
flow/density:        f_last_cap f_mean_cap f_trend d_last d_mean d_ratio
queue history:       qnow qmed3 qmed6 qmean3 qmean6 qcount n_nan
rank neighbours:     r_last_up1 r_last_dn1 r_med3_up1 r_med3_dn1
                     qnow_up1 qnow_dn1 qnow_up2 qnow_dn2 qnow_up3 qnow_dn3
                     qnow_up5 qnow_dn5 qsum_up5 qsum_dn5 dist_up dist_dn
climatology:         rate_tgt rate_last onset_freq ong_freq elig_rate
                     qnow_rate_tgt r_anom_last r_anom_med3 fcap_anom_last
                     fcap_anom_mean
static link:         rel_rank lanes free_speed capacity k_jam length_km
                     on_ramp off_ramp cap_per_lane
window context:      corr_r_last corr_f_last_cap nq_hist coverage
horizon step:        hour_sin hour_cos step_k is_ongoing panel_id
```

`r_*` = speed / `v_cut`; `d_*` = density = flow/speed normalised by `k_jam`;
`rate_*` = per-link 288-slot time-of-day queue rate; `*_anom` = deviation from the
per-link time-of-day mean; rank order comes from `lwr_mainline_topology.csv`
(`order_index`, upstream -> downstream), never from the id sort.

## Model

Two LightGBM binary models, all eight panels jointly (panel id as a categorical
feature), `objective=binary`, `num_leaves=127`, `lr=0.05`,
`min_data_in_leaf=40`, `feature_fraction=0.8`, `bagging_fraction=0.8`,
`seed=42`, `deterministic=true`, `force_row_wise=true`; 450 trees (onset), 700
trees (ongoing), refit on all train windows for the submission.

* **onset**: `P(link queued at T+30)` per (window, link), last horizon step only.
* **ongoing**: `P(link queued at step k)` per (window, link, step), six steps, so
  persistence and dissipation are both learned (step index + per-step targets).

### Decision rules (selected on the January block, then frozen)

| condition | rule |
|---|---|
| queue_onset | predict every link with `p >= 0.30` at `T+30` only (about 4 links/window) |
| queue_ongoing | predict every link with `p >= 0.25` at every one of the six steps |

The rules are global (per condition); per-panel thresholds were tested and did
not help (Jan selection noise exceeded the gain).

## Validation protocol (honest)

Queue truth is withheld for every split, so local scoring uses two proxies:

1. **Primary: February pseudo-CV.** Models are trained on windows from train days
   before 2031-01-01, the decision rule and model config are selected on the
   January block, and the score is computed exactly once on the untouched
   February block with the official nesting (window -> panel x condition ->
   panel -> family -> 4 families).
2. **Secondary: released-train-window proxy.** Score on the actual 80 released
   train windows with the smoothed observed horizon of the train plain layer as
   truth. Optimistic (the submission model is refit on all train windows,
   including those), reported for reference only.

## Results

### Primary pseudo-CV (February 2031 hold-out, official nesting)

| method | overall | onset | ongoing |
|---|---:|---:|---:|
| persistence (score on same windows; official validation baseline 0.3017) | 0.3161 | 0.000 | 0.632 |
| static v1: hotspot links + TOD climatology persistence | 0.6530 | 0.6130 | 0.7109 |
| LightGBM v1: 2,131 greedy-accepted onset windows, 61 features | 0.8043 | 0.7525 | 0.8720 |
| LightGBM v2: all candidates + anomaly/tod-profile features, base config | 0.8024 | 0.7583 | 0.8653 |
| **LightGBM v3 (FINAL): all candidates, 127-leaf config, th 0.30/0.25** | **0.8071** | **0.7572** | **0.8736** |
| LightGBM v4 (experiment): v3 + 6 ramp-flow features (71 features) | 0.7990 | 0.7471 | 0.8728 |

v4 (attached on/off-ramp demand features) was rejected: it did not improve the
honest February CV (and its January tuning score was also lower), so the final
model is v3. The 6 ramp features remain implemented in `features.py` /
`build_dataset.py` for the record; v3 uses the 65-column subset listed in
`models/v3/gbm_meta.json["features"]` (predict-time column selection handles it).

Also tested and rejected on the honest split (see `exp_rules.py`, `out/exp_rules.json`
and `out/exp_rules.log`; it retrains the frozen v3 configuration on the pre-January
windows and reproduces the 0.8071 baseline exactly):

* **explicit set-size calibration** (predict the top-k links instead of
  thresholding the probability): per-panel or global k selected on January, or
  k from an estimated expected size `k = round(alpha * sum(p))` for alpha in
  {0.6, 0.8, 1.0, 1.25, 1.5}, for both conditions. Everything scored below the
  plain probability threshold on the January block (onset: threshold 0.30 ->
  0.7662 vs best top-k/alpha rule 0.7088; ongoing: threshold 0.25 -> 0.8984 vs
  best size rule 0.8349), so the delivered rule stays a probability threshold.
  Interpretation: the calibrated per-link probability threshold already adapts
  the predicted set size per window, and `sum(p)` is a poor size estimator
  because the uncertain links carry much of that mass.
* **hybrid GBM + train climatology ranking** (rank by `w*p + (1-w)*onset_freq`,
  w in {0.3,0.5,0.7}, January): 0.7034 at best, worse than pure GBM 0.7662.
* **per-panel decision thresholds** (chosen on January, applied to February:
  onset 0.848 -> 0.846, ongoing 0.857 -> 0.851; selection noise exceeded the
  gain) and per-panel probability scaling: both rejected.
* February diagnostic only (not deployed, would be tuning on the reporting
  block): the best possible February threshold is onset 0.20 (0.7593 vs 0.7572
  at 0.30) and ongoing 0.30 (0.8741 vs 0.8736). The deployed rules are within
  noise of the February optimum, so rule/threshold tuning has no headroom left;
  the remaining error is the forecast itself.

The deployed rules are therefore global per condition and the final model is v3.

### Month-adaptivity / prior-shift robustness (rejected, `month_adapt.py`)

The last hypothesis tested was that the validation/private month's queue
realisation differs from the train months in a way the February hold-out cannot
see, and that a month-specific climatology from the split's own released
observation layer (window gaps are NaN, hence excluded) could correct it.

**Part A - is the target month an outlier?** Per-link observed queue frequency
from the split's masked layer vs the train reference (mean of the last four train
months), measured as Spearman correlation and overlap of the top-8 hotspot links.
Reference scale = how different train months are from each other:

| panel | train month-to-month sp (min/mean/max) | val sp / top8 overlap | priv sp / top8 overlap |
|---|---:|---:|---:|
| D7_I10_E | .67/.82/.95 | .86 / .50 | .98 / .88 |
| D7_I10_W | .61/.75/.89 | .90 / .75 | .36 / .62 |
| D7_I210_E | .14/.47/.77 | .58 / .75 | .60 / .75 |
| D7_I210_W | .62/.76/.91 | .84 / .88 | .84 / .88 |
| D7_I405_N | .99/1.00/1.00 | .99 / .88 | .99 / .88 |
| D7_I405_S | .83/.89/.95 | .88 / .12 | .69 / .12 |
| D12_I5_N | .88/.94/.99 | .76 / .62 | .57 / .00 |
| D12_I5_S | .91/.94/.98 | .83 / .25 | .83 / .75 |
| mean | .71/.82/.93 | .83 / .59 | .73 / .61 |

So the validation/private months are **not** outliers: their structural agreement
with train (~0.83 / 0.73 Spearman) is at the same level as train months are with
each other (~0.82), and the observed fraction is comparable (0.57-0.60). The
hotspot *sets* do drift somewhat (top-8 overlap 0.59-0.61), most visibly on
D7_I405_S and D12_I5_N/S - the same panels that are hardest in the pseudo-CV, so
part of their weakness is genuine scenario drift rather than model error.

**Part B - leave-one-month-out adaptation.** A causal month-to-date climatology
is built per link and time-of-day slot from the masked layer of days *before* each
window's origin day, then the frozen v3 probabilities (weights trained on
pre-January windows only) are shifted by
`lam * (logit(month_rate) - logit(train_rate))`. Sweep on the February LOMO
block (official nesting):

| lam | overall | onset | ongoing |
|---:|---:|---:|---:|
| 0.00 (v3) | 0.8071 | 0.7572 | 0.8736 |
| 0.10 | 0.8076 | 0.7580 | 0.8739 |
| **0.20** | **0.8087** | 0.7621 | 0.8721 |
| 0.30 | 0.8075 | 0.7601 | 0.8724 |
| 0.50 | 0.8059 | 0.7587 | 0.8709 |
| 0.80 | 0.7995 | 0.7500 | 0.8669 |
| 1.20 | 0.7945 | 0.7449 | 0.8620 |

The best adaptation is +0.0016 overall at lam = 0.2 (well under the +0.01
promotion bar) and over-weighting clearly hurts, so **v5 was not produced**: the
month-adaptive adjustment is neutral-to-negative and the v3 artefacts stay final.
Diagnostic files: `out/month_structure.csv`, `out/month_adapt.json`,
`out/month_adapt.log`.

Note on which model is frozen: the submitted v3 artefacts are refit on *all*
train windows (including February), which is legitimate for validation/private
prediction because they are legal "past" data for those months. Consequently
scores computed on train windows with the refit model (e.g. the released-train
proxy below, or February scored with the refit model) are optimistic; the honest
number is the 0.8071 from the pre-January-fitted model reported above.

Live leaderboard context (Lead's submissions): the first full submission scored
public 0.79509 with the v2 queue model; swapping in v3 raised it to 0.79775
(+0.0027 total, i.e. +0.009 Task-2 IoU), which confirms the pseudo-CV ranking.
The implied Task-2 validation IoU is ~0.75-0.76 versus the 0.807 honest
pseudo-CV, i.e. the validation month is somewhat harder but the method transfers;
Part A above shows it is within the natural month-to-month variation of the train
year. For the team log: the parallel Task-4 entropy candidate was also probed on
the LB and lost badly (0.76803 vs 0.79775, dS_od ~ -0.33), so the NNLS baseline
stays; that does not affect Task 2.

Per-panel (v3): D7_I210_E .947, D7_I10_E .871, D7_I405_N .827, D7_I405_S .796,
D7_I210_W .788, D12_I5_N .758, D7_I10_W .757, D12_I5_S .712.
Per family: D7_I210 .868, D7_I10 .814, D7_I405 .811, D12_I5 .735.

Model selection on the January block (mean IoU over January windows):
base (63 leaves, lr .06) onset .761 / ongoing .894; big (127 leaves, lr .05)
onset .766 / ongoing .898; small (31 leaves, lr .08) onset .739 / ongoing .889.
"big" was selected.

### Secondary released-train-window proxy (v3)

overall 0.879 (onset 0.899, ongoing 0.858); per family D7_I405 .906, D7_I210
.872, D7_I10 .870, D12_I5 .866. Persistence on the same windows ≈ 0.30.

## Submission files

* `out/queue_gbmv3_validation.csv`, `out/queue_gbmv3_private.csv` — **final
  candidate (LightGBM v3)**: exactly 87,000 template rows per split (80 windows,
  no missing keys, binary `queue_pred`; onset positives only at `T+30`).
* `out/queue_gbm_validation.csv`, `out/queue_gbm_private.csv` — v1/v2 model
  predictions used in the first full submission (kept untouched).
* `out/queue_static_validation.csv`, `out/queue_static_private.csv` — static v1
  fallback.

### Frozen final model / exact reproduction

`models/v3/` holds the frozen artefacts of the final model
(`gbm_onset.txt`, `gbm_ongoing.txt`, `gbm_meta.json` with the config, rounds,
decision rules, the exact 65-column feature list, and the February CV numbers;
`gbm_cv.csv` is the February per-window detail). The submitted CSVs are
reproduced *byte for byte* by

```
python predict.py --model gbmv3 --split both --output out/queue_gbmv3.csv
```

(verified with md5: both splits identical to the submitted files). The
`--model gbmv3` path loads `models/v3/*` and selects the model's 65 feature
columns out of the current feature builder, so it keeps working even though the
builder now also carries the rejected ramp features.

Exact model configuration recorded in `models/v3/gbm_meta.json`:
`objective=binary, num_leaves=127, learning_rate=0.05, min_data_in_leaf=40,
feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=42,
deterministic=true, force_row_wise=true`, 450 trees (onset), 700 trees
(ongoing); decision rules onset `p>=0.30` (last step only), ongoing `p>=0.25`.

## Files / how to reproduce

```
work/task2/
  t2core.py            data loading, queue labels, candidate table, greedy
                       selector, official IoU aggregation
  features.py          the 65-feature window matrix (history only)
  static_model.py      rule-based predictor (hotspot links + persistence)
  build_dataset.py     pseudo-window dataset -> cache/ds_<PANEL>.npz
  run_static_cv.py     static rule sweep + honest pseudo-CV
  fit_static.py        static parameters -> models/static_<PANEL>.npz
  train_gbm.py         dataset loading, GBM training helpers
  tune_final.py        config + rule selection on January, February scoring,
                       writes out/gbm_meta.json, out/gbm_cv.csv
  refit.py             refit the selected config on all train windows ->
                       models/gbm_onset.txt, models/gbm_ongoing.txt
  predict.py           build a submission CSV for a split (--model gbm|static)
  eval_baselines.py    persistence / simple baselines pseudo-CV
  eval_released.py     score on the released train windows
  analyze_onset.py     onset-set structure diagnostics
  analyze_gbm.py       decision-rule diagnostics
  NOTES.md             this file

# rebuild the *pipeline* end to end (about 15 minutes; the builder currently
# emits 71 features including the rejected ramp block, so this reproduces the
# model-selection pipeline, not necessarily the frozen v3 weights)
python build_dataset.py            # cache/ds_<PANEL>.npz
python run_static_cv.py            # static rules + out/static_cv_info.json
python fit_static.py               # models/static_<PANEL>.npz
python tune_final.py               # config/rule selection -> out/gbm_meta.json,
                                   # out/gbm_cv.csv (February CV detail)
python refit.py                    # models/gbm_{onset,ongoing}.txt
python predict.py --model gbm --split both --output out/queue_candidate.csv

# reproduce the SUBMITTED v3 CSVs exactly (frozen artefacts, no retraining):
python predict.py --model gbmv3 --split both --output out/queue_gbmv3.csv
```

Deterministic: no sampling in the dataset builder, fixed LightGBM seed and
`deterministic=true`. The submitted CSVs are reproduced byte for byte from the
frozen `models/v3` artefacts with `predict.py --model gbmv3` (md5-verified on
both splits); the full pipeline above reproduces the model-selection procedure,
whose current feature builder also carries the rejected ramp block.
