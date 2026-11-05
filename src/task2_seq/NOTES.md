# Task 2 — long-history causal queue forecaster (second attempt, "v5")

Owner: **task2-seq** (second, independent attempt at Task 2). Workspace: `work/task2_seq/`.
Shipped first attempt: `work/task2/` (LightGBM v3, honest Feb pseudo-CV 0.8071,
onset .7572 / ongoing .8736, LB-implied validation IoU ~0.76). This folder only
adds to it; nothing in `work/task2/` was modified.

## TL;DR

The rules allow **any** data with timestamp `<= T`, so this attempt replaces the
60-minute `window_history` input with a **3-hour causal history ending at the
origin T itself**, taken from the split's released `mainline_states_masked`
layer (NaN-aware aggregation, no imputation model). 80 new features are stacked
on top of the 65 frozen v3 features and a **v3+v5 probability blend** is
validated:

| protocol | v3 (same code/rows) | v5 (long history) | v3+v5 blend |
|---|---:|---:|---:|
| B — February pseudo-CV, official nesting | 0.8085 | 0.8131 | **0.8195** (+0.0110) |
| A — 80 released train windows (June 2030), June excluded from fit | 0.8158 | 0.8200 | **0.8292** (+0.0134) |

Both protocols improve by more than the +0.005 gate, so a candidate submission
(`out/queue_v5_validation.csv`, `out/queue_v5_private.csv`) is emitted. The
blend parameters (`w`, thresholds) are chosen on the January block only;
February and the June windows were never used for any selection.  At the
smaller stride-2 first pass the single v5 model failed protocol B — that
negative result and the lean-block variant are kept below because they are the
interesting feature finding: *the long history only pays off with the full fit
block and when blended, not as a standalone replacement.*

**Live confirmation (Lead, 2026-11-02):** the v3+v5 blend was probed on the
public LB: **0.82902 vs 0.82448** for the shipped v3 queue layer = **+0.00454
total score = +0.015 S_queue IoU**, and it became the queue layer of the best
submission.  Subsequent single-variable probes (thresholds, morphological
closing) all scored below that operating point, so the shipped thresholds
(onset 0.30 / ongoing 0.25) are kept.  Day-end follow-ups (smarter onset
set-size rules, ramp jam-sensor features) were both negative — see §9.

## 1. The new signal

* Task 2 is online: only data with timestamp `<= T` may be used, `T` included.
  v3 used only the released 12-step `window_history` (`T-60 … T-5`), i.e. it
  discarded the allowed row at `T` and everything older than one hour.
* The split's **whole `mainline_states_masked` layer is released data**, so for
  every window the 3 hours before `T` and the row at `T` are legal inputs.
  The `window_history` is exactly the same observation layer (verified: the 12
  history steps are unmasked in the masked layer and equal `window_history` to
  `4e-6`; for the train split they equal the plain layer).
* The masked layer blanks the horizon cells (`T+5 … T+30`), so nothing after
  `T` is even available; the row at `T` is only partially observed (35-65 %
  NaN depending on the day's mask regime) and is handled NaN-aware.

## 2. Causal construction (no timestamp > T ever read)

Per window, a `(37, L)` history matrix ending at the origin:

| rows | offsets | source |
|---|---|---|
| 0..23 | `-180 … -65 min` | split masked layer, NaN where masked |
| 24..35 | `-60 … -5 min` | **observed layer**: `window_history` for released windows, train plain layer for arbitrary train origins (same observed layer, identical feature function for fitting and scoring) |
| 36 | `0` (T) | split masked layer (partially masked) |

* **No imputation model is used.** All long-history statistics are NaN-aware
  (nanmedian/nanmean/nansum; queue runs skip unknown rows). Missingness
  patterns are exactly the released mask, identical in distribution between the
  train masked layer and the validation/private masked layers (per-day NaN
  fraction quantiles .30/.32/.39/.55/.55 vs .32/.32/.40/.55/.58; regime counts
  ~equal).
* **The Lead's Task-1 reconstruction (`dense_gbm`) was deliberately not used.**
  It is an *offline* smoother (temporal neighbours, spatial smoothing, iterated
  fills) — its value for a cell at `t <= T` can depend on observed cells after
  `T` (e.g. `T+35`, which is released outside the blanked horizon), which would
  silently violate the `timestamp <= T` rule. Only raw released observations
  with `timestamp <= T` enter the features. (`dense_smooth/` same reasoning.)
* Labels: 3-step centred rolling median of the train plain speed thresholded at
  `v_cut = 0.60 * free_speed`, masked to `is_score_eligible` — the v3 convention.
  Verified identical to the frozen v3 dataset's truth arrays for every window.
* No hidden truth is accessed; the June/February evaluation labels exist only
  in the released train plain layer and were not used for fitting (June) or
  selection (February).
* Hyper-parameters, thresholds and the blend weight are selected on the
  **January block** of the pseudo-window protocol; February and June are scored
  once.

## 3. Features (80 new, see `lhcore.LH_FEATURES`)

All per link, in travel (rank/milepost) order from `lwr_mainline_topology.csv`:

* **origin-row state (new information v3 did not have)**: speed/v_cut, queue
  flag, flow/capacity at `T`; same-timestamp rank-neighbour fill of the masked
  cells at `T` (+ an observation flag); the fill falls back to the 3-hour level
  when a masked run is longer than 5 links.
* **3 h temporal levels and trends**: speed/v_cut nan-medians over the last
  65 min / `-120..-65` / `-180..-125`, min over the last 2 h, net level change,
  recent-vs-old trend, `r_T - r_med_last`; flow/capacity levels and slope;
  density/critical-density (congestion pressure).
* **queue history over 3 h**: queued fraction over the whole 3 h / old / middle
  / last hour, current queue run length (queue age, unknown rows skipped),
  steps since the last queued observation, longest run, number of
  on/off transitions.
* **queue-front geometry**: rank distance to the nearest queued link upstream
  and downstream at `T` and at `T-30`, their difference (front motion /
  propagation speed), queued mass within 5 and 10 rank positions upstream and
  downstream.
* **rank-order neighbourhood**: queued fraction (last hour and 3 h) and speed
  level shifted by ±1, ±2, ±3, ±5 links; shifted origin-row speed ±1, ±2.
* **panel-level context**: number of queued links at `T`, mean queued fraction
  in the last hour, its change vs the previous hour, mean flow/capacity, mean
  queue age (broadcast to every link).
* **clock**: hour and day-of-week sin/cos of the origin.
* **causal ramp demand over 3 h**: attached on-ramp flow/capacity in the last
  hour and the oldest hour, slope, off-ramp level, ramp validity rate.

The 65 v3 features are reused **row-for-row from the frozen v3 dataset cache**
(`work/task2/cache/ds_<PANEL>.npz`), so the A/B comparison differs only in the
80 new columns.

## 4. Validation protocol

**Equivalence gate (passed).** The window list of the v3 dataset is
reconstructed deterministically (`lhcore.train_index_arrays`) and verified
exact: same number of windows per panel (731/360/708/961/925/707/714/752
onset, 148/45/229/215/224/150/136/137 ongoing), identical truth arrays,
eligibility arrays, accepted flags and day indices; re-computing the v3
features from scratch gives a max abs difference of **0** against the cache.
End-to-end: applying the **frozen shipped v3 boosters** to the released-window
feature block built here reproduces v3's released-train-window proxy exactly:
overall **0.8786**, onset **0.8989**, ongoing **0.8582** (v3 reported 0.879).

* **Protocol B (primary, pseudo-CV).** Fit on all selector-valid pseudo windows
  with origin before `2031-01-01` (the v3 dataset: greedy-accepted windows plus
  a chronological 1-in-4 subsample of valid candidates before `2031-02-01`);
  pick the threshold (and blend weight) on the January block; score the
  untouched February block with the official nesting
  window → (panel, condition) → panel → family → 4 families.
* **Protocol A (released-window proxy).** Same fit block with every June 2030
  origin removed; score the 80 released **train** windows (all in June 2030)
  whose horizon truth is fully present in the train plain layer; parameters are
  the January-selected ones (no June information in fitting or selection).
* Rows at stride 1 (no window subsampling, the decisive run): onset 915,893
  fitting rows, ongoing 926,496 — the same scale as the shipped v3 fit
  (940k/951k).  Stride 2 halves the windows and is used as a stability check.

## 5. Results

### 5.1 Stride 1 (decisive; same rows/config/rule grid for v3 and v5)

Protocol B (February pseudo-CV, official nesting):

| model | overall | onset | ongoing | Jan tune |
|---|---:|---:|---:|---:|
| v3 (65 feats) | 0.8085 | 0.7627 | 0.8719 | th 0.25 → 0.7770 |
| v5 (65+80 feats) | 0.8131 | 0.7653 | 0.8784 | th 0.25 → 0.7790 |
| **blend v3+v5** | **0.8195** | **0.7714** | **0.8812** | w 0.5/th 0.3 → 0.7860 |

Protocol A (released June windows, June excluded from fit, January parameters):

| model | overall | onset | ongoing |
|---|---:|---:|---:|
| v3 | 0.8158 | 0.7773 | 0.8543 |
| v5 | 0.8200 | 0.8060 | 0.8339 |
| **blend v3+v5** | **0.8292** | **0.8050** | **0.8534** |

Per family (stride 1):

| protocol | model | D12_I5 | D7_I10 | D7_I210 | D7_I405 |
|---|---|---:|---:|---:|---:|
| B | v3 | .732 | .814 | .868 | .819 |
| B | blend | .746 | .834 | .874 | .823 |
| A | v3 | .775 | .842 | .834 | .812 |
| A | blend | .795 | .827 | .877 | .817 |

Per panel, protocol B blend: D12_I5_N .766, D12_I5_S .727, D7_I10_E .873,
D7_I10_W .795, D7_I210_E .952, D7_I210_W .797, D7_I405_N .839, D7_I405_S .808.

The blend gain is broad (7/8 panels at stride 1 protocol B; 6/8 protocol A).
Onset gains come from the origin-row/3-hour state (row `T` itself is new); the
ongoing gains come from the blend stabilising the queue-front features.

### 5.2 Stride 2 (first pass, the negative result that shaped the design)

Same protocol, half the fitting windows, single models only (this run is what
made me test blending and lean blocks):

| protocol | v3 | v5 | delta |
|---|---:|---:|---:|
| B — overall | 0.7960 | 0.7892 | −0.0068 |
| B — onset / ongoing | .7537 / .8508 | .7409 / .8564 | |
| A — overall | 0.8022 | 0.8119 | +0.0097 |
| A — onset / ongoing | .7529 / .8515 | .7753 / .8484 | |

So a standalone v5 was inconsistent across protocols (Feb onset overfit despite
a small January tune gain), which is why the final candidate is the blend and
why the full fit block was used. A lean long-history block (28 features: origin
row + front geometry only, 93 features total) scored Feb onset 0.7491 vs v3
0.7589 at stride 1 — the lean block is *worse* than no long history at all,
the 80-feature block is better (0.7642). The information is in the combination
(row-T state + 3 h levels + front geometry), not in a single hand-picked subset.

### 5.3 What the long history actually adds

* The origin row `T` is genuinely new information: v3's newest input is `T-5`.
  It shifts onset predictions toward links already below `v_cut` at `T`
  (near-queue) instead of relying purely on the 3-hour recurrent hotspot
  climatology v3 already had, and it improves onset in both protocols
  (B .7627→.7714 blended, A .7773→.8050).
* For ongoing windows the queue-front geometry (distance to the nearest queued
  link at `T` and `T-30`, and its motion) adds a small but consistent gain,
  mostly via the blend (B .8719→.8812).
* The standalone 145-feature model is noisier on onset: with only ~218 January
  onset windows for rule selection, the extra 80 columns let it overfit, hence
  blend instead of replace. The blend keeps v3's stronger onset behaviour and
  uses v5 only for the residual ordering.

## 6. Deliverables

* `out/queue_v5_validation.csv`, `out/queue_v5_private.csv` — candidate Task 2
  submission: exactly 87,000 template rows per split (80 windows x 6 steps x
  1450 links across the 8 scored panels), binary `queue_pred`, unique
  `(window_id, timestamp, link_id)` keys, deterministic from the saved
  artefacts.  Prediction = `w * p_v3 + (1-w) * p_v5` with per-condition
  `w = 0.5` (onset) / `0.6` (ongoing), onset threshold 0.30 on the last horizon
  step only (release fact: onset truth is only at `T+30`), ongoing threshold
  0.25 on all six steps.  All chosen on the January block.
* Frozen artefacts: `models/v5_onset.txt`, `models/v5_ongoing.txt` (v5 LightGBM
  trained **jointly** on every released train window, 1,095,438 onset rows /
  1,535,022 ongoing rows, the exact CV configuration of 450/700 trees),
  `models/v5_blend_meta.json` (blend weights, thresholds, feature list, CV
  source), plus the frozen shipped v3 boosters in `work/task2/models/v3/`.
  (The per-panel incremental refit helper `fit_final.py` is kept for the record
  but under-produced trees on some panels — a LightGBM continued-training
  quirk — so the deployment models are the joint `fit_joint.py` ones.)
* Validation of the two CSVs: exactly 87,000 rows per split, 80 windows, the 6
  forecast timestamps per window, all template keys covered, unique
  `(window_id, timestamp, link_id)`, binary only; onset positives appear only at
  the `T+30` step (release fact); rerunning `ship_blend.py --no-refit` reproduces
  the files byte-for-byte (md5 validation 49fa501f…, private 761a6150…). The
  candidate changes only 114 (validation) / 135 (private) of 87,000 cells
  relative to the shipped v3 file (0.13-0.16 %) — it is a light, near-threshold
  correction, which is what a +0.01 IoU blend looks like on 80 windows.
* Rollback: keep the shipped v3 files `work/task2/out/queue_gbmv3_*.csv`.
  The honest gain of the blend over the comparable v3 baseline is +0.011
  (protocol B) and +0.013 (protocol A) overall; the realistic LB-implied gain
  is likely smaller (the validation month is harder than the pseudo-CV, see
  `work/task2/NOTES.md`), so this is a probe candidate, not a guaranteed gain.

## 7. Reproduction

```powershell
cd C:\Users\lenovo\Desktop\kaggle\work\task2_seq
python build_train.py          # pseudo-window long-history caches: cache/tr_<PANEL>.npz (~3 min)
python build_released.py       # released-window features (all splits): cache/rel_<PANEL>.npz
python run_final_cv.py         # stride-1 A/B CV, v3 vs v5 vs blend -> out/final_cv_stride1.json
python combine_cv.py           # official-nesting combination of the per-condition details
python fit_joint.py            # joint refit of the v5 deployment models (450/700 trees)
python ship_blend.py --no-refit  # blend with the frozen v3 boosters -> the two CSVs
# (ship_blend.py with refit uses the incremental fit_final.py path, kept for the record)
# smoke tests / equivalences
python verify_core.py D7_I10_E # window-list + truth + v3 feature alignment (max diff 0)
python check_v3_rel.py         # frozen v3 boosters on this pipeline -> 0.8786 released proxy
```

The stride-2 first pass is reproduced by
`python run_cv.py --feats v3 v5 --protocols B A --stride 2`.
Everything is deterministic: no sampling in the feature builders, fixed
LightGBM seed 42 with `deterministic=true`, feature caches on disk; re-running
`ship_blend.py` reproduces the CSVs byte-for-byte.

## 8. Files

```
work/task2_seq/
  lhcore.py            causal history assembly, 80 long-history features,
                       window-list reconstruction, v3-cache alignment, IoU
  build_train.py       pseudo-window caches      -> cache/tr_<PANEL>.npz
  build_released.py    released-window features  -> cache/rel_<PANEL>.npz
  verify_core.py       window-list / feature / truth equivalence checks
  check_v3_rel.py      frozen-v3 end-to-end proxy check (0.8786)
  run_cv.py            stride-N protocol A/B runner (v3 vs v5)
  run_exp.py           lean-block + blend experiments (used for the negative result)
  run_final_cv.py      decisive stride-1 A/B run with the v3+v5 blend
  combine_cv.py        official-nesting aggregation of the per-condition details
  fit_final.py         per-panel incremental refit (record only; see fit_joint.py)
  fit_joint.py         joint refit of the deployment v5 models (450/700 trees)
  ship_blend.py        blend with the frozen v3 boosters, write out/queue_v5_*.csv
  validate_csv.py      template coverage / binary / v3-diff diagnostics
  check_csv.py         onset-last-step structure + md5 determinism checks
  rule_exp.py          onset set-size rule experiments (§9.1)
  ramp_tod.py          train-split ramp time-of-day profiles (§9.2)
  ramp_cv.py           ramp jam-sensor ongoing A/B (§9.2)
  ship_variants.py     threshold / fallback variants of the final blend (§9.3)
  ship_variants_hi.py  extra onset thresholds 0.45/0.50/0.55 (§9.3)
  onset_size.py        predicted onset set-size distribution per threshold (§9.3)
  fit_ramp.py, ship_ramp.py   ramp deployment path (not shipped, §9.2)
  v6core.py            v6 feature block: recurrence / incidents / imbalance /
                       shockwaves (§10)
  build_v6.py          v6 feature caches -> cache/v6_<PANEL>.npz
  run_v6_cv.py         v6 single-model A/B (§10 stage 1)
  run_v6_stage2.py     v3+v6 blend vs the shipped v3+v5 blend (§10 stage 2)
  out/v6_results.md    one-page v6 result table (negative, not shipped)
  out/v6_stage{1,2}.json, out/v6_{...}.csv   v6 per-window details
  build_v7.py          sigma-2.5 hard denoised labels/truth -> cache/v7_<PANEL>.npz
  run_v7_cv.py         v7 two-protocol CV vs denoised truth (§11)
  build_v9.py          sigma_v estimate, soft Bayesian labels, sigma-1 mild
                       truth -> cache/v9_<PANEL>.npz (§12)
  run_v9_cv.py         soft-label CV on two proxies (§12)
  ship_v7.py, ship_candidates.py   deployment CSVs (v7 hard / v9 soft sources)
  validate_v7.py       template + footprint check of the v7 candidate
  out/queue_v7_{validation,private}.csv   hard-sigma2.5 LB probe candidate
  out/v7_cv.json, out/v9_cv.json, out/v9_sigma.json, out/build_v{7,9}.log
  out/final_cv_stride1.json         all stride-1 numbers (A/B, v3/v5/blend)
  out/final_{on,ong}_{v3,v5,blend}_{feb,june}.csv   per-window IoU details
  out/cv_*_s2.{csv,json}            stride-2 first-pass details
  out/queue_v5_validation.csv       candidate submission (validation)
  out/queue_v5_private.csv          candidate submission (private)
```

## 9. Day-end follow-ups (2026-11-02): rules, thresholds, ramp features

All three were requested by the Lead as single-variable probes around the
shipped operating point; everything is measured with the honest protocol A/B
apparatus above and nothing was tuned on February/June.

### 9.1 Smarter onset set-size rules — NEGATIVE (keep the 0.30 threshold)

Probs = the January-selected blend (w = 0.5 onset).  Rule parameters chosen on
January only, then scored once on February and once on the released June
windows (`rule_exp.py`, `out/rule_exp.json`):

| onset rule (best on Jan) | Jan | FEB onset | JUNE onset |
|---|---:|---:|---:|
| plain threshold p >= 0.30 (shipped) | 0.7860 | **0.7714** | **0.8050** |
| threshold 0.30 ∪ top-1 fallback | 0.7767 | 0.7680 | 0.8050 |
| threshold 0.30 ∪ top-2/3 | 0.7810 | 0.7653 | 0.7581 |
| adaptive k = round(a + b·#{p>=0.3}) | 0.7767 | 0.7680 | 0.8050 |
| k = round(alpha·sum(p)) (best alpha 1.25) | 0.7283 | 0.7200 | 0.7579 |
| low threshold with a cap on k (best .30/6) | 0.7663 | 0.7512 | 0.8035 |

Every explicit set-size rule is below the plain threshold on January *and* on
both held-out protocols: the calibrated probability threshold is already the
best size adaptor, which matches the LB probes (onset top-3 cost -0.024, ongoing
0.40 threshold cost -0.0025).

### 9.2 Ramp jam-sensor features for ongoing — MIXED / REJECTED

New feature block (`LH.RAMP_FEATURES`, 8 features, `ramp_tod.py`): the attached
on/off-ramp flow divided by its own train time-of-day profile (the public
write-up's jam sensor: on-ramps drop to ~0.2x when the adjacent mainline link
jams), last-hour ratio, older-hour ratio, min, trend, off-ramp ratio, upstream
and downstream neighbour ratios, and a "< 0.35" drop flag.  Causal (ramp rows
<= T).  Rebuilt caches (`build_train.py`/`build_released.py` now also store
`on_lhr`/`ong_lhr`/`lhr`), retrained the ongoing halves (`ramp_cv.py`):

| ongoing-only model (same rows/config) | FEB IoU | JUNE IoU |
|---|---:|---:|
| v3 (65 feats) | 0.8580 | **0.8543** |
| v5 (65+80) | **0.8621** | 0.8339 |
| v5r (65+80+8 ramp) | 0.8597 | 0.8452 |
| blend v3+v5 (w 0.6, th 0.25) | **0.8679** | — |
| blend v3+v5r (w 0.6, th 0.25) | 0.8669 | — |

Ramp improves over v5 on June (+0.011) but degrades February ongoing (-0.002)
and the blended total (-0.001); February is the protocol that transferred to the
LB, so the ramp block is **not shipped** (files kept for the record: v5r model
artefacts were not produced/kept in `models/`).  The ramp profile + cache keys
stay in the pipeline so the experiment is reproducible.

### 9.3 Threshold variants delivered (frozen artefacts, no retraining)

Generated by `ship_variants.py` / `ship_variants_hi.py` from the *same* frozen
`models/v5_{onset,ongoing}.txt` + frozen v3 boosters; each pair is a complete
87,000-row submission with only the listed variable changed.  Filenames in
`out/` (both `_validation.csv` and `_private.csv`):

| file tag | change vs shipped blend |
|---|---|
| `queue_v5_*` | **shipped** blend: onset th 0.30 (last step), ongoing th 0.25 |
| `queue_v5blend_ong015_*` | ongoing th 0.15 |
| `queue_v5blend_ong035_*` | ongoing th 0.35 |
| `queue_v5blend_on020_*` | onset th 0.20 |
| `queue_v5blend_on040_*` | onset th 0.40 |
| `queue_v5blend_on045_*` | onset th 0.45 |
| `queue_v5blend_on050_*` | onset th 0.50 |
| `queue_v5blend_on055_*` | onset th 0.55 |
| `queue_v5blend_onfb1_*` | onset th 0.30 ∪ top-1 fallback |
| `queue_v5blend_ong015_onfb1_*` | ongoing 0.15 + onset fallback |

Long aliases also exist (`queue_v5blend_onset_0.40_*`, `..._ongoing_0.15_*`).

Predicted onset set size (links/window at T+30; official/released-train truth
mean 2.75 links, public report 3.52):

| onset th | validation mean (range) | private mean (range) | zero-pred windows (val/priv) |
|---|---:|---:|---:|
| 0.30 (shipped) | 3.80 (0-11) | 3.95 (0-7) | 6 / 3 |
| 0.40 | 3.38 (0-10) | 3.67 (0-7) | 8 / 3 |
| 0.45 | 3.17 (0-10) | 3.50 (0-7) | 9 / 5 |
| 0.50 | 2.98 (0-9) | 3.30 (0-7) | 11 / 5 |
| 0.55 | 2.85 (0-9) | 3.08 (0-7) | 12 / 6 |

Caveat for the probe: raising the onset threshold toward the truth mean also
increases the number of windows with an empty prediction (IoU 0 when the truth
is non-empty), so 0.45-0.55 is a precision/recall trade-off, not free.

### 9.4 CatBoost onset blend — NOT TESTED

The public 0.865 recipe uses LGB+CatBoost for onset, but `pip install catboost`
stalled (network/wheel) and was killed at the Lead's request; no CatBoost
artefact exists and nothing was trained.  The LGB-only blend remains final.

### 9.5 Final artefacts / exact reproduction (verified)

```
python fit_joint.py            # v5 models joint-refit on all train windows (450/700 trees)
python ship_blend.py --no-refit  # v3(blend half) freeze from work/task2/models/v3
# -> out/queue_v5_validation.csv   md5 49fa501fd4b3f65d5fbaca7c416ea1f8
# -> out/queue_v5_private.csv      md5 761a6150d9fa875fd3c982ffc727a446
```

The md5s were re-verified after the ramp cache extension (caches changed shape,
the 145-column model input unchanged), i.e. the shipped CSVs are byte-for-byte
reproducible from `models/` + `cache/`.  The shipped candidate is the blend
`0.5*p_v3 + 0.5*p_v5` on onset (last step, th 0.30) and
`0.6*p_v3 + 0.4*p_v5` on ongoing (all six steps, th 0.25), where the v3 half is
the frozen shipped v3 boosters and the v5 half is `models/v5_{onset,ongoing}.txt`.
Honest CV: **protocol B 0.8195 (onset .7714 / ongoing .8812), protocol A 0.8292
(onset .8050 / ongoing .8534)**; public LB 0.82902 (+0.015 S_queue vs v3).

## 10. v6 feature block — NEGATIVE, not shipped (2026-11-03)

Requested bounded experiment: long-horizon recurrence (24 h/72 h same-TOD queue
flags, recurring speed level, hours since last queue), panel incident context
(>30 km/h/15 min drop events, 6 h/24 h counts, recency, fraction below the 24 h
profile), upstream/downstream demand-capacity imbalance and LWR shockwave speed /
front motion — 26 strictly causal features (`v6core.py`, `build_v6.py`).  Same
stride-1 harness, identical rows/config/rule grid for v5 vs v6.

| model | protocol | overall | onset | ongoing |
|---|---|---:|---:|---:|
| v5 (145 feats) | B | 0.8131 | 0.7653 | 0.8784 |
| v6 (171 feats) | B | 0.8020 | 0.7402 | 0.8785 |
| v5 | A | 0.8200 | 0.8060 | 0.8339 |
| v6 | A | 0.8073 | 0.7802 | 0.8345 |
| **v3+v5 blend (SHIPPED)** | B | **0.8195** | **0.7714** | **0.8812** |
| v3+v6 blend | B | 0.8115 | 0.7622 | 0.8794 |
| **v3+v5 blend (SHIPPED)** | A | **0.8292** | **0.8050** | **0.8534** |
| v3+v6 blend | A | 0.8260 | 0.7968 | 0.8551 |

The v6 blend is worse on both protocols (B -0.0080 overall / -0.0092 onset;
A -0.0032 overall / -0.0082 onset) — the ship bar (>= +0.005 both protocols, no
condition regression > 0.005) is failed, so **nothing was shipped and the v5
files are untouched**.  Feature importance shows why: the new block carries only
6.0 % (onset) / 4.4 % (ongoing) of the model gain and its best feature ranks
14-15 overall; the extra feature mass dilutes the onset model (same pattern as
the earlier stride-2/lean experiments — onset only benefits from blending, not
from more features).  One-page detail: `out/v6_results.md`; reproduce with
`python build_v6.py && python run_v6_cv.py --stage 1 && python run_v6_stage2.py`.

## 11. Spec finding: official truth is the latent state; denoised-label attempt (v7)

`SCORING_SPEC.md` says the ground truth is taken from the *underlying state*
rather than the noisy observation, "so measurement error cannot flip the
indicator back and forth at the threshold", and that every scored window has a
queue somewhere in the horizon.  Our training labels and all CV truth so far use
the 3-step-median observed speed, i.e. they do contain measurement flips near
`v_cut`.  Attempt v7 replaces them with hard labels from a stronger denoiser:
NaN-aware temporal Gaussian (sigma = 2.5 steps = 12.5 min, reflect boundaries,
mask-normalised) thresholded at `v_cut` and eligibility-masked (`build_v7.py`).

* **Flip counts** (noisy median-3 label vs sigma-2.5 label): pseudo onset 0.20 %
  of all cells / **24.7 % of cells within ±5 km/h of the cut**; ongoing 0.19 % /
  22.9 %; released-window truth 0.12 % / 29.6 %.  Positives: onset pseudo
  72,688 -> 61,308 (-15.6 %), ongoing 117,869 -> 115,296 (-2.2 %), released
  6,426 -> 6,119 (-4.8 %).  On the 40 released onset windows the T+30 truth-set
  size drops **2.75 -> 1.43 links/window and 13/40 windows become empty**, which
  contradicts the spec guarantee (the official public report says the onset truth
  is 1-7 links, **mean 3.52**).  So sigma 2.5 over-smooths — it removes short
  queue episodes, it is not the measurement-error model.
* **Honest two-protocol CV** (same stride-1 rows/config; fixed LB-verified rules:
  onset w 0.3 / th 0.30 / top-1 fallback, ongoing w 0.4 / th 0.20 + window
  fallback), scored against the sigma-2.5 truth:

| models | protocol | sigma-2.5 truth overall | onset | ongoing |
|---|---|---:|---:|---:|
| noisy labels (current) | B | 0.6236 | 0.3962 | 0.8581 |
| v7 denoised labels | B | 0.6655 | 0.4817 | 0.8700 |
| noisy labels | A | 0.6077 | 0.3850 | 0.8305 |
| v7 denoised labels | A | 0.6446 | 0.4992 | 0.7900 |

  Retraining on denoised labels does improve against denoised truth (+0.042 B /
  +0.037 A overall), but protocol A ongoing regresses by 0.041 (> 0.005), so the
  v7 model was **not** the default ship.  Per the Lead's request the v7 candidate
  is nevertheless produced as an LB probe (`queue_v7_{validation,private}.csv`,
  models `models/v7hard_{onset,ongoing}_{v3,v5}.txt`, same fixed rules): it
  predicts a **smaller** onset footprint than the current best file —
  links/window 2.83 vs 3.90 (validation) and 2.77 vs 4.00 (private); 231/87,000
  (validation) and 260/87,000 (private) cells differ from
  `queue_w30ong40fb1_*.csv` (171/217 best-only, 60/43 v7-only).  Validation: 5,489
  positives (6.309 %) vs 5,600 (6.437 %).
* **Decision-rule footnote (fallback).**  The earlier pseudo-CV verdict against
  the top-1 fallback was an artefact of empty-truth pseudo-windows: the selector
  gate "queue in horizon" was built on noisy observations, so some pseudo-windows
  have no queue under any cleaned truth, and there an extra predicted cell scores
  0 instead of 1.  Official scored windows are guaranteed to contain a queue, so
  the fallback is safe there; the LB confirmed it (+0.0075 total).  The fallback
  rule is kept in the fixed rule set used here.

## 12. Observation-noise estimate and soft Bayesian labels (v9) — negative

**sigma_v estimate** (robust, from second differences of the plain speed at
consecutive 5-min steps: `var(v_{t+1}-2v_t+v_{t-1}) = 6 sigma^2`): per panel
0.68-1.62 km/h, **pooled ~1.2 km/h** (MAD-based; trimmed-std cross-check
agrees).  This is small: only 0.05-0.14 % of horizon cells flip between the
median-3 truth and a sigma-1 Gaussian truth, so measurement noise is *not* the
main gap between our labels and the official latent state.

**Soft labels** `y = Phi((v_cut - v_obs)/sigma_v) * eligible` (`build_v9.py`,
same rows/config; LightGBM binary accepts continuous labels), evaluated on two
hard proxies: (i) noisy median-3 truth (current regime) and (ii) mild sigma-1
truth:

| models | protocol | proxy (i) noisy: overall (on / ong) | proxy (ii) sigma-1: overall (on / ong) |
|---|---|---:|---:|
| noisy labels (current) | B | 0.8155 (0.7661 / 0.8792) | 0.7873 (0.6984 / 0.8778) |
| soft labels | B | 0.7705 (0.7172 / 0.8582) | 0.7406 (0.6478 / 0.8547) |
| noisy labels | A | 0.8347 (0.8162 / 0.8532) | 0.8073 (0.7674 / 0.8472) |
| soft labels | A | 0.7724 (0.7110 / 0.8339) | 0.7541 (0.6815 / 0.8267) |

Soft labels are **clearly worse on both proxies and both protocols** (-0.045 to
-0.062 overall), so the ship bar (within 0.005 of current on proxy (i), better on
proxy (ii)) fails and **no `queue_v9_*.csv` files were produced**.  Mechanism:
the soft expectation at T+30 is 3.95 onset links/window (hard median-3: 3.80),
but the soft-trained probabilities lose sharpness, so the fixed 0.30 threshold
over-selects; tuning the decision threshold on the sigma-1 proxy only gives
onset 0.50 (IoU 0.662) and ongoing 0.35 (0.873), still far below the current
model — i.e. if a soft/sharpened-label variant is ever probed, it must be used at
onset threshold ~0.50, ongoing ~0.35, not at the shipped 0.30/0.20.

**Interpretation for the report.**  The official onset truth (mean 3.52
links/window) is *larger* than our released-window noisy truth (2.75) and much
larger than the sigma-2.5 denoised truth (1.43); the soft-label expectation
(3.95) is the closest of the three.  So the remaining label-side uncertainty is
in the direction of *more* queued cells (the latent field is sharper/broader
than the smoothed observation), not fewer — aggressive denoising moves the wrong
way.  The shipped v5 blend predicts 3.80-3.90 onset links/window, already close
to the official 3.52.  This is why the v7 (smaller-set) candidate is only an LB
probe and the v5 blend stays the final queue layer.

## 13. Label-regime test (H1 raw vs H2 median-3) — no change needed, nothing shipped

The inverse-analysis hypothesis is **H2 = 3-step temporal median of the observed
speed, thresholded at v_cut** (mild denoise).  Key verification: **the shipped
v3 cache labels and all CV truths already are H2** — recomputing the median-3
labels from the train plain layer reproduces the cached truth arrays bit-exactly,
while the raw single-step (H1) labels do not (verified per cell; H1 vs H2 differ
in only 0.09 % of cells, positives H1 +4.1 %).  So "retrain on median-3 labels"
is a no-op: the current models are the H2-label models, and the requested ship
bar (H2-label model beating the current blend under H2 by +0.01) cannot be met by
construction.

To still give the Lead the label-regime table, the *other* regime was built and
cross-evaluated (`build_h1.py`, `run_v10_cv.py`): models trained on H2 labels
(current) vs models trained on H1 raw labels, both scored under H1 and H2 truth,
identical stride-1 rows/config and fixed LB rules:

| models | protocol | H2 truth: overall (on / ong) | H1 truth: overall (on / ong) |
|---|---|---:|---:|
| H2 labels (current shipped regime) | B | **0.8155** (0.7661 / 0.8792) | 0.8736 (0.8831 / 0.8794) |
| H1 labels (raw) | B | 0.8000 (0.7639 / 0.8499) | 0.8566 (0.8783 / 0.8506) |
| H2 labels (current) | A | **0.8347** (0.8162 / 0.8532) | 0.8682 (0.8831 / 0.8533) |
| H1 labels (raw) | A | 0.8098 (0.7790 / 0.8406) | 0.8623 (0.8784 / 0.8463) |

Conclusions:

* Under the H2 truth hypothesis the current (H2-label) models are best on both
  protocols (B 0.8155 vs 0.8000; A 0.8347 vs 0.8098); training on raw H1 labels
  loses 0.016-0.025 overall, so the shipped label regime is already the right
  one and no models/files were changed.
* Proxy sensitivity is large: the same shipped models score 0.8736 (B) / 0.8682
  (A) under H1 truth but 0.8155 / 0.8347 under H2 truth.  H2 is the
  conservative proxy and is consistent with the LB-implied queue level (the
  earlier v7/v9 experiments pointed the other way only because they over-smoothed
  or softened the labels); the H1 proxy is too optimistic to guide selection.
* **No `queue_v10_*` files were produced** (ship condition not applicable), and
  the current best queue file `out/queue_w30ong40fb1_all_{validation,private}.csv`
  remains the one to probe; the shipped v5 blend / v3+v5 artifacts are untouched.
  Artifacts: `cache/h1_<PANEL>.npz`, `out/v10_cv.json`, `out/build_h1.log`,
  `out/v10_cv.log`; reproduce with `python build_h1.py && python run_v10_cv.py`.




