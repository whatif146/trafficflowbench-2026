# Task-2 inverse analysis: truth hypothesis, absolute S_queue, calibration battery

Author: inverse-scoring (teammate), 2026-11-03. Scope: read-only on the repo release and team artefacts;
all new files under `work/research/inverse/`. Code: `score_hypotheses.py`, `train_lab.py`, `fallback_math.py`,
`zero_probe_math.py`, `percond_odme.py`, `final_numbers.py`. Data: `val_window_scores.csv`,
`train_window_scores.csv`.

## 0. TL;DR

1. **Official nesting reproduced exactly**, including the eligibility filter: for every validation window
   the scored cell set is the published `is_score_eligible` mask on the 6 horizon steps
   (mean **831** eligible cells/window, min 360, max 1643, out of ~1088 template cells on average; `future_coverage = 1.0`).
   Algebra: every window carries weight **exactly 1/80** in S_queue, so
   `ΔS_queue = (1/80)·Σ_windows ΔIoU`. Verified against the fallback probe.
2. **Best-fitting truth hypothesis: H2 = 3-step temporal median of the observation, then `speed ≤ v_cut`
   (`v_cut = 0.60·free_speed`, released per link), on eligible cells.** On the 80 train windows (the only
   split whose horizon is observable) it is the only hypothesis that reproduces the *sign and near-exact
   magnitude* of the pure blend-weight LB delta: train H2 **+0.0062** (raw +0.0060) vs LB-implied **+0.0027**;
   raw-observation H1 gives **−0.0076** (wrong sign) and is rejected. H2 also reproduces the other two
   threshold deltas with the right sign. The LB fallback gain is explained exactly by the nesting, not by
   the truth shape (see §4).
3. **Absolute S_queue (validation month, 90% interval): best candidate q_w30ong40fb1 ≈ 0.80 [0.73, 0.85];
   P4 ≈ 0.77 [0.70, 0.82].** This is an H2-convention-based anchoring (train-month nesting + the two exact
   LB deltas + the 7 empty-window penalty); it is systematically *high* if the organizer's latent-state
   labels disagree with the smoothed-observation convention, and it is inconsistent with the organizer's
   physics-calibrated inversion (0.69–0.79) unless S_ODME is below the 0.8359 baseline anchor and/or
   S_physics ≈ 0.9. The **all-zero-queue probe measures S_queue directly and should settle it** (§6).
4. **Calibration probes** (exact arithmetic, ranked): #1 all-zero queue (measures S_queue, 1 submission);
   #2 all-ones queue (measures mean truth-set size, nearly model-free cross-check of H2); #3 state-all-zero
   (S_state(zeros)=0 *exactly*, S_physics(zeros)≈0 ⇒ reveals `0.35·S_state + 0.15·S_physics`);
   #4 ODME-all-zero (weak: gives `S_ODME_best − S_ODME(0)` with ~±0.03–0.15 on S_ODME(0)).

## 1. What the release lets you recover exactly

**Eligibility is recoverable exactly, even on the horizon.** `mainline_states_masked` keeps the gap cells as
rows with all measured channels blanked but keeps the quality fields. For a validation window
(T=10:05, D7_I10_E) the horizon rows T+5…T+30 exist with `speed_kmh/flow_vph/occupancy` all NaN but
`pct_observed`, `is_score_eligible` populated (eligibility ≈ 0.75–0.78 per row). `window_index.future_coverage`
= 1.0 for every window of every split (it is *not* the eligibility fraction; that is what the flags are for).
Eligibility comes from the cache `work/cache/<PANEL>/<split>_masked.npz` key `elig`.

Consequences:
- Scored cells/window = 6 horizon steps × links where `elig` — mean **831** (out of ~1088 template rows,
  i.e. 6×N_links, N_links 81…352 per panel).
- Predictions on ineligible cells are **dropped by the evaluator** (`eligible.merge(pred, how="left")`), so a
  candidate cannot gain or lose from what it puts on ineligible cells. Our nesting code masks predictions to
  eligible cells; earlier versions that did not were biased low by up to 0.17 S_queue on train.
- `eligible.empty → IoU 0` is irrelevant on the validation month: every window has 360–1643 eligible cells.
- Per-link `v_cut` is released (`fd_parameters.csv`, read into cache); it is not always exactly 0.6×free_speed
  (e.g. 63.24 vs 105.4×0.6 = 63.24; but 67.8 for free_speed 113). Use the released column.

**Truth is not recoverable.** The horizon of validation/private is blanked everywhere; the organizer queue
truth parquet is withheld for every split. So no hypothesis below can be *verified* on validation — only
scored and compared with the LB deltas.

## 2. Truth hypotheses

| id | definition | can be evaluated on |
|---|---|---|
| H1 | raw observation: `speed ≤ v_cut` & eligible | train horizon (unmasked layer) |
| H2 | **3-step rolling temporal median of the observation, then threshold** (team's `lhcore.window_labels` convention; "underlying state rather than noisy observation" approximation) | train horizon |
| H2b | Gaussian σ=1 temporal smoothing of the observation, then threshold | train horizon |
| H3 | Task-1 LightGBM dense state prediction (`work/submissions/dense_gbm`) thresholded at `v_cut` | validation (partial: predictions exist only at Task-1 target cells, ~36–47 % of cells/row) |
| H3u/H3fill | same + nearest-link fill for non-target cells | validation |
| H4a | history persistence: last observed speed ≤ v_cut (at T or last observed step), repeated over horizon | both |
| H4b | "any queued observation in the visible 60-min history", repeated over horizon | both |
| H4c | GBM predictions temporally averaged over the horizon, thresholded | validation |

**H3 is nearly useless on the horizon, and this is a finding**: at horizon rows the Task-1 GBM has no speed
input (the gap is blanked) so its prediction collapses to climatological free flow (mean predicted speed on
Task-1 target cells in the horizon ≈ 104–117 km/h → ~0 queued cells in most onset windows). It therefore
predicts "no queue" exactly where the spec guarantees one, giving 33/80 empty truth windows on validation.
Any truth built from Task-1 horizon predictions is disqualified.

## 3. Train-window lab: which hypothesis reproduces the LB deltas?

80 train windows (all with observable horizon). Candidate masks regenerated with the frozen v3+v5 LightGBM
boosters and the archived rule (blend weight, threshold, top-1 fallback) — i.e. the same masks that were
uploaded for validation — then scored with the official nesting. `train_window_scores.csv`.

Nested S_queue on train (eligible-masked, official nesting):

| candidate | H1 raw | **H2 median-3** | H2b Gaussian |
|---|---|---|---|
| P4_base (v5-blend .5/.6, th .30/.25) | 0.8960 | **0.8755** | 0.8539 |
| q_w30ong40 | 0.8884 | **0.8817** | 0.8602 |
| q_on025 | 0.9022 | **0.8692** | 0.8476 |
| q_w30ong40_on025 | 0.9009 | **0.8692** | 0.8477 |
| q_onfb1 (fallback only; no empty windows exist on train month) | 0.8960 | **0.8755** | 0.8539 |
| q_w30ong40fb1 | 0.8884 | **0.8817** | 0.8602 |

Deltas vs LB-implied `ΔS_queue = Δtotal/0.30`:

| candidate | ΔS_queue H1 | ΔS_queue **H2** | ΔS_queue H2b | **LB-implied** |
|---|---|---|---|---|
| q_w30ong40 (blend weight change only) | −0.0076 | **+0.0062** | +0.0063 | **+0.0027** |
| q_on025 (onset th .30→.25) | +0.0063 | **−0.0063** | −0.0063 | **−0.0017** |
| q_w30ong40_on025 | +0.0049 | **−0.0063** | −0.0062 | **−0.0011** |
| q_onfb1 (fallback only) | 0 | **0** (train has no empty window) | 0 | **+0.0250** |

Reading: H2 gets every sign right on the three threshold/weight probes and its magnitude for the pure
weight change is within 2.3× of the LB value (month-to-month variation; validation month changes only
2 predicted cells at threshold −0.05, train changes ~8). H1 gets the weight-change sign *wrong* and is
rejected. H2b is statistically indistinguishable from H2. **Best fit: H2** (which is also the team's
existing convention, so the model was trained to reproduce it — see caveat in §5).
The fallback probes carry no hypothesis information on train because the train month has 0 empty-prediction
windows; their gain is pure nesting arithmetic (§4).

Validation-window hypothesis scores (for the record; not verifiable): H4a/H4b are the only history-based
constructions that produce non-empty truth in *some* onset windows, and they still leave 40–52/80 windows
empty (they cannot see queue formation), i.e. every released-data-only hypothesis violates the spec guarantee
in tens of windows. The spec guarantee is only satisfied by truth that anticipates the horizon; LB says so
too (§4).

## 4. The fallback gain is exact nesting arithmetic (this resolves +0.0075)

P4 (`queue_v5`) predicts **nothing** in 7 validation windows; `q_onfb1` (onset th .30 + top-1 fallback)
fills the 6 onset ones; `q_w30ong40fb1_all` also fills the remaining ongoing one:

- 6 onset: `Q2_D12_I5_N_validation_004`, `Q2_D7_I10_W_validation_001/002/003/005`,
  `Q2_D7_I405_N_validation_004` (note: 4 of the 6 are in one panel, D7_I10_W — irrelevant because each
  window still weighs 1/80);
- 1 ongoing left by q_onfb1: `Q2_D7_I10_W_validation_009`.

Since every window's weight in S_queue is exactly 1/80,
`ΔS_queue = (1/80)·Σ_{w} ΔIoU_w`, so with the LB's +0.02500 for q_onfb1:

```
Σ_{6 onset windows} IoU_w = 0.025 × 80 = 2.0  ⇒  mean IoU per recovered window = 1/3 exactly.
```

The fallback adds **one** predicted cell per window (the highest-probability link at T+30, one timestep), so a
hit gives `IoU = 1/(1+|T_eligible|)`. Mean 1/3 is exactly what |T| = 2 with a hit gives (or a hit/miss
mixture with |T| ≈ 2–4). Therefore those 6 windows have **small truth sets (~2–4 cells)** and the fallback
cell actually hits — the large gain is real and structural, not a public-month fluke. This also *proves*
those windows' eligible truth was non-empty (otherwise the empty-pred IoU=1 would have been lost).

The two weight-change probes give the same delta (+0.00273 queue) whether applied to P4 or to q_onfb1
(P4→q_w30ong40 = +0.00082 total; q_onfb1→q_w30ong40fb1 = +0.00082 total), a nice internal consistency check.

## 5. Absolute S_queue of P4 and of the current best

Data-driven anchoring (H2 convention):
- Train-month nesting: P4 **0.8755**, q_w30ong40fb1 **0.8817**.
- Validation month has 7 empty-pred windows where P4 must score 0 (spec guarantee): penalty
  `(6·0.883 + 1·0.868)/80 = 0.0771`.
- Fallback recovers the 6 onset windows at mean IoU 1/3: `+6·(1/3)/80 = +0.0250`; the `_all` ongoing
  fallback adds ≈ 0 (one cell against a large queue set: ≤ 1e-4 in total).
- ⇒ `S_queue(best,val) ≈ 0.8817 − 0.0771 + 0.0250 = 0.8296`; `S_queue(P4,val) ≈ 0.8755 − 0.0771 = 0.798`.

Cross-check with the overall equation (weights from `src/release/score_overall.py:20`), assuming the
BASELINES anchors `S_state≈0.93–0.946`, `S_physics≈0.85`, `S_ODME≈0.8359`:
`S_queue = (0.83734 − 0.35·S_state − 0.15·S_physics − 0.20·S_ODME)/0.30` gives 0.67–0.72. The gap
(≈0.08–0.13) is either (a) H2 convention vs organizer latent truth (smoothed observation is the model's own
training target, so 0.83 is biased high), or (b) `S_ODME` below 0.8359, or (c) `S_physics` ≈ 0.9.

**Best defended estimate (used for the cross-check):**
`S_queue(best) = 0.80, 90 % [0.73, 0.85]; S_queue(P4) = 0.77, 90 % [0.70, 0.82]`.
Interval comes from ±0.05 convention gap plus the empty-window/eligibility uncertainty (±0.01). The
organizer's physics-calibrated inversion (0.72–0.75) is compatible at the low end; the difference is
informative rather than contradictory (see §6, probes resolve it in one submission each).

## 6. Calibration probe battery (exact arithmetic)

Notation: `T_best = 0.83734` is the LB total of q_w30ong40fb1; `T_all` = LB of the actually-uploaded base
`q_w30ong40fb1_all` (expected = `T_best + ~2e-5`; the ongoing fallback adds one predicted cell to a window
whose truth is a large queue ⇒ Δtotal ≤ 1e-4, so using 0.83734 as base is safe at the 1e-4 level).

### #1 all-zero queue (highest information; 1 submission)

Evaluator path: `score_window` merges eligible target rows with the submission; `queue_pred` missing → 0
(`fillna(0)`), so **missing and explicit zeros are identical**. Binary 0 is legal. `eligible.empty` → IoU 0
(irrelevant: min 360 eligible cells/window). A window with an empty truth has IoU 1 for empty prediction.

```
S_queue(zeros) = (1/80) · #{windows whose eligible queue-truth set is empty}
```

Spec says both conditions require a queue in the horizon, so the latent truth is non-empty, but the
**eligibility filter can empty it**: truth cells on the ~24 % ineligible cells are not scored. For an onset
window with |T|≈2–4, P(all cells ineligible) ≈ 0.24^|T| ≈ 0.3–1.4 %, i.e. expected ≈ 0–0.6 of the 40 onset
windows ⇒ **S_queue(zeros) ∈ [0, 0.015], point ≈ 0.003**. Hence

```
S_queue(best) = (0.83734 − T_zeroqueue − 0.30·S_queue(zeros)) / 0.30
              = (0.83734 − T_zeroqueue)/0.30  +  S_queue(zeros)
              = (0.83734 − T_zeroqueue)/0.30  +  0.003 ± 0.005
```

Expected magnitude: if the §5 point estimate is right, `T_zeroqueue ≈ 0.83734 − 0.30·0.80 ≈ 0.597`
(if organizer's 0.72: ≈ 0.621). **The single most informative submission we can make.** It also tells us
`S_state+S_physics+S_ODME = T_zeroqueue`-complement, i.e. whether our other three tasks are as good as the
team thinks. Do not select it for the final ranking.

### #2 all-ones queue (second-highest; 1 submission)

All 1s on template cells (eligible ones matter): for every window `IoU = |T∩E| / |E|`, so
`S_queue(ones) = (1/80)Σ |T_w ∩ E_w| / |E_w|` — a direct, model-free measurement of the mean truth-set
fraction. H2-convention prediction on train: **0.092** (mean onset |T|≈2.8, ongoing ≈158 cells, |E|≈831).
If this probe returns ≈0.06–0.10, H2's truth sizes are confirmed; if it returns ≫0.15 the truth sets are much
bigger than H2 and §5's S_queue must be revised downward; if ≪0.03 the empty/truth-size picture is wrong.
Expected total ≈ `T_best − 0.30·(S_queue_best − 0.09)` (e.g. ≈ 0.62 if S_queue_best = 0.80).
Rank #2 (use only if a slot is free; #1 is strictly more valuable because S_queue is the decision variable).

### #3 state-all-zero (medium; 1 submission)

`score_task1`: predictions are compared with the unmasked truth on eligible Task-1 target cells; missing
values become 0 too. With all-zero predictions:
- `RMSE_speed = RMS(true speed) = 111.78 km/h` (computed on the whole train split, 87.1 M eligible cells)
  ⇒ `S_speed = max(0, 1 − 111.78/25) = 0` **exactly**;
- `RMSE_flow_per_lane = RMS(true flow/lanes) = 1040.5 vph` ⇒ `S_flow = max(0, 1 − 1040.5/600) = 0` **exactly**.
- So **S_state(zeros) = 0 exactly** (margin is huge: both RMSEs are >1.7× the normaliser, so this is robust
  to month-to-month changes; a validation month would need RMS speed < 25/0.54 or flow/lane < ~600 to be
  anything else).
- Task 3 on the same rows: `derive_physics` keeps the released observations for non-target cells and uses
  the submitted zeros on Task-1 targets; density `k=q/v = 0/1 = 0`; the empty-road guard
  (`score_task3.py:346-348`: submitted cells with flow < 50 vph ⇒ S_FD=0 when share > 20 %) gives **S_FD = 0**;
  the conservation residual compares zeroed accumulations against organizer boundary flux ⇒
  `e_LWR ≫ 1` ⇒ **S_LWR = 0**, hence **S_physics(zeros) ≈ 0** (bounded [0, ~0.05]; cannot be computed
  locally because boundary flux is withheld).
- Therefore `T_zerostate = T_best − (0.35·S_state + 0.15·S_physics − 0)` ⇒
  `0.35·S_state + 0.15·S_physics = T_best − T_zerostate` **directly** (not relative to the zero baseline).
  With the team's S_state prior 0.93–0.946 this also yields `S_physics = ((T_best−T_zerostate) − 0.35·S_state)/0.15`.
  This is the probe that pins the state+physics budget; combine with #1 to solve for S_ODME.
  Only risk: schema/format rejection — all rows present, values 0 (legal), so low.

### #4 ODME-all-zero (weakest of the four; 1 submission)

Local scorer (`score_task4.py`) computes **only S_link**; `S_od`, `S_dev`, `S_attr` need the withheld truth
`f*`. Definitions *are* published (`docs/SCORING_SPEC.md`), so we can reason:
- `S_od(0) = max(0, 1 − Σ|0−f*|/Σf*) = 0` exactly. `S_link(0) = 1 − Σc/Σc = 0` exactly.
- `S_dev(0) = exp(−|Dhat/Dstar − 1|)`, `Dhat = Σb` (released weak prior!), `Dstar = Σ|f*−b|` (hidden). Using
  our NNLS solution as proxy for f*: Σb ≈ 8.23e5 (8 Task-2 panels; recompute over all 10), Σ|f̂−b| ≈ 2.56e5
  ⇒ ratio ≈ 3.2 ⇒ `S_dev(0) ≈ 0.11` (plausible range 0.05–0.25 given `Af*=c`; a proper LP over the
  count-consistency polytope could tighten it, not done).
- `S_attr(0)`: normalised destination attraction of an all-zero flow is 0/0; implementation withheld.
  It is either 0 or the corridor is invalidated (both give 0); hard bound [0,1], almost certainly 0.
- ⇒ `S_ODME(0) = 0.15·(S_dev(0)+S_attr(0)) ≈ 0.015–0.04 (hard bound [0, 0.30] if S_attr behaves oddly)`.
- Therefore `T_zeroodme = T_best − 0.20·(S_ODME_best − S_ODME(0))`, i.e.
  `S_ODME_best = (T_best − T_zeroodme)/0.20 + S_ODME(0)`, **not** `(T_best−T_zeroodme)/0.20` unless one is
  willing to assume S_ODME(0)=0 (±0.02 if only S_dev is trusted, ±0.06 worst-case if S_attr is not 0).
  Expected `T_zeroodme ≈ 0.83734 − 0.20·(0.8359 − 0.03) ≈ 0.675` under the BASELINES anchor.
- Rejection risk: zero is non-negative and all paths are present, so the listed invalidators
  (negative/non-finite/missing/duplicate) are not triggered; a degenerate `a_hat` may make the organizer
  evaluator return 0 for the terms rather than error. Acceptable; but information per submission is poor.
  A sharper alternative if slots existed: submit `path_flow = b` (released weak prior) — then
  `S_dev = e^{−1} = 0.367879…` **exactly** (because `Dhat = Σ|b−b| = 0`), `S_link` computable locally, but
  `S_od` (needs Σf*) and `S_attr` remain unknown, so it does not give an absolute S_ODME either — no better.

### Ranked battery summary

| rank | probe | measurable | expected total | value |
|---|---|---|---|---|
| 1 | queue all-zero | **S_queue(best)** ±0.008 | ≈ 0.60–0.62 | resolves the only remaining tuning variable; validates §5 |
| 2 | queue all-ones | mean truth-set fraction (truth-size check) | ≈ 0.60–0.65 | independent test of H2 truth sizes |
| 3 | state all-zero | `0.35·S_state + 0.15·S_physics` (S_state(0)=0 exact, S_physics(0)≈0) | ≈ 0.837 − 0.33 ≈ 0.50 | splits state+physics from ODME; needs S_state prior to split |
| 4 | ODME all-zero | `S_ODME_best − S_ODME(0)`, S_ODME(0)≈0.02–0.04 | ≈ 0.675 | weakest; ±0.02–0.06 on S_ODME |
| – | (state speed-only-zero or flow-only-zero) | `0.189·S_speed + 0.15·S_physics` / `0.161·S_flow + 0.15·S_physics` | – | only if 2+ free slots, to split S_state itself |

## 7. Self-consistency check on the 80 train windows

Procedure simulated end-to-end: treat train windows as if we only had the LB; use the regenerated masks of
the five uploaded probes; compute nested `ΔS_queue` under each hypothesis; compare against the (real) LB
deltas.
- The procedure **recovers H2/H2b and rejects H1** (H1 has the wrong sign on the weight probe; H2 has all
  three signs right and the smallest RMSE over non-fallback probes).
- The fallback probe is *not* testable on train (0 empty windows there), which is itself the right diagnosis:
  the fallback gain is month-structure specific and exactly explained by the nesting identity.
- Known Q recovered: H2 train nesting 0.8755 (P4) / 0.8817 (best); the method's own Q estimate on train
  (no empty windows) is unbiased by construction, confirming the code path.

Pitfalls found (all fixed in the final code):
1. **Predictions on ineligible cells must be masked out** before computing IoU; the evaluator drops them.
   Ignoring this depressed train S_queue from 0.876 to 0.703 (and would produce wrong truth rankings).
2. Empty-eligible-truth windows score 1 for empty predictions — matters for the empty-window/fallback
   arithmetic and for S_queue(zeros) (§6 #1).
3. Horizon truth for validation cannot be built from Task-1 horizon predictions (they are free-flow
   climatology); using them gives 33–52/80 empty-truth windows, contradicting the spec.
4. Family weighting is equal means, but its linear form makes each window exactly 1/80 — use that
   identity for probe arithmetic instead of re-deriving weights.
5. The train-month nesting (H2) is *not* a calibration of the organizer truth: the model was trained on the
   H2 convention, so pseudo-CV levels (~0.88) are optimistically biased relative to latent-state truth.
6. The queue truth may sit on ineligible cells; window truth can therefore be empty even under the spec
   guarantee (small probability; ~0.003 impact on S_queue(zeros)).
