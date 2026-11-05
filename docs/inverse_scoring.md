# SCORING_MECHANISM.md — how TrafficFlowBench scores us, what the hidden data is, and where we lose points

**Author:** `organizer-intent` (delegated by the Lead)
**Date:** 2026-11-03 (Asia/Hong_Kong). Kaggle deadline per Lead brief **2026-11-07 06:55 UTC**; team docs say 11-06 — **confirm on the competition page** (see §7.4). Code + PDF due **2026-11-10 23:59 AoE**.
**Scope:** read-only analysis. Every claim is either **M** (measured by me from the released assets, this session), **C** (read from the released evaluator/code with file:line), **D** (read from released docs) or **A** (assumption/inference — labelled). Nothing outside `work/research/organizer/` was modified.
**Primary sources:** `repo/trafficflowbench-public-main/src/**` (evaluators), `repo/.../docs/*.md`, `kaggle_public/config/*.json`, local team evidence in `work/{EXPERIMENTS,task1,task2,task2_seq,task3,task4}/`. Public notebooks: `work/research/{tfb-report,lb-0-87141,trees-unet-cnn-lb-0-87,trafficflowbench-eleven-anomalies-of-the-data}/cells.txt`.

---

## 0. Executive summary

1. **The total is a fixed weighted sum, and a missing task scores 0, not "skip"**: `S_total = 0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME` (`src/release/score_overall.py:20`, `:51-58`; docs/SCORING_SPEC.md:11-13). A component that is missing/empty/non-numeric is read as **0.0** (`release/score_overall.py:23-40`).
2. **Each task aggregates over equal-weight panels/families, not over rows** — and the hierarchy differs per task. Task 2 uses **4 families (8 panels)**; state/physics/ODME use all **10 panels (5 families)**. Physics aggregates over the **10 panels directly** (no family layer). See §1.5.
3. **Task 1's truth is the noisy observation layer, not the latent state** (`src/task1/score_task1.py:3-6, 39-45, 144-147`). Task 2's truth is the **latent state** thresholded at `v_cut = 0.60*free_speed` (docs/SCORING_SPEC.md:73-75). This asymmetry is the single most under-appreciated fact: Task-1's ceiling is the measurement-noise floor (~0.93-0.95 `S_state`), while Task-2's ceiling is not noise-limited in the same way.
4. **Task 3 has no submission of its own and no free parameters.** It is computed from our Task-1 speeds/flows with organizer-held boundary flux "projected onto the released observations" (`src/task3/score_task3.py:219-229, 244-256`). The public evaluator's topology fallback floors `S_LWR` at 0 for *everyone* (docs/RUN_LOCAL.md:49-64), so Task 3 is only observable through the leaderboard.
5. **Task 4's hidden truth is a path-flow vector whose identified part is fixed by the released counts**: `c = A f*` exactly (noiseless), `rank(A_counted) = nzones-1` for 630-5050 paths, so `S_link` is saturated (~0.99993) and all headroom is in `S_od`/`S_dev`/`S_attr` against `f*` we cannot see (work/task4/NOTES.md §1-2; docs/SCORING_SPEC.md:133-152).
6. **The only scoring convention that is provably exploitable is Task 2's empty-prediction case**: every scheduled window is guaranteed to contain a queue in its horizon (`kaggle_public/config/synthetic_release_v1.json` task2.requires_queue_in_horizon; docs/SCORING_SPEC.md:66-71), so for a window where our rule predicts nothing, marking the single most likely link is **weakly dominant**. This is our fallback family: LB `+0.0075…+0.0083` total. Nothing on Task 1/3/4 is similarly exploitable — see §5.
7. **Our current submission loses most on Task 2** (~0.083 of the 0.163 remaining loss), then Task 4 (~0.033, invisible), then Task 1 (~0.025, near its noise floor), then Task 3 (~0.023, of which ~0.006 is the organizer's own ceiling). Component table in §4.2.
8. **Measured data-side asymmetry that matters:** on the **scored** splits the 18 steps `T+5…T+90` after every Task-2 window origin are blanked in the **input** masked layer, while on **train** they are not. Extra unscored input blanks: 3.5-4.7% of the template, only on the 6-8 window days. But only **~0.3-0.6% of Task-1 target cells** are materially affected (§ Appendix A.8) — this is a small covariate shift, not the main holdout→LB transfer gap.
9. **The calibration probe plan needs a third submission** (`cal_zerostate`: both state channels zeroed). The queue-zero probe measures `Q_best` to ±0.005-0.013 (window weight is exactly 1/80), but the ODME-zero probe only measures a *difference* (`S_ODME(0) ∈ [~0.015, ~0.10]`, not 0). A **state-all-zero probe has an exactly-zero baseline** (`S_state(zeros)=0`, `S_physics(zeros)=0`) and therefore pins `0.35·S_state + 0.15·S_physics` exactly; combined with the queue-zero probe it resolves `S_ODME` with no degenerate-score assumption. This also settles the one real tension in the table: **`inverse-scoring` independently estimates `Q(best)=0.80 [0.73,0.85]`** (from the 80 train windows: `0.8817 − 0.0771 + 0.0250 − ~0.03`), which under the LB identity requires either `S_physics ≈ 0.70` or `S_ODME ≈ 0.72` or `S_state ≈ 0.865` — see the scenario table in §4.6. All scenarios agree queue is the largest gap.
10. **Honest bottom line for the last two days:** Tasks 1, 3 and 4 are frozen (no evidence-based lever survives); Task 2's fallback family is the only axis still yielding gains and is structurally safe for the *private* month; everything else is public-month noise. Freeze the final candidate by 2026-11-06 and spend the remaining time on final-artifact validation and the code+PDF deliverable (§7).

---

## 1. The scoring chain, exactly

### 1.0 One upload, four scores

The upload is one table keyed by the organizer's `submission_id`
(`submission_id,task,speed_kmh,flow_vph,queue_pred,path_flow`; `src/merge_submissions.py:1-11, OUT_COLUMNS`). It contains **both** months in one file: 3,451,376 validation + 3,284,419 private state rows, 87,000+87,000 queue rows, 35,354+35,354 ODME rows = **6,980,503 rows** (M; reconstructed from `kaggle_public/submission_key.csv` and the templates). Public LB = validation month; private LB = private month (README.md:11-25, DATA.md:38-58). This matters strategically: **the rank-deciding score is a month we never see.**

`merge_submissions.py` silently fills any unmatched cell with **0** and only prints a warning (`:84-100`). A final upload must therefore be gap-checked, not assumed.

### 1.1 Task 1 — state reconstruction (weight 0.35)

For each panel `p` and mask regime `r ∈ {R1,R2,R3}`:

```text
RMSE_speed(r)      = sqrt( Σ (v_hat - v_true)^2 / n )            # over all target cells of (p,r)
RMSE_flow(r)       = sqrt( Σ ((q_hat - q_true)/lanes)^2 / n )
S_speed(r)         = max(0, 1 - RMSE_speed(r) / 25)
S_flow(r)          = max(0, 1 - RMSE_flow(r)  / 600)
S_state(p,r)       = 0.54 * S_speed(r) + 0.46 * S_flow(r)
S_state(p)         = mean_r S_state(p,r)                          # equal over R1,R2,R3
S_state(family)    = mean over the family's 2 panels
S_state            = mean over the 5 families
```

- Code: `src/task1/score_task1.py:185-189` (RMSE + formula), `:220` (equal mean over regimes), `:260-276` (family), `:278-294` (overall). Constants: `src/task1/baseline_task1_historical_mean.py:22-27` (`SPEED_NORMALIZER=25`, `FLOW_NORMALIZER=600`, `SPEED_WEIGHT=0.54`, `FLOW_WEIGHT=0.46`, `REGIMES={R1:.2,R2:.3,R3:.5}`). Docs: SCORING_SPEC.md:30-50.
- **Per-lane flow**: `(pred_flow - true_flow)/lanes`, lanes keyed by station from `fd_parameters.csv` (station→link fallback), clipped ≥1 (`score_task1.py:166-168`; `baseline_task1_historical_mean.py:140-192`). Docs: SCORING_SPEC.md:41-46.
- **RMSE is pooled over all cells of a (panel,regime)**, not a mean of per-day RMSEs (`:169-171, :185-186`).
- **Targets** = the released template rows (`task1/<PANEL>/<split>/sample_submission_state.csv`); in the evaluator the target set is `is_score_eligible ∧ both-channels-blank` (`:140-143`), which is identical on **train** but **not** on the scored splits (see §2.2). Docs: MASK_SPEC.md:9-16, 74-88.
- **Missing / non-finite prediction → 0 and still counted in the denominator** (`:156-172`); extra rows are ignored (`:151` merge from truth, left join); duplicate keys are rejected outright (`:79-90`); unknown regime is only a warning (`:93-95`).
- **A (panel,regime) with no scoreable cell scores `S_state=0.0`, not 1.0** (`:175-183`, deliberate).
- **Clipping**: the two channel scores are each floored at 0 *before* the 0.54/0.46 mix, so a regime cannot drag the panel below its other two regimes' contributions.

### 1.2 Task 2 — queue forecasting (weight 0.30)

Per window, over **eligible** truth cells only:

```text
IoU(window) = |Qpred ∧ Qtrue| / |Qpred ∨ Qtrue|      ; 1.0 if the union is empty
S_queue(panel,condition) = mean of the panel's 5 windows of that condition
S_queue(panel)           = mean of the panel's 2 condition means
S_queue(family)          = mean of the family's 2 panels
S_queue                  = mean of the 4 scored families          # uniform per-window weight = 1/80
```

- Code: `src/task2/score_task2.py:54-78` (window IoU, eligibility, union=0→1, missing→0), `:112-131` (window→(panel,condition)→panel→family→overall). Docs: SCORING_SPEC.md:52-78.
- **Binary only**: a single non-{0,1} value rejects the whole file (`:37-45`). Duplicate keys rejected (`:46-51`). Missing rows become `queue_pred=0` (`:73`).
- **Eligible cells only**: `eligible = target[target.is_score_eligible]` (`:56`); a window with **no** eligible cells returns **0.0** (`:57-67`), not 1.0.
- Windows: exactly **5 `queue_onset` + 5 `queue_ongoing` per panel per split**, 8 scored panels, `D12_I405_N/S` excluded from Task 2 only (window_index M; SCORING_SPEC.md:64-65, 77-78; contract `tasks.queue.excluded_panels`).
- Truth definition: **latent/underlying state**, `speed <= v_cut`, `v_cut = 0.60*free_speed` (SCORING_SPEC.md:73-75); selector guarantees a queue in the horizon and `history_coverage >= 0.7`, `maximum_persistence_iou = 0.9`, minimum origin spacing 360 min (`synthetic_release_v1.json` task2 block).

### 1.3 Task 3 — physical consistency (weight 0.15)

No submission. It reads our **Task-1 speeds and flows** and reconstructs a full field: truth observations at non-target cells, our values at target cells (`src/task3/score_task3.py:48-71, 244-322`).

```text
k = q / max(v,1)                     density is an identity
N = k * L                            accumulation follows
FD (per lane):  q_fd = v_f*k                    if k <= k_crit
                       = cap/(k_jam-k_crit) * (k_jam-k)   otherwise
                S_FD = max(0, 1 - Σ|q_lane - q_fd| / Σ|q_lane|)
LWR (total flow, forward 5-min steps only, dt ∈ (0, 5.1/60] h):
                rhs  = dt*(q_in + r_on - q_out - r_off)
                E_LWR = Σ|dN - rhs| / (Σ|rhs| + 1e-9)
                S_LWR = max(0, 1 - min(1, E_LWR))
S_physics(p,r) = (S_FD + 2*S_LWR)/3
S_physics(p)   = mean over R1,R2,R3 ;  S_physics = mean over the 10 panels
```

- Code: `:271-273` (density/accumulation), `:337-345` (FD per lane), `:394-396` (LWR + mix), `:447-448` (panel mean then 10-panel mean). Docs: SCORING_SPEC.md:80-129.
- **Empty-road guard**: if `>20%` of *submitted* (`is_target`) cells in a (panel,regime) have `flow < 50 vph`, `S_FD = 0` (`:26-27, :346-348`). A pure penalty; our flows are nowhere near it.
- **Ramp/transition filters (all organizer-side, submission-independent)**: mode A keeps only transitions with valid on- and off-ramp evidence (`:388-391`; structural no-ramp links are valid, `:288-294, :174`); only links whose mainline neighbours are present are "observable" (`:362-381`, a no-op for a template-complete submission); forward interval only, `dt <= 5.1 min` (`:382-387`). All ten panels are **mode A** (`config/task3_lwr_modes.json`; docs/TASK3_LWR_COVERAGE_MODES.md:25-48).
- **Boundary flux** is organizer-held and **never published**; it is "projected onto the released observations, so a participant who reproduces what was published satisfies the conservation identity exactly" (`:219-229`, `:296-307`; CLI `--boundary-flux` `:407-409`). Without it the evaluator falls back to topology flux and `S_LWR = 0` for everybody including the truth (docs/RUN_LOCAL.md:49-64; tfb-report §3.3).
- **Aggregation is over the 10 panels, not over families** (contract `overall_score.aggregation.physics`; `:447-448`). An empty (panel,regime) scores 0.0 (`:326`).

### 1.4 Task 4 — ODME (weight 0.20)

```text
S_od   = max(0, 1 - Σ|f_hat - f*| / max(Σf*, eps))
S_link = max(0, 1 - Σ|A f_hat - c| / max(Σc, eps))          # on the counted links only
S_dev  = exp(-|D_hat/D_star - 1|),  D_hat = Σ|f_hat - b|,  D_star = Σ|f* - b|
S_attr = max(0, 1 - 0.5*L1(a_hat, a_star)),  a = normalised destination-attraction distribution
S_ODME = 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr
```

- Local evaluator computes **S_link only** against the released per-split counts (`src/task4/score_task4.py:53-85`), on the counted links (`:62-66`), with the split's own prior used only for the reference build. Docs: SCORING_SPEC.md:133-152; README.md:295-329.
- Missing/duplicate/invalid (negative/non-finite) or a non-matching path set → **S_link = 0 for that panel** (`:73-82`). Per docs "invalidate the affected corridor" (SCORING_SPEC.md:150-152).
- Aggregation: panel → family (2 panels) → mean over 5 families (`:108-115`).
- `S_od`, `S_dev`, `S_attr` need `f*` and are hidden; the local builder's own reference must **not** be used as a local score (RUN_LOCAL.md:98-101).

### 1.5 Aggregation map (equal means at every level — the reason panel size does not matter)

| task | level 1 | level 2 | level 3 | level 4 | panels |
|---|---|---|---|---|---|
| state | `(panel, regime)` pooled RMSE | panel = mean of 3 regimes | family = mean of 2 panels | total = mean of 5 families | 10 |
| queue | per-window IoU | `(panel,condition)` = mean of 5 windows | panel = mean of 2 conditions; family = mean of 2 panels | total = mean of 4 families | 8 |
| physics | `(panel, regime)` mean `S_physics` | panel = mean of 3 regimes | — | total = mean of the 10 panels | 10 |
| ODME | panel | family = mean of 2 panels | — | mean of 5 families | 10 |

C: contract `overall_score.aggregation`; code refs in §1.1-1.4. **Consequences we act on:**

- A panel counts as much as any other panel: **one broken panel costs 1/10 of the state score (0.35/10 = 0.035), 1/10 of physics (0.015), 1/10 of ODME (0.02)**. Insurance on per-panel completeness is worth more than any tuning.
- **Equal-over-regimes** means a regime with few days (M: `D7_I405_N` validation R3 has only **4 days**; `D7_I210_W` private R1 has 6; max 15) is one third of that panel's state score — same weight for every participant, but it makes `S_state` noisier per panel and means R3 accuracy matters as much as R1.
- Task 2's uniform per-window weight is exactly **1/80** (5 windows × 2 conditions × 2 panels × 4 families), which is why any single-window convention (empty prediction, fallback) has a quantifiable value of 1/80 = 0.0125 IoU.

---

## 2. Public (local) evaluator vs the hidden evaluator

### 2.1 What each scorer needs, and what is withheld

| task | evaluator needs | released? | local runnable? | what is actually hidden |
|---|---|---|---|---|
| 1 | submission; unmasked observation layer at target cells; lane counts; target set | observations only for `train`; lanes + template for all splits | **train only** (`score_task1.py:39-45, 102-106`) | the unmasked values at target cells for validation/private |
| 2 | submission; `task2/<split>/queue_targets.parquet` (`queue_true`, `is_score_eligible`, panel/condition) | **no** for every split (`score_task2.py:95-101`; RUN_LOCAL.md:77-80) | **no** | the latent-state queue labels |
| 3 | Task-1 submission; unmasked obs at **non-target** cells; organizer boundary flux; ramp obs; FD params; frozen mode table | all but (a) and (b) | **no** (fallback floors `S_LWR`) | (a) plain layer for scored splits; (b) `organizer_truth/task3/physics_truth.parquet` |
| 4 | submission; `f*`; `c` per split; `b` per split; `A`, path set | counts/prior/A/paths/`b` all released per split | **S_link only** | `f*` (hence `D_star`, `a_star`) |

`release/score_overall.py:1-8` says it plainly: the reference aggregation "is not something you can run end to end on the public package: Task 2 labels and the Task 3 boundary flows are withheld, so those two reports cannot be produced locally and count as zero here. A local total is therefore always lower than a leaderboard total, and the two are not comparable."

### 2.2 Documented and measured differences

1. **Task-1 target selection on scored splits.** `score_task1.py` defines targets as eligible cells whose speed *and* flow are blank (`:140-143`). On **train** that equals the template exactly (M: for 2030-06-03, a day hosting 3 released windows, template = eligible∧blank = 5,662 rows, and `|template − blank| = |blank − template| = 0`; same on 2031-01-31 and four other sampled days). On **validation/private** it does **not**: the release also blanks the Task-2 horizon + buffer, which are *not* Task-1 targets (MASK_SPEC.md:66-72). M: extra unscored eligible blanks are 11,675 (D7_I10_E/validation), 25,295 (D12_I5_N/validation), 12,653 and 25,655 (private) = **3.5-4.7% of the template**, and they exist only on the 6-8 window days. If the hidden evaluator scored those as targets-with-zero-predictions, `S_state` would fall to ~0.35-0.40 and even a **perfect** queue and the physics ceiling would cap the total at ~0.75 (0.35·0.40 + 0.30·1 + 0.15·0.9636 + 0.20·0.836 = 0.75) — below our 0.837. So **the hidden evaluator uses the template / pure mask, not the blank-selection.** (This is also stated indirectly in `score_task3.py:32-39`: the mask hash draws "a slightly larger set than the release publishes, because a cell that falls inside a released Task 2 window is dropped from the Task 1 targets".)
2. **Task-1 truth**: observed values, not latent (`score_task1.py:3-6, 144-147`). Task 2 truth: latent (SCORING_SPEC.md:73-75). Tasks 1 and 2 therefore have different noise regimes from the same field.
3. **Task 3 hidden path**: the organizers run the same evaluator with `--boundary-flux organizer_truth/task3/physics_truth.parquet` on a release that ships the plain layer for the scored split, so `files()` returns `mainline_states` (plain) at non-target cells (`baseline_task1_historical_mean.py:30-40`). The published "perfect answer" anchors (0.9636 / 0.9598) are the empirical evidence that the organizer flux is a *projection* onto observations with a ~5% irreducible residual (BASELINES.md:70-74; work/task3/NOTES.md §1).
4. **Task 4 `S_link`** is scored against the released per-split counts (`score_task4.py:56-66`); the trees-unet notebook's +0.031 from "solve on each split's own counts" (cells.txt §Experiment log) confirms the released per-split counts are the scored counts. `"private complete counts"` are listed as withheld (SCORING_SPEC.md:156) — the only plausible reading is that the hidden `S_link`/`S_od` frame may extend to links beyond the released "counted links"; our NNLS fits all incidence links, so this is harmless.

### 2.3 What the hidden evaluator most likely is (A, with the evidence above)

The hidden evaluator is the same four scripts with the withheld inputs passed in: `score_task1` on the organizer's plain layer and the template target set; `score_task2` on `queue_targets.parquet`; `score_task3` with `--boundary-flux` and the plain non-target layer; `score_task4` with the withheld reference and the released counts for `S_link`. The four task weights, the nesting and the missing-task-zero convention are fixed in `config/competition_contract.json` and `release/score_overall.py`. **Practical implication: every convention we can read in the public scripts is the scoring convention** — there is no separate hidden metric, only hidden inputs and (per the spec) a hidden private evaluator *configuration*.

---

## 3. Hidden data structure per task, and what it implies

### 3.1 Task 1: truth = observation, not latent state

- **What is withheld:** the unmasked `speed_kmh/flow_vph` at the template target cells for validation/private. Everything else about the target set is released.
- **Structure:** truth `y = x + ε`, `ε` is white multiplicative-ish noise measured on the release as **1.1-1.3 km/h on speed** and **~2.3% per detector on flow** (public analysis `trafficflowbench-eleven-anomalies-of-the-data`, cell 7; team's second-difference estimate 2.06 km/h / 31 vph, `EXPERIMENTS.md:60-61`). Detector availability is a one-step-memory coin (`is_score_eligible` ≈ 76%), no block outages (same notebook, cell 11).
- **Ceiling:** with `σ_v ≈ 1.2-2.0 km/h` and `σ_q/lane ≈ 30-31 vph`, `S_state^max ≈ 0.54(1-1.3/25) + 0.46(1-30/600) ≈ 0.949` (lower with the 2.06 estimate: 0.932). Our GBM holdout is 0.9291-0.9461 (4 panels; `work/task1/gbm/NOTES.md` multi-panel table), i.e. **at the floor**. The public top teams are at ~0.94-0.95 (tfb-report §2.1, 0.9408 on their own holdout).
- **Best estimator:** the conditional **mean** of the observed value = the conditional mean of the latent value (noise zero-mean and independent), which is exactly what an L2 GBM estimates. Post-hoc bias/smoothing corrections at level <0.3% relative are noise (tfb-report §3.2); our own density-target and tensor-completion variants lost (`task1/gbm/NOTES.md` v3/v4; `EXPERIMENTS.md:128-130`). **Task 1 is closed.**

### 3.2 Task 2: truth = latent-state threshold; the selector's invariants are the exploitable structure

**Quoted spec** (docs/SCORING_SPEC.md):
> "The ground truth is taken from the **underlying state** rather than the noisy observation, so measurement error cannot flip the indicator back and forth at the threshold. The threshold is `speed <= v_cut`, `v_cut = 0.60 * free_speed`." (`:73-75`)
> "Both conditions require a queue somewhere in the horizon, so an empty-empty window cannot be drawn as a free mark." (`:66-71`)
> "A window that genuinely has no queue and for which none was predicted scores 1." (`:62-63`)

**Selector invariants (contract, released config):**
- exactly 5 onset + 5 ongoing windows per panel per split; 8 panels; minimum origin spacing 360 min (`synthetic_release_v1.json`; M window_index).
- `history_coverage ∈ [0.7212, 0.8008]`, all above the 0.7 gate (M over all 240 windows).
- `future_coverage = 1.0` for **all** 240 windows (M) — the horizon is fully covered (unlike the history), which is consistent with the truth being defined on every horizon cell.
- `maximum_persistence_iou = 0.9`: the selector rejects ongoing windows whose horizon is too close to persistence, i.e. **scored ongoing windows are the ones where the queue changes**.
- Onset windows are the greedy-earliest origin, so their truth sits at `T+30` (verified on all 40 released train onset windows; tfb-report measured 97.5% / mean 3.52 cells).

**Ceiling / best estimator.** IoU is a set-level score, not log-loss:
- the optimal decision is the set maximising `E[|S∩T|/|S∪T|]` per window; empirically a **per-link probability threshold much lower than 0.5** (0.25-0.30) is optimal because the truth sets are tiny (eligible onset truth mean 3.10 links, M; official reported 3.52) and misses cost the union twice;
- **onset: predict the last horizon step only.** Broadcasting to all six steps collapses IoU (tfb: 0.972 → 0.166; our model writes `T+30` only);
- **empty prediction is provably dominated** (see §5/L1) — this is the fallback;
- training labels from noisy observations are *nearly* right: the latent-vs-noisy threshold flips only 0.05-0.14% of cells (team's σ estimate, `task2_seq/NOTES.md` §12), and attempts to retrain on denoised/soft labels lost on both honest protocols and would have changed the LB representation negatively (`EXPERIMENTS.md:422-478`).

### 3.3 Task 3: organizer flux projected onto the observations; no free parameters

- **What is withheld:** (a) the plain observation layer for the scored split at **non-target** cells (needed because the reconstructed field must be complete); (b) `physics_truth.parquet` boundary flux, described as organizer-held and "projected onto the released observations" (`score_task3.py:219-229`).
- **What that means:** the kept-transition set and both denominators are **submission-independent**. The residual equals, to first order, `e(t+1) − e(t)` where `e = N_sub − N_obs` is non-zero only at Task-1 target cells (each target cell enters ≤2 transitions with coefficient ±1). So:

  **`S_LWR` is, to first order, an L1 penalty on the accumulation error `N = (q/v)L` at our Task-1 target cells, with a fixed normaliser.** (Confirmed by the calibrated proxy: truth `S_LWR≈0.952`, GBM `≈0.787`, baseline `≈0`; `work/task3/NOTES.md` §1-2.)
- **Ceiling:** the official perfect-answer anchor is `S_physics = 0.9636` validation / 0.9598 on D12_I5_N (BASELINES.md:57-74; RUN_LOCAL.md:60-64) — the ~0.95 `S_LWR` residual is the flux projection's own noise and is unreachable by anyone. `S_FD` is effectively fixed at ~0.98 in practice because its denominator is dominated by truth non-target cells (and the empty-road guard is a pure penalty).
- **Best estimator:** minimise `E Σ|e_c|` over target cells ⇒ the conditional **median** of `N`. Task 1 separately wants the conditional **mean** of `q` and `v`. Under symmetric multiplicative noise these agree to `O(σ²)` (the bias is `E[q/v] ≈ (Eq/Ev)(1+σ_v²/μ_v²)` ≈ 2e-4), which is exactly why every density/smoothing post-process tested neutral-to-negative (`task3/NOTES.md` §3, §6, §7; `EXPERIMENTS.md:294-313`). **No metric-level free lunch; Task 3 is closed.**
- **The one genuine convention-level asymmetry** is distributional, not a trick: on scored splits the 18 steps `T+5…T+90` after each window origin are blanked in the *input* masked layer; on train they are observed (M: 2030-06-03 hosts 3 released windows and still has zero extra blanks). The Task-1 model is therefore evaluated on a support pattern it never saw, precisely at the cells whose `dN` drives the conservation residual. But M shows only **~0.3-0.6% of all Task-1 targets** are materially affected (Appendix A.8), so fixing it is worth at most a few thousandths (§5/L4).

### 3.4 Task 4: hidden `f*`; the identified part is exact and the null space is a modelling choice

- **What is withheld:** `f*` (path flows), hence `D_star = Σ|f*−b|` and `a_star`.
- **Structure we verified (M by the task-4 teammate, re-read here):** `c = A f*` **exactly** (within-segment count std ≤1e-12); `rank(A_counted) = nzones−1` (42-100 constraints) for 630-5050 paths; connectors are in `A` but never counted; the released weak prior `b` is **not** a noisy version of the truth (its cut profile is 21-27% relative-L1 off after best scaling, with a systematic downstream trend), but its split-specific cut wobble tracks the truth's (corr 0.42-0.73) and the error is dominated by **additive marginal/row-space** discrepancies (`work/task4/NOTES.md` §1, §5-6).
- **Ceiling / best estimator:** `S_link` is saturated (`0.999935` official; our NNLS reproduces it). All headroom is `S_od` (0.45). Because the denominator `Σf*` is **fixed**, `S_od` is an L1 score whose minimiser is the conditional **median** of `f*`; the LB probe shows the truth is much closer to the **Euclidean min-norm projection of `b` onto `{A f = c, f ≥ 0}`** than to any multiplicative/potential (entropy) reallocation — the entropy swap lost `ΔS_ODME = −0.1486` (`ΔS_od ≈ −0.33`) while `S_link`, total flow and `Σ|f−b|` were held roughly fixed (`work/task4/NOTES.md` §5; `EXPERIMENTS.md:12-14`). Under the revised DGP ("truth ≈ prior + row-space correction"), the small-λ NNLS is *the* estimator; increasing λ trades `S_link` for no null-space gain, and all reallocation families are rejected by the same evidence. **Task 4 is closed.**
- **The hidden-Task-4 unknowns we cannot resolve:** the hidden total `Σf*` (only bracketed to `[0.73,1.57]×Σf_baseline` by the published baseline `S_ODME = 0.8359` and `S_od ≥ 0.6353`), `D_star`, and `a_star`.

### 3.5 Summary table

| task | hidden quantity | measurable now? | ceiling | best estimator | our status |
|---|---|---|---|---|---|
| 1 | observed values at targets | train only | `S_state ≈ 0.93-0.95` (noise floor) | conditional mean | at floor (0.929-0.946 holdout) |
| 2 | latent queue at eligible horizon cells | no | 1.0 (but public best ~0.85-0.90) | per-window IoU-optimal set rule + dominant fallback | LB-implied ~0.72-0.77; **largest gap** |
| 3 | organizer flux + plain non-target field | proxy only | `S_physics = 0.9636` (projection residual) | conditional mean of q,v (≈ median of N) | proxy ~0.85, LB-implied ~0.78-0.90 |
| 4 | `f*`, `D_star`, `a_star` | `S_link` only | 1.0 | min-norm projection of the prior onto the counts | S_ODME = baseline 0.8359 (anchor); S_od ~0.64-0.77 |

---

## 4. Where our current submission loses points

Current best: **public LB 0.83734** (ref `q_w30ong40fb1`; `EXPERIMENTS.md:365-396`). The improvement ledger we can trust (all single-variable probes): v2 0.79775 → v4 0.82448 (Task-1 GBM) → P4 0.82902 (queue blend) → `q_onfb1` 0.83652 → `q_w30ong40fb1` 0.83734, plus two rejected probes (Task-4 entropy −0.02972; queue threshold/top-k −0.0005…−0.0237).

### 4.1 The one measured identity

Only one equation constrains the four component scores:

```text
0.35*S_state + 0.30*S_queue + 0.15*S_physics + 0.20*S_ODME = 0.83734
```

Everything below the task level is a *split* of that equation using independent local proxies. I label each split **measured** / **proxy** / **assumed** explicitly.

### 4.2 Component-score table

| task | weight | estimate (public month) | range | evidence and status | addressed headroom |
|---|---:|---:|---|---|---:|
| **state** | 0.35 | **0.930** | 0.915-0.945 | **proxy**: holdout `S_state` 0.9291-0.9461 on 4 panels (official formula, train split, `task1/gbm/NOTES.md`) minus a 0.005-0.015 month/panel transfer discount; consistent with the LB identity. Not measured on the scored month. | ~0.008 (rest is noise) |
| **queue** | 0.30 | **0.724** (A) / **0.78-0.80** (reconciled) | 0.72-0.82 | **inferred**: residual of the identity after the other three, anchored by the fallback probe (+0.0075 total = +0.025 S_queue, self-consistent with 6/40 onset windows × mean IoU 0.33). Independent estimate from `inverse-scoring`: **0.80 [0.73, 0.85]**, from the 80 train windows (`0.8755/0.8817` proxy, minus the hard 7-empty-window penalty 0.0771, plus 0.0250 fallback recovery, minus a 0.03-0.05 observation-vs-latent discount). Never measured absolutely — the queue-zero probe fixes it. | **~0.05-0.08 (largest)** |
| **physics** | 0.15 | **0.850** | 0.78-0.90 | **proxy**: calibrated Task-3 proxy on the GBM holdout (`S_FD .982`, `S_LWR .787`, `S_physics .852`; `task3/NOTES.md` §5), calibrated to both published anchors to ±0.03/panel. But the v4−v2 LB delta implies the **real** smoother→GBM physics gain is only 0.11-0.16, not the proxy's 0.26 ⇒ if the proxy is optimistic in the delta, the absolute is also lower (0.78-0.85). | ~0.011-0.017 (plus 0.0055 unavoidable) |
| **ODME** | 0.20 | **0.836** | 0.82-0.85 | **anchor**: our Task-4 file is the organizer NNLS baseline, whose published `S_ODME = 0.8359` (BASELINES.md:63); `S_link = 0.999935` measured locally. `S_od/S_dev/S_attr` individually unknown; the entropy probe bounds the *reallocation* headroom as negative. | ~0.03 (nearly all invisible) |
| **total** | 1.00 | **0.8373** | — | matches LB 0.83734 | **0.163** |

**Joint-constraint caveat.** The state/physics/queue triangle is one equation with three proxies; the table is the *central* split, not three independent measurements. If physics is 0.78 rather than 0.85, S_queue becomes 0.765; if 0.70, 0.80 (§4.6 scenario B). **The "queue is the largest gap" conclusion is robust** (queue loss 0.054-0.083 vs ODME 0.033, state 0.025, physics 0.023 across the whole range). **The probes in §4.5-4.6 turn this from three proxies into measured values** — the state-all-zero probe is the missing anchor.

### 4.3 Headroom table (loss = weight × (1 − score))

| task | remaining loss | irreducible part | realistically addressable | comment |
|---|---:|---:|---:|---|
| state | 0.0245 | ~0.017 | ~0.008 | noise floor; public leaders are ~0.94-0.95 |
| **queue** | **0.060-0.083** | ~0.03 (public best ~0.90) | **~0.04-0.05** | the only axis with confirmed LB gains |
| physics | 0.0225 | 0.0055 (organizer ceiling) | ~0.017 | no lever found beyond state quality |
| ODME | 0.0328 | ~0.01? | ~0.02 | requires hidden `f*`; one structural bet already lost |
| **total** | **0.1627** | | **~0.10** | |

**Largest remaining gap: Task 2 (0.060-0.083, ≈40-51% of the remaining loss).** Second: Task 4 (0.033, but 80% of it is in `S_od` and has no local signal and one decisive negative probe). Task 3 and Task 1 are noise/because-of-Task-1 limited. This ordering is the basis for the time-boxed plan: **only Task 2 can pay in two days.**

### 4.4 Measured vs assumed, by component

- **Measured:** all Task-4 `S_link` values (official scorer, local); the LB total; all single-variable LB deltas; the fallback's structural validity; the 80 train-window queue proxies.
- **Proxy:** Task-1 holdout `S_state` (train distribution, 4/10 panels, no scored-split input asymmetry); Task-3 calibrated proxy (anchors matched, but the LB delta suggests it overstates *gains* by ~1.6-2.4×).
- **Assumed/inferred:** absolute `S_queue`; absolute `S_physics`; `S_od/S_dev/S_attr`; the private-month values of everything.
- **What the LB probes already proved:** the queue blend (+0.01513 S_queue), the fallback family (+0.025 S_queue), the entropy reallocation (−0.33 S_od), and the Task-1 GBM switch (+0.02673 total, which *bounds* the real physics gain).

### 4.5 The two calibration submissions queued for 2026-11-04 — what they will and will not measure

**`submission_cal_zeroqueue` (queue_pred all 0; state + path_flow unchanged).**

```text
total_zeroqueue = total_best − 0.30*(S_queue(best) − S_queue(0))
S_queue(0) = E/80        (M: window weights are exactly uniform 1/80)
   E = number of the 80 validation windows whose ELIGIBLE truth set is empty.
=> Q_best = (total_best − total_zeroqueue)/0.30 + E/80
```

- The `eligible.empty → score 0` guard cannot fire: M, every validation window has **360-1643** eligible horizon cells (mean 831); none has zero.
- `E` is very likely 0 and at worst 1-2. On the 80 released **train** windows (observed-speed proxy, eligibility-masked) the eligible truth set was empty in **1/40 onset, 0/40 ongoing** — and that one window is empty even *without* eligibility filtering (all-cells onset mean 3.52 vs eligible 3.10), so it is a truth-definition/observed-vs-latent artefact, not eligibility removing a real queue. The official truth is latent (less noisy) and the selector guarantees a queue, so **E ≈ 0; report Q_best with +0.000/+0.025 (E = 0..2), i.e. ±0.0075 on the total**. This probe is the single highest-information submission we have left.

**`submission_cal_zeroodme` (path_flow all 0; queue + state unchanged).**

```text
S_od(0)   = 0 EXACTLY   (f_hat=0 => Σ|f_hat−f*| = Σf*)
S_link(0) = 0 EXACTLY   (f_hat=0 => Σ|A f_hat−c| = Σc; no invalid-guard fires: all-zero is finite/non-negative/complete)
S_dev(0)  = exp(−|Σb/D_star − 1|) > 0     (D_hat(0) = Σb; Σb ≈ 823k across the 10 panels, NNLS-proxy D_star ≈ 256k ⇒ ≈ 0.11)
S_attr(0) ∈ [0, 0.75]                     (0/0 destination distribution; 0.5 if a_hat=0-vector, 0.25-0.75 if uniform, 0 if unguarded)
=> S_ODME(0) ∈ [~0.015, ~0.10], central ~0.03-0.05.  NOT 0.
=> T4_best = drop4/0.20 + S_ODME(0);  drop4 alone is ΔS_ODME.
```

- So the probe measures a **difference**, not the absolute. Absolute anchor already exists: our file is the organizer NNLS baseline, so `S_ODME(best) = 0.8359` (BASELINES.md:63); predicted `drop4 = 0.20*(0.8359 − S_ODME(0)) ≈ 0.156` (range 0.147-0.166). **If the observed drop exceeds 0.167 = 0.20 × 0.8359, then either our file is not the published baseline or the docs anchor is from a different seed — that is the diagnostic value.** If it lands in range, `S_ODME(0) = 0.8359 − drop4/0.20` is measured, and the Task-4 table becomes fully anchored.
- **But the ODME-zero probe is redundant** once the state-all-zero probe below exists: from `T_zerostate` and `T_zeroqueue` alone, `S_ODME = (T_zerostate − 0.30·Q)/0.20` with no assumption about `S_ODME(0)`. Keep it as a cross-check if slots allow, not as the anchor.

### 4.6 Reconciliation with the independent Task-2 inverse analysis (`inverse-scoring`) — and the probe that ends the ambiguity

`inverse-scoring` independently estimates (memo `work/research/inverse/TASK2_INVERSE.md`, per-window CSVs there):

```text
Q(P4)   = 0.77, 90% [0.70, 0.82]
Q(best) = 0.80, 90% [0.73, 0.85]
```

Derivation: score our frozen v3+v5 masks on the 80 released train windows with the official nesting (truth = 3-step median of the observation, thresholded at the released `v_cut`, eligible cells): P4 = 0.8755, best = 0.8817. On the **validation** month P4's rule predicts **nothing** on 7 windows (6 onset: `D12_I5_N_004`, `D7_I10_W_001/002/003/005`, `D7_I405_N_004`; 1 ongoing: `D7_I10_W_009`) ⇒ penalty `(6·0.883+1·0.868)/80 = 0.0771`; the top-1 fallback recovers the six onset windows at mean IoU ≈ 1/3 ⇒ `+0.0250`. So `Q_best ≈ 0.8817 − 0.0771 + 0.0250 = 0.830`, discounted ~0.03 for the smoothed-observation convention being optimistic vs the organizer latent truth ⇒ **0.80**. The nesting identity `ΔS_queue = (1/80)·ΣΔIoU` (each window exactly 1/80, regardless of panels/families) is confirmed by both of us; `+0.0075 total ⇒ ΣIoU = 2.0 ⇒ mean 1/3` on the six recovered windows, and since a fallback adds exactly one cell, a hit gives `IoU = 1/(1+|T_elig|)` ⇒ those six windows' eligible truth size ≈ 2.

**Why this does not match my table.** The LB identity plus `T_best = 0.83734` admits a family of consistent splits, and `Q = 0.80` sits in it only if something else is lower than my proxy:

| scenario | S_state | S_queue | S_physics | S_ODME | consistent with v4−v2 = +0.02673? | comment |
|---|---:|---:|---:|---:|---|---|
| **A** (my proxy-central) | 0.930 | 0.724 | 0.850 | 0.836 | yes (`ΔS_state +0.018 ⇒ ΔS_physics 0.136`) | proxy physics absolute; Q from inversion |
| **B** | 0.930 | 0.800 | 0.700 | 0.836 | yes (`ΔS_physics 0.136`) | Q from inverse-scoring; physics ~0.70 ⇒ the calibrated proxy overstates the *absolute* by ~0.15 |
| **C** | 0.930 | 0.800 | 0.850 | 0.722 | yes, but then our NNLS scores 0.11 below the published baseline ⇒ our file ≠ the organizer baseline (red flag) | |
| **D** | 0.865 | 0.800 | 0.850 | 0.836 | yes only if the GBM state *regressed* vs the smoother on the month (`ΔS_state −0.047 ⇒ ΔS_physics 0.288`) | possible but uncomfortable |
| **E** (reconciled central) | 0.930 | 0.780 | 0.738 | 0.836 | yes (`ΔS_physics 0.184`) | if the train-window proxy discount is 0.05 (the team's own earlier estimate) rather than 0.03 |

All of A/B/D/E are equally consistent with every LB probe we have; C is disfavoured because it contradicts the Task-4 anchor. **The queue conclusion is unchanged in all of them: queue loss 0.06-0.09, still the largest gap.** The physics/state/ODME split is what is unresolved, and it is resolvable with **one extra probe**:

**The missing anchor: a state-all-zero submission.** Zero *both* state channels on every Task-1 row (queue and path_flow unchanged). Then, exactly:
- `S_state(zeros) = 0` (RMS true speed ≈ 111.8 km/h >> 25, RMS true flow/lane ≈ 1040 >> 600 — inverse-scoring verified this on all 87.1M train eligible cells; exact, not approximate);
- `S_physics(zeros) = 0` (the empty-road guard fires: 100% of submitted cells have flow < 50 ⇒ `S_FD=0`; zeroed `N` ⇒ `E_LWR >> 1 ⇒ S_LWR=0`; speed is clipped at 1 in `k=q/v` so there is no NaN);
- therefore **`T_zerostate = T_best − (0.35·S_state + 0.15·S_physics)` exactly**, with a zero baseline of exactly 0 (unlike the ODME probe).

Combining it with the queue-zero probe gives the whole table without any prior on `S_ODME(0)`:

```text
Y  = T_best − T_zerostate = 0.35·S_state + 0.15·S_physics          (state+physics budget, exact)
Q  = Q0 + (T_best − T_zeroqueue)/0.30                              (Q0 = E/80 ≈ 0.003)
S_ODME = (T_zerostate − 0.30·Q)/0.20                               (S_ODME resolved, no S_ODME(0) assumption)
S_physics = (Y − 0.35·S_state)/0.15                                (with S_state from the 0.929-0.946 holdout, ±0.01)
```

Predicted values to discriminate the scenarios above: `T_zerostate ≈ 0.384` under A (Y = 0.453), `≈ 0.407` under B/E (Y = 0.431-0.436), `≈ 0.402` under D. The LB resolves ~1e-4, so one state-all-zero submission settles it. **This is a higher-value probe than the ODME-zero one:** if `Q ≈ 0.80` and `T_zerostate` lands near 0.384, then `S_ODME ≈ 0.72` and our Task-4 file is not the published baseline (worth knowing for the report and for a possible late Task-4 fix); if it lands near 0.407, physics is ≈0.70 and the Task-4 anchor holds.

**Optional third probe for the state split: flow-column-zero** (zero `flow_vph` only, keep speed). It is clean: `S_flow(zeros)=0` (true flow/lane RMS ≈ 1040 > 600), the empty-road guard fires ⇒ `S_physics=0`, so `T_flow0 = T_best − 0.161·S_flow − 0.15·S_physics`, and `T_flow0 − T_zerostate = 0.189·S_speed` gives **`S_speed` exactly**. **Do NOT build a speed-column-zero probe**: with speed zeroed and flow kept the empty-road guard does *not* fire (flows are normal), `k = q/1 >> k_jam` makes `q_fd = 0` only on the submitted cells, so `S_FD ≈ 1 − (share of flow in target cells) ≈ 0.5-0.8` and `S_physics ≈ 0.17-0.27 ≠ 0` — the probe is contaminated and its physics term is not computable on the scored splits.

---

## 5. LEVERAGE LIST — legitimate, ranked by expected score gain vs cost

> Ground rule: we have **2.7 days and 5 submissions/day**. Every item below states what it is, why it is legitimate, expected gain, cost and evidence. Items L1-L3 are worth doing; L4-L6 are optional; L7-L10 are explicitly not worth it. Everything is scoring-convention or estimator change, **not** hidden-data recovery or probe-tuning.

**L1. Finish the Task-2 empty-window fallback family (highest EV; provably non-negative).**
- *Convention read:* docs/SCORING_SPEC.md:62-71: every scheduled window contains a queue in the horizon ⇒ for a window where our rule predicts **nothing**, `IoU = 0` with certainty; marking its single most-likely eligible link changes the score from 0 to `1/|T| > 0` if right and stays 0 if wrong. Weakly dominant, no assumption about the month. Empirically +0.0075 total (6/40 onset windows) and +0.0008 more from the weight recalibration.
- *Still to do (pre-built candidates, ~1 slot each):* **fallback depth top-2 / minimum set 2** on windows where the rule predicts exactly one link (E[Δ] analysis with |T|≈3.1-3.5 says adding a second link pays iff its precision > ~1/3); apply the same fallback to **ongoing** windows (1/40 had an all-empty mask; spec guarantees a queue there too — already in `_all`, verify it shipped); re-check the onset threshold **with** the fallback in place (`q_on040fb1`, `q_on025fb1`) because the threshold trade-off changes once empty windows are recovered.
- *Expected:* +0.000 to +0.004 total; *cost:* 2-3 slots; *risk:* essentially zero (the fallback core is provable; the refinements are month-specific).

**L2. Use the calibration probes to resolve the component table (information, not score).**
- `cal_zeroqueue` gives `S_queue` to ±0.005-0.013 (window weight exactly 1/80; `Q = Q0 + drop/0.30`).
- **ADD a `cal_zerostate` probe (both state channels zeroed, queue + path_flow unchanged).** It is the only *exact zero-baseline* anchor for the state+physics budget: `T_zerostate = T_best − (0.35 S_state + 0.15 S_physics)` exactly (§4.6). Combined with `cal_zeroqueue` it resolves **`S_ODME = (T_zerostate − 0.30·Q)/0.20`** with no assumption about the degenerate ODME score, which is what makes it more valuable than `cal_zeroodme`.
- `cal_zeroodme` then becomes an optional cross-check (`drop4` must equal `0.20·(S_ODME − S_ODME(0))` with `S_ODME(0) ≈ 0.015-0.10`).
- Optional 4th probe: **flow-column-zero** (zero `flow_vph` only) → `T_flow0 − T_zerostate = 0.189·S_speed` exactly. **Do not build a speed-column-zero probe** (physics does not collapse; it is contaminated, §4.6).
- *Expected score gain:* 0; *value:* prevents mis-allocating the last 2 days and is required for an honest report. *Cost:* 2-4 slots; the state-zero probe is a ~15-min build (copy the final submission, zero the state rows' `speed_kmh/flow_vph`).

**L3. Freeze and insure the final upload (expected value is higher than any remaining probe).**
- One broken panel or one `merge_submissions` gap costs ≥0.015-0.035 total; the 5/day quota and the private month make late errors unrecoverable. Checklist: row count 6,980,503 and `submission_id` order = `sample_submission.csv`; no NaN in any of the 6 columns; `queue_pred ∈ {0,1}`; state template-exact per panel (10 validation + 10 private) and per regime; Task-4 path key set complete per panel (missing path → `S_link=0` for that panel → −0.02 total); `flow<50` share ≤0.2 on submitted state cells (empty-road guard); both splits present; reproducibility package matching the uploaded bytes.
- *Expected:* avoids a −0.02…−0.15 tail event; *cost:* 2-3 hours.

**L4. (Optional, low priority) Train the Task-1 model with the scored-split input pattern (18-step horizon-shaped blanks).**
- *Convention/data read:* horizon+buffer blanks exist only on scored splits (M §2.2; Appendix A.3-A.4), and Task-3's residual is concentrated at gap-adjacent cells (`task1/gbm/NOTES.md` gap table). But M shows only ~0.3-0.6% of all Task-1 targets are materially affected (Appendix A.8), so the addressable gain is small.
- *Expected:* +0.001 to +0.002 total (mostly physics); *cost:* 0.5 day retrain + 1 slot + an LB probe that cannot resolve <0.002. **Only if L1 is exhausted and a slot is free.** Do not let it delay L3.

**L5. (Optional) Ship the bigger Task-1 model (v2).**
- `task1/gbm/NOTES.md`: +0.0024 holdout `S_state` on all 4 panels, consistently positive, below the old +0.003 ship bar. At weight 0.35 that is **+0.0008 total** if it transfers. One slot, one retrain. Low risk but below LB resolution — prefer keeping the incumbent (reproducibility already proven).

**L6. (Optional, information only) One more Task-4 λ probe** (`λ ∈ [0.005, 0.5]`, tiny perturbation: `L1(cand, baseline)/Σf < 0.05`). *Expected:* ≈0 (+2e-5 per the public report's λ sweep); *cost:* 1 slot. Skip unless free.

**L7. Do NOT repeat the Task-4 reallocation (entropy / marginal correction / gravity denoise).** Decisively rejected by the LB (−0.02972 total, ΔS_od ≈ −0.33); every member of that family reallocates the null space multiplicatively, which is the wrong DGP (`task4/NOTES.md` §5). There is no local validator for `S_od`, so the LB is the only evidence and it is negative.

**L8. Do NOT touch Task-3 post-processing.** The metric is L1 on `N` with a submission-independent denominator; the conditional mean and conditional median coincide to O(σ²); the density/interpolation adjuster, the flux-integration adjuster and the public-notebook density correction are all neutral-to-negative (`task3/NOTES.md` §3, §6, §7). The FD term is ~0.98 fixed and the empty-road guard is a pure penalty.

**L9. Do NOT chase Task-1 bias/smoothing corrections or U-Net/tensor stacks.** All tested; all below a 0.3%-relative gate or negative (`tfb-report` §3.2; `task1/gbm/NOTES.md` v3/v4; `EXPERIMENTS.md:128-130`).

**L10. Do NOT tune Task-2 thresholds on the March windows alone.** The private month is generated independently (README.md:20-26; `synthetic_release_v1.json` seed policy). The +0.0008 weight recalibration is plausible (multi-month sim agreed in sign, `EXPERIMENTS.md:244-268`) but it is a public-month micro-gain; the structural fallback is the part that transfers. Anything below a +0.0015 pre-committed bar is not worth a slot.

### 5.1 The direct answer to "can ineligible/hidden cells be exploited legitimately?"

**No.** Checked per task:
- **Task 1:** the scored set is exactly the released template; ineligible cells are neither scored nor needed; extra rows are ignored; missing rows are self-penalising (`score_task1.py:140-172`). There is no "free" cell to gain from — only ~24% of cells that we correctly do not submit.
- **Task 2:** predictions on cells absent from the truth file are free (`score_task2.py:69` merge-from-truth), but the window template already enumerates all 876 cells and `future_coverage = 1.0` (M) means nothing is dropped; predicting a superset of the true queue only grows the union.
- **Task 3:** the transition set and denominators are submission-independent (organizer data); the only submission-dependent quantity is `N` at target cells.
- **Task 4:** every counted link is scored; a missing/duplicate/negative/invalid path zeroes the panel's `S_link` (`score_task4.py:73-82`).
- The **one** exploitable convention is the Task-2 empty-prediction case, which is exactly L1 — and it is a *structural* guarantee, not a hidden-data leak.

### 5.2 The specific interactions the Lead asked about — checked, all negative

1. **Task-1 mask regime × Task-3 scoring.** The per-regime weighting does create an asymmetry: a target cell's `N` error enters only the transitions that survive the filters, and the share of transitions with ≥1 target endpoint is `1-(1-rate)²` = **36% (R1), 51% (R2), 75% (R3)** (matches `task3/NOTES.md` §3), while `Σ|rhs|` scales with all kept transitions. So per unit of target accuracy, R3 days are worth ~2× an R1 day in `S_LWR`, and `S_physics(panel)` weights the three regimes equally. **But there is no exploitable slack:** the mask regime is a property of the calendar day (MASK_SPEC.md:37-52), the state model already uses regime as a feature (`task1/gbm/NOTES.md` feature list), and per-(panel,regime) recalibration is worth 0.004% relative (tfb-report §3.2). Predicting density to game `S_LWR` is the same as predicting density honestly, and both the median adjuster and the public-notebook density correction were neutral/negative on the GBM (`task3/NOTES.md` §3, §7).
2. **Extra blanks are input holes, not scored cells.** The 3.5-4.7% extra blanks on scored splits are excluded from the Task-1 target set (template) and are filled with the organizer's *plain-layer truth* in Task 3 (`derive_physics` uses `files(panel_dir, split)` = `mainline_states` when the organizer has it). So they neither cost nor pay score directly; their only effect is the small support degradation quantified in Appendix A.8.
3. **Task-4 structural prior — none left unused.** The only structural facts are (a) `c = A f*` exactly — already used by fitting the counts; (b) the cut profile identifies only `O_k − D_k` per zone, so origin/destination marginals are *not* separately identified and `S_attr` is only weakly controllable; the released prior's destination distribution is therefore the best available `a_hat`, which is what NNLS already produces; (c) the total `Σf*` is not identified (bracketed only by the published baseline score). No unused prior, no usable normalisation trick (the denominators are all truth-fixed).
4. **Task-2 template superset.** The window template already enumerates every cell (876/window class), so there is nothing to add; the only decision is which cells to mark, and that is the threshold/fallback question.
5. **Task-3 guard arithmetic.** The empty-road guard is a pure penalty (`>20%` of submitted cells under 50 vph ⇒ `S_FD=0`), and `S_FD`'s denominator is dominated by truth non-target rows, so the FD term is effectively fixed at ~0.98 (worth ≤0.001 of total). Do not spend any time on it beyond the L3 checklist.

---

## 6. What could invalidate our current approach

1. **Private month ≠ validation month.** Everything we selected on the public LB (thresholds, blend weights, the +0.0008) is a March estimate. April has its own 7 incidents and demand draws (README.md:20-26). The **fallback is structural and transfers**; the weight recalibration is the part at risk. Final selection should prefer structural justification over the last 4 decimal places.
2. **The absolute component scores are not measured** — the inverted table is one equation plus three proxies. The `inverse-scoring` estimate `Q=0.80` vs my inversion `Q=0.724` is exactly this ambiguity: within the LB identity it forces one of B/C/D/E in §4.6 (physics ≈0.70, or ODME ≈0.72, or state ≈0.865). **The state-all-zero probe resolves it in one submission**; until it runs, do not treat the split as known (the "queue is biggest" ordering *is* robust).
3. **`S_physics` proxy overstates gains.** The v4−v2 LB delta (0.02673) is 56% of the proxy-predicted 0.048, i.e. the real smoother→GBM physics gain is ~0.11-0.16 rather than 0.26. Any future proposal justified by "proxy S_physics +0.04" must be discounted by ~2×. (Most likely causes: the proxy's calibration at the extremes does not interpolate linearly; the scored-split input asymmetry; month transfer.)
4. **Task-4's hidden total.** `Σf*` is only bracketed to `[0.73,1.57]×Σf_baseline`; if the true total sits near 1.5×, even a perfect *shape* caps `S_od` at ~0.67, and our min-norm solution is already at that cap only if the total is right. Do not scale.
5. **Any submission-format or completeness regression** (blank turned into 0 by the merger, a dropped panel, a non-binary queue value, a duplicate key, a wrong timestamp spelling) is catastrophic and silent. The canonical timestamp text matters even for scoring joins (`baseline_task1_historical_mean.py:55-88`).
6. **Deadline ambiguity.** Lead brief says Kaggle closes **2026-11-07 06:55 UTC** (14:55 HKT); team docs say 2026-11-06. Confirm on the competition page and treat the earlier date as real for planning.
7. **Quota exhaustion.** Two calibration probes + queue refinements can eat the 5/day quota. Reserve ≥2 slots on 2026-11-06 for the final re-upload/hedge.
8. **Code + PDF deliverable** (2026-11-10 23:59 AoE) is mandatory for prizes; a final-days bug scramble that leaves the report unsubmitted is a total loss regardless of LB.
9. **Team name** on the leaderboard still reads "Hsu Yooyoo" (`EXPERIMENTS.md:12`) — fix before the deadline for awards/website.

---

## 7. Time-boxed recommendations 2026-11-04 → 2026-11-06

Quota resets 00:00 UTC = 08:00 HKT; so 5 slots on Nov 4, 5 on Nov 5, 5 on Nov 6 (15 total). Treat Nov 6 evening HKT as the freeze. Before every submission: run the L3 checklist.

### 7.1 Day plan

| when (HKT) | action | slots | expected gain | rationale |
|---|---|---:|---|---|
| **Nov 4 morning** | 1. `cal_zeroqueue` 2. **`cal_zerostate` (NEW — build it tonight; both state channels = 0)** 3. `cal_zeroodme` (cross-check) | 3 | 0 (information) | together (1)+(2) resolve `S_queue` to ±0.005-0.013 and `S_ODME` to ±0.02 with *no* degenerate-score assumption; (3) cross-checks the 0.8359 anchor. This is the full component table. |
| Nov 4 afternoon | 4. upload `q_w30ong40fb1_all` (must be on the board to be selectable; one cell better and weakly dominant on the structural argument) 5. `q_on040fb1` **or** `q_min2` (whichever is pre-built; top-2/min-set on onset windows that predict exactly one link) | 2 (+1 spare) | +0.000…+0.002 | only L1 candidates; single-variable, pre-built |
| optional Nov 4/5 | flow-column-zero probe → `S_speed` exactly (one slot, §4.6) | 1 | 0 (information) | completes the state split without relying on the holdout prior |
| **Nov 5 morning** | consolidate: if a queue probe beats `q_w30ong40fb1` by ≥ +0.0015 upload it; else keep the incumbent. Optional: Task-1 v2 probe (L5) or horizon-shape retrain (L4) if it is ready and validated | 2-3 | +0.000…+0.001 | pre-committed bar; private-month transfer argument |
| Nov 5 afternoon | run the full L3 validation on the intended final candidate (both splits, all panels, all keys); build the reproducibility command list | 0 | insurance | — |
| **Nov 6 morning** | upload the **final** candidate; upload the **hedge** (P4-family / no-weight-change fallback variant); verify both files' md5 vs the validated artifacts | 2 | insurance | diversity hedge for the independent private month |
| Nov 6 afternoon | select final submission(s); confirm platform final-selection semantics; freeze; hand artifacts to the report owner | 0 | — | avoid last-hour changes |
| Nov 7-10 | code + PDF report (use this memo's formulas/line refs), reproducibility check on the uploaded bytes | — | mandatory | prize eligibility |

### 7.2 Adoption bar and freeze criterion

- **Bar:** any queue candidate must beat the incumbent by **≥ +0.0015** public LB *and* be structurally justified (not a March-only threshold tweak), following the team's pre-committed rule. The proven fallback core (+0.0075) already cleared it; the refinements are expected to be smaller.
- **Freeze by 2026-11-06 20:00 HKT** with the final upload validated. After that, no file changes — only selection.
- **Final selection:** prefer the fallback candidate (structural) over the weight-recalibrated one if forced to choose; if the platform permits two final selections, pair the fallback candidate with a *diverse* family (P4-style: same structural fallback, different thresholds/weights) rather than two near-identical files. Confirm the exact selection rule on the Kaggle page.

### 7.3 The honest "do nothing / freeze" case

- Tasks 1, 3, 4 have **no remaining evidence-based lever**: Task 1 is at its noise floor; Task 3 has no submission and every post-processing attempt is neutral/negative; Task 4's only structural bet was decisively rejected. Freeze them now.
- For Task 2, the *provable* gain (fallback) is already in the candidate. Remaining refinements are worth **0-0.002**, while a broken final upload is worth **−0.02 to −0.15**, and the private month is a different month anyway. **If the Nov 4-5 probes do not clear +0.0015, the correct decision is to freeze the current best and spend Nov 6 on validation and the report.** That is a legitimate, honest recommendation, not a failure to optimise.
- The only exception: **run the calibration probes regardless** (queue-zero + state-zero, optionally flow-zero and ODME-zero) — they cost 2-4 slots and they are the only way the final report can state the component scores honestly.

### 7.4 Deadline check

Lead brief: Kaggle deadline **2026-11-07 06:55 UTC** (= 14:55 HKT). Team docs say 2026-11-06. **Verify the exact timestamp on the competition page today**; plan to freeze at least 12 h before the earlier reading. Code + PDF e-mail due **2026-11-10 23:59 AoE**.

---

## Appendix A — measurement log (all run this session, scripts in `work/research/organizer/`)

A.1 **Template sizes** (M). Validation state 3,451,376 rows / 10 panels; private 3,284,419. Task-2 queue 87,000 per split = 10 windows/panel × 6 steps × links(panel), 1,450 links across the 8 scored panels (per-panel links 81-352; rows/window 486-2,112). Task-4 35,354 per split. Upload total 6,980,503 = 6,735,795 state + 174,000 queue + 70,708 path-flow.

A.2 **Regime day counts per panel-split** (M; `probe_regimes.py`). Every panel has all 3 regimes in both scored splits. Range: 4 days (`D7_I405_N` validation R3) to 15 days (`D7_I210_W` validation R3); e.g. `D7_I210_W` validation R1 = 6, `D7_I405_S` private R1 = 5. `S_state` weights each regime 1/3 regardless, so a 4-day regime carries a third of that panel's state score.

A.3 **Extra unscored input blanks on scored splits** (M; `probe_extra_blanks.py`): eligible∧blank minus template:
- D7_I10_E validation **11,675** (3.459% of template; 6 days; per-day 1,017-2,936)
- D7_I10_E private **12,653** (4.172%; 6 days; 980-4,907)
- D12_I5_N validation **25,295** (4.298%; 8 days; 1,782-5,762)
- D12_I5_N private **25,655** (4.724%; 7 days; 1,752-5,675)
These are the Task-2 horizon (`T+5…T+30`) + following-hour removals (MASK_SPEC.md:66-72). They exist **only** on window days.

A.4 **Train has no horizon blanks** (M; `probe_targets.py`): 2030-06-03 hosts 3 released windows (`window_index`), yet template = eligible∧blank = 5,662 rows with zero difference. Same on 2031-01-31 (R1), 2030-10-04 (R2), 2031-02-28 (R3). ⇒ the train masked layer is *not* a faithful sample of the scored-split input.

A.5 **Window index invariants** (M): 240 windows total (80 per split); exactly 5+5 per panel for the 8 scored panels; `D12_I405_N/S` have no Task-2 files; `future_coverage = 1.0` for all 240; `history_coverage` 0.7212-0.8008 (mean 0.7634), all above the 0.7 gate.

A.6 **Validation horizon eligibility** (M): per window eligible horizon cells min 360, mean 831, max 1,643; **0 windows** with zero eligible cells ⇒ the `eligible.empty → 0` guard cannot fire, and the zero-queue probe cannot be contaminated by it.

A.7 **Released-train-window queue structure** (M; observed-speed proxy): onset eligible truth 3.10 links (all-cells 3.52, matching the official report), ongoing 158 (174); empty eligible truth: 1/40 onset (that one is empty even without eligibility), 0/40 ongoing.

A.8 **Horizon-block input-support degradation** (M; `probe_shadow.py`, D7_I10_E, after excluding in-block cells which are not targets): on window days, 2.0-3.6% of that day's targets have at least one temporal anchor moved (day-weighted 2.2%); the share whose farther side is ≥8 steps goes 0.9-1.9% → 2.1-5.0%. Window days carry ~16% of a split's template rows ⇒ **~0.3-0.6% of all Task-1 targets are materially affected**. This bounds the value of an L4 gap-shape retrain to ~+0.001-0.002 total.

A.9 **Fallback arithmetic consistency** (C/M): 6/40 onset windows predicted nothing at threshold 0.30; the LB gain +0.0075 = 0.30·ΔS_queue ⇒ ΔS_queue = 0.025; with uniform window weight, `ΔS_queue = 0.5·(6/40)·IoU_avg` ⇒ mean IoU on the recovered windows ≈ **0.33**, exactly what a 1-predicted-link vs ~3-truth-link IoU looks like. This confirms the nesting and the weight arithmetic.

A.10 **Regime day-count examples** (M): `D7_I405_N` validation R3 = 4 days; `D7_I210_W` validation R3 = 15 days.

## Appendix B — line-referenced formula index

| quantity | file:line |
|---|---|
| task weights / total / missing→0 | `src/release/score_overall.py:20, 51-58, 23-40` |
| Task-1 RMSE, formula, clamps | `src/task1/score_task1.py:185-189, 156-172` |
| Task-1 regimes / normalisers / weights | `src/task1/baseline_task1_historical_mean.py:22-27` |
| Task-1 targets (blank∧eligible) | `src/task1/score_task1.py:117-146` |
| Task-1 per-lane flow | `src/task1/score_task1.py:166-168`; `baseline:140-192` |
| Task-1 empty→0 | `src/task1/score_task1.py:175-183` |
| Task-2 window IoU / union=0→1 / empty→0 | `src/task2/score_task2.py:54-78` |
| Task-2 nesting | `src/task2/score_task2.py:112-131` |
| Task-2 binary / duplicate rejection | `src/task2/score_task2.py:37-51` |
| Task-2 truth point | `src/task2/score_task2.py:95-101`; SCORING_SPEC.md:73-75 |
| Task-3 density / accumulation | `src/task3/score_task3.py:271-273` |
| Task-3 FD per lane | `src/task3/score_task3.py:337-345` |
| Task-3 empty-road guard | `src/task3/score_task3.py:26-27, 346-348` |
| Task-3 ramp validity / mode A | `src/task3/score_task3.py:174, 288-294, 388-391` |
| Task-3 kept transitions | `src/task3/score_task3.py:362-387` |
| Task-3 S_LWR / mix | `src/task3/score_task3.py:392-396` |
| Task-3 boundary flux (hidden) | `src/task3/score_task3.py:219-242, 296-307` |
| Task-3 aggregation | `src/task3/score_task3.py:447-448` |
| Task-3 mode table | `repo/.../config/task3_lwr_modes.json` |
| Task-4 S_link | `src/task4/score_task4.py:53-85` |
| Task-4 validity guard | `src/task4/score_task4.py:73-82`; SCORING_SPEC.md:150-152 |
| Task-4 formulas / weights | SCORING_SPEC.md:138-148 |
| aggregation contract | `kaggle_public/config/competition_contract.json` `overall_score` |
| selector invariants | `kaggle_public/config/synthetic_release_v1.json` `task2` |

## Appendix C — open questions for `inverse-scoring` (status after the 2026-11-03 cross-check)

1. **Resolved in principle, pending the probe:** absolute `S_queue` on the validation month = `Q0 + drop0/0.30`, and `inverse-scoring`'s independent estimate is 0.80 [0.73,0.85]. Their derivation identifies the exact 7 empty-rule windows on validation (6 onset, 1 ongoing) — that is a hard fact, not an estimate, and the fallback recovers the 6 at mean IoU 1/3.
2. **Re-framed:** the ODME-zero drop is predicted to be 0.147-0.166 for `S_ODME(0) ∈ [0.015,0.10]`; a drop outside that band means our file ≠ the published baseline or a different seed. But the **state-all-zero probe** makes this redundant: `S_ODME = (T_zerostate − 0.30·Q)/0.20` needs no degenerate assumption.
3. **Still open and now decisive:** which of §4.6 B/C/D/E is true. The state-zero probe (`T_zerostate ≈ 0.384` for A, `≈ 0.407` for B/E, `≈ 0.402` for D) plus the queue-zero probe answers it. Until then, the honest statement is the scenario table, not a point estimate.
4. **New (for the report):** why would the calibrated Task-3 proxy match both published anchors yet overstate the GBM's absolute `S_physics` by ~0.15 (scenario B)? If the state-zero probe lands near 0.407, that is a real calibration finding about projection-residual jitter and should be in the report's limitations section.
