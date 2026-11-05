# Task 4 - ODME / path-flow estimation: findings, methods, candidates

Owner: task4-odme. All work in `work/task4/`. Data is read directly from
`C:\Users\lenovo\Desktop\kaggle\kaggle_public` (read-only); the Lead's cache is not required.

## 1. What the Task-4 data actually is (verified, differs from the task brief)

* Ten directional panels. Each panel is a **linear corridor**: one path per
  ordered zone pair `(i, j)`, `i < j`, i.e. `C(nzones,2)` paths, one path per OD
  pair. Panel sizes vary a lot:

  | panel | zones | paths | counted links | incidence links | rank(A_counted) |
  |---|---:|---:|---:|---:|---:|
  | D12_I405_S | 36 | 630 | 35 | 132 | 35 |
  | D12_I405_N | 43 | 903 | 81 | 152 | 42 |
  | D7_I210_E | 73 | 2628 | 294 | 396 | 72 |
  | D7_I210_W | 82 | 3321 | 81 | 398 | 81 |
  | D7_I10_E | 93 | 4278 | 137 | 373 | 92 |
  | D12_I5_N | 93 | 4278 | 257 | 343 | 92 |
  | D7_I405_N | 96 | 4560 | 350 | 516 | 95 |
  | D7_I405_S | 97 | 4656 | 96 | 496 | 96 |
  | D7_I10_W | 101 | 5050 | 100 | 427 | 100 |
  | D12_I5_S | 101 | 5050 | 100 | 370 | 100 |

  Total template rows per split = **35,354** (not 10 x 4,278); the merged upload
  file needs 70,708 Task-4 rows (35,354 `SYN_PM_VALIDATION` + 35,354 `SYN_PM_PRIVATE`).
* `path(i,j)` = the contiguous run of mainline links between zone boundaries i
  and j. Connector links (`RON-Zk`, `ROFF-Zk`) are in the incidence but are
  **never counted**; do not force them to zero.
* **The released counts are noiseless**: all counted links inside one
  zone-to-zone "segment" have *identical* counts (max within-segment std
  <= 1e-12), which is only possible if `c = A f*` was computed from the hidden
  path-flow vector itself. Consequences:
  * the truth satisfies `A f* = c` **exactly**, `f* >= 0`;
  * the identifiable part of `f*` is only the cut profile
    `C_k = sum_{i<=k<j} f_ij`; `rank(A_counted) = nzones-1` (42-100 rows)
    constrains 630-5050 unknowns. The null-space allocation is a modelling
    choice, not data;
  * **S_link is saturated**: any cut-consistent answer scores ~1.0. Official
    `score_task4.py` on the NNLS baseline: **0.999935** (validation),
    0.999917 (private), 0.999931 (train). So 0.25*S_link ~= 0.25 is free and
    the whole Task-4 margin is in S_od (0.45) + S_dev (0.15) + S_attr (0.15).
* **The weak prior is not a noisy version of the truth.** Its implied cut
  profile differs from `c` by a stable **21-27% relative L1** after the best
  global scale (per-segment cut-profile correlation only 0.08-0.60), and it has
  a systematic trend error (implied flow grows 5-9x along the corridor while the
  true cut profile is far flatter). `sum(A b) / sum(c)` ranges 0.69-2.35.
* Prior structure: a log-linear gravity model (origin + destination +
  separation effect) explains R^2 0.71-0.78 of `log b`, residual sd 0.49-0.61.
  The three splits' priors are correlated at **0.72-0.79** (log 0.73-0.75,
  median per-path ratio 0.92-1.06): shared synthetic structure + split noise.
  The prior's split-specific cut-profile deviations track the truth's
  split-specific cut deviations (corr 0.42-0.73) but with 1.5-4x too much
  amplitude.
* Task-1 / ramp observations are **unrelated** to the Task-4 OD: per-segment
  cut increments are +-200-1300 vph while released ramp net flows are only
  +-30-260 (sum|dC| = 9,331 vs sum|ramp net| = 1,931 on D12_I405_N/val). Do not
  use state/ramp data for Task 4.

## 2. Score decomposition and what the baseline implies

`S_ODME = 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr`.

With official baseline `S_ODME = 0.8359` and `S_link = 0.9999`:
`0.45*S_od + 0.15*S_dev + 0.15*S_attr = 0.586`. If `S_dev ~= S_attr ~= 0.9`
then **S_od(baseline) ~= 0.70**. Also, since `L1(f,f*) >= |sum f - sum f*|`, the
official baseline total implies the hidden total `T = sum f*` lies roughly in
`[0.73, 1.57] * sum f_baseline` per panel (because S_od cannot drop below 0.6353
unless S_dev/S_attr are perfect). Baseline totals (validation) are in
`work/task4/reports/candidate_diagnostics.csv` (mean 77,635 over panels).

## 3. Estimators implemented (`odme.py`, `cand.py`, `entropy.py`, `predict.py`)

| method | equations / construction |
|---|---|
| `baseline` | official: `min_f ||A f - c||^2 + 0.05 ||f - b||^2`, `f >= 0` (L-BFGS-B; matches `scipy.nnls` to 1e-10 on small panels) |
| `pool` | same, prior `b :=` mean of the train/validation/private priors |
| `autoscale` | same, prior scaled by the scalar that matches the cut-profile level: `s = (c . Ab) / (Ab . Ab)` |
| `scale:<s>` | same, prior = `s*b` (scale axis) |
| `ridge:<lam>` | same with a different lambda (lambda sweep) |
| `margcorr` | `f_ij = b_ij * exp(a_i + d_j)`, per-zone multiplicative marginals fitted so `A f = c` (damped Gauss-Newton on <=202 params) |
| `denoise_margcorr` | gravity log-linear fit of the prior (origin+dest+separation) then `margcorr` |
| `entropy` | **max-entropy / minimum-KL projection of the prior onto the cuts**: for a linear corridor a per-segment multiplier is a potential, so `f_ij = b_ij * exp(L_j - L_i)`, `L_0 = 0`; 42-100 potentials = number of independent cut constraints, solved by damped Gauss-Newton; `A f = c` to machine precision (`S_link = 1.0`), prior per-OD relative structure preserved as far as the counts allow. This is the canonical ODME estimator and the natural "gravity recompose + project" family. |

All deterministic, one panel at a time, low memory (< 300 MB), <= 90 s per split.

## 4. Delivered submissions (combined validation+private, 70,708 rows, key-complete)

Primary probes (Lead naming, all verified 0 missing / 0 extra keys, finite,
non-negative, no duplicate rows):

| file | method | local S_link (val, official scorer) | total flow (mean/panel) | sum|f-b|/sum b | one-line description | expected direction / risk |
|---|---|---:|---:|---:|---|---|
| `out/path_flow_validation.csv`, `..._private.csv` | baseline NNLS lam=0.05 | 0.999935 | 77,635 | 0.362 | official regularised ODME | reference; reproduces the organizer baseline |
| `out/cand_a_validation_plus_private.csv` | `pool` | 0.999894 | 78,115 | 0.584 | prior = mean of the 3 split priors, same lam=0.05 NNLS | small change; wins if the prior's split-specific wobble is noise, loses if it tracks month-specific demand (it partly does). Low risk. |
| `out/cand_b_validation_plus_private.csv` | `entropy` (own prior) | 1.000000 | 77,458 | 0.360 | min-KL projection of the own prior onto the cut constraints via 42-100 segment potentials | **REJECTED by public LB**: swapping only this file into the submission gave v3 = 0.76803 vs v2 = 0.79775 (Delta S_ODME = -0.1486, Delta S_od ~= -0.33). Do not ship. Kept for the record. |
| `out/cand_c_validation_plus_private.csv` | `ridge:100` | 0.966114 | ~baseline | smaller | strongly prior-weighted NNLS (prior-weight axis; S_link no longer saturated: -0.0085 S_ODME floor). Untested. |

Spares / next round (same format):

| file | method | S_link (val) | mean total | sum|f-b|/sum b | note |
|---|---|---:|---:|---:|---|
| `out/cand_b2_validation_plus_private.csv` | `entropy` (pooled prior) | 1.000000 | 78,967 | 0.498 | same rejected entropy family (do not probe) |
| `out/cand_c_lam10_validation_plus_private.csv` | `ridge:10` | 0.992281 | ~baseline | | gentler lambda point |
| `out/cand_x_autoscale_validation_plus_private.csv` | `autoscale` + NNLS | 0.999915 | 47,132 | 0.642 | the scale-axis probe: total = 0.61x baseline. Wins only if the hidden total is ~45-50k; high risk / high information |
| `out/cand_x_margcorr_validation_plus_private.csv` | `margcorr` | 1.000000 | 59,875 | 0.534 | marginal correction only |
| `out/cand_x_denoise_margcorr_validation_plus_private.csv` | `denoise_margcorr` | 1.000000 | 45,133 | 0.795 | gravity-smoothed prior + marginal correction; most aggressive (total -42%) |
| `out/cand_x_ent_blend35_validation_plus_private.csv` | `entropy` of `b^0.65 * m^0.35` (m = gravity fit of pooled prior) | 1.000000 | 71,652 | 0.445 | entropy projection with a mild gravity denoise: intermediate between cand_b and cand_x_denoise_margcorr, keeps a 0.92x-baseline total |
| `out/cand_x_ent_blend70_validation_plus_private.csv` | `entropy` of `b^0.3 * m^0.7` | 1.000000 | 65,094 | 0.593 | stronger gravity denoise (0.84x baseline total) |
| `out/final_recommended_validation_plus_private.csv` | = **baseline NNLS** (`path_flow_validation_plus_private.csv`) | 0.999935 | 77,635 | 0.362 | **FINAL**: the official NNLS baseline, see section 5 (LB rejection of the entropy family) and section 7 |

## 5. Public-LB probe result: the entropy hypothesis is REJECTED (decisive)

Single-variable Task-4 swap, everything else in the upload identical:

| submission | Task-4 file | public LB | Delta |
|---|---|---:|---:|
| v2 | NNLS baseline (`lam=0.05`, our reproduction) | **0.79775** | - |
| v3 | `cand_b` entropy projection of the own prior | **0.76803** | **-0.02972** |

Consequences:

* `Delta S_ODME = -0.02972 / 0.20 = -0.1486`.
* S_link is saturated in both files (1.0000 vs 0.9999) and the two files have the
  same total flow and the same `sum|f-b|/sum b` (0.360 vs 0.362), so S_dev and
  S_attr moved little; hence `Delta S_od ~= Delta S_ODME / 0.45 ~= -0.33`
  (the Lead's estimate: `Delta S_od ~= -0.33`).
* This is consistent with the size of the reallocation: the entropy solution is
  `L1(cand_b, baseline)/sum f_baseline = 0.307` per panel away from the
  baseline. If the baseline is already at `L1/sum f* ~= 0.30` (S_od ~= 0.70, the
  value implied by the organizer baseline's 0.8359) and that displacement adds
  to the error rather than cancelling it, the expected result is
  `S_od(cand_b) ~= 0.39`, `Delta S_od ~= -0.31` - exactly what the LB shows.
* Interpretation: the hidden path-flow truth's null-space allocation is **not**
  the multiplicative/potential form implied by the prior. It is much closer to
  the baseline's **Euclidean min-norm projection of the prior onto the cut
  constraints**. Equivalently: the prior's error is dominated by **additive
  (row-space / marginal) discrepancies**, while its null-space allocation is
  already close to the truth's, and the entropy projection distorted it.
* Everything in the same family is rejected without further quota: `cand_b2`
  (entropy + pooled prior), `cand_x_ent_blend35/70`, and the
  `margcorr`/`denoise_margcorr` marginal-correction family all reallocate the
  null space in the same multiplicative way.

## 6. Simulation study: which estimator would win, and the revised DGP belief

`sim_study.py` builds synthetic truths that all satisfy `A f* = c` exactly (the
same identified part as the real task) but differ in the null-space allocation
and in the total. Each truth is the entropy projection of a latent OD onto the
released cuts; latent OD = own prior / pooled prior / gravity-smoothed pooled
prior, optionally with independent lognormal per-OD noise. All estimators are
scored with the exact S_ODME formula. Mean S_ODME over 4 panels
(`reports/sim_summary_S_ODME.csv`, `sim_summary_S_od.csv`, ...):

| estimator | A: own prior +n0.3 | B: pooled prior (circular*) | C: pooled prior +n0.4 | D: gravity-smooth truth | E: gravity-smooth +n0.4 |
|---|---|---|---|---|---|
| baseline NNLS lam=0.05 | 0.801 | 0.742 | 0.703 | 0.480 | 0.473 |
| **cand_b entropy, own prior** | **0.883** | 0.817 | 0.759 | 0.568 | 0.540 |
| cand_b2 entropy, pooled prior | 0.803 | 1.000* | 0.853 | 0.635 | 0.602 |
| cand_a pooled-prior NNLS | 0.707 | 0.827 | 0.801 | 0.533 | 0.520 |
| cand_x_margcorr | 0.791 | 0.784 | 0.745 | 0.666 | 0.632 |
| cand_x_denoise_margcorr | 0.567 | 0.616 | 0.609 | 0.812 | 0.768 |
| cand_x_autoscale | 0.679 | 0.687 | 0.674 | 0.608 | 0.589 |

(*scenario B uses the pooled prior as the latent truth, so `ent_pool` reproduces
it by construction - not evidence.)
S_od follows the same ranking, e.g. scenario A baseline 0.629 vs cand_b 0.796;
scenario C baseline 0.473 vs cand_b 0.588. Simulation truth totals: 57.6-58.8 k
for the smooth scenarios vs 75.1-76.9 k for the prior-like scenarios.

**Simulation caveat (important, recorded after the LB result).** Every simulated
truth in this study was itself built by *entropy projection* of a latent OD onto
the cuts, which structurally favours the entropy family. The public-LB probe
(section 5) showed the real truth is instead close to the *additive* min-norm
projection, so the simulation's ordering did **not** transfer. Lesson for the
report: a DGP study for ODME must include additive (not only multiplicative)
prior error - here `f* ~= b + additive correction` - and the public LB was the
only reliable model-selection signal.

**Revised DGP belief (after the LB probe).** The evidence is
(a) released counts are *exactly* `A f*`; (b) the prior's implied cut profile is
21-27% relative-L1 off the truth after best scaling, with a *systematic
downstream trend* - so the prior's zone-level marginals are structurally wrong
(additive/row-space error in `b`); (c) the prior is gravity-like (log-linear
R^2 0.71-0.78, residual sd ~0.55) and the three splits' priors correlate
0.72-0.79; (d) the prior's *split-specific* cut-profile wobble tracks the
truth's split-specific wobble (corr 0.42-0.73) with 1.5-4x too much amplitude;
(e) the organizer baseline's 0.8359 S_ODME forces the hidden total to be close
to the baseline total (~78 k); (f) **new, decisive: the entropy reallocation of
the prior's null space lost 0.149 S_ODME**, while the same total scale and the
same `sum|f-b|` kept S_dev/S_attr roughly fixed - so the truth's null-space
allocation is close to the baseline's Euclidean min-norm projection of `b` onto
the cuts.

The revised belief is therefore: **the prior shares (most of) the truth's
null-space allocation, and its error is dominated by additive marginal/level
discrepancies.** The generator most consistent with everything is a synthetic
demand model whose truth and prior differ mainly in zone-level origin/attraction
draws (hence the wrong cut profile), with the within-cut OD allocation close
between the two. Under that belief the NNLS baseline is already the right
estimator: `lam=0.05` is so small relative to `A'A` (diagonal = path length)
that it effectively returns `f ~= b + A^T(AA^T)^+(c - Ab)`, i.e. the minimum
Euclidean displacement of the prior that reproduces the cuts exactly - exactly
the "additive correction" form the LB selected. Increasing `lam` only trades
S_link for no null-space gain; the entropy family reallocates and loses.

## 7. Final recommendation

**Ship the official NNLS baseline**: `out/path_flow_validation_plus_private.csv`
(= `out/final_recommended_validation_plus_private.csv`; identical to
`out/path_flow_validation.csv` + `out/path_flow_private.csv`, local
S_link = 0.999935 / 0.999917). It is in LB submission v2 (0.79775 overall) and no
tested alternative beats it.

Evidence: single-variable LB swap of `cand_b` (entropy projection, the only
family with a plausible structural argument) cost -0.02972 total =
-0.1486 S_ODME = -0.33 S_od (section 5). The remaining untested directions are
either rejected by the same argument (`cand_b2`, `cand_x_ent_blend*`,
`margcorr`, `denoise_margcorr`), lose S_link (`cand_c*`), or bet against the
baseline-implied total bracket (`cand_x_autoscale`, total 0.61x). A
sparser/concentrated solution was suggested by the Lead as the last speculative
idea; there is no positive evidence for it (the prior has no zeros, the
baseline's NNLS clips ~60% of the paths to zero already, and the LB says the
truth's null-space allocation is close to the prior's, i.e. dense), so it should
not be probed without a new argument. If a probe is ever free, the only
defensible candidates are *small* perturbations of the baseline (e.g. lambda in
[0.005, 0.5], `L1(cand, baseline)/sum f < 0.05`), not reallocations.

## 8. Risks / open questions

* **Hidden total flow is not identified by the counts.** The cut profile fixes
  `E_i - X_i` per zone but not the balanced exchange, so `sum f*` is only
  bounded below (~ minimum-total feasible: 8.6 k-27 k per panel) and above by
  nothing. The baseline-implied bracket (section 2) is the only handle, which is
  why all recommended candidates keep the baseline total scale.
* **Prior marginals are structurally wrong** (systematic downstream trend), but
  the LB probe shows the prior's *null-space* allocation is nevertheless close
  to the truth's; the NNLS baseline exploits exactly that.
* S_dev is a second, coarse copy of the same "distance from the prior" trade:
  candidates that move far from b (`cand_x_*`, `sum|f-b|/sum b` = 0.53-0.80 vs
  baseline 0.36) risk it if Dstar is small.
* S_attr (normalised destination attraction) is only weakly controllable: the
  cut constraints do not identify origin/destination marginals separately. The
  candidates' destination distributions differ by L1 0.10-0.41 vs baseline
  (`reports/candidate_allocation_compare.csv`), i.e. up to ~0.03 S_ODME, so it
  is a secondary, not a primary, lever.

## 9. Reproduce

```powershell
$env:PYTHONPATH='C:\Users\lenovo\Desktop\kaggle\libs'
# FINAL submission = official NNLS baseline (validation and private; concatenate
# for the merged upload). This is out/path_flow_validation_plus_private.csv and
# out/final_recommended_validation_plus_private.csv.
python work\task4\predict.py --split validation --output work\task4\out\path_flow_validation.csv
python work\task4\predict.py --split private    --output work\task4\out\path_flow_private.csv
# rejected candidate family for the record (entropy projection):
python work\task4\predict.py --split validation --method entropy \
    --output work\task4\out\cand_b_validation.csv
python work\task4\predict.py --split private    --method entropy \
    --output work\task4\out\cand_b_private.csv
python work\task4\build_entropy.py    # entropy candidates (cand_b, cand_b2, shrink)
python work\task4\batch.py            # NNLS families, val+private
python work\task4\compare.py          # allocation / attraction diagnostics
python work\task4\sim_study.py        # simulation study
python work\task4\sim_summary.py      # clean report tables (+ final_recommended copy)
python work\task4\build_lam_cand.py   # cand_c / cand_c_lam10 combined files
```

Reports: `reports/s_link_baseline_validation.csv`,
`s_link_cand_a_validation.csv`, `s_link_cand_b_validation.csv`,
`s_link_cand_c_validation.csv`, `candidate_diagnostics.csv`,
`entropy_diagnostics.csv`, `candidate_allocation_compare.csv`,
`structure_diag.csv`, `prior_cut_diag.csv`, `prior_gravity_diag.csv`,
`prior_cut_match.csv`, `cut_increment_diag.csv`, `simulation_study.csv`,
`sim_summary_S_ODME.csv`, `sim_summary_S_od.csv`, `sim_summary_S_dev.csv`,
`sim_summary_S_attr.csv`.
