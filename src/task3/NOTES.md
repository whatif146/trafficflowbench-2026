# Task 3 physics consistency — proxy evaluator, calibration and post-processing

Owner: `task3-physics`. Live directory: `work/task3/`. Nothing outside this
folder (and `work/submissions/state_adjusted_all.csv`) is modified.

Deliverables:

| file | what it is |
|---|---|
| `local_eval.py` | local train-split proxy of the official Task 3 evaluator (`score_task3.py` replicated field for field, frozen observation-derived boundary flux) |
| `adjust.py` | conservation/accumulation-aware post-processing of Task 1 predictions; streaming CLI for train/validation/private |
| `parity_check.py` | exact replication check of `local_eval.py` against the official repo scorer on a staged 9-day release |
| `calibrate_flux.py`, `test_calibrated.py` | boundary-flux calibration against the published official anchors |
| `measure_adjust.py`, `experiment3.py`, `experiment4.py`, `final_tradeoff.py` | adjustment trade-off measurements |
| `reports/` | all measurement CSVs |
| `work/submissions/state_adjusted_all.csv` | the delivered adjusted Task 1 state (validation + private, 6,735,795 rows, original columns/order) |

---

## 1. What the official Task 3 score really measures (calibration)

The organizer boundary flows are hidden, so `score_task3.py` run locally floors
`S_LWR` at 0 and is useless (docs/RUN_LOCAL.md: perfect answer 0.3301 locally vs
0.9598 with organizer flows). Two published anchors pin down the hidden flux:

* perfect answer: `S_physics = 0.9636` (10-panel validation), `0.9598` on D12_I5_N;
* naive historical baseline: `S_physics = 0.3549`, `0.3242` on D12_I5_N.

The FD term needs no boundary flux, so it can be computed exactly for the
observed field: **`S_FD(observed) = 0.9815` (D7_I10_E), `0.9858` (D7_I10_W),
`0.9828` (D7_I210_W)**. Therefore a perfect answer's `S_LWR` must be

```
S_LWR(perfect) = (3*0.9636 - 0.982)/2 ~ 0.954   (D12_I5_N: (3*0.9598-0.982)/2 ~ 0.949)
```

A smooth/simulated flux instead gives `E_LWR ~ 1.0` for *everybody* (including the
observed truth): measured on train, `rim = sum|dN-rhs| / sum|rhs| = 1.01-1.04`
for the exact observed field, and temporal smoothing of the simulated-like flux
makes it worse, not better (the observed accumulation change is
noise-dominated: `mean|dN_obs| ~ 0.9` vehicles vs `mean|dN_smoothed| ~ 0.3`).

Conclusion: the organizer flux is a **projection onto the released observations**
(the docstring says so literally: "projected onto the released observations, so a
participant who reproduces what was published satisfies the conservation identity
exactly"), with a residual of order 5% (the 0.95 not 1.00 ceiling). In first
order, with `N = (q/v)*L`:

```
target at t+1 only : residual ~= N_sub(t+1) - N_obs(t+1)   (level error)
target at t only   : residual ~= N_obs(t)   - N_sub(t)     (level error)
target at both ends: residual ~= dN_sub - dN_obs
no target endpoint : residual ~= -jitter (irreducible)
```

So S_LWR is, to first order, an **L1 penalty on the submission's accumulation
(density) error at the Task 1 target cells**. It is not a smoothness bonus.

## 2. The proxy (`local_eval.py`)

Pipeline (all vectorised numpy; one panel at a time, float32 where safe):

1. regime labels = frozen `regime_of_dates` blake2b hash of `panel|date` (verified
   to match the published `mask_regime` views exactly);
2. state = unmasked observations at non-target cells + candidate values at the
   published Task 1 template cells of that regime; official eligibility filter;
3. FD term per lane, including the empty-road guard, exactly as `score_panel`;
4. observable-transition filter (mainline neighbours present), forward
   `N(t+dt)-N(t)` on adjacent 5-min steps, `dt <= 5.1 min`, mode-A ramp-valid
   transitions (structural no-ramp links valid, unobserved ramps excluded, never
   treated as zero);
5. rhs from the **frozen** flux proxy built once from the unmasked observation
   layer (candidate never enters it): `rhs = observed dN` (exact projection),
   optionally with a calibration residual `flux_jitter` (default 0.06, see below);
6. per-regime -> per-panel -> macro mean, matching the official aggregation.

Validation of the replication: `parity_check.py` builds a staging release with 3
days per regime of one panel and runs the official `src/task3/score_task3.py`
against it. With `--flux fallback` (official topology fallback, computed from the
candidate like the official code) the two scorers agree **exactly** on
S_FD / S_LWR / S_physics and on the number of scored transitions for all three
regimes (staged D7_I10_E: 0.980515 / 0.981855 / 0.981684 S_FD, S_LWR 0,
S_physics 0.326838 / 0.327285 / 0.327228 in both).

Calibration to the anchors (`--flux projected --flux-window 1 --flux-jitter 0.06`):

| candidate (train) | panel | S_FD | S_LWR | S_physics | official reference |
|---|---|---|---|---|---|
| observed truth (Task 1 perfect) | D7_I10_E | .9815 | .9521 | **.9619** | perfect 0.9636 |
| observed truth | D7_I210_W | .9828 | .9524 | **.9625** | perfect 0.9636 |
| historical mean baseline | D7_I10_E | .9741 | .0000 | **.3247** | naive 0.3549 |
| historical mean baseline | D7_I210_W | .9791 | .0588 | **.3656** | naive 0.3549 |
| lead smoother `state_train_D7_I10_E.csv` | D7_I10_E | .9803 | .3996 | **.5931** | (this submission) |
| observed + noise (sd 5 km/h, 10% flow) | D7_I10_E | .9712 | .0117 | .3315 | — |
| observed lagged by one 5-min step | D7_I210_W | .9828 | .3837 | .5834 | — |

Both anchors are reproduced to ~0.01-0.03 and the ordering is exactly the
intuitive one: truth > good reconstruction > lagged/partially-right > historical
> noisier model. A residual level of 0.06 was chosen to match the .95 S_LWR of
the perfect-answer anchor; the ranking of methods is unchanged for 0.0-0.12
(measured, see `final_tradeoff.py`).

**Headroom analysis.** An honest model cannot predict the observation noise at
target cells, so the S_LWR ceiling for a perfect *smooth* predictor is finite:
candidate = temporal smoother (window 3) of the truth field gives
`S_LWR = .635` at `S_state = .947` (window 5: .568, window 9: .476). The lead
smoother reaches .400 at S_state .912. So ~0.23 S_LWR (~+0.077 S_physics,
~+0.012 S_total) is the realistic headroom above the current Task 1 answer; the
remaining gap to the perfect-answer 0.95 is unpredictable noise.

## 3. The post-processor (`adjust.py`)

`invert q = N*v/L`: keep the submitted speed, replace the submitted accumulation
at each target cell by a local robust estimate of the noiseless observed N built
only from released observations plus the candidate itself, then invert to a new
flow. Estimators: `median` (default; matches the L1 objective and robust to
spikes), `mean`, `linear` (equivalent to mean on symmetric windows), and
`conservation` (iterative min-norm symmetric projection of every scored 5-min
transition using the observation-derived rhs, observed cells kept fixed).
Corrections are shrunk by `strength` and capped by `max_rel_change` (0.35) and
`max_abs_change` (400 vph). Panels are processed one at a time; a panel that
cannot be loaded is written through unchanged (never dropped).

Grid search on D7_I10_E train with the lead smoother (proxy):

| method | window | strength | dS_state | dS_physics | 0.35 dS_state + 0.15 dS_physics |
|---|---|---|---|---|---|
| median | 3 | 0.75 | -0.0020 | +0.0416 | +0.00555 |
| **median** | **5** | **0.75** | **-0.0024** | **+0.0444** | **+0.00580** |
| median | 5 | 0.50 | -0.0013 | +0.0408 | +0.00565 |
| median | 7 | 0.75 | -0.0029 | +0.0437 | +0.00555 |
| mean | 3 | 0.75 | -0.0023 | +0.0388 | +0.00501 |
| conservation (Jacobi) | 1 | 0.5 / 20 it | -0.0024 | +0.0141 | +0.00127 |
| no adjustment | — | — | 0 | 0 | 0 |

Final delivered setting — **median, window 5, strength 0.75** — measured twice
(`final_tradeoff.py`, `reports/adjust_final_tradeoff.csv`):

| panel / candidate | base S_state -> adj | dS_state | base S_physics -> adj | dS_physics | trade-off |
|---|---|---|---|---|---|
| D7_I10_E, lead smoother (proxy residual .00/.06/.12) | .9120 -> .9095 | -0.0024 | .6083->.6526 / .5931->.6375 / .5770->.6216 | +.0443/+.0444/+.0446 | **+0.0058** |
| D7_I210_W, synthetic noisy candidate | .8908 -> .9002 | +0.0093 | .4869->.6430 | +.1531 | +0.0262 |

The gain is remarkably insensitive to the flux-residual calibration (S_physics
gain .0443-.0446 across residual 0-.12), so it is not an artefact of the 0.06
anchor calibration. On the noisier synthetic candidate the adjustment *improves*
Task 1 S_state as well, and the physics gain is much larger: the adjustment is
worth more the noisier the Task 1 model is.

Effect on score: +0.15*0.044 = **+0.0066 total** if validation behaves like the
train holdout (larger if the uploaded Task 1 predictions are noisier than the
train smoother; I expect the validation S_state of the shipped model to differ
slightly from the train holdout .912).

### Delivered file

```
work/submissions/state_adjusted_all.csv
```

* same rows (6,735,795), same order, same columns as `state_all.csv`
  (verified: panel/timestamp/station_id/link_id/mask_regime/speed_kmh identical,
  only `flow_vph` changed);
* validation block (10 panels, 3,451,376 rows) + private block (10 panels,
  3,284,419 rows), split auto-detected from the timestamp grids;
* 6,728,136 rows actually changed (7,659 rows the estimator left within 1e-9 of
  the input; all other rows are byte-identical in formatting);
* mean |Δflow| per panel 20-56 vph (`reports/adjust_coverage.csv`,
  `reports/adjust_diag_all.csv`);
* every submitted cell is a Task 1 target cell, so every one is eligible for the
  adjustment. S_LWR only scores transitions whose start cell has valid released
  ramp observations and whose mainline neighbours are present; all mainline
  links are present in the full submission (so all links are observable) and
  ~76% of ramp cells are valid by the frozen coverage mode table — i.e. the
  adjustment acts on exactly the submitted endpoints of those scoreable
  transitions. Mask leverage by regime: R1 1-(0.8)^2 = 36%, R2 51%, R3 75% of a
  regime's transitions contain at least one submitted endpoint.

Reproduce with:

```powershell
python work/task3/adjust.py --state work/submissions/state_all.csv `
  --output work/submissions/state_adjusted_all.csv `
  --method median --window 5 --strength 0.75 `
  --diagnostics work/task3/reports/adjust_diag_all.csv
```

## 4. How to use the proxy for model selection

```powershell
# score a Task 1 train submission on one or more cached panels
python work/task3/local_eval.py --state-submission work/submissions/state_train_D7_I10_E.csv `
  --panel D7_I10_E --panel D7_I10_W --flux projected --flux-window 1 --flux-jitter 0.06 `
  --output work/task3/reports/eval_candidate.csv
```

`--flux fallback` reproduces the official local (hidden-flux) behaviour; use it
only for parity checks. Use the calibrated mode (`projected`, window 1, jitter
0.06) to rank methods. Because S_physics is 15% of the total while S_state is
35%, the selection objective for a Task 1 model is

```
0.35 * S_state(train holdout) + 0.15 * S_physics(proxy)
```

Both terms are printable from `score_candidates.py`.

## 5. GBM Task 1 model (strongest candidate) — measured 2026-11-01 21:15

`work/task1/gbm/holdout_D7_I10_E.csv` (holdout-day rows, same protocol as
`work/task1/eval.py --holdout-days 60`), calibrated proxy, `reports/gbm_holdout_adjust.csv`:

| candidate | S_state | S_FD | S_LWR (residual 0 / .06 / .12) | S_physics (.06) |
|---|---|---|---|---|
| **GBM raw** | **.9391** | .9820 | .8244 / **.7873** / .7495 | **.8522** |
| GBM + adjust median w5 strength .50 | .9384 | .9820 | .8269 / .7899 / .7522 | .8539 |
| GBM + adjust median w5 strength .75 | .9377 | .9820 | .8255 / .7885 / .7508 | .8530 |
| GBM + adjust median w5 strength 1.0 | .9369 | .9819 | .8225 / .7855 / .7479 | .8510 |
| (lead smoother, all train days) | .9120 | .9803 | .4223 / .3996 / .3754 | .5931 |

Interpretation:

* the GBM's implied accumulation already agrees with the released-observation
  flux (S_LWR .79 vs .40 for the profile+residual-smoothing model), so switching
  Task 1 to the GBM is worth roughly
  `0.35*dS_state + 0.15*dS_physics = 0.35*0.027 + 0.15*0.259 ~ +0.048` total;
* the accumulation adjuster is **neutral to slightly negative** on the GBM
  (trade-off -0.0004 at strength .75, +0.00002 at .50) — it moves the inherited,
  already-correct level structure. **Do not adjust the GBM**; submit its raw
  state. The adjuster remains the right lever for noisier reconstructions
  (smoother: +0.0058 trade-off; synthetic noisy candidate: +0.026).

Note: the GBM validation/private CSVs produced at 21:02 were missing the whole
`D12_I405_S` panel (93,718 / 107,416 template rows) plus 8 rows of
`D12_I405_N`; that must be regenerated before merging (missing Task 1
predictions are scored as zero, and the Task 3 state would be incomplete).

## 6. Flux-integration adjuster (Lead's hypothesis) — tested, rejected

Idea: rhs(t) = dt*(q_in + r_on - q_out - r_off) from released observations, then
anchor the accumulation at observed (non-target) cells and integrate that rhs
forward/backward along each regime-day, so the submitted N change equals the
observation-derived flux by construction (`adjust.py --method flux`).

Measured on the GBM holdout (`reports/flux_adjust_test.csv`):

| variant | S_state | S_LWR (residual 0/.06/.12) | S_physics (.06) | dS_state | dS_physics |
|---|---|---|---|---|---|
| GBM raw | .9391 | .8244/.7873/.7495 | .8522 | 0 | 0 |
| flux strength .25 | .9026 | .6908/.6539/.6163 | .7631 | -0.0365 | -0.0891 |
| flux strength .50 | .8977 | .6400/.6031/.5657 | .7292 | -0.0415 | -0.1230 |
| flux strength 1.0 | .8936 | .5906/.5537/.5165 | .6963 | -0.0456 | -0.1559 |

It is strongly negative (mean |dq| = 248-317 vph near anchor gaps). Reason: the
*released topology + ramp* flux is NOT consistent with the released accumulation
field -- that is exactly why the official local fallback floors S_LWR at zero.
Measured on train: mean|rhs_topology| ~ 5.8 vehicles per 5-min transition while
mean|dN_obs| ~ 0.9, and the residual of observed dN against topology flux gives
`E_LWR ~ 1.0` even for the exact observed field. Integrating that flux injects
~5 vehicles of random-walk error per step into N, an order of magnitude more
than the ~0.3-0.9 vehicle signal the score measures. The organizer flux is a
projection onto the observations (section 1); on a scored split the observations
of the submitted cells are hidden, so the best available estimate of the target
N is a level estimate borrowed from the observed neighbours - which is exactly
the median adjuster (neutral on the already-good GBM), not flux integration.

Conclusion: **stop physics post-processing.** The raw GBM Task 1 state is the
best available physics answer (proxy S_physics ~0.85); the remaining gap to the
perfect-answer ceiling is unpredictable observation noise at the submitted cells.

## 7. Public-notebook density correction (LB 0.87141) — tested, rejected (2026-11-02)

The strong public notebook `work/research/lb-0-87141/cells.txt` proposes pulling
`k=q/v` towards `k_interp = interp_N/L` (linear N between nearest released
anchors) in log space, split between speed and flow by `alpha`, for hidden runs
`<= max_run`. Replicated exactly in `density_adjust.py` and tested honestly on
the shipped GBM holdout (4 panels) with the calibrated proxy; full report:
`reports/DENSITY_CORRECTION_REPORT.md`.

* all 80 grid configs have **negative** `est_LB = 0.35*dS_state +
  0.15*(dS_FD/3+2*dS_LWR/3)`; best (w=.25, alpha=.75, max_run=1):
  dS_state -0.0004, dS_FD 0, dS_LWR -0.0002, dS_physics -0.0001,
  **est_LB -0.00015** (bar +0.001);
* reason: on the holdout target cells the GBM density is 30-40% better than the
  anchor interpolation (MAE 1.07-1.38 vs 1.49-1.92; interpolation wins on only
  ~35% of cells), so the correction dilutes a better estimate. The notebook's
  CatBoost was worse than interpolation in density; ours is not;
* the median adjuster already rejected in section 3 has the same conclusion at
  equal strength (est_LB +0.00001 / -0.00037 for strength .5/.75), i.e. the
  notebook recipe is not different in outcome;
* deployment check on the shipped validation+private state (identical to
  `submission_v4.csv` state rows): exact S_FD +0.00003, ~50.7% of cells changed;
  the interp-consistent proxy shows dS_LWR +0.0365 but that is self-fulfilling
  (same interpolation target used both sides), not evidence.

No validation/private files were written. Verdict: **no adjustment; raw GBM
state is the final Task 3 answer.**

## 8. Caveats

* The proxy's boundary flux is calibrated, not known: absolute S_physics is
  accurate to ~±0.03 per panel and ranking is what is trusted.
* Flux residual calibration (0.06) was fitted to the two published perfect-answer
  anchors; S_LWR gains measured under residual 0 / 0.06 / 0.12 are
  indistinguishable (.0443 / .0444 / .0446), so the adjustment decision does not
  hinge on it.
* Adjuster hyper-parameters were tuned on one train panel (D7_I10_E, real
  smoother) and cross-checked on a second small panel with a synthetic noisy
  candidate; the optimum region (median, window 3-7, strength .5-.75) is broad.
* `speed_kmh` is deliberately left unchanged: changing flow is the direct lever
  on N and keeps the submitted speeds exactly as the Task 1 model produced them.
  An optimal speed/flow split would reduce the S_state cost by ~20% but adds FD
  risk and was not needed at the measured operating point.
* RAM: the evaluator and adjuster stream one panel at a time and hold peak
  ~0.5-1.3 GB (largest 352-link panels). `--flux fallback` and the full
  streaming path are validated.
* Formatting: `adjust.py` reads identifier columns and `speed_kmh` as text and
  writes untouched values back byte-identically, so only `flow_vph` of adjusted
  rows differs. (`state_adjusted_all.csv`, produced before this hardening, has
  ~1 ulp difference in the text rendering of a small fraction of *unchanged*
  speed values; the numeric values are identical. The GBM adjusted files were
  regenerated after the fix and are byte-identical except for adjusted flows.)
