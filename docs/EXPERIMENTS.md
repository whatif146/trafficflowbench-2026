# Experiment log — TrafficFlowBench (Lead keeps this file)

Competition: 2026 IEEE Big Data Cup Track 2 TrafficFlowBench (Kaggle).
Deadline: 2026-11-06 final submission; 2026-11-10 code+PDF.
Local reference scores (official numbers from repo docs): baseline total 0.5534,
perfect 0.9945. Public LB top on 2026-11-01: 0.93089.

## Submissions

| # | date/time (HKT) | ref | contents | public score | delta | note |
|---|---|---|---|---|---|---|
| 1 | 2026-11-01 20:45 | 57871841 | T1 profile+residual smoother (train CV S_state 0.912), T2 GBM v2 (Feb pseudo-CV 0.8024), T4 NNLS baseline (S_link 0.9999) | **0.79509** | — | first end-to-end; team name shows "Hsu Yooyoo" (needs website change) |
| 2 | 2026-11-01 21:01 | 57872136 | v1 + T2 GBM v3 (Feb pseudo-CV 0.8071) | **0.79775** | **+0.00266** | Task2 v3 confirmed; ΔS_queue = +0.0089 |
| 3 | 2026-11-01 21:3x | 57872437 | v2 + T4 cand_b (max-entropy projection; S_link 1.0) | **0.76803** | **-0.02972** | Task4 entropy family REJECTED (ΔS_od ≈ -0.33); revert to NNLS |
| 4 | 2026-11-01 21:5x | pending | v2 + Task1 GBM (holdout S_state 0.939; proxy ΔS_physics 0.593->0.852) | pending | expect ~+0.048 | the big one: isolates T1+physics |

Implied decomposition of v1 (using our local estimates): 0.35*~0.90 + 0.30*Q + 0.15*~0.59 + 0.20*~0.836 = 0.795
=> Q_validation ≈ 0.75 (our GBM v2 pseudo-CV was 0.8024 on Feb holdout).

## A/B plan (single variable changed per submission)

| probe | change vs previous | expected delta | status |
|---|---|---|---|
| 2 | Task2 GBM v3 instead of v2 (Feb CV 0.8071 vs 0.8024) | +0.002 | **done: +0.00266** |
| 3 | Task4 cand_b (entropy projection of prior onto cuts; S_link=1.0) | 0..+0.016 | uploading |
| 4 | Task1 GBM (holdout S_state 0.9391 vs 0.912; proxy S_physics 0.852 vs 0.593) | +0.01 T1 + 0.04 physics ≈ +0.048 | build after task1-gbm fixes missing D12_I405_S panel |
| 5 | Task4 cand_b2 (if cand_b wins) or skip | 0..+0.01 | depending on probe 3 |

## Key model results so far

- **Task 1 GBM** (task1-gbm): holdout D7_I10_E S_state **0.9391** (R1 .9414 / R2 .9427 /
  R3 .9333; speed RMSE 1.6-2.0 km/h, flow RMSE 30-31 vph/lane). Base = 0.9*nearest
  temporal interp + 0.1*spatial neighbours; LightGBM residual on 60 features from the
  masked view. THIS IS THE PRODUCTION T1 MODEL (pending the missing-panel fix).
- **Task 3 proxy on the GBM** (task3 calibrated proxy): S_physics **0.852**
  (S_FD .982, S_LWR .787) vs smoother 0.593 (S_LWR .400) and truth 0.962.
  => the T1 switch alone is worth ~+0.048 total. The task3 adjuster is neutral on
  the GBM (recommendation: submit raw GBM state).
- **Task 2 v3**: final (honest Feb pseudo-CV .8071 / onset .7572 / ongoing .8736;
  LB-confirmed +0.0027 over v2). No rule-side headroom left; ramps/hybrid/set-size
  experiments rejected honestly.
- **Task 4**: S_link saturated (0.99993; c = A f* noiseless, rank(A)=nzones-1).
  All headroom in S_od. Sim study: cand_b (entropy projection) best under the
  prior-like DGP scenarios (+0.06..0.08 S_ODME expected), probing now.

## LB context

Public LB is compressed high: of the first 100 teams, p50 = 0.887, p99 = 0.927,
top = 0.93089 (2026-11-01). Our v2 0.79775 is well below the top-100, so the
remaining levers (T1 GBM + physics, Task4 S_od) are essential.

## Local model scores (official formula, train only)

- Task 1 historical profile baseline: S_state 0.7046 on D7_I10_E last-60-day holdout
  (official scorer on full D7_I10_E train reproduces 0.912 for the smoother and
  matches our scorer to 1e-4).
- Task 1 profile + Jacobi residual smoothing (lam_t=1.0, lam_s_speed=0.10,
  lam_s_flow=0.30, 60 iters): holdout S_state **0.9092** (speed RMSE 3.06,
  flow RMSE ~33 vph/lane); official full-train D7_I10_E 0.9120.
- Measurement-noise floor estimate from second differences: sigma_speed ≈ 2.06 km/h,
  sigma_flow/lane ≈ 31 vph => ceiling S_state ≈ 0.93.
- Task 2: persistence 0.3161 / static v1 0.6530 / GBM v2 0.8024 / GBM v3 0.8071
  (honest Feb 2031 holdout pseudo-CV, official window->panel/condition->panel->family nesting).
- Task 3 proxy (task3 calibrated against official anchors): observed truth
  S_physics 0.9619; our smoother 0.5931 (S_FD 0.9803, S_LWR 0.3996); historical
  0.3247. All headroom in S_LWR (organizer flux ≈ observed dN + 6% residual).
- Task 4: NNLS baseline local S_link = 0.999935 (saturated). Counts are noiseless
  A f*, and only rank(A)=nzones-1=42..100 cut constraints exist for 630..5050
  paths, so all Task-4 headroom is S_od (0.45) + S_dev (0.15) + S_attr (0.15).

## Environment

Python 3.13.7, pip user site; pyarrow 26.0.0, pandas 3.0.2, numpy 2.2.6,
scipy 1.15.3, sklearn 1.9.1, lightgbm 4.6.0, kaggle 2.2.5, psutil 7.2.2.
`work/setup/verify_env.py` checks everything. Kaggle auth via
`~/.kaggle/access_token` (bearer). Upload workaround: www.googleapis.com is
unreachable from this network; `work/tools/upload_submit.py` uploads the signed
GCS resumable session through a reachable Google frontend (142.250.69.187) and
then creates the submission with the token.

## Day 2 (2026-11-02): Task-2 calibration probes and research

Public material review: 28 public notebooks + discussions for this competition
were downloaded into `work/research/` (see `work/tools/kaggle_research.py`).
The most useful public references:
- `cacinie/tfb-report` (0.86535 public, rank 224/660 on 2026-10-28; snapshot
  leader 0.92817, p10 0.88155, median 0.84810): detailed per-task report. Their
  LB-implied component scores are state 0.941 / queue 0.887 / physics 0.66 /
  ODME 0.854. Their queue recipe: onset threshold 0.55 (last step only),
  ongoing threshold 0.50, LGB+CatBoost blend; they measured that thresholds
  chosen on train windows mis-transfer to the validation month.
- `plarmuseau/trafficflowbench-eleven-anomalies-of-the-data`: generator
  anomalies measured from the release. Key for us: (1) observation noise is
  white multiplicative (flow sd 2.3%, speed 1.1-1.3 km/h) => Task-1 ceiling;
  (2) on-ramp flow drops to ~0.2x its TOD profile in the same step when the
  adjacent mainline jams ("ramp as a jam sensor"); using horizon ramp rows is a
  leak, legal history rows were worth +0.008 IoU in their test; (3) the
  conservation score is exactly 0 for truth under the released topology, so
  Task 3 is only knowable through the LB; (4) jam head fixed at a bottleneck,
  tail grows upstream, sharp in space/smeared in time => the Task-2 onset place
  is knowable, the moment is not.

Our public-LB probes today (v4 baseline 0.82448, all single-variable on Task 2):

| probe | change | public LB | delta | verdict |
|---|---|---|---|---|
| P1 | ongoing threshold 0.25 -> 0.40 | 0.82197 | -0.00251 | rejected: lower threshold better for our model |
| P2 | onset rule threshold 0.30 -> top-3 links | 0.80081 | -0.02367 | rejected: threshold adapts set size better than fixed top-k |
| P4 | task2-seq long-history v3+v5 blend (114/135 cells changed) | 0.82902 | +0.00454 | ACCEPTED: new best queue layer |
| P5 | ongoing threshold 0.25 -> 0.15 | 0.82431 | -0.00017 | neutral: shipped 0.25 is at the optimum |
| P6b | v5 blend + morphological closing of ongoing mask | 0.81537 | -0.01365 | rejected: closing adds false positives on the validation month |

Reproducibility verified end-to-end on 2026-11-02: `work/pipeline/run_all.py`
with isolated output dirs regenerated all 6,980,503 rows; vs the submitted v4
file: speed max|diff| 5.0e-4 km/h, flow 5.9e-3 vph (output rounding), queue_pred
identical (0 differing cells), path_flow max|diff| 6.0e-5 (float32 storage), i.e.
reproduction is exact to well under 1e-5 relative. `code/setup/verify_env.py`
passes all checks.

Research also suggested two Task-1 stacking ideas from public write-ups
(41-link position-window profile averaging, low-rank tensor completion of the
month) worth at most ~+0.003 state => ~+0.001 total; not pursued while Task 2
probes were closing the larger gap.


## Day-2 evening wrap (2026-11-02)

- Task-1 low-rank / day-similarity tensor completion stacked on the GBM:
  holdout mean S_state 0.9367 -> 0.9373 (+0.0006, bar was +0.003) => NOT shipped.
  Task 1 is closed at the noise-dominated ceiling.
- Task-2 ramp jam-sensor (on-ramp flow / TOD profile) ongoing model: Feb 0.8669 vs
  blend 0.8679; June v3 0.8543 vs v5r 0.8452 => mixed/worse, NOT shipped.
- CatBoost onset-blend experiment blocked by a stalled catboost wheel download;
  LGB-only blend stays.
- Pre-built Task-2 threshold-variant submissions for tomorrow's probes:
  submission_q_on040/q_on045/q_on050/q_on055/q_on020/q_onfb1/q_ong035 (.zip).
  Validation onset set size: th 0.30 -> 3.8 links/window; 0.40 -> 3.38;
  0.45 -> 3.18; 0.50 -> 2.98; 0.55 -> 2.85; train-window truth ~2.75-3.5.
- Current best: P4 ref 57906197 public 0.82902. Package + report updated to it.
- Next (2026-11-03): probe onset thresholds (quota resets 08:00 HKT), pick best,
  build combined candidate, then freeze for the 2026-11-06 deadline.

## Day-2 late: offline onset-threshold analysis (released train windows)

Truth for the 80 released train windows is available from the train plain layer;
the shipped blend boosters are refit on all train windows, so this is an
in-sample but *window-population-correct* check of the onset decision rule
(the released window population is what validation uses, unlike pseudo-windows):

| onset threshold | 0.30 | 0.40 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| June onset IoU | 0.734 | 0.833 | 0.900 | 0.910 | 0.940 | 0.925 | 0.890 |

For validation we cannot see truth, but the predicted onset set size at th 0.30
is 3.80 links/window with 6/40 windows predicting NOTHING (IoU 0), while
th 0.55 + top-1 p-fallback gives 2.90 links/window and 0 empty windows
(th 0.60 + fallback: 2.58; truth estimate ~2.75-3.5). On June, the p-fallback
hit the truth link 1/1 at th 0.55/0.60, while climatology-based fallbacks missed.
=> tomorrow's probes: q_on055fb1, q_on060fb1, q_on050fb1, q_on055 (no fallback,
to isolate), then a refinement. New candidates built for all of these.

## Day-2 late #2: rule-family comparison on released windows (in-sample)

On the 40 released train onset windows (truth available), fixed probability
thresholds dominate any fixed top-k rule:
  th0.60 IoU 0.9404 (mean set 2.60) | th0.55 0.9102 (2.75) | th0.50 0.8996 (3.05)
  top-k: top1 0.498, top2 0.620, top3 0.618, top4 0.576, top5 0.498,
  per-panel-k (k = mean truth size per panel) 0.721.
=> threshold rules adapt the set size per window; top-k does not. This also
explains the P2 LB result (onset top-3 lost -0.024). Per-panel June truth mean
set size varies 1.2-4.0 links, so any fixed k is wrong for most panels.
Combined with the empty-window issue (6/40 at th 0.30 on validation), the
shipped probe family is threshold 0.50-0.60 + top-1 probability fallback, which
is exactly what is pre-built and queued for tomorrow's LB probes.

## Round-9 CORRECTION: honest CV of onset threshold+fallback rules

The earlier "released-window threshold analysis" in this log was IN-SAMPLE (the
boosters had been refit on all train windows including the June windows being
scored). Refit with every June origin excluded (same protocol as run_cv.py) and
scored on the untouched February block (B) and released June windows (A):

| rule | B_feb | A_june | mean |
|---|---:|---:|---:|
| th0.20 | 0.7665 | 0.7963 | 0.7814 |
| th0.25 | 0.7727 | 0.8087 | 0.7907 |
| th0.25 + top1 fallback | 0.7739 | 0.8087 | 0.7913 |
| th0.30 (shipped) | 0.7664 | 0.8050 | 0.7857 |
| th0.30 + top1 fallback | 0.7629 | 0.8050 | 0.7840 |
| th0.40 | 0.7482 | 0.8010 | 0.7746 |
| th0.50 | 0.6921 | 0.7789 | 0.7355 |
| th0.60 | 0.6389 | 0.6664 | 0.6527 |

=> thresholds > 0.30 degrade monotonically on both honest protocols; 0.25 is the
best point and top-1 fallback is neutral-to-slightly-positive there (it targets
the 6/40 empty onset windows seen on validation). New candidates built:
q_on025, q_on025fb1, q_on020fb1 (plus the earlier fallback/threshold family, now
mostly informative rather than likely winners). Tomorrow's probe order is
updated accordingly (q_on025fb1, q_on025, q_on020fb1 first; high-threshold
candidates only as a check).

## Round-12: honest CV grid over the v3/v5 onset blend weight

The shipped blend used w_on = 0.5 (onset), selected only on the February pseudo-CV.
A two-protocol honest grid (refit excluding June; B = untouched Feb block,
A = released June windows) gives, for the onset rule:

| w_on | th | fallback | B_feb | A_june | mean |
|---:|---:|---|---:|---:|---:|
| 0.3 | 0.30 | no | 0.7690 | 0.8162 | **0.7926** |
| 0.3 | 0.25 | top-1 | 0.7741 | 0.8102 | 0.7921 |
| 0.5 | 0.25 | top-1 | 0.7739 | 0.8087 | 0.7913 |
| 0.4 | 0.25 | top-1 | 0.7728 | 0.8087 | 0.7908 |
| 0.5 | 0.30 | no | 0.7664 | 0.8050 | 0.7857 |
| 0.6 | 0.25 | top-1 | 0.7661 | 0.8046 | 0.7853 |
| 0.8 | 0.30 | no | 0.7548 | 0.7831 | 0.7690 |

w_on = 0.3 improves BOTH protocols over the shipped 0.5 (B +0.003, A +0.011),
so the onset half of the blend should shift toward the long-history model.
Full grid: work/task2_seq/out/blend_weight_cv.csv; candidates generated with
ship_w30.py (q_w30, q_w30_on025fb1, q_w30_on030fb1 submissions pre-built) and
moved to the front of tomorrow's probe plan.

## Round-13: honest CV grid over the ONGOING blend weight

Same two-protocol protocol (models refit excluding June, stride-2 training to
bound memory; relative ordering only). Best rows:

| w_ong | th | B_feb | A_june | mean |
|---:|---:|---:|---:|---:|
| 0.3 | 0.20 | 0.8634 | 0.8481 | 0.8558 |
| 0.4 | 0.20 | 0.8626 | 0.8489 | 0.8558 |
| 0.5 | 0.20 | 0.8612 | 0.8504 | 0.8558 |
| 0.6 (shipped) | 0.25 | 0.8605 | 0.8442 | 0.8523 |
| 0.6 | 0.20 | 0.8596 | 0.8498 | 0.8547 |
| 0.8 | 0.20 | 0.8546 | 0.8493 | 0.8519 |

So the ongoing half prefers a slightly lower weight on v3 (0.3-0.5) and a lower
threshold (0.20); the shipped .6/.25 is ~0.0035 IoU behind. Combined with the
onset weight finding (w_on 0.3) this gives a full "w30ong40" candidate family.
Generated by ship_w30.py; submissions q_w30ong40 and q_w30ong40_on025fb1 are
pre-built and second in tomorrow's probe plan.

## 2026-11-03: multi-month pseudo-validation simulation of the Task-2 rule grid

Beyond the single Feb/June protocols, the queue decision-rule grid was fitted at
three cut-offs (fit on all selector-valid pseudo windows before the block) and
scored on the held-out blocks Dec-2030, Jan-2031, Feb-2031
(work/task2_seq/sim_multimonth.py, out/sim_multimonth_rules.csv).

Estimated total-score deltas vs the shipped P4 rule (0.30*0.5*(d_on+d_ong)):

| candidate | Dec-30 | Jan-31 | Feb-31 | mean |
|---|---:|---:|---:|---:|
| q_w30ong40 (onset w.3; ongoing w.4/th.20) | +0.00052 | +0.00031 | +0.00057 | +0.00047 |
| q_on025 (onset th .25) | +0.00060 | +0.00002 | +0.00039 | +0.00034 |
| q_w30ong40_on025fb1 | +0.00067 | -0.00051 | +0.00049 | +0.00022 |
| q_w30 (onset weight only) | -0.00008 | 0.00000 | +0.00038 | +0.00010 |
| q_onfb1 (top-1 fallback) | +0.00007 | -0.00102 | 0.00000 | -0.00032 |

Findings: the combined weight candidate is the only one positive in ALL three
blocks (magnitude small, +0.0005). Onset th .25 is also positive everywhere. The
top-k fallback HURTS on the pseudo months (Jan block), most likely because some
pseudo-window truth sets are empty after eligibility filtering while a fallback
always predicts links -> in the real released window populations the fallback
never triggered in the honest June protocol either, so it stays a diagnostic
only. Note the blocks' onset IoU levels differ a lot (Dec/Jan ~0.43, Feb ~0.77),
so the main value is the sign consistency, not the absolute level.

Also reviewed two additional public notebooks: LB 0.87141 (Task-3 density
post-processing via N interpolation between observed anchors, log-space w/alpha
correction, est-LB selection) and LB 0.87 Trees+U-Net+CNN (condition-specific
models, onset top-2 fallback, selector-matched training windows, 30% U-Net state
blend whose local RMSE is worse but LB higher because Task-3 likes smoothness).
The Task-3 density correction is being tested on our predictions by
task3-physics; the queue ideas were already partly adopted (condition split,
selector-based training) and the fallback is now deprioritised by the simulation.

## 2026-11-03: is a physical queue-propagation prior justified? No.

On the 40 released ongoing train windows (truth available), the queued-cell sets
churn strongly between T and T+30: windows growing 17 / shrinking 19 / constant 4;
mean queued cells 27.7 -> 25.1; mean cells gained 28.9 vs lost 31.5
(work/task2_seq/analyze_queue_physics.py). So the "queue tail grows upstream
monotonically / head fixed" picture does NOT hold for the scored binary masks --
the masks are thresholded on noisy speed, so links toggle. This is why our
morphological-closing probe lost 0.0137 on the LB, and it means no simple
shockwave-based binary post-processing is justified; the statistically optimal
decision remains the calibrated per-link probability threshold (plus, possibly,
a fallback for empty windows). Any further Task-2 gain must come from a better
posterior (features/model), not from forcing kinematic-wave geometry onto masks.

## 2026-11-03: public-notebook Task-3 density correction tested -> NEGATIVE

Replicated the LB-0.87141 notebook's density post-processing exactly (interpolate
N=(q/v)L between nearest observed anchors on the same link, hidden-run length run,
log-space correction d=w(log k_i - log k_hat), speed/flow split alpha, run<=max_run;
grid w 0.25-1.0, alpha 0-1, max_run 1-9999 = 80 configs) and evaluated it with the
calibrated Task-3 proxy on the shipped GBM holdout (4 panels), selecting with
est = 0.35*dS_state + 0.15*(dS_FD/3 + 2*dS_LWR/3), bar +0.001.

All 80 configs negative; best est_LB = -0.00015 (w .25 / alpha .75 / run 1:
dS_state -0.00040, dS_LWR -0.00016, dS_physics -0.00010; 291,888 cells changed).
Aggressive correction is much worse (est -0.026). Reason: on holdout target cells
our GBM density is 30-40% more accurate than anchor interpolation (MAE 1.07-1.38 vs
1.49-1.92; interpolation wins only ~35% of cells), so pulling q/v toward the
interpolation dilutes a better estimate. Same conclusion as the median adjuster
(est +0.00001 at strength .5, -0.00037 at .75). Deployment check on validation:
S_FD +0.00003, ~50.7% cells changed, but the interp-consistent dS_LWR +0.0365 is
self-fulfilling and not evidence.

=> Task 3 post-processing is exhausted; ship the raw GBM state. Full report:
work/task3/reports/DENSITY_CORRECTION_REPORT.md.

## 2026-11-03 evening: Task-2 LB probes (P4 = 0.82902 baseline)

| probe | change vs P4 | public LB | delta | note |
|---|---|---:|---:|---|
| q_w30ong40_on025 | onset w .5->.3 + th .30->.25; ongoing w .6->.4 + th .25->.20 | 0.82868 | -0.00034 | tie/negative; the th .25 part costs ~0.001 |
| q_w30ong40 | onset w .5->.3 (th .30); ongoing w .6->.4 / th .20 | **0.82984** | **+0.00082** | new public best but below the pre-committed +0.0015 win bar -> treat as tie until paired probes finish |
| q_on025 | onset th .25 only (shipped weights) | pending | | paired with #2: isolates the threshold effect |
| q_onfb1 | onset th .30 + top-1 fallback | pending | | targets 6/40 empty validation onset windows; simulation suggested negative |

Reading: the weight recalibration is slightly positive on the validation month while
moving the onset threshold down from .30 to .25 is slightly negative -- i.e. the
honest-CV threshold gain did not transfer, but the weight gain did (small). The
multi-month simulation had predicted +0.0005 for the weights change, consistent
with the observed +0.0008.

## 2026-11-03: cross-task "density-space target" Task-1 model -> NEGATIVE

Hypothesis: since Task 3 scores an L1 penalty on N=(q/v)L, training a model
directly on density k=q/v and reconstructing q = k_hat * v_hat could improve the
combined score even if Task-1 RMSE moves slightly. Tested on the same 4-panel
holdout protocol as the shipped v1 GBM (same features/config), variants:
a = shipped flow, b = density reconstruction, c = untuned 0.5 blend.

| panel | est_total a | b | c | Δ(b-a) |
|---|---:|---:|---:|---:|
| D7_I10_E | 0.4565 | 0.4552 | 0.4564 | -0.0014 |
| D12_I5_N | 0.4644 | 0.4605 | 0.4635 | -0.0039 |
| D7_I405_S | 0.4549 | 0.4528 | 0.4545 | -0.0021 |
| D7_I210_W | 0.4505 | 0.4463 | 0.4496 | -0.0042 |
| mean | 0.4566 | 0.4537 | 0.4560 | **-0.0029** |

Strict loss: flow RMSE/lane rises 14-57% AND S_LWR falls on every panel; the
density model's cell-level L1 error is not better than the baseline's derived
q/v, and q=k*v re-injects the speed model's ~2% error into flow, which is
already at the 2.3% multiplicative noise floor. Bar was +0.001 => not shipped.
This closes the last cross-task idea for Task 1; both channels are at their
independent noise floors and the q=v*k identity cannot beat separate estimation.

## 2026-11-03: probe 2 changes the leaderboard order

q_w30ong40 (onset w_on .3, ongoing w_ong .4/th .20; 203 cells changed) scored
**0.82984** vs P4 0.82902 = +0.00082, while the same candidate plus onset th .25
(q_w30ong40_on025) scored 0.82868 (-0.00034). So on the real validation month the
*weight recalibration* transfers (small positive) but the *lower onset threshold*
does not (about -0.001 relative). Current best candidate: q_w30ong40 (ref 57946739);
P4 remains the diverse fallback. Refinement probes queued: ongoing threshold .25
vs .20 at the same weights, w_ong .3, w_on .4.

## 2026-11-03 (evening): the empty-window fallback is the largest gain since P4

Full single-variable probe record (P4 baseline 0.82902):

| probe | change | public LB | delta |
|---|---|---:|---:|
| q_w30ong40_on025 | weights + onset th .25 | 0.82868 | -0.00034 |
| q_w30ong40 | onset w .3, ongoing w .4/th .20 | 0.82984 | +0.00082 |
| q_on025 | onset th .25 | 0.82852 | -0.00050 |
| **q_onfb1** | **onset th .30 + top-1 fallback on empty windows** | **0.83652** | **+0.00750** |
| **q_w30ong40fb1** | **weights + fallback** | **0.83734** | **+0.00832** |

Interpretation: our 40 validation onset windows include 6 whose model probabilities
are all below 0.30, so the shipped rule predicts NOTHING there -> guaranteed IoU 0
for those windows (they are 15% of onset windows, half of the queue score). The
fallback (if a window predicts no queued link, mark the single most likely link)
recovers them, and the official spec guarantees every onset window contains a
queue, so this is a structural fix, not public-month fitting. The weight
recalibration adds its +0.0008 on top (additive).

IMPORTANT METHODOLOGICAL CORRECTION: our honest pseudo-window CV had shown the
fallback as neutral/negative (B 0.7629 vs 0.7664) and the multi-month simulation
as negative (-0.0003). That was an artefact: in pseudo-window truth, eligibility
filtering can empty a window's truth set, and then an empty prediction scores
IoU 1 while any fallback scores 0. The *scored* window populations (validation /
private, and the 80 released train windows) always contain a queue, so those
protocols systematically mis-grade the fallback. For decision rules, the
released-window protocols and the LB are the valid evidence; pseudo-window truth
must exclude empty-truth windows before judging fallback-style rules.

New best candidate: q_w30ong40fb1 (ref 57947511, public 0.83734). Next probes
(tomorrow): fallback depth (top-2, q_onfb2), minimum-set size 2 (q_min2), and the
onset threshold with the fallback in place (q_on040fb1, q_on025fb1), then the
combined winner.

## 2026-11-03 (late): two scoring-spec facts that change Task 2

Reading repo/trafficflowbench-public-main/docs/SCORING_SPEC.md carefully:

1. "The ground truth is taken from the **underlying state** rather than the noisy
   observation, so measurement error cannot flip the indicator back and forth at
   the threshold." Our Task-2 training labels and all local CV truths were built
   by thresholding the *noisy observed* speed -- i.e. exactly the flips the
   official truth excludes. We are therefore training against a noisier target
   than the scorer uses (and our CV numbers measure that noisier target). Action:
   rebuild pseudo-window labels and evaluation truths from a NaN-aware temporal
   Gaussian smoothing of the observations (sigma 2-3 steps) and retrain the same
   v3+v5 pipeline on them (delegated to task2-seq, strict two-protocol ship bar).
2. "Both require a queue somewhere in the horizon, so an empty-empty window
   cannot be drawn as a free mark." This applies to **ongoing** windows as well as
   onset. Our best queue layer still had 1/40 validation ongoing windows with an
   all-empty mask (guaranteed IoU 0); marking its single most likely cell weakly
   dominates (if wrong the window still scores 0, if right it scores >0). Added as
   config `fb_ong` and built candidates q_w30ong40fb1_all / q_onfb1_all
   (validation: exactly 1 cell changed, private: 0 cells; no downside).

New candidate: q_w30ong40fb1_all (best + ongoing-window fallback, weakly dominant
over q_w30ong40fb1 = 0.83734). Pending: denoised-label retrain (v7) from task2-seq.

## 2026-11-03: v7 = retraining on heavily denoised labels (sigma 2-3) -> mixed, not shipped

Built denoised labels (NaN-aware temporal Gaussian sigma 2-3 on the observed
speed, then <= 0.60*free_speed) for pseudo-window training labels and for the
evaluation truths, retrained the same v3/v5 pipeline, and evaluated with the two
honest protocols. Table from task2-seq (v7_cv.log):

| protocol | model | vs denoised truth | vs noisy truth |
|---|---|---|---|
| B (Feb block) | shipped-style noisy-label blend | 0.6236 (on .3962 / ong .8581) | 0.8155 (on .7661 / ong .8792) |
| B | v7 denoised-label model | 0.6655 (on .4817 / ong .8700) | 0.7178 (on .6389 / ong .7966) |
| A (June) | noisy-label blend | 0.6077 (on .3850 / ong .8305) | 0.8347 (on .8162 / ong .8532) |
| A | v7 denoised-label model | 0.6446 (on .4992 / ong .7900) | 0.7178 (on .6389 / ong .7966) |

Reading: (i) the label regime matters enormously -- the same predictions score
0.77-0.82 onset IoU against noisy truth but only 0.39 against sigma-2-3 denoised
truth; (ii) v7 improves its own regime (+0.037-0.042) but regresses ongoing on
protocol A and is much worse against noisy truth; (iii) the LB-implied queue
score (~0.7-0.8 depending on the unknown physics term) is incompatible with the
official truth being sigma-2-3 denoised, so the aggressive smoother is almost
certainly not the right model of the "underlying state".

Decision: do not ship v7. The principled follow-up (delegated) trains on SOFT
Bayesian labels P(latent queued | observed speed) = Phi((v_cut - v_obs)/sigma_v),
which encodes the measurement-error model directly instead of guessing a
smoothing kernel; it is evaluated under a noisy and a MILD (median-3) proxy truth
and only an LB probe decides whether the latent-truth regime helps.

## 2026-11-03: v9 = soft Bayesian labels P(latent queued | observed speed) -> NEGATIVE

Motivation: the spec says Task-2 truth is the *underlying state*, so hard labels
from noisy observations are noisy. Instead of guessing a smoothing kernel (v7,
which over-smoothed), encode the measurement-error model directly: target =
Phi((v_cut - v_obs)/sigma_v), i.e. the Bayes probability that the latent speed is
below the cut given the observation (sigma_v estimated per panel from robust
second differences, sigma ~0.7-2.0 depending on the estimator). Same features,
configs, stride-1 rows; evaluated under two hard proxy truths (noisy labels and a
MILD median-3/sigma-1 denoise) on both honest protocols:

| protocol | proxy | current noisy-label blend | v9 soft-label | delta |
|---|---|---|---|---|
| B (Feb) | noisy | 0.8155 | 0.7705 | -0.0449 |
| B | mild-s1 | 0.7873 | 0.7406 | -0.0466 |
| A (June) | noisy | 0.8347 | 0.7724 | -0.0623 |
| A | mild-s1 | 0.8073 | 0.7541 | -0.0532 |

Soft labels are worse under BOTH proxies and BOTH protocols: near-boundary soft
targets (many at ~0.5) weaken the sharp decision boundary that IoU needs, and the
sigma estimate/AUC mismatch dominates any label-denoising benefit. Combined with
v7 (heavy denoise: better on its own denoised truth, much worse on noisy truth),
the conclusion is that the official truth is close to the noisy-label population
(otherwise our LB-implied queue score of ~0.7-0.8 would be inconsistent), and no
label-regime trick ships. No v7/v9 files are used.

The remaining Task-2 axis is the structural fallback family, probed on 2026-11-04:
threshold up/down with fallback, top-2 fallback depth, minimum-set 2, and the
ongoing threshold on top of the winning family.

## 2026-11-03 (late): final-artifact reproducibility verification

Verified the artifacts that the organizers will check, without a full 30-min
re-run (the end-to-end pipeline reproduction was already verified on 2026-11-02:
all 6,980,503 rows matched the upload with max |diff| 5e-4 km/h / 5.9e-3 vph,
i.e. output-rounding level):

- Task-2 final queue layer is byte-identical on re-run:
  `queue_w30ong40fb1_all_validation.csv` md5 e9240ff4f466093a128c02ba614877a1,
  `..._private.csv` md5 1c0c5e22c9f89aa9ddbe816c49b73b43 (same after re-running
  ship_w30.py w30ong40fb1_all).
- Queue patching is byte-identical: rebuilding the 6,980,503-row upload from the
  base gives submission CSV md5 c994b07558d68622bfc88fde722a03a1, identical to
  the shipped candidate (submission_q_w30ong40fb1_all.csv).
- The package's final submission zip is md5 e950a4d2b60adabe15f1ca358b0a5c6c
  (identical to work/submissions/submission_q_w30ong40fb1_all.zip).

Probe-plan tweak for 2026-11-04: the fifth slot is used to UPLOAD the true final
candidate q_w30ong40fb1_all (one cell better than the uploaded best 0.83734 and
weakly dominant by the spec); q_on025fb1 stays as a spare for 2026-11-05. A
candidate can only be selected if it has been uploaded.


## 2026-11-03 (late): inverse-analysis calibration probes prepared (user approved)

The scoring mechanism was re-verified from the official source. Confirmed:
- score_overall weights 0.35/0.30/0.15/0.20;
- Task-1 truth = the *unmasked observation layer* (not the latent state);
- Task-2 truth = organizer queue_true (latent-state threshold) restricted to
  is_score_eligible cells, aggregated window -> (panel,condition) mean -> panel
  mean -> family mean -> overall mean; empty union -> IoU 1, but the spec
  guarantees both conditions contain a queue somewhere in the horizon;
- Task-3 derives density as q/v and uses organizer boundary flux (perfect answer
  0.9636, i.e. finite noise ceiling);
- Task-4 S_ODME = 0.45*S_od + 0.25*S_link + 0.15*S_dev + 0.15*S_attr; only S_link
  is computable locally.

Absolute component scores are hidden, so two calibration submissions are built
from the final candidate (submission_q_w30ong40fb1_all) and queued for 2026-11-04:
- submission_cal_zeroqueue: all 174,000 queue rows = 0 -> every scored window
  has a queue (spec) -> S_queue = 0 -> Q = (0.83734 - total_zeroqueue)/0.30.
- submission_cal_zeroodme: all 70,708 path_flow rows = 0 -> S_link = 0, and
  S_od/S_dev/S_attr are expected 0 for zero flows -> T4 ~ (0.83734 -
  total_zeroodme)/0.20 (uncertainty to be bounded by the subagent reading
  src/task4/score_task4.py).
Both files pass check_lean (6,980,503 rows; binary queue; finite non-negative
path_flow). Residual 0.35*S1+0.15*P = 0.83734 - 0.30Q - 0.20T4, split using the
single-task v2->v4 LB delta (0.02673) and the train-proxy physics/state ratio.


## 2026-11-03 (late): scorer inverse analysis corrections (organizer-intent)

1. Task-4 has an absolute documented anchor: docs/BASELINES.md (validation split,
   ten corridors) publishes the reference NNLS baseline (released prior, lambda
   0.05) at S_ODME = 0.8359. Our Task-4 file is that baseline, so T4_best is
   ANCHORED at 0.8359 and the planned zero-ODME calibration probe was dropped
   (all-zero S_ODME is not 0: S_od=0 and S_link=0 exactly, but S_dev(0)~0.17 and
   S_attr(0) in [0,0.75], so the probe would only give a lower bound).
2. Zero-queue inversion formula corrected: S_queue(all-zero) = E/80 where E is
   the number of scored windows whose *eligible* truth set is empty (empty union
   scores 1). Measured proxy on the 80 released train windows: E=0 (one onset
   window's proxy truth is empty even without eligibility filtering, so a
   conservative E<=2, i.e. <= +0.025 on Q = <= +0.0075 total). Every scored
   window has 360-1643 eligible horizon cells, so the 'no eligible cells' guard
   never fires. Hence Q = (0.83734 - total_zeroqueue)/0.30 + E/80.
3. No eligibility/union/physics exploit exists: Task-3 numerator is linear in the
   target densities and both denominators are submission-independent; S_LWR wants
   the conditional median of N=(q/v)L while Task-1 wants the conditional mean of
   q,v -- identical to O(sigma^2), which is why density post-processing is
   neutral. S_FD ~0.98 is effectively fixed.
4. Real distribution-shift lever found: on validation/private the INPUT masked
   layer carries extra unscored 18-step horizon blanks (T+5..T+90 after every
   Task-2 window origin; +11,675 (D7_I10_E val), +25,295 (D12_I5_N val),
   +12,653/+25,655 (private) cells, 3.5-4.7% of the Task-1 template, only on the
   6-8 window days per month). Our T1 LightGBM only simulates R1/R2/R3 target
   masking, so the scored splits give it a different missingness pattern exactly
   where gap-adjacent RMSE is worst (2.05 vs 1.77 km/h) and where the Task-3
   S_LWR residual lives. task1-gbm is retraining with horizon-shaped blanks;
   estimated +0.001-0.004 total, LB-probe required.


## 2026-11-03 (late): inverse-scoring results (task-2 truth hypothesis + absolute Q)

1. Best-fitting Task-2 truth hypothesis from the LB deltas: **H2 = 3-step temporal
   median of the observed speed, thresholded at v_cut on eligible cells**. Train-lab
   (frozen v3+v5 masks, official nesting): H2 reproduces the sign and rough size of
   the weight/threshold probe deltas (weight change +0.0062 vs LB-implied +0.0027;
   on025 -0.0063 vs -0.0017; w30ong40_on025 -0.0063 vs -0.0011), while the raw
   single-step observation truth H1 gives the wrong sign for the weight change
   (-0.0076) and is rejected; the Task-1-horizon reconstruction H3 is disqualified
   (horizon rows are blanked -> climatology -> 33/80 empty truth windows, contra
   spec). A sigma2.5 denoise is far too strong (v7 already failed).
2. Absolute S_queue estimate (independent, cross-check): Q(best) = 0.80 [0.73,0.85],
   Q(P4) = 0.77 [0.70,0.82], from train-H2 nesting (P4 0.8755 / best 0.8817) plus
   the validation empty-window penalty (-0.077) and the fallback gain (+0.025),
   minus a ~0.03 convention gap (H2 is our own training target, biased high).
   This sits above the physics-calibrated estimate (0.69-0.79); the zero-queue
   probe settles it: S_queue(zeros) = (1/80)*#windows with empty eligible truth
   in [0,0.015] (point ~0.003), so
   Q_best = (0.83734 - T_zeroqueue)/0.30 + 0.003 +- 0.005
   and expected T_zeroqueue ~ 0.597-0.64 (Q 0.72-0.80).
3. Fallback arithmetic confirmed exactly: Delta_S_queue = (1/80)*sum_w Delta_IoU;
   the +0.0075 total gain = +0.025 queue means the 6 recovered onset windows gained
   IoU 1/3 each (consistent with one top-1 hit on an eligible truth set of size ~2).
4. ODME-zero probe dropped: it measures S_ODME_best - S_ODME(0), and
   S_ODME(0) in [0.015,0.30] is itself uncertain (S_attr(0)=0/0 convention), so the
   T4 anchor 0.8359 (docs/BASELINES.md) remains the better absolute value.
5. Consequence: a model retrained on median-3 hard labels (mild denoise) is now the
   only model-level lead; delegated to task2-seq (v10 candidate, two-proxy bar).


## 2026-11-03 (late): reconciled inverse analysis -> tomorrow's calibration battery

The two inverse teammates reconciled their estimates: Q=0.724 (physics-calibrated)
and Q=0.80 [0.73,0.85] (LB-delta calibrated) are both consistent with the same LB
identity; they force different splits of the remaining 0.163 loss. Reconciled
central: Q ~ 0.78, S_physics ~ 0.74 (queue remains the largest gap, 0.054-0.083,
in every scenario). Full scenario table A-E in
work/research/organizer/SCORING_MECHANISM.md §4.6.

Final probe battery for 2026-11-04 (two EXACT inversions + one upload + two
structural probes; all files built and LEAN-validated):
1. q_w30ong40fb1_all  - upload the true final candidate.
2. cal_zeroqueue      - S_queue(zeros) = E/80 ~ +0.003 +-0.005;
                        R = 0.35*S1 + 0.15*P = T_zq - 0.20*T4.
3. cal_zerostate      - state rows all zero => S_state = 0 and S_physics = 0
                        EXACTLY (RMS speed 111.8>>25, flow/lane 1040>>600; the
                        empty-road guard fires because 100% of flows < 50 vph;
                        zeroed accumulation makes e_LWR>>1) so
                        T_zs = 0.30*Q + 0.20*T4 and, with the documented T4
                        anchor 0.8359, Q = (T_zs - 0.16718)/0.30 exactly
                        (predicted T_zs 0.384-0.407 discriminates the scenarios).
4. q_on040fb1         - onset threshold .40 with fallback.
5. q_onfb2            - top-2 fallback depth.
Spares: q_min2, cal_allones (truth-set-size test, H2 predicts 0.092),
cal_statezero (sentinel state=1), cal_zeroodme (Delta-ODME only), q_v10
(median-3-label retrain, pending task2-seq).

Correction recorded: a speed-column-zero probe is NOT clean (empty-road guard
does not fire when flow stays normal; S_physics ~ S_FD/3 ~ 0.17-0.27). The clean
split of the state channels is a flow-only-zero probe: T_flow0 - T_zerostate =
0.189*S_speed exactly (optional 4th calibration slot, not currently planned).


## 2026-11-03 (late): label-regime question CLOSED - our shipped models already use H2 labels

task2-seq recomputed the labels from the plain train layer with a NaN-aware 3-step
median of speed, then threshold at v_cut and eligibility: the result reproduces the
cached v3 labels/truth arrays BIT-IDENTICALLY. In other words the shipped v3+v5
models are already trained on the H2 (median-3) regime that inverse-scoring
identified as the best-fitting description of the official truth; only 0.09% of
cells differ from the raw single-step H1 labels.

Cross-regime table (identical rows/config/fixed rules, official two-protocol):
| models | B under H2 | A under H2 |
|---|---|---|
| H2-label models (current shipped regime) | 0.8155 | 0.8347 |
| H1-label models (raw single-step) | 0.8000 | 0.8098 |
=> training on raw H1 labels LOSES 0.0155 (B) / 0.0249 (A) under the H2 truth, so
the shipped regime is right and no v10 candidate exists; v7 (sigma2.5: truth sets
too small) and v9 (soft labels: blurred boundary) failed precisely because they
moved away from this regime. Proxy sensitivity is large (same shipped models:
0.874/0.868 under H1 truth vs 0.816/0.835 under H2), and H2 is the conservative
proxy consistent with the LB-implied queue level.

Consequence: the label/model-regime line is CLOSED. Remaining Task-2 levers are
only the structural fallback-family rules and the calibration probes already
queued for 2026-11-04; the ~0.10 queue gap to the top ten is model-class
(condition-specific CNN/sequence ensembles), not label regime.


## 2026-11-03 (late): T1 horizon-gap shift -> measured, not shipped

task1-gbm measured the exact scored-split blanking pattern: per split-month the
input loses 10 blocks of 18 consecutive 5-min steps (forecast_origin+1..+18),
blanked at ALL links and BOTH channels (entire 90-min slices; offsets 0 and >=+19
have zero extra blanks). Train has 0 such blanks; validation/private have 11,675
(3.46%) / 25,295 (4.30%) extra blanks on D7_I10_E / D12_I5_N (private 12,653 /
25,655). Under a matched-rate validation-like mask the shipped GBM loses 0.0056
S_state (0.9367 -> 0.9311); retraining with matched block augmentation recovers
+0.0016 (~30%), denser augmentation and day-similarity features do not beat that
(+0.0010/+0.0011, and they regress the plain-mask case), so NO SHIP (bar +0.002).
~70% of the loss is irreducible (both temporal and spatial neighbours vanish).

Practical calibration implication: the true scored-month S_state is ~0.005 below
the plain-mask holdout number (so ~0.931 rather than 0.9367). This shifts the
identity-based queue estimate slightly (Q ~ 0.72-0.80 depending on S_physics) and
is exactly what the zero-queue / zero-state probes on 2026-11-04 will pin down.


## 2026-11-03 (late): GPU enabled (RTX 5060 Laptop, sm_120)

The machine has an NVIDIA GeForce RTX 5060 Laptop GPU (8.55 GB VRAM, driver CUDA
12.8, compute capability 12.0). The pre-installed torch was CPU-only, so we
installed the matching CUDA wheels:
  python -m pip install --user torch==2.9.1+cu128 torchvision==0.24.1+cu128       --index-url https://download.pytorch.org/whl/cu128
Verified: torch 2.9.1+cu128, torch.cuda.is_available() True, GPU matmul OK,
torchvision 0.24.1+cu128 (the old CPU torch 2.11 and torchvision 0.26 were
replaced; nothing in the pipeline depends on them). A dedicated teammate
	ask1-unet is now training a compact U-Net on 288 x L daily corridor images and
will test a 70% GBM + 30% U-Net state blend on the honest 60-day holdout with the
official Task-1 formula + Task-3 proxy; ship bar est_total >= +0.0015. This
targets the main model-class gap to the top ten (their state ensemble uses a U-Net
share, and Task 3's density L1 prefers the smoother field). A Task-2 onset CNN is
the fallback idea if the U-Net run leaves GPU time tonight.


## 2026-11-03 (late): U-Net state blend meets the ship bar on the honest holdout

task1-unet trained a compact U-Net (~1.0 M params, 3 pools, 16 base channels) on
288 x L daily corridor images, fit on train days <213 and scored on the last 60
days from each split's masked view only (GBM holdout side reproduced exactly).
Pooled over 4 panels (single seed 42, 80 epochs):

| variant | mean S_state | mean S_physics (proxy) | mean est_total delta |
|---|---:|---:|---:|
| shipped GBM | 0.9367 | 0.8584 | 0 |
| pure U-Net | 0.9375 | 0.8474 | -0.0014 |
| 0.6 GBM + 0.4 U-Net | **0.9408** | **0.8612** | **+0.0019** |
| 0.7 GBM + 0.3 U-Net | 0.9403 | 0.8610 | +0.0017 |
| 0.8 / 0.9 | 0.9393 / 0.9381 | - | +0.0013 / +0.0008 |

All 4 panels improve S_state at w=0.6 (max regression 0.0000); S_physics gains
+0.0016..+0.0023 on every panel (S_FD flat, S_LWR +0.004..+0.005). This is exactly
the pattern the strong public notebook reported: the U-Net alone is slightly worse
than the GBM, but the blend is better for both Task 1 and Task 3 (the density L1
prefers the smoother field). SHIP: global w=0.6 (grid-edge checked; deployment
will refit all 10 panels on all 273 train days with a 2-3 seed ensemble), then the
combined submission (U-Net-blend state + current best queue layer + NNLS ODME) is
probed on 2026-11-04.


## 2026-11-04: exact component inversion from the calibration probes (LB)

Three submissions with only one task layer changed (all other tasks fixed at the
current best): q_w30ong40fb1_all = 0.83734 (T_best); cal_zeroqueue = 0.62725
(queue rows all 0); cal_zerostate = 0.40830 (state rows all 0). With the official
weights and T4 anchored at 0.8359 (docs/BASELINES.md):

  Q = (T_best - T_zq)/0.30 + S_queue(zeros) = 0.7003 + E/80   (E<=2, so 0.700..0.725)
  R = 0.35*S1 + 0.15*P = T_zq - 0.20*T4 = 0.46007
  P0 := S_physics(zero state) = (T_zs - 0.30*Q - 0.20*T4)/0.15 = 0.2069

So the validation-month component table is now measured, not inferred:
| component | value | how |
|---|---|---|
| S_state | 0.92-0.94 (use 0.93) | R minus physics; T1 shift analysis puts the scored-month S_state ~0.005 below holdout |
| S_queue | **0.700** (0.700-0.725) | EXACT from the queue-zero probe (window weight 1/80) |
| S_physics | **~0.89** (0.87-0.92) | from R with S1 ~0.93 |
| S_ODME | 0.8359 | docs anchor (our Task-4 file = organizer NNLS baseline) |
| total | 0.8373 | matches the LB |

Correction: organizer-intent assumed S_physics(all-zero-state)=0; the zero-state
probe shows it is ~0.21 (under the T4 anchor), i.e. the Task-3 LWR term is not
fully zeroed by an all-zero state (the empty-road guard zeroes S_FD but the LWR
residual remains partially favourable). This does not change R or Q (both derive
from differences), it only matters for interpreting the third probe.

Remaining loss decomposition vs the public top (0.931): queue ~0.060 (dominant),
ODME ~0.023 (hidden), physics ~0.011 (to the 0.9636 ceiling), state ~0.004-0.010.
This confirms that Task 2 remains by far the largest gap (the top-ten queue is
~0.90+ vs our 0.70) while Task 3 is already near its practical ceiling.


## 2026-11-04: q_on040fb1 rejected; U-Net blend candidate built

- q_on040fb1 (onset threshold .40 + top-1 fallback, shipped weights) = 0.83376
  vs best 0.83734 (-0.00358): raising the onset threshold hurts even with the
  fallback in place. Keep the .30 threshold; the fallback is the gain, not the
  threshold shift.
- U-Net blend candidate built from the seed-43 deploy files
  (work/task1/unet/state_gbm_unet_{validation,private}.csv, 0.6*shipped GBM +
  0.4*U-Net, U-Net refit on all 273 train days): pack_state -> patch_state onto
  submission_q_w30ong40fb1_all (best queue layer + NNLS ODME), check_lean PASSED
  (6,980,503 rows; state speed min 4.33 km/h, mean 111.4), zip 101.1 MB ->
  submission_q_unet_blend.zip. Submitted as the 5th slot of 2026-11-04.


## 2026-11-04: U-Net blend candidate scores 0.83996 on the LB (new best, +0.00262)

q_unet_blend (ref 57968783) = 0.83996 vs the previous best 0.83734 (+0.00262),
larger than the holdout prediction (est_total +0.0019) -- the Task-3 part
transfers better than projected. The candidate keeps the best Task-2 queue layer
(w30ong40fb1_all) and the NNLS ODME, changing ONLY the Task-1 state rows:
0.6*shipped GBM + 0.4*U-Net, with the U-Net refit on all 273 train days, seed 43
(60 epochs, fixed step budget ~2100). Validation footprint vs the shipped state:
mean|dv| 0.287 km/h, mean|dq| 4.45 vph/lane, 16.5% of flow cells changed >1%.

Today's full probe ledger (2026-11-04):
| probe | public LB | verdict |
|---|---:|---|
| q_w30ong40fb1_all | 0.83734 | final candidate uploaded (was already the best) |
| cal_zeroqueue | 0.62725 | inversion: Q = 0.700 |
| cal_zerostate | 0.40830 | inversion: R = 0.4601, S_physics(0) ~ 0.207 |
| q_on040fb1 | 0.83376 | rejected: onset threshold .40 hurts (-0.00358) |
| **q_unet_blend** | **0.83996** | ACCEPTED: U-Net state blend, new best |

New best: q_unet_blend (ref 57968783, 0.83996). Next (2026-11-05): 2-seed U-Net
mean, the w-edge variant if the holdout prefers it, U-Net combined with the
structural fallback depth/breadth probes, then freeze and select.


## 2026-11-04 (midday): U-Net 2-seed deployment and the 2026-11-05 candidate set

task1-unet finished the 2-seed (43+44) deployment refit on all 273 train days.
Seed-averaged 4-panel holdout (fit days<213, official formula + Task-3 proxy):
| variant | S_state | S_physics | est_total | Delta vs GBM |
|---|---:|---:|---:|---:|
| shipped GBM | 0.9367 | 0.8584 | 0.4566 | - |
| pure U-Net (2-seed) | 0.9425 | 0.8586 | 0.4587 | +0.0021 |
| w=0.5 | 0.9431 | 0.8590 | 0.4597 | +0.0031 |
| w=0.55 | 0.9428 | 0.8589 | 0.4595 | +0.0029 |
| w=0.6 | 0.9423 | 0.8587 | 0.4594 | +0.0028 |
| w=0.7 | 0.9413 | 0.8581 | 0.4589 | +0.0023 |
All 4 panels improve at w=0.6, max S_state regression 0.0000; seed averaging makes
even the pure U-Net positive. Deployment footprint (2-seed w=0.6, validation):
mean|dv| 0.289 km/h (p99 1.47), mean|dq| 4.18 vph/lane, 14.0% of cells changed
>1%; private 0.302 / 4.16 / 13.6%.

2026-11-05 probe candidates (all built and LEAN-validated; 2-seed U-Net state +
best queue layer unless stated):
q_unet2_w06, q_unet2_w05, q_unet2_fb2 (onset top-2 fallback), q_unet2_min2
(onset min-2), q_unet2_w05_fb2. A Task-2 onset CNN attempt was delegated to the
task2-queue teammate on the free GPU (time box 3.5 h; ship bar +0.01 onset IoU on
both protocols); if it passes, the 5th slot goes to U-Net2-state + v11 queue.


## 2026-11-04 (afternoon): Task-2 onset CNN -> decisive NO-SHIP

task2-queue trained an onset-only corridor CNN (103k params, [23 channels x 13
steps x L] images, AMP, 1.8-2.2 s/epoch, <0.3 GB VRAM) on 5,858 onset windows
(honest fit block pre-2026 with June-2030 origins removed; Jan tuning; Feb eval;
40 released June windows). Reference recomputed from scratch reproduces the team
numbers exactly (B 0.7656, A 0.8162), so rows are identical.

| model | Jan | B_feb | A_june | Jan-selected blend |
|---|---:|---:|---:|---|
| reference (shipped v3+v5, th .30, fb1) | 0.7835 | 0.7656 | 0.8162 | - |
| CNN 17ch 60ep | 0.7081 | 0.6980 | 0.7237 | w=0 |
| CNN dilated 100ep (overfits ~epoch 15) | 0.6412 | 0.6084 | 0.6405 | w=0 |
| CNN 23ch + aug + soft-IoU 80ep | 0.6899 | 0.6555 | 0.7073 | w=0 |
| CNN 23ch + aug + BCE 40ep | 0.6749 | 0.6683 | 0.6914 | w=0 |

Any positive CNN weight trades one protocol for the other; the CNN ties the
reference on 2/8 panels and collapses on D7_I210_W / D7_I405_S. Diagnosis: onset
is dominated by static link identity + climatology + queue-front geometry, which
the 145-feature GBM ensemble encodes; a small CNN cannot learn those aggregates
from 4.3k windows. No v11 files shipped; shipped queue artefacts untouched.

This closes the Task-2 model-class line: the remaining queue gap is not reachable
with a small image CNN on this data size. Final queue layer stays
queue_w30ong40fb1_all (freezing at LB 0.83996 combined with the U-Net state).


## 2026-11-04 (evening): package reproducibility finalized

- final/ package rebuilt around the new best (submission_q_unet_blend, LB 0.83996):
  153 files, ~522 MB incl. U-Net deploy models (20 x 4.5 MB), frozen deploy
  predictions (20 npz), compressed 2-seed state components (78 MB gz), shipped
  queue components, report/PDF, full code, EXPERIMENTS, FINAL_CHOICE.
- lib/traffic.py now derives ROOT/REL/WORK from its own location with
  TRAFFIC_ROOT / TRAFFIC_REL / TRAFFIC_WORK env overrides, so the package runs
  outside the development tree; the workspace behaviour is unchanged (verified).
- New entry script code/pipeline/reproduce_final.py: run_all --skip-t2 ->
  reproduce_unet_state (0.6 GBM + 0.4 U-Net from frozen npz) -> pack_state ->
  patch_state -> patch_queue (shipped component) -> check_lean.
- New code/pipeline/verify_final.py compares the submitted zip with the shipped
  per-row components; result on the current package: VERIFIED (speed max |d|
  1.4e-14, flow 2.3e-13, queue 0, path_flow 6e-5 from float32 rounding).


## 2026-11-04 (late): package end-to-end reproduction verified in a clean copy

Copied final/ to an isolated directory and ran the package entry script with only
the released data:

  python work/pipeline/reproduce_final.py --release-root <kaggle_public>

The full chain rebuilt the caches, the Task-1 LightGBM state, the Task-2 v3 queue,
the Task-4 NNLS path flows, blended the frozen U-Net deploy predictions
(0.6 GBM + 0.4 U-Net, seeds 43+44), patched the state and the shipped queue layer,
and validated the 6,980,503-row upload (LEAN CHECK PASSED). The reproduced file
matches the intended 2-seed candidate at CSV rounding only: speed max |d| 8.0e-4
km/h, flow 8.8e-3 vph, queue exact, path_flow 6e-5. Wall time in the clean copy:
~14 min with caches, ~24 min cold.

Paths in the package are relocatable (lib/traffic.py derives ROOT/REL/WORK from
its location; scripts honour TFB_ROOT/TFB_REL and TRAFFIC_* env overrides), and
the package now ships the complete code tree (253 code files) + U-Net weights
(20 checkpoints) + frozen deploy predictions (20 npz) + compressed state
components. work/pipeline/verify_final.py re-checks the submitted zip against
the shipped per-row component arrays: VERIFIED (speed 1.4e-14, flow 2.3e-13,
queue 0, path_flow 6.0e-5 from float32 rounding).


## 2026-11-04 (afternoon): 3-seed U-Net deployment as a variance-reduction spare

task1-unet trained deploy seed 45 (all 10 panels, 273 days, same 2100-step budget;
seeds 43/44 reused) and wrote 3-seed mean blends in work/task1/unet/:
state_gbm_unet3_*.csv (w=0.6) and state_gbm_unet3_w05_*.csv (w=0.5), template-exact
(3,451,376 / 3,284,419 rows, no NaNs). Footprint vs the 2-seed files: validation
mean|dv| 0.055 km/h, 0.28% of flow cells >1% (0.0008% >5%); private 0.057 km/h,
0.30% >1% - a pure variance reduction (the 3-seed-vs-GBM footprint is unchanged
at 0.273 km/h / 13.9% >1%). The 3-seed blend is not itself holdout-evaluated; its
evidence is the 43+44 seed-averaging table and the probed seed-43 LB result.
Candidates built and LEAN-validated: q_unet3_w06, q_unet3_w05 (spares for the
2026-11-06 morning slots; the 2026-11-05 slots stay with the 2-seed candidates).


## 2026-11-04 (late): combined structural rule candidate built

The onset fallback rules act on disjoint windows (top-2 fallback only on windows
that would predict nothing; min-set-2 only on windows with a single predicted
link), and both leave ongoing windows unchanged, so the combined rule is the
per-cell union of the base, q_onfb2 and q_min2 queue layers. Built and validated:
  q_unet2_fb2min2  (2-seed U-Net state + combined queue rule)
  q_unet3_fb2min2  (3-seed variant)
Both LEAN-passed; combined queue positives 10,795 / 174,000. The 2026-11-05 probe
plan now uses the combined rule in the 5th slot (instead of the weight x rule
interaction, which stays as a 2026-11-06 spare).


## 2026-11-04 (final readiness audit)

Report: added section 3.5 (the Task-2 truth-hypothesis analysis: H2 median-3
reproduces all probe deltas, H1 has the wrong sign, H3 is disqualified, shipped
labels are bit-identical H2); refreshed section 6 (exact single-task component
inversion) and section 7 (new U-Net reproduction chain + clean-copy verification);
PDF 11 pages. Package rebuilt and verify_final.py VERIFIED (579 MB).

2026-11-05 probe set audited: all five zips exist and are LEAN-validated -
q_unet2_w06 1E047B1C3C, q_unet2_w05 6670B3152C, q_unet2_fb2 6DDC9348BF,
q_unet2_min2 39224B4AEB, q_unet2_fb2min2 B2EAAE0EBD (md5 prefixes). Quota is
5/5 used today; resets 08:00 HKT. After the batch, finalize_after_probes.py
selects the best, rebuilds the package/report and prints the two refs to select.


## 2026-11-04 (afternoon): wider U-Net -> NO-SHIP (capacity saturated)

A wider U-Net (widths [32,64,128,192], 2.23M params vs 1.12M; same 4 levels, same
loss, 16 channels; AMP batch 8, ~0.9 GB VRAM, 1.2 s/epoch on the 88-link panel)
was trained with 2 seeds (42, 43) on the 4-panel holdout protocol and compared
against the shipped 1.0M 2-seed blend:
| variant | S_state | S_physics | est_total | d vs GBM |
|---|---:|---:|---:|---:|
| shipped GBM | 0.9367 | 0.8584 | 0.4566 | - |
| pure wide U-Net (2-seed) | 0.9429 | 0.8591 | 0.4589 | +0.0023 |
| wide w=0.6 | 0.9424 | 0.8640 | 0.4594 | +0.0028 |
Head-to-head vs the current 1.0M 2-seed blend at w=0.6: d_est_total = +0.0001
(S_state +0.0001, S_physics +0.0003), far below the +0.0007 replace bar. The
wider model is genuinely slightly better alone (+0.0023 vs +0.0021) but the
blend objective is saturated by GBM<->U-Net complementarity, not by U-Net
capacity; seed averaging already removed most of the variance width would reduce.
No files shipped; no checkpoints touched. This closes the Task-1 model line.


## 2026-11-05: 2-seed probes -> new best 0.84052; "more cells" rules hurt

Five single-variable probes on top of the previous best (all use the 2-seed
U-Net state blend; the queue layer is unchanged):

| candidate | public LB | delta vs q_unet_blend 0.83996 |
|---|---:|---:|
| q_unet2_w06 (2-seed state, w=0.6) | 0.84035 | +0.00039 |
| **q_unet2_w05 (2-seed state, w=0.5)** | **0.84052** | **+0.00056** |
| q_unet2_fb2 (+ onset top-2 fallback) | 0.83766 | -0.00230 |
| q_unet2_min2 (+ min-set-2) | 0.83579 | -0.00417 |
| q_unet2_fb2min2 (combined rule) | 0.83556 | -0.00440 |

Conclusions: (1) the 2-seed U-Net state blend transfers positively and w=0.5 is
the better weight (consistent with the clean holdout); (2) adding extra predicted
cells to empty / one-link onset windows HURTS, because the eligible truth sets
there are tiny (the top-1 fallback was already the right structural rule). The
top-1 fallback stays. New best: q_unet2_w05 = 0.84052 (ref 58007101).

2026-11-06 plan (pending quota): weight curve (w=0.45 / 0.55), 3-seed variance
reduction at the best weight, and one ongoing-threshold nuance; freeze the two
selected submissions by 12:00 HKT.
