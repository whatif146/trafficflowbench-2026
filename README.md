# TrafficFlowBench — 2026 IEEE Big Data Cup (Track 2)

A reproducible, evidence-driven solution to the **TrafficFlowBench** challenge:
reconstruct masked freeway speed/flow, forecast queue propagation, satisfy
freeway physics, and invert link counts into path flows — all at 5-minute
resolution over ten directional corridors.

| | |
|---|---|
| Competition | [2026 IEEE Big Data Cup, Challenge 06, Track 2](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench) |
| Team | CAMPUS SUNNY (Kaggle `hsuyooyoo`) |
| Best public score | **0.84052** (`S_total`), rank 430 / 766 at submission time |
| Top of public LB | 0.93124 |
| Deadline / submission | 2026-11-07 |
| Code, report, predictions | this repository + the **final-package release asset** |

## Why this repository

Most competition repos are a dump of notebooks. This one is organised as an
engineering case study:

* **Every accepted change is a single-variable A/B probe** on the official
  leaderboard, so each delta maps to one scoring component.
* The four hidden components were **measured, not guessed**, by submitting
  single-task-crippled files (all-zero queue / all-zero state) and inverting the
  official weighted-sum identity.
* **Negative results are first-class**: five independent model families were
  built, measured and rejected with evidence (see `docs/EXPERIMENTS.md`).
* The final submission is **reproducible end-to-end from the released data** and
  the package is verified row-by-row against the uploaded file.

## Results

### Public leaderboard progression (single-variable probes)

| step | change | `S_total` |
|---|---|---:|
| v1 | temporal/spatial smoother + Task-2 v3 | 0.79509 |
| v2 | + Task-2 v3 queue model | 0.79775 |
| v4 | + Task-1 LightGBM residual model | 0.82448 |
| v5 | + long-history Task-2 blend | 0.82902 |
| v6 | + blend weights | 0.82984 |
| v7 | + empty-window fallback (structural) | 0.83652 |
| v8 | + weights + fallback | 0.83734 |
| v9 | + U-Net state blend (0.6 GBM / 0.4 U-Net) | 0.83996 |
| **v10** | **+ two-seed U-Net average, weight 0.5** | **0.84052** |

### Measured score decomposition (validation month)

The official score is
`S_total = 0.35·S_state + 0.30·S_queue + 0.15·S_physics + 0.20·S_ODME`.
Three single-task probes (queue zeroed `0.62725`, state zeroed `0.40830`, full
`0.84052`) plus the documented Task-4 baseline anchor (`S_ODME = 0.8359`) give:

| component | value | evidence |
|---|---:|---|
| `S_state` | ~0.93 | holdout 0.9367 (GBM) → 0.9423 (U-Net blend) |
| `S_queue` | **0.700** | exact inversion from the queue-zero probe |
| `S_physics` | ~0.895 | residual `0.35·S_state + 0.15·S_physics = 0.4601` |
| `S_ODME` | 0.8359 | published reference baseline (our file is that baseline) |

The remaining loss is dominated by Task 2 (~0.06 of `S_total`), which is a
model-class/data-size gap, not a tuning gap — see the negative results below.

## Approach

| task | method |
|---|---|
| **Task 1 — state** | 0.6 × LightGBM residual model on a temporal-interpolation / spatial-neighbour base + 0.4 × compact 2-D U-Net over `[16 × 288 × L]` daily corridor images, loss expressed directly in official score units. U-Net refit on all 273 train days, seeds averaged. |
| **Task 2 — queue** | LightGBM v3 + strictly-causal 3-hour-history LightGBM probability blend (onset `w=0.3`, threshold 0.30 on the last step; ongoing `w=0.4`, threshold 0.20), plus a **spec-justified fallback**: every scored window contains a queue somewhere in its horizon, so an empty prediction is a guaranteed IoU of 0 — marking the single most likely link is weakly dominant (+0.0075 on the LB, transfers across months). |
| **Task 3 — physics** | No submission file: scored on the Task-1 rows. We showed the S_LWR term is an L1 penalty on the accumulation error and that no legal post-processing improves it beyond fixing Task 1. |
| **Task 4 — ODME** | Official regularised NNLS baseline form (prior λ=0.05), `S_link = 0.999935` locally. Entropy/null-space reallocation was tested and decisively rejected (−0.149 `S_ODME`). |

## Key findings

1. **Structural fallback beats tuning.** The scoring spec guarantees a non-empty
   queue truth in every scored window; the empty-window fallback is the only
   provably non-negative change available, worth +0.0075 total.
2. **The Task-2 truth is a latent state, not the raw observation.** A 3-step
   temporal-median-then-threshold hypothesis reproduces every probe delta; the
   shipped labels are already that construction, so the residual gap is model
   class, not label noise.
3. **Model capacity saturates the blend.** A 2.23M-parameter U-Net (vs 1.12M)
   changed the blend holdout `est_total` by +0.0001 — the 0.6/0.4 mix is limited
   by GBM↔U-Net complementarity, not by capacity.
4. **An image CNN does not win onset.** Five variants never beat the
   145-feature GBM ensemble (Jan 0.71–0.78 vs 0.784), because onset is dominated
   by static link identity, climatology and queue-front geometry.

## Repository layout

```
src/                    code used for the submitted predictions
  lib/traffic.py        released-data parsing + per-panel caches
  task1/                Task-1 LightGBM + U-Net + evaluation
  task2/ task2_seq/     Task-2 v3, long-history blend, decision rules
  task2_cnn/            Task-2 onset CNN (negative result, kept for evidence)
  task3/                calibrated physics proxy + conservation analysis
  task4/                ODME NNLS solver and alternatives
  pipeline/             cache packing, merging, validation, reproducibility
docs/
  TrafficFlowBench_Report.pdf   full write-up (11 pages)
  EXPERIMENTS.md                single-variable A/B ledger incl. all negatives
  method.md                     pipeline details and component inversion math
  inverse_scoring.md            how the hidden scoring mechanism was reverse-engineered
results/                LB progression, component table, submission fingerprints
```

## Quick start

```bash
python -m pip install -r requirements.txt
# download the released data from Kaggle (competition files) into ./kaggle_public
python src/build_cache.py                     # ~8 min, one-off per panel set
python src/task1/gbm/predict.py --split validation private --outdir work/task1/gbm
python src/task2/predict.py --model gbmv3 --split both --output work/task2/out/queue_gbmv3.csv
python src/task4/predict.py --split validation --method baseline --output work/task4/out/path_flow_validation.csv
```

For the **exact final submission**, use the release asset
`submission_package.zip` (code + frozen predictions + weights + report) and run

```
python work/pipeline/reproduce_final.py --release-root /path/to/kaggle_public
```

CPU only, ~25 minutes cold, and `work/pipeline/verify_final.py` checks the
produced file against the shipped row-level components (VERIFIED on the
submitted package).

## What is not in this repository

* The competition release data (11.5 GB) is **not redistributed**; it belongs to
  the organizers and must be downloaded from Kaggle.
* Large artifacts (U-Net weights, frozen deploy predictions, the final package)
  are attached to the **GitHub release**, not committed to git history.

## License

Code: MIT. Report and documentation: CC BY 4.0. The competition data and any
organizer-provided files remain under their original terms.
