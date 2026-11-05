# Final submission selection — instructions for the user

Updated 2026-11-03 20:35 HKT. The competition allows selecting **up to 2
successful submissions** for final scoring (private leaderboard). Selection is on
Kaggle: competition page → **Submissions** tab → tick the checkboxes of the
chosen submissions → **Submit / Select**. Deadline: **2026-11-06**.

## Current ranking (public LB)

| rank | ref | public LB | contents |
|---:|---:|---:|---|
| 1 | **57947511** | **0.83734** | T1 LightGBM + T2 blend (onset w .3/th .30 + top-1 fallback on empty windows; ongoing w .4/th .20) + T4 NNLS |
| 2 | 57947197 | 0.83652 | T2 onset th .30 + top-1 fallback, shipped blend weights (the structural fix alone) |
| 3 | 57946739 | 0.82984 | T2 blend weights only (no fallback) |
| 4 | 57906197 | 0.82902 | P4: shipped v3+v5 blend (th .30 / .25), no fallback |
| 5 | 57872768 | 0.82448 | v4: T2 v3 queue model + T1 LightGBM + T4 NNLS |
| - | 57946946 | 0.82852 | q_on025 — rejected |
| - | 57946514 | 0.82868 | weights + th .25 — rejected |
| - | 57906566 | 0.81537 | morphological closing — rejected |

## Why the fallback is not public-month overfitting

The 40 validation `queue_onset` windows include 6 whose model probabilities are all
below the 0.30 threshold, so the shipped rule predicts **nothing** there (IoU 0 by
construction). The official scoring specification guarantees every onset window
contains a queue, so marking the single most likely link instead is a structural
fix that should transfer to the private month. On the LB it is worth +0.0075
(0.82902 -> 0.83652) and the blend-weight recalibration adds +0.0008 on top
(-> 0.83734).

## Recommendation (re-checked after the 2026-11-04 refinement probes)

1. **Entry 1: the best public-LB candidate** (currently 57947511: fallback +
   blend weights).
2. **Entry 2: 57947197** (fallback + shipped weights) — same structural fix,
   but *without* the small weight recalibration, which is the component most at
   risk of not transferring to the private month. This is a deliberately diverse
   pair on the only axis we changed.
3. Alternative conservative entry if the user prefers: P4 (57906197), the
   fallback-free configuration — but the fallback is structural, so the two
   fallback entries above are recommended.

Do **not** select the rejected probes (closing, top-k onset, th .25, ongoing
threshold .40/.15, Task-4 entropy reallocation).

## Other items only the user can do

- **Team name**: change `Hsu Yooyoo` → `CAMPUS SUNNY` on the Kaggle **Team**
   page (the public API cannot rename a team).
- Code + PDF report submission: **2026-11-10 23:59 AoE** (package in `final/`).

## How to submit (authoritative rules, fetched 2026-11-03 21:20 HKT)

1. **Leaderboard**: Competition -> **Submissions** tab -> tick up to 2 of our
   uploaded submissions -> **Select**. Kaggle's authoritative cutoff is
   `2026-11-07 06:55 UTC` (2026-11-07 14:55 HKT / 2026-11-06 23:55 US Pacific);
   the organizers' timetable lists November 6. Do not wait for the last hour.
2. **Code + PDF report (required for awards)**: by **2026-11-10 23:59 AoE** e-mail
   the package (GitHub link or zip) and the PDF report to
   `trafficflowbench@gmail.com` with subject
   `Traffic Flow Bench Final Package - CAMPUS SUNNY`. It must reproduce the
   selected final submission: the `final/` package contains a README,
   environment details and a single entry script (`code/reproduce_final.py`).
3. **Award registration** was due 2026-10-25 23:59 AoE by e-mail to the same
   address (team name, usernames, full names, affiliations). If that e-mail was
   not sent, the team can still submit and appear on the leaderboard but may not
   be eligible for the cash awards - check with the organizers.
4. **Team name**: change it to `CAMPUS SUNNY` on the Kaggle **Team** page (the
   API cannot rename a team); the report and the final-package e-mail use it.

## Uploaded vs final-candidate subtleties (2026-11-03)

- The best *uploaded* submission is ref 57947511 (q_w30ong40fb1, 0.83734).
- The best *candidate* is q_w30ong40fb1_all, which additionally applies the
  spec-guaranteed ongoing-window fallback (validation: exactly 1 cell changed,
  private: 0). It is weakly dominant (it can only turn a guaranteed-zero window
  into a positive IoU), but it must be uploaded to be selectable: it is now in
  the 2026-11-04 probe plan, so it will have a ref by tomorrow.
- Reproducibility fingerprints of the final candidate: queue val md5
  e9240ff4f466093a128c02ba614877a1, priv md5 1c0c5e22c9f89aa9ddbe816c49b73b43,
  merged submission CSV md5 c994b07558d68622bfc88fde722a03a1, package zip md5
  e950a4d2b60adabe15f1ca358b0a5c6c (all verified identical on re-run).

## Team-name check (2026-11-03 21:20 HKT)

The live Kaggle API leaderboard (fetched just now, team id 17201406, rank 427,
score 0.83734) still reports the competition team name as **Hsu Yooyoo**, and the
submission objects store the same historical name. The registration e-mail is
confirmed sent by the user (awards eligibility OK). If the competition Team page
itself shows CAMPUS SUNNY, the API/leaderboard is just lagging; otherwise check
that the rename was applied on the *competition Team tab* (not the profile
display name) and save it. The report and the final-package e-mail use
CAMPUS SUNNY either way; re-check the leaderboard row before the final e-mail.

## 2026-11-04 update — U-Net state blend is the new best

| rank | ref | public LB | contents |
|---:|---:|---:|---|
| 1 | **57968783** | **0.83996** | T1 = 0.6*LightGBM + 0.4*U-Net state blend (U-Net refit on all 273 train days, seed 43) + T2 best queue layer (fallback + weights) + T4 NNLS |
| 2 | 57967779 | 0.83734 | T1 GBM only + same T2/T4 (U-Net-free fallback candidate) |
| 3 | 57947511 | 0.83734 | T1 GBM + Q fallback with shipped weights |
| 4 | 57947197 | 0.83652 | T2 fallback only |

Recommendation (re-check after the 2026-11-05 probes): **57968783 (U-Net blend) +
57947511 or 57967779 (GBM-only hedge)**. The pair is diverse on the newest
component (U-Net state); if the U-Net does not transfer to the private month the
GBM-only entry is the safe fallback. Both entries share the structural Task-2
fallback, which is spec-justified and should transfer.

## 2026-11-04 late: measured components and final selection rule

Exact component inversion from the three single-task probes (official weights,
T4 = 0.8359 documented anchor): S_queue = 0.700 (0.700-0.725), R = 0.35*S1 +
0.15*P = 0.4601, S_state ~ 0.93, S_physics ~ 0.895. Queue remains the largest
gap (0.060 of the 0.094 total), and a decisive onset-CNN experiment showed it is
not reachable with an image CNN at this data size; the remaining Task-2 idea is
tree-side features, out of scope for the deadline.

2026-11-05 probes (all 2-seed U-Net, refit on all 273 train days): w=0.6, w=0.5,
w=0.6+onset top-2 fallback, w=0.6+min-2, w=0.5+top-2. Decision rule: keep the
highest public score; among entries within +0.0005 of the best prefer the
structurally justified rule (fallback depth/breadth over a threshold change) and
the 2-seed blend. Freeze by 2026-11-06 12:00 HKT.

Selection plan (two picks): best probe (U-Net blend family) + a GBM-only hedge
(57967779 or 57947511) so the pair is robust if the U-Net does not transfer to the
private month. Do not pick two entries that differ only in the fourth decimal.

## 2026-11-05 probe results

| candidate | public LB | ref |
|---|---:|---:|
| q_unet_blend | 0.83996 | 57968783 |
| q_w30ong40fb1 | 0.83734 | 57947511 |
| q_w30ong40fb1_all | 0.83734 | 57967779 |
| q_onfb1 | 0.83652 | 57947197 |
| q_w30ong40 | 0.82984 | 57946739 |
| q_w30ong40_on025 | 0.82868 | 57946514 |
| q_on025 | 0.82852 | 57946946 |

**Selected primary: q_unet_blend (ref 57968783, 0.83996).**
Backup (GBM-only hedge): q_w30ong40fb1 ref 57947511 (0.83734).

## 2026-11-05 probe results

| candidate | public LB | ref |
|---|---:|---:|
| q_unet2_w05 | 0.84052 | 58007101 |
| q_unet2_w06 | 0.84035 | 58006844 |
| q_unet_blend | 0.83996 | 57968783 |
| q_unet2_fb2 | 0.83766 | 58007346 |
| q_w30ong40fb1 | 0.83734 | 57947511 |
| q_w30ong40fb1_all | 0.83734 | 57967779 |
| q_onfb1 | 0.83652 | 57947197 |
| q_unet2_min2 | 0.83579 | 58007575 |
| q_unet2_fb2min2 | 0.83556 | 58007768 |
| q_w30ong40 | 0.82984 | 57946739 |
| q_w30ong40_on025 | 0.82868 | 57946514 |
| q_on025 | 0.82852 | 57946946 |

**Selected primary: q_unet2_w05 (ref 58007101, 0.84052).**
Backup (GBM-only hedge): q_w30ong40fb1 ref 57947511 (0.83734).
