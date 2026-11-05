# Method details

## Pipeline

```
released data ──► per-panel cache ──► Task 1 (LightGBM + U-Net) ─┐
                                   ├─► Task 2 (v3 + long history) ─┼─► merge ──► submission.csv
                                   └─► Task 4 (NNLS) ─────────────┘
                                          Task 3 scored on Task-1 rows
```

Each task is produced as a separate keyed file, merged onto the 642 MB
`submission_key.csv`, and validated against `sample_submission.csv` (6,980,503
rows, id order, task tags, value domains, binary queue flags).

## Task 1: residual GBM + U-Net

* Base: 0.9 × temporal interpolation + 0.1 × spatial-neighbour mean on masked cells.
* GBM: LightGBM residual on ~60 masked-view features, one model per panel.
* U-Net: 4 levels, widths [24,48,96,128], GroupNorm/SiLU, 16 input channels,
  2 output channels; loss `0.54*r_s^2 + 0.46*r_q^2` in official score units.
* Blend: `0.6 * GBM + 0.4 * mean(U-Net seeds)`.

## Task 2: probability blend + structural rule

* Two models: v3 static/kinematic LightGBM and a strictly-causal 3-hour-history
  LightGBM. Probabilities blended (onset 0.3, ongoing 0.4).
* Thresholds: onset 0.30 (last horizon step only), ongoing 0.20.
* Structural fallback: when a window would predict nothing, mark the top-1
  (onset) / top-1 (ongoing) most likely link. The spec guarantees a non-empty
  queue truth, so this is weakly dominant.

## Score decomposition (exact inversion)

With the official weights and three probes that zero one task at a time:

```
T_best = 0.35*S1 + 0.30*Q + 0.15*P + 0.20*T4      = 0.84052
T_zq   = 0.35*S1 + 0.15*P + 0.20*T4               = 0.62725   (queue zeroed)
T_zs   = 0.30*Q + 0.15*P0 + 0.20*T4               = 0.40830   (state zeroed)
Q      = (T_best - T_zq)/0.30 + E/80              = 0.700 (+0.000..0.025)
R      = 0.35*S1 + 0.15*P = T_zq - 0.20*T4         = 0.4601
P0     = (T_zs - 0.30*Q - 0.20*T4)/0.15            ~ 0.21
```

`T4 = 0.8359` is the published reference-baseline `S_ODME`; `E` is the number of
scored windows with an empty eligible truth set (`<= 2`).

## Reproducibility

* All models are seeded; LightGBM is deterministic.
* U-Net deploy predictions are frozen arrays shipped in the release asset, so
  reproduction needs no GPU.
* `verify_final.py` re-derives the submitted file from the shipped row-level
  components and reports VERIFIED (speed max |d| 1.4e-14, flow 2.3e-13, queue 0,
  path_flow 6e-5 from float32 rounding).
