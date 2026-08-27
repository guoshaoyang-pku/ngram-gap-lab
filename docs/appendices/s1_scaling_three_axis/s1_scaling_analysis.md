# V5 scaling statistics

- Source revision for the training batch: `7583ae3222ffb4bbfb13262295a6a828e1f08d3f`.
- Every row below is a single seed-42 run; these are descriptive fits, not uncertainty intervals.
- Gap is fixed validation loss minus the same-step current-batch online train loss.

## Table-size axis
- bigram: log-log slope `0.429009`, R² `0.976185`, rank correlation `0.997936`, n=`18`; K=`3541098`, and K/R is a load ratio only. Collision rate was not measured.
- trigram: log-log slope `0.657470`, R² `0.995109`, rank correlation `1.000000`, n=`18`; K=`19027841`, and K/R is a load ratio only. Collision rate was not measured.

## Epoch-length axis
- Quadratic descriptive fit in ln(L4 multiplier): vertex `0.411502×L4`, predicted gap `2.648528`, R² `0.774155`; this summarizes the observed U-shape and is not a mechanistic law.

## Dose axis
- Positive-gap points through 5× only: log-log slope `-1.726983`, R² `0.887225`, rank correlation `-1.000000`; the sign change is bracketed between 5× and 6×, so no global power law is reported.

## Exact-frequency axis
- bigram: token-mass-weighted geometric-bin log-log slope `-0.252746`, R² `0.997165`, rank correlation `-1.000000`, n=`7`; positive-gap bins only, with exact-f points retained in the CSV.
- trigram: token-mass-weighted geometric-bin log-log slope `-0.318121`, R² `0.995548`, rank correlation `-1.000000`, n=`7`; positive-gap bins only, with exact-f points retained in the CSV.

## Files
- `s1_table_size_points.csv`: one row per formal bigram-R/trigram-R run.
- `s1_epoch_length_points.csv`: one row per formal 3-epoch point.
- `s1_epoch_long_replay_points.csv`: epoch-boundary values for both/no-gram 10-epoch replay.
- `s1_dose_points.csv`: final dose endpoints.
- `s1_dose_frequency_gap.csv`: raw final frequency-bin rows, including token fractions.
- `s1_frequency_exact_points.csv`: shared-context exact-f rows from the formal frequency run.
- `s1_scaling_fits.csv`: descriptive fit coefficients and rank correlations.
- `v5_optimizer_points.csv`: final metrics for the clean 11-arm optimizer refresh.
