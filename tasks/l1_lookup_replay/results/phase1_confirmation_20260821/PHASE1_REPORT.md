# Phase 1 confirmatory analysis (toy6, seeds 20001-20040)

raw: raw/metrics.jsonl
rows: 2560; seeds: 40

## Gates (runbook 6.2)

| gate | result | value |
|---|---|---|
| g1_noise0_all_table_residual_lt_1e-12 | PASS | max_abs=0.000e+00 |
| g2_k1_fixed_fresh_identity_lt_1e-12 | PASS |  |
| g3_k64_gap_did_positive | PASS | mean=+0.079375 CI=[+0.077567,+0.081182] |
| g4_k64_val_pos_train_neg | PASS |  |
| g5_mean_gap_did_monotone | PASS |  |
| g6_beta_ci_intersects_0p8_1p2 | PASS | beta=0.9692 90%CI=[0.9414,0.9970] n=40 |

Overall: PASS

## DID series (population_gap, fixed*memory interaction, seed-level mean)

| K | mean | 95% CI |
|---|---|---|
| 2 | +0.009428 | [+0.009139,+0.009716] |
| 4 | +0.023523 | [+0.023034,+0.024013] |
| 8 | +0.042280 | [+0.041341,+0.043220] |
| 16 | +0.059811 | [+0.058347,+0.061275] |
| 32 | +0.072896 | [+0.071228,+0.074563] |
| 64 | +0.079375 | [+0.077567,+0.081182] |

## val / train DID @ K=64

- population_ce DID: mean=+0.036195 CI=[+0.035397,+0.036993]
- train_ce DID: mean=-0.043180 CI=[-0.044212,-0.042147]

Note: Phase 1 confirms the Adam toy phenomenon only; no optimizer/LLM external validity.