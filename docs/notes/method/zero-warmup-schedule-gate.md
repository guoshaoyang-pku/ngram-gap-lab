# Zero-warmup LR schedule gate (2026-08-25)

## Status

**Done — one seed / one clean-table anchor.** This is a convergence-quality
gate for a learning-rate schedule, not evidence for a table-size law or an
injection-position conclusion.

## Question

The v4 `--lr_schedule constant` setting set LR to `0.004` from the first
optimizer update. Does its near-zero gap reflect a clean mechanism result, or
does zero warmup prevent the backbone from reaching the regime in which the
gap appears? Does a short, epoch-independent warmup recover the regime without
reintroducing warmdown?

## Matched contract

All arms used the same implementation (`code/train.py` MD5
`232da27e8e02849a27ba9e1f0ea31386`; task copy MD5
`89e62e70c0d37e446aede43f532f9c2f`), data, and seed:

| Coordinate | Value |
|---|---|
| model / injection | vanilla nanoGPT 8L/6H/768D, `input` |
| table | clean bigram + trigram, `R_bigram = R_trigram = 1,048,576` |
| optimizer | backbone AdamW LR `0.004`; table RMSProp `(0.0, 0.99)`, LR scale `2.0` |
| data | fixed replay, train shard `1`, non-overlap val shards `2..10,6542`, `epoch_batches=337` |
| budget | seed `42`, bf16, 1000 steps, no compile |
| scalar measurement | online train loss and fixed-val loss at steps 100 / 250 / 500 / 1000 |

Frequency diagnostics were intentionally disabled: this gate asks only whether
the optimizer reaches a comparable train/validation regime. It must not be
used for a frequency-decomposition figure.

The sole changed coordinate was schedule:

| run_id | schedule | warmup behavior |
|---|---|---|
| `schedcheck_v5_warmdown_r1048576_both_s42` | `warmdown` | historical 35%-of-progress warmup, then decay |
| `schedcheck_v5_constant_r1048576_both_s42` | `constant` | zero warmup, LR `0.004` from step 1 |
| `schedcheck_v5_warmup100_r1048576_both_s42` | `warmup_constant` | step 1–100 linear `0.1× → 1×`, then fixed `0.004` |

Artifacts are under `data/runs_scaling/<run_id>_fixed/` on 360-1. The
launcher is `code/cluster/run_schedule_compare.sh`; code is from scheduler
commit `b04077b` and launcher commit `e6be4dd`.

## Current-standard validation (running)

The current SSOT subsequently changed the warmup start from `0.1×` to
`0.25×` (`0.001`) in commit `c79ffd2`. The matched three-arm gate above still
establishes the zero-warmup failure, but it is not exact evidence for that
new start value. The following two one-seed, same-host anchors are registered
on `ophis-gpu`, with the contract above unchanged and `code/train.py` MD5
`88e3dacae052c7add4ed6484ba373a6a`:

| run_id | schedule | status |
|---|---|---|
| `schedcheck_v5_warmup100_1e3_r1048576_both_s42_ophisretry1` | `warmup_constant`, step 1–100 `0.001 → 0.004`, then fixed | running on GPU 0 |
| `schedcheck_v5_warmdown_r1048576_both_s42_ophis` | historical warmdown baseline | planned on GPU 3 |

Because the completed 360-1 `0.0004 → 0.004` result and the current
`0.001 → 0.004` result are on different software stacks, a third same-host
diagnostic is also registered: `schedcheck_v5_warmup100_0p4e3_r1048576_both_s42_ophis`.
It uses `code/train.py` from commit `b04077b` (MD5
`232da27e8e02849a27ba9e1f0ea31386`). A source diff establishes that this file
and commit `c79ffd2` differ in executable behavior only at the
`warmup_constant` start multiplier (`0.1` versus `0.25`); the remaining
changes are comments or argparse help.

The corresponding token shards `1`, `2..10`, and `6542` have identical MD5
checksums on 360-1 and ophis-gpu. The validation therefore isolates schedule
within one software environment; it does not overwrite the completed 360-1
gate or relabel its artifacts.

## Results

| run_id | train @1000 | val @1000 | online gap @1000 |
|---|---:|---:|---:|
| warmdown | 3.6036 | 4.2837 | +0.6801 |
| zero-warmup constant | 6.1442 | 6.1759 | +0.0318 |
| warmup-100 then constant | 3.9826 | 4.6816 | +0.6990 |

At step 500, warmdown / zero-warmup constant / warmup-100 had online train
loss 4.6751 / 6.0737 / 4.7221 and gap +0.0934 / +0.0289 / +0.1682,
respectively.

## Decision boundary

The zero-warmup `constant` arm fails this quality gate: its small gap is
coupled to high train loss, so it cannot support a clean-table gap conclusion.
`warmup_constant --warmup_steps 100` restores a low-loss, positive-gap regime
without a warmdown phase; it is the new standard schedule.

This is one anchor and one seed. Before claiming a table-size or
injection-position result, create new `v5` run IDs under the warmup-100
contract and rerun the relevant grid/ablation. Do not overwrite or re-label
`ctbl_v4_*` artifacts. The aborted first attempt with full freq-eval cadence is
retained as `schedcheck_v5_warmdown_r1048576_both_s42_aborted_freq10_fixed` and
is not evidence.
