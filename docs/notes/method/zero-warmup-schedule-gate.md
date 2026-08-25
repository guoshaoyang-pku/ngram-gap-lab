# Zero-warmup LR schedule gate (2026-08-25)

## Status

**Done — one seed / one clean-table anchor per software environment.** This is
a convergence-quality gate for a learning-rate schedule, not evidence for a
table-size law or an injection-position conclusion.

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

## Current-standard validation (complete)

The current SSOT subsequently changed the warmup start from `0.1×` to
`0.25×` (`0.001`) in commit `c79ffd2`. The matched three-arm gate above still
establishes the zero-warmup failure, but it is not exact evidence for that
new start value. The following one-seed, same-host anchors ran on `ophis-gpu`,
with the contract above unchanged and current `code/train.py` MD5
`88e3dacae052c7add4ed6484ba373a6a`:

| run_id | schedule | status |
|---|---|---|
| `schedcheck_v5_warmup100_1e3_r1048576_both_s42_ophisretry1` | `warmup_constant`, step 1–100 `0.001 → 0.004`, then fixed | done |
| `schedcheck_v5_warmdown_r1048576_both_s42_ophis` | historical warmdown baseline | done |

Because the completed 360-1 `0.0004 → 0.004` result and the current
`0.001 → 0.004` result are on different software stacks, a third same-host
diagnostic also ran: `schedcheck_v5_warmup100_0p4e3_r1048576_both_s42_ophis`.
It uses `code/train.py` from commit `b04077b` (MD5
`232da27e8e02849a27ba9e1f0ea31386`). A source diff establishes that this file
and commit `c79ffd2` differ in executable behavior only at the
`warmup_constant` start multiplier (`0.1` versus `0.25`); the remaining
changes are comments or argparse help.

The corresponding token shards `1`, `2..10`, and `6542` have identical MD5
checksums on 360-1 and ophis-gpu. The validation therefore isolates schedule
within one software environment; it does not overwrite the completed 360-1
gate or relabel its artifacts.

## Results: 360-1 matched gate

| run_id | train @1000 | val @1000 | online gap @1000 |
|---|---:|---:|---:|
| warmdown | 3.6036 | 4.2837 | +0.6801 |
| zero-warmup constant | 6.1442 | 6.1759 | +0.0318 |
| warmup-100 then constant | 3.9826 | 4.6816 | +0.6990 |

At step 500, warmdown / zero-warmup constant / warmup-100 had online train
loss 4.6751 / 6.0737 / 4.7221 and gap +0.0934 / +0.0289 / +0.1682,
respectively.

## Results: ophis-gpu same-host scale control

The three rows below have the same seed, data checksums, model, clean-table
capacities, optimizer, evaluation steps, and host. The two fixed-LR rows are
the historical and current source variants described above; they differ only
in the warmup start (`0.0004` versus `0.001`).

| schedule | train @500 | gap @500 | train @1000 | val @1000 | online gap @1000 |
|---|---:|---:|---:|---:|---:|
| warmdown | 4.6848 | +0.1333 | 3.6387 | 4.3031 | **+0.6644** |
| warmup-100, `0.0004 → 0.004`, then fixed | 5.0997 | +0.0924 | 5.7240 | 5.8392 | **+0.1152** |
| warmup-100, `0.001 → 0.004`, then fixed | 5.4753 | +0.0917 | 5.9251 | 6.0271 | **+0.1020** |

Thus the `0.001` start is modestly worse than `0.0004`, but neither
100-step warmup variant reaches the warmdown quality regime on this matched
anchor. The earlier 360-1 `0.0004` arm did reach it (3.9826 / 4.6816 / +0.6990),
so warmup-then-fixed-LR is stack-sensitive rather than a robust replacement.

## Decision boundary

The zero-warmup `constant` arm fails this quality gate: its small gap is
coupled to high train loss, so it cannot support a clean-table gap conclusion.
More importantly, **100-step `warmup_constant` followed by a permanent
`0.004` also fails the gate on the same-host control**, regardless of whether
it starts at `0.0004` or `0.001`. It must not be presented as a validated
replacement for warmdown or used for a clean-table conclusion.

The current setting SSOT is not silently changed by this diagnostic. Until an
epoch-independent no-warmdown schedule passes this gate robustly, new
clean-table grid/ablation launches are blocked: do not start or relabel
`ctbl_v4w_*` as evidence. A user decision is required to either retain
warmdown temporarily or authorize a new no-warmdown schedule search. The
aborted first attempt with full freq-eval cadence is retained as
`schedcheck_v5_warmdown_r1048576_both_s42_aborted_freq10_fixed` and is not
evidence.

## Phase 2: minimal no-warmdown schedule search (planned)

**Question.** Is the failed fixed-LR plateau caused chiefly by too-short
warmup, or does the post-warmup decay itself supply the late-stage convergence?
The only changed coordinate is the LR schedule. All arms use the clean-table
anchor above: input injection, `R_bigram = R_trigram = 1,048,576`, AdamW
backbone LR `0.004`, RMSProp tables `(0.0, 0.99)` at LR scale `2.0`, shard 1
train / shards `2..10,6542` fixed validation, seed 42, bf16/no-compile,
1000 steps, and `epoch_batches=337`.

The runner is `code/cluster/run_schedule_search.sh` from commit `a5efb31`.
It records only fixed scalar checkpoints at steps 100/250/500/750/1000; no
frequency diagnostic is enabled during this convergence screen. `warmup_cosine`
is step-anchored: the shared linear warmup is `0.001 → 0.004`, then a cosine
decay reaches multiplier `0.05` exactly at step 1000. It never reads an epoch
boundary. This floor is explicit as `--cosine_min_lr_mult 0.05`.

| run_id stem | changed flags | status |
|---|---|---|
| `schedgrid_v1_hold_w200_r1048576_both_s42` | `--lr_schedule warmup_constant --warmup_steps 200` | done; fails gate |
| `schedgrid_v1_hold_w300_r1048576_both_s42` | `--lr_schedule warmup_constant --warmup_steps 300` | done; fails gate |
| `schedgrid_v1_cosine_w100_r1048576_both_s42` | `--lr_schedule warmup_cosine --warmup_steps 100 --cosine_min_lr_mult 0.05` | done; fails gate |
| `schedgrid_v1_cosine_w200_r1048576_both_s42` | `--lr_schedule warmup_cosine --warmup_steps 200 --cosine_min_lr_mult 0.05` | done; fails gate |
| `schedgrid_v1_cosine_w300_r1048576_both_s42` | `--lr_schedule warmup_cosine --warmup_steps 300 --cosine_min_lr_mult 0.05` | done; best first-pass cosine, still fails gate |
| `schedgrid_v1_cosine_w400_r1048576_both_s42` | `--lr_schedule warmup_cosine --warmup_steps 400 --cosine_min_lr_mult 0.05` | done; fails gate |
| `schedgrid_v1_warmdown_current_r1048576_both_s42` | `--lr_schedule warmdown` | failed before step 1: CUDA launch failure on known-bad 360-1 GPU 7 |
| `schedgrid_v1_cosine_w500_r1048576_both_s42` | `--lr_schedule warmup_cosine --warmup_steps 500 --cosine_min_lr_mult 0.05` | stalled; no known-good 360-1 card became available |

**Falsifiable gate.** A candidate passes only if its final online train loss is
at most `4.3` and its final online gap is at least `+0.5` on this anchor; this
is deliberately looser than the same-host warmdown reference (3.6387 / +0.6644)
but rejects the fixed-LR plateau. A passing arm must then be rerun on a second
software stack before it becomes an SSOT setting. A failed arm remains a
recorded diagnostic, not a table-size result.

### First-pass results (360-1)

| schedule | train @1000 | val @1000 | online gap @1000 |
|---|---:|---:|---:|
| hold, warmup 200 | 6.1105 | 6.2048 | +0.0943 |
| hold, warmup 300 | 5.8822 | 6.0011 | +0.1189 |
| cosine, warmup 100, floor 0.05 | 5.2921 | 5.4500 | +0.1578 |
| cosine, warmup 200, floor 0.05 | 5.2379 | 5.4566 | +0.2187 |
| cosine, warmup 300, floor 0.05 | 4.7952 | 5.0659 | +0.2707 |
| cosine, warmup 400, floor 0.05 | 5.1785 | 5.3760 | +0.1975 |

The hold arms show that longer warmup alone does not cure the plateau. Cosine
decay helps but none passes the convergence gate. The best first-pass arm,
cosine warmup 300, has a lower late-stage multiplier than warmdown; Phase 3
therefore changes only its explicit cosine floor.

## Phase 3: cosine-tail floor control (planned)

Both arms retain `warmup_cosine`, a 300-step `0.001 → 0.004` warmup, all
Phase-2 coordinates, and the scalar checkpoint cadence. Only
`--cosine_min_lr_mult` changes. They run against the existing ophis-gpu
warmdown anchor, which has the same data checksums and unchanged warmdown
implementation.

| run_id stem | changed flag | status |
|---|---|---|
| `schedgrid_v1_cosine_w300_floor10_r1048576_both_s42` | `--cosine_min_lr_mult 0.10` | done; fails gate |
| `schedgrid_v1_cosine_w300_floor20_r1048576_both_s42` | `--cosine_min_lr_mult 0.20` | done; best floor control, still fails gate |

| cosine floor | train @1000 | val @1000 | online gap @1000 |
|---:|---:|---:|---:|
| 0.10 | 4.7531 | 5.0237 | +0.2705 |
| 0.20 | 4.4065 | 4.7373 | +0.3308 |

## Phase 4: cosine-tail continuation (planned)

Phase 3 improves monotonically from a 0.10 to 0.20 floor without changing
the warmup or model. To locate whether this is still an improving direction
or an overshoot, the next two one-coordinate arms keep the exact Phase-3
contract and set only the terminal cosine multiplier to 0.30 or 0.40.

| run_id stem | changed flag | status |
|---|---|---|
| `schedgrid_v1_cosine_w300_floor30_r1048576_both_s42` | `--cosine_min_lr_mult 0.30` | planned on ophis-gpu GPU 0 |
| `schedgrid_v1_cosine_w300_floor40_r1048576_both_s42` | `--cosine_min_lr_mult 0.40` | planned on ophis-gpu GPU 3 |
