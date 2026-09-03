# RMSProp Stage 1: table beta2 x table learning rate

## Question

Does the epoch-edge shape of the frequency-bucket curves depend on the n-gram
table's RMSProp second-moment memory (`beta2`) or on its update scale?

## Fixed setting

- Model/data: baseline input injection, bigram + trigram, seed 42, fixed replay
  of train shard 1, 1000 optimizer steps.
- Backbone: AdamW `(0.8, 0.95)`, base LR `0.004`, existing warmup/warmdown,
  and weight decay `0.1`; unchanged for every run.
- Table: bias-corrected RMSProp with no momentum, epsilon `1e-10`, and zero
  weight decay. Only `beta2` and `table_lr_scale` vary.
- Measurement: same 4-batch fixed train/validation probe and same 100 fixed
  occurrences per frequency bucket. Checkpoints are every 50 steps and at
  `center +/- 10` with spacing 5 around replay edges and middle probes.
- Fixed-gram sampling: all conditions reuse the baseline's manifest after it
  is validated against this identical shard/seed/sample specification. This
  preserves exactly the same occurrence set without repeating a full scan per
  condition.

The base table LR for a condition is `0.004 * table_lr_scale`, then receives
the unchanged global schedule. The center condition already exists as the
completed run `nglab_baseline_input_midprobe_sparse_20260812`.

## Matrix

Status: `[~]` running; `[ ]` queued; `[!]` incomplete OOM attempt that will
be rerun on a GPU with sufficient free memory; `[x]` locally validated.

| beta2 | table LR scale 0.5 | table LR scale 1.0 | table LR scale 1.5 |
|---:|---|---|---|
| 0.990 | - [x] `nglab_rms_b0990_lr050_s1` | - [x] `nglab_rms_b0990_lr100_s1` | - [x] `nglab_rms_b0990_lr150_s1` |
| 0.995 | - [x] `nglab_rms_b0995_lr050_s1` | - [x] `nglab_rms_b0995_lr100_s1` | - [x] `nglab_rms_b0995_lr150_s1` |
| 0.999 | - [x] `nglab_rms_b0999_lr050_s1` | - [x] `nglab_baseline_input_midprobe_sparse_20260812` | - [x] `nglab_rms_b0999_lr150_s1` |

## Completion criteria

For each unchecked condition: `summary.json`, all three frequency JSONLs,
`frequency_measurement_meta.json`, and the generated HTML must exist. The
report records the effective table beta2 and LR scale so that results can be
audited independently of the launcher.
