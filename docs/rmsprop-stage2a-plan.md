# RMSProp Stage 2A execution matrix

Fixed setting: input injection, bigram + trigram, seed 42, shard 1 fixed replay,
1000 optimizer steps, table RMSProp with epsilon `1e-10` and zero weight decay.

Status notation: `[ ]` queued, `[>]` running, `[x]` completed and locally validated,
`[!]` failed/incomplete and not a result.

| beta2 | LR scale | Run ID | Status |
|---:|---:|---|---|
| 0.500 | 1.000 | `nglab_s2a_b0500_lr1000_s42` | [x] locally validated; final gap `+0.7909`; Liu H200 GPU 1 |
| 0.900 | 1.000 | `nglab_s2a_b0900_lr1000_s42` | [x] locally validated; final gap `+0.9608`; Tencent A100 GPU 4 |
| 0.999 | 0.000 | `nglab_s2a_b0999_lr0000_s42_r1` | [x] locally validated; final gap `+0.0977`; Liu H200 GPU 2; initial run rejected zero LR |
| 0.999 | 0.125 | `nglab_s2a_b0999_lr0125_s42` | [x] locally validated; final gap `+0.4619`; Tencent A100 GPU 5 |
| 0.999 | 0.250 | `nglab_s2a_b0999_lr0250_s42` | [x] locally validated; final gap `+0.6240`; Liu H200 GPU 7 |
| 0.999 | 0.375 | `nglab_s2a_b0999_lr0375_s42` | [x] locally validated; final gap `+0.6068`; Tencent A100 GPU 6 |
| 0.999 | 0.625 | `nglab_s2a_b0999_lr0625_s42` | [x] locally validated; final gap `+0.8107`; Liu H200 GPU 4 |
| 0.999 | 0.750 | `nglab_s2a_b0999_lr0750_s42` | [x] locally validated; final gap `+0.8040`; Tencent A100 GPU 7 |
| 0.999 | 0.875 | `nglab_s2a_b0999_lr0875_s42` | [x] locally validated; final gap `+0.8969`; Liu H200 GPU 4 |
| 0.500 | 0.250 | `nglab_s2a_b0500_lr0250_s42` | [x] locally validated; final gap `+0.3572`; Liu H200 GPU 1 |
| 0.500 | 0.500 | `nglab_s2a_b0500_lr0500_s42` | [x] locally validated; final gap `+0.5381`; Liu H200 GPU 7 |
| 0.900 | 0.250 | `nglab_s2a_b0900_lr0250_s42` | [x] locally validated; final gap `+0.5149`; Liu H200 GPU 2 |
| 0.900 | 0.500 | `nglab_s2a_b0900_lr0500_s42` | [x] locally validated; final gap `+0.7207`; Liu H200 GPU 4 |

Existing Stage 1 runs supply `beta2 in {0.990, 0.995, 0.999}` at LR scale
`1.0` and `beta2=0.999` at LR scale `0.5`; they are not rerun here.
