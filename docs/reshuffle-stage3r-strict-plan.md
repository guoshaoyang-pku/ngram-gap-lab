# Stage 3R strict 2×2 checkpoint forks

The final matrix crosses epoch-1 order (original vs seed-101 random) with
later-epoch policy (reuse epoch-1 order vs reshuffle every epoch).

Fixed setting: input injection, bigram + trigram, model/data seed 42, order seed
101, shard 1, 1000 optimizer steps, table RMSProp beta2 0.999 and LR scale 1.0.
Fixed probe is disabled; only online and fixed-gram observables are retained.

For each epoch-1 order, the first 337 optimizer steps are executed once. The
exact post-update model, AdamW, RMSProp, RNG, and validation-iterator states are
saved, then no-shuffle and shuffle conditions resume from that checkpoint at
step 338. Thus all four conditions use data seed 42 and order seed 101; only the
two intended order factors vary.

| phase / condition | run id | status |
|---|---|---|
| shared sequential prefix through step 337 | `nglab_s3r2_common_seq_s42` | [!] GPU 6 was claimed by another process; OOM at step 1, not a result |
| shared sequential prefix rerun | `nglab_s3r2_common_seq_s42_r1` | [x] Liu H200 GPU 7; checkpoint SHA256 `cc487a7acd2042f70d893d789bb7331ebbd5c303464dbe17496f0cdc8c0946ca` |
| sequential continuation | `nglab_s3r2_sequential_s42` | [x] completed and locally validated; final gap `+0.9212` |
| epoch reshuffle continuation | `nglab_s3r2_reshuffle_s42_p101` | [x] completed and locally validated; final gap `+0.5727` |
| shared random-order prefix through step 337 | `nglab_s3r3_common_random_s42_p101` | [x] checkpoint SHA256 `3a616d781269cf7879ce8fd1959a16f5bf1ccbb4406dfd77d9fb86797a831665` |
| repeat epoch-1 random order | `nglab_s3r3_random_frozen_s42_p101` | [x] completed and locally validated; final gap `+1.2695` |
| reshuffle after random epoch 1 | `nglab_s3r3_random_reshuffle_s42_p101` | [x] completed and locally validated; final gap `+0.7882` |

Acceptance requirements:

- prefix checkpoint is post-update step 337 and records a parameter-state SHA256;
- both branches reference the same checkpoint SHA256;
- both complete logs contain steps 1–1000 and share identical rows through 337;
- sequential order is `0..336` in every epoch;
- reshuffle order is `0..336` in epoch 1 and a seed-101 permutation thereafter;
- no fixed-probe JSONL exists;
- online frequency is dense around starts 338 and 675;
- fixed-gram checkpoints are exactly 328, 333, 337, 338, 339, 343, 348,
  665, 670, 674, 675, 676, 680, 685, and 1000.
