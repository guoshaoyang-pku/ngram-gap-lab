# Stage 3R: paired epoch reshuffle

Fixed setting: input injection, bigram + trigram, model seed 42, shard 1,
1000 optimizer steps, table RMSProp beta2 `0.999`, table LR scale `1.0`,
epsilon `1e-10`, zero table weight decay, and the shared fixed-gram manifest.

The permutation seed is `101`. Both conditions use the same epoch-1 logical
optimizer-batch permutation. The frozen condition repeats it; the reshuffle
condition draws a deterministic new permutation for each later epoch.

Fixed-probe construction and evaluation are disabled. Only online and
fixed-gram measurements are results for this stage.

Status notation: `[ ]` queued, `[>]` running, `[x]` completed and locally
validated, `[!]` failed/incomplete and not a result.

| train order | Run ID | Status |
|---|---|---|
| frozen permutation | `nglab_s3r_frozen_p101_s42` | [!] GPU 7 launch failure with volatile uncorrected ECC at step 10; not a result |
| frozen permutation rerun | `nglab_s3r_frozen_p101_s42_r1` | [x] completed and locally validated |
| epoch reshuffle | `nglab_s3r_reshuffle_p101_s42` | [x] completed and locally validated |

Primary boundaries are step 337/338 and 674/675. Online frequency evaluation
is dense at every step within +/-20 of the old boundary. Fixed-gram evaluation
uses new-epoch-relative offsets `-10,-5,-1,0,1,5,10` and the final step.
