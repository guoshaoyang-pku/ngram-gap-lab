# NanoGPT gap campaign exclusion ledger

Generated: 2026-07-16T12:18:32Z

This ledger is fail-closed. A listed result root, including any otherwise successful arm inside that root, must not enter a formal estimate. Failed directories are retained permanently and replacements use new immutable source snapshots and new result roots.

| Result root | Status | Failure boundary | Replacement |
|---|---|---|---|
| `/data2/ncpl-pathA/work/nanogpt_gap_single_exposure_smoke_seed42_20260716_v2` | excluded | The first compiled background forward failed with `RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`; the arm returned rc=1 after 412.89 s. | `single_exposure_v4_4c103993_a883a941_67ff34ac`; result root `nanogpt_gap_single_exposure_smoke_seed42_20260716_v3` |
| `/data2/ncpl-pathA/work/nanogpt_gap_boundary_remap_20260716_v2` | excluded | Compiled identity arms hit a Triton device-side bounds assertion (`index ... < 524288`). The dynamic int64 modulus path and an incorrectly registered modulus made this implementation ineligible. | `boundary_address_remap_v4_0c8bbbc6_c4759126_6f3cda9d` after raw-vs-compiled address/lookup and sequence-QC qualification |
| `/data2/ncpl-pathA/work/nanogpt_gap_single_exposure_pilot_seed42_20260716_v3` | excluded | The sham arm correctly failed closed at background update 21 because the batch re-exposed two protected probe samples. Permitting the exposure would invalidate the single-exposure estimand. | `single_exposure_v5_ff79623c_b3cd8cec_c701d0c6`, using a cross-arm exact rejection stream; result root `nanogpt_gap_single_exposure_pilot_seed42_20260716_v4` |
| `/data2/ncpl-pathA/work/nanogpt_gap_vanilla_graft_smoke_20260716_v3_read_delta` | excluded | Post-QC treated each arm's private absolute `data_split.json` work path as paired identity. Disabled and frozen matched on content hashes, but the campaign root stopped before full. No partial reuse is allowed. | immutable graft v4 source with path canonicalization and independent data-split content SHA; result root `nanogpt_gap_vanilla_graft_smoke_20260716_v4_read_delta` |
| `/data2/ncpl-pathA/work/nanogpt_gap_boundary_remap_20260716_v4` | quarantined pending root-cause repair | Seed42 identity completed with post-run and sequence QC PASS, but affine A failed at update 0 because `affine_collision_equivalence=false`. Seed43/44 were not launched, and the successful identity arm is not reusable as a formal estimate. | Not yet assigned. A new immutable snapshot and new result root are required after an exact diagnostic reproduces and repairs the collision-equivalence failure. |
| `/data2/ncpl-pathA/work/nanogpt_gap_causal_20260715_v1/phase34_v8_qualification_20260716_v1` | excluded infrastructure failure | The first compiled qualification arm stopped before training because system `/` and `/tmp` had zero free bytes. The v8 runner did not itself force all temporary/compiler paths under `/data2`, so the qualification is not reproducible as launched. | New immutable Phase34 snapshot with runner-enforced `/data2` TMP/Triton/Inductor paths, followed by all tests and three CUDA qualification gates. |

## Formal-analysis rule

- Do not pool partial, failed, quarantined, or failed-post-QC arms.
- Do not reuse a successful control arm from an excluded campaign root unless a new qualification explicitly proves cross-root identity and the analysis contract permits it. The current campaign does not grant such an exception.
- Every replacement receives a new source hash, new result root, fresh run manifest, and post-run QC.
- `EXCLUDED_DO_NOT_USE.json` is create-if-absent; existing markers must never be overwritten.
