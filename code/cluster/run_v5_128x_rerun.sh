#!/usr/bin/env bash
# Rerun all current-standard v5 experiments at table LR scale 128x
# (2026-08-29 standard switch). New run IDs use a `_128x` suffix so they do
# not overwrite the historical 2x evidence.
#
# Families rerun at 128x:
#   M2 injection        nglab1x_{input,y,v,nogram}_v5_128x_freq10      (2000 steps)
#   M5 dose             nglab{d}_input_v5_128x_freq10                   (2000 steps, 11 doses)
#   Causal              causalv5c_{...}_128x                            (1000 steps, 6 arms)
#   X2 row width        ctbl_dim{192,48,12,768}_input_v5_128x           (1000 steps)
#   X1 optimizer        optv5c_{rms,adamw,sgd}_s128x                    (1000 steps)
#
# Usage: GROUP=<m2|dose|causal|x2|x1|maskhigh|net|trim|lep|ffq|mix|all|m2+causal|...> NGLAB_PY=python3 bash run_v5_128x_rerun.sh <gpu> [<gpu> ...]
# Each GPU owns a disjoint slice of the selected group's spec list
# (create-only: skips existing). GROUP may be a '+' separated list of groups.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"
export NGLAB_TABLE_LR_SCALE="${NGLAB_TABLE_LR_SCALE:-128.0}"
GROUP="${GROUP:-all}"
GPUS=("$@")

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "usage: GROUP=<m2|dose|causal|x2|x1|all> $0 <gpu> [<gpu> ...]" >&2
  exit 2
fi

VAL="2,3,4,5,6,7,8,9,10,6542"

M2=(
  "nglab1x_input_v5_128x_freq10|1|${VAL}|2000|"
  "nglab1x_y_v5_128x_freq10|1|${VAL}|2000|--injection_position y"
  "nglab1x_v_v5_128x_freq10|1|${VAL}|2000|--injection_position v"
  "nglab1x_nogram_v5_128x_freq10|1|${VAL}|2000|--enable_bigram 0 --enable_trigram 0"
)
DOSE=(
  "nglab0_25x_input_v5_128x_freq10|62|${VAL}|2000|"
  "nglab0_5x_input_v5_128x_freq10|60|${VAL}|2000|"
  "nglab0_75x_input_v5_128x_freq10|63|${VAL}|2000|"
  "nglab1_5x_input_v5_128x_freq10|1,61|3,4,5,6,7,8,9,10,6542|2000|"
  "nglab2x_input_v5_128x_freq10|1,2|3,4,5,6,7,8,9,10,6542|2000|"
  "nglab2_5x_input_v5_128x_freq10|1,2,64|4,5,6,7,8,9,10,6542|2000|"
  "nglab3x_input_v5_128x_freq10|1,2,3|4,5,6,7,8,9,10,6542|2000|"
  "nglab4x_input_v5_128x_freq10|1,2,3,4|5,6,7,8,9,10,6542|2000|"
  "nglab5x_input_v5_128x_freq10|1,2,3,4,5|6,7,8,9,10,6542|2000|"
  "nglab6x_input_v5_128x_freq10|1,2,3,4,5,6|7,8,9,10,6542|2000|"
  "nglab8x_input_v5_128x_freq10|1,2,3,4,5,6,7,8|9,10,6542|2000|"
)
CAUSAL=(
  "causalv5c_none_128x|1|${VAL}|1000|"
  "causalv5c_freeze_table_e1_128x|1|${VAL}|1000|--intervention freeze_table --intervention_epoch 1"
  "causalv5c_freeze_backbone_e1_128x|1|${VAL}|1000|--intervention freeze_backbone --intervention_epoch 1"
  "causalv5c_hash_reseed_e1_128x|1|${VAL}|1000|--intervention hash_reseed --intervention_epoch 1"
  "causalv5c_mask_low_f200_e1_128x|1|${VAL}|1000|--intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 200"
  "causalv5c_mask_high_f200_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 200"
)
X2=(
  "ctbl_dim768_input_v5_128x|1|${VAL}|1000|"
  "ctbl_dim192_input_v5_128x|1|${VAL}|1000|--bigram_table_dim 192 --trigram_table_dim 192"
  "ctbl_dim48_input_v5_128x|1|${VAL}|1000|--bigram_table_dim 48 --trigram_table_dim 48"
  "ctbl_dim12_input_v5_128x|1|${VAL}|1000|--bigram_table_dim 12 --trigram_table_dim 12"
)
X1=(
  "optv5c_rms_s128x|1|${VAL}|1000|"
  "optv5c_adamw_s128x|1|${VAL}|1000|--table_optimizer adamw --table_betas 0.0,0.99"
  "optv5c_sgd_m0_s128x|1|${VAL}|1000|--table_optimizer sgd --table_betas 0.0,0.99"
)
# mask_high threshold scan (epoch 2 boundary): masks f >= thr (inclusive),
# threshold high→low.
# The f=200 point already exists as causalv5c_mask_high_f200_e1_128x and is reused
# in the curve (not relaunched here).
MASKHIGH=(
  "causalv5m_mask_high_t12800_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 12800"
  "causalv5m_mask_high_t6400_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 6400"
  "causalv5m_mask_high_t3200_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 3200"
  "causalv5m_mask_high_t1600_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 1600"
  "causalv5m_mask_high_t800_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 800"
  "causalv5m_mask_high_t400_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 400"
  "causalv5m_mask_high_t100_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 100"
  "causalv5m_mask_high_t50_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 50"
  "causalv5m_mask_high_t25_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 25"
  "causalv5m_mask_high_t10_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 10"
  "causalv5m_mask_high_t5_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 5"
  "causalv5m_mask_high_t2_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 2"
  "causalv5m_mask_high_t1_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 1"
  # t=0 is the explicit full-context boundary, including novel contexts.
  # It uses a new ID because the old t0 output was generated by the bugged
  # seen-context-only implementation.
  "causalv5m_mask_high_t0_full_e1_128x|1|${VAL}|1000|--intervention mask_high_freq --intervention_epoch 1 --intervention_freq_threshold 0"
)

# Net-benefit batch (2026-08-31): does a constrained n-gram table beat nogram
# on VAL loss (not gap)? 2000 steps = 6 epochs of shard 1, val/freq=10.
#   *_reseed_eall: hash reseed at every epoch boundary (0-indexed due epochs
#     1..5 = start of epochs 2..6); inference uses the last table.
#   *_masklowf8_e0: mask f<=8 (+novel) from step 0; the table only ever learns
#     high-frequency contexts. Works for input/y/v (mask extended to the
#     per-layer y/v path in train.py, 2026-08-31).
NET=(
  "netv5_input_reseed_eall_128x|1|${VAL}|2000|--intervention hash_reseed --intervention_epoch 1 --intervention_epochs 2,3,4,5"
  "netv5_y_reseed_eall_128x|1|${VAL}|2000|--injection_position y --intervention hash_reseed --intervention_epoch 1 --intervention_epochs 2,3,4,5"
  "netv5_v_reseed_eall_128x|1|${VAL}|2000|--injection_position v --intervention hash_reseed --intervention_epoch 1 --intervention_epochs 2,3,4,5"
  "netv5_input_masklowf8_e0_128x|1|${VAL}|2000|--intervention mask_low_freq --intervention_epoch 0 --intervention_freq_threshold 8 --intervention_low_inclusive 1"
  "netv5_y_masklowf8_e0_128x|1|${VAL}|2000|--injection_position y --intervention mask_low_freq --intervention_epoch 0 --intervention_freq_threshold 8 --intervention_low_inclusive 1"
  "netv5_v_masklowf8_e0_128x|1|${VAL}|2000|--injection_position v --intervention mask_low_freq --intervention_epoch 0 --intervention_freq_threshold 8 --intervention_low_inclusive 1"
)

# Trigram-only mask-low sweep (2026-08-31, user): separate the branches.
# Single trigram table (bigram disabled), 1000 steps, val/freq=10, mask from
# epoch 2 (due epoch 1, inclusive f<=t) -- comparable to the joint causalv5m
# sweep. Control included so removal % is measured against a tri-only
# baseline with identical eval cadence.
TRI_FLAGS="--enable_bigram 0 --enable_trigram 1 --bigram_clean_table 0 --trigram_clean_table 1048576"
TRIM=(
  "s1v5_128_tri_masklow_ctl|1|${VAL}|1000|${TRI_FLAGS}"
  "s1v5_128_tri_masklowf1_e1|1|${VAL}|1000|${TRI_FLAGS} --intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 1 --intervention_low_inclusive 1"
  "s1v5_128_tri_masklowf2_e1|1|${VAL}|1000|${TRI_FLAGS} --intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 2 --intervention_low_inclusive 1"
  "s1v5_128_tri_masklowf4_e1|1|${VAL}|1000|${TRI_FLAGS} --intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 4 --intervention_low_inclusive 1"
  "s1v5_128_tri_masklowf8_e1|1|${VAL}|1000|${TRI_FLAGS} --intervention mask_low_freq --intervention_epoch 1 --intervention_freq_threshold 8 --intervention_low_inclusive 1"
)

# Last-epoch-only table (2026-08-31, user): readout fully masked for epochs
# 1-5 (table gets no gradient, backbone sees no table), unmasked at the start
# of epoch 6 (step 1686). 2000 steps, val/freq=10. Tests whether a table
# trained only in the final epoch beats reseed-every-epoch (§43).
LEP=(
  "netv5_input_lastep_128x|1|${VAL}|2000|--intervention readout_last_epoch --intervention_epoch 5 --intervention_epochs 0"
  "netv5_y_lastep_128x|1|${VAL}|2000|--injection_position y --intervention readout_last_epoch --intervention_epoch 5 --intervention_epochs 0"
  "netv5_v_lastep_128x|1|${VAL}|2000|--injection_position v --intervention readout_last_epoch --intervention_epoch 5 --intervention_epochs 0"
)

# Freeze four-factor long-range batch (2026-09-01): control / freeze_table /
# freeze_backbone / freeze_both from the SAME e2 boundary, 10 full epochs
# (3370 steps). Separates table-environment vs backbone-active contributions
# over the full long-run window. Controls reuse the existing
# s1v5_128_ep1xL4_10ep_{both,nogram} runs (identical contract, no
# intervention). e2 = 0-indexed epoch 2 = step 675 (§38 semantics).
FFQ_VALSTEPS="337,674,1011,1348,1685,2022,2359,2696,3033,3370"
FFQ=(
  "ffqv5_freeze_table_e2_10ep|1|${VAL}|3370|--epoch_batches 337 --intervention freeze_table --intervention_epoch 2 --val_steps ${FFQ_VALSTEPS}"
  "ffqv5_freeze_backbone_e2_10ep|1|${VAL}|3370|--epoch_batches 337 --intervention freeze_backbone --intervention_epoch 2 --val_steps ${FFQ_VALSTEPS}"
  "ffqv5_freeze_both_e2_10ep|1|${VAL}|3370|--epoch_batches 337 --intervention freeze_both --intervention_epoch 2 --val_steps ${FFQ_VALSTEPS}"
)

# Shuffle experiment (2026-09-01, §38 planned): same shard-1 tokens x3 passes.
#   replay arm: standard fixed-order epoch replay (3 epochs of 337 batches).
#   mixed arm:  --replay_mix_passes 3 consumes ONE globally chunk-shuffled
#               stream (chunk-level permutation preserves n-gram continuity)
#               with NO epoch structure. H-DILUTE predicts the same endpoint
#               gap and no epoch teeth; a significant endpoint shift means the
#               model is missing a repeat-interval variable.
MIX=(
  "mixv5_tri_replay_3pass|1|${VAL}|1011|${TRI_FLAGS} --epoch_batches 337 --val_steps 337,674,1011"
  "mixv5_tri_mixed_3pass|1|${VAL}|1011|${TRI_FLAGS} --epoch_batches 337 --replay_mix_passes 3 --val_steps 337,674,1011"
)

case "$GROUP" in
  all) SPECS=("${M2[@]}" "${DOSE[@]}" "${CAUSAL[@]}" "${X2[@]}" "${X1[@]}" "${MASKHIGH[@]}" "${NET[@]}" "${TRIM[@]}" "${LEP[@]}" "${FFQ[@]}" "${MIX[@]}") ;;
  *)
    SPECS=()
    IFS='+' read -ra PARTS <<< "$GROUP"
    for part in "${PARTS[@]}"; do
      case "$part" in
        m2) SPECS+=("${M2[@]}") ;;
        dose) SPECS+=("${DOSE[@]}") ;;
        causal) SPECS+=("${CAUSAL[@]}") ;;
        x2) SPECS+=("${X2[@]}") ;;
        x1) SPECS+=("${X1[@]}") ;;
        maskhigh) SPECS+=("${MASKHIGH[@]}") ;;
        net) SPECS+=("${NET[@]}") ;;
        trim) SPECS+=("${TRIM[@]}") ;;
        lep) SPECS+=("${LEP[@]}") ;;
        ffq) SPECS+=("${FFQ[@]}") ;;
        mix) SPECS+=("${MIX[@]}") ;;
        *) echo "unknown GROUP part=$part (m2|dose|causal|x2|x1|maskhigh|net|trim|lep|ffq|mix|all)" >&2; exit 2 ;;
      esac
    done
    [[ "${#SPECS[@]}" -gt 0 ]] || { echo "empty GROUP=$GROUP" >&2; exit 2; }
    ;;
esac

# Optional disjoint slice of the chosen group, e.g. NGLAB_SPEC_SLICE="0:8"
# runs SPECS[0..7]. Used to split one group across machines without overlap.
if [[ -n "${NGLAB_SPEC_SLICE:-}" ]]; then
  slice_start="${NGLAB_SPEC_SLICE%%:*}"
  slice_end="${NGLAB_SPEC_SLICE##*:}"
  SPECS=("${SPECS[@]:$slice_start:$((slice_end - slice_start))}")
fi

launch() {
  local gpu="$1"
  local idx="$2"
  local total="$3"
  # round-robin partition: each GPU owns specs idx, idx+total, idx+2*total, ...
  local i
  for ((i = idx; i < ${#SPECS[@]}; i += total)); do
    local spec="${SPECS[$i]}"
    local run_id train val steps extra
    IFS='|' read -r run_id train val steps extra <<< "$spec"
    if [[ -e "$OUT_DIR/${run_id}_fixed/summary.json" ]]; then
      echo "[gpu $gpu] skip existing: $run_id"
      continue
    fi
    echo "[gpu $gpu] launch: $run_id ($steps steps, train=$train)"
    if ! bash "$SCRIPT_DIR/run_v5_clean.sh" "$gpu" "$run_id" "$train" "$val" "$steps" $extra; then
      echo "[gpu $gpu] FAILED: $run_id" >&2
    fi
  done
}

NGPU="${#GPUS[@]}"
for idx in "${!GPUS[@]}"; do
  launch "${GPUS[$idx]}" "$idx" "$NGPU" &
done
wait
echo "[done] all GPU queues finished"
