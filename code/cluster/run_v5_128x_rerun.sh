#!/usr/bin/env bash
# Rerun all current-standard v5 experiments at table LR scale 128x
# (2026-08-29 standard switch). New run IDs use a `_128x` suffix so they do
# not overwrite the historical 2x evidence.
#
# Families rerun at 128x:
#   M2 injection        nglab1x_{input,y,v,nogram}_v5_128x_freq10      (2000 steps)
#   M5 dose             nglab{d}_input_v5_128x_freq10                   (2000 steps, 11 doses)
#   Causal              causalv5c_{...}_128x                            (1000 steps, 9 arms)
#   X2 row width        ctbl_dim{192,48,12,768}_input_v5_128x           (1000 steps)
#   X1 optimizer        optv5c_{rms,adamw,sgd}_s128x                    (1000 steps)
#
# Usage: GROUP=<m2|dose|causal|x2|x1|all|m2+causal|...> NGLAB_PY=python3 bash run_v5_128x_rerun.sh <gpu> [<gpu> ...]
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
  "causalv5c_reset_table_e1_128x|1|${VAL}|1000|--intervention reset_table --intervention_epoch 1"
  "causalv5c_reset_table_e2_128x|1|${VAL}|1000|--intervention reset_table --intervention_epoch 2"
  "causalv5c_mask_readout_e1_128x|1|${VAL}|1000|--intervention mask_readout --intervention_epoch 1"
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

case "$GROUP" in
  all) SPECS=("${M2[@]}" "${DOSE[@]}" "${CAUSAL[@]}" "${X2[@]}" "${X1[@]}") ;;
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
        *) echo "unknown GROUP part=$part (m2|dose|causal|x2|x1|all)" >&2; exit 2 ;;
      esac
    done
    [[ "${#SPECS[@]}" -gt 0 ]] || { echo "empty GROUP=$GROUP" >&2; exit 2; }
    ;;
esac

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
