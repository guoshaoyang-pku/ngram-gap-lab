#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
GROUP="${V5_GROUP:?set V5_GROUP to inj, dose, epoch, causal, rho, or long}"
GPUS=("$@")

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "usage: V5_GROUP=<group> NGLAB_PY=python3 $0 <gpu> [gpu...]" >&2
  exit 2
fi

case "$GROUP" in
  inj)
    SPECS=(
      "nglab1x_input_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab1x_y_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|--injection_position y"
      "nglab1x_v_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|--injection_position v"
      "nglab1x_nogram_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|--enable_bigram 0 --enable_trigram 0"
    )
    ;;
  dose)
    SPECS=(
      "nglab0_25x_input_v5|62|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab0_5x_input_v5|60|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab0_75x_input_v5|63|2,3,4,5,6,7,8,9,10,6542|2000|"
      "nglab1_5x_input_v5|1,61|3,4,5,6,7,8,9,10,6542|2000|"
      "nglab2x_input_v5|1,2|3,4,5,6,7,8,9,10,6542|2000|"
      "nglab2_5x_input_v5|1,2,64|4,5,6,7,8,9,10,6542|2000|"
      "nglab3x_input_v5|1,2,3|4,5,6,7,8,9,10,6542|2000|"
      "nglab4x_input_v5|1,2,3,4|5,6,7,8,9,10,6542|2000|"
      "nglab5x_input_v5|1,2,3,4,5|6,7,8,9,10,6542|2000|"
      "nglab6x_input_v5|1,2,3,4,5,6|7,8,9,10,6542|2000|"
      "nglab8x_input_v5|1,2,3,4,5,6,7,8|9,10,6542|2000|"
    )
    ;;
  epoch)
    SPECS=(
      "nglab0_25x_e5_v5|62|2,3,4,5,6,7,8,9,10,6542|420|"
      "nglab0_5x_e5_v5|60|2,3,4,5,6,7,8,9,10,6542|840|"
      "nglab0_75x_e5_v5|63|2,3,4,5,6,7,8,9,10,6542|1260|"
      "nglab1x_e5_v5|1|2,3,4,5,6,7,8,9,10,6542|1685|"
      "nglab1_5x_e5_v5|1,61|3,4,5,6,7,8,9,10,6542|2525|"
      "nglab2x_e5_v5|1,2|3,4,5,6,7,8,9,10,6542|3350|"
      "nglab2_5x_e5_v5|1,2,64|4,5,6,7,8,9,10,6542|4190|"
      "nglab3x_e5_v5|1,2,3|4,5,6,7,8,9,10,6542|5000|"
      "nglab4x_e5_v5|1,2,3,4|5,6,7,8,9,10,6542|6700|"
    )
    ;;
  causal)
    SPECS=(
      "nglab1x_reset_e1_v5|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention reset_table --intervention_epoch 1"
      "nglab1x_reset_e2_v5|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention reset_table --intervention_epoch 2"
      "nglab1x_mask_e1_v5|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention mask_readout --intervention_epoch 1"
      "nglab1x_freeze_table_e1_v5|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention freeze_table --intervention_epoch 1"
      "nglab1x_freeze_backbone_e1_v5|1|2,3,4,5,6,7,8,9,10,6542|1000|--intervention freeze_backbone --intervention_epoch 1"
    )
    ;;
  rho)
    SPECS=(
      "nglab1x_input_rho_v5|1|2,3,4,5,6,7,8,9,10,6542|2000|--fixed_train_probe 4 --probe_eval_interval 10"
      "nglab2x_input_rho_v5|1,2|3,4,5,6,7,8,9,10,6542|2000|--fixed_train_probe 4 --probe_eval_interval 10"
    )
    ;;
  long)
    SPECS=(
      "nglab1x_nogram_long_v5|1|2,3,4,5,6,7,8,9,10,6542|8000|--enable_bigram 0 --enable_trigram 0"
    )
    ;;
  *)
    echo "unknown V5_GROUP=$GROUP" >&2
    exit 2
    ;;
esac

run_one() {
  local gpu="$1"
  local spec="$2"
  local run_id train_shards val_shards steps extra result_dir
  IFS='|' read -r run_id train_shards val_shards steps extra <<< "$spec"
  result_dir="$ROOT/data/runs_fixed/${run_id}_fixed"
  if [[ -f "$result_dir/summary.json" ]]; then
    echo "[v5-$GROUP] skip complete $run_id"
    return 0
  fi
  [[ ! -e "$result_dir" ]] || {
    echo "[v5-$GROUP] refusing partial directory $result_dir" >&2
    return 2
  }
  echo "[v5-$GROUP] $run_id gpu=$gpu steps=$steps"
  NGLAB_PY="$PY" bash "$SCRIPT_DIR/run_v5_clean.sh" "$gpu" "$run_id" \
    "$train_shards" "$val_shards" "$steps" $extra
}

active=0
slot=0
gpu_count="${#GPUS[@]}"
for spec in "${SPECS[@]}"; do
  while [[ "$active" -ge "$gpu_count" ]]; do
    wait -n
    active=$((active - 1))
  done
  run_one "${GPUS[$slot]}" "$spec" &
  active=$((active + 1))
  slot=$(( (slot + 1) % gpu_count ))
done
wait
echo "[v5-$GROUP] complete"