#!/usr/bin/env bash
# Run the clean V5 optimizer/LR curve grid on supplied GPU slots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
GPUS=("$@")

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "usage: NGLAB_PY=python3 $0 <gpu> [gpu...]" >&2
  exit 2
fi

run_one() {
  local gpu="$1"
  local run_id="$2"
  shift 2
  local result="$ROOT/data/runs_fixed/${run_id}_fixed"
  if [[ -f "$result/summary.json" ]]; then
    echo "[optv5] skip complete $run_id"
    return 0
  fi
  if [[ -e "$result" ]]; then
    echo "[optv5] refusing partial directory $result" >&2
    return 2
  fi
  echo "[optv5] $run_id on GPU $gpu"
  NGLAB_PY="$PY" bash "$SCRIPT_DIR/run_v5_clean.sh" "$gpu" "$run_id" \
    1 2,3,4,5,6,7,8,9,10,6542 1000 "$@"
}

launch_slot() {
  while [[ "$ACTIVE" -ge "$NGPU" ]]; do
    wait "${PIDS[0]}"
    PIDS=("${PIDS[@]:1}")
    ACTIVE=$((ACTIVE - 1))
  done
  local gpu="${GPUS[$SLOT]}"
  SLOT=$(( (SLOT + 1) % NGPU ))
  ACTIVE=$((ACTIVE + 1))
  run_one "$gpu" "$@" &
  PIDS+=("$!")
}

NGPU="${#GPUS[@]}"
ACTIVE=0
SLOT=0
PIDS=()

launch_slot optv5c_rms_b099_s0p5 --table_lr_scale 0.5
launch_slot optv5c_rms_b099_s1p0 --table_lr_scale 1.0
launch_slot optv5c_rms_b099_s2p0 --table_lr_scale 2.0
launch_slot optv5c_rms_b099_s3p0 --table_lr_scale 3.0
launch_slot optv5c_rms_b099_s4p0 --table_lr_scale 4.0
launch_slot optv5c_rms_b095_s2p0 --table_betas 0.0,0.95
launch_slot optv5c_rms_b098_s2p0 --table_betas 0.0,0.98
launch_slot optv5c_rms_b0995_s2p0 --table_betas 0.0,0.995
launch_slot optv5c_rms_b0999_s2p0 --table_betas 0.0,0.999
launch_slot optv5c_adamw_b099_s2p0 --table_optimizer adamw --table_betas 0.0,0.99
launch_slot optv5c_sgd_m0_s2p0 --table_optimizer sgd --table_betas 0.0,0.99

while [[ "$ACTIVE" -gt 0 ]]; do
  wait "${PIDS[0]}"
  PIDS=("${PIDS[@]:1}")
  ACTIVE=$((ACTIVE - 1))
done
echo "[optv5] sweep complete"