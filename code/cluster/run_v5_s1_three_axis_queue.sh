#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LAUNCHER="$SCRIPT_DIR/run_v5_s1_three_axis.sh"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_scaling}"
QUEUE_LOG="${S1_QUEUE_LOG:-$OUT_DIR/s1v5_128_final_queue.log}"
GPUS=(0 1 2 3 4 5 6 7)

run_group() {
  local group="$1"
  echo "[s1-v5-128-queue] start group=$group $(date)"
  if ! S1_GROUP="$group" NGLAB_OUT_DIR="$OUT_DIR" NGLAB_PY="${NGLAB_PY:-python3}" \
    bash "$LAUNCHER" "${GPUS[@]}"; then
    echo "[s1-v5-128-queue] group failed=$group; continuing" >&2
  fi
  echo "[s1-v5-128-queue] finish group=$group $(date)"
}

run_group table_size_bi
run_group table_size_tri
run_group frequency_main
run_group epoch_length
echo "[s1-v5-128-queue] all groups complete $(date)"