#!/usr/bin/env bash
# v10 standard: run v/y/input/nogram ablation in PARALLEL on ophis-gpu.
# Validation + freq-bin eval every 10 steps, 2000 steps.
# Usage: ./run_injpos_parallel.sh [gpu_list] [steps]
set -euo pipefail

GPU_LIST="${1:-0,5,6,7}"
STEPS="${2:-2000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
TRAIN_SHARDS=1
VAL_SHARDS=2,3,4,5,6,7,8,9,10,6542

IFS=',' read -r -a GPUS <<< "$GPU_LIST"

run_one() {
  local GPU="$1"; local EXP="$2"; local INJ="$3"; shift 3
  local RESULT_DIR="$ROOT/data/runs_fixed/${EXP}_fixed"
  mkdir -p "$RESULT_DIR"
  echo "[launch] $EXP on GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "$EXP" \
    --injection_position "$INJ" \
    --steps "$STEPS" \
    --seed 42 \
    --data_dir "$DATA_DIR" \
    --out_dir "$ROOT/data/runs_fixed" \
    --train_shards "$TRAIN_SHARDS" \
    --val_shards "$VAL_SHARDS" \
    --device_batch_size 72 \
    --total_batch_size 147456 \
    --val_interval 10 \
    --val_batches 4 \
    --table_norm_interval 10 \
    --lr 0.004 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index "$ROOT/data/freq_index.npz" \
    --table_optimizer rmsprop \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    "$@" \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[launch] $EXP exit=$? at $(date)"
}

run_one "${GPUS[0]}" "nglab1x_v10_v" "v" &
PIDS=("$!")
run_one "${GPUS[1]}" "nglab1x_v10_y" "y" &
PIDS+=("$!")
run_one "${GPUS[2]}" "nglab1x_v10_input" "input" &
PIDS+=("$!")
run_one "${GPUS[3]}" "nglab1x_v10_nogram" "input" --enable_unigram 0 --enable_bigram 0 --enable_trigram 0 &
PIDS+=("$!")

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
