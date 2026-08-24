#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-1,2,3}"
STEPS="${2:-2000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
FREQ_INDEX="$ROOT/data/freq_index_train2x.npz"
TRAIN_SHARDS=1,2
VAL_SHARDS=3,4,5,6,7,8,9,10,6542

IFS=',' read -r -a GPUS <<< "$GPU_LIST"

run_one() {
  local GPU="$1"
  local EXP="$2"
  local INJ="$3"
  local RESULT_DIR="$ROOT/data/runs_fixed/${EXP}_fixed"
  mkdir -p "$RESULT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "${EXP}_fixed" \
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
    --freq_index "$FREQ_INDEX" \
    --freq_eval_interval 10 \
    --table_optimizer rmsprop \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    --freq_eval_batches 4 \
    > "$RESULT_DIR/train.log" 2>&1
}

run_one "${GPUS[0]}" "nglab2x_v" "v" &
PIDS=("$!")
run_one "${GPUS[1]}" "nglab2x_y" "y" &
PIDS+=("$!")
run_one "${GPUS[2]}" "nglab2x_input" "input" &
PIDS+=("$!")

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
exit "$status"