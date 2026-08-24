#!/usr/bin/env bash
# High-end shard-size asymptote runs on 360-2 (gpu02): 5x / 6x / 8x,
# same v10 fixed-val standard, 2000 steps, seed 42, input injection.
#
# Usage: ./run_shard_sweep_360.sh [gpu5x] [gpu6x] [gpu8x]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_fixed"
G5="${1:-0}" G6="${2:-1}" G8="${3:-2}"

run_one() {  # run_one <gpu> <run_id> <train_shards> <val_shards> <steps> <freq_index>
  local GPU="$1" RUN_ID="$2" SHARDS="$3" VAL="$4" STEPS="$5" FREQ="$6"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  mkdir -p "$RESULT_DIR"
  echo "[sweep360] $RUN_ID (shards=$SHARDS steps=$STEPS) -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input --steps "$STEPS" --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 2.0 \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 10 --val_batches 4 --table_norm_interval 10 --lr 0.004 \
    --enable_unigram 0 --enable_bigram 1 --enable_trigram 1 \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --freq_eval_interval 10 --freq_eval_batches 4 \
    --train_shards "$SHARDS" --val_shards "$VAL" --freq_index "$FREQ" \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[sweep360] $RUN_ID done (exit=$?) at $(date)"
}

run_one "$G5" nglab5x_input_fv "1,2,3,4,5"       "6,7,8,9,10,6542" 2000 "$ROOT/data/freq_index_train5x.npz" &
run_one "$G6" nglab6x_input_fv "1,2,3,4,5,6"     "7,8,9,10,6542"   2000 "$ROOT/data/freq_index_train6x.npz" &
run_one "$G8" nglab8x_input_fv "1,2,3,4,5,6,7,8" "9,10,6542"       2000 "$ROOT/data/freq_index_train8x.npz" &
wait
echo "=== shard sweep (360-2) done at $(date) ==="
