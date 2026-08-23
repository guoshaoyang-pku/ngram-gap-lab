#!/usr/bin/env bash
# Launch v/y/input injection-point ablation on ophis-gpu.
# Runs 4 experiments SERIALLY on one GPU (avoids timestamp collisions).
# v10 standard: validation + freq-bin eval every 10 steps, 2000 steps.
# Usage: ./run_injpos.sh [gpu_id] [steps]
set -euo pipefail

GPU="${1:-0}"
STEPS="${2:-2000}"

ROOT=/data3/guoshaoyang/ngram-gap-lab
PY="$ROOT/.venv/bin/python"
DATA_DIR="$ROOT/data/tokenized"
TRAIN_SHARDS=1
VAL_SHARDS=2,3,4,5,6,7,8,9,10,6542

run_one() {
  local EXP="$1"; local INJ="$2"; shift 2
  local RESULT_DIR="$ROOT/data/runs/$EXP"
  mkdir -p "$RESULT_DIR"
  echo ""
  echo "=========================================="
  echo "=== $EXP (injection=$INJ, GPU $GPU) ==="
  echo "=== $(date) ==="
  echo "=========================================="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "$EXP" \
    --injection_position "$INJ" \
    --steps "$STEPS" \
    --seed 42 \
    --data_dir "$DATA_DIR" \
    --out_dir "$ROOT/data/runs" \
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
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    "$@" \
    > "$RESULT_DIR/train.log" 2>&1
  local RC=$?
  echo "=== $EXP finished (exit $RC) at $(date) ==="
  return $RC
}

run_one "nglab1x_v10_v" "v"
run_one "nglab1x_v10_y" "y"
run_one "nglab1x_v10_input" "input"
run_one "nglab1x_v10_nogram" "input" --enable_unigram 0 --enable_bigram 0 --enable_trigram 0

echo ""
echo "=========================================="
echo "=== All 4 runs done at $(date) ==="
echo "=========================================="
