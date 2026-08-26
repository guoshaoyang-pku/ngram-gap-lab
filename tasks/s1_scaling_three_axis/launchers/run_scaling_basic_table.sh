#!/usr/bin/env bash
# Basic-scaling table axis (user decision 2026-08-24): LR x2 + bf16, no compile.
#
# Goal: quickly re-derive table-size scaling under the NEW standard
# (table_lr_scale=2.0, beta2=0.99, bf16, no compile) before the full grid.
# This launcher only spawns the ANCHOR points:
#   table axis : 1M & 16K logical x bigram-only @ L4 (1k steps)
#                + 1M trigram-only @ L4 (1k steps)
#
# Everything else is frozen: vanilla nanoGPT 8L/6H/768D, input injection,
# natural corpus nested-prefix shard 1, train/val zero overlap,
# table RMSProp(0.0,0.99), backbone AdamW(0.8,0.95), online gap +
# exact-frequency + freq-bin diagnostics.
#
# Usage: ./run_scaling_basic_table.sh [gpu1] [gpu2] [gpu3]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
FREQ_IDX="$ROOT/data/freq_index.npz"
mkdir -p "$OUT_DIR"

# GPU allocation (edit per machine): 3 free cards
G1="${1:-0}" G2="${2:-1}" G3="${3:-2}"

run_arm() {  # run_arm <gpu> <run_id> <table_mult> <bigram> <trigram>
  local GPU="$1" RUN_ID="$2" TM="$3" BI="$4" TRI="$5"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  mkdir -p "$RESULT_DIR"
  echo "[basic-table] $RUN_ID table_mult=$TM bigram=$BI trigram=$TRI -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps 1000 --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 10 --val_batches 4 --table_norm_interval 10 --lr 0.004 \
    --enable_unigram 0 --enable_bigram "$BI" --enable_trigram "$TRI" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --freq_eval_interval 10 --freq_eval_batches 4 \
    --exact_freq_eval_interval 10 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --freq_index "$FREQ_IDX" \
    --epoch_batches 337 \
    --fixed_train_probe 0 \
    --table_betas 0.0,0.99 --table_lr_scale 2.0 \
    --table_mult "$TM" \
    --dtype bf16 \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[basic-table] $RUN_ID done (exit=$?) at $(date)"
}

# Anchor points (L4, fixed-step 1k steps)
run_arm "$G1" basic_tbl_1M_bigram   64 1 0 &
run_arm "$G2" basic_tbl_16K_bigram   1 1 0 &
run_arm "$G3" basic_tbl_1M_trigram  64 0 1 &
wait

echo "=== basic table anchors done at $(date) ==="
