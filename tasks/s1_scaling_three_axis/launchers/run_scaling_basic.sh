#!/usr/bin/env bash
# Basic-scaling gate (user decision 2026-08-24): LR x2 + bf16, no compile.
#
# Goal: quickly re-derive the three axis curves under the NEW standard
# (table_lr_scale=2.0, beta2=0.99, bf16) before the full grid.
# This launcher only spawns the ANCHOR points:
#   epoch axis : L1 & L4 x {both, no-ngram} @ 1k steps (fixed-step)
#   table axis : 1M & 16K logical x bigram-only @ L4 (1k steps)
#
# Everything else is frozen: vanilla nanoGPT 8L/6H/768D, input injection,
# natural corpus nested-prefix shard 1, train/val zero overlap,
# table RMSProp(0.0,0.99), backbone AdamW(0.8,0.95), online gap +
# exact-frequency + freq-bin diagnostics.
#
# Usage: ./run_scaling_basic.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
FREQ_IDX="$ROOT/data/freq_index.npz"
mkdir -p "$OUT_DIR"

# GPU allocation (edit per machine): 4 free cards
G1="${1:-0}" G2="${2:-1}" G3="${3:-2}" G4="${4:-3}"

# L -> batches per epoch (nested prefix of shard 1)
declare -A EPB=( [L1]=42 [L2]=84 [L3]=168 [L4]=337 )

run_arm() {  # run_arm <gpu> <run_id> <epoch_batches> <steps> <bigram> <trigram> [table_mult]
  local GPU="$1" RUN_ID="$2" EPB="$3" STEPS="$4" BI="$5" TRI="$6" TM="${7:-64}"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  mkdir -p "$RESULT_DIR"
  echo "[basic] $RUN_ID epb=$EPB steps=$STEPS bigram=$BI trigram=$TRI table_mult=$TM -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps "$STEPS" --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 10 --val_batches 4 --table_norm_interval 10 --lr 0.004 \
    --enable_unigram 0 --enable_bigram "$BI" --enable_trigram "$TRI" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --freq_eval_interval 10 --freq_eval_batches 4 \
    --exact_freq_eval_interval 10 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --freq_index "$FREQ_IDX" \
    --epoch_batches "$EPB" \
    --fixed_train_probe 0 \
    --table_betas 0.0,0.99 --table_lr_scale 2.0 \
    --table_mult "$TM" \
    --dtype bf16 \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[basic] $RUN_ID done (exit=$?) at $(date)"
}

# Anchor points (fixed-step, 1k steps)
run_arm "$G1" basic_L1_both_fs       42 1000 1 1 64 &
run_arm "$G2" basic_L4_both_fs      337 1000 1 1 64 &
run_arm "$G3" basic_L1_nogram_fs     42 1000 0 0 64 &
run_arm "$G4" basic_L4_nogram_fs    337 1000 0 0 64 &
wait

echo "=== basic epoch anchors done at $(date) ==="
