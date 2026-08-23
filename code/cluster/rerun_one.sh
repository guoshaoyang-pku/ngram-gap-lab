#!/usr/bin/env bash
# Rerun all experiments with fixed train.py.
# This script runs ONE experiment on a given GPU, then exits.
# A wrapper (run_gpu_queue.sh) chains multiple experiments per GPU.
#
# Usage: run_one.sh <gpu_id> <run_id> <train_shards> <val_shards> <steps> <freq_index> [extra args...]
set -euo pipefail

GPU="$1"; shift
RUN_ID="$1"; shift
TRAIN_SHARDS="$1"; shift
VAL_SHARDS="$1"; shift
STEPS="$1"; shift
FREQ_INDEX="$1"; shift
EXTRA_ARGS="$@"

ROOT="/data/home/guoshaoyang/ngram-gap-lab"
PY="python3"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs"
RESULT_DIR="$OUT_DIR/$RUN_ID"
LOG_FILE="$RESULT_DIR/train.log"

mkdir -p "$RESULT_DIR"

echo "[$(date)] START $RUN_ID on GPU $GPU (steps=$STEPS shards=$TRAIN_SHARDS)"

CUDA_VISIBLE_DEVICES="$GPU" $PY -u "$ROOT/code/train.py" \
    --run_id "$RUN_ID" \
    --injection_position input \
    --steps "$STEPS" \
    --seed 42 \
    --data_dir "$DATA_DIR" \
    --out_dir "$OUT_DIR" \
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
    --freq_eval_batches 4 \
    $EXTRA_ARGS \
    > "$LOG_FILE" 2>&1

RC=$?
echo "[$(date)] DONE $RUN_ID exit=$RC"
exit $RC
