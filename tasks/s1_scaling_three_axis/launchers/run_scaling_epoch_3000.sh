#!/usr/bin/env bash
# Epoch-length scaling @ 3000 steps (fixed-step) — online-only extension.
#
# Rationale (user 2026-08-24): the fixed train probe (first 4 train batches,
# re-read every replay) creates sawtooth artifacts at epoch boundaries. It is
# DEPRECATED.  These runs log ONLY the online train/val gap (train_log.jsonl,
# val_interval=50) to inspect the long-run saturation of the gap with epoch
# count.
#
# Runs: L1..L4 x {bigram, trigram, both, no-ngram}, fixed-step alignment,
#       3000 steps, step-anchored LR, table RMSProp(0.0,0.99) lr_scale 2.0,
#       bf16, no compile, NO fixed train probe (--fixed_train_probe 0).
#
# Usage: ./run_scaling_epoch_3000.sh <gpu_id...>   (default: 0 1 2 3)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$(command -v python3)}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then GPUS=(0 1 2 3 4 5 6 7); fi
mkdir -p "$OUT_DIR"

declare -A EPB=( [L1]=42 [L2]=84 [L3]=168 [L4]=337 )
STEPS=3000

run_arm() {  # run_arm <gpu> <run_id> <epoch_batches> <bigram> <trigram>
  local GPU="$1" RUN_ID="$2" EPB="$3" BI="$4" TRI="$5"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  if [ -f "$RESULT_DIR/summary.json" ]; then
    echo "[epoch-3000] SKIP $RUN_ID (summary.json already present)"
    return 0
  fi
  mkdir -p "$RESULT_DIR"
  echo "[epoch-3000] $RUN_ID epb=$EPB steps=$STEPS bi=$BI tri=$TRI -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps "$STEPS" --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 50 --val_batches 4 --table_norm_interval 50 --lr 0.004 \
    --enable_unigram 0 --enable_bigram "$BI" --enable_trigram "$TRI" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --epoch_batches "$EPB" \
    --fixed_train_probe 0 \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    --dtype bf16 \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[epoch-3000] $RUN_ID done (exit=$?) at $(date)"
}

NGPU=${#GPUS[@]}
ACTIVE=0
SLOT=0
for L in L1 L2 L3 L4; do
  for ARM in bigram trigram both nogram; do
    case "$ARM" in
      bigram) BI=1 TRI=0 ;;
      trigram) BI=0 TRI=1 ;;
      both) BI=1 TRI=1 ;;
      nogram) BI=0 TRI=0 ;;
    esac
    while [ "$ACTIVE" -ge "$NGPU" ]; do
      wait -n
      ACTIVE=$((ACTIVE - 1))
    done
    GPU="${GPUS[$SLOT]}"
    SLOT=$(( (SLOT + 1) % NGPU ))
    run_arm "$GPU" "ep3000_${L}_${ARM}_fs" "${EPB[$L]}" "$BI" "$TRI" &
    ACTIVE=$((ACTIVE + 1))
  done
  while [ "$ACTIVE" -gt 0 ]; do
    wait -n
    ACTIVE=$((ACTIVE - 1))
  done
  SLOT=0
done

echo "=== epoch 3000 grid done at $(date) ==="
