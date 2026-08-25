#!/usr/bin/env bash
# Bigram dense-fill arms: off-power-of-2 table_mult points (2026-08-25).
#
# Motivation: prove the gap vs log(table size) relation is continuous, not a
# power-of-2 sampling artifact; and resolve the mult=128 outlier (jamming
# critical fluctuation vs hash artifact) by bridging 64-128 and 128-256 and
# densifying the sawtooth region 40-64.
#
# Points: 44 52 60 (sawtooth fill) | 80 96 112 (64-128 bridge) | 160 192 224
# (128-256 bridge). All bigram-only, sparse (--val_steps 1000), + occupancy.
#
# Usage: ./run_bigram_dense_fill.sh <gpu1> [gpu2] ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
FREQ_IDX="$ROOT/data/freq_index.npz"
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then GPUS=(0 1); fi
mkdir -p "$OUT_DIR"

MULTS=(80 96 112 160 192 224 44 52 60)  # large first: longest runs start early

run_one() {  # run_one <mult> <gpu>
  local TM="$1" GPU="$2"
  local RUN_ID="tbl_${TM}_bigram"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  if [ -f "$RESULT_DIR/summary.json" ]; then
    echo "[fill] SKIP $RUN_ID (summary.json present)"; return 0
  fi
  mkdir -p "$RESULT_DIR"
  echo "[fill] $RUN_ID (mult=$TM) -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps 1000 --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_batches 4 --lr 0.004 \
    --enable_unigram 0 --enable_bigram 1 --enable_trigram 0 \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --freq_index "$FREQ_IDX" \
    --epoch_batches 337 \
    --fixed_train_probe 0 \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    --dtype bf16 \
    --table_mult "$TM" --val_steps 1000 --exact_freq_eval_interval 10 \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[fill] $RUN_ID done (exit=$?) at $(date)"
  "$PY" -u "$ROOT/code/table_occupancy.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --epoch_batches 337 \
    --table_mult "$TM" --out "$RESULT_DIR/table_occupancy.json" \
    > "$RESULT_DIR/occupancy.log" 2>&1 \
    || echo "[fill] occupancy failed mult=$TM"
}

NGPU=${#GPUS[@]}
ACTIVE=0
SLOT=0
launch() {  # launch <cmd...> (gpu appended as LAST argument)
  while [ "$ACTIVE" -ge "$NGPU" ]; do wait -n; ACTIVE=$((ACTIVE - 1)); done
  local GPU="${GPUS[$SLOT]}"
  SLOT=$(( (SLOT + 1) % NGPU ))
  "$@" "$GPU" &
  ACTIVE=$((ACTIVE + 1))
}

for TM in "${MULTS[@]}"; do
  launch run_one "$TM"
done

while [ "$ACTIVE" -gt 0 ]; do wait -n; ACTIVE=$((ACTIVE - 1)); done
echo "=== bigram dense-fill done at $(date) ==="
