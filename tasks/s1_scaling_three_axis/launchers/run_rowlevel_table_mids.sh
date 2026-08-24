#!/usr/bin/env bash
# Row-level table-size intermediate sampling (2026-08-24).
#
# The 36-run formal table grid (mult 64..1) did NOT save final_model.pt, so it
# cannot feed row-level gap analysis.  This launcher re-trains intermediate
# table sizes WITH --save_final_model so we can recover per-row train/val gap
# at several table sizes between the two pilot extremes (tbl_64 / tbl_1).
#
# Design decisions:
#   - MONITOR=sparse (--val_steps 1000) so each run is ~10x faster; row-level
#     analysis only needs the FINAL model + an explicitly enabled diagnostic
#     probe, not the per-step curve.
#   - bigram-only for this pass (the pilot was bigram; keeps cost and analysis
#     simple).  Trigram/both can reuse the same launcher later if wanted.
#   - outputs to data/runs_scaling_rowlevel/ (separate namespace, does NOT
#     touch the formal data/runs_scaling runs).
#   - epoch_batches 337 (L4, full shard 1), seed 42, beta2=0.99, lr_scale 2.0:
#     identical to the formal grid so the row-level gap is comparable.
#
# Usage: ./run_rowlevel_table_mids.sh <gpu_id1> [gpu_id2] ...
# Optional env: TABLE_MULTS="32 16 8 4 2"  (default middle five)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling_rowlevel"
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then GPUS=(0 1 2); fi
mkdir -p "$OUT_DIR"

TABLE_MULTS="${TABLE_MULTS:-32 16 8 4 2}"

CACHE_ROOT="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.inductor_cache}"

run_arm() {  # run_one <gpu> <run_id> <table_mult>
  local GPU="$1" RUN_ID="$2" TM="$3"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_row"
  if [ -f "$RESULT_DIR/final_model.pt" ] && [ -f "$RESULT_DIR/probe_tokens.npz" ]; then
    echo "[row-mid] SKIP $RUN_ID (final_model + probe already present)"
    return 0
  fi
  mkdir -p "$RESULT_DIR"
  echo "[row-mid] $RUN_ID mult=$TM -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/${RUN_ID}_row" \
  "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_row" --injection_position input \
    --steps 1000 --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_batches 4 --lr 0.004 \
    --enable_unigram 0 --enable_bigram 1 --enable_trigram 0 \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --epoch_batches 337 \
    --fixed_train_probe 0 \
    --val_steps 1000 \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    --table_mult "$TM" \
    --dtype bf16 \
    --save_final_model \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[row-mid] $RUN_ID done at $(date)"
}

NGPU=${#GPUS[@]}
ACTIVE=0
SLOT=0
launch() {  # launch <run_id> <table_mult>
  local RUN_ID="$1" TM="$2"
  while [ "$ACTIVE" -ge "$NGPU" ]; do
    wait -n
    ACTIVE=$((ACTIVE - 1))
  done
  local GPU="${GPUS[$SLOT]}"
  SLOT=$(( (SLOT + 1) % NGPU ))
  run_arm "$GPU" "$RUN_ID" "$TM" &
  ACTIVE=$((ACTIVE + 1))
}

for TM in $TABLE_MULTS; do
  launch "row_tbl_${TM}_bigram" "$TM"
done

while [ "$ACTIVE" -gt 0 ]; do
  wait -n
  ACTIVE=$((ACTIVE - 1))
done

echo "=== row-level mid table grid done at $(date) ==="
