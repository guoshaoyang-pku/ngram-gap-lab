#!/usr/bin/env bash
# Full table-size scaling grid (plan §5) on 360-2.  Run AFTER pilot QC.
#
# Downward-only from default 1M logical addresses (2R):
#   mult=64->2R=1M (reuse pilot run), 32->512K, 16->256K, 8->128K (pilot),
#   4->64K, 2->32K, 1->16K (pilot).  Modules {bigram, trigram, both}.
# L4 = 337 batches/epoch (FULL shard 1), 1000 steps, step-anchored LR,
# beta2=0.99.  Ordinary grid runs do NOT compute exact-frequency / freq-bin
# diagnostics (no --freq_index); occupancy is still emitted per run.
#
# Usage: ./run_scaling_table_full.sh <gpu_id1> [gpu_id2] [gpu_id3] ...
#   Pass the CUDA device ids of the free GPUs (any number >= 1).  The 3 module
#   arms for each table size are scheduled round-robin across the given GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then GPUS=(0 1 2); fi
mkdir -p "$OUT_DIR"

run_arm() {  # run_one <gpu> <run_id> <table_mult> <bigram> <trigram>
  local GPU="$1" RUN_ID="$2" TM="$3" BI="$4" TRI="$5"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  mkdir -p "$RESULT_DIR"
  echo "[table-full] $RUN_ID mult=$TM bi=$BI tri=$TRI -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps 1000 --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 10 --val_batches 4 --table_norm_interval 10 --lr 0.004 \
    --enable_unigram 0 --enable_bigram "$BI" --enable_trigram "$TRI" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --epoch_batches 337 \
    --fixed_train_probe 4 --probe_eval_interval 10 \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    --table_mult "$TM" \
    --dtype bf16 --compile \
    > "$RESULT_DIR/train.log" 2>&1
  "$PY" -u "$TASK_ROOT/code/table_occupancy.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --epoch_batches 337 \
    --table_mult "$TM" --out "$RESULT_DIR/table_occupancy.json" \
    > "$RESULT_DIR/occupancy.log" 2>&1 || echo "[table-full] occupancy failed $RUN_ID"
  echo "[table-full] $RUN_ID done (exit=$?) at $(date)"
}

# 7 table sizes x 3 modules (mult=64 and 8 and 1 already in pilot; rerun all
# for uniform naming & full collision curve)
GPUS_IDX=0
for TM in 64 32 16 8 4 2 1; do
  for ARM in bigram trigram both; do
    case "$ARM" in
      bigram) BI=1 TRI=0 ;;
      trigram) BI=0 TRI=1 ;;
      both) BI=1 TRI=1 ;;
    esac
    GPU="${GPUS[$GPUS_IDX]}"
    GPUS_IDX=$(( (GPUS_IDX + 1) % ${#GPUS[@]} ))
    run_arm "$GPU" "tbl_${TM}_${ARM}" "$TM" "$BI" "$TRI" &
  done
  wait
  GPUS_IDX=0
done

echo "=== table full grid done at $(date) ==="
