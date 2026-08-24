#!/usr/bin/env bash
# Full table-size scaling grid (plan §5) on 360-2.  Run AFTER pilot QC.
#
# Downward-only from default 1M logical addresses (2R):
#   mult=64->2R=1M (reuse pilot run), 32->512K, 16->256K, 8->128K (pilot),
#   4->64K, 2->32K, 1->16K (pilot).  Modules {bigram, trigram, both}.
# L4 = 336 batches/epoch, 1000 steps, step-anchored LR, beta2=0.99.
#
# Usage: ./run_scaling_table_full.sh [gpu1] [gpu2] [gpu3]
set -euo pipefail

ROOT=/data/home/guoshaoyang/ngram-gap-lab
PY=/usr/bin/python3
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
FREQ_IDX="$ROOT/data/freq_index.npz"
G1="${1:-0}" G2="${2:-1}" G3="${3:-2}"
mkdir -p "$OUT_DIR"

run_arm() {  # run_one <gpu> <run_id> <table_mult> <bigram> <trigram>
  local GPU="$1" RUN_ID="$2" TM="$3" BI="$4" TRI="$5"
  local RESULT_DIR="$OUT_DIR/$RUN_ID"
  mkdir -p "$RESULT_DIR"
  echo "[table-full] $RUN_ID mult=$TM bi=$BI tri=$TRI -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "$RUN_ID" --injection_position input \
    --steps 1000 --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 50 --val_batches 4 --table_norm_interval 50 --lr 0.004 \
    --enable_unigram 0 --enable_bigram "$BI" --enable_trigram "$TRI" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --freq_eval_interval 50 --freq_eval_batches 4 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --freq_index "$FREQ_IDX" \
    --epoch_batches 336 \
    --fixed_train_probe 4 --probe_eval_interval 50 \
    --table_betas 0.0,0.99 \
    --table_lr_scale 1.0 \  # frozen: pilots ran under the pre-2026-08-24 default
    --table_mult "$TM" \
    > "$RESULT_DIR/train.log" 2>&1
  "$PY" -u "$ROOT/code/table_occupancy.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --epoch_batches 336 \
    --table_mult "$TM" --out "$RESULT_DIR/table_occupancy.json" \
    > "$RESULT_DIR/occupancy.log" 2>&1 || echo "[table-full] occupancy failed $RUN_ID"
  echo "[table-full] $RUN_ID done (exit=$?) at $(date)"
}

# 7 table sizes x 3 modules (mult=64 and 8 and 1 already in pilot; rerun all
# for uniform naming & full collision curve)
for TM in 64 32 16 8 4 2 1; do
  run_arm "$G1" "tbl_${TM}_bigram"  "$TM" 1 0 &
  run_arm "$G2" "tbl_${TM}_trigram" "$TM" 0 1 &
  run_arm "$G3" "tbl_${TM}_both"    "$TM" 1 1 &
  wait
done

echo "=== table full grid done at $(date) ==="
