#!/usr/bin/env bash
# Table-size scaling grid (plan §5) on 360-2.
#
# Horizontal axis = logical addresses 2R (per n-gram, per layer, two
# decorrelated hash embeddings).  table_mult maps R = vocab_size * table_mult:
#   mult=64 -> 2R=1,048,576 (default 1M)   mult=32 -> 524,288
#   mult=16 -> 262,144    mult=8 -> 131,072    mult=4 -> 65,536
#   mult=2  -> 32,768     mult=1 -> 16,384
# Downward-only from 1M.
#
# Training: L4 = 336 batches/epoch (nested prefix of shard 1), 1000 steps,
# step-anchored LR, β₂=0.99, modules {bigram-only, trigram-only, both}.
# Each run outputs exact-frequency + occupancy diagnostics.
#
# Usage: ./run_scaling_table.sh [gpu1] [gpu2] [gpu3]
set -euo pipefail

ROOT=/data/home/guoshaoyang/ngram-gap-lab
PY=/usr/bin/python3
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
FREQ_IDX="$ROOT/data/freq_index.npz"
G1="${1:-0}" G2="${2:-1}" G3="${3:-2}"
mkdir -p "$OUT_DIR"

# module -> enable flags
run_arm() {  # run_one <gpu> <run_id> <table_mult> <bigram> <trigram>
  local GPU="$1" RUN_ID="$2" TM="$3" BI="$4" TRI="$5"
  local RESULT_DIR="$OUT_DIR/$RUN_ID"
  mkdir -p "$RESULT_DIR"
  echo "[table] $RUN_ID mult=$TM bigram=$BI trigram=$TRI -> GPU $GPU at $(date)"
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
    --table_lr_scale 1.0 \
    --table_mult "$TM" \
    > "$RESULT_DIR/train.log" 2>&1
  # occupancy diagnostic (offline, cheap, per run)
  "$PY" -u "$ROOT/code/table_occupancy.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --epoch_batches 336 \
    --table_mult "$TM" --out "$RESULT_DIR/table_occupancy.json" \
    > "$RESULT_DIR/occupancy.log" 2>&1 || echo "[table] occupancy failed for $RUN_ID"
  echo "[table] $RUN_ID done (exit=$?) at $(date)"
}

# --- Pilot gate: logical addresses {1M, 128K, 16K} × 3 modules ---
run_arm "$G1" tbl_pilot_1M_bigram    64 1 0 &
run_arm "$G2" tbl_pilot_128K_bigram  8  1 0 &
run_arm "$G3" tbl_pilot_16K_bigram   1  1 0 &
wait
run_arm "$G1" tbl_pilot_1M_trigram   64 0 1 &
run_arm "$G2" tbl_pilot_128K_trigram 8  0 1 &
run_arm "$G3" tbl_pilot_16K_trigram  1  0 1 &
wait
run_arm "$G1" tbl_pilot_1M_both      64 1 1 &
run_arm "$G2" tbl_pilot_128K_both    8  1 1 &
run_arm "$G3" tbl_pilot_16K_both     1  1 1 &
wait

# --- Full grid: 7 table sizes × 3 modules ---
for TM in 64 32 16 8 4 2 1; do
  run_arm "$G1" "tbl_${TM}_bigram"  "$TM" 1 0 &
  run_arm "$G2" "tbl_${TM}_trigram" "$TM" 0 1 &
  run_arm "$G3" "tbl_${TM}_both"    "$TM" 1 1 &
  wait
done

echo "=== table scaling grid done at $(date) ==="
