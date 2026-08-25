#!/usr/bin/env bash
# Bigram large-table + collision-free (perfect-map) arms (2026-08-25).
#
# Motivation (superposition/localization 相图): the existing 23-point grid only
# reaches K/N = 0.30 for bigram (N = 3.54M distinct contexts). These arms
# extend to K/N ~ 1.19 (jamming point) and add a zero-collision anchor:
#
#   tbl_128_bigram_fixed        mult=128, K/N = 0.59   (sparse: final-step only)
#   tbl_256_bigram_fixed        mult=256, K/N = 1.19   (sparse: final-step only)
#   tbl_64_bigram_l1_fixed      mult=64 single-layer control, freq=50 curve
#   tbl_perfect_bigram_l1_fixed collision-free map (N rows + UNK), single layer,
#                               freq=50 curve -> Delta_inf anchor + forking test
#
# The two large-mult runs write to data/runs_scaling/ with table_occupancy.json
# so analyze_scaling_table.py absorbs them into the formal grid automatically.
# The l1 pair uses --bigram_single_layer: the perfect table is single-layer to
# fit fp32 (4 layers x 3.54M x 768 x 12B > H200), so the control must match.
#
# Usage: ./run_bigram_large_perfect.sh <gpu1> [gpu2] [gpu3] ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
FREQ_IDX="$ROOT/data/freq_index.npz"
PERFECT_MAP="$ROOT/data/bigram_perfect_map_s1.npz"
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then GPUS=(0 1); fi
mkdir -p "$OUT_DIR"

# 0. build the collision-free map once (packed bigram -> row id, OOV -> UNK row)
if [ ! -f "$PERFECT_MAP" ]; then
  echo "[big-perfect] building perfect map -> $PERFECT_MAP"
  "$PY" -u "$ROOT/code/tools/make_bigram_perfect_map.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --val_shards 2,3 --out "$PERFECT_MAP"
fi

run_one() {  # run_one <gpu> <run_id> <extra args...>
  local GPU="$1" RUN_ID="$2"; shift 2
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  if [ -f "$RESULT_DIR/summary.json" ]; then
    echo "[big-perfect] SKIP $RUN_ID (summary.json present)"; return 0
  fi
  mkdir -p "$RESULT_DIR"
  echo "[big-perfect] $RUN_ID -> GPU $GPU at $(date)"
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
    "$@" \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[big-perfect] $RUN_ID done (exit=$?) at $(date)"
}

run_sparse_mult() {  # large-mult sparse arm + occupancy (formal-grid产物) <mult> <gpu>
  local TM="$1" GPU="$2"
  run_one "$GPU" "tbl_${TM}_bigram" \
    --table_mult "$TM" --val_steps 1000 --exact_freq_eval_interval 1000
  "$PY" -u "$ROOT/code/table_occupancy.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --epoch_batches 337 \
    --table_mult "$TM" --out "$OUT_DIR/tbl_${TM}_bigram_fixed/table_occupancy.json" \
    > "$OUT_DIR/tbl_${TM}_bigram_fixed/occupancy.log" 2>&1 \
    || echo "[big-perfect] occupancy failed mult=$TM"
}

# l1 pair (freq=50 curve for the forking comparison) <gpu>
run_l1_pair_perfect() {
  run_one "$1" "tbl_perfect_bigram_l1" \
    --bigram_perfect_map "$PERFECT_MAP" \
    --val_interval 50 --freq_eval_interval 50 --exact_freq_eval_interval 100
}
run_l1_pair_control() {
  run_one "$1" "tbl_64_bigram_l1" \
    --table_mult 64 --bigram_single_layer \
    --val_interval 50 --freq_eval_interval 50 --exact_freq_eval_interval 100
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

launch run_sparse_mult 256
launch run_sparse_mult 128
launch run_l1_pair_perfect
launch run_l1_pair_control

while [ "$ACTIVE" -gt 0 ]; do wait -n; ACTIVE=$((ACTIVE - 1)); done
echo "=== bigram large+perfect arms done at $(date) ==="
