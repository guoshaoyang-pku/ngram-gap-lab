#!/usr/bin/env bash
# Shard-size sweep (dose-response of epoch length -> gap), v10 standard,
# fixed-val, freq eval every 10 steps, 2000+ steps, seed 42, input injection.
#
# Grid (multiplier of shard_00001 = 24264 rows):
#   0.25x (shard 62) 0.75x (shard 63) 1.5x (1,61) 2.5x (1,2,64)
#   3x (1,2,3) 4x (1,2,3,4)
# 2.5x/3x/4x run longer so they also reach ~5.5 observed epochs.
#
# Usage: ./run_shard_sweep.sh           # GPUs 0-5
#        GPU_SET="3 4 5" ./run_shard_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_fixed"
GPU_SET="${GPU_SET:-0 1 2 3 4 5}"
LOCKDIR="$ROOT/logs/.sweep_locks"
mkdir -p "$LOCKDIR"

acquire_gpu() {
  while true; do
    for g in $GPU_SET; do
      if mkdir "$LOCKDIR/gpu_$g" 2>/dev/null; then
        if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")" -lt 1000 ]; then
          echo "$g"; return
        fi
        rmdir "$LOCKDIR/gpu_$g"
      fi
    done
    sleep 60
  done
}

run_one() {  # run_one <run_id> <train_shards> <val_shards> <steps> <freq_index>
  local RUN_ID="$1" SHARDS="$2" VAL="$3" STEPS="$4" FREQ="$5"
  local GPU; GPU="$(acquire_gpu)"
  trap "rmdir $LOCKDIR/gpu_$GPU 2>/dev/null || true" EXIT
  local RESULT_DIR="$OUT_DIR/$RUN_ID"
  mkdir -p "$RESULT_DIR"
  echo "[sweep] $RUN_ID (shards=$SHARDS steps=$STEPS) -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "$RUN_ID" --injection_position input --steps "$STEPS" --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 2.0 \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 10 --val_batches 4 --table_norm_interval 10 --lr 0.004 \
    --enable_unigram 0 --enable_bigram 1 --enable_trigram 1 \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --freq_eval_interval 10 --freq_eval_batches 4 \
    --train_shards "$SHARDS" --val_shards "$VAL" --freq_index "$FREQ" \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[sweep] $RUN_ID done (exit=$?) at $(date)"
  trap - EXIT
  rmdir "$LOCKDIR/gpu_$GPU" 2>/dev/null || true
}

STD_VAL="2,3,4,5,6,7,8,9,10,6542"
BIG_VAL="3,4,5,6,7,8,9,10,6542"

run_one nglab0_25x_input_fv "62"              "$STD_VAL" 2000 "$ROOT/data/freq_index_train0_25x.npz" &
run_one nglab0_75x_input_fv "63"              "$STD_VAL" 2000 "$ROOT/data/freq_index_train0_75x.npz" &
run_one nglab1_5x_input_fv  "1,61"            "$BIG_VAL" 2000 "$ROOT/data/freq_index_train1_5x.npz"  &
run_one nglab2_5x_input_fv  "1,2,64"          "$BIG_VAL" 3200 "$ROOT/data/freq_index_train2_5x.npz"  &
run_one nglab3x_input_fv    "1,2,3"           "$BIG_VAL" 3800 "$ROOT/data/freq_index_train3x.npz"    &
run_one nglab4x_input_fv    "1,2,3,4"         "$BIG_VAL" 5000 "$ROOT/data/freq_index_train4x.npz"    &
wait
echo "=== shard sweep (ophis) done at $(date) ==="
