#!/usr/bin/env bash
# Epoch-length scaling arms, v10 standard (validation + freq eval every 10 steps),
# 2000 steps, seed 42, input injection (the blog's default setting).
#
#   nglab2x_input_v10_fv : train shards 1,2 (~674 steps/epoch — epoch-1 doubled)
#   nglab0_5x_input_fv   : train shard  60 (~168 steps/epoch — epoch-1 halved;
#                          shard_00060 = first 12132 rows of shard_00001)
#
# _fv = fixed-val: train.py captures fixed val batches at startup and reuses them
# for every val eval (and the val-side freq-bin eval), so val curves always
# measure the SAME val data.  The earlier nglab2x_input_v10 / nglab0_5x_input
# runs (moving-window val) are superseded by these.
#
# Usage: ./run_epoch_scale_v10.sh [gpu_2x] [gpu_0_5x]
set -euo pipefail

GPU2X="${1:-0}"
GPU05X="${2:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"

run_one() {
  local GPU="$1"; local EXP="$2"; shift 2
  local RESULT_DIR="$ROOT/data/runs_fixed/${EXP}_fixed"
  mkdir -p "$RESULT_DIR"
  echo "[launch] $EXP on GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "${EXP}_fixed" \
    --injection_position input \
    --steps 2000 \
    --seed 42 \
    --data_dir "$DATA_DIR" \
    --out_dir "$ROOT/data/runs_fixed" \
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
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer rmsprop \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    "$@" \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[launch] $EXP exit=$? at $(date)"
}

# 2x: train = shards 1,2 ; val = 3..10,6542 (same split as the completed nglab2x batch)
run_one "$GPU2X" "nglab2x_input_v10_fv" \
  --train_shards 1,2 \
  --val_shards 3,4,5,6,7,8,9,10,6542 \
  --freq_index "$ROOT/data/freq_index_train2x_fine.npz" &

# 0.5x: train = shard 60 (first half of shard 1) ; val = standard val split
run_one "$GPU05X" "nglab0_5x_input_fv" \
  --train_shards 60 \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --freq_index "$ROOT/data/freq_index_train0_5x.npz" &

wait
echo "=== epoch-scale v10 batch done at $(date) ==="
