#!/usr/bin/env bash
# Rerun V2 单 run 执行器 —— 新标准：β₂=0.99（无动量）· 表学习率 ×2
# 用户 2026-08-24 拍板；所有参数显式传递，不依赖默认值。
#
# Usage: ./run_rerun_v2.sh <gpu_id> <run_id> <train_shards> <val_shards> <steps> <extra...>
#   extra 可追加: --injection_position v/y/input（默认 input）
#                  --enable_bigram 0 --enable_trigram 0（nogram 臂）
#                  --intervention reset_table --intervention_epoch 1（因果臂）
#                  --table_mult 16（表大小扫描臂）
set -euo pipefail

GPU="${1:?gpu id}"
RUN_ID="${2:?run id}"
TRAIN_SHARDS="${3:?train shards}"
VAL_SHARDS="${4:?val shards}"
STEPS="${5:?steps}"
shift 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
RESULT_DIR="$ROOT/data/runs_fixed/${RUN_ID}_fixed"
mkdir -p "$RESULT_DIR"

# --- freq index 自动选择：小剂量臂用旧波次对应的 train-set 专属索引（保证
# --- freq-bin 曲线与旧 _fixed run 同口径可比）；没有对应专属索引时回退全局。
TRAIN_SHARDS="$(echo "$TRAIN_SHARDS" | tr -d ' ')"
pick_freq_index() {
  local name
  case "$TRAIN_SHARDS" in
    62)      name="freq_index_train0_25x" ;;
    60)      name="freq_index_train0_5x" ;;
    63)      name="freq_index_train0_75x" ;;
    1,61)    name="freq_index_train1_5x" ;;
    1,2,64)  name="freq_index_train2_5x" ;;
    1,2)     name="freq_index_train2x_fine" ;;
    1,2,3)   name="freq_index_train3x" ;;
    1,2,3,4) name="freq_index_train4x" ;;
    1,2,3,4,5) name="freq_index_train5x" ;;
    1,2,3,4,5,6) name="freq_index_train6x" ;;
    1,2,3,4,5,6,7,8) name="freq_index_train8x" ;;
    *)       name="freq_index" ;;
  esac
  if [ -f "$ROOT/data/${name}.npz" ]; then
    echo "$ROOT/data/${name}.npz"
  else
    echo "$ROOT/data/freq_index.npz"
  fi
}
FREQ_INDEX="${FREQ_INDEX:-$(pick_freq_index)}"

echo "=== $RUN_ID  GPU=$GPU  shards=$TRAIN_SHARDS  steps=$STEPS  freq_index=$(basename "$FREQ_INDEX")  extra=$*  $(date) ==="

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" \
  --injection_position input \
  --steps "$STEPS" \
  --seed 42 \
  --data_dir "$ROOT/data/tokenized" \
  --out_dir "$ROOT/data/runs_fixed" \
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
  --table_optimizer rmsprop \
  --table_betas 0.0,0.99 \
  --table_lr_scale 2.0 \
  --table_mult 64 \
  --freq_index "$FREQ_INDEX" \
  --freq_eval_interval 10 \
  --freq_eval_batches 4 \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  "$@" \
  > "$RESULT_DIR/train.log" 2>&1

echo "=== $RUN_ID finished (exit $?) at $(date) ==="
