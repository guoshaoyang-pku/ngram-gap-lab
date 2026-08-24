#!/usr/bin/env bash
# Probe-v3 单 run 执行器 —— 新标准 + uniform train probe
# 在 run_rerun_v2.sh 基础上加：
#   --train_probe_mode uniform（train 全域均匀采样固定 probe，与 val 平行）
# 用法: ./run_probe_v3.sh <gpu_id> <run_id> <train_shards> <val_shards> <steps> [extra...]
#   extra: --enable_bigram 0 --enable_trigram 0（nogram）
#          --fixed_train_probe N（probe batch 数，默认 4）
#   compute dtype 固定 bf16（新标准 §1.4，不 compile）
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

PROBE_MODE="${NGLAB_PROBE_MODE:-uniform}"
PROBE_BATCHES="${NGLAB_PROBE_BATCHES:-4}"

echo "=== $RUN_ID  GPU=$GPU  shards=$TRAIN_SHARDS  steps=$STEPS  probe_mode=$PROBE_MODE  probe_batches=$PROBE_BATCHES  extra=$*  $(date) ==="

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
  --fixed_train_probe "$PROBE_BATCHES" \
  --train_probe_mode "$PROBE_MODE" \
  --probe_eval_interval 10 \
  --dtype bf16 \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  "$@" \
  > "$RESULT_DIR/train.log" 2>&1

echo "=== $RUN_ID finished (exit $?) at $(date) ==="
