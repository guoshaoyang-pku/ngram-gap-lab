#!/usr/bin/env bash
# 表大小扫描（自然语言数据，极简 setting）
# 复现 DEPRECATED 结论「表大小在无碰撞点饱和，参数量不是主导变量」
# （原 toy 数据：vocab=2048, M=16 无碰撞，见 docs/_archive/docs/table-size-sweep-results-20260811.md）
# 本次在自然语言 shard1 上扫 M=16/32/64/128/256，table size = 8192 * M。
# 注意：M=64 即 SSOT 默认（1M 行）。其余 setting 与 §14 完全一致。
#
# Usage: ./run_table_sweep_minimal.sh <gpu_id> <M>
set -euo pipefail

GPU="${1:?gpu id}"
M="${2:?table mult}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"

EXP="nglab1x_table_mult_${M}"
RESULT_DIR="$ROOT/data/runs_fixed/${EXP}_fixed"
mkdir -p "$RESULT_DIR"

echo "=== $EXP  table_mult=$M  GPU=$GPU  $(date) ==="

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "$EXP" \
  --injection_position input \
  --steps 1000 \
  --seed 42 \
  --data_dir "$DATA_DIR" \
  --out_dir "$ROOT/data/runs_fixed" \
  --train_shards 1 \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --device_batch_size 72 \
  --total_batch_size 147456 \
  --val_interval 10 \
  --val_batches 4 \
  --table_norm_interval 10 \
  --lr 0.004 \
  --enable_unigram 0 \
  --enable_bigram 1 \
  --enable_trigram 1 \
  --table_mult "$M" \
  --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 2.0 \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  > "$RESULT_DIR/train.log" 2>&1

echo "=== $EXP finished (exit $?) at $(date) ==="
