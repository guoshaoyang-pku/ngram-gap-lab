#!/usr/bin/env bash
# P1/P2 因果干预 · 极简 setting 重跑（干净 vanilla nanoGPT + input 注入）
# 旧版极简 causal launcher；仅保留当前允许的干预类型。
#
# 干预在 epoch 边界触发（--intervention_epoch 为 0-indexed epoch）：
#   e1 边界 = 0-indexed epoch 1（step ~337）；e2 边界 = 0-indexed epoch 2（step ~674）。
# 当前支持：freeze_table、freeze_backbone、hash_reseed、mask_low_freq、
# mask_high_freq。
#
# NOTE: 干预臂必须与 §14 控制臂 vanilla_input_1000_seed42 完全同 setting，
# 仅加 --intervention / --intervention_epoch。控制臂实跑配置：
#   table_betas 0.0,0.99 · table_lr_scale 2.0 · val_shards 2 · dtype fp32 · 无 compile
#
# 默认从当前仓库的 code/train.py 运行；集群副本可通过环境变量覆盖。
#
# Usage: ./run_causal_minimal.sh <gpu_id> <arm>
#   arm: freeze_table_e1 | freeze_backbone_e1 | hash_reseed_e1 |
#        mask_low_f200_e1 | mask_high_f200_e1
set -euo pipefail

GPU="${1:?gpu id}"
ARM="${2:?arm}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
TRAIN_PY="${NGLAB_TRAIN_PY:-$ROOT/code/train.py}"

EXP="nglab1x_input_${ARM}"
RESULT_DIR="$ROOT/data/runs_fixed/${EXP}_fixed"
mkdir -p "$RESULT_DIR"
COMPILE_CACHE="$ROOT/data/cache/torchinductor_${EXP}"
mkdir -p "$ROOT/data/cache"
mkdir -p "$COMPILE_CACHE"
COMPILE_ARGS=()
if [[ "${NGLAB_COMPILE:-0}" == "1" ]]; then
  COMPILE_ARGS+=(--compile)
fi

declare -A INTERV
INTERV[freeze_table_e1]="freeze_table"
INTERV[freeze_backbone_e1]="freeze_backbone"
INTERV[hash_reseed_e1]="hash_reseed"
INTERV[mask_low_f200_e1]="mask_low_freq"
INTERV[mask_high_f200_e1]="mask_high_freq"

declare -A EPOCH
EPOCH[freeze_table_e1]=1
EPOCH[freeze_backbone_e1]=1
EPOCH[hash_reseed_e1]=1
EPOCH[mask_low_f200_e1]=1
EPOCH[mask_high_f200_e1]=1

INTERV_VAL="${INTERV[$ARM]}"
EPOCH_VAL="${EPOCH[$ARM]}"
FREQ_ARGS=()
if [[ "$INTERV_VAL" == "mask_low_freq" || "$INTERV_VAL" == "mask_high_freq" ]]; then
  FREQ_ARGS=(
    --freq_index "$ROOT/data/freq_index.npz"
    --freq_eval_interval 10
    --freq_eval_batches 4
    --intervention_freq_threshold 200
  )
fi

echo "=== $EXP  intervention=$INTERV_VAL  epoch=$EPOCH_VAL  GPU=$GPU  $(date) ==="

TORCHINDUCTOR_CACHE_DIR="$COMPILE_CACHE" CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TRAIN_PY" \
  --run_id "${EXP}_fixed" \
  --injection_position input \
  --steps 1000 \
  --seed 42 \
  --data_dir "$DATA_DIR" \
  --out_dir "$ROOT/data/runs_fixed" \
  --train_shards 1 \
  --val_shards 2 \
  --device_batch_size 72 \
  --total_batch_size 147456 \
  --val_interval 10 \
  --val_batches 4 \
  --table_norm_interval 10 \
  --lr 0.004 \
  --enable_unigram 0 \
  --enable_bigram 1 \
  --enable_trigram 1 \
  --table_mult 64 \
  --table_optimizer rmsprop \
  --table_betas 0.0,0.99 \
  --table_lr_scale 2.0 \
  --dtype fp32 \
  "${COMPILE_ARGS[@]}" \
  --intervention "$INTERV_VAL" \
  --intervention_epoch "$EPOCH_VAL" \
  "${FREQ_ARGS[@]}" \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  2>&1 | tee "$RESULT_DIR/train.log"

echo "=== $EXP finished (exit $?) at $(date) ==="
