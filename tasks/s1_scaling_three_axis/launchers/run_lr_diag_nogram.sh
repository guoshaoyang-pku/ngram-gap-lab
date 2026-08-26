#!/usr/bin/env bash
# LR-schedule 诊断（2026-08-25）：nogram 纯 backbone，排除表干扰，只比 LR schedule 对收敛的影响。
# 回答用户："是 warmup 少导致不收敛，还是缺 cooldown 导致？"
#
# 臂设计（全部 nogram：--enable_bigram 0 --enable_trigram 0）：
#   C: warmup_constant 4e-3（warmup 100 步 1e-3→4e-3，之后恒定）——v4 候选，已知疑似不收敛
#   D: warmup_cosine  4e-3（warmup 100 步 + cosine decay 到 5%）——对照"有 cooldown"
#   E: constant 4e-4（低 lr 恒定）——对照"纯低 lr 能否收敛"（绝对 lr 而非 schedule）
# 已有复用臂：A = ng lab1x_v10_nogram_fixed（warmdown 收敛 2.98），
#            B = v4 constant 4e-3（若无现成 run 由本脚本可选补跑）
#
# 产出：data/runs_fixed/<run_id>_fixed/train_log.jsonl（step, train_loss, val_loss, lr）
# Usage: ./run_lr_diag_nogram.sh <gpu0> <gpu1> <gpu2> [--with-constant]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_fixed"
mkdir -p "$OUT_DIR"

WITH_CONSTANT=0
GPUS=()
for a in "$@"; do
  if [ "$a" = "--with-constant" ]; then WITH_CONSTANT=1; else GPUS+=("$a"); fi
done
if [ "${#GPUS[@]}" -lt 3 ]; then echo "need >=3 gpus"; exit 1; fi

run_one() {  # run_one <gpu> <run_id> <schedule> <extra...>
  local GPU="$1" RUN_ID="$2" SCHED="$3"; shift 3
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  if [ -f "$RESULT_DIR/summary.json" ]; then
    echo "[lrdiag] SKIP $RUN_ID (summary.json present)"; return 0
  fi
  mkdir -p "$RESULT_DIR"
  echo "[lrdiag] $RUN_ID ($SCHED) -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps 1000 --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_batches 4 --lr 0.004 \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --enable_unigram 0 --enable_bigram 0 --enable_trigram 0 \
    --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 2.0 \
    --lr_schedule "$SCHED" --warmup_steps 100 \
    --val_interval 10 --dtype bf16 \
    "$@" \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[lrdiag] $RUN_ID done (exit $?) at $(date)"
}

# C: warmup_constant
run_one "${GPUS[0]}" lrdiag_nogram_warmup_const warmup_constant &
# D: warmup_cosine
run_one "${GPUS[1]}" lrdiag_nogram_warmup_cosine warmup_cosine &
# E: constant 4e-4（低 lr，无 warmup）
run_one "${GPUS[2]}" lrdiag_nogram_const_4e4 constant --lr 0.0004 &
# B: constant 4e-3（若需补跑）
if [ "$WITH_CONSTANT" = "1" ] && [ "${#GPUS[@]}" -ge 4 ]; then
  run_one "${GPUS[3]}" lrdiag_nogram_const_4e3 constant &
fi

wait
echo "[lrdiag] all arms done at $(date)"
