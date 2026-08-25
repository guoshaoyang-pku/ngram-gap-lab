#!/usr/bin/env bash
# clean 单表加密取点 wave-2（2026-08-25 下午，用户拍板）：
#   1) bigram 加密 17 点（低 R 区重点）：16K 32K 48K 96K 160K 192K 320K 448K
#      640K 896K 1.25M 1.75M 2.25M 2.75M 3.5M 5M 6M —— 全部 sparse 末端。
#      与 wave-1 的 13 点合计 30 点 bigram 相图。
#   2) 曲线补跑（freq=50，用户要求只画 3 张曲线：小/大/零碰撞）：
#      ctbl_65536 是 wave-1 sparse 跑的，需要 freq=50 版补一条"小表"曲线；
#      ctbl_4194304 同理补"大表"曲线。run_id 加 _curve 后缀，不覆盖 sparse 口径。
#      （零碰撞曲线 wave-1 已有 ctbl_perfect_bigram。）
#   3) trigram clean 网格（排队到最后）：64K 256K 1M 2M 4M 8M，sparse 末端。
#      N_tri = 18,989,467 distinct contexts；R=8M → K/N = 0.42（19M 行 fp32 放
#      不下，trigram 不扫零碰撞锚点）。
#
# Usage: ./run_clean_table_dense2.sh <gpu1> [gpu2] ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
FREQ_IDX="$ROOT/data/freq_index.npz"
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then GPUS=(0 1); fi
mkdir -p "$OUT_DIR"

run_one() {  # run_one <gpu> <run_id> <extra args...>
  local GPU="$1" RUN_ID="$2"; shift 2
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  if [ -f "$RESULT_DIR/summary.json" ]; then
    echo "[ctbl2] SKIP $RUN_ID (summary.json present)"; return 0
  fi
  mkdir -p "$RESULT_DIR"
  echo "[ctbl2] $RUN_ID -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps 1000 --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_batches 4 --lr 0.004 \
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
  echo "[ctbl2] $RUN_ID done (exit=$?) at $(date)"
}

occ_clean() {  # occ_clean <run_dir> <branch_flag> <R>
  "$PY" -u "$ROOT/code/table_occupancy.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --epoch_batches 337 \
    "$2" "$3" --out "$1/table_occupancy.json" \
    > "$1/occupancy.log" 2>&1 || echo "[ctbl2] occupancy failed $1"
}

run_bi_sparse() {  # <R> <gpu>
  local R="$1" GPU="$2"
  run_one "$GPU" "ctbl_${R}_bigram" \
    --enable_unigram 0 --enable_bigram 1 --enable_trigram 0 \
    --bigram_clean_table "$R" --val_steps 1000 --exact_freq_eval_interval 1000
  occ_clean "$OUT_DIR/ctbl_${R}_bigram_fixed" --bigram_clean_table "$R"
}

run_bi_curve() {  # <R> <gpu>  (freq=50 curve variant, _curve suffix)
  local R="$1" GPU="$2"
  run_one "$GPU" "ctbl_${R}_bigram_curve" \
    --enable_unigram 0 --enable_bigram 1 --enable_trigram 0 \
    --bigram_clean_table "$R" \
    --val_interval 50 --freq_eval_interval 50 --exact_freq_eval_interval 100
}

run_tri_sparse() {  # <R> <gpu>
  local R="$1" GPU="$2"
  run_one "$GPU" "ctbl_${R}_trigram" \
    --enable_unigram 0 --enable_bigram 0 --enable_trigram 1 \
    --trigram_clean_table "$R" --val_steps 1000 --exact_freq_eval_interval 1000
}

NGPU=${#GPUS[@]}

wave() {  # wave <spec...>  (gpu appended last; wait for the whole wave)
  local i=0
  for spec in "$@"; do
    local GPU="${GPUS[$((i % NGPU))]}"
    i=$((i + 1))
    eval "$spec" "$GPU" &
  done
  wait
}

# --- wave A: bigram 低 R 区加密（最快的先跑完腾出卡） + 2 个曲线补跑 -------
wave \
  "run_bi_curve 4194304" \
  "run_bi_curve 65536" \
  "run_bi_sparse 6291456" \
  "run_bi_sparse 5242880" \
  "run_bi_sparse 3670016" \
  "run_bi_sparse 2883584" \
  "run_bi_sparse 2359296"

wave \
  "run_bi_sparse 1835008" \
  "run_bi_sparse 1310720" \
  "run_bi_sparse 917504" \
  "run_bi_sparse 655360" \
  "run_bi_sparse 458752" \
  "run_bi_sparse 327680" \
  "run_bi_sparse 196608"

wave \
  "run_bi_sparse 163840" \
  "run_bi_sparse 98304" \
  "run_bi_sparse 49152" \
  "run_bi_sparse 32768" \
  "run_bi_sparse 16384"

# --- wave B: trigram clean（排队到最后） ------------------------------------
wave \
  "run_tri_sparse 8388608" \
  "run_tri_sparse 4194304" \
  "run_tri_sparse 2097152" \
  "run_tri_sparse 1048576" \
  "run_tri_sparse 262144" \
  "run_tri_sparse 65536"

echo "=== clean-table dense wave-2 done at $(date) ==="
