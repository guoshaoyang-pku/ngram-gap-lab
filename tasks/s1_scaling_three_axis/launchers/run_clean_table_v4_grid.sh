#!/usr/bin/env bash
# clean 单表 v4 加密网格（2026-08-25 用户拍板：重刷全部 v4 实验 + 两者加密取点）。
#
# v4 口径 = warmup_constant LR（warmup 100 步 1e-3→4e-3，无 decay，用户 2026-08-25
# 拍板）+ β₂=0.99 + 表 LR ×2 + bf16 不 compile（与 run_rerun_v4.sh 基线一致）。
#
# 网格：
#   bigram  24 点对数均匀（16K -> 6.5M，K/N 0.0045->1.53）+ perfect 零碰撞锚点
#   trigram 14 点对数均匀（64K -> 8M，K/N 0.0035->0.44；19M 行放不下，无零碰撞锚点）
# 全部 sparse 末端（--val_steps 1000，只取 final gap，最快）。
#
# Usage: ./run_clean_table_v4_grid.sh <gpu...>
#   GPU 不够时会按 wave 分批；跨机并行用 scp + ssh（见 skills/ngram-gap-rerun-v4）。
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
    echo "[ctbl4] SKIP $RUN_ID (summary.json present)"; return 0
  fi
  mkdir -p "$RESULT_DIR"
  echo "[ctbl4] $RUN_ID -> GPU $GPU at $(date)"
  local rc=1 attempt=0
  while [ "$rc" -ne 0 ] && [ "$attempt" -lt 2 ]; do
    attempt=$((attempt + 1))
    set +e
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
      --table_optimizer rmsprop \
      --table_betas 0.0,0.99 \
      --table_lr_scale 2.0 \
      --lr_schedule warmup_constant \
      --warmup_steps 100 \
      --dtype bf16 \
      "$@" \
      > "$RESULT_DIR/train.log" 2>&1
    rc=$?
    set -e
    if [ "$rc" -ne 0 ] && [ "$attempt" -lt 2 ]; then
      echo "[ctbl4] $RUN_ID attempt $attempt failed (rc=$rc), retrying on $(date)"
      sleep 10
    fi
  done
  echo "[ctbl4] $RUN_ID done (rc=$rc) at $(date)"
}

occ_clean() {  # occ_clean <run_dir> <branch_flag> <R>
  "$PY" -u "$ROOT/code/table_occupancy.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --epoch_batches 337 \
    "$2" "$3" --out "$1/table_occupancy.json" \
    > "$1/occupancy.log" 2>&1 || echo "[ctbl4] occupancy failed $1"
}

run_bi() {  # <R> <gpu>
  local R="$1" GPU="$2"
  if [ -f "$OUT_DIR/ctbl_v4w_${R}_bigram_fixed/summary.json" ]; then
    echo "[ctbl4] SKIP run_bi $R (summary.json present)"
    return 0
  fi
  run_one "$GPU" "ctbl_v4w_${R}_bigram" \
    --enable_unigram 0 --enable_bigram 1 --enable_trigram 0 \
    --bigram_clean_table "$R" --val_steps 1000 --exact_freq_eval_interval 1000
  occ_clean "$OUT_DIR/ctbl_v4w_${R}_bigram_fixed" --bigram_clean_table "$R"
}

run_bi_perfect() {  # <gpu>
  local GPU="$1"
  if [ -f "$OUT_DIR/ctbl_v4w_perfect_bigram_fixed/summary.json" ]; then
    echo "[ctbl4] SKIP run_bi_perfect (summary.json present)"
    return 0
  fi
  run_one "$GPU" "ctbl_v4w_perfect_bigram" \
    --enable_unigram 0 --enable_bigram 1 --enable_trigram 0 \
    --bigram_perfect_map "$ROOT/data/bigram_perfect_map_s1.npz" \
    --val_steps 1000 --exact_freq_eval_interval 1000
}

run_tri() {  # <R> <gpu>
  local R="$1" GPU="$2"
  if [ -f "$OUT_DIR/ctbl_v4w_${R}_trigram_fixed/summary.json" ]; then
    echo "[ctbl4] SKIP run_tri $R (summary.json present)"
    return 0
  fi
  run_one "$GPU" "ctbl_v4w_${R}_trigram" \
    --enable_unigram 0 --enable_bigram 0 --enable_trigram 1 \
    --trigram_clean_table "$R" --val_steps 1000 --exact_freq_eval_interval 1000
}

# ---- bigram 24 点（对数均匀）+ perfect ----
BIGRAM=(16000 20612 26553 34207 44068 56770 73134 94215 121372 156358 201428
        259490 334287 430646 554779 714694 920704 1186096 1527988 1968430
        2535829 3266781 4208429 5421506)
# ---- trigram 14 点（对数均匀）----
TRIGRAM=(65536 95186 138250 200798 291643 423590 615231 893576 1297850
         1885027 2737856 3976525 5775596 8388608)

NGPU=${#GPUS[@]}

# 滚动 slot 调度：每发一个 run 占用一张卡，等任一 run 结束才回收。
# 绝不把第二个 run 发到还在跑的卡上（避免一卡多 run 并发 OOM——wave 循环
# 分配在 spec 数 > NGPU 时会把 run 摊到同一卡，2026-08-25 踩过 perfect OOM）。
launch_slot() {  # launch_slot <fn args...>  (gpu appended last)
  while [ "$ACTIVE" -ge "$NGPU" ]; do wait -n; ACTIVE=$((ACTIVE - 1)); done
  local GPU="${GPUS[$SLOT]}"
  SLOT=$(( (SLOT + 1) % NGPU ))
  ACTIVE=$((ACTIVE + 1))
  eval "$1" "$GPU" &
}

# 分工：CTBL4_ONLY=ab -> bigram（wave A+B，含 perfect；360-2）
#         CTBL4_ONLY=c  -> trigram（wave C；360-1）
# 缺省 = 全部跑（单机全跑）。
CTBL4_ONLY="${CTBL4_ONLY:-all}"

if [ "$CTBL4_ONLY" = "all" ] || [ "$CTBL4_ONLY" = "ab" ]; then
# 先发最快的 bigram（sparse 1000 步 ~5-8 分钟/run）腾卡
echo "=== [ctbl4] wave A: bigram 前半 + perfect at $(date) ==="
ACTIVE=0 SLOT=0
launch_slot "run_bi_perfect"
launch_slot "run_bi 16000"
launch_slot "run_bi 20612"
launch_slot "run_bi 26553"
launch_slot "run_bi 34207"
launch_slot "run_bi 44068"
launch_slot "run_bi 56770"
launch_slot "run_bi 73134"
launch_slot "run_bi 94215"
launch_slot "run_bi 121372"
launch_slot "run_bi 156358"
launch_slot "run_bi 201428"
launch_slot "run_bi 259490"
wait

echo "=== [ctbl4] wave B: bigram 后半 at $(date) ==="
ACTIVE=0 SLOT=0
launch_slot "run_bi 334287"
launch_slot "run_bi 430646"
launch_slot "run_bi 554779"
launch_slot "run_bi 714694"
launch_slot "run_bi 920704"
launch_slot "run_bi 1186096"
launch_slot "run_bi 1527988"
launch_slot "run_bi 1968430"
launch_slot "run_bi 2535829"
launch_slot "run_bi 3266781"
launch_slot "run_bi 4208429"
launch_slot "run_bi 5421506"
wait
fi

if [ "$CTBL4_ONLY" = "all" ] || [ "$CTBL4_ONLY" = "c" ]; then
echo "=== [ctbl4] wave C: trigram 14 点 at $(date) ==="
ACTIVE=0 SLOT=0
launch_slot "run_tri 65536"
launch_slot "run_tri 95186"
launch_slot "run_tri 138250"
launch_slot "run_tri 200798"
launch_slot "run_tri 291643"
launch_slot "run_tri 423590"
launch_slot "run_tri 615231"
launch_slot "run_tri 893576"
launch_slot "run_tri 1297850"
launch_slot "run_tri 1885027"
launch_slot "run_tri 2737856"
launch_slot "run_tri 3976525"
launch_slot "run_tri 5775596"
launch_slot "run_tri 8388608"
wait
fi

echo "=== [ctbl4] clean v4 grid done at $(date) ==="
