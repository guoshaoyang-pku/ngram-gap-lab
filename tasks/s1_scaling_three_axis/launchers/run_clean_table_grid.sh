#!/usr/bin/env bash
# clean 单表 bigram R 网格（2026-08-25，SSOT: docs/notes/method/clean-table-rework.md）。
#
# 新框架重扫 table-size 轴：--bigram_clean_table R（单 nn.Embedding(R, 768)、
# 单层、单 hash、R 任意）。取代 [HISTORICAL 4-LAYER FRAMEWORK] 的 34 点网格。
#
# 取点（K/N = R / 3,538,293 distinct bigram contexts）：
#   平滑段:  64K 128K 256K 384K 512K          (K/N 0.019-0.148)
#   jamming: 768K 1M 1.5M 2M 2.5M 3M 4M       (K/N 0.222-1.185，加密重检锯齿)
#   锚点:    perfect (R = N+1 = 3,538,294，零碰撞，复用 bigram_perfect_map_s1.npz)
# R=1M 与 perfect 用 freq=50（同时产出 forking 对比曲线：碰撞 vs 零碰撞），
# 其余 sparse（--val_steps 1000 只测末端）。全部 bigram-only、1000 步、seed 42。
#
# Usage: ./run_clean_table_grid.sh <gpu1> [gpu2] ...
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

if [ ! -f "$PERFECT_MAP" ]; then
  echo "[ctbl] building perfect map -> $PERFECT_MAP"
  "$PY" -u "$ROOT/code/tools/make_bigram_perfect_map.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --val_shards 2,3 --out "$PERFECT_MAP"
fi

run_one() {  # run_one <gpu> <run_id> <extra args...>
  local GPU="$1" RUN_ID="$2"; shift 2
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  if [ -f "$RESULT_DIR/summary.json" ]; then
    echo "[ctbl] SKIP $RUN_ID (summary.json present)"; return 0
  fi
  mkdir -p "$RESULT_DIR"
  echo "[ctbl] $RUN_ID -> GPU $GPU at $(date)"
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
  echo "[ctbl] $RUN_ID done (exit=$?) at $(date)"
}

run_hash_sparse() {  # <R> <gpu>
  local R="$1" GPU="$2"
  run_one "$GPU" "ctbl_${R}_bigram" \
    --bigram_clean_table "$R" --val_steps 1000 --exact_freq_eval_interval 1000
  "$PY" -u "$ROOT/code/table_occupancy.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --epoch_batches 337 \
    --bigram_clean_table "$R" \
    --out "$OUT_DIR/ctbl_${R}_bigram_fixed/table_occupancy.json" \
    > "$OUT_DIR/ctbl_${R}_bigram_fixed/occupancy.log" 2>&1 \
    || echo "[ctbl] occupancy failed R=$R"
}

run_hash_curve() {  # R=1M collision control with freq=50 forking curve <gpu>
  run_one "$1" "ctbl_1048576_bigram" \
    --bigram_clean_table 1048576 \
    --val_interval 50 --freq_eval_interval 50 --exact_freq_eval_interval 100
}

run_perfect_curve() {  # zero-collision anchor with freq=50 forking curve <gpu>
  run_one "$1" "ctbl_perfect_bigram" \
    --bigram_clean_table 3538294 \
    --bigram_perfect_map "$PERFECT_MAP" \
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

# longest runs first
launch run_perfect_curve
launch run_hash_curve
for R in 4194304 3145728 2621440 2097152 1572864 786432 524288 393216 262144 131072 65536; do
  launch run_hash_sparse "$R"
done

while [ "$ACTIVE" -gt 0 ]; do wait -n; ACTIVE=$((ACTIVE - 1)); done
echo "=== clean-table grid done at $(date) ==="
