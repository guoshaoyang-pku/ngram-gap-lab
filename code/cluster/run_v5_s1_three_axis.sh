#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_scaling}"
VAL_SHARDS="2,3,4,5,6,7,8,9,10,6542"
GROUP="${S1_GROUP:?set S1_GROUP to table_size_bi, table_size_tri, frequency_main, or epoch_length}"
GPUS=("$@")

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "usage: S1_GROUP=<group> NGLAB_PY=python3 $0 <gpu> [gpu...]" >&2
  exit 2
fi

TABLE_ROWS=(16000 22000 30000 41000 56000 76000 104000 142000 194000 265000 362000 494000 675000 922000 1259000 1719000 2000000 2347000)
# small-R extension (2026-08-29): 1e4 → 1e0, ~1/3 decade spacing; extends the
# load factor K/R from ~221 (bigram) up to K itself (R=1, all contexts collide).
SMALL_TABLE_ROWS=(10000 4642 2154 1000 464 215 100 46 22 10 5 2 1)
SPECS=()

if [[ "$GROUP" == "frequency_main" ]]; then
  SPECS+=("s1v5_128_frequency_main|1000|--table_lr_scale 128.0 --val_steps 337,674,1000")
fi

case "$GROUP" in
  # —— 单表轴（用户拍板：只开被扫描的那张表，另一张关闭）——
  table_size_bi1)
    for rows in "${TABLE_ROWS[@]}"; do
      SPECS+=("s1v5_128_tbl_bi1_R${rows}|1000|--enable_bigram 1 --enable_trigram 0 --bigram_clean_table ${rows} --trigram_clean_table 0 --table_lr_scale 128.0 --val_steps 337,674,1000")
    done
    ;;
  table_size_tri1)
    for rows in "${TABLE_ROWS[@]}"; do
      SPECS+=("s1v5_128_tbl_tri1_R${rows}|1000|--enable_bigram 0 --enable_trigram 1 --bigram_clean_table 0 --trigram_clean_table ${rows} --table_lr_scale 128.0 --val_steps 337,674,1000")
    done
    ;;
  # —— 小 R 扩展轴（2026-08-29：R 从 1e4 扫到 1e0，观察 gap 是否塌缩到 no-gram 水平）——
  table_size_bi1_small)
    for rows in "${SMALL_TABLE_ROWS[@]}"; do
      SPECS+=("s1v5_128_tbl_bi1_R${rows}|1000|--enable_bigram 1 --enable_trigram 0 --bigram_clean_table ${rows} --trigram_clean_table 0 --table_lr_scale 128.0 --val_steps 337,674,1000")
    done
    ;;
  table_size_tri1_small)
    for rows in "${SMALL_TABLE_ROWS[@]}"; do
      SPECS+=("s1v5_128_tbl_tri1_R${rows}|1000|--enable_bigram 0 --enable_trigram 1 --bigram_clean_table 0 --trigram_clean_table ${rows} --table_lr_scale 128.0 --val_steps 337,674,1000")
    done
    ;;
  # —— 历史双表轴（superseded，保留脚本兼容）——
  table_size_bi)
    for rows in "${TABLE_ROWS[@]}"; do
      SPECS+=("s1v5_128_tbl_bi2_R${rows}|1000|--bigram_clean_table ${rows} --trigram_clean_table 1048576 --table_lr_scale 128.0 --val_steps 337,674,1000")
    done
    ;;
  table_size_tri)
    for rows in "${TABLE_ROWS[@]}"; do
      SPECS+=("s1v5_128_tbl_tri2_R${rows}|1000|--bigram_clean_table 1048576 --trigram_clean_table ${rows} --val_steps 337,674,1000")
    done
    ;;
  frequency_main) ;;
  # —— 单表 epoch 轴（用户拍板：优先只开 trigram）——
  epoch_length_tri)
    EPOCH_MULTS=(0p125 0p1667 0p25 0p3333 0p5 0p6667 0p75 1p0 1p25 1p5 1p75 2p0)
    EPOCH_BATCHES=(42 56 84 112 168 224 253 337 421 506 590 674)
    for i in "${!EPOCH_BATCHES[@]}"; do
      batches="${EPOCH_BATCHES[$i]}"
      multiplier="${EPOCH_MULTS[$i]}"
      steps=$((batches * 3))
      SPECS+=("s1v5_128_ep_tri_${multiplier}xL4_3ep|${steps}|--epoch_batches ${batches} --enable_bigram 0 --enable_trigram 1 --bigram_clean_table 0 --trigram_clean_table 1048576 --table_lr_scale 128.0 --val_steps ${batches},$((batches * 2)),${steps}")
    done
    SPECS+=(
      "s1v5_128_ep_tri_1xL4_10ep|3370|--epoch_batches 337 --enable_bigram 0 --enable_trigram 1 --bigram_clean_table 0 --trigram_clean_table 1048576 --table_lr_scale 128.0 --val_steps 337,674,1011,1348,1685,2022,2359,2696,3033,3370"
      "s1v5_128_ep_tri_1xL4_10ep_nogram|3370|--epoch_batches 337 --enable_bigram 0 --enable_trigram 0 --table_lr_scale 128.0 --val_steps 337,674,1011,1348,1685,2022,2359,2696,3033,3370"
    )
    ;;
  # —— 历史双表 epoch 轴（superseded，保留脚本兼容）——
  epoch_length)
    EPOCH_MULTS=(0p125 0p1667 0p25 0p3333 0p5 0p6667 0p75 1p0 1p25 1p5 1p75 2p0)
    EPOCH_BATCHES=(42 56 84 112 168 224 253 337 421 506 590 674)
    for i in "${!EPOCH_BATCHES[@]}"; do
      batches="${EPOCH_BATCHES[$i]}"
      multiplier="${EPOCH_MULTS[$i]}"
      steps=$((batches * 3))
      SPECS+=("s1v5_128_ep${multiplier}xL4_3ep|${steps}|--epoch_batches ${batches} --table_lr_scale 128.0 --val_steps ${batches},$((batches * 2)),${steps}")
    done
    SPECS+=(
      "s1v5_128_ep1xL4_10ep_both|3370|--epoch_batches 337 --table_lr_scale 128.0 --val_steps 337,674,1011,1348,1685,2022,2359,2696,3033,3370"
      "s1v5_128_ep1xL4_10ep_nogram|3370|--epoch_batches 337 --enable_bigram 0 --enable_trigram 0 --table_lr_scale 128.0 --val_steps 337,674,1011,1348,1685,2022,2359,2696,3033,3370"
    )
    ;;
  *)
    echo "unknown S1_GROUP=$GROUP" >&2
    exit 2
    ;;
esac

run_one() {
  local gpu="$1"
  local spec="$2"
  local run_id steps extra result_dir
  IFS='|' read -r run_id steps extra <<< "$spec"
  result_dir="$OUT_DIR/${run_id}_fixed"
  if [[ -f "$result_dir/summary.json" ]]; then
    echo "[s1-$GROUP] skip complete $run_id"
    return 0
  fi
  [[ ! -e "$result_dir" ]] || {
    echo "[s1-$GROUP] refusing partial directory $result_dir" >&2
    return 2
  }
  echo "[s1-$GROUP] $run_id gpu=$gpu steps=$steps"
  NGLAB_PY="$PY" NGLAB_OUT_DIR="$OUT_DIR" bash "$SCRIPT_DIR/run_v5_clean.sh" \
    "$gpu" "$run_id" 1 "$VAL_SHARDS" "$steps" \
    --table_lr_scale 128.0 \
    $extra
}

active=0
slot=0
gpu_count="${#GPUS[@]}"
pids=()
for spec in "${SPECS[@]}"; do
  while [[ "$active" -ge "$gpu_count" ]]; do
    wait "${pids[0]}" || echo "[s1-$GROUP] a run failed; continuing" >&2
    pids=("${pids[@]:1}")
    active=$((active - 1))
  done
  run_one "${GPUS[$slot]}" "$spec" &
  pids+=("$!")
  active=$((active + 1))
  slot=$(( (slot + 1) % gpu_count ))
done
while [[ "$active" -gt 0 ]]; do
  wait "${pids[0]}" || echo "[s1-$GROUP] a run failed; continuing" >&2
  pids=("${pids[@]:1}")
  active=$((active - 1))
done
echo "[s1-$GROUP] complete"