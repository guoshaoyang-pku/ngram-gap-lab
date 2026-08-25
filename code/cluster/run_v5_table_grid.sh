#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
GPUS=("$@")

if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "usage: NGLAB_PY=python3 $0 <gpu> [gpu...]" >&2
  exit 2
fi

BIGRAM=(16000 22000 30000 41000 56000 76000 104000 142000 194000 265000 362000 494000 675000 922000 1259000 1719000 2347000 3206000 4380000 5980000)
TRIGRAM=(65536 86000 113000 149000 196000 258000 339000 446000 587000 772000 1016000 1336000 1757000 2310000 3039000 4000000 5260000 7000000)

run_one() {
  local gpu="$1"
  local run_id="$2"
  local branch="$3"
  local rows="$4"
  local result_dir="$ROOT/data/runs_scaling/${run_id}_fixed"
  if [[ -f "$result_dir/summary.json" ]]; then
    echo "[v5-table] skip complete $run_id"
    return 0
  fi
  [[ ! -e "$result_dir" ]] || {
    echo "[v5-table] refusing partial directory $result_dir" >&2
    return 2
  }
  mkdir -p "$result_dir"
  local branch_args=()
  if [[ "$branch" == "bigram" ]]; then
    branch_args=(--enable_bigram 1 --enable_trigram 0 --bigram_clean_table "$rows")
  else
    branch_args=(--enable_bigram 0 --enable_trigram 1 --trigram_clean_table "$rows")
  fi
  echo "[v5-table] $run_id gpu=$gpu R=$rows branch=$branch"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/code/train.py" \
    --run_id "${run_id}_fixed" \
    --out_dir "$ROOT/data/runs_scaling" \
    --data_dir "$ROOT/data/tokenized" \
    --train_shards 1 \
    --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --seed 42 \
    --steps 1000 \
    --dtype bf16 \
    --injection_position input \
    --enable_unigram 0 \
    "${branch_args[@]}" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --total_batch_size 147456 \
    --lr 0.0006 \
    --lr_schedule warmup_constant --warmup_steps 100 \
    --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 2.0 \
    --val_steps 1000 --val_batches 4 \
    --freq_index "$ROOT/data/freq_index.npz" \
    --freq_eval_interval 1000 --freq_eval_batches 4 \
    --exact_freq_eval_interval 1000 --table_norm_interval 1000 \
    --fixed_train_probe 0 \
    > "$result_dir/train.log" 2>&1
  test -s "$result_dir/summary.json"
}

SPECS=()
for rows in "${BIGRAM[@]}"; do
  SPECS+=("ctbl_v5_${rows}_bigram|bigram|$rows")
done
for rows in "${TRIGRAM[@]}"; do
  SPECS+=("ctbl_v5_${rows}_trigram|trigram|$rows")
done

active=0
slot=0
gpu_count="${#GPUS[@]}"
for spec in "${SPECS[@]}"; do
  while [[ "$active" -ge "$gpu_count" ]]; do
    if ! wait -n; then
      echo "[v5-table] a run failed; continuing remaining queue" >&2
    fi
    active=$((active - 1))
  done
  IFS='|' read -r run_id branch rows <<< "$spec"
  run_one "${GPUS[$slot]}" "$run_id" "$branch" "$rows" &
  active=$((active + 1))
  slot=$(( (slot + 1) % gpu_count ))
done
while [[ "$active" -gt 0 ]]; do
  if ! wait -n; then
    echo "[v5-table] a run failed; continuing remaining queue" >&2
  fi
  active=$((active - 1))
done
echo "[v5-table] complete"