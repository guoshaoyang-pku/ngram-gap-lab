#!/usr/bin/env bash
# Reproducibility gate for the LR-schedule confound discovered in v4.
#
# This runs one clean-table input anchor three times on ONE GPU. The only
# differing flag is --lr_schedule / --warmup_steps. Scalar validation is fixed
# and sampled at four checkpoints; frequency diagnostics are intentionally off
# because this is a convergence gate, not a frequency-decomposition result.
# It does not replace a registered main-line experiment or overwrite any prior
# run.
#
# Usage:
#   bash code/cluster/run_schedule_compare.sh <gpu>
set -euo pipefail

GPU="${1:?gpu id required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
if [ -n "${NGLAB_PY:-}" ]; then
  PY="$NGLAB_PY"
  if [ ! -x "$PY" ]; then
    PY="$(command -v "$PY" || true)"
  fi
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3)"
fi
DATA_DIR="${NGLAB_DATA_DIR:-$ROOT/data/tokenized}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_scaling}"

if [ ! -x "$PY" ] || [ ! -d "$DATA_DIR" ]; then
  echo "missing python interpreter or tokenized data" >&2
  exit 2
fi

run_one() {  # <label> <schedule> <warmup_steps>
  local LABEL="$1" SCHEDULE="$2" WARMUP_STEPS="$3"
  local STEM="schedcheck_v5_${LABEL}_r1048576_both_s42"
  local RESULT_DIR="$OUT_DIR/${STEM}_fixed"
  if [ -e "$RESULT_DIR" ]; then
    echo "refusing to overwrite existing result: $RESULT_DIR" >&2
    exit 2
  fi
  mkdir -p "$RESULT_DIR"
  echo "[schedule-check] $STEM schedule=$SCHEDULE warmup_steps=$WARMUP_STEPS"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${STEM}_fixed" --out_dir "$OUT_DIR" \
    --data_dir "$DATA_DIR" --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --seed 42 --steps 1000 --epoch_batches 337 --dtype bf16 \
    --injection_position input --enable_unigram 0 --enable_bigram 1 --enable_trigram 1 \
    --bigram_clean_table 1048576 --trigram_clean_table 1048576 \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --total_batch_size 147456 \
    --lr 0.004 --lr_schedule "$SCHEDULE" --warmup_steps "$WARMUP_STEPS" \
    --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 2.0 \
    --val_steps 100,250,500,1000 --val_batches 4 \
    --table_norm_interval 100 --fixed_train_probe 0 \
    > "$RESULT_DIR/train.log" 2>&1
  test -s "$RESULT_DIR/summary.json"
  test -s "$RESULT_DIR/train_log.jsonl"
}

# Historical schedule / failed zero-warmup control / proposed standard.
run_one warmdown warmdown 100
run_one constant constant 100
run_one warmup100 warmup_constant 100
