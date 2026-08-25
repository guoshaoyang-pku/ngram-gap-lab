#!/usr/bin/env bash
# One create-only arm of the clean-table LR-schedule quality gate.
#
# Usage:
#   bash code/cluster/run_schedule_search.sh <gpu> <label> <schedule> <warmup_steps>
#
# This intentionally enables only scalar checkpoints: it ranks schedule
# candidates by convergence first, before any full frequency decomposition.
set -euo pipefail

GPU="${1:?gpu id required}"
LABEL="${2:?unique label required}"
SCHEDULE="${3:?lr schedule required}"
WARMUP_STEPS="${4:?warmup steps required}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
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
STEM="schedgrid_v1_${LABEL}_r1048576_both_s42"
RESULT_DIR="$OUT_DIR/${STEM}_fixed"

case "$SCHEDULE" in
  warmup_constant|warmup_cosine|warmdown) ;;
  *)
    echo "schedule search accepts warmup_constant, warmup_cosine, or warmdown, got: $SCHEDULE" >&2
    exit 2
    ;;
esac
if [ ! -x "$PY" ] || [ ! -d "$DATA_DIR" ] || [ -e "$RESULT_DIR" ]; then
  echo "missing interpreter/data or refusing to overwrite: $RESULT_DIR" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
echo "[schedule-grid] $STEM schedule=$SCHEDULE warmup_steps=$WARMUP_STEPS"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${STEM}_fixed" --out_dir "$OUT_DIR" \
  --data_dir "$DATA_DIR" --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --seed 42 --steps 1000 --epoch_batches 337 --dtype bf16 \
  --injection_position input --enable_unigram 0 --enable_bigram 1 --enable_trigram 1 \
  --bigram_clean_table 1048576 --trigram_clean_table 1048576 \
  --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
  --device_batch_size 72 --total_batch_size 147456 \
  --lr 0.004 --lr_schedule "$SCHEDULE" --warmup_steps "$WARMUP_STEPS" \
  --cosine_min_lr_mult 0.05 \
  --table_optimizer rmsprop --table_betas 0.0,0.99 --table_lr_scale 2.0 \
  --val_steps 100,250,500,750,1000 --val_batches 4 \
  --table_norm_interval 100 --fixed_train_probe 0 \
  > "$RESULT_DIR/train.log" 2>&1
test -s "$RESULT_DIR/summary.json"
test -s "$RESULT_DIR/train_log.jsonl"
