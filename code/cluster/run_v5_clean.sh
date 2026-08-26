#!/usr/bin/env bash
# V5 clean-table launcher: 6e-4 backbone LR, table ×2, RMSProp (0,0.99),
# 100-step warmup_constant, bf16 without torch.compile.
set -euo pipefail

GPU="${1:?gpu id}"
RUN_ID="${2:?run id without _fixed}"
TRAIN_SHARDS="${3:?train shards}"
VAL_SHARDS="${4:?validation shards}"
STEPS="${5:?steps}"
shift 5

case "$RUN_ID" in
  *_fixed) echo "run_id must not include _fixed" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="${NGLAB_DATA_DIR:-$ROOT/data/tokenized}"
OUT_DIR="${NGLAB_OUT_DIR:-$ROOT/data/runs_fixed}"
RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"

if [[ "$PY" != */* ]]; then
  PY="$(command -v "$PY")"
fi

pick_freq_index() {
  local compact="${TRAIN_SHARDS// /}"
  local name="freq_index"
  local dose_index=0
  case "$compact" in
    62) name="freq_index_train0_25x"; dose_index=1 ;;
    60) name="freq_index_train0_5x"; dose_index=1 ;;
    63) name="freq_index_train0_75x"; dose_index=1 ;;
    1,61) name="freq_index_train1_5x"; dose_index=1 ;;
    1,2) name="freq_index_train2x_fine"; dose_index=1 ;;
    1,2,64) name="freq_index_train2_5x"; dose_index=1 ;;
    1,2,3) name="freq_index_train3x"; dose_index=1 ;;
    1,2,3,4) name="freq_index_train4x"; dose_index=1 ;;
    1,2,3,4,5) name="freq_index_train5x"; dose_index=1 ;;
    1,2,3,4,5,6) name="freq_index_train6x"; dose_index=1 ;;
    1,2,3,4,5,6,7,8) name="freq_index_train8x"; dose_index=1 ;;
  esac
  if [[ -f "$ROOT/data/${name}.npz" ]]; then
    printf '%s\n' "$ROOT/data/${name}.npz"
  elif [[ "$dose_index" -eq 1 ]]; then
    echo "missing dose-specific frequency index: $ROOT/data/${name}.npz" >&2
    return 2
  else
    printf '%s\n' "$ROOT/data/freq_index.npz"
  fi
}

FREQ_INDEX="${NGLAB_FREQ_INDEX:-$(pick_freq_index)}"
[[ -x "$PY" ]] || { echo "python unavailable: $PY" >&2; exit 2; }
[[ -d "$DATA_DIR" ]] || { echo "data unavailable: $DATA_DIR" >&2; exit 2; }
[[ -f "$FREQ_INDEX" ]] || { echo "frequency index unavailable: $FREQ_INDEX" >&2; exit 2; }
[[ ! -e "$RESULT_DIR" ]] || { echo "refusing existing run: $RESULT_DIR" >&2; exit 2; }

mkdir -p "$RESULT_DIR"
echo "[v5] run=$RUN_ID gpu=$GPU steps=$STEPS train=$TRAIN_SHARDS"

  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${RUN_ID}_fixed" \
  --out_dir "$OUT_DIR" \
  --data_dir "$DATA_DIR" \
  --train_shards "$TRAIN_SHARDS" \
  --val_shards "$VAL_SHARDS" \
  --seed 42 \
  --steps "$STEPS" \
  --dtype bf16 \
  --injection_position input \
  --enable_unigram 0 \
  --enable_bigram 1 \
  --enable_trigram 1 \
  --bigram_clean_table 1048576 \
  --trigram_clean_table 1048576 \
  --n_layer 8 \
  --n_head 6 \
  --n_embd 768 \
  --vocab_size 8192 \
  --sequence_len 2048 \
  --device_batch_size 72 \
  --total_batch_size 147456 \
  --lr 0.0006 \
  --lr_schedule warmup_constant \
  --warmup_steps 100 \
  --table_optimizer rmsprop \
  --table_betas 0.0,0.99 \
  --table_lr_scale 2.0 \
  --val_interval 10 \
  --val_batches 4 \
  --freq_index "$FREQ_INDEX" \
  --freq_eval_interval 10 \
  --freq_eval_batches 4 \
  --exact_freq_eval_interval 10 \
  --table_norm_interval 10 \
  --fixed_train_probe 0 \
  "$@" \
  > "$RESULT_DIR/train.log" 2>&1

test -s "$RESULT_DIR/summary.json"
echo "[v5] complete: $RESULT_DIR"