#!/usr/bin/env bash
# Pure backbone-LR arm: vary backbone LR while locking the absolute table LR.
# The base launcher implements table_lr(t) = backbone_lr(t) * table_lr_scale,
# so this wrapper sets table_lr_scale = fixed_table_lr / backbone_lr.
set -euo pipefail

GPU="${1:?gpu id}"
RUN_ID="${2:?run id without _fixed}"
BACKBONE_LR="${3:?backbone lr}"
STEPS="${4:?steps}"
ARM="${5:?arm: input or nogram}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABLE_LR_ABS="${NGLAB_TABLE_LR_ABS:-0.0768}"
TABLE_LR_SCALE="$({ awk -v t="$TABLE_LR_ABS" -v b="$BACKBONE_LR" '
  BEGIN {
    if (b <= 0 || t <= 0) exit 2
    printf "%.12g", t / b
  }
'; })"

case "$ARM" in
  input)
    ARM_ARGS=(
      --enable_bigram 1
      --enable_trigram 1
      --bigram_clean_table 1048576
      --trigram_clean_table 1048576
    )
    ;;
  nogram)
    ARM_ARGS=(
      --enable_bigram 0
      --enable_trigram 0
      --bigram_clean_table 0
      --trigram_clean_table 0
    )
    ;;
  *)
    echo "arm must be input or nogram, got: $ARM" >&2
    exit 2
    ;;
esac

ACTUAL_TABLE_LR="$(awk -v b="$BACKBONE_LR" -v s="$TABLE_LR_SCALE" 'BEGIN { printf "%.12g", b * s }')"
echo "[blr-abslock] run=$RUN_ID arm=$ARM gpu=$GPU backbone_lr=$BACKBONE_LR table_lr_abs=$ACTUAL_TABLE_LR table_lr_scale=$TABLE_LR_SCALE steps=$STEPS"

NGLAB_TABLE_LR_SCALE="$TABLE_LR_SCALE" \
  bash "$SCRIPT_DIR/run_v5_clean.sh" \
  "$GPU" "$RUN_ID" \
  "1" "2,3,4,5,6,7,8,9,10,6542" "$STEPS" \
  --lr "$BACKBONE_LR" \
  --epoch_batches 337 \
  --val_interval 50 \
  --freq_eval_interval 50 \
  --exact_freq_eval_interval 50 \
  --table_norm_interval 50 \
  "${ARM_ARGS[@]}"
