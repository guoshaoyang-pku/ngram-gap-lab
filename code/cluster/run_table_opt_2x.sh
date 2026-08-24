#!/usr/bin/env bash
# Table-optimizer ablation on the 2x-epoch setting (train shards 1,2; val 3..10,6542).
# Mirrors run_table_opt.sh but with train shards 1,2 and 2000 steps (default), so the
# "epoch length x2, 2000 steps" LR/beta2 curves can be compared against the 1x arms.
#
#   rmsprop_1x            : RMSProp lr x1, beta2=0.999          (baseline, cf. nglab2x_input_v10_fv)
#   rmsprop_2x / _4x      : RMSProp lr x2 / x4, beta2=0.999
#   *_b2_09999            : same LR, beta2=0.9999 (longer grad^2 EMA window)
#   *_b2_099999           : same LR, beta2=0.99999 (even longer)
#
# Usage: ./run_table_opt_2x.sh <arm> <gpu> [steps] [seed]
set -euo pipefail

ARM="${1:?arm required}"
GPU="${2:?gpu_id required}"
STEPS="${3:-2000}"
SEED="${4:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
FREQ_IDX="$ROOT/data/freq_index_train2x_fine.npz"

case "$ARM" in
  rmsprop_1x)          EXTRA=(--table_optimizer rmsprop --table_lr_scale 1.0 --table_betas 0.0,0.999);;
  rmsprop_2x)          EXTRA=(--table_optimizer rmsprop --table_lr_scale 2.0 --table_betas 0.0,0.999);;
  rmsprop_4x)          EXTRA=(--table_optimizer rmsprop --table_lr_scale 4.0 --table_betas 0.0,0.999);;
  rmsprop_1x_b2_09999) EXTRA=(--table_optimizer rmsprop --table_lr_scale 1.0 --table_betas 0.0,0.9999);;
  rmsprop_2x_b2_09999) EXTRA=(--table_optimizer rmsprop --table_lr_scale 2.0 --table_betas 0.0,0.9999);;
  rmsprop_4x_b2_09999) EXTRA=(--table_optimizer rmsprop --table_lr_scale 4.0 --table_betas 0.0,0.9999);;
  rmsprop_2x_b2_099999) EXTRA=(--table_optimizer rmsprop --table_lr_scale 2.0 --table_betas 0.0,0.99999);;
  rmsprop_2x_b2_099)   EXTRA=(--table_optimizer rmsprop --table_lr_scale 2.0 --table_betas 0.0,0.99);;
  rmsprop_4x_b2_099)   EXTRA=(--table_optimizer rmsprop --table_lr_scale 4.0 --table_betas 0.0,0.99);;
  rmsprop_2x_b2_098)   EXTRA=(--table_optimizer rmsprop --table_lr_scale 2.0 --table_betas 0.0,0.98);;
  rmsprop_4x_b2_098)   EXTRA=(--table_optimizer rmsprop --table_lr_scale 4.0 --table_betas 0.0,0.98);;
  *) echo "unknown arm: $ARM" >&2; exit 1;;
esac

if [ "$SEED" = "42" ]; then EXP="nglab2x_opt_$ARM"; else EXP="nglab2x_opt_${ARM}_s${SEED}"; fi
RESULT_DIR="$ROOT/data/runs_fixed/${EXP}_fixed"
mkdir -p "$RESULT_DIR"

echo "=== $EXP on GPU $GPU at $(date) ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "$EXP" \
  --injection_position input \
  --steps "$STEPS" \
  --seed "$SEED" \
  --data_dir "$DATA_DIR" \
  --out_dir "$ROOT/data/runs_fixed" \
  --train_shards 1,2 \
  --val_shards 3,4,5,6,7,8,9,10,6542 \
  --device_batch_size 72 \
  --total_batch_size 147456 \
  --val_interval 10 \
  --val_batches 4 \
  --table_norm_interval 10 \
  --lr 0.004 \
  --enable_unigram 0 \
  --enable_bigram 1 \
  --enable_trigram 1 \
  --n_layer 8 \
  --n_head 6 \
  --n_embd 768 \
  --vocab_size 8192 \
  --sequence_len 2048 \
  --freq_index "$FREQ_IDX" \
  --freq_eval_interval 10 \
  --freq_eval_batches 4 \
  "${EXTRA[@]}" \
  > "$RESULT_DIR/train.log" 2>&1
echo "=== $EXP exit=$? at $(date) ==="
