#!/usr/bin/env bash
# [HISTORICAL LEGACY TABLE] Table-optimizer ablation for the old blog setting
# (input injection, 1x shard). It does not select clean-table R; retain it for
# provenance only, not as a new main-line optimizer baseline.
# Goal: n-gram table RMSProp (no momentum) learns slowly/plateaus; test faster alternatives.
# All arms share: input injection, train shard 1, val 3..10,6542, seed 42, steps (default 1000),
# val + freq eval + table norm every 10 steps. Only the TABLE optimizer differs.
#
#   rmsprop_2x     : RMSProp (beta2=0.999), table LR x2            -> pure LR speedup
#   adamw_090999   : AdamW (0.9, 0.999) on table                   -> momentum + adaptive
#   adamw_080950   : AdamW (0.8, 0.95) on table (backbone betas)   -> momentum, aligned w/ backbone
#   sgd_09         : SGD momentum 0.9 on table                     -> momentum only (no adaptive)
#
# Baseline: nglab1x_v10_input (v10/2000, in flight) and nglab_input (v50/1000, done).
#
# Usage: ./run_table_opt.sh <arm> <gpu> [steps] [seed]
set -euo pipefail

ARM="${1:?arm required: rmsprop_2x|rmsprop_4x|adamw_090999|adamw_080950|sgd_09}"
GPU="${2:?gpu_id required}"
STEPS="${3:-1000}"
SEED="${4:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"

case "$ARM" in
  rmsprop_2x)   EXTRA=(--table_optimizer rmsprop --table_lr_scale 2.0);;
  rmsprop_4x)   EXTRA=(--table_optimizer rmsprop --table_lr_scale 4.0);;
  adamw_090999) EXTRA=(--table_optimizer adamw --table_betas 0.9,0.999);;
  adamw_080950) EXTRA=(--table_optimizer adamw --table_betas 0.8,0.95);;
  sgd_09)       EXTRA=(--table_optimizer sgd --table_betas 0.9,0.0);;
  rmsprop_2x_b2_099) EXTRA=(--table_optimizer rmsprop --table_lr_scale 2.0 --table_betas 0.0,0.99);;
  rmsprop_4x_b2_099) EXTRA=(--table_optimizer rmsprop --table_lr_scale 4.0 --table_betas 0.0,0.99);;
  rmsprop_2x_b2_098) EXTRA=(--table_optimizer rmsprop --table_lr_scale 2.0 --table_betas 0.0,0.98);;
  rmsprop_4x_b2_098) EXTRA=(--table_optimizer rmsprop --table_lr_scale 4.0 --table_betas 0.0,0.98);;
  *) echo "unknown arm: $ARM" >&2; exit 1;;
esac

if [ "$SEED" = "42" ]; then EXP="nglab1x_opt_$ARM"; else EXP="nglab1x_opt_${ARM}_s${SEED}"; fi
RESULT_DIR="$ROOT/data/runs_fixed/${EXP}_fixed"
mkdir -p "$RESULT_DIR"

echo "=== $EXP on GPU $GPU at $(date) ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${EXP}_fixed" \
  --injection_position input \
  --steps "$STEPS" \
  --seed "$SEED" \
  --data_dir "$DATA_DIR" \
  --out_dir "$ROOT/data/runs_fixed" \
  --train_shards 1 \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --device_batch_size 72 \
  --total_batch_size 147456 \
  --val_interval 10 \
  --val_batches 4 \
  --table_norm_interval 10 \
  --lr 0.004 \
  --lr_schedule warmdown \
  --enable_unigram 0 \
  --enable_bigram 1 \
  --enable_trigram 1 \
  --n_layer 8 \
  --n_head 6 \
  --n_embd 768 \
  --vocab_size 8192 \
  --sequence_len 2048 \
  --freq_index "$ROOT/data/freq_index.npz" \
  --freq_eval_interval 10 \
  --freq_eval_batches 4 \
  "${EXTRA[@]}" \
  > "$RESULT_DIR/train.log" 2>&1
echo "=== $EXP exit=$? at $(date) ==="
