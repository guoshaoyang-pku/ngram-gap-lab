#!/usr/bin/env bash
# Short-epoch x beta2 ablation (v10, input injection, 2000 steps, seed 42).
# Goal: does beta2=0.99 change the per-epoch stair-step clarity at high replay
# frequency (0.25x / 0.5x epoch) vs the §10 beta2=0.999 reference runs?
#
#   0.25x : train shard 62  (~55 steps/epoch obs), val 2..10,6542, freq idx 0.25x
#   0.5x  : train shard 60  (~120 steps/epoch obs), val 2..10,6542, freq idx 0.5x
#   beta2 : 0.99 (beta1=0.0, RMSProp, no momentum)
#
# Usage: ./run_epoch_short_b2.sh <arm> <gpu> [steps] [seed]
set -euo pipefail

ARM="${1:?arm: 025x_b2_099|05x_b2_099}"
GPU="${2:?gpu_id required}"
STEPS="${3:-2000}"
SEED="${4:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"

case "$ARM" in
  025x_b2_099) SHARDS=62;   FREQ="$ROOT/data/freq_index_train0_25x.npz"; B2=0.99;;
  05x_b2_099)  SHARDS=60;   FREQ="$ROOT/data/freq_index_train0_5x.npz";  B2=0.99;;
  *) echo "unknown arm: $ARM" >&2; exit 1;;
esac

EXP="nglab${ARM}"
RESULT_DIR="$ROOT/data/runs_fixed/${EXP}_fixed"
mkdir -p "$RESULT_DIR"

echo "=== $EXP (shards=$SHARDS, beta2=$B2) on GPU $GPU at $(date) ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "${EXP}_fixed" \
  --injection_position input \
  --steps "$STEPS" \
  --seed "$SEED" \
  --data_dir "$DATA_DIR" \
  --out_dir "$ROOT/data/runs_fixed" \
  --train_shards "$SHARDS" \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 \
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
  --freq_index "$FREQ" \
  --freq_eval_interval 10 \
  --freq_eval_batches 4 \
  --table_optimizer rmsprop \
  --table_betas 0.0,$B2 \
  --table_lr_scale 2.0 \
  > "$RESULT_DIR/train.log" 2>&1
echo "=== $EXP exit=$? at $(date) ==="
