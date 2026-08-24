#!/usr/bin/env bash
# Full epoch-length scaling grid (plan §3) on 360-2.  Run AFTER pilot QC.
#
# All runs: vanilla nanoGPT 8L·6H·768D + input n-gram injection,
# natural corpus nested-prefix shard 1, train/val zero overlap,
# table RMSProp(0.0, 0.99) / backbone AdamW(0.8, 0.95), online gap.
#
# User decisions 2026-08-24:
#   - L4 = 337 batches/epoch = FULL shard 1 (24264 chunks / 72). L1-L3 are
#     nested prefixes (42 / 84 / 168). L4 is no longer 336.
#   - Ordinary grid runs do NOT compute exact-frequency / freq-bin
#     diagnostics (no --freq_index).  The frequency axis is a separate
#     small run set (see run_scaling_frequency_axis.sh).
#
# Fixed-step (fs): 1000 steps, step-anchored LR, all L1-L4 x 4 modules.
# Fixed-epoch (fe): 6 full epochs, epoch-anchored LR (--lr_schedule_epochs 6),
#   target steps L1=252 L2=504 L3=1008 L4=2022.  no-ngram is run at
#   every L so each n-gram arm has a same-L baseline.
#
# Usage: ./run_scaling_epoch_full.sh <gpu_id1> [gpu_id2] [gpu_id3] [gpu_id4] ...
#   Pass the CUDA device ids of the free GPUs (any number >= 1).  Arms are
#   scheduled round-robin across the given GPUs; a full wave finishes when
#   every arm of the current L has been launched on some GPU.
#
# Optional env:
#   SEED=42|43|44         training seed (default: 42)
#   MONITOR=dense|sparse  dense logs every 10 steps; sparse only final step
#   NGLAB_PY=python3      python interpreter on the cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then GPUS=(0 1 2 3); fi
mkdir -p "$OUT_DIR"

SEED="${SEED:-42}"
MONITOR="${MONITOR:-dense}"
if [ "$MONITOR" != "dense" ] && [ "$MONITOR" != "sparse" ]; then
  echo "[epoch-full] MONITOR must be dense or sparse" >&2
  exit 2
fi
if [ "$SEED" = "42" ]; then
  RUN_SUFFIX=""
else
  RUN_SUFFIX="_s${SEED}"
fi

# Nested-prefix epoch lengths (device batches per epoch).
# L4 = 337 = full shard 1 (24264 chunks / 72). L1/L2/L3 are 1/8, 1/4, 1/2 of it.
declare -A EPB=( [L1]=42 [L2]=84 [L3]=168 [L4]=337 )
declare -A FESTEPS=( [L1]=252 [L2]=504 [L3]=1008 [L4]=2022 )

run_arm() {  # run_arm <gpu> <run_id> <epoch_batches> <steps> <schedule_epochs> <bigram> <trigram>
  local GPU="$1" RUN_ID="$2" EPB="$3" STEPS="$4" SCHED="$5" BI="$6" TRI="$7"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  if [ -f "$RESULT_DIR/summary.json" ]; then
    echo "[epoch-full] SKIP $RUN_ID (summary.json already present)"
    return 0
  fi
  mkdir -p "$RESULT_DIR"
  if [ "$MONITOR" = "sparse" ]; then
    EVAL_ARGS=(--val_steps "$STEPS" --table_norm_interval "$STEPS")
  else
    EVAL_ARGS=(--val_interval 10 --table_norm_interval 10)
  fi
  echo "[epoch-full] $RUN_ID epb=$EPB steps=$STEPS sched=$SCHED bi=$BI tri=$TRI seed=$SEED monitor=$MONITOR -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps "$STEPS" --seed "$SEED" \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_batches 4 --lr 0.004 \
    --enable_unigram 0 --enable_bigram "$BI" --enable_trigram "$TRI" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --epoch_batches "$EPB" \
    --fixed_train_probe 0 \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    --dtype bf16 \
    "${EVAL_ARGS[@]}" \
    ${SCHED:+--lr_schedule_epochs "$SCHED"} \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[epoch-full] $RUN_ID done (exit=$?) at $(date)"
}

# run_wave <label> <schedule_epochs> <steps_fn> -- schedule every arm across
# the GPUS list with a slot counter so that at most one training process runs
# per GPU at any time (avoids CUDA OOM when len(GPUS) < #arms).
run_wave() {
  local LABEL="$1" SCHED="$2" STEPS_FN="$3"
  local NGPU=${#GPUS[@]}
  local ACTIVE=0                     # number of training jobs currently running
  local SLOT=0                       # next GPU slot to use (round-robin)
  local L ARM BI TRI STEPS GPU
  for L in L1 L2 L3 L4; do
    for ARM in bigram trigram both nogram; do
      case "$ARM" in
        bigram) BI=1 TRI=0 ;;
        trigram) BI=0 TRI=1 ;;
        both) BI=1 TRI=1 ;;
        nogram) BI=0 TRI=0 ;;
      esac
      # Wait until a GPU slot is free (at most NGPU concurrent jobs).
      while [ "$ACTIVE" -ge "$NGPU" ]; do
        wait -n
        ACTIVE=$((ACTIVE - 1))
      done
      GPU="${GPUS[$SLOT]}"
      SLOT=$(( (SLOT + 1) % NGPU ))
      STEPS=$("$STEPS_FN" "$L")
      run_arm "$GPU" "ep_${L}_${ARM}_${LABEL}${RUN_SUFFIX}" "${EPB[$L]}" "$STEPS" "$SCHED" "$BI" "$TRI" &
      ACTIVE=$((ACTIVE + 1))
    done
    # Wait for all arms of this L to finish before the next L (keeps the
    # no-ngram baseline of each L on the same timeline).
    while [ "$ACTIVE" -gt 0 ]; do
      wait -n
      ACTIVE=$((ACTIVE - 1))
    done
    SLOT=0
  done
}

steps_fs() { echo 1000; }
steps_fe() { echo "${FESTEPS[$1]}"; }

# Fixed-step: L1-L4 x {bigram, trigram, both, no-ngram}, 1000 steps
run_wave fs 0 steps_fs
# Fixed-epoch: L1-L4 x {bigram, trigram, both, no-ngram}, 6 epochs each
run_wave fe 6 steps_fe

echo "=== epoch full grid done at $(date) ==="
