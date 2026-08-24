#!/usr/bin/env bash
# Full table-size scaling grid (plan §5) on 360-2.  Run AFTER pilot QC.
#
# Downward-only from default 1M logical addresses (2R).  Two monitoring modes:
#   MONITOR=dense (default): val/table-norm every 10 steps (full curve).
#   MONITOR=sparse:         val/table-norm ONLY at the final step
#                            (--val_steps <max_steps>).  ~10x faster per run;
#                            you only get the final online gap, no curve.
#
# Table sizes (log-spaced "encrypted sampling"): mult in
#   64 48 32 24 16 12 8 6 4 3 2 1   -> 12 sizes per module (36 runs).
# (table size = vocab_size * mult; hash uses % size so any integer mult works.)
# L4 = 337 batches/epoch (FULL shard 1), 1000 steps, step-anchored LR,
# beta2=0.99.  Ordinary grid runs do NOT compute exact-frequency / freq-bin
# diagnostics (no --freq_index); occupancy is still emitted per run.
#
# Usage: ./run_scaling_table_full.sh <gpu_id1> [gpu_id2] ...
#   Pass the CUDA device ids of the free GPUs (any number >= 1).  The module
#   arms for each table size are scheduled with a slot counter so that at most
#   one training process runs per GPU at any time.
#
# Optional env:
#   MONITOR=sparse            only evaluate at the final step (fast)
#   TABLE_MULTS="64 48 ..."   override the mult list
#   ARMS="bigram trigram"     override the module arms (default: bigram trigram both)
#   SEED=42|43|44             training seed (default: 42)
#   NGLAB_PY=python3          python interpreter on the cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
GPUS=("$@")
if [ "${#GPUS[@]}" -eq 0 ]; then GPUS=(0 1 2); fi
mkdir -p "$OUT_DIR"

MONITOR="${MONITOR:-dense}"
TABLE_MULTS="${TABLE_MULTS:-64 48 32 24 16 12 8 6 4 3 2 1}"
ARMS="${ARMS:-bigram trigram both}"
SEED="${SEED:-42}"
if [ "$MONITOR" != "dense" ] && [ "$MONITOR" != "sparse" ]; then
  echo "[table-full] MONITOR must be dense or sparse" >&2
  exit 2
fi
if [ "$SEED" = "42" ]; then
  RUN_SUFFIX=""
else
  RUN_SUFFIX="_s${SEED}"
fi

# 360-2 /tmp is tiny (974MB) and torch.compile writes there by default.
# Redirect the inductor cache into the repo (gitignored) and isolate it per
# run so concurrent compiles do not clobber each other.
CACHE_ROOT="${TORCHINDUCTOR_CACHE_DIR:-$ROOT/.inductor_cache}"

run_arm() {  # run_one <gpu> <run_id> <table_mult> <bigram> <trigram> <max_steps>
  local GPU="$1" RUN_ID="$2" TM="$3" BI="$4" TRI="$5" MAX_STEPS="$6"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  if [ -f "$RESULT_DIR/summary.json" ] && [ -f "$RESULT_DIR/table_occupancy.json" ]; then
    echo "[table-full] SKIP $RUN_ID (summary + occupancy already present)"
    return 0
  fi
  mkdir -p "$RESULT_DIR"
  echo "[table-full] $RUN_ID mult=$TM bi=$BI tri=$TRI seed=$SEED monitor=$MONITOR -> GPU $GPU at $(date)"
  if [ "$MONITOR" = "sparse" ]; then
    # only final-step eval -> ~10x faster
    EVAL_ARGS=(--val_steps "$MAX_STEPS" --table_norm_interval "$MAX_STEPS")
  else
    EVAL_ARGS=(--val_interval 10 --table_norm_interval 10)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/${RUN_ID}" \
  "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps "$MAX_STEPS" --seed "$SEED" \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_batches 4 --lr 0.004 \
    --enable_unigram 0 --enable_bigram "$BI" --enable_trigram "$TRI" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --epoch_batches 337 \
    --fixed_train_probe 0 \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    --table_mult "$TM" \
    --dtype bf16 \
    "${EVAL_ARGS[@]}" \
    > "$RESULT_DIR/train.log" 2>&1
  "$PY" -u "$TASK_ROOT/code/table_occupancy.py" \
    --data_dir "$DATA_DIR" --train_shards 1 \
    --vocab_size 8192 --sequence_len 2048 \
    --device_batch_size 72 --epoch_batches 337 \
    --table_mult "$TM" --out "$RESULT_DIR/table_occupancy.json" \
    > "$RESULT_DIR/occupancy.log" 2>&1 || echo "[table-full] occupancy failed $RUN_ID"
  echo "[table-full] $RUN_ID done at $(date)"
}

# Slot scheduling: at most one training job per GPU at any time.
NGPU=${#GPUS[@]}
ACTIVE=0
SLOT=0
launch() {  # launch <run_id> <table_mult> <bigram> <trigram> <max_steps>
  local RUN_ID="$1" TM="$2" BI="$3" TRI="$4" MAX_STEPS="$5"
  while [ "$ACTIVE" -ge "$NGPU" ]; do
    wait -n
    ACTIVE=$((ACTIVE - 1))
  done
  local GPU="${GPUS[$SLOT]}"
  SLOT=$(( (SLOT + 1) % NGPU ))
  run_arm "$GPU" "$RUN_ID" "$TM" "$BI" "$TRI" "$MAX_STEPS" &
  ACTIVE=$((ACTIVE + 1))
}

for TM in $TABLE_MULTS; do
  for ARM in $ARMS; do
    case "$ARM" in
      bigram) BI=1 TRI=0 ;;
      trigram) BI=0 TRI=1 ;;
      both) BI=1 TRI=1 ;;
      *) echo "[table-full] unknown arm: $ARM"; exit 2 ;;
    esac
    launch "tbl_${TM}_${ARM}${RUN_SUFFIX}" "$TM" "$BI" "$TRI" 1000
  done
done

# wait for all
while [ "$ACTIVE" -gt 0 ]; do
  wait -n
  ACTIVE=$((ACTIVE - 1))
done

echo "=== table full grid done (MONITOR=$MONITOR) at $(date) ==="
