#!/usr/bin/env bash
# Full epoch-length scaling grid (plan §3) on 360-2.  Run AFTER pilot QC.
#
# All runs: vanilla nanoGPT 8L·6H·768D + input n-gram injection,
# natural corpus nested-prefix shard 1, train/val zero overlap,
# table RMSProp(0.0, 0.99) / backbone AdamW(0.8, 0.95), fixed train probe +
# exact-frequency diagnostics.
#
# Fixed-step (fs): 1000 steps, step-anchored LR, all L1-L4 x 3 modules.
# Fixed-epoch (fe): 6 full epochs, epoch-anchored LR (--lr_schedule_epochs 6),
#   target steps L1=252 L2=504 L3=1008 L4=2016.
#
# Usage: ./run_scaling_epoch_full.sh [gpu1] [gpu2] [gpu3]
set -euo pipefail

ROOT=/data/home/guoshaoyang/ngram-gap-lab
PY=/usr/bin/python3
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
FREQ_IDX="$ROOT/data/freq_index.npz"
G1="${1:-0}" G2="${2:-1}" G3="${3:-2}"
mkdir -p "$OUT_DIR"

declare -A EPB=( [L1]=42 [L2]=84 [L3]=168 [L4]=336 )
declare -A FESTEPS=( [L1]=252 [L2]=504 [L3]=1008 [L4]=2016 )

run_arm() {  # run_arm <gpu> <run_id> <epoch_batches> <steps> <schedule_epochs> <bigram> <trigram>
  local GPU="$1" RUN_ID="$2" EPB="$3" STEPS="$4" SCHED="$5" BI="$6" TRI="$7"
  local RESULT_DIR="$OUT_DIR/$RUN_ID"
  mkdir -p "$RESULT_DIR"
  echo "[epoch-full] $RUN_ID epb=$EPB steps=$STEPS sched=$SCHED bi=$BI tri=$TRI -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "$RUN_ID" --injection_position input \
    --steps "$STEPS" --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 50 --val_batches 4 --table_norm_interval 50 --lr 0.004 \
    --enable_unigram 0 --enable_bigram "$BI" --enable_trigram "$TRI" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --freq_eval_interval 50 --freq_eval_batches 4 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --freq_index "$FREQ_IDX" \
    --epoch_batches "$EPB" \
    --fixed_train_probe 4 --probe_eval_interval 50 \
    --table_betas 0.0,0.99 \
    ${SCHED:+--lr_schedule_epochs "$SCHED"} \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[epoch-full] $RUN_ID done (exit=$?) at $(date)"
}

# Fixed-step: L1-L4 x {bigram, trigram, both}  (no-ngram baseline from pilot)
for L in L1 L2 L3 L4; do
  run_arm "$G1" "ep_${L}_bigram_fs"  "${EPB[$L]}" 1000 0 1 0 &
  run_arm "$G2" "ep_${L}_trigram_fs" "${EPB[$L]}" 1000 0 0 1 &
  run_arm "$G3" "ep_${L}_both_fs"    "${EPB[$L]}" 1000 0 1 1 &
  wait
done

# Fixed-epoch: L1-L4 x {bigram, trigram, both}
for L in L1 L2 L3 L4; do
  run_arm "$G1" "ep_${L}_bigram_fe"  "${EPB[$L]}" "${FESTEPS[$L]}" 6 1 0 &
  run_arm "$G2" "ep_${L}_trigram_fe" "${EPB[$L]}" "${FESTEPS[$L]}" 6 0 1 &
  run_arm "$G3" "ep_${L}_both_fe"    "${EPB[$L]}" "${FESTEPS[$L]}" 6 1 1 &
  wait
done

echo "=== epoch full grid done at $(date) ==="
