#!/usr/bin/env bash
# v2 rerun of the 2.5x / 3x / 4x arms with CORRECTED val shards.
# v1 bug: val = 3..10,6542 overlapped train (2.5x/3x/4x include shard 3),
# so the fixed val batches were partially memorized train data -> gap biased
# negative. v2 uses val strictly AFTER the last train shard.
#
# Usage: ./run_shard_sweep_v2.sh [gpu25] [gpu3] [gpu4]
set -euo pipefail

ROOT=/data3/guoshaoyang/ngram-gap-lab
PY="$ROOT/.venv/bin/python"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs"
G25="${1:-3}" G3="${2:-4}" G4="${3:-5}"

run_one() {  # run_one <gpu> <run_id> <train_shards> <val_shards> <steps> <freq_index>
  local GPU="$1" RUN_ID="$2" SHARDS="$3" VAL="$4" STEPS="$5" FREQ="$6"
  local RESULT_DIR="$OUT_DIR/$RUN_ID"
  mkdir -p "$RESULT_DIR"
  echo "[sweep2] $RUN_ID (shards=$SHARDS val=$VAL steps=$STEPS) -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "$RUN_ID" --injection_position input --steps "$STEPS" --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 10 --val_batches 4 --table_norm_interval 10 --lr 0.004 \
    --enable_unigram 0 --enable_bigram 1 --enable_trigram 1 \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --freq_eval_interval 10 --freq_eval_batches 4 \
    --train_shards "$SHARDS" --val_shards "$VAL" --freq_index "$FREQ" \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[sweep2] $RUN_ID done (exit=$?) at $(date)"
}

run_one "$G25" nglab2_5x_input_fv_v2 "1,2,64"     "4,5,6,7,8,9,10,6542" 3200 "$ROOT/data/freq_index_train2_5x.npz" &
run_one "$G3"  nglab3x_input_fv_v2   "1,2,3"      "4,5,6,7,8,9,10,6542" 3800 "$ROOT/data/freq_index_train3x.npz"   &
run_one "$G4"  nglab4x_input_fv_v2   "1,2,3,4"    "5,6,7,8,9,10,6542"   5000 "$ROOT/data/freq_index_train4x.npz"   &
wait
echo "=== shard sweep v2 done at $(date) ==="
