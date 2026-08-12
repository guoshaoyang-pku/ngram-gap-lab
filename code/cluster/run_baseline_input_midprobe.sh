#!/usr/bin/env bash
# Baseline input-injection run with the fixed train probe centered in a replay
# epoch.  Dense checkpoints are deliberately sparse: five points (center +/-
# 10, spacing 5) around each epoch boundary and each probe position.
# Usage: ./run_baseline_input_midprobe.sh [gpu_id] [steps]
set -euo pipefail

GPU="${1:-0}"
STEPS="${2:-1000}"

ROOT=/data/home/yushanbin/ngram-gap-shaoyang-2
PY=/usr/bin/python3
RUN_ID="${RUN_ID:-nglab_baseline_input_midprobe_sparse_20260812}"

mkdir -p "$ROOT/data/runs/$RUN_ID"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "$RUN_ID" \
  --injection_position input \
  --steps "$STEPS" \
  --seed 42 \
  --data_dir "$ROOT/data/tokenized" \
  --out_dir "$ROOT/data/runs" \
  --train_shards 1 \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --device_batch_size 72 \
  --total_batch_size 147456 \
  --val_interval 50 \
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
  --freq_index "$ROOT/data/freq_index.npz" \
  --fixed_gram_samples_per_bucket 100 \
  --fixed_gram_seed 42 \
  --online_frequency_interval 50 \
  --online_frequency_epoch_window 10 \
  --online_frequency_dense_interval 5 \
  --online_frequency_probe_window 10 \
  --online_frequency_probe_dense_interval 5 \
  --online_frequency_val_batches 1 \
  --fixed_probe_batches 4 \
  --fixed_probe_train_offset_steps 168 \
  > "$ROOT/data/runs/$RUN_ID/train.log" 2>&1

"$PY" "$ROOT/docs/generate_frequency_gap_report.py" \
  --run-dir "$ROOT/data/runs/$RUN_ID" \
  --out "$ROOT/docs/frequency-gap-by-hit-count-$RUN_ID.html"
