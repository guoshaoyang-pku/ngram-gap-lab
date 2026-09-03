#!/usr/bin/env bash
# Stage 1 RMSProp ablation. Only table beta2 and table LR scale vary.
# Usage: ./run_rmsprop_stage1.sh GPU RUN_ID BETA2 TABLE_LR_SCALE [STEPS]
set -euo pipefail

GPU="${1:?GPU id is required}"
RUN_ID="${2:?run id is required}"
BETA2="${3:?table beta2 is required}"
TABLE_LR_SCALE="${4:?table LR scale is required}"
STEPS="${5:-1000}"

ROOT=/data/home/yushanbin/ngram-gap-shaoyang-2
PY=/usr/bin/python3
OUT="$ROOT/data/runs/$RUN_ID"
# Every Stage 1 condition uses the same deterministic fixed-gram occurrence
# sample. Reuse the already validated baseline manifest rather than having
# concurrent runs rescan the full shard set independently.
MANIFEST_SOURCE="$ROOT/data/runs/nglab_baseline_input_midprobe_sparse_20260812/fixed_gram_probe_manifest.json"

mkdir -p "$OUT"
if [[ ! -f "$OUT/fixed_gram_probe_manifest.json" ]]; then
  [[ -f "$MANIFEST_SOURCE" ]] || {
    echo "missing shared fixed-gram manifest: $MANIFEST_SOURCE" >&2
    exit 1
  }
  cp "$MANIFEST_SOURCE" "$OUT/fixed_gram_probe_manifest.json"
fi
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
  --table_beta2 "$BETA2" \
  --table_lr_scale "$TABLE_LR_SCALE" \
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
  > "$OUT/train.log" 2>&1

echo "Run data complete. Generate the consolidated report locally with docs/generate_report.py after syncing data/runs/."
