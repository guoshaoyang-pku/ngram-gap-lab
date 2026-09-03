#!/usr/bin/env bash
# Run one cumulative exact-frequency masking condition.
# Usage: run_frequency_mask_sweep.sh GPU THRESHOLD [RUN_ID]
# THRESHOLD is one of: none, all, or a non-negative integer.
set -euo pipefail

GPU="${1:?GPU id is required}"
THRESHOLD="${2:?mask threshold is required}"
ROOT="${NGLAB_ROOT:-/data/home/yushanbin/ngram-gap-shaoyang-2}"
if [[ -n "${NGLAB_PYTHON:-}" ]]; then
  PY="$NGLAB_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY=/usr/bin/python3
fi
RUN_ID="${3:-nglab_freqmask_${THRESHOLD}_s42}"
OUT="$ROOT/data/runs/$RUN_ID"
INDEX="$ROOT/data/freq_index.npz"

[[ -f "$ROOT/code/train.py" ]] || { echo "missing train.py under $ROOT" >&2; exit 1; }
[[ -f "$INDEX" ]] || { echo "missing frequency index: $INDEX" >&2; exit 1; }
[[ -d "$ROOT/data/tokenized" ]] || { echo "missing tokenized data" >&2; exit 1; }
if [[ -e "$OUT/summary.json" ]]; then
  echo "refusing to overwrite completed run: $OUT" >&2
  exit 1
fi
if [[ -d "$OUT" ]] && find "$OUT" -mindepth 1 -print -quit | grep -q .; then
  echo "refusing to overwrite non-empty run directory: $OUT" >&2
  exit 1
fi
mkdir -p "$OUT"

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'threshold=%s\n' "$THRESHOLD"
  printf 'host=%s\n' "$(hostname)"
  printf 'gpu=%s\n' "$GPU"
  printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "$OUT/job_meta.txt"

/usr/bin/time -f 'wall_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kb=%M' \
  -o "$OUT/runtime.txt" \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
    --run_id "$RUN_ID" \
    --injection_position input \
    --steps 1011 \
    --seed 42 \
    --data_seed 42 \
    --order_seed 42 \
    --train_order sequential \
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
    --table_beta2 0.999 \
    --table_lr_scale 1.0 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --ngram_mask_index "$INDEX" \
    --ngram_mask_threshold "$THRESHOLD" \
    --online_gap_interval 50 \
    --online_gap_epoch_offsets=-1,0,1 \
    --online_gap_val_batches 1 \
    --fixed_probe_batches 0 \
    > "$OUT/train.log" 2>&1

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "$OUT/job_meta.txt"
echo "completed $RUN_ID on GPU $GPU"
