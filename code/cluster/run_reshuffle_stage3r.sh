#!/usr/bin/env bash
# Stage 3R paired logical-batch order experiment on H200.
# Usage: run_reshuffle_stage3r.sh GPU MODE RUN_ID
# MODE is one of: frozen_permutation, epoch_reshuffle
set -euo pipefail

GPU="${1:?GPU id is required}"
MODE="${2:?train-order mode is required}"
RUN_ID="${3:?run id is required}"

case "$MODE" in
  frozen_permutation|epoch_reshuffle) ;;
  *) echo "unknown train-order mode: $MODE" >&2; exit 2 ;;
esac

ROOT="${NGLAB_ROOT:-/data/home/yushanbin/ngram-gap-shaoyang-2}"
PY="${NGLAB_PYTHON:-/usr/bin/python3}"
OUT="$ROOT/data/runs/$RUN_ID"
MANIFEST_SOURCE="$ROOT/data/shared/fixed_gram_probe_manifest.json"

[[ -x "$PY" ]] || { echo "missing Python: $PY" >&2; exit 1; }
[[ -f "$ROOT/code/train.py" ]] || { echo "missing train.py under $ROOT/code" >&2; exit 1; }
[[ -f "$ROOT/data/freq_index.npz" ]] || { echo "missing frequency index" >&2; exit 1; }
[[ -f "$MANIFEST_SOURCE" ]] || { echo "missing shared fixed-gram manifest" >&2; exit 1; }
if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing run directory: $OUT" >&2
  exit 1
fi

mkdir -p "$OUT"
cp "$MANIFEST_SOURCE" "$OUT/fixed_gram_probe_manifest.json"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" - "$OUT/runtime_environment.json" "$GPU" <<'PY'
import json
import platform
import sys

import torch

out, physical_gpu = sys.argv[1:]
payload = {
    "hostname": platform.node(),
    "profile": "h200",
    "physical_gpu": int(physical_gpu),
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "visible_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "device_batch_size": 72,
}
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "$RUN_ID" \
  --injection_position input \
  --steps 1000 \
  --seed 42 \
  --data_seed 101 \
  --train_order "$MODE" \
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
  --freq_index "$ROOT/data/freq_index.npz" \
  --fixed_gram_samples_per_bucket 100 \
  --fixed_gram_seed 42 \
  --online_frequency_interval 50 \
  --online_frequency_epoch_window 20 \
  --online_frequency_dense_interval 1 \
  --online_frequency_probe_window 0 \
  --online_frequency_probe_dense_interval 1 \
  --online_frequency_val_batches 1 \
  --fixed_probe_batches 0 \
  --fixed_gram_epoch_relative_steps=-10,-5,-1,0,1,5,10 \
  --fixed_gram_frequency_interval 0 \
  > "$OUT/train.log" 2>&1

echo "completed $RUN_ID on GPU $GPU"
