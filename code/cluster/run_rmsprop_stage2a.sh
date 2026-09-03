#!/usr/bin/env bash
# RMSProp Stage 2A launcher for the H200 and A100 hosts.
# Usage: run_rmsprop_stage2a.sh PROFILE GPU RUN_ID BETA2 TABLE_LR_SCALE
# PROFILE is one of: h200, a100
set -euo pipefail

PROFILE="${1:?profile is required (h200 or a100)}"
GPU="${2:?GPU id is required}"
RUN_ID="${3:?run id is required}"
BETA2="${4:?table beta2 is required}"
TABLE_LR_SCALE="${5:?table LR scale is required}"

case "$PROFILE" in
  h200)
    ROOT=/data/home/yushanbin/ngram-gap-stage2a
    PY="$ROOT/.venv/bin/python"
    DEVICE_BATCH_SIZE=72
    VAL_BATCHES=4
    FIXED_PROBE_BATCHES=4
    ONLINE_FREQUENCY_VAL_BATCHES=1
    ;;
  a100)
    ROOT=/data0/yushanbin/ngram-gap-shaoyang-2
    PY=/data0/yushanbin/conda-envs/ngramgap/bin/python
    DEVICE_BATCH_SIZE=4
    VAL_BATCHES=72
    FIXED_PROBE_BATCHES=72
    ONLINE_FREQUENCY_VAL_BATCHES=18
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    exit 2
    ;;
esac

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

CUDA_VISIBLE_DEVICES="$GPU" "$PY" - "$OUT/runtime_environment.json" "$PROFILE" "$GPU" "$DEVICE_BATCH_SIZE" <<'PY'
import json
import os
import platform
import sys

import torch

out, profile, physical_gpu, device_batch_size = sys.argv[1:]
payload = {
    "hostname": platform.node(),
    "profile": profile,
    "physical_gpu": int(physical_gpu),
    "python": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "visible_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "device_batch_size": int(device_batch_size),
}
with open(out, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "$RUN_ID" \
  --injection_position input \
  --steps 1000 \
  --seed 42 \
  --data_dir "$ROOT/data/tokenized" \
  --out_dir "$ROOT/data/runs" \
  --train_shards 1 \
  --val_shards 2,3,4,5,6,7,8,9,10,6542 \
  --device_batch_size "$DEVICE_BATCH_SIZE" \
  --total_batch_size 147456 \
  --val_interval 50 \
  --val_batches "$VAL_BATCHES" \
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
  --online_frequency_val_batches "$ONLINE_FREQUENCY_VAL_BATCHES" \
  --fixed_probe_batches "$FIXED_PROBE_BATCHES" \
  --fixed_probe_train_offset_steps 168 \
  > "$OUT/train.log" 2>&1

echo "completed $RUN_ID on $PROFILE GPU $GPU"
