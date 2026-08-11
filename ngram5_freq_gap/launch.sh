#!/usr/bin/env bash
# Launch one 5-gram frequency-gap run.  Usage:
#   ./launch.sh <alpha> <gpu_id> [extra env ...]
# Example:
#   ./launch.sh 0.0 0                 # baseline, no resampling
#   ./launch.sh 0.5 1                # mild low-freq up-sampling
#   ./launch.sh 0.5 1 MAX_TRAINING_STEPS=2000
set -euo pipefail

ALPHA="${1:?alpha required}"; GPU="${2:?gpu_id required}"; shift 2 || true
EXTRA=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
DATA_BASE="$ROOT/data"

# Derive data dir from alpha (must have been generated already by data_gen.py).
DATA_DIR="$DATA_BASE/alpha${ALPHA}"
if [[ ! -f "$DATA_DIR/meta.json" ]]; then
  echo "ERROR: $DATA_DIR/meta.json missing. Run data_gen.py --alpha $ALPHA --out-dir $DATA_DIR first." >&2
  exit 1
fi

RESULT_DIR="$ROOT/runs/alpha${ALPHA}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULT_DIR"

COMMON=(
  NGRAM5_DATA_DIR="$DATA_DIR"
  REMOTE_RESULT_DIR="$RESULT_DIR"
  SEED=42
  MAX_TRAINING_STEPS=1000
  DEVICE_BATCH_SIZE=72
  MAX_SEQ_LEN=2048
  LEARNING_RATE=9e-4
  WARMUP_RATIO=0.05
  WARMDOWN_RATIO=0.5
  FINAL_LR_FRAC=0.0
  WEIGHT_DECAY=0.1
  BETA1=0.9
  BETA2=0.95
  VAL_LOSS_INTERVAL_STEPS=50
  VAL_LOSS_BATCHES=4
  NGRAM5_PROBE_STEPS=100,200,300,400,500,750,1000
  NGRAM5_BUCKET_EDGES=0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001
  TORCH_COMPILE=1
)

echo ""
echo "=========================================="
echo "=== alpha=${ALPHA} (GPU ${GPU}) ==="
echo "=== $(date) ==="
echo "=== data:  ${DATA_DIR}"
echo "=== out:   ${RESULT_DIR}"
echo "=========================================="
CUDA_VISIBLE_DEVICES="$GPU" env "${EXTRA[@]}" "${COMMON[@]}" \
  "$PY" -u "$ROOT/trainer.py" 2>&1 | tee "$RESULT_DIR/train.log"
RC=${PIPESTATUS[0]}
echo "=== alpha=${ALPHA} finished (exit ${RC}) at $(date) ==="
exit "$RC"
