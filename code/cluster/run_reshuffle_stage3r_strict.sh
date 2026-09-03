#!/usr/bin/env bash
# Strict Stage 3R checkpoint-fork experiment on H200.
# Usage:
#   run_reshuffle_stage3r_strict.sh GPU prefix RUN_ID
#   run_reshuffle_stage3r_strict.sh GPU sequential RUN_ID CHECKPOINT
#   run_reshuffle_stage3r_strict.sh GPU reshuffle RUN_ID CHECKPOINT
set -euo pipefail

GPU=${1:?GPU index required}
PHASE=${2:?phase required: prefix|sequential|reshuffle}
RUN_ID=${3:?run id required}
CHECKPOINT=${4:-}
ROOT=${NGLAB_ROOT:-/data/home/yushanbin/ngram-gap-stage2a}
PY=${NGLAB_PYTHON:-$ROOT/.venv/bin/python}
OUT="$ROOT/data/runs/$RUN_ID"
SHARED_MANIFEST="$ROOT/data/shared/fixed_gram_probe_manifest.json"

case "$PHASE" in
  prefix)
    MODE=sequential
    EXTRA_ARGS=(--stop_after_step 337 --save_checkpoint_step 337)
    ;;
  sequential)
    MODE=sequential
    [[ -n "$CHECKPOINT" && -f "$CHECKPOINT" ]] || {
      echo "missing checkpoint: $CHECKPOINT" >&2
      exit 2
    }
    EXTRA_ARGS=(--resume_checkpoint "$CHECKPOINT")
    ;;
  reshuffle)
    MODE=sequential_then_reshuffle
    [[ -n "$CHECKPOINT" && -f "$CHECKPOINT" ]] || {
      echo "missing checkpoint: $CHECKPOINT" >&2
      exit 2
    }
    EXTRA_ARGS=(--resume_checkpoint "$CHECKPOINT")
    ;;
  *)
    echo "unknown phase: $PHASE" >&2
    exit 2
    ;;
esac

[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 3; }
[[ -x "$PY" ]] || { echo "missing Python: $PY" >&2; exit 4; }
[[ -f "$SHARED_MANIFEST" ]] || { echo "missing shared manifest: $SHARED_MANIFEST" >&2; exit 5; }

mkdir -p "$OUT"
cp "$SHARED_MANIFEST" "$OUT/fixed_gram_probe_manifest.json"

"$PY" - "$OUT/runtime_environment.json" "$GPU" "$PHASE" "$MODE" "$CHECKPOINT" <<'PY'
import json, os, platform, socket, subprocess, sys, torch
out, gpu, phase, mode, checkpoint = sys.argv[1:]
payload = {
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": gpu,
    "phase": phase,
    "train_order": mode,
    "checkpoint": os.path.abspath(checkpoint) if checkpoint else None,
}
try:
    payload["nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
except Exception as exc:
    payload["nvidia_smi_error"] = repr(exc)
with open(out, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
PY

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/code/train.py" \
  --run_id "$RUN_ID" \
  --injection_position input \
  --steps 1000 \
  --seed 42 \
  --data_seed 42 \
  --order_seed 101 \
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
  "${EXTRA_ARGS[@]}" \
  > "$OUT/train.log" 2>&1

echo "completed $RUN_ID phase=$PHASE mode=$MODE on GPU $GPU"
