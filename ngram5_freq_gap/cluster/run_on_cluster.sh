#!/usr/bin/env bash
# Launch an n-gram frequency-gap run on the ophis-gpu cluster.
#
# This script:
#   1. Syncs the isolated ngram5_freq_gap/ package to the cluster.
#   2. Generates the controlled n-gram dataset on the cluster (from the real climbmix
#      shard 1 cache) if not already present.
#   3. Launches the trainer with the baseline_input n-gram injection setting.
#
# Usage:
#   ./run_on_cluster.sh <alpha> <gpu_id> [max_steps] [extra env ...]
# Examples:
#   ./run_on_cluster.sh 0.0 0 2000              # baseline, no resampling
#   ./run_on_cluster.sh 0.5 1 2000              # mild low-freq up-sampling
#   ./run_on_cluster.sh 0.5 1 2000 SEED=43
#   NGRAM5_DATA_DIR_OVERRIDE=/path/to/audited/dataset \
#     ./run_on_cluster.sh 0.0 1 10              # reuse an audited pilot dataset

set -euo pipefail

ALPHA="${1:?alpha required}"; GPU="${2:?gpu_id required}"; MAXSTEPS="${3:-2000}"; shift 3 || true
EXTRA=("$@")

# ---- cluster config ----
SSH_HOST="${NGLAB_SSH_HOST:-ophis-gpu}"
CLUSTER_ROOT="${NGLAB_CLUSTER_ROOT:?set NGLAB_CLUSTER_ROOT to the repository path on the target host}"
CLUSTER_PY="$CLUSTER_ROOT/.venv/bin/python"
REMOTE_CACHE="${NGLAB_CLUSTER_CACHE:?set NGLAB_CLUSTER_CACHE to the tokenizer/parquet cache on the target host}"
LOCAL_CACHE="${NGLAB_CACHE_DIR:-}"
if [[ -n "$LOCAL_CACHE" ]]; then
  CACHE="$LOCAL_CACHE"
else
  CACHE="$REMOTE_CACHE"
fi
RUNTIME_TMP="$CLUSTER_ROOT/.tmp/ngram5"
DATA_BASE="$CLUSTER_ROOT/data/ngram5_controlled"
DATA_DIR="${NGRAM5_DATA_DIR_OVERRIDE:-$DATA_BASE/fivegram_alpha${ALPHA}}"
LOCAL_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # ngram5_freq_gap/
REPO_ROOT="$(cd "$LOCAL_HERE/.." && pwd)"

# Trunk variant: transformer (baseline_input) or mlp (mlp_input).
TRUNK="${NGRAM5_TRUNK:-transformer}"
case "$TRUNK" in
  mlp) RUN_LABEL="mlp_input" ;;
  *)   RUN_LABEL="baseline_input" ;;
esac

echo "=== 5-gram freq-gap run: alpha=${ALPHA} steps=${MAXSTEPS} gpu=${GPU} trunk=${TRUNK} ==="

# ---- 1. sync local files to cluster ----
echo "=== [1/3] syncing local files to cluster ==="
rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude 'data/' --exclude 'run_artifacts/' --exclude 'runs/' \
  "$LOCAL_HERE/" "$SSH_HOST:$CLUSTER_ROOT/ngram5_freq_gap/"
rsync -avz "$REPO_ROOT/code/train.py" "$SSH_HOST:$CLUSTER_ROOT/train.py"
# trainer.py loads ngram5_freq_gap/lib.py under a private module name.  Keep
# the repository-root lib.py untouched.

# ---- 2. generate dataset if missing ----
echo "=== [2/3] ensuring dataset exists at $DATA_DIR ==="
ssh "$SSH_HOST" "
  set -e
  if [[ ! -f '$DATA_DIR/meta.json' || ! -f '$DATA_DIR/metadata.json' || ! -f '$DATA_DIR/exact_ngram_counts.npz' ]] || \
     ! '$CLUSTER_PY' -c \"import json,sys; m=json.load(open('$DATA_DIR/meta.json')); sys.exit(0 if m.get('frequency_definition') == 'exact_train_epoch_context_count' and m.get('frequency_source_split') == 'train' and m.get('frequency_key_type') == 'exact_context' and not m.get('hash_bucket_occupancy_diagnostic', False) else 1)\"; then
    echo '[gen] generating 5-gram dataset alpha=$ALPHA ...'
    mkdir -p '$DATA_BASE'
    cd '$CLUSTER_ROOT'
    AUTORESEARCH_CACHE_DIR='$CACHE' FIXED_TOKENIZER_DIR='' \
      '$CLUSTER_PY' -u ngram5_freq_gap/data_gen.py \
        --out-dir '$DATA_DIR' \
        --alpha $ALPHA \
        --bucket-count 5000000 \
        --order 5 \
        --f-train 0.8 --f-val 0.2 \
        --k-min 0.25 --k-max 8.0 \
        --r-ref-mode median \
        --dataset-seed 20260805 \
        --doc-len 2048
  else
    echo '[gen] dataset already exists, skipping'
  fi
"

# ---- 3. launch trainer ----
echo "=== [3/3] launching trainer ==="
echo "=== trace: every train batch + 4 fixed validation batches per step; token_loss=float32; norm=sqrt(mean(x^2)); compile=0 ==="
RUN_ID="${NGRAM5_RUN_ID:-ngram5_alpha${ALPHA}_${RUN_LABEL}_$(date +%Y%m%d-%H%M%S)}"
RESULT_DIR="$CLUSTER_ROOT/data/runs_fixed/${RUN_ID}_fixed"
ssh "$SSH_HOST" "
  set -e
  mkdir -p '$RESULT_DIR'
  mkdir -p '$RUNTIME_TMP'
  cd '$CLUSTER_ROOT'
  # baseline_input n-gram injection setting (from run_injpos_input.sh)
  # + 5-gram block data mode
  CUDA_VISIBLE_DEVICES='$GPU' env \\
    TMPDIR='$RUNTIME_TMP' TMP='$RUNTIME_TMP' TEMP='$RUNTIME_TMP' \\
    TORCHINDUCTOR_CACHE_DIR='$CLUSTER_ROOT/torchinductor_guoshaoyang' \\
    AUTORESEARCH_CACHE_DIR='$CACHE' FIXED_TOKENIZER_DIR='' \\
    ARCH_VARIANT=nanogpt_original \\
    NGRAM5_TRUNK='$TRUNK' \\
    NANOGPT_NGRAM_OPTIMIZER=mixed NGRAM_TABLE_OPTIMIZER=rmsprop \\
    NANOGPT_MATRIX_OPTIMIZER=adamw NANOGPT_OPTIMIZER_GROUPING=nanogpt \\
    NANOGPT_ATTENTION_IMPL=fused NANOGPT_NGRAM_INJECTION_IMPL=nanogpt \\
    NANOGPT_NGRAM_INJECTION_POSITION=input \\
    NANOGPT_ENABLE_NGRAM_VE=1 ENABLE_UNIGRAM_VE=0 ENABLE_BIGRAM_VE=1 ENABLE_TRIGRAM_VE=1 ENABLE_FOURGRAM_VE=0 \\
    NANOGPT_ADAM_LR=0.0006 NGRAM_TABLE_BETAS=0.0,0.99 NGRAM_TABLE_LR_SCALE=2.0 \\
    BIGRAM_CLEAN_TABLE=1048576 TRIGRAM_CLEAN_TABLE=1048576 \\
    POSITION_ENCODING=learned_abs CURRENT_NORMALIZATION=layernorm \\
    CURRENT_EMBEDDING_TYING=tied CURRENT_NGRAM_INJECTION_IMPL=none \\
    CURRENT_EMBEDDING_INIT=nanogpt_like CURRENT_BLOCK_INIT=nanogpt_style \\
    CURRENT_ATTENTION_NORM=none CURRENT_HEAD_GATE=none \\
    CURRENT_RESIDUAL_PATH=plain CURRENT_LAYER_POOL=none \\
    CURRENT_MLP=gelu CURRENT_LOGIT_SOFTCAP=none CURRENT_LINEAR_BIAS=none \\
    WINDOW_PATTERN=LLLL SEED=42 \\
    TRAIN_DATA_MODE=ngram5_blocks TRAIN_DATA_SEED=42 \\
    NGRAM5_DATA_DIR='$DATA_DIR' \\
    MAX_TRAINING_STEPS=$MAXSTEPS DEVICE_BATCH_SIZE=72 TOTAL_BATCH_SIZE=147456 \\
    VAL_LOSS_INTERVAL_STEPS=10 VAL_LOSS_BATCHES=4 LR_SCHEDULE_MODE=warmup_constant WARMUP_STEPS=100 \\
    NGRAM5_PROBE_STEPS=100,200,400,600,800,1000,1500,2000 \\
    NGRAM5_PROBE_FREQUENCY_MODE=exact_context \\
    NGRAM5_BUCKET_EDGES=0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001 \\
    NGRAM5_TRACE_ALL_BATCHES=1 \\
    NGRAM5_TRACE_COMPRESSION=1 \\
    NGRAM_GLOBAL_FREQUENCY_MODE=baseline \\
    NGRAM_GLOBAL_FREQUENCY_DIR='$DATA_DIR' \\
    TORCH_COMPILE=0 \\
    REMOTE_RESULT_DIR='$RESULT_DIR' \\
    RUN_ID='$RUN_ID' \\
    ${EXTRA[@]:+} ${EXTRA[*]:-} \\
    '$CLUSTER_PY' -u ngram5_freq_gap/trainer.py > '$RESULT_DIR/train.log' 2>&1 || rc=\$?
  rc=\${rc:-0}
  printf '%s\n' \"\$rc\" > '$RESULT_DIR/exit_code.txt'
  if [[ \"\$rc\" -eq 0 ]]; then
    touch '$RESULT_DIR/done'
  fi
  echo \"=== run finished, exit \$rc ===\"
  exit \"\$rc\"
"
echo "=== done. results at $SSH_HOST:$RESULT_DIR ==="
echo "=== fetch with: rsync -avz $SSH_HOST:$RESULT_DIR/ ./data/runs_fixed/$(basename "$RESULT_DIR")/ ==="
