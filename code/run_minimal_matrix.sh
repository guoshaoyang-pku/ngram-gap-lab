#!/usr/bin/env bash
# Minimal 5-gram experiment matrix on ophis-gpu: shuffle on/off x injection
# {none, +trigram, +bigram} = 6 arms.  Runs trainer.py locally on the cluster.
#
# Usage (on ophis-gpu):
#   ./run_minimal_matrix.sh <steps> <gpu_list>
#   ./run_minimal_matrix.sh 1000 2            # 1000 steps, GPU 2
#   ./run_minimal_matrix.sh 1000 "2 4 7"      # 3 arms in parallel on 3 GPUs
#
# Env overrides:
#   NGLAB_DATA_DIR   path to make_ngram_blocks output
#   NGLAB_RUNS_DIR   where run dirs land
set -euo pipefail

STEPS="${1:-1000}"
GPU_LIST="${2:-2}"

CLUSTER_ROOT="/data3/guoshaoyang/ngram-gap-exp"
PY="${PYTHON:-$CLUSTER_ROOT/.venv/bin/python}"
CACHE="/data2/ncpl-pathA/work/vbird_autoresearch/cache"
RUNTIME_TMP="$CLUSTER_ROOT/.tmp/ngram5"
DATA_DIR="${NGLAB_DATA_DIR:-$CLUSTER_ROOT/ngram5_data/minimal_order5_full}"
RUNS_DIR="${NGLAB_RUNS_DIR:-$CLUSTER_ROOT/runs/ngram5_minimal}"
TRAINER="$CLUSTER_ROOT/ngram5_freq_gap/trainer.py"

if [[ ! -f "$DATA_DIR/meta.json" ]]; then
  echo "ERROR: $DATA_DIR/meta.json missing. Run make_ngram_blocks.py first." >&2
  exit 1
fi

# arms: <name> <NANOGPT_ENABLE_NGRAM_VE> <ENABLE_BIGRAM_VE> <ENABLE_TRIGRAM_VE> <shuffle>
ARMS=(
  "pure_transformer_fixed      0 0 0 0"
  "pure_transformer_shuffled   0 0 0 1"
  "plus_trigram_fixed          1 0 1 0"
  "plus_trigram_shuffled       1 0 1 1"
  "plus_bigram_fixed           1 1 0 0"
  "plus_bigram_shuffled        1 1 0 1"
)

TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUNS_DIR"

i=0
for arm in "${ARMS[@]}"; do
  read -r NAME NGRAM_EN BIGRAM TRI SHUF <<<"$arm"
  GPU="$(echo "$GPU_LIST" | awk -v n="$i" '{print $( (n % NF) + 1 )}')"
  RESULT_DIR="$RUNS_DIR/${NAME}_${TS}"
  mkdir -p "$RESULT_DIR"

  echo ""
  echo "=========================================="
  echo "=== arm: ${NAME} (GPU ${GPU}, ${STEPS} steps) ==="
  echo "=== data: ${DATA_DIR}  out: ${RESULT_DIR} ==="
  echo "=== ngram_ve=${NGRAM_EN} bigram=${BIGRAM} trigram=${TRI} shuffle=${SHUF} ==="
  echo "=========================================="

  CUDA_VISIBLE_DEVICES="$GPU" env \
    TMPDIR="$RUNTIME_TMP" TMP="$RUNTIME_TMP" TEMP="$RUNTIME_TMP" \
    TORCHINDUCTOR_CACHE_DIR="$CLUSTER_ROOT/torchinductor_guoshaoyang" \
    AUTORESEARCH_CACHE_DIR="$CACHE" FIXED_TOKENIZER_DIR="" \
    ARCH_VARIANT=nanogpt_original \
    NGRAM5_TRUNK=transformer \
    NANOGPT_NGRAM_OPTIMIZER=mixed NGRAM_TABLE_OPTIMIZER=rmsprop \
    NANOGPT_MATRIX_OPTIMIZER=adamw NANOGPT_OPTIMIZER_GROUPING=nanogpt \
    NANOGPT_ATTENTION_IMPL=fused NANOGPT_NGRAM_INJECTION_IMPL=nanogpt \
    NANOGPT_NGRAM_INJECTION_POSITION=input \
    NANOGPT_ENABLE_NGRAM_VE="$NGRAM_EN" \
    ENABLE_UNIGRAM_VE=0 ENABLE_BIGRAM_VE="$BIGRAM" ENABLE_TRIGRAM_VE="$TRI" ENABLE_FOURGRAM_VE=0 \
    NANOGPT_ADAM_LR=0.004 NGRAM_TABLE_BETAS=0.0,0.999 NGRAM_TABLE_LR_SCALE=1.0 \
    POSITION_ENCODING=learned_abs CURRENT_NORMALIZATION=layernorm \
    CURRENT_EMBEDDING_TYING=tied CURRENT_NGRAM_INJECTION_IMPL=none \
    CURRENT_EMBEDDING_INIT=nanogpt_like CURRENT_BLOCK_INIT=nanogpt_style \
    CURRENT_ATTENTION_NORM=none CURRENT_HEAD_GATE=none \
    CURRENT_RESIDUAL_PATH=plain CURRENT_LAYER_POOL=none \
    CURRENT_MLP=gelu CURRENT_LOGIT_SOFTCAP=none CURRENT_LINEAR_BIAS=none \
    WINDOW_PATTERN=LLLL SEED=42 \
    TRAIN_DATA_MODE=ngram5_blocks TRAIN_DATA_SEED=42 \
    NGRAM5_BLOCK_SHUFFLE="$SHUF" \
    NGRAM5_DATA_DIR="$DATA_DIR" \
    MAX_TRAINING_STEPS="$STEPS" DEVICE_BATCH_SIZE=72 TOTAL_BATCH_SIZE=147456 \
    VAL_LOSS_INTERVAL_STEPS=10 VAL_LOSS_BATCHES=4 LR_SCHEDULE_MODE=baseline \
    NGRAM5_PROBE_STEPS=100,200,400,600,800,1000 \
    NGRAM5_PROBE_FREQUENCY_MODE=exact_context \
    NGRAM5_BUCKET_EDGES=0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001 \
    NGRAM5_TRACE_ALL_BATCHES=1 \
    NGRAM5_TRACE_COMPRESSION=1 \
    NGRAM_GLOBAL_FREQUENCY_MODE=baseline \
    NGRAM_GLOBAL_FREQUENCY_DIR="$DATA_DIR" \
    TORCH_COMPILE=0 \
    REMOTE_RESULT_DIR="$RESULT_DIR" \
    RUN_ID="ngram5_minimal_${NAME}" \
    "$PY" -u "$TRAINER" > "$RESULT_DIR/train.log" 2>&1 &
  echo "[launched] $NAME pid=$!"
  i=$((i + 1))
done

echo ""
echo "=== launched ${i} arms; waiting... ==="
FAIL=0
for pid in $(jobs -p); do
  if wait "$pid"; then
    echo "[OK] pid $pid"
  else
    echo "[FAIL] pid $pid"
    FAIL=1
  fi
done
echo "=== matrix finished (fail=$FAIL) ==="
exit "$FAIL"
