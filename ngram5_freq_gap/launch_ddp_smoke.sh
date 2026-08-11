#!/usr/bin/env bash
# 4-GPU DDP smoke on the small sample285_v3 dataset (old format, .txt loader):
#   run1: MAX_TRAINING_STEPS=100, ckpt at 50 (model-only) and 100 (full)
#   run2: resume from step_00100.pt up to step 120 (verifies resume path)
# Uses GPUs 2,4,5,6.  Safe to run while data_gen tokenizes (CPU-bound, 1 core).
set -euo pipefail

SSH_HOST="${SSH_HOST:-ophis-gpu}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/data3/guoshaoyang/ngram-gap-exp}"
CLUSTER_PY="${CLUSTER_PY:-$CLUSTER_ROOT/.venv/bin/python}"
CACHE="${AUTORESEARCH_CACHE_DIR:-/data2/ncpl-pathA/work/vbird_autoresearch/cache}"
DATA_DIR_OLD="${DATA_DIR_OLD:-$CLUSTER_ROOT/ngram5_data/trigram_exact_alpha0.0_sample285_v3}"
GPU_IDS="${GPU_IDS:-2,4,5,6}"
RESULT_DIR="${RESULT_DIR:-$CLUSTER_ROOT/runs/ngram5_big/ddp_smoke_$(date +%Y%m%d-%H%M%S)}"
MASTER_PORT="${MASTER_PORT:-29501}"

COMMON_ENV=(
  AUTORESEARCH_CACHE_DIR="$CACHE" FIXED_TOKENIZER_DIR=''
  ARCH_VARIANT=nanogpt_original NGRAM5_TRUNK=transformer
  NANOGPT_NGRAM_OPTIMIZER=mixed NGRAM_TABLE_OPTIMIZER=rmsprop
  NANOGPT_MATRIX_OPTIMIZER=adamw NANOGPT_OPTIMIZER_GROUPING=nanogpt
  NANOGPT_ATTENTION_IMPL=fused NANOGPT_NGRAM_INJECTION_IMPL=nanogpt
  NANOGPT_NGRAM_INJECTION_POSITION=input
  NANOGPT_ENABLE_NGRAM_VE=1 ENABLE_UNIGRAM_VE=0 ENABLE_BIGRAM_VE=1
  ENABLE_TRIGRAM_VE=1 ENABLE_FOURGRAM_VE=0
  NANOGPT_ADAM_LR=0.004 NGRAM_TABLE_BETAS=0.0,0.999 NGRAM_TABLE_LR_SCALE=1.0
  POSITION_ENCODING=learned_abs CURRENT_NORMALIZATION=layernorm
  CURRENT_EMBEDDING_TYING=tied CURRENT_NGRAM_INJECTION_IMPL=none
  CURRENT_EMBEDDING_INIT=nanogpt_like CURRENT_BLOCK_INIT=nanogpt_style
  CURRENT_ATTENTION_NORM=none CURRENT_HEAD_GATE=none
  CURRENT_RESIDUAL_PATH=plain CURRENT_LAYER_POOL=none
  CURRENT_MLP=gelu CURRENT_LOGIT_SOFTCAP=none CURRENT_LINEAR_BIAS=none
  WINDOW_PATTERN=LLLL SEED=42 TRAIN_DATA_SEED=42
  TRAIN_DATA_MODE=ngram5_blocks NGRAM5_DATA_DIR="$DATA_DIR_OLD"
  DEVICE_BATCH_SIZE=72 TOTAL_BATCH_SIZE=589824
  VAL_LOSS_INTERVAL_STEPS=10 VAL_LOSS_BATCHES=4 LR_SCHEDULE_MODE=baseline
  NGRAM5_PROBE_STEPS=50,100
  NGRAM5_PROBE_FREQUENCY_MODE=exact_context
  NGRAM5_BUCKET_EDGES='0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001'
  NGRAM5_TRACE_ALL_BATCHES=0 NGRAM5_TRACE_COMPRESSION=1
  NGRAM_GLOBAL_FREQUENCY_MODE=baseline
  NGRAM_GLOBAL_FREQUENCY_DIR="$DATA_DIR_OLD"
  TORCH_COMPILE=0
  NGRAM5_CKPT_DIR="$RESULT_DIR/checkpoints"
  NGRAM5_CKPT_INTERVAL_STEPS=50 NGRAM5_CKPT_STEPS=100
  REMOTE_RESULT_DIR="$RESULT_DIR" RUN_ID=ngram5_ddp_smoke
  MASTER_PORT="$MASTER_PORT"
)

launch() {  # $1 = max_steps, $2 = init_ckpt ("" = none), $3 = log label
  local max_steps="$1" init_ckpt="$2" label="$3"
  ssh "$SSH_HOST" "
    set -e
    mkdir -p '$RESULT_DIR'
    cd '$CLUSTER_ROOT'
    ENV_VARS=(
      CUDA_VISIBLE_DEVICES=$GPU_IDS
      ${COMMON_ENV[*]}
      MAX_TRAINING_STEPS=$max_steps
    )
    if [[ -n '$init_ckpt' ]]; then
      ENV_VARS+=(NGRAM5_INIT_CKPT='$init_ckpt')
    fi
    nohup env \"\${ENV_VARS[@]}\" '$CLUSTER_PY' -m torch.distributed.run \
      --nproc-per-node=4 --nnodes=1 --standalone --master_port=$MASTER_PORT \
      ngram5_freq_gap/trainer.py > '$RESULT_DIR/$label.log' 2>&1 &
    echo \"\$!\"
  "
}

wait_done() {  # $1 = pid, $2 = label
  local pid="$1" label="$2" t=0
  while ssh "$SSH_HOST" "kill -0 $pid 2>/dev/null"; do
    sleep 20
    t=$((t + 20))
    if (( t % 120 == 0 )); then
      echo "  [smoke] $label still running (${t}s) ..."
      ssh "$SSH_HOST" "tail -1 '$RESULT_DIR/$label.log'" || true
    fi
    if (( t > 3600 )); then
      echo "  [smoke] TIMEOUT waiting for $label" >&2
      return 1
    fi
  done
}

echo "=== smoke run1: 100 steps, 4-GPU DDP on $DATA_DIR_OLD ==="
ssh "$SSH_HOST" "mkdir -p '$RESULT_DIR'"
PID1="$(launch 100 "" run1)"
PID1="${PID1##*$'\n'}"
echo "  pid=$PID1 log=$RESULT_DIR/run1.log"
wait_done "$PID1" run1
echo "=== smoke run1 done; tail + checkpoints ==="
ssh "$SSH_HOST" "tail -6 '$RESULT_DIR/run1.log'; echo '---'; ls -la '$RESULT_DIR/checkpoints/'"

CKPT100="$RESULT_DIR/checkpoints/step_00100.pt"
echo "=== smoke run2: resume from step_00100.pt to step 120 ==="
PID2="$(launch 120 "$CKPT100" resume_00100)"
PID2="${PID2##*$'\n'}"
echo "  pid=$PID2 log=$RESULT_DIR/resume_00100.log"
wait_done "$PID2" resume_00100
echo "=== smoke run2 done; tail + checkpoints ==="
ssh "$SSH_HOST" "tail -6 '$RESULT_DIR/resume_00100.log'; echo '---'; ls -la '$RESULT_DIR/checkpoints/'"

echo "=== DDP smoke complete: $RESULT_DIR ==="
