#!/usr/bin/env bash
# Stage 2: launch the 4-GPU DDP continuous pretraining run on the full-163
# n-gram block dataset.  Run AFTER run_big_continuous.sh has produced
# $GEN_OUT/meta.json.  Supports resume via INIT_CKPT.
set -euo pipefail

SSH_HOST="${SSH_HOST:-ophis-gpu}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/data3/guoshaoyang/ngram-gap-exp}"
CLUSTER_PY="${CLUSTER_PY:-$CLUSTER_ROOT/.venv/bin/python}"
CACHE="${AUTORESEARCH_CACHE_DIR:-/data2/ncpl-pathA/work/vbird_autoresearch/cache}"
GEN_OUT="${GEN_OUT:-ngram5_data/trigram_exact_alpha0.0_full163_20260808}"
GPU_IDS="${GPU_IDS:-2,4,5,6}"
MAXSTEPS="${MAXSTEPS:-70000}"
INIT_CKPT="${INIT_CKPT:-}"
RESULT_DIR="${RESULT_DIR:-$CLUSTER_ROOT/runs/ngram5_big/continuous_v1_$(date +%Y%m%d-%H%M%S)}"

PROBE_STEPS="1000,2000,4000,6000,8000,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,70000"
CKPT_FULL="10000,20000,30000,40000,50000,60000,70000"

echo "=== launching DDP continuous run on GPUs [$GPU_IDS] ==="
echo "result dir: $RESULT_DIR"

ssh "$SSH_HOST" "
  set -e
  mkdir -p '$RESULT_DIR'
  cd '$CLUSTER_ROOT'
  if [[ ! -f '$CLUSTER_ROOT/$GEN_OUT/meta.json' ]]; then
    echo '[error] dataset not ready: $GEN_OUT/meta.json missing' >&2
    exit 1
  fi
  CMD="'$CLUSTER_PY' -m torch.distributed.run \\
    --nproc-per-node=4 --nnodes=1 --standalone --master_port=29500 \\
    ngram5_freq_gap/trainer.py"
  ENV_VARS=(
    CUDA_VISIBLE_DEVICES=$GPU_IDS
    AUTORESEARCH_CACHE_DIR='$CACHE' FIXED_TOKENIZER_DIR=''
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
    TRAIN_DATA_MODE=ngram5_blocks NGRAM5_DATA_DIR='$CLUSTER_ROOT/$GEN_OUT'
    MAX_TRAINING_STEPS=$MAXSTEPS DEVICE_BATCH_SIZE=72 TOTAL_BATCH_SIZE=589824
    VAL_LOSS_INTERVAL_STEPS=10 VAL_LOSS_BATCHES=4 LR_SCHEDULE_MODE=baseline
    NGRAM5_PROBE_STEPS='$PROBE_STEPS'
    NGRAM5_PROBE_FREQUENCY_MODE=exact_context
    NGRAM5_BUCKET_EDGES='0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001'
    NGRAM5_TRACE_ALL_BATCHES=0 NGRAM5_TRACE_COMPRESSION=1
    NGRAM_GLOBAL_FREQUENCY_MODE=baseline
    NGRAM_GLOBAL_FREQUENCY_DIR='$CLUSTER_ROOT/$GEN_OUT'
    TORCH_COMPILE=0
    NGRAM5_CKPT_DIR='$RESULT_DIR/checkpoints'
    NGRAM5_CKPT_INTERVAL_STEPS=1000 NGRAM5_CKPT_STEPS='$CKPT_FULL'
    NGRAM5_CKPT_KEEP_MODEL_ONLY=2
    REMOTE_RESULT_DIR='$RESULT_DIR' RUN_ID=ngram5_full163_continuous
    MASTER_PORT=29500
  )
  if [[ -n '$INIT_CKPT' ]]; then
    ENV_VARS+=(NGRAM5_INIT_CKPT='$INIT_CKPT')
  fi
  nohup env \"\${ENV_VARS[@]}\" \$CMD > '$RESULT_DIR/train.log' 2>&1 &
  echo \"[train] pid: \$! result: $RESULT_DIR\"
  echo \"[train] log: $RESULT_DIR/train.log\"
"

echo "=== monitor: ssh $SSH_HOST 'tail -f $RESULT_DIR/train.log' ==="
