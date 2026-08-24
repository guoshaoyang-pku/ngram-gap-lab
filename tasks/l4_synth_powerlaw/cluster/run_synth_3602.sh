#!/usr/bin/env bash
# run_synth_3602.sh — synthetic-transition (order=5) pilot runs on 360-2.
# 2026-08-06 user instruction: toy-model experiments on 360-2, val interval 10.
# Usage:
#   bash run_synth_3602.sh smoke            # 30-step smoke (GPU auto/default 3)
#   bash run_synth_3602.sh all             # A/B inject parallel (2 free GPUs), then A/B no-ngram
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GPU_A=${SYNTH_GPU_A:-3}
GPU_B=${SYNTH_GPU_B:-6}
PY="${NGLAB_PY:-$REPO_ROOT/.venv/bin/python}"
RUNS_DIR="${SYNTH_RUNS_DIR:-$TASK_ROOT/results/runs}"
DATA_A="${SYNTH_DATA_A:-$TASK_ROOT/results/inputs/synth_A_sparse_restart}"
DATA_B="${SYNTH_DATA_B:-$TASK_ROOT/results/inputs/synth_B_lowrank_sparse}"
CACHE_DIR="${SYNTH_CACHE_DIR:?set SYNTH_CACHE_DIR to the external tokenizer/cache directory}"
TRAINER="$REPO_ROOT/ngram5_freq_gap/trainer.py"

pick_gpus() {
  local free
  free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1}')
  echo "$free" | head -2 | tr '\n' ' '
}

run_job() {
  local gpu=$1 name=$2 data_dir=$3 inject=$4 maxsteps=$5 probesteps=$6
  local out=$RUNS_DIR/$name
  mkdir -p "$out"
  echo "=== [$name] GPU $gpu inject=$inject steps=$maxsteps -> $out ==="
  CUDA_VISIBLE_DEVICES=$gpu env \
    AUTORESEARCH_CACHE_DIR=$CACHE_DIR \
    FIXED_TOKENIZER_DIR=$CACHE_DIR/tokenizer \
    ARCH_VARIANT=nanogpt_original NGRAM5_TRUNK=transformer \
    NANOGPT_NGRAM_OPTIMIZER=mixed NGRAM_TABLE_OPTIMIZER=rmsprop \
    NANOGPT_MATRIX_OPTIMIZER=adamw NANOGPT_OPTIMIZER_GROUPING=nanogpt \
    NANOGPT_ATTENTION_IMPL=fused \
    NANOGPT_NGRAM_INJECTION_IMPL=nanogpt NANOGPT_NGRAM_INJECTION_POSITION=input \
    NANOGPT_ENABLE_NGRAM_VE=$([ "$inject" = nanogpt ] && echo 1 || echo 0) \
    ENABLE_UNIGRAM_VE=0 ENABLE_BIGRAM_VE=1 ENABLE_TRIGRAM_VE=1 ENABLE_FOURGRAM_VE=0 \
    NANOGPT_ADAM_LR=0.004 NGRAM_TABLE_BETAS=0.0,0.99 NGRAM_TABLE_LR_SCALE=2.0 \
    POSITION_ENCODING=learned_abs CURRENT_NORMALIZATION=layernorm \
    CURRENT_EMBEDDING_TYING=tied CURRENT_NGRAM_INJECTION_IMPL=none \
    CURRENT_EMBEDDING_INIT=nanogpt_like CURRENT_BLOCK_INIT=nanogpt_style \
    CURRENT_ATTENTION_NORM=none CURRENT_HEAD_GATE=none \
    CURRENT_RESIDUAL_PATH=plain CURRENT_LAYER_POOL=none \
    CURRENT_MLP=gelu CURRENT_LOGIT_SOFTCAP=none CURRENT_LINEAR_BIAS=none \
    WINDOW_PATTERN=LLLL SEED=42 TRAIN_DATA_SEED=42 TRAIN_DATA_MODE=ngram5_blocks \
    NGRAM5_DATA_DIR=$data_dir \
    MAX_TRAINING_STEPS=$maxsteps DEVICE_BATCH_SIZE=4 TOTAL_BATCH_SIZE=8192 \
    VAL_LOSS_INTERVAL_STEPS=10 VAL_LOSS_BATCHES=4 LR_SCHEDULE_MODE=baseline \
    NGRAM5_PROBE_STEPS=$probesteps \
    NGRAM5_PROBE_FREQUENCY_MODE=exact_context \
    NGRAM5_BUCKET_EDGES=0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001 \
    NGRAM5_TRACE_ALL_BATCHES=1 NGRAM5_TRACE_COMPRESSION=1 \
    NGRAM_GLOBAL_FREQUENCY_MODE=baseline \
    NGRAM_GLOBAL_FREQUENCY_DIR=$data_dir \
    TORCH_COMPILE=0 \
    TORCHINDUCTOR_CACHE_DIR=$ROOT/.inductor_cache TRITON_CACHE_DIR=$ROOT/.triton_cache \
    REMOTE_RESULT_DIR=$out RUN_ID=$name \
    "$PY" -u "$TRAINER" > "$out/train.log" 2>&1
  local rc=$?
  printf '%s\n' "$rc" > "$out/exit_code.txt"
  [ "$rc" -eq 0 ] && touch "$out/done"
  echo "=== [$name] finished rc=$rc ==="
  return "$rc"
}

cd "$REPO_ROOT"
MODE=${1:-all}
case "$MODE" in
  smoke)
    run_job "$GPU_A" synth_smoke_A_sparse_restart $DATA_A nanogpt 30 10,20,30 || exit 1
    ;;
  all)
    if [ -z "${SYNTH_GPU_A:-}" ] || [ -z "${SYNTH_GPU_B:-}" ]; then
      read -r GPU_A GPU_B <<< "$(pick_gpus)"
      GPU_A=${GPU_A:-3}; GPU_B=${GPU_B:-6}
      echo "auto-selected GPUs: $GPU_A / $GPU_B"
    fi
    run_job "$GPU_A" synth_A_sparse_restart_s42 $DATA_A nanogpt 2000 100,200,400,600,800,1000,1500,2000 &
    p1=$!
    run_job "$GPU_B" synth_B_lowrank_sparse_s42 $DATA_B nanogpt 2000 100,200,400,600,800,1000,1500,2000 &
    p2=$!
    wait "$p1" || exit 1
    wait "$p2" || exit 1
    run_job "$GPU_A" synth_A_sparse_restart_nogram_s42 $DATA_A none 2000 100,200,400,600,800,1000,1500,2000 &
    p3=$!
    run_job "$GPU_B" synth_B_lowrank_sparse_nogram_s42 $DATA_B none 2000 100,200,400,600,800,1000,1500,2000 &
    p4=$!
    wait "$p3" || exit 1
    wait "$p4" || exit 1
    ;;
  nog)
    if [ -z "${SYNTH_GPU_A:-}" ] || [ -z "${SYNTH_GPU_B:-}" ]; then
      read -r GPU_A GPU_B <<< "$(pick_gpus)"
      GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}
      echo "auto-selected GPUs: $GPU_A / $GPU_B"
    fi
    run_job "$GPU_A" synth_A_sparse_restart_nogram_s42 $DATA_A none 2000 100,200,400,600,800,1000,1500,2000 &
    p1=$!
    run_job "$GPU_B" synth_B_lowrank_sparse_nogram_s42 $DATA_B none 2000 100,200,400,600,800,1000,1500,2000 &
    p2=$!
    wait "$p1" || exit 1
    wait "$p2" || exit 1
    ;;
esac
echo "=== queue done ==="
