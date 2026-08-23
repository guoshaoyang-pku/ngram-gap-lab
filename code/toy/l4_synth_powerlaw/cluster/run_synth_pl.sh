#!/usr/bin/env bash
# run_synth_pl.sh — clean power-law gap toy on 360-2 (fine buckets, probabilistic
# rule, honest context-uniform val). 2026-08-07.
#   bash run_synth_pl.sh smoke        # 30-step smoke on auto GPU
#   bash run_synth_pl.sh all          # table on/off x seeds 42,43 (2 free GPUs, 2 waves)
set -uo pipefail

ROOT=${SYNTH_ROOT:-/data/home/guoshaoyang/ngram-gap-exp}
PY=python3
RUNS_DIR=$ROOT/runs/ngram5
DATA=$ROOT/ngram5_data/synth_pl_A
CACHE_DIR=$ROOT/toy/cache/base

pick_gpus() {
  local free
  free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1000 {print $1}')
  echo "$free" | head -2 | tr '\n' ' '
}

run_job() {
  local gpu=$1 name=$2 inject=$3 seed=$4 maxsteps=$5 probesteps=$6
  local out=$RUNS_DIR/$name
  mkdir -p "$out"
  echo "=== [$name] GPU $gpu inject=$inject seed=$seed steps=$maxsteps -> $out ==="
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
    NANOGPT_ADAM_LR=0.004 NGRAM_TABLE_BETAS=0.0,0.999 NGRAM_TABLE_LR_SCALE=1.0 \
    POSITION_ENCODING=learned_abs CURRENT_NORMALIZATION=layernorm \
    CURRENT_EMBEDDING_TYING=tied CURRENT_NGRAM_INJECTION_IMPL=none \
    CURRENT_EMBEDDING_INIT=nanogpt_like CURRENT_BLOCK_INIT=nanogpt_style \
    CURRENT_ATTENTION_NORM=none CURRENT_HEAD_GATE=none \
    CURRENT_RESIDUAL_PATH=plain CURRENT_LAYER_POOL=none \
    CURRENT_MLP=gelu CURRENT_LOGIT_SOFTCAP=none CURRENT_LINEAR_BIAS=none \
    WINDOW_PATTERN=LLLL SEED=$seed TRAIN_DATA_SEED=$seed TRAIN_DATA_MODE=ngram5_blocks \
    NGRAM5_DATA_DIR=$DATA \
    MAX_TRAINING_STEPS=$maxsteps DEVICE_BATCH_SIZE=4 TOTAL_BATCH_SIZE=8192 \
    VAL_LOSS_INTERVAL_STEPS=10 VAL_LOSS_BATCHES=4 LR_SCHEDULE_MODE=baseline \
    NGRAM5_PROBE_STEPS=$probesteps NGRAM5_PROBE_BATCHES=8 \
    NGRAM5_PROBE_FREQUENCY_MODE=exact_context \
    NGRAM5_BUCKET_EDGES=0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001 \
    NGRAM5_TRACE_ALL_BATCHES=1 NGRAM5_TRACE_COMPRESSION=1 \
    NGRAM_GLOBAL_FREQUENCY_MODE=baseline \
    NGRAM_GLOBAL_FREQUENCY_DIR=$DATA \
    TORCH_COMPILE=0 \
    TORCHINDUCTOR_CACHE_DIR=$ROOT/.inductor_cache TRITON_CACHE_DIR=$ROOT/.triton_cache \
    REMOTE_RESULT_DIR=$out RUN_ID=$name \
    "$PY" -u ngram5_freq_gap/trainer.py > "$out/train.log" 2>&1
  local rc=$?
  printf '%s\n' "$rc" > "$out/exit_code.txt"
  [ "$rc" -eq 0 ] && touch "$out/done"
  echo "=== [$name] finished rc=$rc ==="
  return "$rc"
}

cd "$ROOT"
MODE=${1:-all}
case "$MODE" in
  smoke)
    run_job 3 synth_pl_smoke nanogpt 42 30 10,20,30 || exit 1
    ;;
  all)
    if [ -z "${SYNTH_GPU_A:-}" ] || [ -z "${SYNTH_GPU_B:-}" ]; then
      read -r GPU_A GPU_B <<< "$(pick_gpus)"
      GPU_A=${GPU_A:-3}; GPU_B=${GPU_B:-6}
      echo "auto-selected GPUs: $GPU_A / $GPU_B"
    fi
    run_job "$GPU_A" synth_pl_A_nanogpt_s42 nanogpt 42 2000 100,200,400,600,800,1000,1500,2000 &
    p1=$!
    run_job "$GPU_B" synth_pl_A_nanogpt_s43 nanogpt 43 2000 100,200,400,600,800,1000,1500,2000 &
    p2=$!
    wait "$p1" || exit 1
    wait "$p2" || exit 1
    run_job "$GPU_A" synth_pl_A_nogram_s42 none 42 2000 100,200,400,600,800,1000,1500,2000 &
    p3=$!
    run_job "$GPU_B" synth_pl_A_nogram_s43 none 43 2000 100,200,400,600,800,1000,1500,2000 &
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
    run_job "$GPU_A" synth_pl_A_nogram_s42 none 42 2000 100,200,400,600,800,1000,1500,2000 &
    p1=$!
    run_job "$GPU_B" synth_pl_A_nogram_s43 none 43 2000 100,200,400,600,800,1000,1500,2000 &
    p2=$!
    wait "$p1" || exit 1
    wait "$p2" || exit 1
    ;;
esac
echo "=== queue done ==="
