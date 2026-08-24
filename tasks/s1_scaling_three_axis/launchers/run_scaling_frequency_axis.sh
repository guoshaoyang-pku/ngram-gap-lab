#!/usr/bin/env bash
# Frequency-axis runs (plan §5 P2).  User decision 2026-08-24:
# ordinary grid runs skip exact-frequency diagnostics; the frequency axis
# is a SEPARATE small run set, one per module arm at the L4 / 1M table point.
#
# Each run computes exact-frequency + freq-bin + fixed-train-probe
# diagnostics (--freq_index set, --exact_freq_eval_interval 100).
#
# Arms (L4 = 337 batches/epoch = FULL shard 1, 1M table = table_mult 64):
#   freq_bigram / freq_trigram / freq_both / freq_nogram
# Both fixed-step (1000 steps) and fixed-epoch (6 epochs, 2022 steps) are run
# so G(E,f) can be read at epoch 3 and epoch 6 cross-sections.
#
# Usage: ./run_scaling_frequency_axis.sh [gpu1] [gpu2] [gpu3] [gpu4]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${NGLAB_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
TASK_ROOT="$ROOT/tasks/s1_scaling_three_axis"
PY="${NGLAB_PY:-$ROOT/.venv/bin/python}"
DATA_DIR="$ROOT/data/tokenized"
OUT_DIR="$ROOT/data/runs_scaling"
FREQ_IDX="$ROOT/data/freq_index.npz"
G1="${1:-0}" G2="${2:-1}" G3="${3:-2}" G4="${4:-3}"
mkdir -p "$OUT_DIR"

run_arm() {  # run_arm <gpu> <run_id> <steps> <schedule_epochs> <bigram> <trigram>
  local GPU="$1" RUN_ID="$2" STEPS="$3" SCHED="$4" BI="$5" TRI="$6"
  local RESULT_DIR="$OUT_DIR/${RUN_ID}_fixed"
  mkdir -p "$RESULT_DIR"
  echo "[freq-axis] $RUN_ID steps=$STEPS sched=$SCHED bi=$BI tri=$TRI -> GPU $GPU at $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$TASK_ROOT/code/train.py" \
    --run_id "${RUN_ID}_fixed" --injection_position input \
    --steps "$STEPS" --seed 42 \
    --data_dir "$DATA_DIR" --out_dir "$OUT_DIR" \
    --device_batch_size 72 --total_batch_size 147456 \
    --val_interval 10 --val_batches 4 --table_norm_interval 10 --lr 0.004 \
    --enable_unigram 0 --enable_bigram "$BI" --enable_trigram "$TRI" \
    --n_layer 8 --n_head 6 --n_embd 768 --vocab_size 8192 --sequence_len 2048 \
    --freq_eval_interval 10 --freq_eval_batches 4 \
    --train_shards 1 --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --freq_index "$FREQ_IDX" \
    --exact_freq_eval_interval 100 \
    --epoch_batches 337 \
    --fixed_train_probe 4 --probe_eval_interval 10 \
    --table_betas 0.0,0.99 \
    --table_lr_scale 2.0 \
    --table_mult 64 \
    --dtype bf16 --compile \
    ${SCHED:+--lr_schedule_epochs "$SCHED"} \
    > "$RESULT_DIR/train.log" 2>&1
  echo "[freq-axis] $RUN_ID done (exit=$?) at $(date)"
}

# Fixed-step arms (1000 steps, step-anchored LR): 4 module arms
run_arm "$G1" freq_bigram_fs  1000 0 1 0 &
run_arm "$G2" freq_trigram_fs 1000 0 0 1 &
run_arm "$G3" freq_both_fs    1000 0 1 1 &
run_arm "$G4" freq_nogram_fs  1000 0 0 0 &
wait

# Fixed-epoch arms (6 full epochs, epoch-anchored LR): 4 module arms
run_arm "$G1" freq_bigram_fe  2022 6 1 0 &
run_arm "$G2" freq_trigram_fe 2022 6 0 1 &
run_arm "$G3" freq_both_fe    2022 6 1 1 &
run_arm "$G4" freq_nogram_fe  2022 6 0 0 &
wait

echo "=== frequency axis done at $(date) ==="
