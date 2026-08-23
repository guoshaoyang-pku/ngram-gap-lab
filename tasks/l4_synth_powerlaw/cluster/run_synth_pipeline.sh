#!/usr/bin/env bash
# run_synth_night.sh — full synthetic-transition pilot on 360-2, inside tmux "synth".
# gen A -> gen B -> prep x2 -> smoke -> all (4 runs, 2 waves, 2 GPUs).
# Stages are skipped when their outputs already exist (resume-safe).
set -uo pipefail
ROOT=/data/home/guoshaoyang/ngram-gap-exp
PY=python3
LOG=/tmp/synth_night.log
DATA_A=$ROOT/ngram5_data/synth_A_sparse_restart
DATA_B=$ROOT/ngram5_data/synth_B_lowrank_sparse
{
  echo "=== $(date '+%F %T') synth night pipeline start ==="
  if [ ! -f "$DATA_A/meta.json" ]; then
    echo "--- gen A (sparse_restart) ---"
    (cd "$ROOT/toy" && "$PY" synthetic_transition_gen.py --out-dir "$DATA_A" --scheme sparse_restart) || { echo "GEN_A FAILED"; exit 1; }
  else
    echo "--- gen A: exists, skip ---"
  fi
  if [ ! -f "$DATA_B/meta.json" ]; then
    echo "--- gen B (lowrank_sparse) ---"
    (cd "$ROOT/toy" && "$PY" synthetic_transition_gen.py --out-dir "$DATA_B" --scheme lowrank_sparse) || { echo "GEN_B FAILED"; exit 1; }
  else
    echo "--- gen B: exists, skip ---"
  fi
  if [ ! -f "$DATA_A/meta.json" ] || [ ! -f "$DATA_B/meta.json" ]; then
    echo "--- prep A ---"
    "$PY" "$ROOT/toy/synthetic_prep.py" --data-dir "$DATA_A" || { echo "PREP_A FAILED"; exit 1; }
    echo "--- prep B ---"
    "$PY" "$ROOT/toy/synthetic_prep.py" --data-dir "$DATA_B" || { echo "PREP_B FAILED"; exit 1; }
  else
    echo "--- prep: meta.json present, skip ---"
  fi
  echo "--- smoke (GPU ${SYNTH_GPU_A:-3}) ---"
  bash "$ROOT/toy/run_synth_3602.sh" smoke || { echo "SMOKE FAILED"; exit 1; }
  echo "--- all (2 waves, 2 GPUs) ---"
  bash "$ROOT/toy/run_synth_3602.sh" all || { echo "ALL FAILED"; exit 1; }
  echo "=== $(date '+%F %T') synth night pipeline DONE ==="
} > "$LOG" 2>&1
