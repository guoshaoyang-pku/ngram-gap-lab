#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=2

exec /usr/bin/python3 -u code/train.py \
  --run_id theory_zipf_iid_ngram_on_freq_s42_fixed_rerun_20260905 \
  --out_dir runs_fixed \
  --data_dir data/theory_zipf_iid_mainline_aligned_20260904 \
  --steps 1000 \
  --seed 42 \
  --lr 0.0006 \
  --lr_schedule warmup_constant \
  --warmup_steps 100 \
  --table_betas 0.0,0.99 \
  --table_lr_scale 128.0 \
  --bigram_clean_table 1048576 \
  --trigram_clean_table 1048576 \
  --freq_index data/theory_zipf_iid_mainline_aligned_20260904/freq_index.npz
