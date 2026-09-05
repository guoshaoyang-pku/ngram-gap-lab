#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

/usr/bin/python3 -u code/train.py \
  --run_id theory_zipf_iid_ngram_on_s42_fixed \
  --out_dir runs_fixed \
  --data_dir data/theory_zipf_iid_mainline_aligned_20260904 \
  --lr 0.0006 \
  --table_lr_scale 128.0 \
  --bigram_clean_table 1048576 \
  --trigram_clean_table 1048576
