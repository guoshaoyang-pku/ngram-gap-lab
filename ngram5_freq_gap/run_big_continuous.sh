#!/usr/bin/env bash
# Stage 1 for the full-corpus continuous pretraining run:
#   - sync ngram5_freq_gap code to the cluster
#   - install the full-163 data split (train = all shards except 06542,
#     val = 06542, the group-canonical unseen validation shard)
#   - launch data_gen in the background (token-cached, binary uint16 output)
set -euo pipefail

SSH_HOST="${SSH_HOST:-ophis-gpu}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/data3/guoshaoyang/ngram-gap-exp}"
# The cluster canonical lib.py resolves DATA_DIR = AUTORESEARCH_CACHE_DIR/data
# and tokenizer = AUTORESEARCH_CACHE_DIR/tokenizer (no DATA_DIR_OVERRIDE
# support).  Point AUTORESEARCH_CACHE_DIR at a cache-home whose data/ and
# tokenizer/ are symlinks to the full-163 corpus and the fixed tokenizer.
FULL_CACHE_HOME="${FULL_CACHE_HOME:-$CLUSTER_ROOT/ngram5_data/full163_cache_home}"
FULL_DATA_DIR="${FULL_DATA_DIR:-/data2/shared/ncpl-pathA/harry_autoresearch_full_data/data}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/data2/ncpl-pathA/work/vbird_autoresearch/cache/tokenizer}"
CLUSTER_PY="${CLUSTER_PY:-$CLUSTER_ROOT/.venv/bin/python}"
GEN_OUT="${GEN_OUT:-ngram5_data/trigram_exact_alpha0.0_full163_20260808}"
TOKEN_CACHE="${TOKEN_CACHE:-ngram5_data/token_cache_full163}"
LOCAL_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/3] syncing ngram5_freq_gap + full-163 data split ==="
rsync -avz --exclude '__pycache__' --exclude '*.pyc' --exclude 'data/' \
  --exclude 'run_artifacts/' --exclude 'runs/' \
  "$LOCAL_HERE/" "$SSH_HOST:$CLUSTER_ROOT/ngram5_freq_gap/"
scp "$LOCAL_HERE/../data_split.full163.json" "$SSH_HOST:$CLUSTER_ROOT/data_split.json"

echo "=== [2/3] launching data generation in the background ==="
ssh "$SSH_HOST" "
  set -e
  cd '$CLUSTER_ROOT'
  mkdir -p '$FULL_CACHE_HOME'
  ln -sfn '$FULL_DATA_DIR' '$FULL_CACHE_HOME/data'
  ln -sfn '$TOKENIZER_DIR' '$FULL_CACHE_HOME/tokenizer'
  if [[ -f '$CLUSTER_ROOT/$GEN_OUT/meta.json' ]]; then
    echo '[gen] dataset already exists: $GEN_OUT (skipping)'
  else
    mkdir -p '$CLUSTER_ROOT/ngram5_data'
    nohup env AUTORESEARCH_CACHE_DIR='$FULL_CACHE_HOME' FIXED_TOKENIZER_DIR='' \\
      '$CLUSTER_PY' -u ngram5_freq_gap/data_gen.py \\
        --out-dir '$GEN_OUT' \\
        --tokenizer-dir '$FULL_CACHE_HOME/tokenizer' \\
        --alpha 0.0 --bucket-count 5000000 --order 3 \\
        --f-train 0.8 --f-val 0.2 --k-min 0.25 --k-max 8.0 \\
        --r-ref-mode median --dataset-seed 20260808 --doc-len 2048 \\
        --val-source test --val-frac 0.02 \\
        --emit-format bin \\
        --fast-scan --fast-emit \\
        --token-cache '$TOKEN_CACHE' \\
        > 'ngram5_data/gen_full163.log' 2>&1 &
    echo \"[gen] pid: \$!\"
  fi
"

echo "=== [3/3] done ==="
echo "monitor:  ssh $SSH_HOST 'tail -f $CLUSTER_ROOT/ngram5_data/gen_full163.log'"
echo "after meta.json appears, launch training with: bash ngram5_freq_gap/launch_ddp_train.sh"
