#!/usr/bin/env bash
# One-shot status snapshot for the full-163 continuous pretraining run.
# Usage: ./monitor_full163.sh
SSH_HOST="${SSH_HOST:-ophis-gpu}"
CLUSTER_ROOT="${CLUSTER_ROOT:-/data3/guoshaoyang/ngram-gap-exp}"
GEN_LOG="$CLUSTER_ROOT/ngram5_data/gen_full163.log"
CACHE="$CLUSTER_ROOT/ngram5_data/token_cache_full163"
GEN_OUT="$CLUSTER_ROOT/ngram5_data/trigram_exact_alpha0.0_full163_20260808"

ssh "$SSH_HOST" "
  echo '=== time ==='; date '+%F %T'
  echo '=== data_gen ==='
  ps -o pid,rss,etime,pcpu -p \$(pgrep -f 'data_gen.py --out-dir ngram5_data/trigram_exact_alpha0.0_full163' | head -1) 2>/dev/null || echo '(not running)'
  echo 'token cache shards: '\$(ls '$CACHE' 2>/dev/null | grep -c npy)
  echo '--- gen log tail ---'
  tail -4 '$GEN_LOG' 2>/dev/null | tr '\r' '\n' | grep -vE '^\s*$' | tail -4
  echo '--- gen outputs ---'
  ls -la '$GEN_OUT' 2>/dev/null | tail -8 || echo '(no outputs yet)'
  echo '=== GPU ==='
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo '=== disk (quota) ==='
  quota -s 2>/dev/null | grep nvme3 || df -h '$CLUSTER_ROOT' | tail -1
  echo '=== training runs ==='
  ls -lat '$CLUSTER_ROOT/runs/ngram5_big/' 2>/dev/null | head -5
  for d in '\$(ls -d '$CLUSTER_ROOT'/runs/ngram5_big/continuous_v1_* 2>/dev/null | tail -1)'; do
    [ -n \"\$d\" ] || continue
    echo \"--- \$d ---\"
    tail -2 \"\$d/train.log\" 2>/dev/null | tr '\r' '\n' | tail -2
    wc -l \"\$d/training_loss.jsonl\" \"\$d/validation_loss.jsonl\" 2>/dev/null
    ls \"\$d/checkpoints/\" 2>/dev/null | tail -5
  done
"
