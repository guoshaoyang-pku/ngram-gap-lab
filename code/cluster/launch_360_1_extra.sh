#!/usr/bin/env bash
# Extra runs on 360-1 GPU 0,2,3,4,5
set -uo pipefail

ROOT="/data/home/guoshaoyang/ngram-gap-lab"

# GPU 0: 1 runs
(
  echo "[GPU 0] 1/1: nglab05x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab05x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab05x_b2_099_fixed \
    --injection_position input \
    --steps 2000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 60 \
    --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --device_batch_size 72 \
    --total_batch_size 147456 \
    --val_interval 10 \
    --val_batches 4 \
    --table_norm_interval 10 \
    --lr 0.004 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train0_5x.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer rmsprop \
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab05x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 0] nglab05x_b2_099_fixed FAILED"
  echo "[GPU 0] done at $(date)"
) &

# GPU 2: 1 runs
(
  echo "[GPU 2] 1/1: nglab025x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab025x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="2" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab025x_b2_099_fixed \
    --injection_position input \
    --steps 2000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 62 \
    --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --device_batch_size 72 \
    --total_batch_size 147456 \
    --val_interval 10 \
    --val_batches 4 \
    --table_norm_interval 10 \
    --lr 0.004 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train0_25x.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer rmsprop \
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab025x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 2] nglab025x_b2_099_fixed FAILED"
  echo "[GPU 2] done at $(date)"
) &

# GPU 3: 2 runs
(
  echo "[GPU 3] 1/2: nglab0_75x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_75x_e6_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab0_75x_e6_fixed \
    --injection_position input \
    --steps 1260 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 63 \
    --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --device_batch_size 72 \
    --total_batch_size 147456 \
    --val_interval 10 \
    --val_batches 4 \
    --table_norm_interval 10 \
    --lr 0.004 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_75x_e6_fixed/train.log" 2>&1 || echo "[GPU 3] nglab0_75x_e6_fixed FAILED"
  echo "[GPU 3] 2/2: nglab0_25x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_25x_e6_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab0_25x_e6_fixed \
    --injection_position input \
    --steps 420 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 62 \
    --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --device_batch_size 72 \
    --total_batch_size 147456 \
    --val_interval 10 \
    --val_batches 4 \
    --table_norm_interval 10 \
    --lr 0.004 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_25x_e6_fixed/train.log" 2>&1 || echo "[GPU 3] nglab0_25x_e6_fixed FAILED"
  echo "[GPU 3] done at $(date)"
) &

# GPU 4: 2 runs
(
  echo "[GPU 4] 1/2: nglab1x_opt_rmsprop_2x_s43_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s43_fixed"
  CUDA_VISIBLE_DEVICES="4" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_rmsprop_2x_s43_fixed \
    --injection_position input \
    --steps 1000 \
    --seed 43 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1 \
    --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --device_batch_size 72 \
    --total_batch_size 147456 \
    --val_interval 10 \
    --val_batches 4 \
    --table_norm_interval 10 \
    --lr 0.004 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer rmsprop \
    --table_lr_scale 2.0 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s43_fixed/train.log" 2>&1 || echo "[GPU 4] nglab1x_opt_rmsprop_2x_s43_fixed FAILED"
  echo "[GPU 4] 2/2: nglab1x_opt_adamw_090999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_090999_fixed"
  CUDA_VISIBLE_DEVICES="4" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_adamw_090999_fixed \
    --injection_position input \
    --steps 1000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1 \
    --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --device_batch_size 72 \
    --total_batch_size 147456 \
    --val_interval 10 \
    --val_batches 4 \
    --table_norm_interval 10 \
    --lr 0.004 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer adamw \
    --table_betas 0.9,0.999 > "$ROOT/data/runs/nglab1x_opt_adamw_090999_fixed/train.log" 2>&1 || echo "[GPU 4] nglab1x_opt_adamw_090999_fixed FAILED"
  echo "[GPU 4] done at $(date)"
) &

# GPU 5: 2 runs
(
  echo "[GPU 5] 1/2: nglab1x_opt_adamw_090999_s44_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_090999_s44_fixed"
  CUDA_VISIBLE_DEVICES="5" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_adamw_090999_s44_fixed \
    --injection_position input \
    --steps 1000 \
    --seed 44 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1 \
    --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --device_batch_size 72 \
    --total_batch_size 147456 \
    --val_interval 10 \
    --val_batches 4 \
    --table_norm_interval 10 \
    --lr 0.004 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer adamw \
    --table_betas 0.9,0.999 > "$ROOT/data/runs/nglab1x_opt_adamw_090999_s44_fixed/train.log" 2>&1 || echo "[GPU 5] nglab1x_opt_adamw_090999_s44_fixed FAILED"
  echo "[GPU 5] 2/2: nglab0_5x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_5x_e6_fixed"
  CUDA_VISIBLE_DEVICES="5" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab0_5x_e6_fixed \
    --injection_position input \
    --steps 840 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 60 \
    --val_shards 2,3,4,5,6,7,8,9,10,6542 \
    --device_batch_size 72 \
    --total_batch_size 147456 \
    --val_interval 10 \
    --val_batches 4 \
    --table_norm_interval 10 \
    --lr 0.004 \
    --enable_unigram 0 \
    --enable_bigram 1 \
    --enable_trigram 1 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_5x_e6_fixed/train.log" 2>&1 || echo "[GPU 5] nglab0_5x_e6_fixed FAILED"
  echo "[GPU 5] done at $(date)"
) &

wait
echo "All extra done at $(date)"
