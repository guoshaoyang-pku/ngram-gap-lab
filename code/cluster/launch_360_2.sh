#!/usr/bin/env bash
# Auto-generated launch script for 360-2
# Each GPU runs its queue serially; all GPUs run in parallel.
# A failed run logs an error but does NOT stop the queue.
set -uo pipefail

ROOT="/data/home/guoshaoyang/ngram-gap-lab"
LOGDIR="$ROOT/logs/rerun"
mkdir -p "$LOGDIR"

# GPU 4: 5 runs, est 160min
(
  echo "[GPU 4] 1/5: nglab4x_input_fv_v3_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab4x_input_fv_v3_fixed"
  CUDA_VISIBLE_DEVICES="4" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab4x_input_fv_v3_fixed \
    --injection_position input \
    --steps 5000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,2,3,4 \
    --val_shards 5,6,7,8,9,10,6542 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train4x.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab4x_input_fv_v3_fixed/train.log" 2>&1 || echo "[GPU 4] nglab4x_input_fv_v3_fixed FAILED"
  echo "[GPU 4] 2/5: nglab1x_opt_rmsprop_2x_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_fixed"
  CUDA_VISIBLE_DEVICES="4" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_rmsprop_2x_fixed \
    --injection_position input \
    --steps 2000 \
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
    --table_optimizer rmsprop \
    --table_lr_scale 2.0 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_fixed/train.log" 2>&1 || echo "[GPU 4] nglab1x_opt_rmsprop_2x_fixed FAILED"
  echo "[GPU 4] 3/5: nglab2x_opt_rmsprop_2x_b2_098_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_098_fixed"
  CUDA_VISIBLE_DEVICES="4" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_2x_b2_098_fixed \
    --injection_position input \
    --steps 2000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,2 \
    --val_shards 3,4,5,6,7,8,9,10,6542 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train2x_fine.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer rmsprop \
    --table_lr_scale 2.0 \
    --table_betas 0.0,0.98 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_098_fixed/train.log" 2>&1 || echo "[GPU 4] nglab2x_opt_rmsprop_2x_b2_098_fixed FAILED"
  echo "[GPU 4] 4/5: nglab025x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab025x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="4" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab025x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 4] nglab025x_b2_099_fixed FAILED"
  echo "[GPU 4] 5/5: nglab1x_opt_adamw_090999_s44_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_090999_s44_fixed"
  CUDA_VISIBLE_DEVICES="4" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.9,0.999 > "$ROOT/data/runs/nglab1x_opt_adamw_090999_s44_fixed/train.log" 2>&1 || echo "[GPU 4] nglab1x_opt_adamw_090999_s44_fixed FAILED"
  echo "[GPU 4] queue done at $(date)"
) &

# GPU 5: 5 runs, est 160min
(
  echo "[GPU 5] 1/5: nglab3x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab3x_e6_fixed"
  CUDA_VISIBLE_DEVICES="5" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab3x_e6_fixed \
    --injection_position input \
    --steps 5000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,2,3 \
    --val_shards 4,5,6,7,8,9,10,6542 \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab3x_e6_fixed/train.log" 2>&1 || echo "[GPU 5] nglab3x_e6_fixed FAILED"
  echo "[GPU 5] 2/5: nglab1x_opt_rmsprop_2x_b2_098_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_b2_098_fixed"
  CUDA_VISIBLE_DEVICES="5" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_rmsprop_2x_b2_098_fixed \
    --injection_position input \
    --steps 2000 \
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
    --table_optimizer rmsprop \
    --table_lr_scale 2.0 \
    --table_betas 0.0,0.98 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_b2_098_fixed/train.log" 2>&1 || echo "[GPU 5] nglab1x_opt_rmsprop_2x_b2_098_fixed FAILED"
  echo "[GPU 5] 3/5: nglab2x_opt_rmsprop_2x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="5" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_2x_b2_099_fixed \
    --injection_position input \
    --steps 2000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,2 \
    --val_shards 3,4,5,6,7,8,9,10,6542 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train2x_fine.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer rmsprop \
    --table_lr_scale 2.0 \
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 5] nglab2x_opt_rmsprop_2x_b2_099_fixed FAILED"
  echo "[GPU 5] 4/5: nglab05x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab05x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="5" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab05x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 5] nglab05x_b2_099_fixed FAILED"
  echo "[GPU 5] 5/5: nglab1x_opt_rmsprop_2x_s43_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s43_fixed"
  CUDA_VISIBLE_DEVICES="5" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_lr_scale 2.0 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s43_fixed/train.log" 2>&1 || echo "[GPU 5] nglab1x_opt_rmsprop_2x_s43_fixed FAILED"
  echo "[GPU 5] queue done at $(date)"
) &

# GPU 6: 6 runs, est 155min
(
  echo "[GPU 6] 1/6: nglab2_5x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2_5x_e6_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2_5x_e6_fixed \
    --injection_position input \
    --steps 4190 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,2,64 \
    --val_shards 4,5,6,7,8,9,10,6542 \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab2_5x_e6_fixed/train.log" 2>&1 || echo "[GPU 6] nglab2_5x_e6_fixed FAILED"
  echo "[GPU 6] 2/6: nglab0_75x_input_fv_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_75x_input_fv_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab0_75x_input_fv_fixed \
    --injection_position input \
    --steps 2000 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train0_75x.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_75x_input_fv_fixed/train.log" 2>&1 || echo "[GPU 6] nglab0_75x_input_fv_fixed FAILED"
  echo "[GPU 6] 3/6: nglab2x_opt_rmsprop_1x_b2_09999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_1x_b2_09999_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_1x_b2_09999_fixed \
    --injection_position input \
    --steps 2000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,2 \
    --val_shards 3,4,5,6,7,8,9,10,6542 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train2x_fine.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer rmsprop \
    --table_lr_scale 1.0 \
    --table_betas 0.0,0.9999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_1x_b2_09999_fixed/train.log" 2>&1 || echo "[GPU 6] nglab2x_opt_rmsprop_1x_b2_09999_fixed FAILED"
  echo "[GPU 6] 4/6: nglab2x_opt_rmsprop_4x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_4x_b2_099_fixed \
    --injection_position input \
    --steps 2000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,2 \
    --val_shards 3,4,5,6,7,8,9,10,6542 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train2x_fine.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer rmsprop \
    --table_lr_scale 4.0 \
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 6] nglab2x_opt_rmsprop_4x_b2_099_fixed FAILED"
  echo "[GPU 6] 5/6: nglab1x_opt_adamw_090999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_090999_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.9,0.999 > "$ROOT/data/runs/nglab1x_opt_adamw_090999_fixed/train.log" 2>&1 || echo "[GPU 6] nglab1x_opt_adamw_090999_fixed FAILED"
  echo "[GPU 6] 6/6: nglab0_25x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_25x_e6_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_25x_e6_fixed/train.log" 2>&1 || echo "[GPU 6] nglab0_25x_e6_fixed FAILED"
  echo "[GPU 6] queue done at $(date)"
) &

# GPU 7: 6 runs, est 159min
(
  echo "[GPU 7] 1/6: nglab3x_input_fv_v3_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab3x_input_fv_v3_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab3x_input_fv_v3_fixed \
    --injection_position input \
    --steps 3800 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,2,3 \
    --val_shards 4,5,6,7,8,9,10,6542 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train3x.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab3x_input_fv_v3_fixed/train.log" 2>&1 || echo "[GPU 7] nglab3x_input_fv_v3_fixed FAILED"
  echo "[GPU 7] 2/6: nglab0_5x_input_fv_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_5x_input_fv_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab0_5x_input_fv_fixed \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_5x_input_fv_fixed/train.log" 2>&1 || echo "[GPU 7] nglab0_5x_input_fv_fixed FAILED"
  echo "[GPU 7] 3/6: nglab1x_opt_rmsprop_4x_b2_098_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_b2_098_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_rmsprop_4x_b2_098_fixed \
    --injection_position input \
    --steps 2000 \
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
    --table_optimizer rmsprop \
    --table_lr_scale 4.0 \
    --table_betas 0.0,0.98 > "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_b2_098_fixed/train.log" 2>&1 || echo "[GPU 7] nglab1x_opt_rmsprop_4x_b2_098_fixed FAILED"
  echo "[GPU 7] 4/6: nglab2x_opt_rmsprop_4x_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_4x_fixed \
    --injection_position input \
    --steps 2000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,2 \
    --val_shards 3,4,5,6,7,8,9,10,6542 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train2x_fine.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 \
    --table_optimizer rmsprop \
    --table_lr_scale 4.0 \
    --table_betas 0.0,0.999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_fixed/train.log" 2>&1 || echo "[GPU 7] nglab2x_opt_rmsprop_4x_fixed FAILED"
  echo "[GPU 7] 5/6: nglab0_75x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_75x_e6_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_75x_e6_fixed/train.log" 2>&1 || echo "[GPU 7] nglab0_75x_e6_fixed FAILED"
  echo "[GPU 7] 6/6: nglab0_5x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_5x_e6_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_5x_e6_fixed/train.log" 2>&1 || echo "[GPU 7] nglab0_5x_e6_fixed FAILED"
  echo "[GPU 7] queue done at $(date)"
) &

echo "All GPU queues launched. Waiting..."
wait
echo "All done at $(date)"
