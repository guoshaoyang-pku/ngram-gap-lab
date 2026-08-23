#!/usr/bin/env bash
# Auto-generated launch script for 360-1
# Each GPU runs its queue serially; all GPUs run in parallel.
# A failed run logs an error but does NOT stop the queue.
set -uo pipefail

ROOT="/data/home/guoshaoyang/ngram-gap-lab"
LOGDIR="$ROOT/logs/rerun"
mkdir -p "$LOGDIR"

# GPU 1: 6 runs, est 160min
(
  echo "[GPU 1] 1/6: nglab2x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_e6_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_e6_fixed \
    --injection_position input \
    --steps 3350 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab2x_e6_fixed/train.log" 2>&1 || echo "[GPU 1] nglab2x_e6_fixed FAILED"
  echo "[GPU 1] 2/6: nglab2x_input_v10_fv_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_input_v10_fv_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_input_v10_fv_fixed \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab2x_input_v10_fv_fixed/train.log" 2>&1 || echo "[GPU 1] nglab2x_input_v10_fv_fixed FAILED"
  echo "[GPU 1] 3/6: nglab1x_opt_rmsprop_4x_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_rmsprop_4x_fixed \
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
    --table_lr_scale 4.0 > "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_fixed/train.log" 2>&1 || echo "[GPU 1] nglab1x_opt_rmsprop_4x_fixed FAILED"
  echo "[GPU 1] 4/6: nglab2x_opt_rmsprop_2x_b2_099999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099999_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_2x_b2_099999_fixed \
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
    --table_betas 0.0,0.99999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099999_fixed/train.log" 2>&1 || echo "[GPU 1] nglab2x_opt_rmsprop_2x_b2_099999_fixed FAILED"
  echo "[GPU 1] 5/6: nglab1x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_e6_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_e6_fixed \
    --injection_position input \
    --steps 1685 \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab1x_e6_fixed/train.log" 2>&1 || echo "[GPU 1] nglab1x_e6_fixed FAILED"
  echo "[GPU 1] 6/6: nglab1x_opt_sgd_09_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_sgd_09_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_sgd_09_fixed \
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
    --table_optimizer sgd \
    --table_betas 0.9,0.0 > "$ROOT/data/runs/nglab1x_opt_sgd_09_fixed/train.log" 2>&1 || echo "[GPU 1] nglab1x_opt_sgd_09_fixed FAILED"
  echo "[GPU 1] queue done at $(date)"
) &

# GPU 3: 6 runs, est 155min
(
  echo "[GPU 3] 1/6: nglab2_5x_input_fv_v3_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2_5x_input_fv_v3_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2_5x_input_fv_v3_fixed \
    --injection_position input \
    --steps 3200 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train2_5x.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab2_5x_input_fv_v3_fixed/train.log" 2>&1 || echo "[GPU 3] nglab2_5x_input_fv_v3_fixed FAILED"
  echo "[GPU 3] 2/6: nglab1x_v10_nogram_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_v10_nogram_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_v10_nogram_fixed \
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
    --enable_bigram 0 \
    --enable_trigram 0 \
    --n_layer 8 \
    --n_head 6 \
    --n_embd 768 \
    --vocab_size 8192 \
    --sequence_len 2048 \
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab1x_v10_nogram_fixed/train.log" 2>&1 || echo "[GPU 3] nglab1x_v10_nogram_fixed FAILED"
  echo "[GPU 3] 3/6: nglab1x_opt_rmsprop_2x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_rmsprop_2x_b2_099_fixed \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 3] nglab1x_opt_rmsprop_2x_b2_099_fixed FAILED"
  echo "[GPU 3] 4/6: nglab2x_opt_rmsprop_2x_b2_09999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_09999_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_2x_b2_09999_fixed \
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
    --table_betas 0.0,0.9999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_09999_fixed/train.log" 2>&1 || echo "[GPU 3] nglab2x_opt_rmsprop_2x_b2_09999_fixed FAILED"
  echo "[GPU 3] 5/6: nglab1x_v10_input_nofb_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_v10_input_nofb_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_v10_input_nofb_fixed \
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
    --sequence_len 2048 > "$ROOT/data/runs/nglab1x_v10_input_nofb_fixed/train.log" 2>&1 || echo "[GPU 3] nglab1x_v10_input_nofb_fixed FAILED"
  echo "[GPU 3] 6/6: smoke_fixed_verify at $(date)"
  mkdir -p "$ROOT/data/runs/smoke_fixed_verify"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id smoke_fixed_verify \
    --injection_position input \
    --steps 400 \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/smoke_fixed_verify/train.log" 2>&1 || echo "[GPU 3] smoke_fixed_verify FAILED"
  echo "[GPU 3] queue done at $(date)"
) &

# GPU 6: 6 runs, est 154min
(
  echo "[GPU 6] 1/6: nglab1_5x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1_5x_e6_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1_5x_e6_fixed \
    --injection_position input \
    --steps 2525 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,61 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab1_5x_e6_fixed/train.log" 2>&1 || echo "[GPU 6] nglab1_5x_e6_fixed FAILED"
  echo "[GPU 6] 2/6: nglab1x_v10_input_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_v10_input_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_v10_input_fixed \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab1x_v10_input_fixed/train.log" 2>&1 || echo "[GPU 6] nglab1x_v10_input_fixed FAILED"
  echo "[GPU 6] 3/6: nglab1_5x_input_fv_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1_5x_input_fv_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1_5x_input_fv_fixed \
    --injection_position input \
    --steps 2000 \
    --seed 42 \
    --data_dir /data/home/guoshaoyang/ngram-gap-lab/data/tokenized \
    --out_dir /data/home/guoshaoyang/ngram-gap-lab/data/runs \
    --train_shards 1,61 \
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
    --freq_index /data/home/guoshaoyang/ngram-gap-lab/data/freq_index_train1_5x.npz \
    --freq_eval_interval 10 \
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab1_5x_input_fv_fixed/train.log" 2>&1 || echo "[GPU 6] nglab1_5x_input_fv_fixed FAILED"
  echo "[GPU 6] 4/6: nglab2x_opt_rmsprop_2x_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_2x_fixed \
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
    --table_betas 0.0,0.999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_fixed/train.log" 2>&1 || echo "[GPU 6] nglab2x_opt_rmsprop_2x_fixed FAILED"
  echo "[GPU 6] 5/6: nglab2x_opt_rmsprop_4x_b2_09999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_09999_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_4x_b2_09999_fixed \
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
    --table_betas 0.0,0.9999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_09999_fixed/train.log" 2>&1 || echo "[GPU 6] nglab2x_opt_rmsprop_4x_b2_09999_fixed FAILED"
  echo "[GPU 6] 6/6: nglab1x_opt_adamw_090999_s43_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_090999_s43_fixed"
  CUDA_VISIBLE_DEVICES="6" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_adamw_090999_s43_fixed \
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
    --table_optimizer adamw \
    --table_betas 0.9,0.999 > "$ROOT/data/runs/nglab1x_opt_adamw_090999_s43_fixed/train.log" 2>&1 || echo "[GPU 6] nglab1x_opt_adamw_090999_s43_fixed FAILED"
  echo "[GPU 6] queue done at $(date)"
) &

# GPU 7: 7 runs, est 160min
(
  echo "[GPU 7] 1/7: nglab1x_v10_v_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_v10_v_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_v10_v_fixed \
    --injection_position v \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab1x_v10_v_fixed/train.log" 2>&1 || echo "[GPU 7] nglab1x_v10_v_fixed FAILED"
  echo "[GPU 7] 2/7: nglab1x_v10_y_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_v10_y_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_v10_y_fixed \
    --injection_position y \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab1x_v10_y_fixed/train.log" 2>&1 || echo "[GPU 7] nglab1x_v10_y_fixed FAILED"
  echo "[GPU 7] 3/7: nglab0_25x_input_fv_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_25x_input_fv_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab0_25x_input_fv_fixed \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_25x_input_fv_fixed/train.log" 2>&1 || echo "[GPU 7] nglab0_25x_input_fv_fixed FAILED"
  echo "[GPU 7] 4/7: nglab1x_opt_rmsprop_4x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_rmsprop_4x_b2_099_fixed \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 7] nglab1x_opt_rmsprop_4x_b2_099_fixed FAILED"
  echo "[GPU 7] 5/7: nglab2x_opt_rmsprop_4x_b2_098_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_098_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab2x_opt_rmsprop_4x_b2_098_fixed \
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
    --table_betas 0.0,0.98 > "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_098_fixed/train.log" 2>&1 || echo "[GPU 7] nglab2x_opt_rmsprop_4x_b2_098_fixed FAILED"
  echo "[GPU 7] 6/7: nglab1x_opt_adamw_080950_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_080950_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_adamw_080950_fixed \
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
    --table_betas 0.8,0.95 > "$ROOT/data/runs/nglab1x_opt_adamw_080950_fixed/train.log" 2>&1 || echo "[GPU 7] nglab1x_opt_adamw_080950_fixed FAILED"
  echo "[GPU 7] 7/7: nglab1x_opt_rmsprop_2x_s44_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s44_fixed"
  CUDA_VISIBLE_DEVICES="7" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
    --run_id nglab1x_opt_rmsprop_2x_s44_fixed \
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
    --table_optimizer rmsprop \
    --table_lr_scale 2.0 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s44_fixed/train.log" 2>&1 || echo "[GPU 7] nglab1x_opt_rmsprop_2x_s44_fixed FAILED"
  echo "[GPU 7] queue done at $(date)"
) &

echo "All GPU queues launched. Waiting..."
wait
echo "All done at $(date)"
