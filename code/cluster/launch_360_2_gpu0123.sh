#!/usr/bin/env bash
# Supplementary launch for 360-2 GPU 0-3 (freed after handover runs)
set -uo pipefail

ROOT="/data/home/guoshaoyang/ngram-gap-lab"
mkdir -p "$ROOT/logs"

# GPU 0: 9 runs, est 192min
(
  echo "[GPU 0] 1/9: nglab0_5x_input_fv_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_5x_input_fv_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_5x_input_fv_fixed/train.log" 2>&1 || echo "[GPU 0] nglab0_5x_input_fv_fixed FAILED"
  echo "[GPU 0] 2/9: nglab1x_opt_rmsprop_2x_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_lr_scale 2.0 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_fixed/train.log" 2>&1 || echo "[GPU 0] nglab1x_opt_rmsprop_2x_fixed FAILED"
  echo "[GPU 0] 3/9: nglab1x_opt_rmsprop_4x_b2_098_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_b2_098_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.98 > "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_b2_098_fixed/train.log" 2>&1 || echo "[GPU 0] nglab1x_opt_rmsprop_4x_b2_098_fixed FAILED"
  echo "[GPU 0] 4/9: nglab2x_opt_rmsprop_2x_b2_098_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_098_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.98 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_098_fixed/train.log" 2>&1 || echo "[GPU 0] nglab2x_opt_rmsprop_2x_b2_098_fixed FAILED"
  echo "[GPU 0] 5/9: nglab2x_opt_rmsprop_4x_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_fixed/train.log" 2>&1 || echo "[GPU 0] nglab2x_opt_rmsprop_4x_fixed FAILED"
  echo "[GPU 0] 6/9: nglab025x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab025x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab025x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 0] nglab025x_b2_099_fixed FAILED"
  echo "[GPU 0] 7/9: nglab1x_opt_adamw_080950_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_080950_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.8,0.95 > "$ROOT/data/runs/nglab1x_opt_adamw_080950_fixed/train.log" 2>&1 || echo "[GPU 0] nglab1x_opt_adamw_080950_fixed FAILED"
  echo "[GPU 0] 8/9: nglab1x_opt_rmsprop_2x_s43_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s43_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_lr_scale 2.0 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s43_fixed/train.log" 2>&1 || echo "[GPU 0] nglab1x_opt_rmsprop_2x_s43_fixed FAILED"
  echo "[GPU 0] 9/9: nglab0_25x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_25x_e6_fixed"
  CUDA_VISIBLE_DEVICES="0" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_25x_e6_fixed/train.log" 2>&1 || echo "[GPU 0] nglab0_25x_e6_fixed FAILED"
  echo "[GPU 0] queue done at $(date)"
) &

# GPU 1: 9 runs, est 192min
(
  echo "[GPU 1] 1/9: nglab0_25x_input_fv_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_25x_input_fv_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_25x_input_fv_fixed/train.log" 2>&1 || echo "[GPU 1] nglab0_25x_input_fv_fixed FAILED"
  echo "[GPU 1] 2/9: nglab1x_opt_rmsprop_2x_b2_098_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_b2_098_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.98 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_b2_098_fixed/train.log" 2>&1 || echo "[GPU 1] nglab1x_opt_rmsprop_2x_b2_098_fixed FAILED"
  echo "[GPU 1] 3/9: nglab1x_opt_rmsprop_4x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 1] nglab1x_opt_rmsprop_4x_b2_099_fixed FAILED"
  echo "[GPU 1] 4/9: nglab2x_opt_rmsprop_2x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 1] nglab2x_opt_rmsprop_2x_b2_099_fixed FAILED"
  echo "[GPU 1] 5/9: nglab2x_opt_rmsprop_4x_b2_098_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_098_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.98 > "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_098_fixed/train.log" 2>&1 || echo "[GPU 1] nglab2x_opt_rmsprop_4x_b2_098_fixed FAILED"
  echo "[GPU 1] 6/9: nglab05x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab05x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab05x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 1] nglab05x_b2_099_fixed FAILED"
  echo "[GPU 1] 7/9: nglab1x_opt_adamw_090999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_090999_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.9,0.999 > "$ROOT/data/runs/nglab1x_opt_adamw_090999_fixed/train.log" 2>&1 || echo "[GPU 1] nglab1x_opt_adamw_090999_fixed FAILED"
  echo "[GPU 1] 8/9: nglab1x_opt_rmsprop_2x_s44_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s44_fixed"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_lr_scale 2.0 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_s44_fixed/train.log" 2>&1 || echo "[GPU 1] nglab1x_opt_rmsprop_2x_s44_fixed FAILED"
  echo "[GPU 1] 9/9: smoke_fixed_verify at $(date)"
  mkdir -p "$ROOT/data/runs/smoke_fixed_verify"
  CUDA_VISIBLE_DEVICES="1" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/smoke_fixed_verify/train.log" 2>&1 || echo "[GPU 1] smoke_fixed_verify FAILED"
  echo "[GPU 1] queue done at $(date)"
) &

# GPU 2: 8 runs, est 187min
(
  echo "[GPU 2] 1/8: nglab0_75x_input_fv_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_75x_input_fv_fixed"
  CUDA_VISIBLE_DEVICES="2" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_75x_input_fv_fixed/train.log" 2>&1 || echo "[GPU 2] nglab0_75x_input_fv_fixed FAILED"
  echo "[GPU 2] 2/8: nglab1x_opt_rmsprop_2x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="2" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab1x_opt_rmsprop_2x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 2] nglab1x_opt_rmsprop_2x_b2_099_fixed FAILED"
  echo "[GPU 2] 3/8: nglab2x_opt_rmsprop_1x_b2_09999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_1x_b2_09999_fixed"
  CUDA_VISIBLE_DEVICES="2" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.9999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_1x_b2_09999_fixed/train.log" 2>&1 || echo "[GPU 2] nglab2x_opt_rmsprop_1x_b2_09999_fixed FAILED"
  echo "[GPU 2] 4/8: nglab2x_opt_rmsprop_2x_b2_09999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_09999_fixed"
  CUDA_VISIBLE_DEVICES="2" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.9999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_09999_fixed/train.log" 2>&1 || echo "[GPU 2] nglab2x_opt_rmsprop_2x_b2_09999_fixed FAILED"
  echo "[GPU 2] 5/8: nglab2x_opt_rmsprop_4x_b2_099_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_099_fixed"
  CUDA_VISIBLE_DEVICES="2" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99 > "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_099_fixed/train.log" 2>&1 || echo "[GPU 2] nglab2x_opt_rmsprop_4x_b2_099_fixed FAILED"
  echo "[GPU 2] 6/8: nglab1x_v10_input_nofb_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_v10_input_nofb_fixed"
  CUDA_VISIBLE_DEVICES="2" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --sequence_len 2048 > "$ROOT/data/runs/nglab1x_v10_input_nofb_fixed/train.log" 2>&1 || echo "[GPU 2] nglab1x_v10_input_nofb_fixed FAILED"
  echo "[GPU 2] 7/8: nglab1x_opt_adamw_090999_s43_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_090999_s43_fixed"
  CUDA_VISIBLE_DEVICES="2" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.9,0.999 > "$ROOT/data/runs/nglab1x_opt_adamw_090999_s43_fixed/train.log" 2>&1 || echo "[GPU 2] nglab1x_opt_adamw_090999_s43_fixed FAILED"
  echo "[GPU 2] 8/8: nglab1x_opt_sgd_09_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_sgd_09_fixed"
  CUDA_VISIBLE_DEVICES="2" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.9,0.0 > "$ROOT/data/runs/nglab1x_opt_sgd_09_fixed/train.log" 2>&1 || echo "[GPU 2] nglab1x_opt_sgd_09_fixed FAILED"
  echo "[GPU 2] queue done at $(date)"
) &

# GPU 3: 9 runs, est 197min
(
  echo "[GPU 3] 1/9: nglab1_5x_input_fv_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1_5x_input_fv_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab1_5x_input_fv_fixed/train.log" 2>&1 || echo "[GPU 3] nglab1_5x_input_fv_fixed FAILED"
  echo "[GPU 3] 2/9: nglab1x_opt_rmsprop_4x_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_lr_scale 4.0 > "$ROOT/data/runs/nglab1x_opt_rmsprop_4x_fixed/train.log" 2>&1 || echo "[GPU 3] nglab1x_opt_rmsprop_4x_fixed FAILED"
  echo "[GPU 3] 3/9: nglab2x_opt_rmsprop_2x_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_fixed/train.log" 2>&1 || echo "[GPU 3] nglab2x_opt_rmsprop_2x_fixed FAILED"
  echo "[GPU 3] 4/9: nglab2x_opt_rmsprop_2x_b2_099999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099999_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099999_fixed/train.log" 2>&1 || echo "[GPU 3] nglab2x_opt_rmsprop_2x_b2_099999_fixed FAILED"
  echo "[GPU 3] 5/9: nglab2x_opt_rmsprop_4x_b2_09999_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_09999_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.9999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_4x_b2_09999_fixed/train.log" 2>&1 || echo "[GPU 3] nglab2x_opt_rmsprop_4x_b2_09999_fixed FAILED"
  echo "[GPU 3] 6/9: nglab1x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_e6_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab1x_e6_fixed/train.log" 2>&1 || echo "[GPU 3] nglab1x_e6_fixed FAILED"
  echo "[GPU 3] 7/9: nglab0_75x_e6_fixed at $(date)"
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
  echo "[GPU 3] 8/9: nglab1x_opt_adamw_090999_s44_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab1x_opt_adamw_090999_s44_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.9,0.999 > "$ROOT/data/runs/nglab1x_opt_adamw_090999_s44_fixed/train.log" 2>&1 || echo "[GPU 3] nglab1x_opt_adamw_090999_s44_fixed FAILED"
  echo "[GPU 3] 9/9: nglab0_5x_e6_fixed at $(date)"
  mkdir -p "$ROOT/data/runs/nglab0_5x_e6_fixed"
  CUDA_VISIBLE_DEVICES="3" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --freq_eval_batches 4 > "$ROOT/data/runs/nglab0_5x_e6_fixed/train.log" 2>&1 || echo "[GPU 3] nglab0_5x_e6_fixed FAILED"
  echo "[GPU 3] queue done at $(date)"
) &

echo "All GPU queues launched. Waiting..."
wait
echo "All done at $(date)"
