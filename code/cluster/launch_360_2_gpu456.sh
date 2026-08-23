#!/usr/bin/env bash
# Extra runs on 360-2 GPU 4,5,6 (freed after original queue)
set -uo pipefail

ROOT="/data/home/guoshaoyang/ngram-gap-lab"

echo "[GPU 4] nglab2x_opt_rmsprop_2x_b2_09999_fixed at $(date)"
mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_09999_fixed"
CUDA_VISIBLE_DEVICES="4" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.9999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_09999_fixed/train.log" 2>&1 || echo "[GPU 4] nglab2x_opt_rmsprop_2x_b2_09999_fixed FAILED"
echo "[GPU 4] nglab2x_opt_rmsprop_2x_b2_09999_fixed done at $(date)"
echo "[GPU 5] nglab2x_opt_rmsprop_2x_b2_099999_fixed at $(date)"
mkdir -p "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099999_fixed"
CUDA_VISIBLE_DEVICES="5" python3 -u /data/home/guoshaoyang/ngram-gap-lab/code/train.py \
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
    --table_betas 0.0,0.99999 > "$ROOT/data/runs/nglab2x_opt_rmsprop_2x_b2_099999_fixed/train.log" 2>&1 || echo "[GPU 5] nglab2x_opt_rmsprop_2x_b2_099999_fixed FAILED"
echo "[GPU 5] nglab2x_opt_rmsprop_2x_b2_099999_fixed done at $(date)"
echo "[GPU 6] nglab2x_opt_rmsprop_4x_b2_09999_fixed at $(date)"
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
echo "[GPU 6] nglab2x_opt_rmsprop_4x_b2_09999_fixed done at $(date)"
echo "All extra runs done at $(date)"
