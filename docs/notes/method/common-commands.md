# 常用命令

> 本文件由 `agents.md` §8 offload（2026-08-26）。命令清单会变，放这里维护。

```bash
# [HISTORICAL LEGACY TABLE] ophis-gpu：跑旧注入点消融（v/y/input/nogram，串行 4 个 run）
cd /data3/guoshaoyang/ngram-gap-lab && bash code/cluster/run_injpos.sh 0 2000

# 新主线：clean 单表 input 基线（bigram / trigram 均为 R=2^20，固定 LR）
cd /data3/guoshaoyang/ngram-gap-lab && \
  bash code/cluster/run_baseline.sh 0 <run_id>

# 本地 CPU 冒烟（非主线 setting，只检验代码路径；不可作为实验结果）
python code/train.py --run_id smoke --injection_position input --steps 10 \
  --data_dir /path/to/tokenized --device_batch_size 4 --total_batch_size 8192 \
  --n_layer 1 --n_head 1 --n_embd 16 --sequence_len 32 --dtype fp32 \
  --bigram_clean_table 64 --trigram_clean_table 64

# 构建频率索引
.venv/bin/python code/ngram_freq.py --data_dir <tokenized> \
  --train_shards 1 --vocab_size 8192 --out data/runs_fixed/<run_id>_fixed/freq_index.npz

# 重新生成全部主线图
python docs/plot_scripts/gen_all_figures.py

# 同步报告与图到 blog（不覆盖 index.html）
bash docs/sync_to_blog.sh
```
