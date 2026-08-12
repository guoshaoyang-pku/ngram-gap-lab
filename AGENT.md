# ngram-gap-lab · Agent 工作区指南

## 0. 工作原则

1. **科研 setting 从简、重说服力**：目标是简单、可信、可复现。
2. **遇到大困难先问用户**：不要擅自大改 setting。
3. **训练代码与文档解耦**：`code/` 只放可运行代码，`docs/` 只放计划/日志/图。

## 1. 集群资源（h200-1，主集群）

- **SSH**: `ssh h200-1`
- **连接**: `yushanbin@10.234.161.2:22`，密钥 `C:/Users/vbird/.ssh/id_rsa`
- **GPU**: 8×NVIDIA H200 (141 GB × 8)，OS Ubuntu 22.04.5，公网 ✅
- **存储**:
  - `/data/home/yushanbin/ngram-gap-shaoyang-2` — 当前最小仓库运行目录
  - `/data/home/yushanbin/ngram-gap-shaoyang-2/data/runs` — 当前实验结果目录
- **环境**: `/usr/bin/python3`（系统 Torch CUDA 环境）
- **运行**: `cd /data/home/yushanbin/ngram-gap-shaoyang-2 && /usr/bin/python3 -u code/train.py`
- **数据**: 放在运行目录下的 `data/tokenized/` 和 `data/freq_index.npz`

## 2. 本 repo 路径

- 代码: `code/`（`train.py`, `ngram_freq.py`, `cluster/`）
- 文档: `docs/`（`plan.md`, `experiment-log.md`, `plot_scripts/`, `figs/`）
- 数据: `data/`（**gitignored**，含 `tokenized/` 和 `runs/`）

## 3. 关键 setting

见 `docs/plan.md`。核心：vanilla nanoGPT + bigram/trigram + input 注入 + mixed optimizer + seed42 + 1000 步。

## 4. 与 OPHIS 的关系

本 repo 从 OPHIS（`/Users/guoshaoyang/Desktop/workdir/OPHIS/`）提炼而来，
只保留 gap 复现必需的最小代码。完整理论体系、current shell 对照、文献调研
仍在 OPHIS_gap 中。
