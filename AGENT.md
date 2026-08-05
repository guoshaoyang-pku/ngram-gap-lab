# ngram-gap-lab · Agent 工作区指南

## 0. 工作原则

1. **科研 setting 从简、重说服力**：目标是简单、可信、可复现。
2. **遇到大困难先问用户**：不要擅自大改 setting。
3. **训练代码与文档解耦**：`code/` 只放可运行代码，`docs/` 只放计划/日志/图。

## 1. 集群资源（ophis-gpu，主集群）

- **SSH**: `ssh ophis-gpu`（别名 `ophis_gpu`, `fcloud-223`）
- **连接**: `guoshaoyang@223.167.85.180:50002`，密钥 `~/.ssh/id_rsa`
- **GPU**: 8×NVIDIA H200 (141 GB × 8)，OS Ubuntu 22.04.5，公网 ✅
- **存储**:
  - `/data3/guoshaoyang` — 7.0 TB NVMe（个人持久存储，配额 500G soft / 600G hard）
  - `/data4/guoshaoyang` — 7.0 TB NVMe（空闲，配额 500G/600G）
  - `/scratch/guoshaoyang` / `/tmp` — 438 GB（临时，**配额 15G soft / 20G hard**，勿写大文件）
  - **⚠️ 注意**：`/tmp` 和 `/` 在同一分区，guoshaoyang 根分区配额已超，**不要往 `/tmp` 写**
- **环境**: 系统 python3 + `uv`（`/data3/guoshaoyang/.local/bin/uv`），`torch==2.9.1`
  - 本 repo venv: `/data3/guoshaoyang/ngram-gap-lab/.venv/bin/python`
- **运行**: `cd /data3/guoshaoyang/ngram-gap-lab && .venv/bin/python code/train.py`
- **数据**: 复用 `/data3/guoshaoyang/ngram-gap-exp/data/`（tokenized shards）

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
