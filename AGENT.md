# ngram-gap-lab · Agent 工作区指南

## 0. 工作原则

1. **科研 setting 从简、重说服力**：目标是简单、可信、可复现。
2. **遇到大困难先问用户**：不要擅自大改 setting。
3. **训练代码与文档解耦**：`code/` 只放可运行代码，`docs/` 只放计划/日志/图。
4. **多机跑实验前先同步代码**：开跑前用 `md5sum` 核对目标机与权威源的 `code/` 一致
   （详见 §1.2），避免多机版本漂移导致口径不一致。

## 1. 集群资源（ophis-gpu 主集群 + 360-1/360-2 内部集群）

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

### 1.1 360-1 / 360-2（内部集群，已就绪 2026-08-06 验证）

- **SSH**: `ssh 360-1` / `ssh 360-2`（需先启动 QConnect 客户端）
- **GPU**: 各 8×NVIDIA H200（143.7GB/卡），当前全部空闲
- **环境**: 系统 python3.10 + `torch 2.13.0+cu130`（cuda OK）+ `numpy 2.2.6`；ngram-gap-lab 无需再装包
- **仓库**: `/data/home/guoshaoyang/ngram-gap-lab`（1.4GB：`code/` + `data/tokenized/` 12 shard + `data/freq_index.npz`）
- **无公网**；与 ophis-gpu 不能直连（223.167.85.180 超时），搬运走 ophis-gpu → Mac → 360 中转
- **已验证**：30 步冒烟通过，step10 loss 与 ophis-gpu 一致（7.5294）；~2.1s/步 → 2000 步约 70–75 min/run

### 1.2 代码同步规则（跑实验前必读）

- **在任何机器上开跑前，先把当前代码同步过去，并 `md5sum` 核对版本一致**（至少核对
  `code/train.py`、`code/ngram_freq.py`、`code/cluster/*.sh`）。
- 权威源：本地 git 仓库（`/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab`，用已 commit 的
  版本）或 ophis-gpu `/data3/guoshaoyang/ngram-gap-lab`。
- 教训（2026-08-06）：360 上曾拷贝到旧版 `train.py`（`f9388473`），与 ophis-gpu 最新版
  （`05bffab8`）存在 val / freq-val 口径差异（旧版 freq-val 是移动窗口，新版固定 batch），
  导致同一批实验口径不一致；同步 + md5 核对后，两台 360 与 ophis-gpu 的 step10 loss 完全一致。
- 同一实验集跨机并行时，必须用同一份代码；改代码前先 commit，再同步到所有目标机。

## 2. 本 repo 路径

- 代码: `code/`（`train.py`, `ngram_freq.py`, `cluster/`）
- 文档: `docs/`（`plan.md`, `experiment-log.md`, `plot_scripts/`, `figs/`）
- 数据: `data/`（**gitignored**，含 `tokenized/` 和 `runs/`）

## 3. 多 Agent 并行实验协议（可持续协作）

> 核心：`docs/experiment-log.md` 是唯一实验登记簿。多 Agent 并行时，每个 Agent
> 只写自己的 section，不碰别人的 section；先登记、后开跑、跑完回填。

### 3.1 实验生命周期（每个实验五步）

1. **登记（planned）**：在 `docs/experiment-log.md` 顶部「实验登记总表」加一行，
   拿到唯一 `run_id`（如 `nglab_<topic>_<variant>`）；同时在正文新建一个 section，
   填 setting、假设、产物路径。
2. **占 GPU**：ophis-gpu 共 8 卡。启动前先 `nvidia-smi` 确认空闲卡，用
   `CUDA_VISIBLE_DEVICES=<id>` 或 `code/cluster/*.sh` 指定，避免与在跑实验撞卡。
3. **开跑**：输出写到 `data/runs/<run_id>/`（`train_log.jsonl`、`summary.json`、
   `train.log`、freq stats 等）。`data/` 整体 gitignored，不入库。
4. **回填（done）**：跑完后把 gap 数值、norm、关键观察写进自己 section，
   总表状态改 `done`，附产物路径与报告位置。
5. **沉淀**：新代码/脚本走 `code/`，新图走 `docs/figs/` + `docs/plot_scripts/`，
   随实验一起 commit；大文件（数据、checkpoint、tar.gz）一律放 `data/`，不进 git。

### 3.2 并行规则（避免冲突）

- **append-only**：只新增 section / 只编辑自己 run_id 对应的行，不重排、不改写他人 section。
- **run_id 唯一**：每个实验独占一个 `run_id`，禁止复用/覆盖在跑或已完成的 run 目录。
- **GPU 互斥**：一个 Agent 同时最多用自己登记的卡，以 `nvidia-smi` 实际占用为准。
- **登记簿同步**：实验结束后 24h 内回填；长期未回填的 planned 行会被标记 `stalled`。
- **commit 粒度**：一个实验一个 commit，格式 `type(scope): 描述`
  （如 `experiment(gap): ...` / `plot(gap): ...` / `code(freq): ...`）。

### 3.3 实验 section 登记模板

```markdown
## <run_id> — <一句话标题>（<日期>，<owner>）
### Setting
| 项 | 值 |  ← 关键配置；与 plan.md 标准 setting 的差异用**粗体**标出
### 结果
| run | gap@step | ... | ← 数值表
### 关键观察 / 结论
- ...
### 产物
- `data/runs/<run_id>/...`；图 `docs/figs/...`；报告 `docs/...`
```

## 4. 关键 setting

见 `docs/plan.md`。核心：vanilla nanoGPT + bigram/trigram + input 注入 + mixed optimizer + seed42 + 2000 步；validation 与 freq-bin eval 每 10 步。

## 5. 与 OPHIS 的关系

本 repo 从 OPHIS（`/Users/guoshaoyang/Desktop/workdir/OPHIS/`）提炼而来，
只保留 gap 复现必需的最小代码。完整理论体系、current shell 对照、文献调研
仍在 OPHIS_gap 中。

## 6. 作图与统计归档

- 作图源码统一放在 `docs/plot_scripts/`，目录索引和统计口径见
  `docs/plot_scripts/README.md`。
- 当前主入口是 `gen_all_figures.py`；它读取 `data/runs/<run_id>/`，生成
  `docs/figs/` 下的 Plotly HTML 和 SVG 备用图。
- 主图的逻辑分别是：global loss/gap、table norm 对齐、frequency-bin 时间曲线、
  命中频次 fraction + 末态 gap、以及 log-x/log-log frequency-to-gap。
- `freq_bin_loss.jsonl` 的 per-bin loss 来自 unreduced per-token loss 的聚合，
  同时保存 token count、fraction、mean loss 和 total contribution；不要把
  aggregate bucket 当成 exact hit-count 数据。
- `novel`（train hit count = 0）可用于 fraction 和 loss 展示，但没有 train
  token loss，不能定义标准 gap，因此不进入 gap/log 图。
- 已注册全局 plotting skill：`ngram-gap-plotting`。需要新增、修改或解释
  本 repo 图表时，先读取该 skill 和 `docs/plot_scripts/README.md`。
