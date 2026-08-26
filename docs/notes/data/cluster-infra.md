# 集群算力与存储坐标

> 本文件由 `agents.md` §4 与 §5 offload（2026-08-26）。规则与可用算力摘要仍保留在
> `agents.md` §4；这里是**完整目录细则**，修改存储路径/配额时同步更新本文件与 agents.md §4 摘要。

## 可用算力（总览，权威）

| 集群 | 连接 | GPU | 公网 | 环境 | 状态 |
|---|---|---|---|---|---|
| **ophis-gpu**（主） | 直连 SSH `guoshaoyang@223.167.85.180:50002`，别名 `ophis-gpu` / `ophis_gpu` / `fcloud-223` | 8×H200 (141 GB) | ✅ | `uv` + torch 2.9.1 | ✅ 可用 |
| **360-1** | QConnect VPN → `10.234.161.2:22`，`ssh 360-1` | 8×H200 (143.7 GB) | ❌ | 系统 python3.10 + torch 2.13.0+cu130 + numpy 2.2.6 | ✅ 可用 |
| **360-2** | QConnect VPN → `10.234.161.3:22`，`ssh 360-2` | 8×H200 (143.7 GB) | ❌ | 同上 | ✅ 可用 |

SSH 配置位于 `~/.ssh/config.d/`（主配置 `Include ~/.ssh/config.d/*.conf`），计算节点在 `20-compute.conf`。

360 系**无公网、且不能直连 ophis-gpu**（223.167.85.180 超时）。跨集群搬运走 **ophis-gpu → Mac → 360** 中转。

## 各集群的内容存储

### ophis-gpu

| 路径 | 容量/配额 | 存什么 |
|---|---|---|
| `/data3/guoshaoyang` | 7.0 TB NVMe，配额 500G soft / 600G hard | **个人持久存储（主力）** |
| `/data3/guoshaoyang/ngram-gap-lab/` | — | **本仓库的集群副本**；venv 在 `.venv/`，跑实验的工作区 |
| `/data3/guoshaoyang/ngram-gap-lab/data/tokenized/` | — | token shards |
| `/data3/guoshaoyang/ngram-gap-lab/data/runs/` | — | 本仓库所有 run 产物 |
| `/data3/guoshaoyang/ngram-gap-exp/` | — | **历史工作区**（OPHIS 时代）：旧 `train.py` / `lib.py`、`ngram5_data/`、`runs/ngram5/`、`toy/` |
| `/data3/guoshaoyang/ngram-gap-exp/ngram5_data/` | — | ngram5 / trigram controlled 数据集 |
| `/data3/guoshaoyang/ngram-gap-exp/runs/ngram5/` | — | ngram5 系列 run 结果 |
| `/data3/guoshaoyang/ophis_gap_local_backup/` | — | 本地镜像备份 |
| `/data2/guoshaoyang` | 7.0 TB，81% used | 共享实验目录 |
| `/data4/guoshaoyang` | 7.0 TB，配额 500G/600G | 空闲，可作扩展 |
| `/scratch/guoshaoyang`、`/tmp` | 438 GB 但**配额仅 15G soft / 20G hard** | ⚠️ **禁止写大文件**：`/tmp` 与 `/` 同分区，根分区配额已超 |

### 360-1 / 360-2

| 路径 | 存什么 |
|---|---|
| `/data/home/guoshaoyang/ngram-gap-lab/` | **本仓库副本**（~1.4 GB）：`code/` + `data/tokenized/`（12 shard）+ `data/freq_index*.npz` |
| `/data/home/guoshaoyang/ngram-gap-lab/data/runs/` | run 产物 |
| `/data/home/guoshaoyang/ngram-gap-exp/toy/` | **toy 实验专用工作区**（360-2，~425 MB）：`ws/`（harness）、`data/`、`cache/`、各 `toy*_launch.sh` |
| `$ROOT/.inductor_cache` | torch inductor 编译缓存 |
| `/tmp` | ⚠️ **360-2 的 `/tmp` 只有 974 MB**，曾被 inductor 缓存写满。**禁止写 `/tmp`**，编译缓存一律走 `$ROOT/.inductor_cache` |

**分工约定**：
- 主线 nanoGPT 实验：ophis-gpu 或 360-1/360-2 均可（代码同步后口径一致）。
- toy / 合成数据实验：默认跑 **360-2**（集中算力）；ophis-gpu 不再启动新 toy run，历史 toy 数据保留在 `/data3/guoshaoyang/ngram-gap-exp/toy`。

## 跑实验前的强制检查

1. `nvidia-smi` 确认目标卡空闲，用 `CUDA_VISIBLE_DEVICES=<id>` 占卡；一个 Agent 只用自己登记的卡。
2. 同步代码到目标机，然后 `md5sum` 核对至少 `code/train.py`、`code/ngram_freq.py`、`code/cluster/*.sh`。
   - 权威源：本地 git 仓库已 commit 的版本，或 ophis-gpu `/data3/guoshaoyang/ngram-gap-lab`。
   - 教训（2026-08-06）：360 上曾残留旧 `train.py`（`f9388473`），与 ophis-gpu（`05bffab8`）的 val / freq-val 口径不同（旧版 freq-val 是移动窗口，新版固定 batch），导致同批实验口径不一致。
3. 改代码前先 commit，再同步到所有目标机。同一实验集跨机并行必须用同一份代码。

## Workspace（本地坐标）

| 用途 | 路径 | 状态 |
|---|---|---|
| **主开发仓库** | `/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab` | ✅ 当前唯一开发地。GitHub: `git@github.com:guoshaoyang-pku/ngram-gap-lab.git` |
| **发布博客仓库** | `/Users/guoshaoyang/Desktop/workdir/guoshaoyang-pku.github.io` | ✅ 主文档发布地。主页面 `blogs/ngram-gap-mechanism-guide/index.html` |
| 旧仓库（弃用） | `/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap` | ⛔ **已弃用**，只读溯源，不再开发。见 `deprecated-list.md` |
| 两因素模型参考 | `/Users/guoshaoyang/Documents/Codex/2026-08-21/xian-xi/outputs/ngram-repeat-gap-two-factor-model.html` | 📄 外部理论文档，待验证对象 |

**规则**：新文件、新代码、新实验一律落在 `ngram-gap-lab`。需要 OPHIS 里的东西就 `cp` 过来并在 `docs/_archive/docs/` 登记来源，不要跨仓库引用路径。
