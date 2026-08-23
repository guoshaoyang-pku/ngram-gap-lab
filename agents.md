# ngram-gap-lab · Agent 指南

> 本仓库是本课题**唯一的开发仓库**。`/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap` 已弃用（见 §7）。
> 本文件只写**不变的规则与坐标**：工作原则、极简 setting、文档权威性、算力与存储、workspace。
> 会变的东西（实验进度、待办、作图清单）一律写进 `docs/experiment-log.md` 和 `docs/plans/`。

---

## 0. 工作原则

Agent 在本仓库工作时，按以下顺序遵守。冲突时，**编号小的优先**。

### P1 · 极简优先（最高原则）

任何实验的 setting 都必须是 §1 的**极简基线**，只允许改动**当前实验要检验的那一个变量**。

- 禁止引入：`current shell`、Muon、RoPE、RMSNorm、fourgram、gate 变体、MuonAdamW grouping。
- 若某个想法必须偏离极简基线，先在 `docs/experiment-log.md` 里写清楚「偏离了哪一项、为什么必须偏离」，再开跑。
- **`current shell` 是历史错误分支**（见 §6.2），任何引用它的结论都不能写进主线，必须在极简 setting 重跑。

### P2 · 改动前先读，读完再改

- 修改任何文件前，先完整读一遍该文件。
- 修改 `code/train.py` 前，先确认改动是否会影响已有 run 的口径；影响口径的改动必须新起 `run_id`，不能覆盖旧 run。
- 不确定某个结论出自哪个 setting 时，去 `docs/claims-ledger.md` 查该断言的 evidence 状态，不要凭印象引用。

### P3 · 一次只做一件事，做完就登记

- 一个实验 = 一个 `run_id` = 一个 `data/runs/<run_id>/` 目录 = `docs/experiment-log.md` 里的一行 + 一个 section。
- 五步生命周期：**登记（planned）→ 占 GPU → 开跑（running）→ 回填（done）→ 沉淀（图/脚本入库）**。
- 跑完 24h 内回填；未回填的 `planned` 行标 `stalled`。

### P4 · 口径一致性优先于结果好看

- gap 定义只有一个：`val_loss − train_loss`，同一 step、同一 batch 口径。
- `val` 必须是 **fixed validation batches**（固定的一组 batch），不能用移动窗口。历史教训见 `docs/method/loss-curve-sawtooth-audit.md`：移动窗 + 50 步间隔曾造出纯显示伪影的「锯齿」。
- `VAL_LOSS_INTERVAL_STEPS = 10`（v10 规范），freq-bin eval 同步每 10 步。
- `novel`（train hit count = 0）没有 train token loss，**不能定义 gap**，不进 gap 图。
- 多机跑同一批实验前，必须 `md5sum` 核对 `code/` 一致（见 §4.3）。

### P5 · 需要用户拍板的事，先问

以下动作必须先征得用户同意：
- 大改极简 setting，或新增一个模型架构变体。
- 跨集群 / 大文件传输（> 1 GB）。
- 删除 `data/runs/` 下已完成的 run，或删除历史存档。
- `git push`、改分支、改远端。

### P6 · 并行优先

非阻塞、可并行的调查/作图/分析任务，直接派给 subagent 并行做，不要串行等待。但**理解代码这一步不能外包**——先自己读懂再分派。

### P7 · 写文档的规矩

- 结论必须附「哪个 run_id、哪个 step、几个 seed」。没有 run 支撑的话写成「假设」而不是「结论」。
- 任何来自 current shell / Muon / RoPE 系的数字，引用时必须显式标注 `[DEPRECATED SETTING]`。
- 作图脚本进 `docs/plot_scripts/`，图进 `docs/figs/`，两者一起 commit。绘图规范见全局 skill `ngram-gap-plotting` 与 `docs/plot_scripts/README.md`。

---

## 1. 极简 setting（SSOT，唯一权威定义）

> 这是本仓库所有实验的出发点。**任何实验的 setting 表都必须以此为基准，只用粗体标出差异项。**
> 训练 1000 步即可看到 train/val forking；2000 步为标准延长口径。

### 1.1 模型

| 项 | 值 | 说明 |
|---|---|---|
| backbone | **vanilla nanoGPT**（Karpathy 风格） | 不是 current shell，不是 Muon 变体 |
| 层数 / 头数 / 宽度 | 8L · 6H · 768D | |
| vocab | 8192 | |
| sequence_len | 2048 | |
| position encoding | learned absolute | 不用 RoPE |
| normalization | LayerNorm | 不用 RMSNorm |
| embedding tying | tied | |
| attention window | LLLL（全 attention） | |

### 1.2 n-gram 模块

| 项 | 值 | 说明 |
|---|---|---|
| 注入点 | **`input` / wte** | over-encoding 风格：`x = wte(idx) + Σ ngram_ve`，不走 attention |
| n-gram 阶数 | bigram + trigram | unigram / fourgram **关闭** |
| **table size** | **1M** | `vocab_size × 64 = 524,288` 行 × 2 个 hash embedding = **1,048,576**。这是默认值，**没有改变过** |
| hash | 每个 n-gram 两组 decorrelated primes，各占一半 embedding dim | |

`v`（pre-attention value residual）与 `y`（post-attention residual）注入**只作为消融对照**存在，不是主线 setting。

### 1.3 优化器

| 项 | 值 | 说明 |
|---|---|---|
| n-gram table | **RMSProp，无动量** | `table_optimizer=rmsprop` |
| table betas | `(0.0, 0.999)` | β₁=0 即无动量；β₂=0.999 是原始设置 |
| **β₂ 备选** | `0.99` | 用户认为 0.99 更合理；已有反向扫描 `nglab*_b2_099` / `_b2_098`，见 `docs/experiment-log.md` §9d |
| backbone | AdamW，betas `(0.8, 0.95)`，weight_decay 0.1 | |
| lr | 0.004 | table_lr_scale = 1.0 |
| lr schedule | warmdown_ratio 0.65 | |

### 1.4 数据与训练

| 项 | 值 |
|---|---|
| data mode | `fixed`（固定顺序 epoch replay）·  data_seed 42 |
| train shards | 1（标准 1x）；shard 数即 epoch 长度剂量变量 |
| val shards | 与 train **完全不重叠**（历史坑：2.5x/3x/4x 首轮因重叠作废） |
| device batch | 72 |
| total batch | 147,456 tokens |
| seed | 42（多 seed 用 43 / 44） |
| steps | 1000（标准）/ 2000（延长） |
| val interval | **10 步**，fixed validation batches |
| freq-bin eval | 每 10 步 |

### 1.5 参考数值（seed 42，2000 步，标准 1x）

| run | 注入点 | final gap |
|---|---|---|
| `nglab1x_v10_input` | **input（主线）** | **1.931** |
| `nglab1x_v10_y` | y | 5.049 |
| `nglab1x_v10_v` | v | 5.041 |
| `nglab1x_v10_nogram` | 无 n-gram（negative control） | 0.231 |

完整登记见 `docs/experiment-log.md`。

---

## 2. 文档权威性（冲突解决规则）

历史上存在**三份**同名主文档，内容互相冲突。**从即日起按下表判定，不再有歧义**：

| 层级 | 文档 | 地位 |
|---|---|---|
| **权威主汇报** | `/Users/guoshaoyang/Desktop/workdir/guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide/index.html` | ✅ **唯一权威版**。9 章极简主线（setting / 注入点 / 曲线 / norm / 频率 / exact-frequency / sample285 / toy / 结论），已剔除 current shell |
| 权威背景页 | 同目录 `background.html` | ✅ 术语、伪代码、debug 过程、以及「current shell 为什么是废弃分支」的说明 |
| 存档 | 同目录 `index_archive_20260805.html` | 🗄️ 旧版完整 guide |
| **已废弃** | `OPHIS_gap/docs/ngram-gap-mechanism-guide.html`（chapter 0–19 全量版） | ⛔ **不再维护**。§2 / §7.9 / §9 / §12 / §15 / §16.7 / §16.11 全部建在 `baseline_current`（current shell）上 |
| 已废弃 | `OPHIS_gap/docs/ngram-gap-mechanism-guide-0728.html` / `-0730.html` | ⛔ 更早的存档 |
| 已废弃 | `OPHIS_gap/docs/ngram-gap-regime-bridge.html` | ⛔ 同事做的 current-shell 实验 |

**规则**：
1. 需要引用「主实验结论」时，一律引用 blog 的 `index.html`。
2. blog 的 `index.html` **不由脚本覆盖**（`docs/sync_to_blog.sh` 只同步独立报告与图）。要改主文档就手工改 blog 仓库那一份。
3. 本仓库内部的实验事实来源是 `docs/experiment-log.md`（登记簿）+ `docs/claims-ledger.md`（断言台账）。
4. OPHIS_gap 里的全量版 guide 只在「查历史怎么做过」时打开，**不能当作结论来源**。

---

## 3. 仓库结构

```
ngram-gap-lab/
├── agents.md                    # 本文件：工作原则 + 极简 setting + 坐标
├── README.md                    # 对外说明
├── code/
│   ├── train.py                 # vanilla nanoGPT + n-gram table + 3 注入点（<1000 行）
│   ├── ngram_freq.py            # per-frequency-bin loss 统计
│   ├── prepare_data.py          # 数据准备
│   ├── make_ngram_blocks.py     # controlled block 构造
│   ├── analyze_minimal.py       # 极简分析入口
│   ├── cluster/                 # 各集群 launcher + setup_env.sh
│   ├── toy/                     # 纯 numpy/torch 理论验证脚本（不依赖 backbone）
│   └── tools/                   # 语料熵、生成器等价性校验等通用工具
├── ngram5_freq_gap/             # order-5 / trigram controlled 实验独立训练包
├── docs/
│   ├── plan.md                  # 标准实验计划（setting 摘要，细节以 agents.md §1 为准）
│   ├── experiment-log.md        # ★ 唯一实验登记簿
│   ├── claims-ledger.md         # ★ 断言台账（C1–C9，标 SUPPORTED / PROXY_ONLY / UNRUN）
│   ├── theory/                  # 理论推导（unigram gap、幂律、Markov、真实长尾修正）
│   ├── literature/              # 文献精读 + related work + references.bib
│   ├── method/                  # 方法论与踩坑（sawtooth 审计、合成任务设计、排除台账模板）
│   ├── plans/                   # plan-1 机制研究总纲、plan-2 文献故事线
│   ├── archive/                 # 历史文档（含 current-shell 结论，仅供溯源）
│   ├── plot_scripts/            # 作图脚本（gen_all_figures.py 为主入口）
│   ├── figs/                    # 主线图
│   │   └── theory/              # 理论/toy 图
│   └── sync_to_blog.sh          # 同步报告与图到 blog（不覆盖 index.html）
└── data/                        # ★ gitignored
    ├── tokenized/               # token shards
    ├── freq_index*.npz          # 频率索引
    ├── runs/<run_id>/           # train_log.jsonl / table_norm.jsonl / freq_bin_loss.jsonl / summary.json / train.log
    └── toy_results/             # toy 实验结果
```

---

## 4. 算力与存储

### 4.1 集群总览

| 集群 | 连接 | GPU | 公网 | 环境 | 状态 |
|---|---|---|---|---|---|
| **ophis-gpu**（主） | 直连 SSH `guoshaoyang@223.167.85.180:50002`，别名 `ophis-gpu` / `ophis_gpu` / `fcloud-223` | 8×H200 (141 GB) | ✅ | `uv` + torch 2.9.1 | ✅ 可用 |
| **360-1** | QConnect VPN → `10.234.161.2:22`，`ssh 360-1` | 8×H200 (143.7 GB) | ❌ | 系统 python3.10 + torch 2.13.0+cu130 + numpy 2.2.6 | ✅ 可用 |
| **360-2** | QConnect VPN → `10.234.161.3:22`，`ssh 360-2` | 8×H200 (143.7 GB) | ❌ | 同上 | ✅ 可用 |

SSH 配置位于 `~/.ssh/config.d/`（主配置 `Include ~/.ssh/config.d/*.conf`），计算节点在 `20-compute.conf`。

360 系**无公网、且不能直连 ophis-gpu**（223.167.85.180 超时）。跨集群搬运走 **ophis-gpu → Mac → 360** 中转。

### 4.2 各集群的内容存储在哪个子文件夹

#### ophis-gpu

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

#### 360-1 / 360-2

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

### 4.3 跑实验前的强制检查

1. `nvidia-smi` 确认目标卡空闲，用 `CUDA_VISIBLE_DEVICES=<id>` 占卡；一个 Agent 只用自己登记的卡。
2. 同步代码到目标机，然后 `md5sum` 核对至少 `code/train.py`、`code/ngram_freq.py`、`code/cluster/*.sh`。
   - 权威源：本地 git 仓库已 commit 的版本，或 ophis-gpu `/data3/guoshaoyang/ngram-gap-lab`。
   - 教训（2026-08-06）：360 上曾残留旧 `train.py`（`f9388473`），与 ophis-gpu（`05bffab8`）的 val / freq-val 口径不同（旧版 freq-val 是移动窗口，新版固定 batch），导致同批实验口径不一致。
3. 改代码前先 commit，再同步到所有目标机。同一实验集跨机并行必须用同一份代码。

---

## 5. Workspace（本地坐标）

| 用途 | 路径 | 状态 |
|---|---|---|
| **主开发仓库** | `/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab` | ✅ 当前唯一开发地。GitHub: `git@github.com:guoshaoyang-pku/ngram-gap-lab.git` |
| **发布博客仓库** | `/Users/guoshaoyang/Desktop/workdir/guoshaoyang-pku.github.io` | ✅ 主文档发布地。主页面 `blogs/ngram-gap-mechanism-guide/index.html` |
| 旧仓库（弃用） | `/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap` | ⛔ **已弃用**，只读溯源，不再开发。见 §7 |
| 两因素模型参考 | `/Users/guoshaoyang/Documents/Codex/2026-08-21/xian-xi/outputs/ngram-repeat-gap-two-factor-model.html` | 📄 外部理论文档，待验证对象 |

**规则**：新文件、新代码、新实验一律落在 `ngram-gap-lab`。需要 OPHIS 里的东西就 `cp` 过来并在 `docs/archive/` 登记来源，不要跨仓库引用路径。

---

## 6. 废弃清单（不得进入主线）

### 6.1 判定标准

一个实验/结论只要命中下面任意一条，就是**非极简 setting**，不能进主线：
- backbone 是 `current shell`（`nanogpt_current_shell`）
- 用了 Muon 或 MuonAdamW grouping
- 用了 RoPE 或 RMSNorm
- n-gram table betas 的 β₁ ≠ 0（有动量），例如 `(0.5, 0.999)`
- `VAL_LOSS_INTERVAL_STEPS` > 10，或 val 用移动窗口

### 6.2 具体废弃项

| 内容 | 位置 | 原因 |
|---|---|---|
| `baseline_current` / `exp6_freqdecomp_current` | OPHIS `remote_training_runs/` | current shell，是同事的错误 setting；全库 11 个作图脚本 + 4 份结果文档都指向它 |
| `ngram-gap-regime-bridge.html` | OPHIS `docs/` | 同事的 current-shell 实验 |
| `ongoing_experiment/` 全目录 | OPHIS | 含 `exp4_hashreseed_current.log` 等 |
| `figB*` 系列图（~40 SVG） | OPHIS `docs/figs/`、`docs/interactive/` | Muon / nofork / rmsprop_freeze 专用 |
| `nanogpt_gap_causal/`、`nanogpt_gap_onset_source/`、`nanogpt_gap_vanilla_graft/`、`remote_gap_snapshot/` | OPHIS | **隐性污染**：不叫 current shell，但 parent 模型带 RoPE + RMSNorm |
| `ngram5_*_aligned_v2_final` | OPHIS `remote_training_runs/` | `run_contract.json` 显示 MuonAdamW + betas `(0.5, 0.999)` + val interval 50 |

### 6.3 「废弃结论，保留问题」

以下结论**方向可能是对的、问题很有价值，但数字全部建在 current shell 上，必须在极简 setting 重跑**：

| 问题 | 原结论（DEPRECATED SETTING） | 存档位置 |
|---|---|---|
| 表内容是不是 gap 的载体 | e2 边界 table 全量 reset → gap −89% | `docs/archive/p12-causal-results.md` |
| readout 通道是不是必要 | 屏蔽 readout → −89% | 同上 |
| table write vs backbone 各占多少 | 冻结 table −49%；冻结 backbone −54% | 同上 |
| 表大小是不是主导变量 | M=16 无碰撞点后 gap 饱和 → 不是参数量，是碰撞区低频涨落加权 | `docs/archive/table-size-sweep-results-20260811.md` |

重跑这些是当前最高价值的实验队列。

---

## 7. 与 OPHIS_gap 的关系（迁移状态）

`OPHIS_gap` 已弃用。2026-08-23 完成资产迁移，**已迁入本仓库**的内容：

| 迁入位置 | 来源 | 内容 |
|---|---|---|
| `docs/theory/` | `docs/theory_notes/` + `markov-unigram-exact-gap-20260811.md` | 5 篇纯理论推导，零 backbone 依赖 |
| `docs/literature/` | `docs/literature/` + 4 篇顶层长综述 | 9 个文件，含 arXiv 复核过的 `references.bib` 与可直接进论文的 related work |
| `docs/method/` | sawtooth 审计、合成任务设计、排除台账 | 方法论与踩坑 |
| `docs/plans/` | `plans/` | plan-1 机制总纲（§3.1a 是极简 setting 的原始定义）、plan-2 文献故事线 |
| `docs/claims-ledger.md` | `docs/claims-ledger-20260808.md` | C1–C9 断言台账 |
| `docs/archive/` | closure-status、p12-causal、table-size-sweep、injpos-log、manual 工作日志 | 历史溯源 |
| `code/toy/` | `toy/` | 9 个纯 numpy/torch 脚本，全库唯一零 current-shell 污染的代码 |
| `code/tools/` | `tools/` | 语料熵计算、生成器等价性校验 |
| `docs/figs/theory/` | `docs/figs/` 中的 markov / gap_vs_samples / synth 系列 | 理论图 |
| `data/toy_results/` | `toy/results/` | `markov_clean` + `rmsprop_v_sawtooth` |
| `data/injpos_*.json` | `remote_training_runs/` | injpos obs summary + 2000 步延长数据 |

**未迁移**（留在 OPHIS_gap，只读溯源）：
- 2.0 GB 的 `remote_training_runs/`（其中 `ngram5_trigram_full_trace/` 单目录 1.7 GB 是一次性调试 trace）
- 所有 current shell / Muon / RoPE 系源码目录与结果
- `ngram5_freq_gap/` 代码（本仓库已有同一份，仅 `lib.py` 存在无逻辑差异的格式差异）
- injpos v/y/input 的 `train.log`（本仓库 `nglab1x_v10_*` 已是更新波次）

⚠️ 已知冲突：OPHIS 的 `injpos_ablation_data.json`（292 KB）与本仓库 `data/injpos_ablation_data.json`（40 KB）不是同一波次。`claims-ledger.md` 记录：`input/train_log.jsonl` 旧波次 gap 为 1.9615，canonical `summary.json` / `train.log` 为 1.9308，**不能混用**。引用 injpos 数字时以本仓库 `summary.json` 为准。

---

## 8. 常用命令

```bash
# ophis-gpu：跑注入点消融（v/y/input/nogram，串行 4 个 run）
cd /data3/guoshaoyang/ngram-gap-lab && bash code/cluster/run_injpos.sh 0 2000

# 本地 CPU 冒烟
python code/train.py --run_id smoke --injection_position input --steps 10 \
  --data_dir /path/to/tokenized --device_batch_size 4 --total_batch_size 8192

# 构建频率索引
.venv/bin/python code/ngram_freq.py --data_dir <tokenized> \
  --train_shards 1 --vocab_size 8192 --out data/runs/<run_id>/freq_index.npz

# 重新生成全部主线图
python docs/plot_scripts/gen_all_figures.py

# 同步报告与图到 blog（不覆盖 index.html）
bash docs/sync_to_blog.sh
```

## 9. 相关 skill

- `ngram-gap-plotting`：本项目图表规范（injection-point loss/gap 曲线、loss-gap-table-RMS 对齐、频率 bin 双轴图、log-x / log-log、Plotly 图例控制、blog 非滚动嵌入）。新增或修改图前先读它和 `docs/plot_scripts/README.md`。
- `blog-deploy`：把 `docs/` 产物发布到 GitHub Pages 并验证 HTTP 200。
