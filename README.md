# ngram-gap-lab

vanilla nanoGPT 上 **n-gram 值记忆导致的 replay 特化 train/val gap** 的最小干净复现仓库。

> **Agent / 协作者从这里开始：先读 [`agents.md`](agents.md)。**
> setting / 实验登记 / 绘图工作流在 [`.agents/skills/`](.agents/skills/) 随仓库版本化。
> 本 README 是 agents.md 的中文速览；**一切以 agents.md 为准**（唯一权威）。

---

## 这是什么

一个独立、干净、自包含的仓库，只隔离一个现象：给 vanilla nanoGPT 加上可训练的
n-gram 值表（bigram + trigram），在小数据集上做固定顺序多 epoch replay 训练，
train/val gap 就会出现。**1000 步即可看到 fork。**

仓库用**尽可能简单的 setting**复现这个 gap——没有 current shell、没有 Muon、
没有 RoPE、没有 RMSNorm、表优化器无动量。

## 极简 setting（SSOT：`agents.md` §1）

| 坐标 | 主线默认 |
|---|---|
| 注入 | `input` / wte（over-encoding：`x = wte(idx) + Σ ngram_ve`，不走 attention） |
| backbone | vanilla nanoGPT · 8L · 6H · 768D · vocab 8192 · seq 2048 · learned abs PE · LayerNorm · tied embedding |
| n-gram | bigram + trigram **clean 单表**（`nn.Embedding(R, n_embd)`），unigram/fourgram 关 |
| table size | 非 table-size 实验固定 `R_bigram = R_trigram = 2^20 = 1,048,576`；旧 `table_mult` 一律不用 |
| table optimizer | **RMSProp 无动量**，betas `(0.0, 0.99)` |
| table LR | `--table_lr_scale 2.0`，实际 `0.0012` |
| backbone optimizer | AdamW `(0.8, 0.95)`，lr `0.0006`，wd 0.1 |
| LR schedule | `--lr_schedule warmup_constant --warmup_steps 100`：step 1–100 从 0.25× 线性升至 1×，之后固定；禁 warmdown |
| 数据 | fixed-order epoch replay · data_seed 42 · train shards 1（标准 1x）· val shards 与 train 完全不重叠 |
| 预算 | seed 42 · **1000 步（标准）/ 2000 步（延长）** · bf16 autocast · 不 `torch.compile` |
| 评估 | **online 当前 batch 的 train loss**（更新前）+ fixed val batches；`VAL_LOSS_INTERVAL_STEPS=10`，freq-bin 同步 |
| 口径 | `gap = val_loss − train_loss`（同 logged step 的 fixed-val − 当前 batch online train） |

`v`（pre-attention value residual）与 `y`（post-attention residual）注入**只作为消融对照**，不是主线。

## 核心发现

gap 取决于 **n-gram 信号能否有效到达输出**而不被 attention 混合淹没。

| 注入 | 风格 | gap @2000 | run |
|---|---|---|---|
| **`input`** | over-encoding（Engram / SCONE 主流） | **1.867** | `nglab1x_v10_input_fixed` |
| `y` | post-attention residual（ResFormer 变体） | 5.804 | `nglab1x_v10_y_fixed` |
| `v` | value residual（pre-attention，被 V 淹没） | 5.450 | `nglab1x_v10_v_fixed` |
| — | 无 n-gram（negative control） | 0.245 | `nglab1x_v10_nogram_fixed` |

## 仓库结构

```
ngram-gap-lab/
├── agents.md              # ★ 工作原则 + 极简 setting SSOT + 文档权威性 + 可用算力
├── .agents/skills/        # ★ 随仓库版本化的 Codex 工作流（setting / 登记 / 绘图）
├── code/                  # 主线 nanoGPT（train.py / ngram_freq.py / cluster/ / tools/）
├── tasks/                 # L1–L6 独立敏捷验证任务（自包含，单机可跑）
├── ngram5_freq_gap/       # 受控数据干预运行时
├── docs/
│   ├── experiment-lines.md  # ★★ 实验线全景 + 权威数据源 + 待办（入口）
│   ├── experiment-log.md    # ★ 实验登记簿
│   ├── claims-ledger.md     # ★ 断言台账 C1–C9
│   ├── report/              # 对外报告（index.html / background.html / versions/）
│   ├── notes/               # theory / literature / method / data
│   ├── plans/               # plan-1..5
│   ├── figs/ + plot_scripts/ + appendices/ + _archive/docs/
│   └── sync_to_blog.sh
└── data/                  # gitignored：tokenized / freq_index*.npz / runs_fixed/
```

**入口顺序**：`agents.md` → `docs/experiment-lines.md` → `docs/experiment-log.md`。

## 快速开始

### GPU 集群

```bash
cd /path/to/ngram-gap-lab
bash code/cluster/setup_env.sh                 # 环境（复用已有 torch）
bash code/cluster/run_baseline.sh 0 <run_id>   # 新主线入口
```

`run_baseline.sh` 是当前主线入口：input 注入、lr 0.0006、RMSProp `(0.0, 0.99)`、
table LR ×2、`warmup_constant(100)`、bf16、1000 步、online train loss、fixed val。
非 table-size 实验锁定双表 `2^20`；table-size 线必须在专属 launcher 里显式写明替代 R。
跑前强制检查：`nvidia-smi` 确认卡空闲 + 同步代码后 `md5sum` 核对（细则见 `agents.md` §4 / `docs/notes/data/cluster-infra.md`）。

### 本地 CPU 冒烟（只验代码路径，不是实验证据）

```bash
pip install torch numpy
python code/train.py --run_id smoke --injection_position input --steps 10 \
  --data_dir /path/to/tokenized --device_batch_size 4 --total_batch_size 8192 \
  --n_layer 1 --n_head 1 --n_embd 16 --sequence_len 32 --dtype fp32 \
  --bigram_clean_table 64 --trigram_clean_table 64
```

### 频率索引

```bash
.venv/bin/python code/ngram_freq.py --data_dir <tokenized> \
  --train_shards 1 --vocab_size 8192 --out data/runs_fixed/<run_id>_fixed/freq_index.npz
```

## 公开报告

权威汇报见 [ngram-gap-mechanism-guide](https://guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide/)（9 章极简主线）。
旧 chapter 0–19 全量版**不再维护**——见 `agents.md` §2。

---

## Handoff prompt（交给协作者 / agent）

> 复制下面整段，发给接手本仓库的协作者或 agent。

```
你接手的是 ngram-gap-lab，一个研究「n-gram 值记忆导致 replay 特化 train/val gap」的仓库。
请严格按以下规范工作：

【先读】仓库根目录 agents.md（唯一权威，工作原则 + 极简 setting SSOT），
然后是 docs/experiment-lines.md（实验线全景 + 权威数据源）和 docs/experiment-log.md（实验登记簿）。
本 README 只是速览，与 agents.md 冲突时以 agents.md 为准。

【setting 铁律】任何实验只用极简基线，只改当前要检验的一个变量：
注入=input/wte；clean 单表 bigram+trigram 各 R=2^20（table-size 实验除外）；
表优化器 RMSProp 无动量 betas(0.0,0.99)、table_lr_scale=2.0；
backbone AdamW(0.8,0.95) wd0.1、lr=0.0006、schedule=warmup_constant(100)（禁 warmdown）；
seed 42、1000 步（标准）/2000（延长）、bf16、不 torch.compile；
口径=online 当前 batch train loss（更新前）+ fixed val（同 step 更新后），gap=val−train，
val 与 freq-bin 每 10 步。禁止 current shell / Muon / RoPE / RMSNorm / fourgram / gate / 表动量。

【权威数据】只有 data/runs_fixed/ 里带 _fixed 后缀的 run 是合法的；data/runs/ 因 freq-bin bug 作废。
val 必须 fixed batches。novel（hit=0）无 train loss，不定义 gap。

【登记纪律】一个实验 = 一个 run_id = 一个 data/runs_fixed/<run_id>_fixed/ 目录 =
experiment-log.md 一行 + 一个 section。生命周期：planned → 占 GPU → running → done(24h 内回填) → 沉淀。
结论必须附 run_id/step/seed；没跑过的写成「假设」；current-shell 系数字标 [DEPRECATED SETTING]。

【算力】三机均为 8×H200：ophis-gpu（SSH 223.167.85.180:50002）、360-1、360-2（VPN，无公网）。
跑前 nvidia-smi 确认卡空闲、CUDA_VISIBLE_DEVICES 占卡；同步代码后 md5sum 核对 code/。
改代码先 commit 再同步，同批实验跨机用同一份代码。细则见 docs/notes/data/cluster-infra.md。

【先问再动】大改 setting / 新架构变体、跨集群大文件(>1GB)、删已完成 run、git push / 改分支——
先征得用户同意。改 code/train.py 影响口径须新起 run_id。

【工具】skill：ngram-gap-settings（setting 审计）、ngram-gap-experiment-registration（登记/回填/handover）、
ngram-gap-plotting（绘图规范，改图前先读）、blog-deploy（发布到 GitHub Pages）。
```

## License

Apache-2.0（继承自 nanoGPT upstream）。
