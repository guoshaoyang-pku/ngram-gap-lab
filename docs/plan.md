# ngram-gap-lab · 标准实验计划

> **setting 的唯一权威定义在 `agents.md` §1（SSOT）。本文件只写现象定义、消融变量与实验登记模板。**
> 若本文件与 `agents.md` §1 冲突，以 `agents.md` 为准。

## 1. 现象定义

含可训练 n-gram value table 的 **vanilla nanoGPT**，小训练集 + 固定顺序多 epoch replay 时：

- train loss 阶梯下降（epoch 边界 cliff），val loss 翘起 → train/val gap。
- 训练 **1000 步**即可看到 forking；2000 步为标准延长口径。
- 关键：gap 只依赖 n-gram memory，**不需要 current shell / Muon / RoPE / RMSNorm**。

## 2. 三种注入点（核心消融变量）

| 注入点 | 技术方案 | 走 attention？ | 信号强度 |
|---|---|---|---|
| `v` | `V = V + gate·ngram_ve`（attention 之前）| ✅ 是（被 softmax 混合）| 弱（norm 只有 V 的 6.5%）|
| `y` | `y = attn(Q,K,V) + gate·ngram_ve`（attention 之后）| ❌ 否 | 中（每层注入）|
| **`input`** | `x = wte(idx) + Σ ngram_ve`（over-encoding，入口一次）| ❌ 否 | 中（一次注入）|

**`input` 是主线 setting**，其余两个只作对照。
- `input` = over-encoding 风格（Engram / SCONE / Over-Tokenized 主流做法）。
- `y` = ResFormer y-variant（gap 最大但非主流）。
- `v` = ResFormer value residual（信号被 V 淹没）。

**结论**：只要 n-gram 信号不走 attention 混合、能有效到达输出，就能产生 gap。

## 3. 标准设置（baseline_input）摘要

完整表见 `agents.md` §1。核心六项：

| 项 | 值 |
|---|---|
| backbone | vanilla nanoGPT（8L · 6H · 768D，vocab 8192，learned abs + LayerNorm + tied）|
| n-gram 模块 | bigram + trigram，`input` / wte 注入 |
| **table size** | **1M**（`vocab_size × 64 = 524,288` 行 × 2 hash embedding），默认值未改动 |
| table 优化器 | **RMSProp 无动量**，betas `(0.0, 0.99)`；历史对照使用 β₂ = `0.999` |
| backbone 优化器 | AdamW `(0.8, 0.95)`，lr 0.004，wd 0.1 |
| 数据 / 评测 | fixed 顺序 epoch replay，seed 42，1000 或 2000 步，**val 每 10 步 + fixed batches** |

## 4. 当前实验队列

优先级从高到低：

1. **在极简 setting 重跑 current-shell 时代的因果实验**（见 `agents.md` §6.3）：
   e2 边界 table reset / 冻结 table / 冻结 backbone / 屏蔽 readout / table size 扫描。
2. **固定 train 采样集合的 loss 曲线**：与 val 同频率（每 10 步）记录一个**不变的** train 子集的 loss，
   用于直接测量记忆进度 ρ 而非在线 batch 平均。同时记录该固定集合的 per-frequency-bin loss。
3. **缩短单 epoch 数据量**加速实验并放大 gap（shard 扫描已做到 0.25x–8x，见 `experiment-log.md` §10）。
   ⚠️ 前置条件：先跑一个**长时程 no-ngram 对照**，确认 backbone 单独长训是否也会产生 gap，
   否则无法把 gap 归因给 n-gram 表。
4. β₂ = 0.99 vs 0.999 的主线对照（已有 4x lr 下的反向扫描，见 §9d，需补主线 lr 下的版本）。

## 5. 实验登记模板

```markdown
## <run_id> — <一句话标题>（<日期>，<owner>）

### Setting
| 项 | 值 |   ← 只列与 agents.md §1 的**差异项**，用粗体标出

### 结果
| run | gap@step | train | val | ... |

### 关键观察 / 结论
- ...

### 产物
- `data/runs_fixed/<run_id>_fixed/`；图 `docs/figs/<line>/`；报告 `docs/report/`
```

## 6. 已完成

1. ✅ 干净 repo（`train.py` < 1000 行 + `ngram_freq.py`）
2. ✅ v/y/input/nogram 四注入点消融（`experiment-log.md` §2 / §8）
3. ✅ 频率索引 + per-bin loss 统计（§3）
4. ✅ 注入点对比图 + table norm 图 + 频率 bin 分解图（`docs/figs/`）
5. ✅ 公开博客重写（blog `ngram-gap-mechanism-guide/index.html`，9 章极简主线）
6. ✅ table 优化器消融含 β₂ 反向扫描（§9 / §9a–9d）
7. ✅ shard 大小扫描 12 点（§10）
8. ✅ predecessor codebase 资产迁移（理论 / 文献 / 方法论 / toy，见 `agents.md` §7）
