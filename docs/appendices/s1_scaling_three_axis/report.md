# 附录 · 三轴 scaling 验证（epoch 长度 / exact frequency / table size）

> **实验线**：T-scaling（极简 setting 下验证三条 scaling 曲线）
> **状态**：🟢 基础 QC 完成（7 锚点）+ `bb_safety` 完成；正式 full grid seed 42
> 已启动（epoch / table / frequency 三轴，运行中）
> **数据源**：`data/runs_scaling/basic_*`（基础 QC）、`data/runs_scaling/bb_safety*`
> （长训安全）及 `data/runs_scaling/<run_id>_fixed/`（正式网格）
> **代码**：`tasks/s1_scaling_three_axis/`（launchers + analysis + 单测）
> **计划**：`docs/plans/plan-5-s1-three-axis-handoff.md`

---

## 0. 这个专题回答什么问题

在**唯一极简 setting**（vanilla nanoGPT 8L·6H·768D + input n-gram 注入，
自然语料）下，验证三条 scaling 曲线：

1. **Epoch 长度** `L`：训练集一个 epoch 的长度如何影响 gap（严格区分
   fixed-step 与 fixed-epoch 两种对齐）。
2. **Exact context frequency** `f`：`G(E,f)` 是否服从两因素模型
   `G(f) = A·f^(−β)·[1 − exp(−c·f^γ)]`（observational consistency）。
3. **Table size**：从默认 1M 逻辑地址**只向下**缩放，判断 gap 由参数量持续
   决定，还是由 hash collision / occupancy 决定后饱和。

**claim 边界**：本专题只用自然语料，frequency 分支是模型一致性检验，
**不能**把 `f` 宣称为严格因果变量（DoD §9）。

---

## 1. 冻结 setting

| 项 | 值 |
|---|---|
| backbone | vanilla nanoGPT 8L/6H/768D，learned abs，LayerNorm，tied |
| n-gram 注入 | `input`（wte over-encoding） |
| 模块臂 | bigram-only / trigram-only / both / no-ngram |
| table 优化器 | RMSProp 无动量，**β₂=0.99**（`--table_betas 0.0,0.99`） |
| S1 table LR | `table_lr_scale=2.0`（当前标准；实际 table lr = 0.008） |
| backbone 优化器 | AdamW (0.8, 0.95)，lr 0.004 |
| 数据 | 自然语料 shard 1 嵌套前缀，train/val 零重叠 |
| 测量 | fixed train probe（4 batches，SHA256）+ exact-frequency + occupancy |
| compute | bf16 autocast + `torch.compile` |
| validation / probe | 基础 QC 每 25 步；正式网格每 10 步；均为 fixed validation + fixed train probe |
| 结果目录 | 基础 QC：`data/runs_scaling/basic_*`；正式网格：`data/runs_scaling/<run_id>_fixed/` |

## 2. 数据分层与当前状态

结果分成两层：

1. **历史 pilot**：`pilot_*`、`tbl_pilot_*` 和旧 safety 目录，只用于方法
   QC 与溯源，不用于当前标准的 scaling 定律。
2. **当前标准基础 QC**：7 个 `basic_*` 锚点，使用 β₂=0.99、
   `table_lr_scale=2.0`、bf16 + compile、固定 train probe，但采用
   25 步打点以降低基础试验开销。它们用于回答“规律是否出现”，不能
   替代 10 步 cadence 的正式 full grid。

正式网格仍要求 `_fixed` 目录、完整 run contract、10 步 validation/probe
cadence，并在多 seed 下复现。

本地已有的 7 个 `basic_*` 结果是在本次降频修正之前生成的：它们的
fixed-probe 与 exact-frequency 记录会同频出现，因此 L1 中日志行数较多。
代码和 launcher 现已拆开两种 cadence：fixed probe 每 25 步，宽 bucket
与 exact-frequency 每 100 步，并保留 epoch 边界与最终步；下一批重跑后，
报告中的运行时比较应以新 contract 为准，不能把旧日志行数当作训练开销。

### 2.0 数据量和完整性

当前本地 `data/runs_scaling/` 有 **15 个实验目录**：

- **8 个历史 pilot/safety run**：epoch 轴 4 个，table 轴 3 个，另有 1 个
  backbone safety；
- **7 个当前标准 basic QC run**：epoch 轴 4 个（L1/L4 × both/no-ngram），
  table 轴 3 个（1M/16K bigram、1M trigram）；
- 暂无 full-grid、fixed-epoch、完整 table-size、多 seed 或 5000 步 safety
  结果，因此当前不能声称三条 scaling 已被完整验证。

7 个 basic run 都有：

- 在线 train/val 记录和 fixed-probe 记录；
- exact-frequency 最后快照覆盖 train/val 共同的 63 个 bigram exact-f 值、
  57 个 trigram exact-f 值（按纳入标准）；
- 在 `>=1024` tokens 且 `>=32` contexts 的 train/val 共同纳入标准下，
  可用于基础曲线比较；table 锚点另已生成逐 branch/layer/hash 的 occupancy 文件。

7 个 basic 结果日志约 **500 MB**；固定 train probe hash 均为
`38d1254a827759d6`。summary 将 bf16/compile 记录在顶层
`compute_dtype` / `torch_compile`，而不是 `config` 内；后续 contract
校验应读取实际字段，不能仅凭目录名判断。

### 2.1 Epoch length（当前标准基础 QC，fixed-step 1000 步）

| run | epoch batches | 重播轮数 | final fixed gap |
|---|---|---|---|
| `basic_L1_both_fs` | 42 | ~24 | **+2.385** |
| `basic_L4_both_fs` | 336 | ~3 | **+2.341** |
| `basic_L1_nogram_fs` | 42 | ~24 | +0.085 |
| `basic_L4_nogram_fs` | 336 | ~3 | **−0.008** |

**基础 QC 观察 E1（尚非 scaling 定律）**：在新标准、1000-step 截面，
both 臂产生约 +2.34–2.39 的 gap，而两个 no-ngram 对照约为 0。这说明
forking 在新标准下清晰出现；但 L1 与 L4 的终点几乎相同，不能再引用旧
pilot 的“L1 明显大于 L4”作为当前标准结论。必须等待 L2/L3、fixed-epoch
对齐和多 seed。

no-ngram 的两个 1000-step 对照暂未显示明显 backbone gap；但独立的
`bb_safety_L1_nogram_5000` 仍需完成后，才能界定长训 backbone 的贡献。

（fixed-epoch 对齐的 full grid 才能分离「重播次数」与「epoch 长度本身」的效应。）

### 2.2 Table size（当前标准基础 QC，L4，fixed-step 1000 步）

| run | logical 2R | module | final fixed gap |
|---|---:|---|---:|
| `basic_tbl_1M_bigram` | 1,048,576 | bigram-only | **+0.801** |
| `basic_tbl_16K_bigram` | 16,384 | bigram-only | **+0.016** |
| `basic_tbl_1M_trigram` | 1,048,576 | trigram-only | **+0.815** |

**基础 QC 观察 T1（尚非完整 table scaling）**：在同一 L4、同一 seed、同一
1000-step 预算下，1M→16K bigram gap 从 +0.801 降至 +0.016，方向与
collision pooling 假设一致；1M trigram 也产生约 +0.815 的 gap。目前仅有
两个 table size、bigram-only 的直接对比。当前 occupancy（L4 的 24,192
train chunks）显示：bigram 的 `K=3,532,481`，1M 点 layer-0/hash-0
collision rate 为 0.8518、mean co-occupants 为 94.7；16K 点分别为
0.9977 和 6048.0。该方向支持 collision pooling 的候选解释，但仍不能
判断参数量与 collision 哪个机制占主导，也不能声称 plateau。

### 2.3 Exact-frequency（当前标准基础 QC）

当前标准 exact-f 图显示，`both` 的 L1/L4 曲线都随 `f` 增大而下降，且
低频段 gap 约在 2–8，高频端降到约 2–3；L1 与 L4 的形状相近。在 table
轴，1M bigram 的 gap 随 `f` 明显下降（约 5.9→0.3），而 16K 曲线整体
接近 0。

这与“低频 context 更容易产生较大 gap”的 observational pattern 一致，
但图中为 exact-f 纳入后再做 log-frequency 分箱中位数的展示，不是两因素
公式的正式拟合。当前只有 seed 42、有限截面，且缺少 fixed-epoch、完整
table 网格和 profile-likelihood，因此不能报告 `A,c,β,γ` 的稳定估计。

### 2.4 Backbone safety

`bb_safety_L1_nogram_5000` 已完成（2026-08-24），**最终 fixed gap +16.66 @5000**
（train 0.0065 / val 16.666）。该 run 为旧 cadence（50 步）+ fp32 无 compile，
只作量级参考，不属当前标准：

| step | fixed train | fixed val | fixed gap |
|---:|---:|---:|---:|
| 1000 | 4.374 | 5.148 | +0.774 |
| 1200 | 4.022 | 5.247 | +1.225 |
| 1400 | 3.491 | 5.418 | +1.927 |
| 1681 | 3.138 | 5.724 | +2.586 |
| 4000 | 0.368 | 13.604 | +13.236 |
| **5000** | **0.007** | **16.666** | **+16.660** |

**结论（量级参考）**：长训 no-ngram backbone 自身就会产生巨大 gap（train 趋
0、val 16.7）—— 1000 步 gap ≈ 0 不能外推到 5000 步。因此 no-ngram 对照必须
在每个 L、每个对齐下重跑当前标准版本，不能假设 backbone gap 恒为零；
`ΔG = G_module − G_no-ngram` 修正口径仍然必要。

## 3. 图片约定

- **一图一变量**：epoch 图中只变 L（其余全冻结）；table 图中只变 table size。
- **三视图**：每张图同时展示 `train loss`、`val loss`、`gap = val − train`。
- 曲线标签直接写在各面板曲线末端；标题写明冻结的 module、alignment 和
  唯一变化变量，不依赖跨面板共享 legend。
- epoch fixed-step 图固定 module 后只比较 `L1/L2/L3/L4`；no-ngram 图是
  独立 backbone safety 对照。table 图固定 `L4 + bigram-only` 后只比较
  `table_mult`。final summary 仍只报告 fixed probe 的 train/val/gap 三项。

## 4. 任务目录与并行运行

开发代码全部位于 `tasks/s1_scaling_three_axis/`，包括训练副本、
`ngram_freq.py`、`table_occupancy.py`、launcher、分析脚本和测试；它们不再
通过相对路径调用根目录 `code/`。附录目录只保存面向阅读的报告、图和结果摘要。

各集群均应使用本仓库的独立副本；路径由启动环境中的 `NGLAB_ROOT` 指定，
结果写入各自的 `data/runs_scaling/<run_id>_fixed/`，不会跨机共享同一 run
目录。启动前必须：

1. 先完成本地 `py_compile` 与测量单测；
2. 同步整个 `tasks/s1_scaling_three_axis/`；
3. 用 `md5sum` 核对任务内 `code/` 与 launcher；
4. 确认目标 GPU 空闲后，再从任务目录启动当前标准 pilot/full launcher。

360-1 当前约有 400 GB 可用空间，可与 360-2 并行排队；这只是资源条件，
不代表 full grid 已启动。backbone safety 和 pilot QC 完成前不启动完整网格。

## 5. 状态与待办

- [x] 测量基础设施（epoch_batches / fixed probe / exact-freq / occupancy）+ 11 单测
- [x] 旧 Pilot QC（7 run）：epoch 4 + table 3，全部完成，但不满足当前标准
- [ ] Epoch full grid（L1–L4 × 4 modules × 2 alignments，seed 42；每个 L 配 no-ngram）
- [ ] Table full grid（7 尺寸 × 3 modules）
- [ ] Backbone safety（L1 no-ngram 5000 步）
- [ ] Frequency 两因素拟合 + manifest
- [ ] Seed 43/44 复现 + 三轴联合报告
