# 附录 · 三轴 scaling 验证（epoch 长度 / exact frequency / table size）

> **实验线**：T-scaling（极简 setting 下验证三条 scaling 曲线）
> **状态**：🟡 旧 pilot QC 已完成；当前标准的 S1 run 尚未形成
> canonical 证据
> **数据源**：`data/runs_scaling/<run_id>_fixed/`；现有不带 `_fixed` 的
> pilot 只作历史 QC
> **代码**：`tasks/s1_scaling_three_axis/`（launchers + analysis + 单测）
> **计划**：`docs/plans/plan-3-fix-and-backfill.md`

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
| validation / probe | 每 10 步；fixed validation + fixed train probe |
| 结果目录 | `data/runs_scaling/<run_id>_fixed/` |

## 2. 已有历史 pilot（不属于当前标准证据）

本地 `data/runs_scaling/` 中现有的 7 个目录均为旧 pilot/safety 产物：
它们的实际 metadata 使用 `table_lr_scale=1.0`，validation 和 fixed-train
probe 周期为 50 步，目录也不带 `_fixed`。因此当前分析脚本会拒绝这些目录；
它们的数字只能作为历史 QC，不能支撑当前 `table_lr_scale=2.0`、v10、
bf16+compile 的 scaling 结论。当前标准 run 必须同时满足 `_fixed` 目录、
`summary.json.run_id` 与物理目录名一致、β 值 `[0.0, 0.99]`、table LR
scale `2.0` 和 10 步 validation cadence。

### 2.0 数据量和完整性

当前本地 `data/runs_scaling/` 有 **7 个历史实验目录**：

- **6 个已完成的 1000-step run**：epoch 轴 4 个，table 轴 3 个；
- **1 个进行中的 backbone safety**：L1 + no-ngram，目标 5000 steps；
- 暂无 full-grid、fixed-epoch、trigram-only、both 的 table-size 或多 seed
  结果，因此当前不能声称三条 scaling 已被完整验证。

每个已完成 run 都有：

- 20 条在线 train/val 记录；
- L1 run 有 43 条 fixed-probe 记录，L4/table run 有 22 条；
- exact-frequency 最后快照覆盖 bigram 3088 个正频率、trigram 1477 个正频率；
- 在 `>=1024` tokens 且 `>=32` contexts 的 train/val 共同纳入标准下，
  最后快照有 63 个 bigram exact-f 值、57 个 trigram exact-f 值可用于
  初步比较；
- 三个 table run 另有逐 branch/layer/hash 的 occupancy 文件。

日志数据约 **555 MB**，其中约 **142 MB** 来自仍在写入的 backbone safety
exact-frequency 日志。所有已完成 run 的固定 train probe hash 都是
`38d1254a827759d6`，可直接横向比较。

### 2.1 Epoch length（历史 pilot，fixed-step 1000 步）

| run | epoch batches | 重播轮数 | final fixed gap |
|---|---|---|---|
| `pilot_ep_L1_both_fs` | 42 | ~23 | **+3.59** |
| `pilot_ep_L4_both_fs` | 336 | ~3 | **+0.74** |
| `pilot_ep_L1_nogram_fs` | 42 | ~23 | +0.17 |
| `pilot_ep_L4_nogram_fs` | 336 | ~3 | −0.004 |

**历史观察 F1（非 canonical）· gap 主要来自 n-gram 表重播，而非 backbone**：两个
1000-step no-ngram 对照的固定 gap 都接近 0，而 both 臂明显更大。这个结论
仍需等待 L1 no-ngram 5000-step safety 完成；当前 safety 在 step 1000 时
gap 为 +0.77，step 1681 时已到 +2.59，说明长训 backbone 也可能产生
non-negligible gap，不能再写成“backbone 自身不会产生 gap”。

**历史观察 F2（非 canonical）· fixed-step 下短 epoch 的 gap 明显更大**：L1（约 23
次重播）gap = 3.59，L4（约 3 次重播）gap = 0.74，约 4.8 倍。它支持
“更多 replay 会放大 gap”的解释，但还不是纯粹的 epoch-length scaling，
因为 fixed-epoch 对齐和完整 L1–L4 网格尚未完成。

（fixed-epoch 对齐的 full grid 才能分离「重播次数」与「epoch 长度本身」的效应。）

### 2.2 Table size（历史 pilot；L4, bigram-only, 1000 步）

| run | logical 2R | final gap | collision rate (bigram L0 h0) |
|---|---|---|---|
| `tbl_pilot_1M_bigram` | 1,048,576 | **+0.40** | 0.998+ |
| `tbl_pilot_128K_bigram` | 131,072 | +0.12 | 1.0 |
| `tbl_pilot_16K_bigram` | 16,384 | +0.03 | 1.0 |

**历史观察 F3（非 canonical）· table 越小 gap 越小（单调）**：1M→128K→16K
（逻辑地址从 1,048,576 降到 16,384），gap 为 **0.394→0.128→0.034**。
occupancy 诊断显示三点的 bigram collision rate 约为 **0.852→0.981→0.998**，
平均 co-occupants 约为 **94.6→756→6048**。这与 collision pooling 的方向
一致：小 table 把更多 context 混入同一 row，削弱逐 context 记忆。但现在只有
3 个 table size、1 个 module、1 个 seed，尚不足以区分 collision 曲线与
参数量曲线，也不能判断是否已经进入 plateau。

### 2.3 Exact-frequency 历史 QC 能说什么

在旧 pilot 满足纳入标准的 exact-f 上，bigram 的 gap 随 f 增大总体下降；例如：

- L1 both：低频 `f=1–10` 的中位 gap 约 7.40，高频 `f=11–100` 约 4.67；
- L4 both：对应约 1.82 和 0.84；
- table 1M / 128K / 16K：低频段约 1.20 / 0.37 / 0.08。

这与“低频 context 更容易产生较大的 train–val gap”的 observational
pattern 一致，但只属于旧 pilot QC。当前图是按 exact-f 先应用纳入标准、再在 log-frequency bins
内取中位数的降噪展示；它不是两因素公式的正式拟合。正式拟合还缺
epoch 截面、trigram-only 分支、权重与 profile-likelihood / 可辨识性分析。

### 2.4 Backbone safety 历史状态

`bb_safety_L1_nogram_5000` 仍在运行，当前本地快照为 step **1681/5000**：

| step | fixed train | fixed val | fixed gap |
|---:|---:|---:|---:|
| 1000 | 4.374 | 5.148 | +0.774 |
| 1200 | 4.022 | 5.247 | +1.225 |
| 1400 | 3.491 | 5.418 | +1.927 |
| 1681 | 3.138 | 5.724 | **+2.586** |

因此，旧 no-ngram 对照曾支持 n-gram-induced gap，但长训 safety 已经显示
backbone-only gap 会增长。最终表述必须以 5000-step 结果为准，不能使用
“backbone gap 恒为零”的强结论。

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
结果写入各自的 `data/runs_scaling/`，不会跨机共享同一 run 目录。启动前必须：

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
