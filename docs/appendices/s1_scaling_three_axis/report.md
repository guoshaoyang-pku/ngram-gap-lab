# 附录 · 三轴 scaling 验证（epoch 长度 / exact frequency / table size）

> **实验线**：T-scaling（极简 setting 下验证三条 scaling 曲线）
> **状态**：🟡 Pilot 完成（seed 42，7 run），full grid 待跑
> **数据源**：`data/runs_scaling/`（新 namespace；β₂=0.99 全部生效）
> **代码**：`tasks/s1_scaling_three_axis/`（launchers + analysis + 单测）
> **计划**：`docs/notes/plans/plan-3-fix-and-backfill.md`

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
| S1 table LR | `table_lr_scale=1.0`，与 pilot 保持一致；不混入全局默认 ×2 |
| backbone 优化器 | AdamW (0.8, 0.95)，lr 0.004 |
| 数据 | 自然语料 shard 1 嵌套前缀，train/val 零重叠 |
| 测量 | fixed train probe（4 batches，SHA256）+ exact-frequency + occupancy |
| 结果目录 | `data/runs_scaling/` |

## 2. Pilot 结果（seed 42，2026-08-24）

### 2.1 Epoch length（fixed-step 1000 步）

| run | epoch batches | 重播轮数 | final fixed gap |
|---|---|---|---|
| `pilot_ep_L1_both_fs` | 42 | ~23 | **+3.59** |
| `pilot_ep_L4_both_fs` | 336 | ~3 | **+0.76** |
| `pilot_ep_L1_nogram_fs` | 42 | ~23 | +0.16 |
| `pilot_ep_L4_nogram_fs` | 336 | ~3 | −0.01 |

**发现 F1 · gap 由 n-gram 表重播产生，非 backbone**：pilot 中 no-ngram 臂在两种 L 下
gap 都 ≈ 0（≤0.16），而 both 臂 gap 大 20–400 倍。backbone 自身不产生 gap。

**发现 F2 · 重播轮数越多 gap 越大**：L1（23 次重播）gap = 3.59，L4（3 次重播）
= 0.76，差 ~4.7 倍。fixed-step 对齐下，短 epoch 因 replay 次数多而放大 gap。

（fixed-epoch 对齐的 full grid 才能分离「重播次数」与「epoch 长度本身」的效应。）

### 2.2 Table size（L4, bigram-only, 1000 步）

| run | logical 2R | final gap | collision rate (bigram L0 h0) |
|---|---|---|---|
| `tbl_pilot_1M_bigram` | 1,048,576 | **+0.40** | 0.998+ |
| `tbl_pilot_128K_bigram` | 131,072 | +0.12 | 1.0 |
| `tbl_pilot_16K_bigram` | 16,384 | +0.03 | 1.0 |

**发现 F3 · table 越小 gap 越小（单调）**：1M→16K（64 倍地址差），gap 从 0.40
降到 0.03（~12 倍）。初步支持 **collision pooling 假设**：小 table 的 hash
碰撞把不同 context 的 embedding 平均化，稀释逐 context 记忆。full grid
（7 尺寸）才能判定是否存在 plateau 与碰撞后的饱和形状。

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

360-1 与 360-2 均使用 `/data/home/guoshaoyang/ngram-gap-lab/` 作为仓库副本，
结果写入各自的 `data/runs_scaling/`，不会跨机共享同一 run 目录。启动前必须：

1. 先完成本地 `py_compile` 与测量单测；
2. 同步整个 `tasks/s1_scaling_three_axis/`；
3. 用 `md5sum` 核对任务内 `code/` 与 launcher；
4. 确认目标 GPU 空闲后，再从任务目录启动 pilot/full launcher。

360-1 当前约有 400 GB 可用空间，可与 360-2 并行排队；这只是资源条件，
不代表 full grid 已启动。backbone safety 和 pilot QC 完成前不启动完整网格。

## 5. 状态与待办

- [x] 测量基础设施（epoch_batches / fixed probe / exact-freq / occupancy）+ 11 单测
- [x] Pilot（7 run）：epoch 4 + table 3，全部完成
- [ ] Epoch full grid（L1–L4 × 4 modules × 2 alignments，seed 42；每个 L 配 no-ngram）
- [ ] Table full grid（7 尺寸 × 3 modules）
- [ ] Backbone safety（L1 no-ngram 5000 步）
- [ ] Frequency 两因素拟合 + manifest
- [ ] Seed 43/44 复现 + 三轴联合报告
