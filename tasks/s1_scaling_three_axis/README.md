# S1 · 三轴 scaling 验证（epoch length / exact frequency / table size）

> **目标**：在唯一极简 setting（vanilla nanoGPT + input n-gram injection，
> 自然语料）下验证三条 scaling 曲线：
> ① **epoch 长度** `L` 如何影响 gap；
> ② **exact context frequency** `f` 的 `G(E,f)` 是否服从两因素模型；
> ③ **table size**（1M 逻辑地址只向下）如何决定 gap，隔离参数量 vs collision。
>
> 本任务**只用自然语料**。frequency 分支是「observational consistency」检验，
> **不是**频率因果证明（计划 §4.5 / DoD §9）。
> 完整计划：`docs/plans/plan-5-s1-three-axis-handoff.md`。

## 冻结 setting（不得改动）

- Backbone：vanilla nanoGPT，8L/6H/768D，learned abs position，LayerNorm，tied embedding。
- N-gram 注入：`input`（wte over-encoding）。
- 模块臂：`bigram-only` / `trigram-only` / `both` / `no-ngram`（共享 table 基线）。
- Table 优化器：RMSProp 无动量，**β₂=0.99**（`--table_betas 0.0,0.99`，全部显式传）。
- S1 launcher 固定 `--table_lr_scale 2.0`，与当前极简基线保持一致。
- Backbone：AdamW (0.8, 0.95) lr 0.004。
- 数据：自然语料、固定顺序 replay；train/val shard 严格不重叠。
- 结果目录：`data/runs_scaling/<run_id>_fixed/`（新 namespace，不与历史
  `runs_fixed/` 混用）。

## 代码

| 文件 | 用途 |
|---|---|
| `code/train.py` | 本任务自包含的主训练副本：`--epoch_batches`（嵌套前缀 epoch 长度）、online train/val gap、可选 `--fixed_train_probe` 诊断、`--probe_eval_interval`、exact-freq 观测、β₂=0.99 默认 |
| `code/ngram_freq.py` | 本任务自包含的 `GlobalFrequencyIndex.build_from_chunks`（chunk-boundary 精确语义）、`ExactFreqLossAccumulator`（向量化 exact-f 充分统计）、shared-context 统计 |
| `code/table_occupancy.py` | 本任务自包含的每 branch/layer/hash occupancy / collision / singleton / freq-weighted load |
| `launchers/run_scaling_epoch.sh` | epoch 网格（pilot + full 两段） |
| `launchers/run_scaling_epoch_full.sh` | epoch full grid（pilot QC 后） |
| `launchers/run_scaling_table.sh` | table 网格（pilot + full 两段） |
| `launchers/run_scaling_table_full.sh` | table full grid（pilot QC 后） |
| `analysis/analyze_scaling_epoch.py` | fixed-step / fixed-epoch 双对齐曲线 + ΔG vs no-ngram |
| `analysis/analyze_scaling_frequency.py` | token-marginal + context-matched gap(f)，两因素拟合 + manifest |
| `analysis/analyze_scaling_table.py` | gap vs 2R / collision / occupancy |
| `analysis/test_scaling_measurement.py` | 11 项正确性单测（epoch 边界、probe 安全、exact key 一致、hash 等价、occupancy 单调、β₂ 生效） |

## Epoch 长度网格（嵌套前缀）

| 标称 | batches/epoch | chunks/epoch | 相对 L4 |
|---|---|---|---|
| L1 | 42 | 3,024 | 1/8 |
| L2 | 84 | 6,048 | 1/4 |
| L3 | 168 | 12,096 | 1/2 |
| L4 | 337 | 24,264 | 1 |

L1/L2/L3 是同一 shard 1 数据流的嵌套前缀；L4 使用 shard 1 的完整
24,264 chunks。L4 不能再按 42 的整数倍近似。
两种对齐：
- **fixed-step (fs)**：1000 步，step-anchored LR，相同算力。
- **fixed-epoch (fe)**：6 完整 epoch，epoch-anchored LR，target steps L1=252 / L2=504 / L3=1008 / L4=2022。

## Table 网格（只向下）

| table_mult | 单 hash rows R | 逻辑地址 2R | 相对默认 |
|---|---:|---:|---:|
| 64 | 524,288 | 1,048,576 | 1 |
| 56 | 458,752 | 917,504 | 7/8 |
| 48 | 393,216 | 786,432 | 3/4 |
| 40 | 327,680 | 655,360 | 5/8 |
| 36 | 294,912 | 589,824 | 9/16 |
| 32 | 262,144 | 524,288 | 1/2 |
| 28 | 229,376 | 458,752 | 7/16 |
| 24 | 196,608 | 393,216 | 3/8 |
| 20 | 163,840 | 327,680 | 5/16 |
| 18 | 147,456 | 294,912 | 9/32 |
| 16 | 131,072 | 262,144 | 1/4 |
| 14 | 114,688 | 229,376 | 7/32 |
| 12 | 98,304 | 196,608 | 3/16 |
| 10 | 81,920 | 163,840 | 5/32 |
| 9 | 73,728 | 147,456 | 9/64 |
| 8 | 65,536 | 131,072 | 1/8 |
| 7 | 57,344 | 114,688 | 7/64 |
| 6 | 49,152 | 98,304 | 3/32 |
| 5 | 40,960 | 81,920 | 5/64 |
| 4 | 32,768 | 65,536 | 1/16 |
| 3 | 24,576 | 49,152 | 3/64 |
| 2 | 16,384 | 32,768 | 1/32 |
| 1 | 8,192 | 16,384 | 1/64 |

其中 64/32/16/8/4/2/1 为原始 21-run dense 网格；第一轮
48/24/12/6/3 为 15 个 sparse 加密 run，第二轮
56/40/36/28/20/18/14/10/9/7/5 为 22 个 bigram/trigram sparse
run。合计 69 个 table run；每个 run 同时输出 `table_occupancy.json`。

## 测量规则（重要）

1. **online gap（主测量）**：`train_log.jsonl` 中当前训练 batch 的
   `val_loss − train_loss`。所有 scaling gap 曲线、最终 gap 和模块比较都使用
   这一口径；它不会把同一批 train tokens 在 epoch 内的已消费比例混入 train loss。
2. **fixed train probe（诊断）**：只有显式传 `--fixed_train_probe N` 才启用；
   独立 dataset 实例抓取固定 train batches，SHA256 记账，全程复用，**不消费
   训练流、不推进 epoch 计数器**。顺序 replay 下它会受到 exposure/训练进度
   污染，`first` 和 `uniform` 都不能用来替代 online gap，也不能证伪
   epoch-1 的 online gap 现象。
3. 原始正式 epoch/table 网格的 validation、table norm 和 online train loss 每
   10 步触发；frequency 轴 exact-frequency 每 100 步。table 加密取点使用
   `MONITOR=sparse`，只在最终 step 1000 触发 val/norm，不产生中间曲线。
4. **exact-frequency**：索引 `GlobalFrequencyIndex.build_from_chunks` 与模型 hash
   逐位置一致（有单测）。f=0 novel 只报 val loss，不定义 gap。
5. 历史 run 可能仍含 `fixed_train_loss.jsonl`；其中的 fixed gap 只作为
   exposure-contaminated 诊断，不与 online gap 混写。

## 历史 pilot 与当前标准 basic QC

`pilot_*` / `tbl_pilot_*` 是旧协议的历史 QC。当前已有 7 个
`basic_*` 锚点，使用 β₂=0.99、table LR×2、bf16（不 compile）；其中旧
结果仍可能带固定 train probe 和 exact-frequency 日志，但为了快速 gate
使用 25 步 cadence。
它们用于判断基础规律是否出现，不替代正式的 10 步 full grid。
新的正式 launcher 会写入 `data/runs_scaling/<run_id>_fixed/`，并由完整
metadata 校验后才能进入 canonical 分析。

### Epoch（fixed-step 1000 步）

| run | final gap | 说明 |
|---|---|---|
| `pilot_ep_L1_both_fs` | **+3.59** | 短 epoch（42 b/ep），23 次重播，gap 最大 |
| `pilot_ep_L4_both_fs` | **+0.76** | 长 epoch（336 b/ep），3 次重播 |
| `pilot_ep_L1_nogram_fs` | +0.16 | backbone 自身几乎无 gap |
| `pilot_ep_L4_nogram_fs` | −0.01 | backbone 无 gap |

**历史观察（非 canonical）**：在该旧 pilot 协议下，gap 与 n-gram 表重播
方向一致；这不是当前标准下的结论。

### Table（L4, bigram-only, 1000 步）

| run | logical 2R | final gap |
|---|---|---|
| `tbl_pilot_1M_bigram` | 1,048,576 | **+0.40** |
| `tbl_pilot_128K_bigram` | 131,072 | +0.12 |
| `tbl_pilot_16K_bigram` | 16,384 | +0.03 |

**历史观察（非 canonical）**：该旧 pilot 中 table 越小 gap 越小；它只支持
后续当前标准 full grid 的候选方向，不能作为当前标准的 collision pooling 证据。

## 运行方法

```bash
# 先在仓库根目录完成 smoke test；任务代码不依赖根目录 code/
python3 -m py_compile tasks/s1_scaling_three_axis/code/*.py
.venv/bin/python tasks/s1_scaling_three_axis/analysis/test_scaling_measurement.py

# 360-1 / 360-2：同步本任务目录和必要数据后，再从任务 launcher 启动
rsync -avz tasks/s1_scaling_three_axis/ user@cluster:/path/to/ngram-gap-lab/tasks/s1_scaling_three_axis/
nohup bash /path/to/ngram-gap-lab/tasks/s1_scaling_three_axis/launchers/run_scaling_epoch.sh \
  > /path/to/ngram-gap-lab/logs/scaling_epoch.log 2>&1 &
```

launcher 默认从自身位置推导仓库根目录，也可用 `NGLAB_ROOT` 和
`NGLAB_PY` 覆盖；正式启动前仍应完成 `md5sum` 核对，并只运行
`run_scaling_epoch_full.sh` / `run_scaling_table_full.sh`（pilot QC 通过后）。
多 seed（43/44）在分析脚本冻结后补跑。

## 图表和报告约定

- 每个组图只允许一个自变量变化：epoch 图只改变 `L`，table 图只改变
  `table_mult`；module、seed、optimizer、数据前缀和训练预算必须固定。
- 每个组图固定为三面板：`train loss`、`val loss`、`gap = val − train`。
  gap 主结论使用 online train loss 与 fixed validation；fixed train probe
  只在显式诊断图中出现，不能替代 online gap。
- 曲线标签直接写在对应曲线末端或面板内，不依赖跨面板共享 legend
  猜测颜色含义；标题必须明确写出冻结条件与唯一变化变量。
- `docs/appendices/s1_scaling_three_axis/` 是同名报告目录，保存报告、
  组图和结果摘要；这里不覆盖 `docs/report/index.html`。

## 当前状态

- [x] 测量基础设施（epoch_batches / online gap / 可选 fixed probe / exact-freq / occupancy）+ 11 单测
- [x] 当前标准基础 QC（7 run，seed 42）：epoch 4 + table 3
- [x] Epoch full grid（32/32，seed 42；L4=337）
- [x] Table full grid + 两轮加密最终取点（69/69，seed 42；23 个 measured table mult；三种 module 各 23 点）
- [x] Backbone safety（L1 no-ngram 5000 步）—— final gap +16.66（旧 cadence，仅量级参考）
- [x] Frequency 轴与探索性拟合（8/8；两因素模型 + manifest）
- [x] 三轴联合报告回填（seed 42）
- [x] Seed 43/44 三 seed 复现（epoch 32×2 / table 36×2 / frequency 8×2，全部 QC 通过）与 H1–H4 检验：ΔG 方向 seed-stable；trigram table 幂律无饱和；两因素 β 可辨识（cv 4–13%）、A/c/γ 不可辨识；模块交互不可合并单公式（附录报告 §7）
- [ ] Frequency 的 epoch-dependent fit（epoch 3/6 截面）与跨 seed profile-likelihood

Table 图默认生成双对数 PNG，并同时生成两个可交互 HTML。HTML 中可点击
legend 隐藏/显示 bigram、trigram、both 曲线，并独立切换 x/y 轴为 linear
或 log。静态图也支持选择曲线和坐标轴，例如：

```bash
python tasks/s1_scaling_three_axis/analysis/analyze_scaling_table.py \
  data/runs_scaling --modules bigram,both --x-scale linear --y-scale log
```

Table 图默认生成双对数 PNG，并同时生成两个可交互 HTML。HTML 中可点击
legend 隐藏/显示 bigram、trigram、both 曲线，并独立切换 x/y 轴为 linear
或 log。静态图也支持选择曲线和坐标轴，例如：

```bash
python tasks/s1_scaling_three_axis/analysis/analyze_scaling_table.py \
  data/runs_scaling --modules bigram,both --x-scale linear --y-scale log
```
