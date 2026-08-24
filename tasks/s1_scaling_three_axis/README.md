# S1 · 三轴 scaling 验证（epoch length / exact frequency / table size）

> **目标**：在唯一极简 setting（vanilla nanoGPT + input n-gram injection，
> 自然语料）下验证三条 scaling 曲线：
> ① **epoch 长度** `L` 如何影响 gap；
> ② **exact context frequency** `f` 的 `G(E,f)` 是否服从两因素模型；
> ③ **table size**（1M 逻辑地址只向下）如何决定 gap，隔离参数量 vs collision。
>
> 本任务**只用自然语料**。frequency 分支是「observational consistency」检验，
> **不是**频率因果证明（计划 §4.5 / DoD §9）。
> 完整计划：`docs/plans/plan-3-fix-and-backfill.md`（plan.md 存档）。

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
| `code/train.py` | 本任务自包含的主训练副本：`--epoch_batches`（嵌套前缀 epoch 长度）、`--fixed_train_probe`（固定 train probe + SHA256）、`--probe_eval_interval`、exact-freq 观测、β₂=0.99 默认 |
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
| L4 | 336 | 24,192 | 1 |

四者严格嵌套前缀（同一 shard 1 数据流）；L4 只舍弃 shard 1 末尾少量 chunks。
两种对齐：
- **fixed-step (fs)**：1000 步，step-anchored LR，相同算力。
- **fixed-epoch (fe)**：6 完整 epoch，epoch-anchored LR，target steps L1=252 / L2=504 / L3=1008 / L4=2016。

## Table 网格（只向下）

| table_mult | 单 hash rows R | 逻辑地址 2R | 相对默认 |
|---|---:|---:|---:|
| 64 | 524,288 | 1,048,576 | 1 |
| 32 | 262,144 | 524,288 | 1/2 |
| 16 | 131,072 | 262,144 | 1/4 |
| 8 | 65,536 | 131,072 | 1/8 |
| 4 | 32,768 | 65,536 | 1/16 |
| 2 | 16,384 | 32,768 | 1/32 |
| 1 | 8,192 | 16,384 | 1/64 |

每个 run 同时输出 `table_occupancy.json`。

## 测量规则（重要）

1. **fixed train probe**：独立 dataset 实例抓取固定 4 个 train batches，SHA256 记账；
   全程复用，**不消费训练流、不推进 epoch 计数器**（防 B1 bug 复发）。
2. 周期评估每 10 步 + 每个 epoch 边界触发。
3. **exact-frequency**：索引 `GlobalFrequencyIndex.build_from_chunks` 与模型 hash
   逐位置一致（有单测）。f=0 novel 只报 val loss，不定义 gap。
4. gap 主量 = **fixed_val − fixed_train**；在线 train loss 仅作诊断。

## 历史 pilot QC（非当前标准证据）

本地已有的 `pilot_*` 目录是 2026-08-24 的旧 pilot。它们的
`summary.json` 实际记录 `table_lr_scale=1.0`、50 步 validation/probe
周期，且目录没有 `_fixed` 后缀，因此分析脚本会主动排除它们。下面的数值只
保留作历史 QC 和迁移溯源，不能用于当前 β₂=0.99、table LR×2、v10、
bf16+compile 标准下的 scaling 结论。新的 launcher 会写入
`data/runs_scaling/<run_id>_fixed/`，并由 metadata 校验后才能进入分析。

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
python3 tasks/s1_scaling_three_axis/analysis/test_scaling_measurement.py

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
  gap 主结论使用固定 train probe 与固定 val probe；在线 train loss 仅作诊断。
- 曲线标签直接写在对应曲线末端或面板内，不依赖跨面板共享 legend
  猜测颜色含义；标题必须明确写出冻结条件与唯一变化变量。
- `docs/appendices/s1_scaling_three_axis/` 是同名报告目录，保存报告、
  组图和结果摘要；这里不覆盖 `docs/report/index.html`。

## 状态

- [x] 测量基础设施（epoch_batches / fixed probe / exact-freq / occupancy）+ 11 单测
- [x] 旧 Pilot QC（7 run，seed 42）：epoch 4 + table 3，全部完成，但不满足当前标准
- [ ] Epoch full grid（L1–L4 × 4 modules × 2 alignments；每个 L 都有 no-ngram 基线）
- [ ] Table full grid（7 尺寸 × 3 modules）
- [ ] Backbone safety（L1 no-ngram 5000 步）—— 跑完后确认 backbone 长训无 gap
- [ ] Frequency 拟合（两因素模型 + manifest）
- [ ] 三轴联合报告回填
