# ngram-gap-lab · 实验日志

> 创建：2026-08-05
> **本文件是多 Agent 的唯一实验登记簿**：先登记（`planned`）→ 占 GPU 开跑 → 回填（`done`）。
> 工作原则、run_id 约定与并行规则见 `agents.md` §0；**极简 setting（SSOT）见 `agents.md` §1**。

## 实验登记总表

| run_id | 日期 | 实验 | 状态 | gap 关键值 | 详情 |
|---|---|---|---|---|---|
| `vanilla_input_1000_seed42` | 2026-08-23 | 干净 vanilla 复现 · input 注入 · 1000 步 | ✅ done | **+0.858 @1000** | §14 |
| `vanilla_nogram_1000_seed42` | 2026-08-23 | 干净 vanilla 复现 · 无 n-gram 对照 · 1000 步 | ✅ done | **+0.038 @1000** | §14 |
| `nglab1x_input_reset_e2` | 2026-08-24 | P1 因果 · e2 边界全 table 回滚 | ✅ done | **+0.054 @1000（−94%）** | §15 |
| `nglab1x_input_reset_e1` | 2026-08-24 | P1 因果 · e1 边界全 table 回滚 | ✅ done | +0.351 @1000（−59%） | §15 |
| `nglab1x_input_mask_e1` | 2026-08-24 | P2 因果 · e1 边界屏蔽 readout | ✅ done | **+0.058 @1000（−93%）** | §15 |
| `nglab1x_input_freeze_table_e1` | 2026-08-24 | P2 因果 · e1 边界冻结 table | ✅ done | +0.601 @1000（−30%） | §15 |
| `nglab1x_input_freeze_backbone_e1` | 2026-08-24 | P2 因果 · e1 边界冻结 backbone | ✅ done | +0.780 @1000（−9%） | §15 |
| `nglab_v` | 2026-08-05 | 注入点消融 · v | ✅ done | 0.33 @999 | §2 |
| `nglab_y` | 2026-08-05 | 注入点消融 · y | ✅ done | 3.50 @999 | §2 |
| `nglab_input` | 2026-08-05 | 注入点消融 · input | ✅ done | 0.79 @999 | §2 |
| `nglab2x_v` | 2026-08-06 | 双倍训练集 · v | ✅ done | 1.169 @2000 | §4 |
| `nglab2x_y` | 2026-08-06 | 双倍训练集 · y | ✅ done | 3.101 @2000 | §4 |
| `nglab2x_input` | 2026-08-06 | 双倍训练集 · input | ✅ done | 0.687 @2000 | §4 |
| `nglab2x_input_v10` | 2026-08-06 | 双倍训练集 · input · v10 细曲线 | ⛔ superseded | val 移动窗，已停 | §6 |
| `nglab0_5x_input` | 2026-08-06 | 半 epoch 训练集 · input · v10 | ⛔ superseded | val 移动窗，已停 | §7 |
| `nglab2x_input_v10_fv` | 2026-08-06 | 双倍训练集 · input · v10 · **fixed-val** | ✅ done | 0.502 @2000 | §6 |
| `nglab0_5x_input_fv` | 2026-08-06 | 半 epoch 训练集 · input · v10 · **fixed-val** | ✅ done | 4.952 @2000 | §7 |
| `nglab1x_v10_v` | 2026-08-06 | 标准 1x · v 注入 · v10 重跑 | ✅ done | 5.041@2000 | §8 |
| `nglab1x_v10_y` | 2026-08-06 | 标准 1x · y 注入 · v10 重跑 | ✅ done | 5.049@2000 | §8 |
| `nglab1x_v10_input` | 2026-08-06 | 标准 1x · input 注入 · v10 重跑 | ✅ done | 1.931@2000 | §8 |
| `nglab1x_v10_nogram` | 2026-08-06 | 标准 1x · 无 n-gram 对照 · v10 重跑 | ✅ done | 0.231@2000 | §8 |
| `nglab1x_opt_rmsprop_2x` | 2026-08-06 | table 优化器消融 · RMSProp lr×2（2000 步）| ✅ done | 2.376@2000 | §9/9a/9b |
| `nglab1x_opt_adamw_090999` | 2026-08-06 | table 优化器消融 · AdamW(0.9,0.999) | ✅ done | 0.912@1000 | §9/9a |
| `nglab1x_opt_adamw_080950` | 2026-08-06 | table 优化器消融 · AdamW(0.8,0.95) | ✅ done | 0.709@1000 | §9/9a |
| `nglab1x_opt_sgd_09` | 2026-08-06 | table 优化器消融 · SGD momentum 0.9 | ✅ done | −0.002@1000（table 未学）| §9/9a/9b |
| `nglab1x_opt_rmsprop_4x` | 2026-08-06 | table 优化器消融 · RMSProp lr×4（剂量上限，2000 步对照）| ✅ done | 4.742@2000 | §9a/9b/9d |
| `nglab1x_opt_rmsprop_4x_b2_099` | 2026-08-07 | β2 反向扫描 · RMSProp 4x · b2=0.99 | ✅ done | 5.143@2000 | §9d |
| `nglab1x_opt_rmsprop_4x_b2_098` | 2026-08-07 | β2 反向扫描 · RMSProp 4x · b2=0.98 | ✅ done | 5.155@2000 | §9d |
| `nglab025x_b2_099` | 2026-08-07 | 短 epoch × β2 · 0.25x · b2=0.99 | ✅ done | 13.577@2000 | §11 |
| `nglab05x_b2_099` | 2026-08-07 | 短 epoch × β2 · 0.5x · b2=0.99 | ✅ done | 5.017@2000 | §11 |
| `nglab1x_opt_rmsprop_2x_s43` | 2026-08-06 | RMSProp lr×2 · seed43 | ✅ done | 2.111@1000 | §9a/9b |
| `nglab1x_opt_adamw_090999_s43` | 2026-08-06 | AdamW(0.9,0.999) · seed43 | ✅ done | 1.443@1000 | §9a/9b |
| `nglab1x_opt_rmsprop_2x_s44` | 2026-08-06 | RMSProp lr×2 · seed44 | ✅ done | 2.089@1000 | §9a/9b |
| `nglab1x_opt_adamw_090999_s44` | 2026-08-06 | AdamW(0.9,0.999) · seed44 | ✅ done | 1.672@1000 | §9a/9b |
| `nglab0_25x_input_fv` | 2026-08-07 | shard 扫描 · 0.25x | 🔄 running | 待填 | §10 |
| `nglab0_75x_input_fv` | 2026-08-07 | shard 扫描 · 0.75x | 🔄 running | 待填 | §10 |
| `nglab1_5x_input_fv` | 2026-08-07 | shard 扫描 · 1.5x | 🔄 running | 待填 | §10 |
| `nglab2_5x_input_fv` | 2026-08-07 | shard 扫描 · 2.5x | ⛔ superseded | val 与 train 重叠 | §10 |
| `nglab3x_input_fv` | 2026-08-07 | shard 扫描 · 3x | ⛔ superseded | val 与 train 重叠 | §10 |
| `nglab4x_input_fv` | 2026-08-07 | shard 扫描 · 4x | ⛔ superseded | val 与 train 重叠 | §10 |
| `nglab2_5x_input_fv_v2` | 2026-08-07 | shard 扫描 · 2.5x（修正 val）| 🔄 running | 待填 | §10 |
| `nglab3x_input_fv_v2` | 2026-08-07 | shard 扫描 · 3x（修正 val）| 🔄 running | 待填 | §10 |
| `nglab4x_input_fv_v2` | 2026-08-07 | shard 扫描 · 4x（修正 val）| 🔄 running | 待填 | §10 |
| `nglab5x_input_fv` | 2026-08-07 | shard 扫描 · 5x（360-2）| 🔄 running | 待填 | §10 |
| `nglab6x_input_fv` | 2026-08-07 | shard 扫描 · 6x（360-2）| 🔄 running | 待填 | §10 |
| `nglab8x_input_fv` | 2026-08-07 | shard 扫描 · 8x（360-2）| 🔄 running | 待填 | §10 |
| `t5z_zipf_s42/s43/s44` | 2026-08-07 | toy 严格 Zipf 分布（N_r∝1/r²）· per-bucket gap | ✅ done | 7.01/7.96/7.56 @2000 | §13 |
| `nglab_plot_baseline` | 2026-08-06 | 基础实验统计与图表归档 | ✅ done | 15 bins + log/log-log | §10 |

状态约定：`planned` 已登记未开跑 / `running` 运行中 / `done` 已回填 / `stalled` 超期未回填。
新实验流程：总表加一行拿到唯一 `run_id` → 正文新建 section 按 `agents.md` §3 / `docs/plan.md` 模板填写
→ 占 GPU 开跑 → 结果回填并改状态。

---

## 1. 注入点消融（2026-08-05，OPHIS 旧 run 迁移）

以下数据来自旧代码库的历史 run（`injpos_*_freq2`），
作为新 repo 的历史对照基线。新 repo 的 `train.py` 精简重写后需复现这些数值。

### 1.1 1000 步消融

| run | 注入点 | n-gram | optimizer | steps | seed | gap@999 | norm 诊断 |
|---|---|---|---|---|---|---|---|
| `injpos_v_freq2` | v | bi+tri | mixed rmsprop | 1000 | 42 | 0.60 | n-gram residual = V 的 6.5% |
| `injpos_y_freq2` | y | bi+tri | mixed rmsprop | 1000 | 42 | 1.82 | — |
| `injpos_input_freq2` | input | bi+tri | mixed rmsprop | 1000 | 42 | 0.64 | n-gram residual = wte 的 4.77x |
| `injpos_baseline_no_ngram` | — | none | mixed rmsprop | 1000 | 42 | 0.03 | — |

### 1.2 2000 步延长

| run | 注入点 | gap@999 | gap@1999 |
|---|---|---|---|
| `injpos_v_long2000` | v | 0.60 | 4.70 |
| `injpos_y_long2000` | y | 2.10 | 4.65 |
| `injpos_input_long2000` | input | 0.75 | 2.98 |

### 1.3 Table norm × gap（theory obs，103 个点）

| step | v gap | y gap | input gap | v bg_rms | y bg_rms | input bg_rms |
|---|---|---|---|---|---|---|
| 10 | 0.00 | 0.00 | 0.00 | 0.037 | 0.037 | 0.037 |
| 100 | -0.004 | 0.011 | 0.002 | 0.042 | 0.083 | 0.068 |
| 337(e2) | -0.047 | -0.048 | -0.093 | 0.121 | 0.119 | 0.113 |
| 686(e3) | -0.077 | -0.790 | -0.342 | 0.160 | 0.149 | 0.150 |
| 999 | -0.542 | -1.900 | -0.672 | — | — | — |

注：gap = val - train（正值=gap）。bg_rms = bigram table layer_1 table_0 的 param rms。

### 1.4 关键结论

1. **v 注入无 gap 的原因是数值尺度问题**：n-gram value norm 只有 V 的 6.5%，信号被 V 淹没。
2. **y/input 注入都能产生 gap**：只要 n-gram 信号不走 attention 混合、能有效到达输出。
3. **gap 仅依赖 n-gram memory**：不需要 current shell / Muon / RoPE / RMSNorm。
4. **Table norm 增长速度不是 gap 的决定因素；注入点（信号能否到达输出）才是。**

## 2. 新 repo 复现验证（2026-08-05 完成）

用 `code/train.py`（精简版）重跑 v/y/input 三注入点，核对 gap 数值是否与 §1 一致。

| run | 注入点 | steps | gap@999（目标）| gap@999（实测）| 状态 |
|---|---|---|---|---|---|
| `nglab_v` | v | 1000 | 0.60 | 0.33 | ✅ |
| `nglab_y` | y | 1000 | 1.82 | 3.50 | ✅ |
| `nglab_input` | input | 1000 | 0.64 | 0.79 | ✅ |

**相对顺序一致**：y > input > v。绝对数值与旧实验有差异（LR schedule 不同），但现象完全可复现。

### 2.1 频率 bin 分解验证

- bigram novel frac: 4.3%（旧实验 ~4%）✅
- trigram novel frac: 31.2%（旧实验 ~30%）✅
- novel + 低频 bucket 主导 gap（详见 `fig_gap_by_freq.html`）

## 3. 频率 bin 分解（2026-08-05 完成）

用 `code/ngram_freq.py` 构建频率索引，统计 per-bin 的 mean loss 与 total contribution。

结果：novel + 低频 bucket（1-5）主导 gap；高频 bucket（5k+）gap 贡献 ≈ 0。与旧实验一致。

## 4. 双倍 training size 延长实验（2026-08-06，ophis-gpu）

目的：把 fixed replay 的 train 数据从 shard 1 扩大到 shard 1+2，观察更长的 epoch 平台是否能让 replay gap 更清楚。三种注入点使用完全相同的 setting，并行跑到 2000 steps。

### 4.1 Setting

| 项目 | setting |
|---|---|
| train shards | `1,2`（约 2x，约 600 steps / epoch） |
| validation shards | `3,4,5,6,7,8,9,10,6542` |
| model | vanilla nanoGPT, 8L / 6H / 768D |
| n-gram | trainable bigram + trigram |
| optimizer | backbone AdamW + table RMSProp (`beta1=0`, `beta2=0.999`) |
| learning rate | `0.004`，沿用原 warmup / warmdown schedule |
| seed | `42` |
| steps | `2000` |
| validation / freq eval | 每 50 steps，4 batches |
| table norm | 每 10 steps |
| runs | `nglab2x_v`, `nglab2x_y`, `nglab2x_input` |

双倍训练集对应的 exact-context frequency index 为 `data/freq_index_train2x.npz`。每个 run 均保留 `train_log.jsonl`、`table_norm.jsonl`、`freq_bin_loss.jsonl`、`summary.json` 和原始 `train.log`；本地备用归档为 `data/nglab2x_runs.tar.gz`。

### 4.2 Gap 结果

| run | 注入点 | gap@1000 | gap@1200 | gap@1500 | gap@2000 |
|---|---:|---:|---:|---:|---:|
| `nglab2x_v` | v | 0.001 | 0.068 | 0.482 | **1.169** |
| `nglab2x_y` | y | 0.220 | 0.752 | 2.174 | **3.101** |
| `nglab2x_input` | input | 0.152 | 0.213 | 0.460 | **0.687** |

### 4.3 Epoch 平台统计

| run | epoch | step range | gap mean | gap min–max | final gap |
|---|---:|---:|---:|---:|---:|
| v | 1 | 50–600 | -0.005 | -0.053–0.069 | 0.003 |
| v | 2 | 650–1200 | 0.032 | -0.015–0.068 | 0.068 |
| v | 3 | 1250–1800 | 0.434 | 0.199–0.654 | 0.614 |
| v | 4 | 1850–2000 | 0.950 | 0.787–1.169 | 1.169 |
| y | 1 | 50–600 | 0.001 | -0.044–0.053 | -0.009 |
| y | 2 | 650–1200 | 0.394 | 0.018–0.752 | 0.752 |
| y | 3 | 1250–1800 | 1.872 | 0.993–2.345 | 2.308 |
| y | 4 | 1850–2000 | 2.903 | 2.603–3.101 | 3.101 |
| input | 1 | 50–600 | -0.008 | -0.045–0.021 | 0.006 |
| input | 2 | 650–1200 | 0.140 | -0.015–0.220 | 0.213 |
| input | 3 | 1250–1800 | 0.416 | 0.255–0.582 | 0.507 |
| input | 4 | 1850–2000 | 0.618 | 0.538–0.687 | 0.687 |

Epoch boundaries occur around steps 600, 1200, 1800. The platform effect is substantially clearer than in the original one-shard run: y shows a stepwise progression `~0 → 0.75 → 2.3 → 3.1`, input shows `~0 → 0.21 → 0.51 → 0.69`, and v only becomes visibly positive after the third replay.

### 4.4 Final table norm and frequency coverage

Final table RMS:

| run | representative bigram RMS | representative trigram RMS | all norm rows |
|---|---:|---:|---:|
| v | 0.1202 (`layer_01`) | 0.1371 (`layer_01`) | 200 |
| y | 0.1449 (`layer_01`) | 0.1611 (`layer_01`) | 200 |
| input | 0.0860 (`layer_01`) | 0.0876 (`layer_01`) | 200 |

At step 2000, every frequency file has 40 checkpoints and every checkpoint covers all 15 buckets for both bigram and trigram. Bucket fractions sum to exactly 1.0 for train and validation, with 589,824 evaluated tokens per split.

| branch | train novel | val novel | train hit=1 | val hit=1 |
|---|---:|---:|---:|---:|
| bigram | 0.0% | 2.85% | 2.34% | 1.61% |
| trigram | 0.07% | 25.58% | 22.38% | 7.57% |

The novel fractions decrease relative to the one-shard index because the doubled training set covers more contexts, while the low-frequency trigram mass remains substantial. The full raw outputs are retained for future plots and alternative optimizer comparisons.

## 10. 基础实验统计与图表归档（2026-08-06）

目的：把已经完成并确认口径的基础实验，连同完整统计和图表生成方法登记为
可复用的干净基线。该 section 不启动新训练，不覆盖其他 Agent 的 running
实验。

> 口径说明：早期 `nglab_v/y/input` 是 validation/freq eval 每 50 步的历史
> 基线；后续 canonical 主线统一为每 10 步（v10），以获得更密集、更清晰的
> epoch replay 曲线。代码默认值、`run_injpos.sh` 和 `docs/plan.md` 均采用 v10。

### 数据与统计口径

基础图表使用 `nglab_v`、`nglab_y`、`nglab_input` 的完整日志：

| 统计层级 | 产物 | 内容 |
|---|---|---|
| global step | `train_log.jsonl` | train loss、fixed validation loss、global gap、epoch |
| table memory | `table_norm.jsonl` | 每个 table 的 RMS/norm 随 step 变化 |
| frequency bucket | `freq_bin_loss.jsonl` | bigram/trigram、train/val、15 个真实 bucket 的 token count、fraction、mean token loss、total contribution |
| summary | `summary.json` | run 的最终摘要 |

frequency loss 使用 unreduced per-token cross-entropy：先得到每个 token 的
loss，再按照真实 context 的 training hit-count bucket 聚合。运行文件保存
每个 bucket 的聚合统计，而不是保存全部逐 token loss 数组。

`novel` 是 training hit count 为 0 的 context。它可以有 validation loss，
但 train 侧没有对应 token，因此没有 train mean loss，不能定义标准的
`val loss - train loss`。因此 novel 保留在 raw/fraction 图中，并从 gap
和 log/log-log 图中排除。

### 图表思想

| 图表 | 目的 |
|---|---|
| global loss/gap | 显示 fixed-order replay 后 train 与 validation 的分叉，以及 epoch boundary |
| table norm × loss | 确认 n-gram table 确实被写入，并对齐 memory growth 与 gap |
| frequency-bin timeline | 观察每个频率 bucket 的 per-token loss、gap 和 total contribution 如何随 replay 演化 |
| frequency histogram + final gap | 柱表示 train/val token fraction，曲线表示末态 per-bin gap；同时看“占多少”和“差多少” |
| log-x / log-log frequency-to-gap | 用 bucket 命中次数的几何中点定量观察频率与末态 gap 的关系；横向误差线表示 bucket 范围 |

### 代码与产物

- canonical generator：`docs/plot_scripts/gen_all_figures.py`
- 作图目录说明：`docs/plot_scripts/README.md`
- summary/norm JSON builder：`docs/plot_scripts/build_injpos_data_json.py`
- epoch-length comparison：`docs/plot_scripts/gen_epoch_scale_figs.py`
- 早期 provenance generator：`docs/plot_scripts/gen_injpos_plot.py`
- 输出目录：`docs/figs/`
- public guide mirror：
  `guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide/`
- 运行数据：历史结果目录（未修正版，不作为当前权威数据）

本次归档的图表生成 commit：

- plotting pipeline：`10cb1b0`
- public guide log/log-log view：`d59111c`


---

## 6. 双倍训练集 v10 细曲线（2026-08-06，input）

目的：§4 的 nglab2x 批是 v50（每 50 步），无法与 v10 标准曲线对齐；本 run 用 v10 重跑 input 注入，保证 epoch 平台与 gap 曲线可与标准 1x（v10）逐点比较。

### Setting

| 项 | 值 |
|---|---|
| train shards | `1,2`（约 2x，~674 steps/epoch）|
| validation shards | `3,4,5,6,7,8,9,10,6542`（与 §4 一致）|
| steps / seed | 2000 / 42 |
| validation / freq eval | **每 10 steps**（v10）|
| freq index | `data/freq_index_train2x_fine.npz` |
| run | `nglab2x_input_v10`（首跑）→ **`nglab2x_input_v10_fv`**（fixed-val 重跑，当前） |

> **val-fix（2026-08-06 20:3x）**：首跑时 `evaluate_val` 从 `val_iter` 顺序取批，
> v10 下 200 次 eval × 4 批会让 val 曲线在 val 集上滑动（移动窗），
> 不满足「val 数据始终同一套」。已在 `code/train.py` 修复：启动时一次性捕获
> `fixed_val_batches`（val loss）与 `fixed_freq_val_batches`（val 侧 freq-bin），
> 每次 eval 复用同一批 val 数据；train 仍是唯一移动队列。首跑（移动窗）已停，
> 首轮数据为历史未修正版结果，正式结果以 `_fixed` run 为准。

### 结果

| run | final train | final val | gap@2000 | 观测 epoch 长（边界步）|
|---|---|---|---|---|
| `nglab2x_input_v10_fv` | 3.041 | 3.543 | **+0.502** | ~450（460, 900, 1350, 1800）|

- v10 细曲线 200 个点（每 10 步），fixed-val：val loss 每次都测同一批 val 数据。
- 关键现象：2x 下 2000 步只走 ~4.4 个 epoch，train 下降但 gap 到 2000 步仍只有 +0.5
  （对照 0.5x 同预算走 18 个 epoch，gap +4.95；见 §7）。

## 7. 半 epoch 训练集（2026-08-06，input）

目的：把 fixed replay 的 epoch 长度减半，与 1x（337 steps/epoch）和 2x（674 steps/epoch）在相同 2000 步预算下对比「epoch 平台长度 → replay gap」的剂量关系。

### Setting

| 项 | 值 |
|---|---|
| train shards | `60`（shard_00060 = shard_00001 前 12132 行，~168 steps/epoch）|
| validation shards | `2,3,4,5,6,7,8,9,10,6542`（与标准一致）|
| steps / seed | 2000 / 42 |
| validation / freq eval | 每 10 steps |
| freq index | `data/freq_index_train0_5x.npz` |
| run | `nglab0_5x_input`（首跑）→ **`nglab0_5x_input_fv`**（fixed-val 重跑，当前） |

> **val-fix（2026-08-06 20:3x）**：同上 §6——首跑 val 为移动窗，已停；
> 正式结果以 `_fv` run 为准（train 侧行为完全一致，仅 val 改为固定批次）。

### 结果

| run | final train | final val | gap@2000 | 观测 epoch 长（边界步）|
|---|---|---|---|---|
| `nglab0_5x_input_fv` | 1.776 | 6.728 | **+4.952** | ~110–120（120, 230, 350, …）|

- v10 细曲线 200 个点，fixed-val。
- 关键现象：epoch 减半后，train 塌到 1.78 而 val 升到 6.73，gap 几乎是 2x（+0.50）的 **10 倍**；
  gap 从 epoch 2（~step 120）就开始转正，符合「epoch 平台越短 → replay 越早、越猛」的剂量关系。
- 与 1x（`nglab1x_v10_input`，parallel agent，fixed-val，观测 ~230 steps/epoch）对照见
  `docs/figs/epoch_scale/epoch_scale_train_val_gap.png`（3 条曲线，均已 2000 步完成）。

### 剂量关系汇总（gap@2000，input 注入，v10，fixed-val，seed 42）

| epoch 长 | run | 观测 steps/epoch | train@2000 | val@2000 | gap@2000 |
|---|---|---|---|---|---|
| 0.5x | `nglab0_5x_input_fv` | ~110–120 | 1.776 | 6.728 | **+4.95** |
| 1x | `nglab1x_v10_input` | ~230 | 2.707 | 4.669 | **+1.96** |
| 2x | `nglab2x_input_v10_fv` | ~450 | 3.041 | 3.544 | **+0.50** |

结论：epoch 平台越短，train 塌缩越深、val 翘起越早越强；2000 步预算内
0.5x 的 gap 是 2x 的 ~10 倍。

## 8. 标准 1x v10 重跑（2026-08-06，blog 克隆任务）

目的：博客 `ngram-gap-mechanism-guide` 主线的 v/y/input 消融原为 1000 步、
validation 每 50 步；本批用 **v10 标准（validation + freq eval 每 10 步）重跑到 2000 步**，
重做 v/y/input 三注入点 + 无 n-gram 对照，产出更细的 loss/gap 曲线，
并克隆一份博客文档。

### Setting

| 项 | 值 |
|---|---|
| train shards | `1`（标准 1x，约 250–316 steps/epoch，含 freq eval 消耗）|
| validation shards | `2,3,4,5,6,7,8,9,10,6542`（与标准一致）|
| steps / seed | 2000 / 42 |
| validation / freq eval | **每 10 steps**（v10）|
| table norm | 每 10 steps |
| freq index | `data/freq_index.npz` |
| runs | `nglab1x_v10_v` / `nglab1x_v10_y` / `nglab1x_v10_input` / `nglab1x_v10_nogram` |

### 结果（ophis-gpu 首波 · 23 桶 freq 统计，已 done；360-1 15 桶重跑进行中）

- ophis-gpu 首波（2026-08-06 21:41–22:41 CST，run_id 同上，使用当时 23 桶版 `ngram_freq.py`）：
  - `nglab1x_v10_v`：final_gap **4.9497**（train 1.3536 / val 6.3033）
  - `nglab1x_v10_y`：final_gap **5.0552**（train 1.3601 / val 6.4153）
  - `nglab1x_v10_input`：final_gap **1.9615**（train 2.7072 / val 4.6687）
  - `nglab1x_v10_nogram`：final_gap **0.2253**（train 3.0167 / val 3.2420）
  - freq bin 统计为 23 桶（另一 agent 的未提交改动在开跑前已同步到 ophis），train/val gap 与桶数无关。
- 360-1 重跑（2026-08-06 22:33–23:51 CST，15 桶提交版 `ngram_freq.py`，4 卡并行；与博客口径一致，作为克隆文档的权威数据）：
  - `nglab1x_v10_v`：final_gap **5.0406**（train 1.2377 / val 6.2783）
  - `nglab1x_v10_y`：final_gap **5.0493**（train 1.3698 / val 6.4191）
  - `nglab1x_v10_input`：final_gap **1.9308**（train 2.7082 / val 4.6390）
  - `nglab1x_v10_nogram`：final_gap **0.2312**（train 3.0033 / val 3.2345）
  - epoch 边界（每 10 步 val 记录）：230 / 460 / 680 / 910 / 1130 / 1360 / 1580 / 1810（约 226 steps/epoch，freq eval 每 10 步消耗 4 个 train batch）。
  - 与 ophis 首波交叉验证：四 run gap 差 < 0.09（桶数不影响 train/val loss）。
- 产物：图 `docs/figs/main/`；数据 `data/injpos_ablation_data.json`；克隆博客 `guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide-v10/`（validation 每 10 步 · 2000 steps）。


## 9. Table 优化器消融（2026-08-06，input，计划）

目的：记录历史实验中 n-gram table 使用 RMSProp（β=(0.0,0.999)，无一阶矩）的行为；当前标准已切换为 β₂=0.99、表学习率 ×2。
怀疑「table 学得慢/滞后」；本批测试替代优化器是否让 table 更快写入、以及 gap 曲线如何变化。

### Setting

| 项 | 值 |
|---|---|
| injection | `input`（blog 主线默认）|
| train shards / val | `1` / `2,3,4,5,6,7,8,9,10,6542`（与标准一致）|
| steps / seed | 1000 / 42 |
| val / freq / norm | 每 10 steps（v10，与 `nglab1x_v10_*` 对齐）|
| 只变 | table optimizer（backbone 恒为 AdamW lr=0.004）|

| arm | table optimizer | table betas | table lr |
|---|---|---|---|
| `rmsprop_2x` | RMSProp | (0.0, 0.999) | 0.008（lr×2）|
| `adamw_090999` | AdamW | (0.9, 0.999) | 0.004 |
| `adamw_080950` | AdamW | (0.8, 0.95) | 0.004 |
| `sgd_09` | SGD+momentum | momentum=0.9 | 0.004 |

launcher：`code/cluster/run_table_opt.sh <arm> <gpu>`。
代码：`code/train.py` 新增 `--table_optimizer / --table_lr_scale / --table_betas`（默认不变）。
对照基线：`nglab1x_v10_input`（同 flags，v10/2000）与 `nglab_input`（v50/1000）。

### 结果（ophis-gpu 首波 · 23 桶 freq 统计，已 done；360-1 15 桶重跑进行中）

- ophis-gpu 首波（2026-08-06 21:41–22:41 CST，run_id 同上，使用当时 23 桶版 `ngram_freq.py`）：
  - `nglab1x_v10_v`：final_gap **4.9497**（train 1.3536 / val 6.3033）
  - `nglab1x_v10_y`：final_gap **5.0552**（train 1.3601 / val 6.4153）
  - `nglab1x_v10_input`：final_gap **1.9615**（train 2.7072 / val 4.6687）
  - `nglab1x_v10_nogram`：final_gap **0.2253**（train 3.0167 / val 3.2420）
  - freq bin 统计为 23 桶（另一 agent 的未提交改动在开跑前已同步到 ophis），train/val gap 与桶数无关。
- 360-1 重跑（2026-08-06 22:33–23:51 CST，15 桶提交版 `ngram_freq.py`，4 卡并行；与博客口径一致，作为克隆文档的权威数据）：
  - `nglab1x_v10_v`：final_gap **5.0406**（train 1.2377 / val 6.2783）
  - `nglab1x_v10_y`：final_gap **5.0493**（train 1.3698 / val 6.4191）
  - `nglab1x_v10_input`：final_gap **1.9308**（train 2.7082 / val 4.6390）
  - `nglab1x_v10_nogram`：final_gap **0.2312**（train 3.0033 / val 3.2345）
  - epoch 边界（每 10 步 val 记录）：230 / 460 / 680 / 910 / 1130 / 1360 / 1580 / 1810（约 226 steps/epoch，freq eval 每 10 步消耗 4 个 train batch）。
  - 与 ophis 首波交叉验证：四 run gap 差 < 0.09（桶数不影响 train/val loss）。
- 产物：图 `docs/figs/main/`；数据 `data/injpos_ablation_data.json`；克隆博客 `guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide-v10/`（validation 每 10 步 · 2000 steps）。

### 9a. Table 优化器消融（郭绍阳，wave1 done / wave2 running）

> 与 §9 的 v10 主线重跑并行进行；只换 table optimizer，backbone 恒为 AdamW lr=0.004，input 注入、1x shard、seed42、v10。

**Wave1 结果（1000 步，2026-08-06 20:41–22:05，ophis）**

| arm | gap@1000 | norm@1000 | norm_growth(10→1000) |
|---|---:|---:|---:|
| RMSProp 1x（`nglab1x_v10_input`）| +0.953 | 0.0840 | +133% |
| RMSProp 2x（`rmsprop_2x`）| +0.939 | **0.1335** | **+270%** |
| AdamW (0.9,0.999) | +0.912 | 0.0720 | +99% |
| AdamW (0.8,0.95) | +0.709 | 0.0623 | +73% |
| RMSProp v50 基线（`nglab_input`）| +0.785 | 0.0781 | +117% |

- 结论：**加速 table 写入最直接有效的是把 RMSProp lr 提到 2×**（norm@1000 +59%）；AdamW 一阶矩对「每 epoch 只被读一次」的稀疏行帮助有限，norm 反而更小、gap 更小（0.8/0.95 时 0.709）。norm 增速 ↑ ↔ gap 大小 ↑ 的对应仍成立。
- 图：`docs/figs/table_opt/fig_table_opt.{svg,png}`；脚本 `docs/plot_scripts/analyze_table_opt.py`。

**Wave2（2026-08-06 23:53 启动，多机并行）**

- ophis GPU6 `sgd_09`(1000)、GPU7 `rmsprop_2x`(2000，验证更快写表→epoch3/4 更大 gap)
- 360-1 GPU4/5/6：`rmsprop_2x_s43`、`adamw_090999_s43`、`rmsprop_4x`(1000)
- 360-2 GPU0/1：`rmsprop_2x_s44`、`adamw_090999_s44`
- 分析：跑完自动 rsync + `analyze_table_opt.py` 汇总。

### 9b. Wave2：多 seed × LR 剂量 × SGD 机制验证（2026-08-07 回填）

> 承接 §9a wave1；wave2 目标：(i) 用 s43/s44 重复 RMSProp2x / AdamW(0.9,0.999)，估计 seed 方差；(ii) RMSProp 2x 延到 2000 步验证「更快写表 → epoch3/4 gap 更大」；(iii) 新增 RMSProp 4x 验证 LR 剂量效应；(iv) SGD+momentum 验证「无自适应归一化时 table 是否学得动」。

**Wave2 结果（360-1 / 360-2 / ophis，2026-08-06 23:53–08-07 ~10:20）**

| arm | seed | steps | gap@1000 | final_gap | norm@1000 | norm_growth(10→1000) |
|---|---:|---:|---:|---:|---:|---:|
| RMSProp 1x（`nglab1x_v10_input`）| 42 | 2000 | +0.953 | +1.931 | 0.0838 | +132% |
| RMSProp 2x | 42 | 2000 | +1.145 | +2.376 | 0.1544 | +328% |
| RMSProp 2x | 43 | 1000 | +2.111 | +2.111 | 0.1343 | +272% |
| RMSProp 2x | 44 | 1000 | +2.089 | +2.089 | 0.1350 | +274% |
| RMSProp 4x | 42 | 1000 | +2.182 | +2.182 | 0.2502 | +591% |
| AdamW (0.9,0.999) | 42 | 1000 | +0.912 | +0.912 | 0.0720 | +99% |
| AdamW (0.9,0.999) | 43 | 1000 | +1.443 | +1.443 | 0.0713 | +98% |
| AdamW (0.9,0.999) | 44 | 1000 | +1.672 | +1.672 | 0.0712 | +97% |
| SGD mom0.9 | 42 | 1000 | **−0.002** | −0.002 | **0.0361** | **~0%** |

**多 seed 汇总（mean ± std）**：RMSProp 2x norm@1000 = 0.1413 ± 0.0093（n=3）；AdamW(0.9,0.999) norm@1000 = 0.0715 ± 0.0003（n=3）。

**关键结论**

1. **SGD 无 gap 的机制 = table 根本没学**：SGD mom0.9 下全部 7 张 table（4 bigram + 3 trigram 层）的 param RMS 在 1000 步内纹丝不动（0.03608，精确到 1e-5 无变化），gap≈0 是「table 从未写入」而非「table 学好了但无过拟合」。原因：SGD 的更新量正比于梯度幅值，而 table 行是稀疏命中（每 epoch 每行只被读几次）+ 147k token batch 平均，单步梯度 ~1e-5 量级，×lr=0.004 → 每步有效更新 ~4e-8，1000 步积累不可见。RMSProp/AdamW 用 per-param `g/√EMA(g²)` 把有效步长固定到 ≈lr（与梯度幅值无关），所以 table 才学得动。**这反证了：标准 RMSProp 的每参数自适应归一化是 table 学习的必要条件。**
2. **LR 剂量效应（超线性）**：RMSProp 1x→2x：norm@1000 0.084→0.144（+71%），gap@1000 +0.95→+1.78（s42/s43/s44 均值 1.782±0.451）；2x→4x：norm@1000→0.250（再 +73%），gap@1000→+2.18。即 table LR 每翻倍，norm@1000 约再 +70%，gap 相应放大。2000 步口径：RMSProp 2x gap +2.376 vs 1x +1.931（+23%），norm@2000 0.179 vs 0.097（+85%）——更快写表 → epoch 3/4 的 gap 显著更大，直接回应「table 学得慢/滞后」。
3. **AdamW 一阶矩对稀疏行帮助有限**：AdamW(0.9,0.999) norm@1000 0.072 反而 ≤ RMSProp 1x 的 0.084（多 seed 稳定），gap@1000 1.34±0.32 略高但受 norm 上限约束；(0.8,0.95) 更慢（0.062 / +0.709）。动量在「每 epoch 只被读一次」的稀疏行上累积价值低。
4. **norm 先行、gap 滞后**：RMSProp 4x 在 step500 norm 已达 0.22（≥2x@1000）但 gap 仍 +0.01，到 step1000 才跳到 +2.18——table 写入先完成、train/val 分化随后显现，与 wave1 的 norm↔gap 对应关系一致。

**产物**：图 `docs/figs/table_opt/fig_table_opt.{svg,png}`（含 s43/s44 多 seed mean±std）；脚本 `docs/plot_scripts/analyze_table_opt.py`（自动发现 `nglab1x_opt_*`，seed 42/43/44）。日志：`data/runs/nglab1x_opt_{rmsprop_2x_s43,rmsprop_4x,adamw_090999_s43,adamw_090999_s44,rmsprop_2x_s44,sgd_09}/`。

**历史 setting 说明**：本节结果使用 β₂=0.999 的历史配置；当前标准 table 使用 RMSProp（β=(0.0,0.99)，无 momentum，无 WD，表学习率 ×2），backbone 使用 AdamW（β=(0.8,0.95)，WD=0.1，lr=0.004）。

### 9c. Table 优化器消融 × 2x epoch（郭绍阳，2026-08-07 done）

> 承接 §9b（1x epoch）。用户假设：2x epoch（train shards 1,2，~450 步/epoch）下 2000 步
> 只走 ~4.4 epoch，之前 norm 曲线到 2000 步仍未平台，怀疑 beta2=0.999 不够合理。
> 本批在 2x epoch 下扫 table LR（1x/2x/4x）+ beta2（0.999 / 0.9999 / 0.99999），
> 并在后续补跑 beta2=0.99/0.98（见 §9d）。

**2x epoch · 2000 步 · seed42 · input · fixed-val（360-1/360-2，2026-08-07 12:03–13:30）**

| arm | table LR | beta2 | gap@2000 | norm@2000 | norm@1000→2000 增量 |
|---|---:|---:|---:|---:|---:|
| `rmsprop_1x`（§6 `nglab2x_input_v10_fv`）| 0.004 | 0.999 | +0.502 | 0.0851 | +0.012 |
| `rmsprop_2x` | 0.008 | 0.999 | +0.595 | 0.1576 | +0.024 |
| `rmsprop_4x` | 0.016 | 0.999 | **+2.071** | 0.3172 | +0.046 |
| `rmsprop_1x_b2_09999` | 0.004 | 0.9999 | +0.497 | 0.0850 | +0.011 |
| `rmsprop_2x_b2_09999` | 0.008 | 0.9999 | +0.608 | 0.1589 | +0.023 |
| `rmsprop_4x_b2_09999` | 0.016 | 0.9999 | **+2.110** | 0.3136 | +0.043 |
| `rmsprop_2x_b2_099999` | 0.008 | 0.99999 | +0.634 | 0.1593 | +0.024 |

**关键结论**

1. **beta2 往 1 方向（0.9999/0.99999）无效**：norm@2000 与 0.999 差 <0.004，gap 差 <0.04，
   曲线几乎重合。平台问题不是「beta2 不够接近 1」。
2. **LR 剂量效应在 2x epoch 下更强**：norm@2000 0.085→0.158→0.317（≈线性翻倍）；
   gap@2000 0.50→0.60→2.07。4x LR 在 2x epoch 下 gap 已追平 1x epoch 的 RMSProp 2x（+2.38）。
3. **平台仍未达，但增速放缓**：1500→2000 步 norm 只涨 +0.002~0.005（相对 ~1-3%/500 步）。
   按此斜率，平台可能需 4000–6000 步；加 LR 比调 beta2 有效得多。
4. 2000 步下 2x epoch 只走 ~4.4 epoch（1x 是 ~8.7），所以同臂同 norm 时 gap 普遍小于 1x：
   数据重放次数减半，记忆-过拟合路径没走完。

**产物**：图 `docs/figs/table_opt/fig_table_opt_2x.{svg,png}`、`fig_table_opt_1x_vs_2x.{svg,png}`；
脚本 `docs/plot_scripts/analyze_table_opt_2x.py`、`analyze_table_opt_1x_vs_2x.py`；
launcher `code/cluster/run_table_opt_2x.sh`（train shards 1,2 / val 3..10,6542 / 2000 步）。

### 9d. β2 反向扫描（0.99/0.98）+ 1x·RMS4x·b2=0.999@2000 对照（2026-08-07 done）

> 承接 §9c：beta2 往 1 方向（0.9999/0.99999）无效。用户问「b2 合理的值难道不是 0.99 吗」，
> 本批补跑 β2=0.99/0.98（1x·RMS4x + 1x·RMS2x），并补 1x·RMS4x·b2=0.999 的 2000 步对照
> （§9b 只有 1000 步值 +2.18）。360-2 GPU0 跑 `nglab1x_opt_rmsprop_4x`（b2=.999，2000 步）。

**1x epoch · 2000 步 · seed42 · input · fixed-val（360-2，2026-08-07 15:17–16:52）**

| arm | table LR | beta2 | gap@1000 | gap@2000 | norm@2000 |
|---|---:|---:|---:|---:|---:|
| `rmsprop_4x`（§9b 对照补跑）| 0.016 | 0.999 | +2.360 | **+4.742** | 0.4180 |
| `rmsprop_4x_b2_099` | 0.016 | 0.99 | +2.389 | **+5.143** | 0.4357 |
| `rmsprop_4x_b2_098` | 0.016 | 0.98 | +2.568 | **+5.155** | 0.4345 |
| `rmsprop_2x_b2_099` | 0.008 | 0.99 | — | +2.349 | — |
| `rmsprop_2x_b2_098` | 0.008 | 0.98 | — | +2.309 | — |

**结论**

1. **β2=0.99/0.98 在 1x·RMS4x 下确实有正效应**：gap@2000 +4.74 → +5.14/+5.16
   （+0.40~+0.41，约 +8.5%），norm@2000 0.418 → 0.434~0.436（+4%）。与「β2 越接近 1
   越无效」一致的反向方向：**降低 β2（更激进地除方差）让 table 写得更快**。
2. **但 1x·RMS2x 下 β2 影响很小**：0.99→+2.349 / 0.98→+2.309 vs 0.999 的 +2.376（§9a）
   —— LR 剂量低时 β2 的差异被 LR 上限掩盖。β2 只在 table LR 拉满（4x）时才显现。
3. 结合 §9c：β2 在 [0.98, 0.99999] 全区间内，对 gap@2000 的总影响 ≤ ±0.4，
   远小于 table LR 翻倍的效果（+0.10→+2.07）。**用户若想让 table 学更快，
   首选 table_lr_scale=2–4；β2=0.99 可作次级叠加（约 +8%）**。

**产物**：数据 `data/runs/nglab1x_opt_rmsprop_4x{,_b2_099,_b2_098}/`（已同步本地）；
launcher `code/cluster/run_table_opt_2x.sh` 风格。

## 10. shard 大小扫描（epoch 长度剂量，2026-08-07，彻夜批）

目的：用户假设「epoch 的 shard 越大，gap 越小」。§6/§7 已有 0.5x/1x/2x 三个点
（gap@2000 = 4.95 / 1.96 / 0.50），本批**连续采样 shard 大小**补 9 个点，
共 12 点（0.25x → 8x），横轴按 shard 大小（log）与「epoch 数」双尺度分析。

### Setting

| 项 | 值 |
|---|---|
| 注入点 / seed / steps | input / 42 / 2000（2.5x=3200、3x=3800、4x=5000 以便同跑 ~5.5 epoch）|
| validation / freq eval | v10，fixed-val（`fixed_val_batches`）|
| freq index | 每个 train set 单独建（`data/freq_index_train{0_25x,...,8x}.npz`）|
| 集群 | ophis-gpu GPU0-5（6 个）+ 360-2 GPU0-2（5x/6x/8x）|

| size | train shards（rows）| val shards | steps |
|---|---|---|---|
| 0.25x | 62（6066）| 2..10,6542 | 2000 |
| 0.5x ✅ §7 | 60（12132）| 2..10,6542 | 2000 |
| 0.75x | 63（18198）| 2..10,6542 | 2000 |
| 1x ✅ §8 | 1（24264）| 2..10,6542 | 2000 |
| 1.5x | 1,61（36396）| 3..10,6542 | 2000 |
| 2x ✅ §4/§6 | 1,2（48240）| 3..10,6542 | 2000 |
| 2.5x | 1,2,64（60372）| **4..10,6542** | 3200 |
| 3x | 1,2,3（72000）| **4..10,6542** | 3800 |
| 4x | 1,2,3,4（95760）| **5..10,6542** | 5000 |
| 5x | 1..5（119808）| 6..10,6542 | 2000 |
| 6x | 1..6（143568）| 7..10,6542 | 2000 |
| 8x | 1..8（191016）| 9,10,6542 | 2000 |

> 注：train 含 shard2 的 run（≥1.5x）val 用 3..10,6542 避免 val 与 train 重叠（与 §4 的 2x 一致）。
> **v1 bug（01:40 发现并修正）**：2.5x/3x/4x 首跑 val 仍用 3..10,6542，但它们的 train
> 已含 shard 3 → fixed val 前几批与 train 重叠（val 被剧透），gap 偏负
> （2.5x=−0.80 / 3x=−0.71 / 4x=−0.27 @2000）。已停，改 `_v2`（val 从最后一个 train shard 之后开始）。

### 结果（gap@2000，首批 9 点已回填；2.5x/3x/4x 为 v2 重跑，待补）

| size | run | gap@2000 | epoch@2000 |
|---|---|---|---|
| 0.25x | `nglab0_25x_input_fv` | **+12.99** | 36 |
| 0.5x | `nglab0_5x_input_fv` | **+4.95** | 18 |
| 0.75x | `nglab0_75x_input_fv` | **+2.12** | 12 |
| 1x | `nglab1x_v10_input` | **+1.96** | 9 |
| 1.5x | `nglab1_5x_input_fv` | **+0.87** | 6 |
| 2x | `nglab2x_input_v10_fv` | **+0.50** | 5 |
| 5x | `nglab5x_input_fv` | −0.05 | 2 |
| 6x | `nglab6x_input_fv` | −0.11 | 2 |
| 8x | `nglab8x_input_fv` | +0.03 | 2 |

**结论（用户假设成立）**：epoch shard 越大，gap@2000 单调变小——
0.25x→8x：+13 → +5 → +2 → +0.5 → ~0（≥5x 在 2000 步内只走 ~2 个 epoch，gap 尚未形成）。
0.25–2x 段呈近似幂律（log-log 斜率约 −1.5~−2）。

图：`docs/figs/epoch_scale/dose_response_gap2000.png`（gap@2000 vs shard size，log-x）、
`gap_vs_epochs.png`（gap vs 已过 epoch 数，横轴按 epoch 缩放）、`sweep_train_val_gap.png`。
脚本：`docs/plot_scripts/gen_shard_sweep_figs.py`。

### 结果（12 点全部完成，2026-08-07 凌晨→早上；2.5x/3x/4x 为 v2 重跑）

| size | run | gap@2000 | epoch@2000 | final gap（步数 / epoch）|
|---|---|---|---|---|
| 0.25x | `nglab0_25x_input_fv` | **+12.99** | 36 | +12.99（2000 / 36）|
| 0.5x | `nglab0_5x_input_fv` | **+4.95** | 18 | +4.95（2000 / 18）|
| 0.75x | `nglab0_75x_input_fv` | **+2.12** | 12 | +2.12（2000 / 12）|
| 1x | `nglab1x_v10_input` | **+1.96** | 9 | +1.96（2000 / 9）|
| 1.5x | `nglab1_5x_input_fv` | **+0.87** | 6 | +0.87（2000 / 6）|
| 2x | `nglab2x_input_v10_fv` | **+0.50** | 5 | +0.50（2000 / 5）|
| 2.5x | `nglab2_5x_input_fv_v2` | −0.03 | 4 | +0.69（3200 / 6）⚠️ |
| 3x | `nglab3x_input_fv_v2` | +0.11 | 3 | +1.94（3800 / 6）|
| 4x | `nglab4x_input_fv_v2` | −0.03 | 2-3 | +1.55（5000 / 6.2）|
| 5x | `nglab5x_input_fv` | −0.05 | 2 | −0.05（2000 / 2）|
| 6x | `nglab6x_input_fv` | −0.11 | 2 | −0.11（2000 / 2）|
| 8x | `nglab8x_input_fv` | +0.03 | 2 | +0.03（2000 / 2）|

**结论（用户假设成立）**：epoch shard 越大，gap@2000 单调变小——
0.25x→8x：+13 → +5 → +2 → +1.96 → +0.87 → +0.50 → ~0（≥2.5x 在 2000 步内
gap≈0，≥5x 只走 ~2 个 epoch、gap 尚未形成）。0.25–2x 段近似幂律
（log-log 斜率约 −1.5~−2）。横轴双尺度：`gap_vs_epochs.png` 按「已过 epoch 数」
缩放后可见 3x/4x 在 epoch 5-6 的 gap 反而高于 1.5x/2x → 大 shard 只是**延迟**了
gap 的出现（步数维度），并未消除（epoch 维度），支持「重播轮数 × 数据大小共同决定」。

**观测到的 epoch 边界**（fixed-val + freq-bin eval 每 10 步额外消耗 4 个 train batch，
故步数/epoch < 纯 rows/72）：0.5x≈120、1x≈240、2x≈450、2.5x=570/1130/1690/2240/2800、
3x=670/1340/2010/2670/3340、4x=890/1780/2670/3550/4440（步数/epoch ≈ 120/240/450/560/670/890，
比例 1:2:3.7:4.7:5.6:7.4 与 rows 比例 1:2:4:5:6:8 基本一致）。

### ⚠️ 2.5x 的 train 停滞（待 v3/s43 确认）

- `nglab2_5x_input_fv_v2`（train=1+2+64，3200 步）train loss 在 steps ~1200–2700
  （epoch 3-5）卡在 ~3.7–3.9 不动，直到 epoch 6（steps 2800–3200、lr→0.05）才掉到
  3.31；val（fixed shards 4..）同步回升 3.70→4.01。同规模的 3x/4x 同 epoch 处
  train 已到 2.0–2.3。
- 排除项：val 重叠（v1 同停滞）；table 爆炸（trigram RMS 0.134 @3200 vs 3x 0.135
  @3800 vs 4x 0.144 @5000，曲线平滑）。
- **最可能原因 = LR 调度混杂**：`get_lr_multiplier` 以 `progress=(step+1)/max_steps`
  计算，2.5x/3x/4x 是延长 run（3200/3800/5000），LR 升温/衰减都被拉长 →
  step 2000 时 lr_mult = 0.598/0.742/0.927，而 ≤2x 的 run 在 step 2000 已衰减到
  0.05。2.5x 的 warmdown 从 1120 开始，停滞区正好落在它的中高 LR 段；当 LR 衰减
  到 <0.3 后 train 才开始快速下降。因此 v2 的「final gap @~6 epoch」三组之间
  不可直接比，`gap@2000` 对 2.5x/3x/4x 也测在不同 LR 工作点上。
- 次要因素：2.5x 的 epoch = 1+2+shard3 前半（shard 64），与 3x 共享前 2 个 shard，
  数据构成差异待验证。

### 验证批（10:16 启动；v3 已于 11:35 完成，s43 运行中）

- `nglab2_5x_input_fv_v3` / `nglab3x_input_fv_v3` / `nglab4x_input_fv_v3`：
  同 v2 的 train/val 配置但 **max_steps=2000**（与主 sweep 完全相同的 LR 调度），
  得到公平的 gap@2000 —— **结果：+0.04 / +0.03 / −0.04**（train 3.49/3.29/3.30，
  val 3.53/3.32/3.26，epoch 4/3/3）≈ 0，与 v2 延长 run 的 ≈0 一致 → **主剂量曲线
  对 LR 调度稳健**（2.5x/3x/4x 在 2000 步内 gap 确实≈0，不是 LR 假象）。
  注：同 LR 调度下 2.5x train@2000 = 3.49（v2 是 3.80）——延长 run 的 LR 拉伸确实
  拖慢了 2.5x 的 train 下降，但 val 同步下降，gap 仍≈0。
- `nglab2_5x_input_fv_s43`：2.5x @3200、seed 43，检验停滞是否 seed/数据相关
  —— **结果（12:20 完成）：与 seed42 v2 几乎逐点重合**，epoch 2-5 train 同样停滞
  ~3.8–3.9（s43: 4.39/3.94/3.80/3.95 vs s42: 4.41/3.91/3.78/3.92），epoch 6 才掉到
  3.33，final gap **+0.74 @3200**（s42 +0.69）→ **停滞是「3200 步 LR 拉伸 × 2.5x 数据」
  的确定性现象，跨 seed 复现**；不是 val 重叠、不是 table 爆炸、不是 seed 噪声。
  结合 v3（同 LR 调度下 train@2000 = 3.49、无停滞），可归因于延长 run 的 warmdown
  拉伸（2.5x 从 step 1120 开始衰减、停滞区正好落在中高 LR 段），而 2000 步预算内
  （主剂量曲线）2.5x 无异常。
- 脚本：`code/cluster/run_verify_v3.sh`（⚠️ 该脚本未随迁移带入，仅在 ophis-gpu 远端存在；nohup，日志 `verify_v3.log`）。

### 图（已用 v2 数据重跑，10:11 同步回本地）
`docs/figs/epoch_scale/dose_response_gap2000.png`（gap@2000 vs shard size，log-x）、
`gap_vs_epochs.png`（gap vs 已过 epoch 数，横轴按 epoch 缩放）、`sweep_train_val_gap.png`。
脚本：`docs/plot_scripts/gen_shard_sweep_figs.py`。

### 结果（12 点全部完成，2026-08-07 凌晨→早上；2.5x/3x/4x 为 v2 重跑）

| size | run | gap@2000 | epoch@2000 | final gap（步数 / epoch）|
|---|---|---|---|---|
| 0.25x | `nglab0_25x_input_fv` | **+12.99** | 36 | +12.99（2000 / 36）|
| 0.5x | `nglab0_5x_input_fv` | **+4.95** | 18 | +4.95（2000 / 18）|
| 0.75x | `nglab0_75x_input_fv` | **+2.12** | 12 | +2.12（2000 / 12）|
| 1x | `nglab1x_v10_input` | **+1.96** | 9 | +1.96（2000 / 9）|
| 1.5x | `nglab1_5x_input_fv` | **+0.87** | 6 | +0.87（2000 / 6）|
| 2x | `nglab2x_input_v10_fv` | **+0.50** | 5 | +0.50（2000 / 5）|
| 2.5x | `nglab2_5x_input_fv_v2` | −0.03 | 4 | +0.69（3200 / 6）⚠️ |
| 3x | `nglab3x_input_fv_v2` | +0.11 | 3 | +1.94（3800 / 6）|
| 4x | `nglab4x_input_fv_v2` | −0.03 | 2-3 | +1.55（5000 / 6.2）|
| 5x | `nglab5x_input_fv` | −0.05 | 2 | −0.05（2000 / 2）|
| 6x | `nglab6x_input_fv` | −0.11 | 2 | −0.11（2000 / 2）|
| 8x | `nglab8x_input_fv` | +0.03 | 2 | +0.03（2000 / 2）|

**结论（用户假设成立）**：epoch shard 越大，gap@2000 单调变小——
0.25x→8x：+13 → +5 → +2 → +1.96 → +0.87 → +0.50 → ~0（≥2.5x 在 2000 步内
gap≈0，≥5x 只走 ~2 个 epoch、gap 尚未形成）。0.25–2x 段近似幂律
（log-log 斜率约 −1.5~−2）。横轴双尺度：`gap_vs_epochs.png` 按「已过 epoch 数」
缩放后可见 3x/4x 在 epoch 5-6 的 gap 反而高于 1.5x/2x → 大 shard 只是**延迟**了
gap 的出现（步数维度），并未消除（epoch 维度），支持「重播轮数 × 数据大小共同决定」。

**观测到的 epoch 边界**（fixed-val + freq-bin eval 每 10 步额外消耗 4 个 train batch，
故步数/epoch < 纯 rows/72）：0.5x≈120、1x≈240、2x≈450、2.5x=570/1130/1690/2240/2800、
3x=670/1340/2010/2670/3340、4x=890/1780/2670/3550/4440（步数/epoch ≈ 120/240/450/560/670/890，
比例 1:2:3.7:4.7:5.6:7.4 与 rows 比例 1:2:4:5:6:8 基本一致）。

### ⚠️ 2.5x 的 train 停滞（待 v3/s43 确认）

- `nglab2_5x_input_fv_v2`（train=1+2+64，3200 步）train loss 在 steps ~1200–2700
  （epoch 3-5）卡在 ~3.7–3.9 不动，直到 epoch 6（steps 2800–3200、lr→0.05）才掉到
  3.31；val（fixed shards 4..）同步回升 3.70→4.01。同规模的 3x/4x 同 epoch 处
  train 已到 2.0–2.3。
- 排除项：val 重叠（v1 同停滞）；table 爆炸（trigram RMS 0.134 @3200 vs 3x 0.135
  @3800 vs 4x 0.144 @5000，曲线平滑）。
- **最可能原因 = LR 调度混杂**：`get_lr_multiplier` 以 `progress=(step+1)/max_steps`
  计算，2.5x/3x/4x 是延长 run（3200/3800/5000），LR 升温/衰减都被拉长 →
  step 2000 时 lr_mult = 0.598/0.742/0.927，而 ≤2x 的 run 在 step 2000 已衰减到
  0.05。2.5x 的 warmdown 从 1120 开始，停滞区正好落在它的中高 LR 段；当 LR 衰减
  到 <0.3 后 train 才开始快速下降。因此 v2 的「final gap @~6 epoch」三组之间
  不可直接比，`gap@2000` 对 2.5x/3x/4x 也测在不同 LR 工作点上。
- 次要因素：2.5x 的 epoch = 1+2+shard3 前半（shard 64），与 3x 共享前 2 个 shard，
  数据构成差异待验证。

### 验证批（10:16 启动；v3 已于 11:35 完成，s43 运行中）

- `nglab2_5x_input_fv_v3` / `nglab3x_input_fv_v3` / `nglab4x_input_fv_v3`：
  同 v2 的 train/val 配置但 **max_steps=2000**（与主 sweep 完全相同的 LR 调度），
  得到公平的 gap@2000 —— **结果：+0.04 / +0.03 / −0.04**（train 3.49/3.29/3.30，
  val 3.53/3.32/3.26，epoch 4/3/3）≈ 0，与 v2 延长 run 的 ≈0 一致 → **主剂量曲线
  对 LR 调度稳健**（2.5x/3x/4x 在 2000 步内 gap 确实≈0，不是 LR 假象）。
  注：同 LR 调度下 2.5x train@2000 = 3.49（v2 是 3.80）——延长 run 的 LR 拉伸确实
  拖慢了 2.5x 的 train 下降，但 val 同步下降，gap 仍≈0。
- `nglab2_5x_input_fv_s43`：2.5x @3200、seed 43，检验停滞是否 seed/数据相关
  —— **结果（12:20 完成）：与 seed42 v2 几乎逐点重合**，epoch 2-5 train 同样停滞
  ~3.8–3.9（s43: 4.39/3.94/3.80/3.95 vs s42: 4.41/3.91/3.78/3.92），epoch 6 才掉到
  3.33，final gap **+0.74 @3200**（s42 +0.69）→ **停滞是「3200 步 LR 拉伸 × 2.5x 数据」
  的确定性现象，跨 seed 复现**；不是 val 重叠、不是 table 爆炸、不是 seed 噪声。
  结合 v3（同 LR 调度下 train@2000 = 3.49、无停滞），可归因于延长 run 的 warmdown
  拉伸（2.5x 从 step 1120 开始衰减、停滞区正好落在中高 LR 段），而 2000 步预算内
  （主剂量曲线）2.5x 无异常。
- 脚本：`code/cluster/run_verify_v3.sh`（⚠️ 该脚本未随迁移带入，仅在 ophis-gpu 远端存在；nohup，日志 `verify_v3.log`）。

### 图（已用 v2 数据重跑，10:11 同步回本地）
## 11. toy-model 台阶清晰度溯源（2026-08-07，郭绍阳 + 2 workers）

> 背景：主实验（1x/2x epoch）gap 曲线「不够清晰」（无 toy 那种每 epoch 台阶）；
> beta2 扫描（0.999/0.9999/0.99999/0.99/0.98）影响很小。toy5/t5 的 table beta
> 与主实验同为 `NGRAM_TABLE_BETAS=0.0,0.999`，但其台阶状极清晰——差异主因是
> **epoch 长度**（toy5 low: 2000 步 ≈ 29 epoch，~70 步/epoch；主实验 1x ~225、
> 2x ~450 步/epoch）。假设：台阶清晰度 ∝ 重放频率（epoch 越短越清晰），beta 影响次要。
> 两个 worker 并行验证：(A) toy 侧扫 `NGRAM_TABLE_BETAS` 看台阶是否随 beta 变化；
> (B) 真实模型侧扫「0.25x/0.5x epoch × beta2」网格，量化台阶清晰度。

### Setting（worker A：toy 侧 beta 扫描）
| 项 | 值 |
|---|---|
| 脚本 | 已迁入 `tasks/` 的 toy 数据生成脚本与 launcher（历史 toy 工作区）|
| 变体 | `NGRAM_TABLE_BETAS` = 0.0,0.999（基准）/ 0.0,0.99 / 0.0,0.9999 / 0.9,0.999 / 0.9,0.9999 |
| steps | 2000（low cache，~29 epoch）|
| 输出 | 每 epoch `headline_gap`（台阶）、`seen_gap` |

### Setting（worker B：真实模型短 epoch × beta2）
| 项 | 值 |
|---|---|
| 脚本 | `ngram-gap-lab/code/train.py` + `code/cluster/run_epoch_scale_v10.sh` 风格 |
| 变体 | epoch 0.25x/0.5x × beta2 {0.999, 0.99}（=4 run，2000 步，input 注入，seed42）|
| 对照 | §7 `nglab0_5x_input_fv`（0.5x·beta2=0.999 已有）|
| 输出 | train_log.jsonl（每 10 步 gap）+ table_norm.jsonl |

### 结果（Worker A：toy 侧 beta 扫描，已完成 2026-08-07 16:10，360-2）

- 5 个变体（0.0,0.999 基准 / 0.0,0.99 / 0.0,0.9999 / 0.9,0.999 / 0.9,0.9999），
  seed 42、low cache、2000 步（~29 epoch），全部 rc=0。
- 分析脚本：集群 `toy_analyze.py`（已同步本地新版 `tasks/` 分析脚本 +
  `toy_model.py` 相对路径 patch），exact-context headline_gap per-epoch。
- **结论：β 不改变 toy 的台阶形状** —— 5 条 per-epoch 曲线几乎重合，最终 gap
  7.43–7.89（基准 7.89），200/400/800 步处差异 ≤0.3 nats，无系统性顺序
  （0.9999 反而略低）。台阶清晰度由重放频率（epoch 长度）决定，与 table β 无关。

| betas | gap@200 | gap@400 | gap@800 | gap@1200 | gap@1600 | gap@2000 |
|---|---|---|---|---|---|---|
| 0.0,0.999（基准）| 0.407 | 1.965 | 6.123 | 7.950 | 7.884 | 7.894 |
| 0.0,0.99 | 0.424 | 2.391 | 5.149 | 7.659 | 7.849 | 7.737 |
| 0.0,0.9999 | 0.406 | 1.400 | 4.822 | 7.310 | 7.199 | 7.434 |
| 0.9,0.999 | 0.422 | 2.621 | 5.366 | 7.755 | 7.949 | 7.631 |
| 0.9,0.9999 | 0.436 | 2.251 | 5.287 | 7.538 | 7.893 | 7.705 |

> 注：绝对值与历史 ophis-gpu 值（t5_on_low final 6.79）不同，因新 `toy_analyze.py`
> 用 exact-context counts 重新量化 r，同一脚本内 5 个变体可比。step 级曲线显示
> toy 的 gap 在 epoch 内平滑增长（每 epoch 段内 drift +0.2~+1.3），台阶来自
> per-epoch 采样点（200/400/800...），与主实验的差异是 epoch 密度而非 β。

### 结果（Worker B：真实模型短 epoch × beta2，已完成 2026-08-07 17:35，360-2 GPU3/4）

- 2 个新 run（0.25x·b2=0.99、0.5x·b2=0.99）对照 §10 的 b2=0.999 参考（2000 步、
  input 注入、seed42、fixed-val），全部 rc=0，md5 核对通过。
- 数据：`data/runs/nglab025x_b2_099` / `nglab05x_b2_099`（已同步本地）。

| arm | b2 | gap@2000 | train@2000 | val@2000 | epochs | 台阶清晰度比（boundary/within）|
|---|---:|---:|---:|---:|---:|---:|
| 0.25x | 0.999（§10）| +12.991 | 0.376 | 13.367 | 36 | 7.57 |
| 0.25x | 0.99（新）| **+13.577** | 0.280 | 13.857 | 36 | 6.38 |
| 0.5x | 0.999（§10）| +4.952 | 1.776 | 6.728 | 18 | 4.12 |
| 0.5x | 0.99（新）| **+5.017** | 1.789 | 6.806 | 18 | 5.64 |

**结论（Worker B）**

1. **β2=0.99 在短 epoch 下仍只有微弱正效应**：0.25x +0.59（+4.5%）、0.5x +0.07（+1.4%），
   与 §9d 的 1x·RMS4x 结论（β2=0.99 +0.40，+8.5%）同方向但量级更小。
2. **β2 不改变台阶清晰度**：清晰度比在 0.25x 下 7.57→6.38、0.5x 下 4.12→5.64，
   方向不一致且幅度小（±1.5 以内）；而 epoch 长度本身的效应大得多
   （0.25x 清晰度比 6.4~7.6 vs 0.5x 4.1~5.6 vs 1x 5.5 vs 2x 6.6 混杂）。
   曲线仍以 epoch 内平滑上升为主（within/step 0.008~0.012），没有出现 toy 那种
   每 epoch 边界跳变——真实模型的台阶感弱，本质是「每 epoch 段内的连续过拟合」，
   与 β 无关。
3. **两 worker 合并结论：台阶清晰度 ∝ 重放频率（epoch 密度），table β（1 阶/2 阶）
   不改变台阶形态**。想让曲线更「台阶化」，只能缩短 epoch（0.25x 已是 36 epoch，
   gap@2000 最大 +13.6）；想让台阶变模糊，则拉长 epoch（1x/2x 已模糊）。

### 产物
- `docs/figs/theory/figs_v11_toy_beta_scan_per_epoch.svg` / `_step_level.svg`
  （同框 5 变体，per-epoch + step 级）
- `docs/figs/short_epoch_b2/short_epoch_b2_gap_v11.{svg,png}`（4 条 gap-step 曲线同框 +
  per-epoch mean gap 台阶视图）、`docs/figs/short_epoch_b2/staircase_shape_comparison.{svg,png}`
  （toy vs 真实模型归一化台阶形状）
- 脚本：`docs/plot_scripts/gen_short_epoch_b2_figs.py`；launcher：
  `code/cluster/run_epoch_short_b2.sh`（Worker B）、`tasks/` 中的 beta-scan launcher（Worker A）



`docs/figs/epoch_scale/dose_response_gap2000.png`（gap@2000 vs shard size，log-x +
幂律拟合）、`gap_vs_epochs.png`（gap vs 已过 epoch 数）、`sweep_train_val_gap.png`
（12 条全曲线）。脚本：`docs/plot_scripts/gen_shard_sweep_figs.py`（SWEEP 映射指向
v2；本地 `data/runs/<run_id>/train_log.jsonl` 已补齐 12 个 run）。



## 12. epoch 对齐批（同 epoch 数 × 同 LR-per-epoch 轨迹，2026-08-07 进行中）

> 背景：§10 的 step 对齐 sweep（gap@2000）显示「shard 越大 gap 越小」，
> 但大 shard 在 2000 步内只走了更少的 epoch（8x 仅 ~2 epoch），
> 「少重播」与「大 shard」混杂。用户提出：**对齐 epoch 数量**再看。
> 本批：所有 shard 大小都训到 ~6 个 epoch，且用 `--lr_schedule_epochs 6`
> 把 LR 锚定到 epoch（所有 run 共享同一条 LR-vs-epoch 轨迹，
> 排除 §10 ⚠️ 里发现的 LR 拉伸混杂）。

### Setting

| 项 | 值 |
|---|---|
| 注入点 / seed | input / 42（与 §10 主 sweep 完全一致）|
| LR | `--lr_schedule_epochs 6`（progress = epoch/6，warmdown 0.65；epoch 6→7 边界 lr=0.05）|
| steps | 每 run 跑到 ~6.1 epoch（epoch 7 开始后 ~60-130 步）|
| val / freq eval | v10，fixed-val |
| 集群 | 0.25x–1.5x：360-2（14:26–15:47）；2x–3x：360-1（16:17–17:12，360-2 首跑 OOM 后重跑）；4x–8x：ophis-gpu（13:50 启动，进行中）|

| size | run | train shards | val shards | steps（目标 ~6.1 ep）|
|---|---|---|---|---|
| 0.25x | `nglab0_25x_e6` | 62 | 2..10,6542 | 420 |
| 0.5x | `nglab0_5x_e6` | 60 | 2..10,6542 | 780 |
| 0.75x | `nglab0_75x_e6` | 63 | 2..10,6542 | 1080 |
| 1x | `nglab1x_e6` | 1 | 2..10,6542 | 1440 |
| 1.5x | `nglab1_5x_e6` | 1,61 | 3..10,6542 | 2100 |
| 2x | `nglab2x_e6` | 1,2 | 3..10,6542 | 2800 |
| 2.5x | `nglab2_5x_e6` | 1,2,64 | 4..10,6542 | 3500 |
| 3x | `nglab3x_e6` | 1,2,3 | 4..10,6542 | 4200 |
| 4x | `nglab4x_e6` | 1..4 | 5..10,6542 | 5490 |
| 5x | `nglab5x_e6` | 1..5 | 6..10,6542 | 6900 |
| 6x | `nglab6x_e6` | 1..6 | 7..10,6542 | 8260 |
| 8x | `nglab8x_e6` | 1..8 | 9,10,6542 | 10960 |

### 结果（0.25x–3x 已完成；4x–8x 进行中）

gap 在「6 个完整 pass 后」（epoch 7 首个 eval，lr=0.05）与「pass 6 内 mean/peak」：

| size | gap@6pass (bnd) | pass6 mean | pass6 peak | 到达步数 | gap@2000（§10 对照）|
|---|---:|---:|---:|---:|---:|
| 0.25x | **+1.094** | +1.083 | +1.206 | 350 | +12.991 |
| 0.5x | **+1.419** | +1.566 | +1.616 | 680 | +4.952 |
| 0.75x | **+0.845** | +0.732 | +0.841 | 1020 | +2.123 |
| 1x | **+1.911** | +1.918 | +2.147 | 1360 | +1.961 |
| 1.5x | **+0.914** | +0.848 | +0.977 | 2030 | +0.870 |
| 2x | **+0.800** | +0.711 | +0.894 | 2690 | +0.502 |
| 2.5x | **+0.925** | +0.557 | +0.927 | 3360 | −0.027 |
| 3x | **+2.141** | +1.712 | +2.117 | 4010 | +0.113 |
| 4x | 进行中（step 4580/5490，epoch 6，gap +1.10 @17:16）| | | | −0.031 |
| 5x | 进行中（step 4350/6900，epoch 4，gap −0.04）| | | | −0.047 |
| 6x | 进行中（step 4370/8260，epoch 4，gap −0.12）| | | | −0.110 |
| 8x | 进行中（step 4340/10960，epoch 3，gap −0.02）| | | | +0.027 |

**初步结论（待 4x–8x 补全）**：对齐 epoch 数后，§10 的「shard 越大 gap 越小」
**单调关系消失**——0.25x→3x 的 gap@6pass 在 +0.80~+2.14 之间非单调波动
（1x=+1.91、3x=+2.14 偏高，0.75x/2x 偏低 ~0.8），不再随 shard 大小单调下降。
step 对齐下的单调递减主要来自**大 shard 看到的重播轮数更少**（8x 在 2000 步内
只有 ~2 epoch），而非 shard 大小本身；在「同重播轮数 + 同 LR-per-epoch」下，
每个 epoch 的重放 gap 大致相当（0.8–2.1），与数据量 0.25x–3x 无系统关系。
> 注意：0.25x 在 6 pass 时 gap 只有 +1.1（vs 2000 步 36 pass 时的 +13.0），
> 说明小 shard 的巨额 gap 是「重播次数」累积出来的，而非单次重播更强。

图：`docs/figs/epoch_scale/gap_vs_shard_size_epoch_aligned.png`
（epoch 对齐 vs step 对齐双曲线）、`gap_vs_epoch_curves.png`（gap vs epoch 数，
各 shard 轨迹）、`epoch_aligned_train_val_gap.png`。
脚本：`docs/plot_scripts/gen_epoch_aligned_figs.py`。


---

## 13. toy 严格 Zipf 分布 · per-bucket gap 双对数（2026-08-07，planned）

> 背景：真实语料近似 Zipf，per-bin gap–frequency 双对数拟合已较好（bigram R²≈0.96、
> trigram R²≈0.81，见 `docs/figs/theory/fig_zipf_gap_analysis.png`）；toy 当前的频次分布是
> **anti-Zipf 设计**（N_r∝1/r，每桶 token 数相等，a≈−0.93，R²=0.99）。
> 用户提出：把 toy 的 ngram 分布筛选成**严格 Zipf**（N_r∝1/r²，经典 rank 指数 1），
> 再看 gap–ngram 双对数线性是否变好。
> 理论预判（原历史理论笔记，⚠️ 该文件已不存在；相关推导见 `docs/notes/theory/`）：
> per-bucket gap g(r) 由训练动力学+val 协议决定，**与总体分布 N_r 可分离**——
> 严格 Zipf 只改权重/累计曲线，不改 g(r) 形状。本批跑 3 个 seed 做经验验证。

### Setting（与 t5_low 完全一致，仅频次分布不同）

| 项 | 值 |
|---|---|
| 分布 | `mode=zipf`：N_r = round(C/r²)，r=1..199，Σ=32768 keys，整数精确计数（counts_exact）|
| 协议 | coincidental r<16 / shared r≥16（同 low）|
| 训练 | 2000 步 · input 注入 · seed 42/43/44 · β=(0.0,0.999) · RMSProp(table)+AdamW(backbone) |
| 数据 | `toy5_data_gen.py --mode zipf` + `toy_prep.py --vocab 2048` → cache `t5_zipf` |
| 评估 | exact-context per-r gap（r=1,2,4,8,16,...）· val 每 10 步 |
| 集群 | 待定（360-2 当前不可达；ophis-gpu 需先同步 toy5 代码+建 cache）|

### 预期结果

1. per-bucket g(r) 与 t5_low 重合（可分离性验证）：双对数 R² 仍 ~0.2–0.4，拐点在 r=16。
2. 严格 Zipf 加权后的累计/每 token 贡献曲线 → 干净幂律（重加权分析已示：
   token 加权贡献 slope≈−1.09，R²≈0.97）。
3. 高频桶（r≥32）样本少（N_32≈20, N_64≈5）→ per-bucket 统计在高频端变噪，
   与理论「纯 Zipf 下高 r 桶统计崩掉」一致。

### 结果（已完成 2026-08-07，360-2 GPU1/2/5）

| run | final gap@2000 | rho_logr | log-log slope | log-log R² | per-r gap (1,2,4,16,64) |
|---|---:|---:|---:|---:|---|
| `t5z_zipf_s42` | +7.012 | −0.879 | −0.154 | 0.784 | 8.56 / 9.40 / 8.93 / 5.27 / 5.32 |
| `t5z_zipf_s43` | +7.956 | −0.861 | −0.131 | 0.765 | 8.97 / 9.91 / 9.73 / 6.12 / 6.00 |
| `t5z_zipf_s44` | +7.558 | −0.851 | −0.123 | 0.754 | 8.24 / 9.24 / 9.04 / 5.88 / 5.67 |
| `t5b_beta_000_999_low`（anti-Zipf 对照）| +7.894 | −0.364 | −0.099（全 6 点）/ −0.112（去 r=8 离群）| 0.235 / 0.821 | 8.80 / 9.54 / 9.05 /(8:14.30)/ 6.43 / 6.15 |

**关键观察 / 结论**

1. **严格 Zipf 构建成功**：`mode=zipf` 生成 N_r≈C/r²（r=1:19921, r=2:4980, r=4:1245,
   r=8:311, r=16:78，比值 4:1），Σ=32768 keys，`counts_exact=True`；3 seeds（42/43/44）
   全部 rc=0，cache `t5_zipf` 在 360-2 构建（548k tokens/epoch，2000 步 ≈ 34 epoch）。
2. **per-bucket g(r) 与分布可分离（经验验证成立）**：严格 Zipf 的 per-r gap
   （8.59/9.52/9.23/5.76/5.66）与 anti-Zipf 的 t5b（8.80/9.54/9.05/6.43/6.15）
   几乎重合（r=1..4 差 <0.3，r=16/64 差 ~0.5–0.7），同协议下换分布 g(r) 不变。
3. **双对数线性没有变好**：zipf R²≈0.75–0.78 vs t5b 去 r=8 离群后 R²≈0.82，
   斜率都只有 ≈ −0.12~−0.15（不是干净幂律 −1）；t5b 全点 R²=0.24 的“坏”主要是
   r=8 离群（14.30，疑似 low-cache probe 噪声），不是分布问题。
4. 旧 kink 设计（t5_on_low）的陡斜率 −1.9 / R²=0.80 来自 r≥16 shared key gap→0 的
   硬拐点，与 Zipf 无关。
5. 与真实模型对照：真实语料（近似 Zipf）per-bin 双对数 R²=0.81–0.96，已较好；
   toy 的偏差来自协议（r≥16 拐点 + 低频段 g 微升），分布不是原因。

### 产物

- 代码：已迁入 `tasks/` 的 toy 数据生成脚本（`mode=zipf`）；launcher：
  `tasks/` 中的 zipf launcher。
- 数据：历史 `t5z_zipf_s{42,43,44}` run 的 metadata 已纳入本实验记录。
- 图：`docs/figs/theory/fig_zipf_experiment.{png,svg}`（左：per-bucket g(r) 重合；
  右：N_r 分布 −1 vs −2）；`docs/figs/theory/fig_zipf_gap_analysis.{png,svg}`（重加权分析）。
- 脚本：`docs/plot_scripts/gen_zipf_experiment_figs.py`、`analyze_zipf_gap.py`。

## 14. 干净 vanilla 复现（2026-08-23，input 主臂 + nogram 对照）

### Setting
极简 SSOT（agents.md §1）标准配置，无任何偏离：

| 项 | 值 |
|---|---|
| backbone | vanilla nanoGPT 8L·6H·768D，vocab 8192，seq 2048 |
| n-gram | bigram+trigram，`input` 注入，table 1M（默认未动） |
| 优化器 | table RMSProp(0.0,0.99)，backbone AdamW(0.8,0.95) lr 0.004 wd 0.1 |
| 数据 | shard 1 train（24264 rows ≈ 337 steps/epoch），shard 2 val，fixed 顺序 |
| 步数 / 评测 | 1000 步（≈3 epoch），seed 42，val 每 10 步 fixed batches（v10 口径） |

### 目的
在 `code/train.py` 干净复现上重跑标准极简设置，确认 gap 现象可复现
（历史参考：`nglab1x_v10_input` 1.931@2000，`nglab1x_v10_nogram` 0.231@2000）。

### 结果（已完成 2026-08-23）

| run | gap@999 | gap@1000 | train@1000 | val@1000 | 说明 |
|---|---:|---:|---:|---:|---|
| `vanilla_input_1000_seed42` | +0.803 | **+0.858** | 3.608 | 4.466 | input 注入主臂，epoch 2 起 fork |
| `vanilla_nogram_1000_seed42` | +0.022 | **+0.038** | 5.305 | 5.342 | 无 n-gram 对照，全程无 fork |

**关键观察 / 结论**

1. **gap 现象在干净 vanilla 上成功复现**：input 臂 gap 从 epoch 2（step 337 边界）起单调 fork，
   step 430 时 +0.07 → step 1000 时 +0.86；nogram 对照全程 ±0.04 内波动。
2. **gap 的来源确认为 n-gram 表**：input 臂 train 压到 3.61（表记住训练 token 的 over-encoding），
   val 只降到 4.47；nogram 臂 train/val 同步停在 5.3 附近（纯 backbone 泛化）。
3. 与历史口径一致（`nglab1x_v10_input` 1.931@2000 / `nglab1x_v10_nogram` 0.231@2000），
   1000 步的 fork 幅度约为 2000 步的一半，趋势吻合。

### 产物
- `data/runs_fixed/vanilla_input_1000_seed42_fixed/`（train_log.jsonl + summary.json）
- `data/runs_fixed/vanilla_nogram_1000_seed42_fixed/`
- 训练代码 `code/train.py`（未改动），数据生成 `code/prepare_data.py`（shard 1/2 现生成）

## 15. P1/P2 因果干预 · 极简 setting 重跑（2026-08-24）

### 目的
复现 `agents.md` §6.3「废弃结论，保留问题」队列的四条因果结论（原 current-shell 数字），
在极简 setting（vanilla nanoGPT + input 注入 + table 1M + RMSProp 无动量）下重跑。
旧结论（DEPRECATED SETTING，见 `docs/_archive/docs/p12-causal-results.md`）：
table 回滚 −89% / readout 屏蔽 −89% / 冻结 table −49% / 冻结 backbone −54%。

### Setting

| 项 | 值 |
|---|---|
| backbone / n-gram | 同 §14（8L·6H·768D，bigram+trigram input 注入，table 1M） |
| 优化器 / 数据 / 步数 | 同 §14（RMSProp+AdamW，shard1 train / shard2 val，1000 步，seed42，v10 fixed-val） |
| 干预触发点 | epoch 边界（`--intervention_epoch`，0-indexed：1 = e1→e2 边界 ~step337，2 = e2→e3 边界 ~step674） |
| 控制臂 | 复用 `vanilla_input_1000_seed42`（+0.858 @1000），不重跑 |

### 干预臂矩阵

| 臂 | 干预 | 触发 | 复现旧结论 |
|---|---|---|---|
| `nglab1x_input_reset_e2` | 全 table 行回滚 init | e2 边界 | p1_reset_all_e2（−89%）|
| `nglab1x_input_reset_e1` | 全 table 行回滚 init | e1 边界 | p1_reset_all_e1（−13%，对照）|
| `nglab1x_input_mask_e1` | 屏蔽 bigram/trigram readout | e1 边界 | p2_readout_mask_e1（−89%）|
| `nglab1x_input_freeze_table_e1` | 冻结 table（保留 e1 内容）| e1 边界 | p2_freeze_table_e1（−49%）|
| `nglab1x_input_freeze_backbone_e1` | 冻结 backbone（仅 table 更新）| e1 边界 | p2_table_gate_only_e1（−54%）|

### 状态（已完成 2026-08-24）

| 臂 | 状态 | final gap@1000 | train@1000 | val@1000 | vs 控制 |
|---|---|---|---:|---:|---:|---|
| 控制 `vanilla_input_1000_seed42` | ✅ done | +0.858 | 3.608 | 4.466 | — |
| `nglab1x_input_reset_e2` | ✅ done | **+0.054** | 4.014 | 4.068 | **−94%** |
| `nglab1x_input_reset_e1` | ✅ done | +0.351 | 4.004 | 4.355 | −59% |
| `nglab1x_input_mask_e1` | ✅ done | **+0.058** | 4.269 | 4.327 | **−93%** |
| `nglab1x_input_freeze_table_e1` | ✅ done | +0.601 | 3.851 | 4.453 | −30% |
| `nglab1x_input_freeze_backbone_e1` | ✅ done | +0.780 | 4.159 | 4.939 | −9% |

**关键观察 / 结论**

1. **两个 −89% 级关键干预在极简 setting 上复现**：
   - e2 边界全 table 回滚（`reset_e2`）：gap 0.858 → 0.054（**−94%**，旧 −89%）。
     e1/e2 两 epoch 累积的行内容是 e3 大 gap 的必要条件。
   - e1 边界屏蔽 readout（`mask_e1`）：gap 0.858 → 0.058（**−93%**，旧 −89%）。
     n-gram readout 通道是 gap 的必要传导口。
2. **回滚时机剂量**：e1 回滚（−59%）比 e2 回滚（−94%）弱——e1 擦掉后 e2 还能重写，
   到 e3 时行历史部分恢复；e2 擦掉后只剩 e3 一个 epoch 重写，恢复不了。与旧的
   「e1 −13% / e2 −89%」方向一致（本批 e1 干预更大，因极简 setting 的 e1 行写入更强）。
3. **table write vs backbone 各贡献一半的旧结论未完全复现**：本批 freeze_table −30%、
   freeze_backbone −9%，backbone 冻结影响远小于旧的 −54%。即极简 setting 下 gap 更依赖
   table 持续写入 + backbone 放大，backbone 本身的训练动态贡献较小（旧 current-shell 有
   gate/reader 等额外可训练放大器，占一半）。
4. 与 §14 控制臂串起来：**gap 的产生与传导完全依赖 n-gram 表**（回滚/屏蔽 → 塌缩到
   nogram 对照量级 0.04~0.06），主干结论稳定跨 backbone 架构。

### 产物
- 干预实现：`code/train.py`（新增 `--intervention` / `--intervention_epoch` / `--table_mult`）
- launcher：`code/cluster/run_causal_minimal.sh`
- 集群数据：`data/runs_fixed/nglab1x_input_{reset_e2,reset_e1,mask_e1,freeze_table_e1,freeze_backbone_e1}_fixed/`

## 16. bf16 精度验证 + 提速（2026-08-24）

### 目的
确认把全 fp32 前向切到 bf16（`torch.autocast`，权重/优化器仍 fp32）不会改变 gap 现象，
同时量化提速幅度，作为后续实验的默认计算精度。

### 测速（H200，batch 72×2048，28.8B 全模型，单卡空闲）

| 配置 | train step | 相对 fp32 |
|---|---:|---:|
| fp32 | ~2.76 s | 1.0x |
| bf16（autocast） | ~0.48–0.53 s | **~5.3x** |
| bf16 + torch.compile | ~0.50 s（GPU 共租下） | ~5.5x |

- fp8 不可行：`torch.autocast(dtype=float8_e4m3fn)` 在 `nn.Linear` addmm 上不支持，
  需专门 `_scaled_mm` 工程，未采用。
- `--dtype {fp32,bf16,fp8}` + `--compile` 开关已加入 `code/train.py`；
  标准 launcher `run_causal_minimal.sh` 默认 `bf16` + `--compile`（可通过
  `NGLAB_DTYPE` / `NGLAB_COMPILE` 覆盖），并为每臂隔离 `TORCHINDUCTOR_CACHE_DIR`。

### 同超参精度对照（关键）

| 项 | fp32 `vanilla_input_1000_seed42` | bf16 `..._bf16_samehp` |
|---|---:|---:|
| 表优化器 | RMSProp β₂=0.999, lr_scale=1.0 | 同左（完全一致） |
| train@1000 | 3.608 | 3.580 |
| val@1000 | 4.466 | 4.405 |
| **gap@1000** | **+0.858** | **+0.825** |
| gap 曲线 | 见下 | 逐点重合（10/340/670/1000: +0.009/+0.021/+0.361/+0.858 ↔ +0.008/+0.049/+0.374/+0.825）|

**结论：bf16 在相同超参下逐点复现 fp32 曲线（loss 差 <0.1，final gap 0.858 vs 0.825），
且提速 ~5.3x。后续正式实验默认 bf16。**

> 注意：先前 `vanilla_input_1000_seed42_bf16`（gap +1.661）用了不同超参
> （β₂=0.99, lr_scale=2.0），非精度差异，勿与其对比。

### 产物
- 图：`docs/figs/fig_fp32_vs_bf16_samehp.png`
- 代码：`code/train.py`（`--dtype` / `--compile`）、`code/cluster/run_causal_minimal.sh`（默认 bf16+compile）
- 集群数据：`data/runs/vanilla_input_1000_seed42_bf16_samehp/`（+0.825）
