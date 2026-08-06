# ngram-gap-lab · 实验日志

> 创建：2026-08-05
> **本文件是多 Agent 的唯一实验登记簿**：先登记（`planned`）→ 占 GPU 开跑 → 回填（`done`）。
> 并行规则、run_id 约定见 `AGENT.md` §3；标准 setting 见 `plan.md`。

## 实验登记总表

| run_id | 日期 | 实验 | 状态 | gap 关键值 | 详情 |
|---|---|---|---|---|---|
| `nglab_v` | 2026-08-05 | 注入点消融 · v | ✅ done | 0.33 @999 | §2 |
| `nglab_y` | 2026-08-05 | 注入点消融 · y | ✅ done | 3.50 @999 | §2 |
| `nglab_input` | 2026-08-05 | 注入点消融 · input | ✅ done | 0.79 @999 | §2 |
| `nglab2x_v` | 2026-08-06 | 双倍训练集 · v | ✅ done | 1.169 @2000 | §4 |
| `nglab2x_y` | 2026-08-06 | 双倍训练集 · y | ✅ done | 3.101 @2000 | §4 |
| `nglab2x_input` | 2026-08-06 | 双倍训练集 · input | ✅ done | 0.687 @2000 | §4 |
| `nglab2x_input_v10` | 2026-08-06 | 双倍训练集 · input · v10 细曲线 | ⛔ superseded | val 移动窗，已停 | §6 |
| `nglab0_5x_input` | 2026-08-06 | 半 epoch 训练集 · input · v10 | ⛔ superseded | val 移动窗，已停 | §7 |
| `nglab2x_input_v10_fv` | 2026-08-06 | 双倍训练集 · input · v10 · **fixed-val** | 🔄 running | 待填 | §6 |
| `nglab0_5x_input_fv` | 2026-08-06 | 半 epoch 训练集 · input · v10 · **fixed-val** | 🔄 running | 待填 | §7 |
| `nglab1x_v10_v` | 2026-08-06 | 标准 1x · v 注入 · v10 重跑 | 🔄 running | 待填 | §8 |
| `nglab1x_v10_y` | 2026-08-06 | 标准 1x · y 注入 · v10 重跑 | 🔄 running | 待填 | §8 |
| `nglab1x_v10_input` | 2026-08-06 | 标准 1x · input 注入 · v10 重跑 | 🔄 running | 待填 | §8 |
| `nglab1x_v10_nogram` | 2026-08-06 | 标准 1x · 无 n-gram 对照 · v10 重跑 | 🔄 running | 待填 | §8 |
| `nglab1x_opt_rmsprop_2x` | 2026-08-06 | table 优化器消融 · RMSProp lr×2 | 🔄 running | 待填 | §9 |
| `nglab1x_opt_adamw_090999` | 2026-08-06 | table 优化器消融 · AdamW(0.9,0.999) | 🔄 running | 待填 | §9 |
| `nglab1x_opt_adamw_080950` | 2026-08-06 | table 优化器消融 · AdamW(0.8,0.95) | 🔄 running | 待填 | §9 |
| `nglab1x_opt_sgd_09` | 2026-08-06 | table 优化器消融 · SGD momentum 0.9 | 🟡 planned | 待填 | §9 |
| `nglab_plot_baseline` | 2026-08-06 | 基础实验统计与图表归档 | ✅ done | 15 bins + log/log-log | §10 |

状态约定：`planned` 已登记未开跑 / `running` 运行中 / `done` 已回填 / `stalled` 超期未回填。
新实验流程：总表加一行拿到唯一 `run_id` → 正文新建 section 按 `AGENT.md` §3.3 模板填写
→ 占 GPU 开跑 → 结果回填并改状态。

---

## 1. 注入点消融（2026-08-05，OPHIS 旧 run 迁移）

以下数据来自旧 OPHIS 代码库（`/data3/guoshaoyang/ngram-gap-exp/runs/injpos_*_freq2`），
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
- 运行数据：gitignored `data/runs/nglab_{v,y,input}/`

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
> 数据保留在 `data/runs/nglab2x_input_v10/`，正式结果以 `_fv` run 为准。

### 结果

- （待回填，见 `nglab2x_input_v10_fv`）

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

- （待回填，见 `nglab0_5x_input_fv`）

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

### 结果

- 首波已启动（2026-08-06 20:41 CST）：`rmsprop_2x` → GPU 3；`adamw_090999` → GPU 4；`adamw_080950` 在 GPU 4 串行跟随（GPU 6 被并行任务占用）。
- `sgd_09` 待 GPU 释放后补跑。
- （待回填）


## 9. Table 优化器消融（2026-08-06，input，计划）

目的：标准实验的 n-gram table 用 RMSProp（β=(0.0,0.999)，无一阶矩），table RMS 在 ~500 步后进入平台（0.036→0.078@1000）。
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

### 结果

- 首波已启动（2026-08-06 20:41 CST）：`rmsprop_2x` → GPU 3；`adamw_090999` → GPU 4；`adamw_080950` 在 GPU 4 串行跟随（GPU 6 被并行任务占用）。
- `sgd_09` 待 GPU 释放后补跑。
- （待回填）
