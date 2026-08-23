# 注入点消融实验文档

> 🗄️ **[ARCHIVE]** 本文档来自已弃用的 `OPHIS_gap` 仓库，仅供历史溯源。
> 其中部分结论建立在 `current shell` / Muon / RoPE 等非极简 setting 上，引用前请对照 `agents.md` §6。


> 创建：2026-08-05
> 状态：input 注入主线，y/v 对照

## 1. 背景与动机

OPHIS 原 B 段实验（vanilla nanoGPT + n-gram value table，注入点 = v / ResFormer style）在 epoch2 目标窗口内无 gap。2026-08-05 的 norm 诊断发现：v 注入的 n-gram residual 只有 V 的 6.5%，信号被 V 淹没。这促使我们测试其他注入点。

注入点消融的核心问题：**n-gram value memory 产生 replay-specific gap 的最小充分条件是什么？**

## 2. 三种注入点

| 注入点 | 代码位置 | 技术方案 | 走 attention？ | 信号强度 |
|---|---|---|---|---|
| `v` | `V = V + gate·ngram_ve`，attention 之前 | ResFormer value residual | ✅ 是（被 softmax 混合） | 弱（norm 只有 V 的 6.5%） |
| `y` | `y = attn(Q,K,V) + gate·ngram_ve`，attention 之后 | ResFormer y-variant | ❌ 否 | 中（每层注入，gate 控制） |
| `input` | `x = wte(idx) + Σ_layer ngram_ve`，入口一次 | Over-encoding（Engram/SCONE/Over-Tokenized 主流） | ❌ 否 | 中（一次注入，所有层 table 求和） |

代码开关：`NANOGPT_NGRAM_INJECTION_POSITION = v | y | input`（`train.py` L3186）

## 3. 实验登记

### 3.1 1000 步消融（2026-08-05，无 theory obs）

| run | 注入点 | n-gram | optimizer | steps | seed | gap@999 | norm 诊断 |
|---|---|---|---|---|---|---|---|
| `injpos_v_bigram_trigram` | v | bi+tri | mixed rmsprop | 1000 | 42 | 0.60 | n-gram residual = V 的 6.5% |
| `injpos_y_bigram_trigram` | y | bi+tri | mixed rmsprop | 1000 | 42 | 1.82 | — |
| `injpos_input_bigram_trigram` | input | bi+tri | mixed rmsprop | 1000 | 42 | 0.64 | n-gram residual = wte 的 4.77x |
| `injpos_baseline_no_ngram` | — | none | mixed rmsprop | 1000 | 42 | 0.03 | — |

集群路径：`/data3/guoshaoyang/ngram-gap-exp/runs/injpos_*`
launcher：`run_injpos_ablation.sh`（v/y）、`run_injpos_input.sh`（input）、`run_injpos_baseline.sh`（baseline）

**注意**：这批 run 没开 `THEORY_OBS_CAPTURE`，没有 table gated norm / row history / direct kernel 等 observable。

### 3.2 2000 步延长（2026-08-05，无 theory obs）

| run | 注入点 | steps | gap@999 | gap@1999 |
|---|---|---|---|---|
| `injpos_v_long2000` | v | 2000 | 0.60 | 4.70 |
| `injpos_y_long2000` | y | 2000 | 2.10 | 4.65 |
| `injpos_input_long2000` | input | 2000 | 0.75 | 2.98 |

集群路径：`/data3/guoshaoyang/ngram-gap-exp/runs/injpos_*_long2000`
launcher：`run_injpos_long.sh`

**注意**：同样没开 `THEORY_OBS_CAPTURE`。

### 3.3 完整 observable 重跑（2026-08-05，已完成）

为获取 table gated norm / row history / frequency 分解等 observable，重跑带完整 theory obs 的 run。

**第一批（并行，时间戳冲突）**：

| run | 注入点 | steps | theory_obs | 状态 |
|---|---|---|---|---|
| `injpos_input_obs` | input | 1000 | ✅ 全开 | ✅ 完成，但 run_artifacts 时间戳冲突 |
| `injpos_y_obs` | y | 1000 | ✅ 全开 | ✅ 完成，但 run_artifacts 时间戳冲突 |
| `injpos_v_obs` | v | 1000 | ✅ 全开 | ✅ 完成，但 run_artifacts 时间戳冲突 |

**问题**：三个并行 run 用了同一时间戳 `20260805-192148`，`run_artifacts/` 下的文件互相覆盖。`layer_observable_curves` 25MB 数据保留（最后一个完成的 v_obs），其余可能丢失。

**第二批（串行，避免时间戳冲突）**：

| run | 注入点 | steps | theory_obs | frequency | 状态 |
|---|---|---|---|---|---|
| `injpos_input_freq2` | input | 1000 | ✅ 全开 | ✅ baseline | ✅ 完成 |
| `injpos_y_freq2` | y | 1000 | ✅ 全开 | ✅ baseline | ✅ 完成 |
| `injpos_v_freq2` | v | 1000 | ✅ 全开 | ✅ baseline | ✅ 完成 |

launcher：`run_injpos_freq2.sh`（串行，GPU 1）

**注意**：`NGRAM_HIT_TRACKING=1` 需要 `trigram_global_counts.npz`（不存在），已关闭。频率分解从 theory obs compact 数据 + global frequency index 后处理。

本地汇总：`remote_training_runs/injpos_obs_summary.json` 已包含 input/y/v 三组；每组有 103 个 `bg_rms`、`tg_rms`、`bg_grad`、`tg_grad` 点（step 10–1000），以及 955 个 gap 点。该文件和 `docs/injpos_ablation_plot.html` 构成当前 observable 重跑交付物。

## 4. Norm 诊断（2026-08-05）

诊断脚本：`diag_ngram_norm.py`（集群 `/data3/guoshaoyang/ngram-gap-exp/`）

```
=== v 注入（加到 V）===
layer    V_norm    ngram_resid   ratio
  1      0.675     0.044         6.5%
  3      0.674     0.044         6.5%
  5      0.673     0.044         6.5%
  7      0.676     0.044         6.5%

=== input 注入（加到 wte）===
wte norm:           0.024
n-gram residual:    0.117
ratio:              477%
```

gate 输出 mean = 1.0000（初始化时 `2·sigmoid(0) = 1.0`，设计正确）。

## 5. 关键结论

1. **v 注入无 gap 的原因是数值尺度问题**：n-gram value norm 只有 V 的 6.5%，信号被 V 淹没。不是"走 attention 被混合"的机制问题（虽然两者都不利于 gap）。
2. **y/input 注入都能产生 gap**：只要 n-gram 信号不走 attention 混合、能有效到达输出，就能产生 replay-specific gap。
3. **gap 的产生仅依赖 n-gram memory**：不需要 current shell / Muon / current optimizer grouping / RoPE / RMSNorm / untied embedding。
4. **B 段叙事需修正**：原结论"vanilla nanoGPT + n-gram 无 gap，需要 current shell"应修正为"vanilla nanoGPT 的 v 注入实现有数值尺度问题；改用 input/y 注入后 vanilla nanoGPT 单独加 bigram+trigram 即可产生 gap"。

## 6. 标准数据消融基线（baseline_input）

> 后续所有数据侧消融（频率遮罩、低频去除、order 对照、row reset/freeze 等）一律基于此 setting。

见 `plans/plan-1-gap-formation.md` §3.1a。

核心配置：
- `ARCH_VARIANT=nanogpt_original`
- `NANOGPT_NGRAM_INJECTION_POSITION=input`
- `NANOGPT_ENABLE_NGRAM_VE=1 ENABLE_UNIGRAM_VE=0 ENABLE_BIGRAM_VE=1 ENABLE_TRIGRAM_VE=1`
- `NANOGPT_NGRAM_OPTIMIZER=mixed NGRAM_TABLE_OPTIMIZER=rmsprop`
- `NANOGPT_ADAM_LR=0.004 NGRAM_TABLE_BETAS=0.0,0.999`
- `POSITION_ENCODING=learned_abs CURRENT_NORMALIZATION=layernorm`
- `CURRENT_EMBEDDING_TYING=tied CURRENT_NGRAM_INJECTION_IMPL=none`
- `WINDOW_PATTERN=LLLL SEED=42 MAX_TRAINING_STEPS=1000`
- `DEVICE_BATCH_SIZE=72 TOTAL_BATCH_SIZE=147456`
- `TRAIN_DATA_MODE=fixed TRAIN_DATA_SEED=42`
- `VAL_LOSS_INTERVAL_STEPS=10`（2026-08-06 起，v10 细曲线；旧 50 步存档见 50 步版 `injpos_ablation_data.json`）

参考 gap（seed42, input 注入）：1000 步 ≈ 0.64；2000 步 ≈ 2.98。

## 7. Table norm + gap 对比图（theory obs，1000 步）

图表：`docs/injpos_ablation_plot.html`

数据来源：`injpos_input_freq2` / `injpos_y_freq2` / `injpos_v_freq2`（串行跑，103 个 theory obs 点）

### 关键数据

| step | v gap | y gap | input gap | v bg_rms | y bg_rms | input bg_rms |
|---|---|---|---|---|---|---|
| 10 | 0.00 | 0.00 | 0.00 | 0.037 | 0.037 | 0.037 |
| 100 | -0.004 | 0.011 | 0.002 | 0.042 | 0.083 | 0.068 |
| 337(e2) | -0.047 | -0.048 | -0.093 | 0.121 | 0.119 | 0.113 |
| 400 | -0.021 | -0.127 | -0.121 | 0.142 | 0.127 | 0.119 |
| 500 | 0.016 | -0.571 | -0.111 | 0.154 | 0.134 | 0.129 |
| 686(e3) | -0.077 | -0.790 | -0.342 | 0.160 | 0.149 | 0.150 |
| 700 | -0.072 | -1.197 | -0.317 | 0.160 | 0.149 | 0.151 |
| 999 | -0.542 | -1.900 | -0.672 | — | — | — |

注：gap = val - train（正值=gap）。bg_rms = bigram table layer_1 table_0 的 param rms。

### 观察

1. **y 注入 gap 最大**（1.90），但 table rms 增长最慢（0.149）——因为 y 注入信号不走 attention，直接到输出，table 不需要很大就能影响 loss。
2. **v 注入 gap 最小**（0.54），但 table rms 增长最快（0.160）——因为 v 注入信号被 V 淹没，table 即使很大也影响不了 attention 输出。
3. **input 注入居中**（0.67，table rms 0.151）。
4. **table norm 的增长速度不是 gap 的决定因素；注入点（信号能否有效到达输出）才是。**

## 8. 频率 bin 分解（待做）

频率 bin 分解需要 `global_frequency_probe_observations.jsonl`（由 vbird 版本的 train.py 生成，含 `NGRAM_GLOBAL_FREQUENCY_MODE` 功能）。当前集群上的 train.py（OPHIS_gap 版本 + input 注入开关）没有这个功能。

**阻塞原因**：input 注入开关在 OPHIS_gap 版本的 train.py 上，频率分桶功能在 vbird 版本的 train.py 上，两个版本不兼容。

**解决方案（待选）**：
1. 把 input 注入开关移植到 vbird 版本的 train.py
2. 把频率分桶功能移植到当前版本
3. 用 `NGRAM_HIT_TRACKING=1` + 正确格式的 `trigram_global_counts.npz`（需要生成 `ctx_keys` 格式的文件）

`trigram_counts.npz`（已存在）的格式是 `keys` (scalar hash) + `counts`，但 `NgramHitTracker` 期望 `ctx_keys` (tuple, shape `(N,3)`)。需要转换格式或修改 `NgramHitTracker`。
