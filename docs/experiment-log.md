# ngram-gap-lab · 实验日志

> 创建：2026-08-05
> 每次实验登记一个 section，记录 setting、gap 数值、关键观察。

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
- novel + 低频 bucket 主导 gap（详见[总报告 §2](frequency-gap-by-hit-count.html#historical-frequency)）

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

## 5. RMSProp Stage 1: table beta2 x table learning rate (2026-08-20)

目的：在 input injection、bigram + trigram、seed 42、shard 1 fixed replay 的基线下，只改变 n-gram table 的 bias-corrected RMSProp `beta2` 和 table LR scale，观察 replay epoch 边缘的频率分桶曲线形态，并以在线训练曲线作为次要指标。每个条件运行 1000 optimizer steps，复用同一 fixed-gram manifest（每个 bucket 100 个 occurrence）。backbone AdamW `(0.8, 0.95)`、base LR `0.004`、weight decay `0.1` 保持不变；table optimizer 无 momentum，`eps=1e-10`，weight decay 为 `0`。

### 5.1 最终在线收敛

| beta2 | table LR scale | table base LR | final train loss | final validation loss | final gap |
|---:|---:|---:|---:|---:|---:|
| 0.990 | 0.5 | 0.002 | 3.7272 | 4.4138 | **+0.6866** |
| 0.990 | 1.0 | 0.004 | 3.6002 | 4.4934 | **+0.8932** |
| 0.990 | 1.5 | 0.006 | 3.4668 | 4.4551 | **+0.9884** |
| 0.995 | 0.5 | 0.002 | 3.6637 | 4.4061 | **+0.7424** |
| 0.995 | 1.0 | 0.004 | 3.6102 | 4.4967 | **+0.8866** |
| 0.995 | 1.5 | 0.006 | 3.7747 | 4.5590 | **+0.7843** |
| 0.999 | 0.5 | 0.002 | 3.7472 | 4.4520 | **+0.7049** |
| 0.999 | 1.0 | 0.004 | 3.5676 | 4.4830 | **+0.9154** |
| 0.999 | 1.5 | 0.006 | 3.5149 | 4.4839 | **+0.9690** |

LR scale 从 0.5 提高到 1.0 时，三个 beta2 的最终 gap 都上升（约 +0.14 至 +0.21）；从 1.0 提高到 1.5 后则不再全局单调：`beta2=0.990` 和 `0.999` 继续上升，而 `beta2=0.995` 回落到 `+0.7843`，并伴随更高的最终 loss。因而 LR 是在线 gap 的主要可见旋钮，但过大的更新尺度可能出现过冲，不能只按“LR 越大越好”解释。

### 5.2 Replay-edge 频率分桶形态

总报告 [RMSProp Stage 1 章节](frequency-gap-by-hit-count.html#rmsprop-stage1-results) 中的第二张表给出每个条件在 step 337 和 674 两个 replay edge 的 `edge+5 - edge-5`：低频侧为 buckets `1` 至 `11-20` 的均值，高频侧为 `21-50` 至 `5k+` 的均值，`tilt` 为低频减高频。

- **Bigram**：step 337 的 tilt 在九个条件中全部为正（`+0.0121` 至 `+0.1368`），说明首个 replay edge 通常表现为低频侧比高频侧更明显的上扬。step 674 仍以正值为主（`-0.0170` 至 `+0.0812`），但高 LR 或较大 beta2 的个别条件已接近零或变负。
- **Trigram**：step 337 的 tilt 除 `beta2=0.990, LR=1.5` 外均为负（范围 `-0.0814` 至 `+0.0123`）；step 674 仍大多为负（范围 `-0.0335` 至 `+0.0172`）。这表明 trigram 的边缘响应不像 bigram 那样稳定地呈现低频 uplift，更容易受 beta2、LR 和训练阶段共同影响。
- 因此，阶段 1 支持的稳健现象是“replay edge 对低频 bucket 更敏感，尤其是 bigram”；但 edge 形状对 optimizer 参数没有简单的单调规律。后续 forking 实验应保留 `beta2=0.999, LR scale=1.0` 的基线，并优先比较 LR scale 0.5/1.5 下首个 edge 的低频 uplift，而不是只比较最终 global gap。

原始 run 目录保留在 `data/runs/`；所有交互曲线统一收录在唯一总报告中，不再生成每条件子报告。九项矩阵见 `docs/rmsprop-stage1-plan.md`，已全部本地验收。

## 6. RMSProp Stage 2A: wider beta2 and low-LR sweep (2026-08-21)

目的：在 Stage 1 固定 setting 下，将 table RMSProp `beta2` 扩展到 `0.5` 和 `0.9`，并对 `beta2=0.999` 的 LR scale 0–1 区间加密采样。13 个新增条件均运行 1000 steps，复用相同 fixed-gram manifest；backbone、epsilon 和 weight decay 均保持不变。

| beta2 | table LR scale | final gap |
|---:|---:|---:|
| 0.500 | 0.250 | +0.3572 |
| 0.500 | 0.500 | +0.5381 |
| 0.500 | 1.000 | +0.7909 |
| 0.900 | 0.250 | +0.5149 |
| 0.900 | 0.500 | +0.7207 |
| 0.900 | 1.000 | +0.9608 |
| 0.999 | 0.000 | +0.0977 |
| 0.999 | 0.125 | +0.4619 |
| 0.999 | 0.250 | +0.6240 |
| 0.999 | 0.375 | +0.6068 |
| 0.999 | 0.625 | +0.8107 |
| 0.999 | 0.750 | +0.8040 |
| 0.999 | 0.875 | +0.8969 |

完整在线 loss、online/fixed/fixed-gram 频率分桶曲线，以及 step 337/674 replay-edge 汇总表，见[总报告 Stage 2A 章节](frequency-gap-by-hit-count.html#rmsprop-stage2a-results)。本节先记录已验收结果，不在曲线集成步骤中提前给出机制结论。

### 6.1 Fixed-beta gap–LR 与 fixed-probe 阅读响应

为避免 validation log 的 50-step 间隔造成 epoch 边缘插值，gap–LR 图直接使用 `online_frequency_gap_contribution.jsonl` 的 step 674：

`G_674 = online_val_loss_674 - train_writer_loss_674`

该 online observable 位于第 674 步 optimizer update 之前。对 `beta2=0.999` 的九个 LR∈[0,1] 条件，gap 从 LR=0 的 `-0.0218` 总体升至 LR=1 的 `+0.4136`，但存在 `.25→.375` 与 `.625→.75` 两处局部回落。线性描述的 `R²=0.7674`；因此当前结果支持“随 LR 总体增加并逐渐趋缓”的经验趋势，但不支持一个无噪声的固定计算公式。`beta2=0.5/0.9` 各只有三个点，即使线性 `R²` 较高，也不足以建立规律；所有结果还仅覆盖单 seed、单 checkpoint。

fixed probe 图使用可加和的 bucket contribution：

`C_bucket = val_frac × val_mean_loss - train_frac × train_mean_loss`

第一次阅读窗口为 step 169–172，比较 `ΔC = C_174 - C_164`；第二次为 506–509，比较 `ΔC = C_511 - C_501`。总报告中的两张交互图固定 beta2 和 branch 后，以 LR 为横轴、每个 frequency bucket 为一条曲线；hover 同时保留阅读前、阅读后和差分值。

## 7. Stage 3R: strict 2×2 order-control matrix (2026-08-25)

最终实验严格交叉两个因素：epoch 1 使用原始 logical optimizer-batch 顺序 `0…336` 或 order-seed-101 随机顺序；后续 epoch 固定重复 epoch-1 顺序或每个 epoch 重新 shuffle。四组均使用 model/data seed 42，其他模型、优化器、数据与 baseline setting 相同；按此前计划关闭 fixed probe，只评估 online 与 fixed-gram sample。

每一种 epoch-1 顺序只训练一次共同前缀至 step 337，保存完整 post-update model、AdamW/RMSProp、RNG 与 validation iterator 状态，再分叉为 no-shuffle/shuffle 两支。原始顺序 pair 的共享参数 SHA256 为 `cc487a7acd2042f70d893d789bb7331ebbd5c303464dbe17496f0cdc8c0946ca`；随机顺序 pair 为 `3a616d781269cf7879ce8fd1959a16f5bf1ccbb4406dfd77d9fb86797a831665`。两套 pair 各自的五类 step 1–337 日志均逐行相同。此前 data/order seed 耦合为 101、且未共享 checkpoint 的 preliminary random pair 不再进入最终 2×2 报告。

| epoch-1 order | later epochs | final train loss | final validation loss | final gap |
|---|---|---:|---:|---:|
| original | no shuffle | 3.5656 | 4.4868 | +0.9212 |
| original | shuffle | 3.7801 | 4.3528 | +0.5727 |
| random | no shuffle | 3.3702 | 4.6397 | +1.2695 |
| random | shuffle | 3.5947 | 4.3829 | +0.7882 |

以新 epoch 第一步（step 338/675，pre-update online observable）为边界：

| epoch-1 order / later epochs | immediate jump @338 / @675 | mean-10 jump @338 / @675 |
|---|---:|---:|
| original / no shuffle | -0.0132 / -0.0475 | +0.0218 / +0.0723 |
| original / shuffle | +0.2389 / +0.2952 | +0.3349 / +0.4362 |
| random / no shuffle | +0.0640 / +0.0262 | +0.0195 / +0.0731 |
| random / shuffle | +0.3072 / +0.3632 | +0.2825 / +0.5273 |

两个 epoch-1 order 条件给出一致方向：reshuffle 都显著放大 epoch 开头的 online gap 跃变，但都降低最终 gap。原始顺序下 final gap 下降 `0.3485`，随机顺序下降 `0.4812`；因此“边界瞬时跃变”和“最终 gap”仍然是不同统计量。随机 epoch-1 本身会提高最终 gap，但该效应也只覆盖单 seed，不宜脱离顺序实现和 checkpoint 口径外推。

总报告的 [Stage 3R 章节](frequency-gap-by-hit-count.html#reshuffle-stage3r-results) 提供四组独立开关的完整 1000-step writer/validation loss。Condition explorer 逐条件新增 global online gap；不再展示单独的 complete gap-contribution 与 epoch-edge comparison 曲线。边界密集采样开关关闭时，online 曲线只保留每 50 step 常规点与 epoch-start step 338/675；frequency 图例移至绘图区外。

## 8. Cumulative exact-frequency masking sweep (2026-08-26)

目的：在最小 input-injection setting 上，同时令训练语料精确命中频次 `freq ≤ x` 的 bigram/trigram context 不产生 n-gram 输出且不更新 table，扫描从不遮罩到全遮罩的 gap 响应。初始 sweep 包含 `none`、23 个数值阈值与 `all` 共 25 个条件，均为 seed 42、sequential one-shard fixed replay、3 epochs（1011 steps）。按计划关闭 fixed probe、fixed-gram sample、online frequency bucket 与 epoch 边缘逐步密集评估，只保留完整 writer loss、50-step validation、table norm，以及 epoch 末尾附近的稀疏 online gap。

主统计量统一为 optimizer update 前的 `online validation loss - writer train loss`，分别取 step 337、674、1011。完整曲线见[总报告累计频率遮罩章节](frequency-gap-by-hit-count.html#frequency-mask-sweep)。曲线模式可在三个 epoch-end 绝对 gap，或 `epoch 2 − epoch 1`、`epoch 3 − epoch 1` 两条增量曲线之间切换。整数输入框默认只显示 `0≤x≤210` 的已测条件。横轴是连续数值 `x`，默认采用 `log10(x+1)` 显示尺度以展开 `0–100` 的主要下降区间，并可切换为线性 `x`；`all` 按联合最大训练命中频次放在 `x=195,964`，`none` 仅作为 `x=0` 处不参与连线的环境/实现对照。代表性数值如下：

| mask threshold x | epoch 1 end | epoch 2 end | epoch 3 end |
|---:|---:|---:|---:|
| none | -0.0815 | +0.3608 | +0.8274 |
| 0 | -0.1193 | +0.2727 | +0.7782 |
| 1 | -0.0993 | +0.2183 | +0.5122 |
| 2 | -0.0932 | +0.1554 | +0.3923 |
| 5 | -0.0810 | +0.0783 | +0.2684 |
| 10 | -0.0746 | +0.0244 | +0.1601 |
| 15 | -0.0781 | -0.0197 | +0.0619 |
| 20 | -0.0675 | -0.0039 | +0.1064 |
| 50 | -0.0752 | -0.0523 | +0.0098 |
| 100 | -0.0771 | -0.0443 | +0.0052 |
| 500 | -0.0830 | -0.0465 | -0.0052 |
| 5k | -0.0789 | -0.0432 | -0.0035 |
| all | -0.0766 | -0.0603 | -0.0198 |

单 seed 下的直接观察是：epoch 1 末尾 gap 对遮罩阈值基本不敏感，初始 25 点始终在约 `-0.12…-0.07`；epoch 2/3 的 replay gap 则主要在 `x=0…15` 区间快速下降。到 `x=15` 时，epoch 2 gap 已接近零，epoch 3 从无数值遮罩时的约 `+0.78` 降至 `+0.0619`；`x≥50` 后两者基本进入零附近的平台，局部非单调变化更像单次 online validation batch 的噪声，不能解释成新的阈值效应。`x=15` 已遮罩约 17% 的 bigram occurrence 和 64% 的 trigram occurrence；由于两支同时累计遮罩，本实验只能定位联合低频区域，不能单独归因给 bigram 或 trigram。

正式曲线保留最初计时用的 `none` 条件。另在 360 软件环境补跑了不进入 49 点曲线的 `none bridge`：其 epoch 1/2/3 gap 为 `-0.0818/+0.3159/+0.7649`，相对正式 `none` 的最大绝对差为 `0.0626`。生成器严格验证全部 49 个条件和 bridge 的 optimizer、mask metadata、固定 batch order与 frequency-index SHA；环境差异说明曲线适合判断大幅度下降区间，不宜把相邻阈值之间百分位级的小波动当成精确函数关系。

### 8.1 Dense `0–210` supplement

为解析下降段和零附近平台，在相同 seed 42 setting 下新增 24 个阈值：`6, 8, 9, 11, 13, 14, 16, 17, 18, 19, 35, 45, 70, 90, 110, 120, 130, 140, 150, 160, 170, 180, 190, 210`。合并后，`0–210` 内共有 41 个数值条件：`0–5` 保留原稀疏点，`5–20` 每个整数一个点，`20–50` 每 5 一个点，`50–210` 每 10 一个点。连同高阈值尾部、`none` 和 `all`，总报告严格验证并展示 49 个正式条件。

加密结果保持原结论：epoch 1 末尾曲线仍近似水平；epoch 2/3 及其相对 epoch 1 的增量在 `x=0–15` 快速下降，`x≈15–50` 是带明显局部波动的过渡区，`x≥50` 总体进入接近零的平台。新增点同时表明这不是严格单调函数，例如 epoch 3 在 `x=15/16/17` 为 `+0.0619/+0.2149/+0.1518`，而 `x=170/180/190` 为 `-0.0019/+0.0502/+0.0395`；因此适合用平滑衰减或平台模型描述总体趋势，但不能把相邻点线性插值解释为确定动力学规律。

以 41 个 `0≤x≤210` 数值点，对两条增量曲线作 robust Hill-decay 描述 `D(x)=c+a/[1+(x/x0)^p]`：`D2=G2−G1` 得到 `c=0.0333, a=0.3585, x0=2.85, p=1.27, R²=0.9825, RMSE=0.0100`；`D3=G3−G1` 得到 `c=0.0906, a=0.8056, x0=1.96, p=0.92, R²=0.9681, RMSE=0.0274`。这说明总体函数更接近“低频阈值处快速衰减至非零平台”，而不是线性关系；拟合仅作单 seed 的经验摘要，不覆盖局部尖峰或跨 seed 不确定性。

`x=120` 首次在 `360-h200-1` GPU3 运行到 step 400 时遇到瞬时 NVLink peer-memory hardware error；失败输出完整留档，正式 run 随后在 `360-h200-2` GPU6 从头重跑成功并通过全部验收，不使用失败尝试的数据。
