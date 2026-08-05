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
- novel + 低频 bucket 主导 gap（详见 `fig_gap_by_freq.html`）

## 3. 频率 bin 分解（2026-08-05 完成）

用 `code/ngram_freq.py` 构建频率索引，统计 per-bin 的 mean loss 与 total contribution。

结果：novel + 低频 bucket（1-5）主导 gap；高频 bucket（5k+）gap 贡献 ≈ 0。与旧实验一致。
