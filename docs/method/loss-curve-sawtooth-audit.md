# Loss 曲线锯齿（Sawtooth）调查报告

日期：2026-08-02
数据：`baseline_current`（current-shell, bigram+trigram RMSProp, 1000 step, ophis-gpu `/data3/guoshaoyang/ngram-gap-exp/runs/baseline_current/train.log`）及 exp4/exp5 同源日志。
结论：**锯齿是 val 评估协议导致的显示伪影，不是记忆机制；其「周期性」= val 重评估周期（50 步）。**

## 1. 现象

用户观察到：loss 曲线有严重锯齿，gap 周期性升高，然后回到 0 或一个基准值。

## 2. 机制（定量证据）

### 2.1 主因：val 每 50 步才重评估一次

`train.py` 默认 `VAL_LOSS_INTERVAL_STEPS = 50`（env 可调），仅当 `completed_step % 50 == 0` 时重算 val（train.py:3912, 8825）。日志中 `val_loss` 在两次评估之间**原样复用**（实测同一 val 值连续出现 50 行）。

因此逐 step 计算 `gap = val_logged − train` 时：

- 每个 50 步窗口内：train 持续下降（phase A 每窗口约 −0.1~−0.3；phase B 每窗口约 −0.2~−0.6）→ gap 单调爬升；
- 窗口末尾（step 99, 149, 199, ...，间隔精确 50 步）：val 重评估 → gap 被重置。

实测重置跳变（baseline_current）：

| 窗口末尾 step | 重置前 gap | 重置后 gap | 跳变 |
|---|---|---|---|
| 98→99 | 0.451 | −0.250 | 0.701 |
| 148→149 | 0.593 | −0.103 | 0.696 |
| 198→199 | 0.278 | −0.052 | 0.331 |
| 248→249 | 0.212 | −0.064 | 0.276 |
| 298→299 | 0.227 | −0.056 | 0.283 |
| 348→349 | 0.120 | −0.046 | 0.167 |
| 398→399 | 0.119 | −0.018 | 0.137 |
| 448→449 | 0.110 | −0.019 | 0.129 |
| 498→499 | 0.125 | +0.013 | 0.113 |

→ 「回到 0」= phase A 的重置值（train ≥ val，gap_reset 略负/≈0）。周期严格 = 50 步。

### 2.2 次因：epoch 3 结构性 gap 让「基准值」上移

step ≥ ~550（epoch 2→3 边界 686、warmdown 750 附近）后，train 继续快速下降而 val 不再降/回升，因此**重置点本身**从 ≈0 一路上移：

| val 评估点 | val | train | gap_reset |
|---|---|---|---|
| 499 | 4.477 | 4.464 | +0.013 |
| 599 | 4.422 | 4.182 | +0.240 |
| 699 | 4.459 | 3.930 | +0.529 |
| 799 | 4.377 | 3.749 | +0.628 |
| 899 | 4.363 | 3.593 | +0.770 |
| 999 | 4.527 | 3.228 | +1.299 |

→ 「回到基准值」= 重置点随 epoch 上移；锯齿只是叠在这条上升基线之上的 50 步小波。

### 2.3 残留周期性检查

train loss 一阶差分自相关：lag 2 处 r=0.69（交替升降），其余峰值（16/20/26/29/32/41/43/46/49）为趋势非平稳伪影，无独立于 50 步的干净第二周期。epoch 边界（337/686）有微小 val 抖动；exp4（hash reseed）在边界后第一次 val 评估出现瞬时尖峰（step 349: 4.924→5.095；step 699: 4.264→4.637），随后恢复——这是 reseed 干预的瞬态，不是普通跑的表现。

## 3. 对论文/图表的含义

1. **锯齿幅度 ≈ 50 步内 train 下降量**（0.1~0.7），不含记忆机制信息；逐 step 的 gap 曲线会把协议伪影与真实信号混在一起。
2. **正式口径应只用 val 评估对齐点**：`run_summary.json` 的 `gap_at`（350/400/.../600）已是对齐口径；fig_exp45_results 等交互图若按逐 step 画 gap 会放大锯齿观感。
3. 比较干预效果时统一 val 间隔（exp4/exp5 与 baseline 均为 50，可比）；若调小 `VAL_LOSS_INTERVAL_STEPS`（如 5），锯齿幅度≈按比例缩小，但评估成本上升。
4. 「gap 周期性回 0」若被读者观察到，应主动解释为评估协议；「gap 不再回 0」才是 epoch-3 结构性信号（val 回升 + train 记忆化下降）。
5. 报告建议：主图表画 (a) val 评估点的对齐 gap；(b) 若要展示窗口内行为，画 train 斜率或 per-window train drop，而不是混排的 val_logged − train。

## 4. 复核路径

- `train.py:3912` `VAL_LOSS_INTERVAL_STEPS = max(1, int(os.environ.get("VAL_LOSS_INTERVAL_STEPS", "50")))`
- `train.py:8825` `if completed_step % VAL_LOSS_INTERVAL_STEPS == 0:`
- 日志证据：`baseline_current/train.log`（同一 val 连续 50 行）、`exp4_hashreseed_current/train.log`、`exp5_lowfreq_gatezero_current/train.log`
- 图：`docs/figs/fig_sawtooth_baseline.svg`（train/val/gap 叠加 + val 评估点（红点，间隔 50 步）+ epoch 边界/ warmdown 虚线）

## 5. 2026-08-08 v10 复核（VAL_LOSS_INTERVAL_STEPS=10）

结论不变：锯齿 = 两层伪影 + 一层真实信号；「周期」严格 = val 重评估间隔。

数据：`toy/runs/t5b_beta_000_999_low/train.log`（v10 配置，val 间隔 10 步，toy t5_low，2000 步 ≈ 29 epochs，`[val_loss]` 每 10 步一行）。

1. **10 步锯齿（主伪影）**：val 每 10 步重评估、窗口内日志原样复用 → `gap = val_logged − train` 在窗口内随 train 下降爬升、评估点重置。实测窗口内 gap 上升均值 0.188、最大 0.882（出现在 train 快速下降段）。幅度随间隔缩小约 5 倍（旧 50 步版窗口上升 0.1–0.7）。
2. **~80 步真周期（epoch 结构，非协议）**：train loss 一阶去趋势后自相关在 lag≈80（=toy epoch 长度，边界 step 70/150/230/…）出现 r=+0.56 峰；epoch 边界后 loss 先陡降再平台/回弹。这是固定顺序 replay 的 epoch 结构，画图时应标注 epoch 边界而不是当噪声。
3. **结构性 gap（「基准值」上移，真实信号）**：epoch 3+ 后 val 不再降而 train 继续塌缩，评估点处的 gap 重置值一路上移；锯齿只是叠在真实信号之上的小波。

图：`docs/figs/fig_sawtooth_baseline.svg`（50 步版）；v10 曲线见 toy beta-scan 图（`toy/figs_v11_*`）。
