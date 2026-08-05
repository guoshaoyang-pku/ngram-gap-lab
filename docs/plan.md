# ngram-gap-lab · 标准实验计划

> 目标：在 vanilla nanoGPT 上用最简配置复现 n-gram value memory 导致的
> replay-specific train/val gap，并量化每个频率 bin 的 loss 贡献。

## 1. 现象定义

含可训练 n-gram value table 的 vanilla nanoGPT，小训练集 + 固定顺序多 epoch replay 时：
- train loss 阶梯下降（epoch 边界 cliff），val loss 翘起 → train/val gap。
- 关键：gap 只依赖 n-gram memory，不需要 current shell / Muon / RoPE / RMSNorm。

## 2. 三种注入点（核心消融变量）

| 注入点 | 技术方案 | 走 attention？ | 信号强度 |
|---|---|---|---|
| `v` | `V = V + gate·ngram_ve`（attention 之前）| ✅ 是（被 softmax 混合）| 弱（norm 只有 V 的 6.5%）|
| `y` | `y = attn(Q,K,V) + gate·ngram_ve`（attention 之后）| ❌ 否 | 中（每层注入）|
| `input` | `x = wte(idx) + Σ ngram_ve`（over-encoding，入口一次）| ❌ 否 | 中（一次注入）|

**结论**：只要 n-gram 信号不走 attention 混合、能有效到达输出，就能产生 gap。
- `input` 注入 = over-encoding 风格（Engram/SCONE/Over-Tokenized 主流做法）。
- `y` 注入 = ResFormer y-variant（gap 最大但非主流）。
- `v` 注入 = ResFormer value residual（信号被 V 淹没，gap 最弱）。

## 3. 标准设置（baseline_input）

**模型与训练**：

| 项 | 值 |
|---|---|
| 模型 | vanilla nanoGPT（Karpathy 风格）|
| 注入点 | `input`（over-encoding）|
| n-gram | bigram + trigram（unigram/fourgram off）|
| 优化器 | mixed：n-gram table 用 RMSProp，backbone 用 AdamW |
| table betas | (0.0, 0.999) — 无 momentum，β₂ 跨 epoch 持久化 |
| AdamW lr | 0.004 |
| position encoding | learned absolute |
| normalization | LayerNorm |
| embedding tying | tied |
| window pattern | LLLL（全 attention）|
| seed | 42 |
| steps | 1000（延长对照 2000）|
| device batch | 72 |
| total batch | 147456 tokens |
| data mode | fixed（固定顺序 epoch replay）|
| val interval | 50 步 |

**参考 gap（seed42）**：
- input 注入：1000 步 ≈ 0.64；2000 步 ≈ 2.98
- y 注入：1000 步 ≈ 1.82；2000 步 ≈ 4.65
- v 注入：1000 步 ≈ 0.60；2000 步 ≈ 4.70（延迟型）

## 4. 待办

1. ✅ 搭建干净 repo（train.py < 1000 行 + ngram_freq.py）
2. ⬜ 集群跑 v/y/input 三注入点 smoke test，核对 gap 数值
3. ⬜ 用 ngram_freq.py 构建频率索引，跑 per-bin loss 统计
4. ⬜ 生成注入点对比图 + table norm 对比图 + 频率 bin 分解图
5. ⬜ 重写公开博客 ngram-gap-mechanism-guide
