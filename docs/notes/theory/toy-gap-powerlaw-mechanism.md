# 干净双对数线性 gap：机理推导与实验设计（2026-08-07）

## 1. 一句话结论

在「概率规则 + 诚实 val + 计数型记忆（n-gram 表）」的小世界里，
`gap(r) = val CE − train CE ≈ (K_eff − 1)/r`，log-log 斜率 = −1，是**精确**的双对数线性。
机理是估计论：经验分布 `p̂`（从 r 个 iid 样本估计真分布 P）的 KL 偏置 `(K−1)/(2r)`，
train 侧与 val 侧各承担一半、符号相反，相减后 H 的常数项完全抵消。

## 2. 推导（计数表 + 概率规则 + 诚实 val）

设 context c 的真分布 `P(·|c)`，train 里抽到 r 个 iid 样本，模型把经验分布
`p̂`（加平滑 α）存进 n-gram 表。r 个样本都是**同分布独立**的（诚实 val：val 也从这个 P 抽）。

- **val CE**：val 问的是 P 的新样本 y~P。
  `CE_val = E_{y~P}[-ln p̂(y)] = H(P) + KL(P || p̂) ≈ H + (K−1)/(2r) + O(r^{-2})`
- **train CE**：train 位置问的是 p̂ 自己的样本（模型把训练位置背到 H(p̂)）。
  `CE_tr = E_{y~p̂}[-ln p̂(y)] = H(p̂) ≈ H − (K−1)/(2r) + O(r^{-2})`
  （熵偏置 `E[H(p̂)] = H − (K−1)/(2r)`，经典结论）
- **gap**：`gap(r) = CE_val − CE_tr ≈ (K−1)/r` —— **H 抵消，只剩 1/r**。

注意点：
1. 概率规则下 **train CE 不是 0**，而是 ≈ H(p̂)，随 r 从 ~0（r=1，p̂≈one-hot）升到 H（r→∞）。
   v1 的确定性规则里 H=0，所以 train CE 恒 ≈ 0——v1 没有这块「地板抵消」。
2. gap 的斜率来自估计论，**与总体分布 N_r 无关**（per-bucket 曲线是 g 的指纹）。
3. `K_eff = exp(H(P))` 是每个 context 的有效符号数；常数 = K_eff − 1。
4. 平滑 α 只在 r≲αK 时压低曲线；无平滑时渐近严格 1/r。

## 3. 数值验证（numpy MC，本地）

计数表 + Dirichlet 平滑 α=0.001，K_eff=8，r=1..512：

| r | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| gap | 5.33 | 4.37 | 3.13 | 1.85 | 0.95 | 0.40 | 0.18 | 0.072 | 0.033 | 0.015 |
| (K−1)/r | 7.00 | 3.50 | 1.75 | 0.88 | 0.44 | 0.22 | 0.11 | 0.055 | 0.027 | 0.014 |

log-log 斜率 = **−0.995**（r=1..512，3 个数量级）。低 r 处有限样本修正让曲线略低于 (K−1)/r，渐近贴合。

## 4. 为什么 v1 的曲线不干净

v1 的三个污染源（对照本设计的三个「干净化」）：
1. **val 协议混杂**：v1 的 designed 键 val 问相同接续（shared），~50% 位置 gap≡0 稀释曲线；
   本设计 val 全部问 P 的新样本（诚实），无稀释。
2. **incidental 窗口**：v1 80% 的训练位置是跨界窗口，bucket 归属混杂；
   本设计沿用 v5/合成的 `[context, y, SEP]` 块结构，附带窗口与目标位置分布一致。
3. **确定性规则**：v1 的 P 是 δ 分布，H=0，train CE 恒≈0，没有「H 抵消」；
   本设计用概率规则（scheme A：私有 8 符号 + 全局，H≈2.57），gap 从估计偏置来。

## 5. GPU 实验（360-2）设计

- 生成器 `toy/synth_powerlaw_gen.py`：vocab 8192、order 5、hub 256；
  细粒度桶 r∈{1,2,4,...,1024} × 每桶 128 contexts（共 1408）；
  val = 每 context 8 个新鲜样本（context-uniform，probe 每桶命中 ~400+）；
  train 1.84M tokens；`K_eff=13.03`，bayes CE=2.567。
- 模型：nanogpt original + bigram/trigram VE（表开 vs 关），2000 步，VAL interval 10。
- 预测：表开 → gap(r) ≈ 12.03/r，log-log slope ≈ −1、R²≈1；
  表关 → backbone 无计数记忆，低 r 处 val CE 被钉在 ln K 量级，曲线形状不同（对照）。
- 分析 `ngram5_freq_gap/analyze_synth_pl.py`：从 probe_details 的 train/val npz 算每 r 桶
  gap = val CE − train CE，拟合 log-log 斜率；理想表假数据验证 slope=−1.038、R²=0.998。
- 启动 `toy/run_synth_pl.sh smoke|all`（360-2，2 GPU，两波）。
