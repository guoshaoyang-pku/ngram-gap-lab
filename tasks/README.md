# Toy 实验线索引

不依赖 nanoGPT backbone 的理论验证实验。**5 条线，编号 L1–L5**。

每个任务目录**自包含**：脚本 + `results/` 结果 + 输入 fixture 都在同一个 `tasks/lN_*/` 下。
理论文档在 `docs/notes/theory/`，对外图在 `docs/figs/theory/`。

---

## L1 · Lookup-Table 记忆 × Replay

**问题**：固定 replay 是否让「每 key 一个标量」的查表记忆去背**标签噪声**？

2×2×2×8 矩阵：记忆开/关 × fixed/fresh replay × 噪声 0/0.2 × 频率桶 1..128。
预测：gap 只在 `fixed + memory + noise>0` 同时成立时出现。

| | |
|---|---|
| 脚本 | `tasks/l1_lookup_replay/toy6_lookup_replay.py`（纯 torch） |
| 结果 | `tasks/l1_lookup_replay/results/main_matrix_20260805/`（160 runs, 4.5 MB）<br>`markov_5ep/`（Markov 数据上的 5 epoch 变体）<br>`phase1_confirmation_20260821/`（独立复现证据） |
| 理论 | 结论交叉引用见 `docs/notes/theory/markov-unigram-exact-gap.md` §7 |

---

## L2 · Markov 链 × 纯计数表：精确 gap 闭式解

**问题**：l1/l2 二阶 Markov 链上，纯 unigram/bigram 计数表的 gap 能否**闭式解出**？

检验 `E[gap∞] = γ(M−1)/N`、`γ = 9`、逐 epoch 瞬态 `β_t`、`std/mean = √(2/(M−1))`，
以及 fixed vs fresh replay 的差异。

| | |
|---|---|
| 脚本 | `l2_markov_exact/markov_clean_unigram.py`（**权威干净版**，M=512，纯 numpy，~35 s）<br>`l2_markov_exact/markov_unigram_gap.py`（探索版，A–F 六节：平稳分布 / 熵率 h / 自相关 τ / E[g∞] / 多 epoch GD / excess-vs-freq） |
| 结果 | `l2_markov_exact/markov_clean/`（M=512 权威）<br>`bigram_bc_m512/`（bigram B→C 臂）<br>`unigram_m64_dense/`（M=64 全批）<br>`unigram_m64_sgd/`（M=64 SGD 对照） |
| 理论 | `docs/notes/theory/markov-unigram-exact-gap.md`（闭式解 + 数值核验全通过）<br>`docs/notes/theory/toy-markov-unigram-gap.md` |
| 图 | `docs/figs/theory/fig_markov_unigram_*` |

⚠️ `bigram_bc_m512` / `unigram_m64_dense` / `unigram_m64_sgd` 是 2026-08-11 的中间迭代产物，
**生成代码已被 `markov_clean_unigram.py` 重写取代**，属于「有结果无可复现代码」的历史数据。

---

## L3 · 单 context 采样律 gap(r) ≈ (K−1)/r

**问题**：一个 context、真分布 P、r 个 iid 样本，gap 与 r 是什么关系？

结论：**解析区** log-log 斜率精确 −1，常数是**支撑 K−1 而非 exp(H)−1**；
**未解析区**（长尾）由未见符号惩罚 `U(r)·log(r/αK)` 主导，斜率漂到 ≈ −0.2 且依赖平滑 α。

| | |
|---|---|
| 脚本 | `l3_sampling_law/gap_vs_samples_unigram.py`（A–G 七节，纯 numpy） |
| 结果 | 无独立目录，直接出图 |
| 理论 | `docs/notes/theory/unigram-gap-vs-samples.md`<br>`docs/notes/theory/real-sparse-unigram-gap.md`（真实长尾修正） |
| 图 | `docs/figs/theory/fig_gap_vs_samples_{bc11,exact,unresolved,longtail,realgen}.*` |

---

## L4 · 幂律频率合成数据 × 真 harness

**问题**：把 L3 的估计论定律放到**真实 nanoGPT + n-gram 表** harness 上是否复现？

构造 vocab 8192 / order 5 / 精确频率桶 `r ∈ {1..1024}` 的合成语料，
检验 per-bucket `gap(r)` 双对数斜率是否 −1、K_eff 是否 ≈ 8。

| | |
|---|---|
| 脚本 | `l4_synth_powerlaw/synthetic_transition_gen.py`（v1 pilot，scheme A sparse_restart / B lowrank_sparse）<br>`l4_synth_powerlaw/synth_powerlaw_gen.py`（v2 干净版：细桶 128 ctx/桶 + 概率规则 + context-uniform val）<br>`l4_synth_powerlaw/synthetic_prep.py`（补 `meta.json` 给 ngram5 trainer） |
| 编排 | `l4_synth_powerlaw/cluster/run_synth_pipeline.sh`（**唯一编排文档**：gen A → gen B → prep ×2 → smoke → all）<br>`run_synth_3602.sh` / `run_synth_pl.sh`（360-2 完整 env，含 `NGRAM_TABLE_BETAS=0.0,0.999` ✅ 符合极简 setting） |
| 结果 | 原始 run 在 360-2 远端（`ngram5_data/synth_*`, `runs/ngram5/`）；本地只有汇总 `docs/figs/theory/synth_{A,B}_summary.json` |
| 理论 | `docs/notes/theory/toy-gap-powerlaw-mechanism.md`<br>`docs/notes/method/synthetic-transition-task-design.md` |
| 图 | `docs/figs/theory/fig_synth_*`、`fig_toy_synth_*`（由 `docs/plot_scripts/build_toy_model_blog_figs.py` 生成） |

---

## L5 · 优化器伪影：RMSProp v 锯齿 & 表容量

**问题 A**：epoch 边界的 loss 锯齿是**优化器伪影**还是真遗忘？
五臂对照分离「v 压缩」「batch 顺序」「trunk 漂移」三条通道：

| 臂 | 含义 |
|---|---|
| A `global_bias` | dense-v 全局衰减（真配置） |
| B `rowwise_bias` | lazy row-wise v |
| C `sgd` | 纯 SGD |
| D `global_bias_fixedorder` | 固定 batch 顺序 |
| E `global_bias_frozenW` | 冻结 readout W |

**问题 B**：表行数 R 变化改变 gap，是**容量**还是**低频涨落加权**？

| | |
|---|---|
| 脚本 | `l5_optimizer_artifact/rmsprop_v_sawtooth.py`（torch；kernel 已内联为 `ngram_rms_kernels.py`，**不再依赖 OPHIS**）<br>`l5_optimizer_artifact/mc_table_size.py`（纯 numpy，R 从 K/8..8K 扫描 + 行冲突平均；目前只 print，无落盘） |
| 输入 | `tasks/l5_optimizer_artifact/results/inputs/exact_frequency_distribution.json`（真实频率直方图 fixture） |
| 结果 | `tasks/l5_optimizer_artifact/results/rmsprop_v_sawtooth/`（5 臂 JSON + `_arms.{png,svg}`） |
| 理论 | `docs/notes/method/loss-curve-sawtooth-audit.md`（**重要踩坑**：移动窗 val + 50 步间隔曾造出纯显示伪影的锯齿） |

---

## 消融附录（M3/M4 线深挖）

| 任务 | 问题 | 脚本 | 结果 | 图 |
|---|---|---|---|---|
| `apendix_lr_beta_ablation/` | 表学习率与 β₂ 的消融及交互；高表学习率体检 | `extract_data.py` + `make_figures.py` + `build_report.py` | `results/appendix_data.json` | `figs/`（7 张 SVG） |

这是主线表优化器消融（M3+M4）的深挖专题，**自包含**：`report.md`（实验报告）、
`report.html`（内嵌 SVG 的阅读版）、`figs/`、`results/` 全部在本目录内。
跑法：`extract_data.py`（可选 `--remote ophis-gpu` 拉补点）→ `make_figures.py` → `build_report.py`。

---

## 运行

所有脚本都可从仓库根目录直接跑，无需额外配置：

```bash
python tasks/l2_markov_exact/markov_clean_unigram.py     # ~35 s，纯 numpy
python tasks/l3_sampling_law/gap_vs_samples_unigram.py   # 纯 numpy
python tasks/l5_optimizer_artifact/mc_table_size.py      # 纯 numpy
python tasks/l5_optimizer_artifact/rmsprop_v_sawtooth.py # torch，支持 MPS
python tasks/l1_lookup_replay/toy6_lookup_replay.py      # torch，160 runs 较慢
```

依赖：`numpy`、`matplotlib`；L1/L5 另需 `torch`。**均不依赖 nanoGPT backbone。**
