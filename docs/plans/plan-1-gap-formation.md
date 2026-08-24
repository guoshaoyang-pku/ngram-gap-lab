# Plan 1 · Gap 形成机制研究

> 🗄️ **迁移说明（2026-08-23）**：本文档来自已弃用的 predecessor codebase。
> **§3.1a「标准数据消融基线（baseline_input）」是本课题极简 setting 的原始定义**，已提升为
> `agents.md` §1（SSOT）。冲突时以 `agents.md` §1 为准。
>
> ⛔ **本文件的 §1 / §2.1 / §3.1 / §3.2 / §5 大量引用 `baseline_current`（current shell）**，
> 那是废弃分支（见 `agents.md` §6）。阅读时请把这些段落当作历史，不要当作结论。
>
> 配套文件（本文件只做索引与状态跟踪）：
> - 理论推导：`../theory/`（unigram gap、幂律机理、Markov、真实长尾修正）
> - 文献与故事线：`plan-2-literature-story.md`、`../literature/`
> - 断言台账：`../claims-ledger.md`
> - **权威报告**：blog `ngram-gap-mechanism-guide/index.html`（9 章极简主线）
> - ⛔ 已废弃报告：OPHIS `docs/report/versions/guide-full-chapter0-19.html`（chapter 0–19 全量版）
> - ⚠️ 断链（图）：文中引用的 `docs/figs/fig_p12_*`、`t5*`、`fig14_7_*`、`docs/interactive/*`
>   全部是 current-shell 时代的图，**未随迁移带入，且其数据源已按「bug 内容彻底删除」原则清除**。
>   这些图不会恢复；对应问题的重跑队列见 `plan-3-fix-and-backfill.md`。
> - ⚠️ 断链：原引用的 `../ngram_gap_theories.md`、`../ngram_gap_ob-th.md`、`../article_plan.md`、
>   `../architecture_gap_experiment_matrix.md`、`../current_vs_nanogpt_bottomup_diff_plan.md`
>   位于 OPHIS 父目录（`OPHIS/`），未随本次迁移带入。需要时去旧仓库查阅。

## 1. 现象定义（要解释的目标）

在含可训练 n-gram value memory 的 nanoGPT/current-shell 模型中，小训练集、固定顺序、多 epoch replay 时：

- **train loss 阶梯下降**（epoch 2/3 边界出现 cliff），**val loss 阶梯上升/翘起**，形成 train/val gap。
- 关键时序：epoch 边界是数据分布切换点；gate/table norm 的上升与 gap 出现**时间对齐**。
- 统一基线 setting（`baseline_current`）：current shell + current-style grouping + bigram+trigram RMSProp，seed42，1000 step，vocab=8192，n_layer=8，epoch 边界 [338, 687]，epoch3 gap 从 ~0.5 扩大到 1.33。
  - 产物在 `remote_training_runs/baseline_current/`（含逐 step norm、allgram 频次分解、频次探针）。

## 2. 理论体系（我们要建立的）

核心假设：**gap 来自 trainable n-gram value memory 在 fixed-order replay 下形成的 train-specific memory advantage**。

### 2.1 机制链（公式级，详见 `../ngram_gap_theories.md` §2）
一次 occurrence `i=(x_i,y_i)`，n-gram 分支 `a` 读 table row `b_{i,a}`：
- logits 残差注入：`z_i = z_i^0 + Σ_a Ψ_{i,a} E_{b_{i,a},a}`（Ψ = reader 通道）
- 一次 update 对 row 的写入：`ΔE_{b,a} = -η_{E,a} P_{b,a} Σ_{i∈B_t} J_{i,a}^T r_i`
- 因此同一 row 后续被读时产生 **self/cross memory kernel**；训练样本的累计 row history（writer-local self-kernel + ever-seen row coverage + 历史 row 内容 + branch-specific reader transport）构成 val 没有的功能性优势。

### 2.2 频率↔gap 核心假设（最重要的 observable 关系）
- 罕见 n-gram（低频 bucket）train 过拟合、val 翘起 → gap 大；高频 bucket（5k+）gap 贡献几乎为 0。
- trigram val novel 占 ~30%（大量 val context 从未出现在 train），是 val 翘起的主要来源；bigram val novel 仅 ~4%。
- 启示：避免过拟合低频 context（如只 over-encode 高频 token 组合 / 无碰撞高频表 / BPE 式合并）。

### 2.3 边界事实（不能过度宣称）
- 旧的 `NanoGPTOriginal` 组件扫描曾显示：在当时的 reader/optimizer/injection 组合下，单独加入 n-gram 未产生目标 gap；这条结果保留为历史边界，不应外推为所有协议的必要条件。较新的 `baseline_input` 与 matched vanilla-graft formal 结果支持更窄的表述：trainable n-gram value memory 在已验证协议内足以诱发 replay-specific gap，但不能据此宣称对所有 plain NanoGPTOriginal 或 shell 组合都成立。

## 3. 实验框架

### 3.1 统一基线
所有 §7.9/§9 的图必须与 `baseline_current` 同一 setting；频次用精确离线索引 `GlobalNgramFrequencyIndex`（不再有 hash 碰撞下限问题）。

### 3.1a 标准数据消融基线（baseline_input，2026-08-05 确立）— ✅ 已提升为 `agents.md` §1 SSOT

> **后续所有数据侧消融（频率遮罩、低频去除、order 对照、row reset/freeze 等）一律基于此 setting，不再用 current-shell baseline。**
> 理由：注入点消融（v/y/input，2026-08-05）证明 vanilla nanoGPT + input 注入即可复现 gap（1000 步 gap≈0.64），不需要 current shell / Muon / current optimizer grouping。input 注入是 over-encoding 风格（Engram/SCONE/Over-Tokenized 主流做法），n-gram value 加到 `wte` 输出，不走 attention。

**模型与训练**：

| 项 | 值 |
|---|---|
| `ARCH_VARIANT` | `nanogpt_original`（vanilla nanoGPT / Karpathy） |
| `NANOGPT_NGRAM_INJECTION_POSITION` | `input`（over-encoding：n-gram value 加到 `wte` 输出，不走 attention） |
| `NANOGPT_ENABLE_NGRAM_VE` | `1` |
| `ENABLE_UNIGRAM_VE` | `0` |
| `ENABLE_BIGRAM_VE` | `1` |
| `ENABLE_TRIGRAM_VE` | `1` |
| `ENABLE_FOURGRAM_VE` | `0` |
| `NANOGPT_NGRAM_OPTIMIZER` | `mixed`（n-gram table 用 RMSProp，backbone 用 AdamW） |
| `NANOGPT_NGRAM_INJECTION_IMPL` | `nanogpt` |
| `NANOGPT_ATTENTION_IMPL` | `fused` |
| `NANOGPT_ADAM_LR` | `0.004` |
| `NGRAM_TABLE_BETAS` | `0.0,0.99` |
| `NGRAM_TABLE_LR_SCALE` | `1.0` |
| `POSITION_ENCODING` | `learned_abs` |
| `CURRENT_NORMALIZATION` | `layernorm` |
| `CURRENT_EMBEDDING_TYING` | `tied` |
| `CURRENT_EMBEDDING_INIT` | `nanogpt_like` |
| `CURRENT_BLOCK_INIT` | `nanogpt_style` |
| `CURRENT_NGRAM_INJECTION_IMPL` | `none` |
| `WINDOW_PATTERN` | `LLLL` |
| `SEED` | `42` |
| `MAX_TRAINING_STEPS` | `1000`（延长对照用 `2000`） |
| `DEVICE_BATCH_SIZE` | `72` |
| `TOTAL_BATCH_SIZE` | `147456` |
| `TRAIN_DATA_MODE` | `fixed`，`TRAIN_DATA_SEED=42` |
| `TRAIN_SHUFFLE_BUFFER_SIZE` | `8192` |
| `TRAIN_REPLAY_NEW_STEPS` | `50` |
| `TRAIN_REPLAY_STEPS` | `50` |
| `TRAIN_REPLAY_SOURCE_PASS_STEPS` | `0` |
| `VAL_LOSS_INTERVAL_STEPS` | `10`（v10 细曲线，2026-08-06 起） |
| `VAL_LOSS_BATCHES` | `4` |
| `LR_SCHEDULE_MODE` | `baseline` |
| `NGRAM_GLOBAL_FREQUENCY_MODE` | `baseline` |

**运行位置**：目标机器上的本仓库副本，由 `NGLAB_ROOT` 指定。
**launcher**：本仓库 `code/cluster/` 下的对应入口。
**代码开关**：`NANOGPT_NGRAM_INJECTION_POSITION=input`（2026-08-05 新增于 `train.py`）

**1000 步 gap 参考**（seed42）：input 注入 final gap ≈ 0.64；y 注入（对照）≈ 1.82；v 注入（旧 B 段，信号被 V 淹没）≈ 0.60。

**norm 诊断**（初始化时 per-token）：v 注入 n-gram residual 只有 V 的 6.5%（信号被淹没）；input 注入 n-gram residual 是 wte 的 4.77 倍（信号充分）。

### 3.2 架构差异矩阵（bottom-up vs current shell）
见 `../architecture_gap_experiment_matrix.md`：逐组件加入（n-gram table → RMSProp → Muon → RoPE → RMSNorm → untied/current init → split q/k/v → current-style grouping/injection → current shell），定位「从哪个组件开始出现目标 gap」。剩余差异表与推荐消融顺序见 `../current_vs_nanogpt_bottomup_diff_plan.md`。

### 3.3 因果拆解（P0–P7，详见 `../article_plan.md` §5）
- P1：historical row-state 因果（row reset/rollback、freeze table/reader、epoch-lag matched probe、history ablation）
- P2：table write vs reader/backbone 拆解（freeze table、freeze reader/backbone、stop-gradient table read、first-pass-only update）
- P3：direct kernel 预测校准（kernel 矩阵、谱量）
- P4：matched probe 与 evaluation artifact 控制（train/val probe 对齐）
- P5：必要机制消融（injection 关闭、high-order n-gram 关闭、unigram 关闭等）
- P6：order specificity、collision 与 hash artifact（table-size/hash-salt sweep、collision-free dictionary、overlap 直方图）
  - ✅ 2026-08-11：table-size sweep（t5_low, M=1/4/8/16/64/256）已完成并验证——**碰撞区（M<16）内 gap 随表大小涨（低频涨落 pooling 稀释），无碰撞点 M=16 后饱和，参数量再多也无用**。详见 `docs/_archive/docs/table-size-sweep-results-20260811.md`。剩余：hash-salt sweep、collision-free dictionary、overlap 直方图
- P7：replay schedule、phase diagram 与 mitigation（old-only/new-only replay、ratio sweep、row clipping、gate regularization、table freeze schedule）

### 3.4 干预实验（是否可消除 gap）
- exp4：hash reseed（`ongoing_experiment/exp4_5_results/exp4_hashreseed_*`）✅ 已验证 −93% 双 shell；统一口径（baseline_current）0.282（−79%，2026-08-02）
- exp5：低频 gate 清零（`exp5_lowfreq_gatezero_*`）✅ 已验证：nanogpt_original −94%、current-shell −58%；统一口径 1-200 无效（1.553，novel 不在范围内）
- exp6：精确重复次数 vs gap 细粒度分桶（`remote_training_runs/exp6_freqdecomp_current/`）— 统一口径 @1000 gap 1.502 与 baseline 同量级（口径差异来自 survey 探针）
- freeze ngram+gate：郭绍阳验证无 gap，vbird 证伪（差异可理解，需记录）

### 3.5 观测集
正式 run 只保留 ob-th-11 列出的几十条关键 series（direct kernel advantage / synergy / spectrum / gap integrand / write / compact rows / row history / token identity / history ablation / compact reader / return coherence），见 `../ngram_gap_ob-th.md` §ob-th-11。

## 4. 数据与实验状态

### 4.1 已验证（有数据支撑）
- [x] 无 n-gram → 无 gap
- [x] 两个 epoch 数据混合训练 → 仍有 gap
- [x] 频次分解：罕见 n-gram gap 大，5k+ 高频 gap 贡献 ≈ 0
- [x] trigram val novel ~30%（val 翘起主因）；bigram ~4%
- [x] gate/table norm 上升与 gap 时间对齐（RMSProp 与 +Muon 两组）
- [x] froze ngram+gate → 郭绍阳侧无 gap（vbird 侧证伪，记录在案）
- [x] epoch2/3 fork 时间差异（table RMSProp：epoch3 才 fork；+Muon：delayed gap）

### 4.2 已验证结果与可选补强
- [x] 每个 epoch 的 n-gram 查询用独立 hash 错位 → 同一 context 查不到 → 预期显著降低 gap（✅ 已验证：guide §10 exp4，nanogpt_original 0.688→0.047、current-shell 3.658→0.252，均 −93%）
- [x] BPE/over-encoding 的低频去除 proxy —— gate-zero 代理（exp7，含 novel 的 0-200/0-1000 低频 gate 清零）
      2026-08-02 验证<strong>失败</strong>：final gap 1.28/1.69 vs 控制 1.10，低频桶 gap 不降反升；机制：gate 输入侧无频率信息，
      清零后记忆转移给 backbone（固定顺序 replay 下 train 更低、val 不变）。<strong>真正 BPE 合并高频组合/低频共享行</strong>
      已由 2026-08-04 的 token 重映射 proxy 验证（guide §16.11：T=3000 −43%、T=8000 −67%、长时程 2000 步稳健 0.28）；真正 BPE/Engram/Over-Encoding 直测仍未做。
- [x] 合成 toy 数据集（高频+低频 n-gram、可控 4-gram 转移矩阵）→ 浮现 forking，细粒度统计频率↔fork 关系（2026-08-02 完成，见 `docs/toy-dataset-results.md` 与 `plans/plan-1-toy-dataset-design.md` 实现状态）：
  - 15/15 runs 复现末 epoch gap（6.5–8.7 nats）；gap(r) 单调（Spearman ≤ −0.9）；r1/r5120 gap 比 ≥63×；novel 解释 val 翘起 96–100%
  - 边界条件：reshuffle（保持 per-context 频率不变）不减少 gap——提示驱动因素是 replay 次数而非位置；字面 Pearson(gap,log r) 仅 n2 达标，幂律形状用 Spearman/log-log ρ 更合适（3 项判据待用户裁定）
- [x] row reset/rollback、freeze table/reader 的因果验证（P1/P2 主线）— 2026-08-02 第一波完成，见 `docs/_archive/docs/p12-causal-results.md`：
  - e2 边界全行回滚 → final gap 0.121（vs 控制 1.096，−89%）；readout mask → 0.116（−89%）
  - freeze table@e1 → 0.559（−49%）；freeze reader/backbone@e1 → 0.507（−54%）；freeze gate → 0.865（−21%）
  - 图表 `docs/figs/fig_p12_gap_curves.svg` + `docs/interactive/fig_p12_causal.html`；run_exp.sh 新增 case（集群）
- [x] P1/P2 3-seed 补跑完成（2026-08-04，guide §12.1/§12.4）：freeze table@e1 0.559/0.656/0.564（−50%）、freeze reader/backbone@e1 0.507/0.580/0.374（−59%）、freeze gate@e1 0.865/1.234/0.838（−19% 弱）、
      ref/rand 小规模行回滚 0.992/1.391/1.076、1.126/1.051/1.332（跨 seed null，排除操作 artifact）——全表级干预稳健，特异性只到「全表内容」级
- [~] P1/P2 第二波（可选）：epoch-lag matched probe、frequency-stratified row zero（需要 epoch1 fork checkpoint 工作流）
      —— 早期 3 seeds 稳健性（2026-08-02）：`p1_reset_all_e2` 0.121/0.123/0.122、`p2_readout_mask_e1` 0.116/0.121/0.126；
      `p2_freeze_both_e1` 0.572 ≈ 仅冻结 table，gate 不叠加
- [x] **历史 toy v5 2×2 干净证明（ngram 表 × 低频键，2026-08-04）**：32768 键、low（30720 键 r<16 train/val next 独立 + 1024 共享）/ high（全 r=64 共享）两数据模式 × 注入 on/off × 3 seeds = 12 runs（所需历史 metadata 未随本仓库迁入）：
  - headline gap（val−train，3-seed 均值±std）：on_low **6.92±0.43**（train 0.004/val 6.80）＞ off_low 2.12±0.003（backbone 底噪）＞ on_high 0.84±0.26（核心键 ≈0.001，0.84 为附带位置效应）≈ off_high 0.064±0.000（干净基准）
  - per-r 精确分桶呈**阶梯函数**：on_low r<16 gap 15–20、r≥16 ≤0.005，断崖恰在 r=16 共享阈值 → 表只在低频键上制造 gap（图 `docs/figs/t5_step_function.svg`）
  - 结论：低频 token + n-gram 表 + 固定顺序重播 三者齐备 → 必然过拟合 gap；去掉任一显著缓解（去表 −69%，去低频核心键 gap→0）。数据为历史结果，guide §16
  - [x] toy 级因果干预（on_low 断表口子）：readout mask@e1 → 2.12 = off_low 逐位相等（−69%，表贡献全走 readout）；freeze table@e1 → 2.49（−64%）；reset@e2 → 7.90（无效，toy 重播 29 epoch 回滚后重新累积）——与真实 P1/P2 互证（guide §16.7）
  - [x] off_low 长训对照：8000 步（104 epoch）gap 只到 3.36，r1 低频键 gap 涨到 20.7——低频键过拟合无表时同样存在，表 = train 完全塌陷的放大器（guide §16.8）
- [x] **P0 wave3 + no-ngram 对照（2026-08-04，guide §15.6，baseline_current seed42）**：
  - comb（只留中频 201–1000）last100 gap 0.148 ≈ 单遮 0–200 → 低频遮罩 gap 缩减 + 高频性能代价近似可加（复现 svbird F）
  - rowzero freq-peak/large 1.175±0.090 / 1.179±0.087 无效（3 seeds：s42 1.096/1.108、s43 1.157/1.152、s44 1.273/1.277；e1 边界 gap_contribution 峰值仅 +0.019 nats）→ 支持「滞后行」机制（e2 写入 e3 重读，对应 p1_reset_all_e2 −89% vs e1 −13%）；comb（0-200+≥1001）3 seeds = 0.149±0.002 ≈ 单遮 0-200（0.155±0.004），高频遮罩只加性能代价
  - replay4/6 → 3.47/6.17（重播次数 = 干净剂量）；shard2/3 → 6.45/8.09（预算越多塌越狠）
  - **no-ngram 对照**：replay6/shard2/shard3 关 ngram 注入 → last100 gap 0.056/0.029/0.020（−99%~−99.8%）；**同 setting 1000 步**（baseline_nongram，3 seeds）→ final gap 0.125±0.001（−90%）→ gap 爆炸全部由表贡献
- [x] **t5f 燃料剂量扫描（2026-08-04，guide §16.10，12 runs）**：巧合低频键占比 0/25/50/75/93.75% → per-context (seen) gap 单调 0.005→0.86→2.36→5.33→7.08；headline（shuffled val）被顺序混杂抹平 ~7 → 引出 same-order 公平考卷
- [x] **toy same-order 公平考卷 2×2（2026-08-04，guide §16.9，12 runs 完成）**：`--val-order same` = val 与 train 同流仅巧合键换答案 → 附带上下文一致，headline 即干净口径（只度量「训练记住 A、验证问 B」的纯记忆错误）。4 象限 × 3 seeds：
  - on × low（表+低频键）**3.94 ± 0.23**（train 0.010/val 3.95）；off × low（无表）1.40 ± 0.002（train 2.57 背不动）；on × high16（全共享）0.000（train 0.02 完美背下且 val 问同样问题）；off × high16 0.000（train=val=3.56）
  - 结论：只有「表 + 低频键」同时在场才有大 gap；去掉任一显著缓解（表→1.40、低频→0.000）——论文主 claim 的完整 2×2 支撑。图 `docs/figs/t5s_2x2.svg` + 交互 `docs/interactive/fig_t5s_2x2.html`
  - [x] same-order frac 剂量（t5sf，9 runs：25/50/75% × 3 seeds）：headline = 0.40/0.98/2.56（+ low 3.94）完美单调（guide §16.10 表格新增 same-order 行）
- [x] **BPE/over-encoding 干预验证（真实数据去低频，2026-08-04，guide §16.11）**：gate-zero 代理失败（exp7）后改用 token 重映射（`tools/remap_rare_tokens.py`，稀有 token→最高频 token + 解码重编码 fixpoint，train+test 全 shard 对称处理）。bigram 唯一上下文 4.03M → 1.84M（−54%）→ 0.54M（−87%）；trigram 22.8M → 14.2M → 5.8M。baseline_current 同 setting（seed42-44、1000 步、epoch 边界 [337-338, 685-687] 几乎不变）：
  - epoch-3 final gap（3 seeds mean）：baseline **1.21±0.28** → T=3000 **0.69±0.20**（−43%）→ T=8000 **0.40±0.05**（−67%）——去除强度与 gap 塌缩单调（toy 剂量在真实数据上复现）
  - 与 no-ngram（≈0）、P0 低频遮罩（−86%~−92%）、svbird F 三线收敛：表 + 低频 token 是 gap 的充分组合，去低频可缓解
  - [x] 长时程对照（2000 步 ≈ 6 epochs，seed42，2026-08-04）：baseline_long 1.07（与 1000 步持平，epoch3 后平台）；nofreq_t3000_long 1.23（部分去除只推迟，剩余中低频键在更多重播下反超 baseline）；nofreq_t8000_long **0.28**（激进去除稳健，键空间 −87% 后无「背错」燃料）——去除强度必须大到键空间塌缩才能长时程抑制

### 4.3 数据位置
- 本地旧结果：`remote_training_runs/`（baseline_current、exp6_freqdecomp_current）、`ongoing_experiment/`
- Explain 分支根目录 `remote_training_runs/2026072x_*`（vbird 的消融 run，含 fetched/ 源码+log）
- 大日志 `remote_training_runs/20260725d/e_*`（~6.3G）未迁入本仓库，保留为外部历史资产
- 集群活跃工作区：由 `NGLAB_ROOT` 指定（含 `runs/`、`run_artifacts/`、`run_exp.sh`）

## 5. 可视化清单与状态（用户逐项检查）

### 5.1 报告章节（`docs/report/versions/guide-full-chapter0-19.html`）
| 章 | 内容 | 状态 |
|---|---|---|
| §1 | n-gram table / hash / gate / shell / optimizer grouping 术语 | ✅ 保留 |
| §2 | Gap 现象与因果链（baseline_current） | ✅ 统一 setting |
| §7.9 | Table Norm × Gate：loss↔norm 时间关系（复合图） | ✅ 统一 setting |
| §9 | 频次分解分析（trigram/bigram 分桶 gap，novel..5k+） | ✅ 全桶有数据 |
| §10 | 干预验证（exp4/exp5） | ⚠️ 待更新到统一 setting |
| §11 | 精确重复次数 vs gap（exp6 细粒度分桶） | ⚠️ 检查中 |
| §15.0 | svbird F 实验本身（人话完整版） | ✅ 2026-08-04 新增 |
| 故事页 | `ngram-gap-story.html`（从零讲起、本地为主+svbird 合作验证） | ✅ 2026-08-04 新增，guide 顶部有入口横幅 |
| §15.1–15.4 | svbird 结果评估 / 论文组织 | ✅ 已闭环 |
| §15.5–15.6 | P0 复现 + wave3（3 seeds、no-ngram） | ✅ 2026-08-04 完成 |

### 5.2 图表文件
- `docs/figs/`：fig14_*（§14 系列）、figB*_norms/freqloss/alignment（norm↔loss 对齐）、fig_gap_by_freq、fig_hitcount_dist、fig_loss_norm、fig_exp6_freq_gap、p0_freqmask_*（§15.5/15.6）、nofreq_gap{,_long}（§16.11）、t5s_2x2 / t5f_dose / t5_*（§16）、p12_3seed_bars（§12.1 新增）
- `docs/interactive/`：fig14_interactive、figB*_interactive、fig_gap_by_freq、fig_hitcount_dist、fig_loss_norm、fig_exp6_freq_gap、fig_exp45_results、fig_p0_freqmask、fig_nofreq{,_long}、fig_t5s_2x2

### 5.3 待办作图（manual.md 遗留）— 2026-08-01 完成
- [x] 图 14.7（gap contribution = frac × (val−train)）：y 轴 0 基线、负值向下延伸
      → `docs/figs/fig14_7_gap_contrib.svg`（bigram/trigram 两面板静态图）+ `docs/interactive/fig_gap_by_freq.html`（gap+总贡献视图）。
      0 基线固定画出并标注 0；无负值时 y 轴最低点 = 0，有负值时向下留 8% 空间。
      总贡献口径 = **val token 占比 × (val−train)**（novel 桶不会被 0 占比掩盖；trigram novel 总贡献 0.34 最大）。
- [x] hit count 分布图 + 累积分布图（低频→高频，展示平均每 token 重复次数）
      → `docs/interactive/fig_hitcount_dist.html`（分桶柱状 + 累积分布，train/val split 切换；
      新增「平均每个 token 重复次数」统计行：bigram train 5019 / val 5160，trigram train 163 / val 176，含 q10/q50/q90）。
      数据 `docs/figs/fig_hitcount_stats.json`（新文件，来自 global_frequency_probe，未改动原 JSON）。
- [x] 所有 gap 图自动带 val 和 train 曲线；支持 per-token 与总贡献切换
      → `fig_gap_by_freq.html`：gap 视图现在三线合一（gap 桶色粗线 + train 蓝细线 + val 红虚细线），
      per-token/总贡献可切，总贡献的 frac 按曲线分别取 train_frac / val_frac；
      `fig_loss_norm.html` 与 `fig_loss_norm_pair.html` 的 gap 视图同样带 train/val 参考线。
- [x] 5k+ 高频 bucket 展示 → 数据存在，未跳过。trigram 5k+ 总贡献 ≈ 0.006（novel ≈ 0.34，最大），
      bigram 5k+ ≈ 0.129（val 占比 ~12%，贡献非零但仍小）；已在 `fig_gap_by_freq.html` note、
      `fig14_7_gap_contrib.svg`（5k+ 深灰粗线）与 hitcount 柱状图中展示。
- [x] RMSProp 与 +Muon 的 norm/loss 时间对齐图（epoch3 fork 与 delayed gap 两组）
      → `docs/interactive/fig_loss_norm_pair.html`（两组并排双轴 loss↔norm + per-epoch 变化率表 + 对齐判定；
      RMSProp：gap 峰值 e3 与 trigram gated RMS 增长峰值 e3 对齐；+Muon：norm 峰值 e1 先行、gap 迟到 e3（delayed gap））。
      数据 `docs/figs/fig_loss_norm_pair.json`；baseline_current 无 +Muon run，故用同配置 B 系列
      `../remote_training_runs/20260725e_nano_rmsprop_fixv3` 与 `_nano_muon_fixv3` 两组对照（nanogpt_original+full n-gram，
      vocab=8192，n_layer=8，seed42，1000 step），已在页面 note 说明。

## 6. 下一步待办（优先级排序）

1. ✅ svbird F 频率遮罩统一框架复现（P0，2026-08-04 完成，guide §15.5/§15.6/§17）：baseline_current + 3 seeds，comb/rowzero 3-seed 补全，no-ngram/shard/replay 对照完成
2. ✅ toy v5 2×2 + same-order 公平考卷 + 剂量扫描（2026-08-04 完成，guide §16.9/§16.10，24+ runs）
3. ✅ BPE/over-encoding 干预验证（2026-08-04 完成，guide §16.11）：gate-zero 代理失败后，token 重映射代理（T=3000/8000）+ 长时程对照完成
4. ✅ P1/P2 因果 3-seed 补跑（2026-08-04 夜跑，10 runs 完成）：freeze table/reader/gate + ref/rand 行回滚 × s43/s44 全部跑完并分析，guide §12.1/§12.4/§17 与 compact_table 已更新
5. （可选）P1/P2 第二波：epoch-lag matched probe、frequency-stratified row zero（需 epoch1 fork checkpoint 工作流）
6. 用户检查可视化反馈 → 修图
