# Plan 2 · 文献调研与故事线

> 配套文件（细节索引）：
> - 调研综述：`docs/ngram-memory-literature-review.md`（方案空间：防碰撞 / SCONE / 频率筛选 / TN-gram）
> - 批判框架：`docs/ngram-literature-critique.md`（Over-Encoding / Engram / LongCat / BLT / N-Grammer 的低频+collision 风险）
> - 架构对照：`docs/scone-vs-engram-optimization.md`、`docs/scone-engram-architecture-pseudocode.md`
> - 文章计划（vbird）：`../article_plan.md` §7 相关工作 + §8 新颖性

## 0. 目标

把「n-gram 记忆方法的最优设计与泛化关系」调研清楚，形成论文的故事线：**我们的频率条件化 gap 分析能同时批判方法设计与评价协议**。

## 1. 核心调研问题

1. **什么样的 n-gram 是最优的？** 表对哪些 key 精确（全部 V^n / 语料中出现过 / 高频筛选）？向量如何学习（独立参数 vs 共享模型 vs 低秩共享）？长度怎么支持（固定 K 最长匹配 / 无界外部检索）？
2. **n-gram 与泛化的关系？** 频率筛选是否数学上合理；高频是否等于高价值（信息量标准）；低频/长尾的 train/val gap 机制。
3. **常见 baseline 是否遇到低频 token 并产生 loss 翘起/gap？** 哪些方法显式依赖 hash（collision 风险）；哪些有低频独立参数（sparse-update 风险）；各自的可批判点与最直接实验。

## 2. 已有调研成果（结论先行）

### 2.1 方案空间（详见 `ngram-memory-literature-review.md`）
- 三种防碰撞：全组合空间无碰撞表（不现实）/ 语料中出现过的 n-gram 无碰撞表 / 只给高频 n-gram 建无碰撞表。
- **SCONE**（Google）最接近我们的想法：f-gram 集合 + 保留 base tokenizer + 精确 key（无 modulo hash）+ 共享 f-gram Transformer 学表示 + 推理外部 KV 查表 + 最长匹配回退。→ 结论：问题不是「防碰撞是否可行」，而是「对哪些 key 精确、向量怎么学、长度怎么支持、高频是否等于高价值」。
- TN-gram：低秩共享结构绕开显式 hash collision，但有 rank bottleneck 与长尾训练分布问题。
- Infini-gram：无界 n 的外部检索路线（非端到端可训练 embedding table）。
- BPE 不等价：BPE 改变离散化；f-gram memory 保留 base tokenizer；贪心 merge 与最长匹配不同。

### 2.2 批判框架（详见 `ngram-literature-critique.md`）
批判点分两类，且不完全一致：
- **低频/长尾训练信号不足**：命中太少 → 参数估计不稳 → 重复 epoch 中快速记住训练模式。
- **hash collision**：不同 n-gram 共享 row → 表示污染、梯度干扰、gate credit-assignment 问题。
- 精确表可消除 collision 但加重 sparse-update；hash 表共享 row 增加统计量但混合 context；SCONE 缓解低频独立参数；Engram / Over-Encoding / BLT / LongCat 显式依赖 hash；TN-gram 低秩；**X-gram 已把 Zipf 长尾 under-training 和 slot collapse 明确列为问题——与我们频率分析最直接相关**。

核心论断（可同时批判设计 + 评价协议）：
> 现有 n-gram memory 方法按 lookup collision 或平均 loss 评价容量，但没有充分区分 n-gram frequency、row collision multiplicity、gate weight 与 route-specific train/validation loss。频率条件化的 gap analysis 可以揭示：同样的平均验证损失背后，可能存在完全不同的长尾记忆化和碰撞干扰机制。

### 2.3 方法对照表（正式版 · 2026-08-02 原文复核）

> 盘点协议：对每个方法回答四个问题——① 表/记忆结构（精确 / hash / 外部检索 / 低秩）；② 是否报告 train/val 分离 loss；③ 是否按频率分桶分析过 loss/gap；④ 对低频 n-gram 的明确处理。
> 核实方式：2026-08-02 对 9 篇论文的 arXiv HTML 原文逐篇检索（train/val loss、frequency、Zipf、bucket、long-tail 等关键词），标注 ✅ 确认 / ⚠️ 部分或待核实 / ❌ 未发现。引用元数据（编号/作者/年份）已全部经 arXiv API 复核（见 `docs/literature/references.bib` 与 §6）。

| 方法 | ① 表/记忆结构 | ② train/val 分离 loss | ③ 频率分桶 loss/gap 分析 | ④ 低频 n-gram 明确处理 | 我们可批判的点（详见 critique） |
|---|---|---|---|---|---|
| **Over-Encoding**（Huang et al., ICML 2025, arXiv:2501.16975） | 1..N 阶 n-gram → 固定 modulo hash 表，多低维 sub-table 求和 | ✅ 主表显式并列 Train Loss/PPL 与 Eval Loss/PPL + downstream | ❌ 未见 | ❌ 无：deterministic hash 与频率无关 | table size 不能说明容量给了高频有用 context 还是长尾 collision 伙伴（§3.3） |
| **Engram**（Cheng et al., 2026, arXiv:2601.07372；⚠️ 第一作者 Xin Cheng，此前笔记写「Tian et al.」有误） | 每 order×head 固定容量 hash 表 + hidden-state gate 中间层注入 | ✅ 主表并列 Pile（训练域）loss 与 Validation Set loss | ❌ 未见；multi-level cache 只按访问频率分层（系统级） | ⚠️ 承认 n-gram Zipf 长尾；长尾放慢速存储，不解决训练统计可靠性 | scalar gate 无法恢复 collision 前的独立身份，可能放大 train shortcut（§4.3） |
| **LongCat N-gram Embedding**（Liu et al., 2026, arXiv:2601.21204） | polynomial rolling hash + 多 sub-table 分解，注入 Longcat-Flash | ✅ 明确监控 training loss + 两个 validation loss | ❌ 未见 gap 分桶；有 collision 数 vs vocab size 分析 | ⚠️ 增大 table/分解缓解 collision；作者显式承认非平滑 collision regime | collision 敏感性与频率/row-load 交互未拆解（§5） |
| **BLT**（Pagnoni et al., ACL 2025, arXiv:2412.09871） | byte n-gram (n=3..8) → RollPolyHash 固定表；附录曾试 frequency-based top-100k 表 | ⚠️ tokenizer-independent PPL/bpb（Train Dist + held-out）；未见 train/val loss 分离 | ❌ 未见 | ✅ 附录明示：frequency-based 表「infrequent n-grams not being represented at all」→ 放弃改 hash；低频明确不保留 | byte 频率分布 ≠ token n-gram；hash 低频行为未分析（§6） |
| **N-Grammer**（Roy et al., ICLR 2023, arXiv:2207.06366） | PQ 离散 latent code 的 bi-gram → 每 head 独立 universal hash 表 | ⚠️ C4 log-PPL + SuperGLUE；未见 train/val 分离 | ❌ 未见 | ❌ 无显式低频处理（universal hashing 保证低碰撞概率） | latent code 组合空间的 collision 与 quantization error 混在一起（§7） |
| **SCONE**（Yu et al., NeurIPS 2025, arXiv:2502.01637） | 精确高频 f-gram 集（min-count 5 剪枝、top-S 预算、最长匹配）+ 共享 f-gram Transformer + 推理 frozen 精确 KV 表 | ⚠️ validation split PPL（WebText/WikiText-103）；未见 train loss 并列；K 截断 K=2→7、K=8→108 | ❌ 未见 gap 分桶；有「K 增大 → 下游匹配率下降」分析（频率-效用问题被部分承认） | ✅ 频率阈值显式砍掉低频；作者承认长 f-gram 更稀有、匹配率更低 | 纯频率筛选无信息量判据；train/val support mismatch 未检查；SFT 应用无多 epoch/冻结分析（§8、scone_notes） |
| **TN-gram**（Zhou et al., 2026, arXiv:2606.08347） | CP 分解低秩共享因子，跨 order 参数共享；无显式 hash collision | ✅ 报告 training loss + validation loss | ❌ 未见 | ❌ 无显式低频处理；rank bottleneck 限制表达能力 | rank 受限 + 高频因子主导低频 tuple 表示（§9） |
| **Infini-gram**（Liu et al., COLM 2024, arXiv:2401.17377） | suffix array 外部检索；非参数统计；无界 n；无学习 row | ⚠️ held-out PPL（带 decontamination）；无 train/val 分离 | ❌ 未见 | ⚠️ 稀有长匹配有覆盖但估计方差大；无训练信号问题 | train/val support mismatch 仍存在；适合做检索上界对照（§11） |
| **X-gram**（Chen et al., 2026, arXiv:2604.21724；补充行，笔记整合见 §2.4） | 频率感知混合哈希：高频 VIP 独占物理行 + 长尾 alias 共享桶 + 多尺度 ShortConv + gate 注入 | ⚠️ val PPL + downstream；有 training PPL 轨迹；全文无 generalization/OOD 分析 | ❌ 频率分析是 token 级行利用率（激活幅度、行更新次数），非 n-gram loss/gap 分桶 | ✅ Zipf 长尾欠训练 + slot collapse 显式列为问题；稀疏感知 LR 放大低频行步长 | 行利用率 ≠ 泛化 gap；频率错配（train/replay/val）下 alias 路由可能加剧 gap（xgram_notes） |

**盘点结论（支撑「评价协议缺口」论断）**：

1. 表结构分三类：**hash 可训练表**（Over-Encoding / Engram / LongCat / BLT / N-Grammer / X-gram 尾部）、**精确高频表**（SCONE）、**低秩共享**（TN-gram）、**外部检索**（Infini-gram）。
2. 8 个核心 baseline 中 5 个报告了分离的 train/val loss（Over-Encoding、Engram、LongCat、TN-gram 明确 ✅；SCONE/N-Grammer/Infini-gram/BLT 只报 held-out），但**没有任何一个做频率条件化的 train/val gap 分析**——这是评价协议缺口的事实基础。
3. 对低频 n-gram 的处理只有三种姿态：**砍掉**（SCONE；BLT 附录的 frequency-based 尝试失败）、**系统级缓存**（Engram multi-level cache）、**不处理**（Over-Encoding / N-Grammer / TN-gram）；X-gram 是唯一把低频欠训练与行利用率当作主要优化目标的工作，但其分析停在训练侧。
4. 因此「同平均 val loss 背后可能是完全不同的长尾记忆化 vs 碰撞干扰机制」在现有文献中不可观测：没有方法同时报告 frequency bucket、row collision multiplicity、gate weight 与 route-specific train/val loss。

### 2.4 三份调研笔记整合（2026-08-02）

原始笔记：`docs/literature/xgram_notes.md`、`docs/literature/scone_notes.md`、`docs/literature/posttraining_freeze_notes.md`（subagent 产出，引用已复核，见 §6 bibtex）。

#### 2.4.1 X-gram：行利用率指标如何对照我们的 gap 分析

- **论文定位（已核对原文）**：arXiv:2604.21724《Beyond N-gram: Data-Aware X-GRAM Extraction for Efficient Embedding Parameter Scaling》（Chen et al., 2026）。摘要原话："Zipfian under-training of the long tail, heterogeneous demand across layers, and 'slot collapse' that produces redundant embeddings"。
- **可对照指标**（xgram_notes §可对照的指标）：
  1. 行利用率：物理行更新数 n_j ≈ N·q_j 按频率桶分解，对比 train/replay 下命中集中度 → 与我们已有的「5k+ 高频 bucket gap 贡献≈0、Novel bucket 领先（贡献 ~1.5）」直接对齐；
  2. slot collapse：slot 间 pairwise cosine similarity，检验 replay 是否加速塌缩、塌缩度与 gap 是否相关；
  3. 频率错配：用 train/replay/val 分别估计 p(ω)，量化桶重路由率、alias 碰撞率与「从未更新覆盖率」，测低频行 LR 放大能否降低尾桶 val gap。
- **与我们的分工（关键）**：X-gram 缓解的是「低频 → 欠训练」（训练侧行利用率）；它**不做泛化/OOD/gap 分析**（已核对原文，全文无 generalization/overfit 分析）。我们的频率条件化 gap 分析补上它缺失的验证侧：**行利用率与泛化 gap 不等价**——一个 bucket 更新频繁仍可能对重复训练 context overfit。

#### 2.4.2 SCONE：纯频率筛选的评价协议缺口

- **筛选细节（已核对原文 §3.1/§4.1.1）**：K−1 次扫描统计 n-gram；min-count 5 剪枝；按原始频率降序取 top-S（预算 10M/20M/1B）；K=2 截断计数 7、K=8 为 108（默认 K=5）。
- **协议缺口**：① 纯频率标准，无信息量判据；② 作者承认频率排序使长 f-gram 更稀有、下游匹配率更低；③ 未见 train/val 分离 loss 与频率分桶 gap；④ E.3 在 Qwen3 SFT 上应用（open-r1 Mixture-of-Thoughts），但**无多 epoch/冻结分析**——直接对接我们的启示层结论。
- **高频 ≠ 高价值**（scone_notes 证据链）：surprise/信息量标准（Levy 2008；RSI arXiv:2606.31575；Entropy-UID arXiv:2502.14366；长文档信息密度 arXiv:2309.06009）；影响函数标准（Koh-Liang 2017；TRAK ICML 2023；LESS ICML 2024；FINDR IJCNLP-AACL 2025——用哈希 n-gram + 影响分数选预训练数据，显式以影响而非频率衡量 n-gram 价值）；重采样/过滤先例（Yang-Pedersen 1997；CCNet LREC 2020；DSIR NeurIPS 2023）；长尾仍有预测力（Infini-gram COLM 2024；SILO ICLR 2024）。
- **对我们的意义**：频率筛选在训练分布内偏置，对验证集低频 n-gram 的分布变化没有保护 → 我们把它写成「频率条件化 gap 分析揭示的盲区」，而不是「SCONE 错了」。

#### 2.4.3 后训练冻结启示的证据链

证据链（posttraining_freeze_notes，引用已核实）：

1. **重复数据/多 epoch → 过拟合与记忆化**：Xue et al.（NeurIPS 2023）token 受限下多 epoch 重复引发过拟合与 degradation；Tirumala et al.（NeurIPS 2022）记忆化先于过拟合，容量越大记忆越多、遗忘越慢；Gao et al.（ICML 2023）RLHF 过度优化扩展律（代理奖励上升、真实奖励先升后降、KL 上漂）。
2. **冻结/只读记忆是既有实践**：kNN-LM（ICLR 2020）记忆完全非参数化、「记忆只读、网络可训」先例；BitFit（ACL 2022）仅训 bias 可逼近全量微调；LoRA Learns Less and Forgets Less（TMLR 2024）受限更新「学得更少但忘得更少」。
3. **记忆组件的泛化短板**：Nishida et al.（NAACL Findings 2025）kNN-LM 增益几乎只来自高频 token，对低频/长尾无改善甚至更差（与我们「低频 context 过拟合」观察一致）；Geng et al.（NAACL 2025）kNN 记忆擅长密集任务、推理任务退化。
4. **结论支撑**：可训练 n-gram 记忆在 SFT/RLHF 多 epoch 重复数据下应冻结或转只读检索，避免把后训练分布偏差固化进记忆。证据链完整性：多 epoch overfit（已有）→ 记忆只读先例（已有）→ 低频记忆泛化短板（已有）→ 我们的 gap 机制（本工作补齐）。

## 3. 故事线（论文叙事）

### 3.1 主线与证据状态（2026-08-08 更新）

状态标记：✅ 已验证（有实验日志/图支撑）｜⚠️ 部分验证或结论有争议，需复核｜❌ 未验证（planned）｜🔍 文献证据（外部工作支撑）。

**逻辑一：频率 ↔ gap 是核心 observable。**
1. ① 绘制 gap 与词频关系 → 确实相关（✅ 已验证：plan-1 §2.2/§9 分桶；Exact-frequency baseline（guide §18，2000 步，22.76M distinct exact context，P50=1）gap 与真实 trigram frequency 单调反比；toy per-key gap 在 r=16 断崖（guide §16.3））。
2. ② 干预：去掉低频分量 → gap 降低（✅ 已验证，位点收敛到 readout/数据侧）：① gate=0 干预（guide §10 实验5，hit 1-200）nanogpt_original −94% / current-shell −58%；② P0 频率遮罩 readout（3 seeds，fork 自 step-337）：0–200 −86%、0–1000 −92%、1–5 −26.5%，遮高频 ≥1001 ≈0（性能代价），comb 可加，与 svbird F 逐位一致（guide §15.5/§15.6）；③ exp7（gate-zero 含 novel 的 0 起点）失败——gate 不是频率干预有效位点，频率结构在表/行内容里（guide §12.4）。
3. ③ 验证：低频键空间去除 proxy 后 gap 变小（✅ 已验证：guide §16.11 token 重映射 proxy——真实数据上把 train 全局频次 < T 的稀有 token 重映射为最高频 token，表保持开启、与 baseline_current 同 setting、3 seeds：T=3000（bigram 键 −54%）epoch-3 gap 1.21→0.69（−43%）；T=8000（键 −87%）→0.40（−67%）；长时程 2000 步（≈6 epochs）T=8000 仍 0.28 稳健、T=3000 反超（1.23）→ 去除强度须大到键空间塌缩。exp7 gate-zero 代理失败；真正 BPE/Engram/Over-Encoding 直测未做）。
4. 文献侧（🔍）：BLT 附录的 frequency-based 表因低频无表示而失败、SCONE 显式按频率剪枝、X-gram 以行利用率为目标——都承认频率-泛化关联，但均未做频率条件化 gap 分析（§2.3 盘点结论）。

**逻辑二：gap 来自多 epoch 重复 + 早期 memorization。**
1. ① 无 n-gram → 无 gap（✅ 已验证：no-ngram 同 setting 3 seeds final gap 0.125±0.001（−90%）；replay6/shard2/shard3 no-ngram last100 0.056/0.029/0.020（−99%，guide §15.6））。
2. ② 两 epoch 混合 → 仍有 gap（✅ 已验证：replay 50-50，article_plan §4.4；重播次数是干净剂量——replay4/replay6（3→4→6 epochs）gap 1.07→3.47→6.17（guide §15.6）；shard/replay epoch 对齐后 gap@6pass 在 +0.8~+2.1 非单调，step 对齐的「shard 越大 gap 越小」是重播轮数假象（manual 08-07））。
3. ③ 部分冻结/回滚 ngram readout 可显著降低 gap（✅ 已验证 3 seeds，guide §12.1/§12.4：e2 全行回滚 −89%（0.122±0.001）、readout mask −89%（0.121±0.005）、freeze table@e1 −50%、freeze reader/backbone@e1 −59%、freeze gate 仅 −19%（弱）；ref/rand 小规模行回滚 null，排除操作 artifact）。结论限于当前协议：历史行内容和 reader 放大共同参与，gate-only 不是充分干预；不同协议存在结果差异。
4. ④ 每 epoch 独立 hash 错位 → 预期显著降低 gap（✅ 已验证：guide §10 实验4 `NGRAM_HASH_RESEED_PER_EPOCH=1`（只改映射不改 row 内容）：nanogpt_original −93%、current-shell −93%；统一口径（baseline_current）final gap 0.282（−79%）；toy 每 epoch 换 hash −72%（3 seeds）。机理：跨 epoch row 恒定性被破坏 → writer-local self-kernel 无法累积）。
5. ⑤ 低频 context 去除 proxy 可降低 gap（✅ 已验证：与逻辑一③共享 token-remapping proxy（guide §16.11，T=8000 −67%、长时程 0.28）；真正 BPE/Engram/Over-Encoding 直测未做）。
6. 机制侧（✅ 已观察）：epoch 边界 train cliff + val 翘起；gate/table norm 增加与 gap 形成时间对齐（manual.md 7-22 记录；plan-1 §2.1 公式链：writer-local self-kernel + historical row state）。

**叙事中的证据依赖关系（写论文时按此标注）**：逻辑一①② 是现象与相关；逻辑一③、逻辑二③④⑤ 是反事实干预（现已全部 ✅）；文献侧（§2.3/§2.4）负责把「同平均 val loss 的不同机制」从不可观测变成可观测问题。注意基线口径：guide §12–§17 在 baseline_current，injpos/P0/§18/§19 在 baseline_input（input 注入，2026-08-05 起为数据侧消融标准），toy 为自包含合成口径。

### 3.2 叙事要点
- 现象层：epoch 边界 train cliff + val 翘起；norm/gate 时间对齐。
- 机制层：writer-local self-kernel + historical row state（公式链见 plan-1 §2.1）。
- 批判层：用频率条件化 gap 分析揭示「同平均 val loss 背后的长尾记忆化 vs 碰撞干扰」。
- 协议层：逐 step gap 曲线存在锯齿伪影——主因是 val 重评估间隔（`VAL_LOSS_INTERVAL_STEPS`：旧 50 步 / v10 起 10 步，窗口内 gap 爬升、评估点重置；2026-08-08 在 v10 toy 日志复核：窗口上升均值 0.19）；次因是 epoch 边界真周期（toy ~80 步，自相关 r≈0.56）；「基准值上移」才是结构性 gap。论文图表统一用 val 评估对齐点，并标注 epoch 边界（`docs/loss-curve-sawtooth-audit.md`）。
- 启示层：训练/后训练中 over-encoding 组件应被冻结（SFT/RL 多 epoch 重复训练会导致 overfit）。

### 3.3 与 vbird 文章计划的分工
- vbird `article_plan.md`：机制理论（Replay Self-Kernel）、P0–P7 因果拆解、投稿目标（arXiv note → ICLR/NeurIPS/ICML/TMLR）。
- 本块（plan-2）：文献定位、新颖性论证（§7/§8）、故事线组装、频率分析作为统一批判语言。
- 协同：plan-1 的实验产出（频率↔gap、干预、bpe 验证）直接喂给故事线的逻辑一/逻辑二。

## 4. 待调研问题

- [x] **X-gram** 精读：Zipf 长尾 under-training 与 slot collapse 定义、证据、可对照指标 → §2.4.1（原文已核对）。
- [x] 各 baseline 的评价协议盘点：train/val 分离 loss 与频率 bucket → §2.3 正式对照表（2026-08-02 原文复核）。
- [x] SCONE 的 f-gram 筛选阈值/预算：min-count 5、top-S、K=2→7 / K=8→108（原文核对）→ §2.4.2。
- [x] 「高频是否等于高价值」：surprise/影响函数替代标准文献 → §2.4.2（RSI/Entropy-UID/TRAK/LESS/FINDR/DSIR/SILO）。
- [x] 后训练冻结文献（SFT/RL 多 epoch → overfit）→ §2.4.3 证据链。
- [x] Infini-gram / retrieval-augmented 对比定位 → §2.3 对照表 + critique §11；写进 related work 的 retrieval 段。

## 5. 待验证/待补实验（支撑故事线）

> 2026-08-08 状态：逻辑一③、逻辑二③④⑤ 全部闭环；新增 8 月 4–7 日证据（guide §15–§19）。

- [x] BPE/over-encoding 干预实验（逻辑一第③点，✅ 已验证：guide §16.11 token 重映射代理，T=3000 −43% / T=8000 −67%，长时程 0.28；如需「真 BPE 重训」版可另起队列）
- [x] 每 epoch hash 错位实验（逻辑二第④点，✅ 已验证：guide §10 实验4，−93% 双 shell；统一口径 −79%；toy −72%）
- [x] 低频 gate 清零干预（✅ 已验证：guide §10 实验5，nanogpt_original −94% / current-shell −58%；但 exp7 含 novel 版失败——gate 不是有效位点，见 §12.4）
- [x] freeze ngram+gate 因果分析（逻辑二第③点，✅ 已验证 3 seeds：e2 回滚 −89%、readout mask −89%、freeze table −50%、freeze reader −59%、gate −19%，guide §12）
- [x] P0 频率遮罩复现 svbird F（✅ 已验证 3 seeds：0–200 −86%、0–1000 −92%、1–5 −26.5%、comb 可加、rowzero 负结果，guide §15.5/§15.6）
- [x] toy 干净 2×2 + 剂量（✅ 已验证：same-order 公平考卷 3.94 vs 1.40/0.000/0.000；seen gap 0.005→7.08 单调；r=16 断崖，guide §16）
- [x] Exact-frequency baseline + 受控样本量复播（✅ 已验证：guide §18/§19；Transformer vs MLP trunk，285 steps/epoch）
- [x] 合成 transition pilot（✅ 已验证：order=5，A/B 两方案，注入贴 Bayes、对照 gap 复现且 excess 随 r 单调，manual 08-07）
- [x] shard/replay epoch 对齐更正（✅ 已验证：step 对齐单调递减是重播轮数假象；epoch 对齐后 gap@6pass 非单调，manual 08-07）
- [x] 表优化器 beta 扫描（✅ 已验证：toy 5 配置 gap 7.4–7.9 几乎不变，beta 不是 gap 开关；toy/toy5_beta_scan_launch.sh）
- [ ] 对 Engram/Over-Encoding 跑最直接实验（critique 里已给出实验设计；本地等价物已覆盖，论文正文可用代理论证）

## 6. 产出物清单

- [x] 方法对照表正式版（§2.3，2026-08-02 原文复核；可直接转论文 Table）
- [x] Related Work 段落（vbird plan §7 合并本调研）→ `docs/literature/related-work-and-novelty.md`
- [x] 新颖性声明（vbird plan §8）→ `docs/literature/related-work-and-novelty.md`
- [x] 引用清单 bibtex（arXiv 编号/作者/年份已逐一复核）→ `docs/literature/references.bib`
- [x] 唯一成果汇报 HTML（docs 内）→ `docs/plan2-literature-story-report.html`（2026-08-08 刷新：证据故事线已闭环，锯齿调查含 v10 复核；论文 Introduction/Background 仍 pending）
- [ ] 论文结构草案中的 Introduction/Background 素材（下一步；可基于 related-work 文档提炼）
