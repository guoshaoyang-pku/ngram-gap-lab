# X-gram 论文精读笔记（subagent Ramanujan，2026-08-01）

> 引用已复核（2026-08-02）：arXiv:2604.21724（v2）《Beyond N-gram: Data-Aware X-GRAM Extraction for Efficient Embedding Parameter Scaling》，Chen et al., 2026；摘要原话与 Zipf/slot-collapse 论述已对照 arXiv HTML 原文确认。已并入 plan-2 §2.4.1。

**论文确认**：arXiv 上无题名完全一致的《X-gram: Training Long-Context Language Model with Multi-Grained Unit》；唯一匹配的是 [Beyond N-gram: Data-Aware X-gram Extraction for Efficient Embedding Parameter Scaling](https://arxiv.org/abs/2604.21724)（arXiv:2604.21724，2026，v2 为最新版）。其内容（可训练 n-gram 记忆、Zipf 长尾欠训练、slot collapse）与主题对应。x-gram = 把 token 索引的 1-gram 检索经多尺度 ShortConv 精炼为可变长度局部 x-gram 特征，并非直接训练 n-gram 表。

**核心方法**（三算子框架：token→参数映射、信息抽取、注入）：
- 频率感知混合哈希：从采样语料估计 p(ω)，按平滑质量 s(ω)=p(ω)^α 分桶；高频 VIP token 独占物理行，长尾 token 路由进共享哈希桶，alias mixing 控制压缩。
- Gated ShortConv：RMSNorm + 因果 depthwise 卷积（多尺度核 3,5,7,9、SiLU 门控、残差融合），从窗口提取 x-gram 特征并打破 slot 冗余。
- 深度感知门控注入 attention value 流与层间残差；稀疏感知学习率（低频行放大步长）。
- 0.73B/1.15B 验证 PPL 与下游准确率，最高比 vanilla 高 4.4 点，50% 表格配置即超基线。

**Zipf 长尾 / slot collapse / 频率讨论（论文动机核心）**：
- 摘要原文："We attribute these limitations to Zipfian under-training of the long tail, heterogeneous demand across layers, and 'slot collapse' that produces redundant embeddings."
- "due to the long-tail distribution of token and n-gram frequencies, most table rows receive very few updates and remain near initialization"；"activation magnitude increases strongly with frequency… uniform scaling primarily adds cold, under-trained parameters"。
- slot collapse："each slot sees almost the same token occurrences and thus receives highly correlated gradients"，slot 间 pairwise cosine similarity 很高。
- 附录：稀有行更新极少、"cumulative parameter movement is disproportionately small"，故设 LR 矫正。频率分析以 token 为主，n-gram 频率一笔带过。

**与低频 n-gram train/val gap 的关系**：
- X-gram 缓解的是「低频→欠训练」（训练侧）；但全文无 generalization/overfit/OOD 分析，不解决「欠训练→泛化 gap」。
- 机制对分布漂移敏感：p(ω)、VIP 集、桶边界、alias 概率均由采样语料估计；若 replay/val 频率与训练语料不一致，尾部单位会被错误路由或碰撞；alias 使一个样本更新影响多个未见 token。replay 场景下 gap 仍可能残留。

**可对照的指标/实验建议**：
1. 复刻行利用率分析：物理行更新数 n_j≈N·q_j 与激活幅度按频率桶分解，对比 train/replay 下命中集中度；尾桶利用率 × val PPL gap 相关分析。
2. 复刻 slot-collapse 度量：多 view/slot 检索嵌入的 pairwise cosine similarity，检验 replay 是否加速塌缩、塌缩度是否与 gap 相关。
3. 频率错配实验：分别用 train/replay/val 语料估计 p(ω)，量化 val n-gram 的桶重路由率、alias 碰撞率与「从未更新覆盖率」；测每行 LR 放大能否降低尾桶 val 端 gap。
