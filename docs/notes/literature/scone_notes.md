# SCONE 调研笔记（subagent Herschel，2026-08-01）

> 注意：SCONE 全称是 **S**calable, **C**ontextualized, **O**ffloaded, **N**-gram **E**mbedding（Google，NeurIPS 2025），不是「Sparse COntext NEtworks」。引用已复核（2026-08-02）：arXiv:2502.01637（v3, NeurIPS 2025 camera-ready），Yu et al.；f-gram 筛选细节（min-count 5、top-S、K=2→7 / K=8→108）已对照原文确认。已并入 plan-2 §2.4.2。

**1. f-gram 筛选（细节公开，§3.1、附录 C 算法 3）**：
- 对语料做 K−1 次线性扫描，逐 n∈[2,K] 统计 n-gram 计数；计数阶段以最小频率 5 剪枝候选省内存（(n+1)-gram 达标则其 n-gram 前后缀必达标，可跳过）。
- 所有 n-gram 按原始频率降序取前 S 个（预算，实验为 10M/20M/1B）。入选阈值由预算隐式决定：固定 20M 时 K=2 截断计数为 7，K=8 为 108；默认 K=5。
- 即「纯频率标准」，不含质量/信息量判据；作者承认频率排序使长 f-gram 更稀有、下游匹配率更低。

**2. 高频 ≠ 高价值的替代标准（有公开工作）**：
- surprise/信息量：surprisal（−log p，Levy 2008）；RSI（arXiv:2606.31575，2026）用相对 surprise 做 token 筛选（剔冗余低 surprise、又剔不稳定高 surprise 尾 token）；Entropy-UID（arXiv:2502.14366）；长文档信息密度（arXiv:2309.06009）。
- 影响函数：Koh-Liang 2017 起源；TRAK（ICML 2023）、LESS（ICML 2024）；FINDR（IJCNLP-AACL 2025）用哈希 n-gram + 影响分数做预训练数据选择，显式以影响而非频率衡量 n-gram 价值；NeurIPS 2025 也有 token 级归因估值。
- 频率 vs 信息标准之争：Yang-Pedersen 1997（DF 简单但 IG/CHI 更优，稀有但判别性强的词信息增益高）；CCNet（LREC 2020）perplexity 过滤；DSIR（NeurIPS 2023）按目标分布重要性重采样；Infini-gram（COLM 2024）与 SILO（ICLR 2024）证明长尾稀有 n-gram 仍有预测力。

**3. 对我们研究的启示**：
- 频率筛选是省内存的工程标准，但信息量 ≠ 频率：低频但关键的 n-gram（领域专名、任务关键短语、验证集独有长尾片段）自信息高、判别性强，恰是频率阈值最先砍掉的。
- 纯频率 f-gram 选择在训练分布内偏置，对验证集低频 n-gram 的分布变化没有保护 → 造成 train/val gap。
- 建议：把预算让给「稀有但高影响」的 n-gram（FINDR 式影响分数或 surprise/信息增益加权替代纯频率排序）；参考 Infini-gram 无限长 n-gram 覆盖与 SILO nonparametric datastore 兜底。

**引用条目**（需复核）：SCONE (Yu et al., NeurIPS 2025, arXiv:2502.01637)；Levy 2008 Cognition；RSI arXiv:2606.31575；Entropy-UID arXiv:2502.14366；Koh-Liang ICML 2017；TRAK ICML 2023 arXiv:2303.14186；LESS ICML 2024 arXiv:2402.04333；FINDR IJCNLP-AACL 2025 DOI:10.18653/v1/2025.ijcnlp-long.184；Yang-Pedersen ICML 1997；CCNet LREC 2020 arXiv:1911.00359；DSIR NeurIPS 2023；Infini-gram COLM 2024 arXiv:2401.17377；SILO ICLR 2024 arXiv:2308.04430（已核实，原 2308.06184 为错误编号）。
