# 文献调研：LLM 中的 over-encoding / 记忆化隐患、post-training 阶段的记忆化动力学，以及「马嘉祺」案例

> 调研日期：2026-05（为 ngram-gap-lab 项目 n-gram 查表模块 train/val gap 研究做背景调研）
> 标注约定：**[实验证据]** = 有公开可复现的实验/数据支持；**[推测]** = 作者或本文的推断、类比，尚无直接实验验证。

---

## 1. 现代 LLM 中是否存在类似的 over-encoding / 记忆化隐患？

### 1.1 总体图景：记忆化是 LLM 的系统性现象

- **[实验证据]** 综述 *Undesirable Memorization in Large Language Models: A Survey*（arXiv:2410.02650）系统梳理了 LLM 中「不良记忆化」的测量方法（提取攻击、成员推断攻击）与缓解策略，明确指出记忆化随重复次数、参数规模、上下文长度增加而增强。
  https://arxiv.org/abs/2410.02650
- **[实验证据]** *SoK: The Landscape of Memorization in LLMs*（arXiv:2507.05578）从机制、测量、缓解三方面系统化，区分「样本级记忆」与「分布级记忆」，后者正是 train/val gap 的来源之一。
  https://arxiv.org/abs/2507.05578
- **[实验证据]** Morris 等的 *How much do language models memorize?*（arXiv:2504.12527）估计 GPT 系列模型的容量约为 3.6 bit/参数，且模型会优先用容量记忆训练数据、剩余容量才用于泛化——即「记忆优先于泛化」的容量分配次序。
  https://arxiv.org/abs/2504.12527 （讨论见 https://www.reddit.com/r/LocalLLaMA/comments/1upq1rc/paper_how_much_do_language_models_memorize/ ）

### 1.2 显式记忆/检索模块：kNN-LM、RETRO、memory-augmented transformer

- **[实验证据]** kNN-LM（Khandelwal et al., ICLR 2020）本身展示了「显式记忆表」的收益：在 Wikitext-103 上无需额外训练即降低 2.9 perplexity；同时论文实验显示关闭 dropout 后训练 loss 可降到 0（模型有足够容量记忆训练集），而「记忆化 LM + 插值」只带来 0.1 的提升，远低于 kNN-LM 的 1.9——说明显式查表与参数化记忆的收益来源不同。
  https://openreview.net/pdf?id=HklBjCEKvH ；https://ai.meta.com/research/publications/generalization-through-memorization-nearest-neighbor-language-models/
- **[实验证据]** *How Much Can CLIP Benefit from VLM's Training? / 更相关的：Norlund et al., "How much do language models copy from their training data?"*（arXiv:2302.12128）对 RETRO 的分析发现：检索带来的性能增益**大部分来自数据库与测试数据之间的 token 重叠**（lexical overlap），即 RETRO 在很大程度上是在「复制训练数据」而非泛化；kNN-LM 中也观察到同样的 lexical overlap 主导效应（Drozdov et al., 2022）。这与我们「n-gram 表在 val 上收益有限、主要吃 train 分布」的观察同构。
  https://arxiv.org/pdf/2302.12128
- **[实验证据]** *Great Memory, Shallow Reasoning: Limits of kNN-LMs*（NAACL 2025）实验表明 kNN-LM 的改进集中在表层 n-gram 匹配，对需要推理的任务几乎无帮助——显式记忆模块「记得多、想得浅」。
  https://aclanthology.org/2025.naacl-short.40.pdf
- **[实验证据]** RETRO 小规模复现 retro-li（arXiv:2410.00004）报告：关闭检索微调时出现明显过拟合（多训几个 epoch 即过拟合），开启检索可部分缓解但 best perplexity 略差——检索模块与 backbone 的过拟合动力学相互耦合。
  https://arxiv.org/html/2410.00004v2
- **[推测]** 隐式记忆模块（如 Transformer FFN 中的 key-value 联想，Geva et al. 2021 的「FFN as key-value memory」视角）与我们的显式 hash table 在功能上同构；现代 LLM 没有显式查表，但 FFN/attention 本身就是分布式 value memory，over-encoding 隐患以更隐蔽的形式存在。此为类比推断，无直接实验。

### 1.3 MoE：专家作为「分区记忆」

- **[实验证据]** *Mixture of Parrots: Experts improve memorization more than reasoning*（arXiv:2410.19034）用受控实验证明：MoE 相对 dense 模型的增益**主要来自记忆型任务（factual recall）而非推理任务**——专家模块天然偏向记忆化，是「结构化的 over-encoding」。
  https://arxiv.org/abs/2410.19034
- **[实验证据]** 后续工作（如 arXiv:2604.23036 *Preserving Long-Tailed Expert Information in MoE*）关注 MoE 中长尾信息在训练中丢失的问题，说明专家路由对低频 token 的覆盖同样不均匀。
  https://arxiv.org/html/2604.23036v1

### 1.4 双时间尺度动力学：快速记忆 + 慢速泛化

- **[实验证据]** *Critical Data Size of Language Models from a Grokking Perspective*（arXiv:2401.10463）在语言模型上复现 grokking，形式化「快速记忆 → 慢速泛化」的相变，划分数据不足/恰好/过剩三个 regime——与我们的「快速表记忆 + 慢速 backbone 共适应」两状态模型直接对应。
  https://arxiv.org/html/2401.10463v1
- **[实验证据]** grokking 系统研究（arXiv:2603.25009 综述性实证；arXiv:2505.11411 玻璃弛豫视角）表明：记忆解与泛化解在 loss landscape 中共存，weight decay 与慢变梯度方向驱动向泛化解迁移；快速降低训练 loss 会把网络「冻结」在记忆态。
  https://arxiv.org/html/2603.25009v1 ；https://arxiv.org/html/2505.11411v5
- **[推测]** 我们的两状态模型（表 = fast component，backbone = slow component）可视为 grokking「快记忆/慢泛化」在架构显式化后的版本：显式表让 fast component 的形成速度和可干预性（mask/freeze/reseed）远超隐式情形。

---

## 2. Post-training（SFT/RLHF/偏好优化）阶段的记忆化：更危险还是被缓解？

### 2.1 SFT 引发遗忘与分布偏移

- **[实验证据]** Luo et al., *An Empirical Study of Catastrophic Forgetting in LLMs during Continual Fine-tuning*（arXiv:2308.08747）：1B–7B 模型在持续指令微调中普遍出现灾难性遗忘，且**模型越大遗忘越严重**（初始性能高、下降更显著）。
  https://arxiv.org/abs/2308.08747
- **[实验证据]** *Mapping Post-Training Forgetting in Language Models at Scale*（arXiv:2510.17776）大规模测绘 post-training 遗忘：RLHF 存在「alignment tax」，模型平均（model averaging）可缓解；Kotha et al. 2024 发现微调并非「擦除」能力而是**扭曲隐式任务推断**；SFT/GRPO 均出现「时间性遗忘」（忘记自己此前能生成的解）。
  https://arxiv.org/html/2510.17776v1
- **[实验证据]** *The Role of On-Policy Data in Mitigating Forgetting*（arXiv:2510.18874）：on-policy 数据（RL 类方法）比 off-policy SFT 更能缓解遗忘——RLHF/RLVR 相对 SFT 对预训练知识更温和。
  https://arxiv.org/html/2510.18874v1
- **[实验证据]** *Improved SFT for LLMs to Mitigate Catastrophic Forgetting*（arXiv:2506.09428）：第三方在开源模型上做领域 SFT 时，因拿不到原始 SFT 数据、rehearsal 数据分布不匹配，反而**加剧**知识退化；需先重建基座指令分布再混入。
  https://arxiv.org/abs/2506.09428

### 2.2 关键结论：post-training 是「分布收窄 + 低频覆盖塌陷」的过程

- **[实验证据]** MiniMax 官方排查报告（见 §3）是迄今最直接的证据：SFT 数据对词表覆盖不均 → 低频 token 的 lm_head 在 SFT 中漂移（baseline 中 4.9% token cos_sim < 0.95，日语 token 29.7% < 0.95），而 input embedding 几乎不变——**理解能力保留、生成能力丢失**。
  https://www.minimaxi.com/blog/sparse-token-forgetting-investigation
- **[推测]** 综合 §2.1–2.2：post-training 对「预训练中已记忆的内容」不是简单缓解或加剧，而是**选择性重写**——高频对齐目标被强化，低频/长尾记忆因梯度稀疏而漂移。对带显式记忆表的架构，若表在 post-training 中继续更新而 SFT 数据分布不同，表的 value 会向 SFT 分布漂移，backbone 与表的共适应关系可能被打破（此为我们实验可检验的推测）。

---

## 3. 真实案例查证：MiniMax 模型「不认识马嘉祺」

### 3.1 事件与官方结论（均有公开来源）

- **事件**：2026 年 5 月，用户发现 MiniMax-M2.5 无法说出时代少年团队长「马嘉祺」的名字——模型知道其履历、团体、出道时间，但生成时输出「马嘉轩」「马丝祺」等编造名或错字。社区在小红书/知乎大量讨论。
  机器之心报道：https://zhuanlan.zhihu.com/p/2036457168669504167 ；新浪科技：https://finance.sina.cn/tech/2026-05-09/detail-inhxhkfi1567582.d.html ；IT之家：https://www.ithome.com/0/948/092.htm ；钛媒体：https://www.tmtpost.com/7982777.html
- **官方排查**（MiniMax Research 博客，**[实验证据]**）：
  https://www.minimaxi.com/blog/sparse-token-forgetting-investigation
  1. 排除 tokenizer 不对齐：「嘉祺」是独立 token（id=190467），预训练中已被充分训练（其 embedding 近邻全是明星人名，语义正确）。
  2. 根因：**后训练数据中含「嘉祺」的样本不足 5 条**，SFT 过程中该 token 的 lm_head 向量发生显著方向漂移，生成概率跌出 top-p 采样范围；而 vocab embedding 几乎不变（梯度逐层衰减 + 低频 token 无有效梯度），故「认识但说不出」。
  3. 系统性发现：lm_head 退化最严重的 token 类别中 40%+ 是日文口语/网页模板 token，与「小语种语言混杂」问题共享同一机制——后训练数据覆盖不足导致 lm_head 漂移与跨语言混淆。
  4. 修复：混入「全词表覆盖合成复读数据」为每个 token 建立生成频率下限。修复后日语→俄文混淆从 47% 降至 1%，马嘉祺 case 及「无痛人流→人流」等 lm_head 退化 case 全部修复；全词表 cos_sim mean 从 0.9837 升至 0.9992，cos_sim<0.95 的 token 从 9,805 个降为 0。
- **定性**：这是**post-training 阶段的稀疏 token 遗忘（sparse token forgetting）**，不是预训练语料覆盖不足，也与记忆模块无关（MiniMax-M2 无显式记忆表）。

### 3.2 与我们架构的关系分析

- **[推测]** 该案例的机制（低频 token 在分布偏移阶段被高频梯度「冲刷」）与我们两状态模型的预测同构：**快速分量（这里是 lm_head/表类参数）对数据分布变化响应快，慢速分量（中间层表征）保持稳定**，两者失配产生「知道但说不出」的解耦行为。我们实验中 mask/freeze 表后 val 行为的变化，正是这种快慢失配的可控版本。
- **[推测]** 若我们带显式 n-gram 表的架构在 pretrain 后接一个「表继续更新、backbone 冻结或低 lr」的 SFT 阶段，且 SFT 语料中某些训练期高频实体的 n-gram 键不再出现，理论上可复现「实体知识保留在 backbone、但表的 value 漂移/门控失配导致推理时输出异常」的现象——这是 MiniMax 案例在显式记忆架构下的可复现类比，值得作为后续实验设计。
- **[推测]** 反向情形（训练语料高频实体在 val 上表现过强）对应我们的 over-encoding 观察：表对 train 高频 n-gram 过拟合，val 上同实体不同上下文反而受损。MiniMax 案例提示：这种 over-encoding 在 post-training 分布收窄时可能**部分被冲掉**（表 value 漂移），即 post-training 对显式记忆表既是威胁也是天然的正则化。

---

## 4. 与我们实验的连接点：mask / freeze / reseed 干预的可迁移启示

1. **快慢分量分离是普遍现象，且可被定位到具体参数组。** 我们的两状态模型（表=fast，backbone=slow）在文献中有三层对应：grokking 的快记忆/慢泛化相变（arXiv:2401.10463）、MiniMax 案例中 lm_head（快）与中间层表征（慢）的解耦、MoE 中专家偏记忆化（arXiv:2410.19034）。启示：现代 LLM 的「记忆化审计」应按参数组（lm_head、FFN、专家、embedding）而非整体 loss 进行——我们的 mask/freeze 干预正是这种参数组级审计的小规模原型。
2. **「冻结快分量」是低成本的过拟合控制手段。** 我们发现 freeze 表可阻断 train/val gap 扩大；文献中对应物是：post-training 遗忘缓解中的正则化/梯度投影方法（arXiv:2602.07892 OGPSA 用正交梯度投影约束更新）与 model averaging（arXiv:2510.17776）。启示：对含显式记忆模块的架构，post-training 阶段对记忆模块做 freeze 或极低 lr，是文献支持的方向。
3. **reseed ≈ 打破记忆态的「扰动」手段，与 grokking 文献一致。** grokking 研究表明扰动（weight decay、参数交换、WanD）能把网络从记忆态推向泛化态（arXiv:2505.11411、arXiv:2608.01833）。我们的 reseed 表干预可解读为对 fast component 的受控重置；迁移启示：现代 LLM 若怀疑某参数组陷入记忆态，重初始化该组 + 短程重训可能比全量重训更高效——此为推测，需实验。
4. **覆盖度下限保障是 MiniMax 给出的可操作工程结论，可直接映射到记忆表维护。** MiniMax 用「全词表复读合成数据」为每个 token 建立生成频率下限。对应到我们的架构：post-training 时为表的键空间做覆盖度监控（哪些 n-gram 键在 SFT 数据中零出现），必要时注入合成回放数据，防止表 value 漂移或门控塌陷。
5. **评估必须区分「复制」与「泛化」。** RETRO/kNN-LM 的增益大部分来自 lexical overlap（arXiv:2302.12128、NAACL 2025 kNN-LM 局限）。我们的 val 评估应报告「表命中 vs 未命中」分层的指标，避免把查表复制误认为泛化收益——这与我们 mask 干预中「表贡献在 val 上接近零」的结论互相印证。

---

## 附：主要来源清单

| 主题 | 来源 |
| --- | --- |
| 记忆化综述 | https://arxiv.org/abs/2410.02650 ；https://arxiv.org/abs/2507.05578 |
| 模型容量与记忆 | https://arxiv.org/abs/2504.12527 |
| kNN-LM | https://openreview.net/pdf?id=HklBjCEKvH |
| kNN-LM 局限 | https://aclanthology.org/2025.naacl-short.40.pdf |
| RETRO 复制效应 | https://arxiv.org/pdf/2302.12128 |
| retro-li 过拟合 | https://arxiv.org/html/2410.00004v2 |
| MoE 偏记忆化 | https://arxiv.org/abs/2410.19034 |
| grokking / 快慢动力学 | https://arxiv.org/html/2401.10463v1 ；https://arxiv.org/html/2505.11411v5 |
| SFT 灾难性遗忘 | https://arxiv.org/abs/2308.08747 |
| post-training 遗忘图谱 | https://arxiv.org/html/2510.17776v1 |
| on-policy 缓解遗忘 | https://arxiv.org/html/2510.18874v1 |
| alignment tax / 梯度投影 | https://arxiv.org/html/2602.07892v1 |
| MiniMax 官方排查 | https://www.minimaxi.com/blog/sparse-token-forgetting-investigation |
| 马嘉祺事件报道 | https://zhuanlan.zhihu.com/p/2036457168669504167 ；https://www.ithome.com/0/948/092.htm ；https://www.tmtpost.com/7982777.html |
