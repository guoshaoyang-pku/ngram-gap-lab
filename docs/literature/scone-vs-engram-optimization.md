# SCONE 与 Engram：时间线、优劣比较与优化方向

更新时间：2026-07-30

## 结论先行

SCONE 并不是 2026 年 2 月才出现：

- SCONE 的 arXiv v1 提交时间是 **2025-02-03**；
- 论文后来成为 **NeurIPS 2025** 论文，arXiv v3 标注为 NeurIPS 2025 camera-ready；
- Engram 的 arXiv v1 提交时间是 **2026-01-12**。

因此，SCONE 在公开时间上早于 Engram 将近一年。两者都建立在更早的 n-gram embedding、hash embedding 和 latent n-gram 工作之上，不能仅因为方法相似就推断两者存在直接继承关系。

两者没有严格的同条件 head-to-head 对比，所以不能简单说某一个绝对更优。更准确的判断是：

- **SCONE 更像“频率筛选 + 精确静态 memory + 共享生成器”**；
- **Engram 更像“全局 hash memory + hidden-state gate + 中间层 conditional memory”**；
- SCONE 更适合解决低频独立 row 和 collision 的问题；
- Engram 更适合解决大规模 MoE 中的中间层知识检索和动态上下文适配；
- 最值得尝试的组合是：**SCONE 的 key selection / shared generator + Engram 的 gated intermediate injection**。

## 1. 为什么 SCONE 看起来合理，却到 2025 年才出现

“频繁 n-gram 建表”本身并不新。传统 n-gram language model、hash embedding、N-Grammer、Over-Encoding 都已经分别使用过类似组件。SCONE 真正的新意不是提出了“n-gram 可以查表”，而是把几个此前分散的问题组合成了一个可扩展系统：

1. 不扩大输出词表，只扩展 input-side representation；
2. 只保留训练语料中的频繁 f-gram；
3. 不为每个 f-gram 独立反向传播，而是用共享的 f-gram Transformer 生成向量；
4. 训练后把向量预计算成精确 key-value 表；
5. 把表放在 host memory 或 NVMe，避免 inference accelerator memory 增长。

这些环节缺一不可。直接说“把高频 n-gram 放到一个表里”会立即遇到：

- 候选集合太大；
- 长尾 key 更新次数太少；
- 输出 embedding 与输入 embedding tied；
- 表无法放进 accelerator；
- 动态生成和静态缓存不一致；
- n-gram lookup 的随机内存访问可能抵消 FLOPs 优势。

所以 SCONE 是一个“在基础思路成熟后，系统条件终于足够”的结果，而不是一个此前没人想到的显然算法。

### 1.1 2025 年之前缺少的关键条件

#### 大规模 embedding offload 逐渐可行

SCONE 依赖 host RAM/NVMe 承载大规模 f-gram 表。传统 Transformer 研究通常把 embedding 看成 accelerator-resident 参数，重点是 FLOPs，而不是把 lookup memory 作为独立的 scaling axis。

#### 输入与输出词表解耦变得明确

如果输入 embedding 与 output/unembedding 共享，扩大输入词表会同步扩大 logits 计算。Over-Encoding 已经明确展示了 input vocabulary scaling 的价值，SCONE 则把这一思路做成了独立的 f-gram cache。

#### 低频更新问题被实验明确暴露

SCONE 的动机实验指出：普通大词表中，很多 row 收不到足够更新。例如在固定训练 token 数下，2M vocabulary 中只有很少比例的 embedding 获得大量更新。这个观察推动了“用共享生成器而不是独立 row”的设计。

#### 数据规模使简单频率筛选变得有效

在更大语料上，top f-grams 可以覆盖相当大的访问质量，同时仍然保持有限表大小。没有足够规模的数据时，频率筛选容易变成 domain-specific memorization。

## 2. SCONE 与 Engram 的核心差异

| 维度 | SCONE | Engram |
|---|---|---|
| 公开时间 | arXiv 2025-02；NeurIPS 2025 | arXiv 2026-01 |
| key 集合 | 训练语料中 top frequent f-grams | canonical suffix n-grams 的 hash 地址 |
| collision | 选中 key 之间可做到精确映射 | multi-head hashing，仍有 collision |
| n-gram 选择 | frequency-based，最大长度 \(K\) | 最大 order \(N\)，通常对所有局部 suffix 做 hash |
| 表示学习 | 共享 f-gram Transformer | 直接训练 sparse embedding tables |
| 上下文适配 | f-gram 内部 contextualization | 当前 hidden state 对 memory 做 context-aware gate |
| 注入位置 | 主要是 input embedding 层 | 选定的中间 Transformer 层 |
| 多阶处理 | longest matching f-gram | 各 order、各 hash head 拼接 |
| 主要优点 | 精确、高频、共享、易 offload | 动态 gate、中间层注入、适合 MoE |
| 主要风险 | train-time/inference-time 表示切换、domain shift | collision、低频 hash row、gate shortcut |
| 训练系统 | 训练额外 f-gram model | table sharding + sparse all-to-all |
| 适合的目标 | 静态局部模式和输入表示增强 | conditional memory 与动态计算协同 |

## 3. 哪一个更优：必须分场景讨论

### 3.1 如果目标是减少低频 row 问题：SCONE 更优

SCONE 的 key 不是所有可能的 n-gram，而是频繁 f-gram 集合。更重要的是，训练时不是让每个 key 从零学习一个独立向量，而是：

\\[
e_g
=
A_{\mathrm{f\text{-}gram}}
\left(
E(x_1),\ldots,E(x_n)
\right).
\\]

不同 f-gram 共享生成器参数，因此低频 key 可以共享 token-level 和 transformer-level 统计信息。

这对你们的 gap 问题很重要：SCONE 会降低“低频独立 table row 欠训练”的风险，但不会自动消除 repeated-epoch overfitting。若一个高频 f-gram 被反复看到，生成器仍然可能学习训练集 shortcut。

### 3.2 如果目标是消除 hash collision：SCONE 更优，但不一定整体更优

SCONE 的 selected f-gram set 可以使用精确 dictionary、trie 或 MPHF，没有 Engram 的 modulo collision。

但 collision 并非永远只带来坏处：

- collision 会让多个 context 共享更新；
- 共享可能构成隐式 regularization；
- 精确表会切断这种统计共享；
- Engram-Nine 的预印本报告过 collision-free hot tier 不一定改善 validation loss。

因此“SCONE 无 collision”是结构优势，不是总 loss 必然优势。

### 3.3 如果目标是动态消歧：Engram 更优

SCONE 的 f-gram vector 主要由局部 token 序列生成。对于同一个局部短语在不同全局上下文中的不同含义，它缺少 Engram 那种显式 hidden-state gate。

Engram 使用：

\\[
\\alpha_t
=
\\sigma
\\left(
\\operatorname{RMSNorm}(h_t)^\top
\\operatorname{RMSNorm}(W_K e_t)
\\right),
\\]

再把 gated value 注入网络。因此它可以根据当前上下文决定 memory 应该注入多少。

不过 gate 只能调节混合向量的强度，不能从 collision row 中恢复每个原始 n-gram 的独立表示。如果 collision 很严重，gate 是缓解器，不是 collision 的逆运算。

### 3.4 如果目标是提升 MoE 主干：Engram 更优

SCONE 主要作用在输入层。它帮助模型一开始获得更好的局部表示，但之后仍然需要主干把这些信息逐层传递。

Engram 直接插入选定的中间层，并把局部 pattern reconstruction 从早期 Transformer computation 中卸载出去。DeepSeek 的实验将其解释为：

- 释放早期 attention/FFN 的静态重建负担；
- 为复杂 reasoning 保留更多有效深度；
- 让 attention 更专注于长程依赖；
- 与 MoE 的 conditional computation 形成互补。

如果研究对象是大型 sparse MoE，Engram 的架构位置是明显优势。

### 3.5 如果目标是推理部署：两者各有优势

SCONE：

- key 集合有限且静态；
- 训练后可直接缓存；
- dense matrix + dictionary 或 LMDB/B+ tree；
- 不需要在推理时运行 f-gram Transformer。

Engram：

- 地址由 token 序列确定；
- 支持 host memory prefetch；
- 可以把大表放到 host memory/NVMe；
- 但推理时需要多 head、多 order 的 memory 读取和 projection/gating。

SCONE 的 lookup 结构更像静态数据库；Engram 的 lookup 更像大规模 conditional memory subsystem。

## 4. 一个更公平的结论

如果必须给出一句判断：

> **SCONE 在“数据筛选、精确寻址、低频训练稳定性”上更干净；Engram 在“动态上下文适配、中间层注入、MoE 协同和规模化系统设计”上更强。**

这不是“SCONE 替代 Engram”，而是两者优化目标不同。

对你们当前研究，推荐优先级是：

1. 用 SCONE 作为 low-frequency/collision 的干净 baseline；
2. 用 Engram 作为 hash + gate + repeated-epoch gap 的现实 baseline；
3. 构造 SCONE key selection + Engram gate 的 hybrid；
4. 再比较 exact hot table、hashed tail 和 shared generator。

## 5. 最值得做的优化：SCONE-Engram Hybrid

建议的结构如下：

```text
token sequence
    ├── base token embedding
    ├── exact frequent f-gram lookup
    │       └── shared f-gram generator / cached vector
    └── hashed tail lookup
            └── multi-head low-dimensional table

exact hot + hashed tail
    └── per-source, per-order context-aware gate
            └── selected intermediate layers
```

### 5.1 Hot tier

对于 top-B 或满足 `count >= c` 的 f-gram：

- 使用精确 key；
- 使用 MPHF、trie 或 sorted dictionary；
- 每个 key 由 shared f-gram model 生成向量；
- 训练后可以缓存到 host memory；
- hot tier 不需要承担 collision。

### 5.2 Tail tier

对于未进入 hot set 的 context：

- 使用小型 multi-head hash table；
- 使用较低维度；
- 可以与 compositional fallback 相加；
- 允许共享统计信号，但不让尾部消耗过多独立容量。

### 5.3 Gate 必须区分 source

不要只使用一个 scalar gate。至少可以拆成：

\\[
\\Delta_t
=
\\sum_{n}
\\left(
\\alpha_{t,n}^{\\mathrm{exact}}v_{t,n}^{\\mathrm{exact}}
+
\\alpha_{t,n}^{\\mathrm{hash}}v_{t,n}^{\\mathrm{hash}}
\\right).
\\]

gate 的输入可以包括：

- 当前 hidden state；
- n-gram order；
- exact/hash source；
- train frequency bucket；
- row load 或 collision confidence；
- 是否发生 longest-match；
- key 是否跨 domain 出现。

这样模型可以学习：

- 高频精确 memory 更可信；
- hash tail 需要更保守；
- 未知或 collision-heavy memory 应降低注入；
- 高阶 n-gram 不一定总比低阶 n-gram 更可靠。

## 6. 其他可行优化技巧

### 6.1 不要只按 raw frequency 选 key

纯频率是强 baseline，但可以使用更好的 utility score：

\\[
S(g)
=
\operatorname{count}(g)^\alpha
\cdot
\operatorname{PMI}(g)^\beta
\cdot
\operatorname{domain\_coverage}(g)^\gamma.
\\]

也可以用小模型估计：

\\[
S(g)
=
\frac{
\text{validation loss reduction from }g
}{
\text{memory bytes of }g
}.
\\]

推荐顺序：

1. top-B frequency；
2. minimum frequency threshold；
3. frequency × PMI；
4. frequency × held-out loss gain；
5. frequency × cross-domain coverage。

不要一开始就用复杂评分，否则会把 key selection 和 representation learning 混在一起。

### 6.2 按 n-gram order 分配预算

不应默认 2-gram、3-gram、4-gram 使用相同容量。建议对每个 order 单独建立预算：

\\[
B_n
\propto
\frac{
\text{covered token mass}_n
\times
\text{validation utility}_n
}{
\text{bytes per key}_n
}.
\\]

通常：

- 2-gram 覆盖率高、训练稳定；
- 3-gram 能表示更多实体和短语；
- 更高阶 n-gram 更容易进入长尾和 train-only 区域。

对你们的 repeated-epoch 实验，建议记录每个 order 的：

- train hit rate；
- validation hit rate；
- average frequency；
- gap contribution；
- memory bytes/token。

### 6.3 不要只取最长 f-gram

SCONE 的 longest-match 策略很省 lookup，但可能丢掉低阶信息：

```text
longest match only:
    New York City → one 3-gram

multi-scale:
    City          → unigram feature
    York City     → bigram feature
    New York City → trigram feature
```

可选方案：

- longest match + lower-order residual；
- 所有命中 order 都 lookup，再由 gate 融合；
- 先取最长 match，再把 lower-order 作为 backoff；
- 使用 order-aware attention，而不是简单求和。

Engram 的多阶拼接在表达上更灵活，但增加了 memory traffic。一个折中方案是 hot tier 只取最长精确 key，tail 或低频区域保留低阶 hash/backoff。

### 6.4 用 compositional residual 处理未登录 key

不要把未命中 key 直接置零。可以定义：

\\[
e(g)
=
e_{\\mathrm{compose}}(g)
+
\\mathbf 1[g\\in\\mathcal F]\\,
\\delta_g.
\\]

其中：

- \(e_{\mathrm{compose}}\) 由 token embedding、低阶 n-gram 或小型 encoder 生成；
- \(\delta_g\) 只给高频 key 学习 residual；
- 未登录或低频 key 仍有泛化表示；
- 高频 key 可以拥有额外的精确记忆。

这是比“频率 cutoff 后完全没有 memory”更平滑的方案。

### 6.5 collision-aware hash tail

hash tail 不必把所有 collision 一视同仁。可以为每个 row 保存轻量 metadata：

- row load；
- top collision partner mass；
- fingerprint；
- update count；
- source order。

然后使用：

\\[
\alpha_t^{\\mathrm{hash}}
=
\alpha_t
\cdot
\operatorname{conf}(r_t),
\\]

其中 `conf` 可以是：

\\[
\operatorname{conf}(r)
=
\frac{1}{
1+\lambda\,\operatorname{row\_load}(r)
}
\\]

或一个可学习的小网络。

如果不希望使用训练语料频率作为模型输入，也可以只使用训练中动态统计的 row load 和 update variance。

### 6.6 对 memory 使用不同于 backbone 的 optimizer schedule

N-gram memory 通常比 backbone 更容易快速记忆局部 pattern，因此可以考虑：

- 更低的 memory learning rate；
- gate 使用更低学习率；
- memory warmup；
- gate warmup；
- memory gradient clipping；
- memory norm cap；
- 在第二个 epoch 前冻结或降低 memory LR；
- 对 memory 使用更强 weight decay；
- 对高频 hot tier 和 hash tail 使用不同 LR。

对你们已有 gap 现象，最直接的实验是：

```text
full LR
memory LR / 2
memory LR / 4
memory frozen after epoch 1
gate frozen after epoch 1
```

如果 gap 主要由重复 epoch 的 memory shortcut 引起，通常会对 memory LR 和 freeze schedule 很敏感。

### 6.7 memory dropout 和 route dropout

训练时随机丢弃 memory branch：

\\[
\tilde v_t
=
z_t \alpha_t v_t,
\qquad
z_t\sim\operatorname{Bernoulli}(1-p).
\\]

可以进一步按 source 做 dropout：

- exact hot dropout；
- hash tail dropout；
- high-order n-gram dropout；
- 随 epoch 增大 dropout。

这会迫使 backbone 保留 token/attention fallback，降低对某个 n-gram row 的依赖。

### 6.8 用 epoch-independent augmentation 检验 shortcut

如果每个 epoch 的同一 context 总是得到完全一致的 lookup identity，memory 很容易记忆重复样本。可以测试：

- 每个 epoch 使用不同 hash seed；
- 对 hash tail 使用 epoch-specific permutation；
- 只在训练阶段改变 collision mapping；
- 保持 validation 使用固定 mapping；
- 对 exact hot tier 不改变 key，对 tail 改变映射。

但要注意：这会破坏 persistent memory semantics。它更适合作为诊断实验，而不是最终模型设计。若 gap 消失，说明 persistent addressability 是重要因素。

### 6.9 训练时共享 generator，推理时缓存

SCONE 的 generator/cache 分离很值得保留，但还可以进一步优化：

- generator 输出用 EMA teacher 稳定；
- 每隔若干 step 更新 cached vectors；
- 对 hot key 进行蒸馏；
- 用小 decoder 复现 generator output；
- cache 只存低维向量，推理时再投影；
- 对 cold tail 只存 token-factor，不存完整 vector。

这可以减少训练时的巨大 table，同时避免推理时运行完整 generator。

### 6.10 分层 memory cache

可采用：

```text
GPU HBM:
    top hot keys

Host DRAM:
    medium-frequency exact keys

NVMe:
    cold exact keys or hashed tail

Fallback:
    compositional token encoder
```

缓存分层应按实际访问概率和 memory read latency 优化，而不是只按 key frequency 排序。真正目标是：

\\[
\text{expected latency}
=
\sum_g p(g)\operatorname{latency}(\operatorname{tier}(g)).
\\]

## 7. 与你们 gap 研究最相关的优化

你们当前最值得做的不是立即追求更大的表，而是测试以下四个方向：

### 7.1 SCONE-style exact frequent baseline

- train-only 统计 2/3-gram；
- 选择 top-B；
- 精确 dictionary；
- 不使用 hash collision；
- 保持现有 value/gate/injection 位置；
- 只改变 key selection 和 addressing。

这能回答：gap 是否来自 hash，还是来自 memory 在重复 epoch 中的学习。

### 7.2 Exact-hot + hash-tail

- 高频 n-gram 使用 exact table；
- 低频 n-gram 使用小 hash tail；
- 相同总参数和总 memory bytes；
- exact/hash 使用独立 gate。

这是最有实际意义的部署方案，也能分解 collision 与 low-frequency 的作用。

### 7.3 Frequency-aware gate

比较：

```text
gate(h_t, memory)
gate(h_t, memory, frequency_bucket)
gate(h_t, memory, frequency_bucket, row_load)
```

重点观察 gate 是否在 epoch 2/3 继续偏爱 train loss 已经下降、但 validation loss 已经升高的 bucket。

### 7.4 Memory regularization schedule

比较：

- memory dropout；
- gate dropout；
- memory LR decay；
- epoch 1 后 freeze；
- high-order n-gram freeze；
- hot/cold 不同 LR。

如果这些方法可以降低 gap，同时不显著损害第一 epoch 的收益，就能形成很清晰的 practical recommendation。

## 8. 仍然需要警惕的 SCONE 问题

SCONE 更干净，但不是完美方案。

### 8.1 train/inference representation mismatch

训练时使用 f-gram Transformer：

\\[
e_g^{\\mathrm{train}}
=
A_{\\mathrm{f\text{-}gram}}(E(x_{1:n})).
\\]

推理时使用最终缓存：

\\[
e_g^{\\mathrm{infer}}
=
F(g).
\\]

如果缓存没有及时用最终 generator 重算，或者 generator 与 main model 的联合训练不稳定，可能出现表示漂移。

### 8.2 固定训练语料集合造成 domain shift

SCONE 的 f-gram 集合由训练语料决定。新 domain 中的高价值短语如果不在集合中，会退回 token/compositional path；而训练 domain 中高频但无用的模板可能占据容量。

### 8.3 frequency 不等于 utility

高频短语可能只是格式化模板，低频实体或 API 名称可能具有更高信息密度。建议加入 held-out utility 或 cross-domain coverage。

### 8.4 longest match 可能遮蔽低阶特征

如果一个高阶 f-gram 被命中，SCONE 的 longest-match 机制可能不再使用 lower-order match。这个策略降低了 memory access，却可能损失 compositional backoff。

## 9. 对论文和实验的最终建议

不要把问题写成：

> SCONE 是正确方案，Engram 是错误方案。

更准确的研究问题是：

> 在固定 memory budget 和 repeated-epoch training 下，exact frequent memory、hashed conditional memory 与 shared compositional memory 如何在 coverage、collision、训练稳定性和动态上下文适配之间折中？

推荐的主实验矩阵：

| 模型 | key 选择 | addressing | representation | 注入 |
|---|---|---|---|---|
| M0 | 无 | 无 | 无 | 无 |
| M1 | 全部局部 context | multi-head hash | trainable table | 现有 gate |
| M2 | top-B frequency | exact dictionary | trainable table | 现有 gate |
| M3 | top-B frequency | exact | shared f-gram generator | 现有 gate |
| M4 | hot top-B + tail | exact + hash | hybrid | per-source gate |
| M5 | 全空间 | tensorized | TN-gram | 现有 gate |

所有模型控制：

- total parameters；
- memory bytes；
- activated FLOPs；
- n-gram order；
- injection layers；
- optimizer；
- training epochs；
- train/validation split。

必须报告：

- train loss；
- validation loss；
- frequency-conditioned gap；
- exact/hash hit rate；
- collision row load；
- gate/value/gated-value norm；
- hot/cold contribution；
- per-token contribution；
- host memory bandwidth；
- cache hit rate。

## 10. 最终判断

SCONE 的合理性来自一个清晰的分解：

```text
先用频率限制 key 集合
再用共享模型解决稀疏训练
最后把结果缓存成精确 memory
```

Engram 的合理性来自另一个分解：

```text
用 hash 获得大规模固定容量
用 hidden-state gate 处理上下文适配
把 memory 放到中间层与 MoE 协同
```

最有潜力的下一代方案不是二选一，而是：

```text
SCONE:
    frequent key discovery
    exact hot table
    shared f-gram generator

Engram:
    intermediate injection
    context-aware gate
    deterministic prefetch

你们的方向:
    frequency-conditioned diagnostics
    collision-aware routing
    repeated-epoch regularization
```

## References

1. [Yu et al., 2025 — Scaling Embedding Layers in Language Models (SCONE)](https://arxiv.org/abs/2502.01637)
2. [Cheng et al., 2026 — Conditional Memory via Scalable Lookup (Engram)](https://arxiv.org/abs/2601.07372)
3. [Huang et al., 2025 — Over-Tokenized Transformer](https://arxiv.org/abs/2501.16975)
4. [Zhou et al., 2026 — Tensorizing Engram](https://arxiv.org/abs/2606.08347)
5. [Chen et al., 2026 — Data-Aware X-gram](https://arxiv.org/abs/2604.21724)
6. [Lin, 2026 — Collision-Free Hot-Tier Extension for Engram](https://arxiv.org/abs/2601.16531)