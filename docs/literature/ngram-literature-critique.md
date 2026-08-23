# N-gram Memory 方法的低频与碰撞风险：文献批判框架

更新时间：2026-07-30

## 结论摘要

你们目前的核心观察——**低频 n-gram 的 train/validation gap 更大，而高频 n-gram 的 gap 贡献很小**——可以用来批判一批现有方法，但批判点需要分成两类：

1. **低频/长尾训练信号不足**：一个 n-gram 或一个 lookup row 被命中次数太少，参数没有得到稳定估计，或者在重复 epoch 中快速记住训练模式。
2. **hash collision**：不同 n-gram 被迫共享一个 row，使向量成为多个 context 的混合；这会造成表示污染、梯度干扰和 gate 的 credit-assignment 问题。

重要的是，这两类风险并不完全一致：

- 精确表可以消除 collision，但如果每个低频 key 都有独立参数，可能加重 sparse-update 问题；
- hash 表可以通过共享 row 增加统计量，但会混合不同 context；
- SCONE 通过高频 key 筛选和共享的 f-gram model，主要缓解低频独立参数问题；
- Engram、Over-Encoding、BLT、LongCat 的 N-gram Embedding 都显式依赖 hash，因此直接暴露在 collision 风险下；
- TN-gram 用低秩共享结构绕开显式 hash collision，但它仍然有 rank bottleneck 和长尾训练分布问题；
- X-gram 已经把 Zipf 长尾 under-training 和 slot collapse 明确列为问题，几乎是你们频率分析最直接的相关工作。

因此，最有力的论文论断不是“所有 hash 都错了”，而是：

> 现有 n-gram memory 方法通常按照 lookup collision 或平均 loss 评价容量，但没有充分区分 n-gram frequency、row collision multiplicity、gate weight 与 route-specific train/validation loss。频率条件化的 gap analysis 可以揭示：同样的平均验证损失背后，可能存在完全不同的长尾记忆化和碰撞干扰机制。

这是一条可以同时批判方法设计和评价协议的路线。

## 1. 你们的方法可以批判什么

你们的“方法”应明确写成一个**诊断和干预框架**，而不仅仅是“把低频 n-gram 删除”：

### 1.1 诊断量

对每个 n-gram context \(g\)，记录：

\[
\operatorname{count}_{\mathrm{train}}(g),
\quad
\operatorname{count}_{\mathrm{val}}(g),
\quad
\Delta_g
=
\mathbb E[\ell_{\mathrm{val}}\mid g]
-
\mathbb E[\ell_{\mathrm{train}}\mid g].
\]

对每个 hash row \(r\)，还应记录：

\[
\mathcal C(r)=\{g:h(g)=r\},
\quad
\operatorname{row\_load}(r)=|\mathcal C(r)|,
\quad
\operatorname{mass}(r)=\sum_{g\in\mathcal C(r)}\operatorname{count}(g).
\]

这样可以把“某个 n-gram 很低频”和“某个 n-gram 与其他 context collision”区分开。

### 1.2 干预量

至少应比较以下干预：

- **frequency cutoff**：只激活 `count_train(g) >= c` 的 context；
- **exact hot table**：高频 context 使用精确 key，尾部保留 hash；
- **collision-free all-hot**：固定总参数预算，用 MPHF 替换高频部分；
- **collision shuffle**：每个 epoch 或每个 split 改变 hash，使模型不能稳定记住同一 row；
- **frozen memory/gate**：观察重复 epoch 中 gap 是否减少；
- **frequency-aware gate**：让 gate 显式接收 count bucket 或 confidence；
- **collision load matching**：在相同参数量下比较不同 row load 分布。

### 1.3 关键识别原则

如果删除低频 n-gram 后 gap 下降，只能说明低频分量与 gap 有关，不能单独证明是 hash collision。

如果 exact hot table 后 gap 下降，也不能单独说明 collision 是原因，因为 MPHF 实验同时改变了：

- 参数共享方式；
- row 的更新次数；
- hot/cold routing；
- 表中每个 key 的有效容量；
- 可能的 gate 输入统计。

所以你们需要把 frequency、collision 和 gate 三个变量分离。

## 2. 方法分类总览

| 方法 | n-gram / memory 形式 | 低频独立参数风险 | 显式 hash collision | 主要可批判点 |
|---|---|---:|---:|---|
| Over-Encoding / Over-Tokenized Transformer | 1 到 \(N\) 阶 n-gram，modulo table，直接相加 | 中 | 高 | 低频 context 无独立身份；collision 与输入 embedding 混合 |
| Engram | suffix n-gram，多头 hash，gate 后注入中间层 | 中 | 高 | collision noise 交给 gate 处理；没有 frequency-aware routing |
| LongCat N-gram Embedding | Engram/Over-Encoding 式 polynomial hash 和多 sub-table | 中 | 高 | collision 通过增大 table 和分解缓解，但仍然存在 |
| BLT hash n-gram embedding | byte n-gram，固定 hash table，加到 byte representation | 低到中 | 高 | byte 频率分布与 token n-gram 不同；frequency baseline 被 hash baseline 取代 |
| N-Grammer | latent code n-gram，product quantization 后查表 | 中 | 有，发生在 latent code 组合空间 | collision 与 quantization error 混在一起 |
| SCONE | 频繁 f-gram 精确集合，训练时共享 f-gram Transformer | 低于独立表 | 低/无 key collision | 对低频显式剪枝，但频率阈值和 domain shift 仍需检查 |
| TN-gram | CP tensorized full-space memory | 不以独立 key 更新 | 无显式 hash collision | 低秩共享可能欠拟合；未见组合未必获得足够有效表示 |
| X-gram | frequency-aware hybrid hash、alias mixing、ShortConv | 主动压缩 tail | 仍有 hash/shared tail | 它已承认 long-tail under-training，但需测试 gap 而非只测 row utilization |
| Infini-gram | suffix-array 外部统计检索 | 无 learned row | 无 embedding collision | 非端到端训练；适合做检索上界而非同类参数化 baseline |

下面逐一展开。

## 3. Over-Encoding：低频和 collision 都存在

原始论文：[Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling](https://arxiv.org/abs/2501.16975)；[HTML](https://arxiv.org/html/2501.16975v2)

### 3.1 它实际做了什么

对于当前位置 \(i\)，Over-Encoding 把当前 token 和前面的 token 组成 n-gram，通过：

\[
x_i^{(-n)}
=
f(x_i,x_{i-1},\ldots,x_{i-n+1}),
\]

再用固定大小的 modulo table：

\[
\mathbb E^{m\times d}(x_i^{(-n)})
=
\mathbf E[x_i^{(-n)}\bmod m].
\]

最终将 1-gram、2-gram、直到 \(N\)-gram 的结果相加，并可通过多个低维 sub-table 和 projection 增加表达能力。

这意味着两个不同 n-gram \(g_1\neq g_2\) 只要满足：

\[
f(g_1)\bmod m=f(g_2)\bmod m,
\]

就会得到同一个 table row。

### 3.2 低频风险

Over-Encoding 没有按 n-gram frequency 建立 exact hot set。一个 context 是否进入某个 row，取决于 deterministic hash，而不是它在训练集出现了多少次。

因此可能同时出现：

- 一个罕见 n-gram 与一个高频 n-gram 共用 row；
- 两个都低频的 n-gram 共用 row，导致 row 的有效更新由多个稀疏 context 混合；
- 一个 validation-only context 命中训练时被其他 context 更新过的 row；
- 一个训练中重复出现的 context 通过同一 row 逐 epoch 累积 shortcut。

这里的关键不是“低频 row 没有更新”——因为 hash 使 row 可能被其他 key 更新——而是**低频 context 没有稳定、语义专属的参数**。

### 3.3 你们可以怎样批判

推荐的原文级批评：

> Over-Encoding reports scaling with the modulo table size, but table size alone does not identify whether the additional capacity is assigned to frequent, useful contexts or to collision partners in the long tail. A frequency-conditioned train/validation analysis is therefore needed.

应避免过强表述：

- 不能说 Over-Encoding 一定会导致 gap；
- 不能说所有 collision 都伤害性能；
- 不能把 modulo collision 和“独立 embedding row 的低频欠训练”混为一谈。

### 3.4 最直接的实验

固定 \(m,N,K\)，对每个 n-gram 做：

```text
frequency bucket
→ hash row id
→ row load / collision partner count
→ train loss
→ validation loss
→ gate or contribution
```

核心图应该是：

\[
\Delta_{\mathrm{bucket}}
\quad\text{vs.}\quad
\operatorname{count}_{\mathrm{train}}
\]

再画：

\[
\Delta_{\mathrm{bucket}}
\quad\text{vs.}\quad
\operatorname{row\_load}.
\]

如果 frequency 能解释 gap，而 row load 在控制 frequency 后没有额外解释力，批判重点应放在长尾记忆化，而不是 collision。

## 4. Engram：最适合被你们直接批判的对象

原始论文：[Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372)；[HTML](https://arxiv.org/html/2601.07372v2)

### 4.1 collision 是架构的一部分

Engram 对每个 order \(n\) 和 hash head \(k\) 建立表：

\[
z_{t,n,k}
=
\varphi_{n,k}(g_{t,n}),
\qquad
e_{t,n,k}
=
E_{n,k}[z_{t,n,k}].
\]

不同 context 在任一 head 中可以碰撞。多 head 的作用是降低多个 head 同时发生不可区分碰撞的概率，但它不保证无碰撞。

Engram 之后把所有 retrieved vectors concat，再生成 key/value，并用当前 hidden state 计算 gate：

\[
\alpha_t
=
\sigma\left(
\frac{\operatorname{RMSNorm}(h_t)^\top
\operatorname{RMSNorm}(k_t)}
\sqrt d
\right).
\]

所以它隐含的假设是：collision 产生的噪声可以由 context-aware gate 学会抑制。

### 4.2 低频问题

Engram 论文明确指出 n-gram 天然服从 Zipf 分布，并提出 multi-level cache：高频 pattern 放在更快的存储层，长尾放在更慢的 memory tier。

这解决的是**系统访问频率**，不是**训练统计可靠性**。一个 rare n-gram 即使可以从 NVMe 被准确读出，也不代表它的 learned embedding 质量足够好。

在你们的设置中，低频 context 还有额外风险：

- 多 epoch 后，相同训练 context 重复激活相同 row；
- gate 可以增大对这些 memorized value 的依赖；
- train loss 下降不一定代表对应 context 在 validation 中可泛化；
- collision partner 的梯度会改变该 row，使其在不同 context 间折中。

Engram 的 gate 可能减少平均 collision damage，但也可能放大训练 shortcut：一旦某个 memory vector 对训练上下文有用，gate 会把它打开；这不保证同一个 hash row 在验证上下文中仍然合适。

### 4.3 最有力的 Engram 批评

你们可以提出：

> Engram treats context-aware gating as a sufficient mechanism for resolving hash ambiguity, but a scalar gate only decides how much of the mixed value to inject. It does not recover the identity of the original n-gram or undo gradients accumulated by collision partners.

这是一个很重要的数学边界：

- gate 可以学习 \(0\leq\alpha\leq1\)；
- 但 collision 后的 row 已经是多个 context 共享的参数；
- 一个 scalar gate 没有额外信息时，无法把同一个 row 分解成各个 context 的独立向量。

当然，hidden state \(h_t\) 提供了额外上下文，所以 gate 不是完全无能；但要证明它解决了 collision，必须测量：

- same-row different-context 的 gate 分布；
- collision partner 的 train/val gap；
- gate 与 frequency、row load 的交互；
- 控制 frequency 后 collision load 对 gap 的增量解释力。

## 5. LongCat N-gram Embedding：碰撞已经被作者显式承认

论文：[Scaling Embeddings Outperforms Scaling Experts in Language Models](https://arxiv.org/abs/2601.21204)；[HTML](https://arxiv.org/html/2601.21204v2)

LongCat 使用 polynomial rolling hash：

\[
\mathcal H_n(t_{i-n+1:i})
=
\left(
\sum_{j=0}^{n-1}t_{i-j}V_0^j
\right)
\bmod V_n.
\]

它的设计包含：

- 每个 n-gram order 的 table；
- 多 sub-table；
- projection；
- 仔细选择 vocabulary/table size；
- 避免某些 table size 产生异常高 collision rate。

论文专门分析了两项指标：

1. vocabulary hit rate：至少被激活过一次的 table entry 比例；
2. hash collisions：由于 modulo indexing 丢失的 unique representation。

这使 LongCat 成为你们最容易引用的证据之一：**作者已经承认 table size 会产生非平滑 collision regime**。这不仅是一个理论可能性，而是 hyperparameter sensitivity。

### 5.1 你们可以补充的批判

LongCat 主要通过选择更好的 table size 来降低 collision。这个策略关注：

\[
\Pr[h(g_1)=h(g_2)]
\]

或整体 collision rate，但你们可以问一个不同问题：

\[
\text{collision damage}(g)
\propto
\operatorname{count}(g)
\times
\text{loss mismatch of its collision partners}.
\]

整体 collision rate 低，不代表高影响的 collision 低。一个低频 context 和一个高频 context 的 collision，可能比两个低频 context 的 collision 更重要；反过来，如果两个 context 的目标行为相似，collision 可能无害甚至有正则化作用。

### 5.2 可验证预测

在 LongCat-style hash table 中：

- table size 接近 base vocabulary 的某些整数倍时，row collision concentration 增加；
- collision concentration 增加时，平均 validation loss 可能变差；
- 但 gap 的峰值位置不一定与 collision rate 峰值一致；
- gap 应更接近“collision load × frequency mismatch”的函数。

## 6. BLT：byte n-gram 的 collision 风险仍然存在

论文：[Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)；[ACL HTML/PDF](https://aclanthology.org/2025.acl-long.453/)

BLT 在 byte-level local encoder 中使用 fixed-size hash n-gram embeddings。论文的配置使用 byte n-gram，并把 hash embedding 加到 byte representation 上；研究了不同 n 和 table size。

### 6.1 为什么 BLT 不能直接排除你们的批评

BLT 的 byte alphabet 很小，远小于 subword vocabulary，因此完整 \(V^n\) 空间增长得慢一些；但它仍然使用固定大小 hash table，因此：

- 对 sufficiently large n，byte n-gram 组合空间仍然远大于 table；
- distinct byte strings 仍然可以共享 slot；
- byte n-gram 的频率也具有长尾；
- rare byte patterns 可能在训练中被重复数据或特定 domain 放大；
- hash table 只位于 local encoder，并不意味着 collision 不会传播到 downstream patch representation。

BLT 还报告 hash n-gram embedding 优于他们测试的 frequency-based n-gram embedding。这个结果不能直接反驳你们，因为两者可能同时改变了：

- candidate selection；
- table capacity；
- injection 位置；
- frequency threshold；
- byte representation 的覆盖率。

### 6.2 适合的批判方式

不要说“BLT 的 hash 一定会造成 gap”。更稳妥的说法是：

> BLT validates the usefulness of hashed local features, but its aggregate comparison between hash-based and frequency-based variants does not establish whether rare byte n-grams, collision partners, or their interaction with repeated training data drive the residual generalization behavior.

可做的诊断：

- 按 byte n-gram frequency bucket 画 train/val loss；
- 按 hash row load 画 gap；
- 区分 byte-level novel、rare、head；
- 比较 patch boundary 和 n-gram hit 的交互。

## 7. N-Grammer：collision 与 latent quantization 同时存在

论文：[N-Grammer: Augmenting Transformers with latent n-grams](https://arxiv.org/abs/2207.06366)；[OpenReview PDF](https://openreview.net/pdf?id=GxjCYmQAody)

N-Grammer 不是直接把 raw token tuple hash 到表里。它先把 token embeddings 离散化为 latent cluster IDs，再对 latent code 的 n-gram 使用 hashing table。论文描述了 universal hashing 和低 collision probability 的设计。

### 7.1 它的风险结构

N-Grammer 至少有两个混合层：

1. 不同原始 token 可能被 product quantization 映射到相同或相近的 latent code；
2. 不同 latent n-gram 可能再 hash 到同一个 n-gram embedding slot。

因此，若观察到某个 latent n-gram 的 gap，不能直接归因于 hash collision。它可能来自：

- quantization cluster 的语义混合；
- latent code frequency imbalance；
- n-gram hash collision；
- sparse table update；
- downstream gate/projection。

### 7.2 你们可以如何批判

N-Grammer 的“低 collision probability”是概率性质，不是 per-key collision-free guarantee。对长尾 context，实际更重要的是：

\[
\Pr(\text{collision with a high-mass partner}\mid g).
\]

这与全局平均 collision probability 不同。

建议在分析中把 collision 定义为两层：

```text
raw token n-gram
    → latent cluster n-gram
    → hash slot
```

分别统计：

- raw-to-latent many-to-one；
- latent-to-slot many-to-one；
- 每层的 frequency-conditioned gap。

## 8. SCONE：主要缓解你们批判的低频问题，但不是免疫

论文：[Scaling Embedding Layers in Language Models](https://arxiv.org/abs/2502.01637)；[HTML](https://arxiv.org/html/2502.01637v3)

SCONE 是最重要的反例，因为它已经采用：

- frequent f-gram selection；
- exact key set；
- BPE-style discovery；
- shared f-gram Transformer；
- inference-time cached embeddings。

### 8.1 它解决了什么

SCONE 不让每个低频 f-gram 都直接拥有一个独立、从零更新的 embedding。训练时用共享 f-gram Transformer 生成表示，避免了“一个 row 只收到几次梯度”的最直接问题。

因此，对 SCONE 不能简单地说：

> 低频 f-gram 会因为独立 table row 更新不足而 overfit。

这是不准确的，因为它的训练结构有共享参数。

### 8.2 它仍可能遇到什么

SCONE 的频率集合仍由训练语料决定，并固定在训练/推理期间。因此仍然可能有：

- threshold 附近的 f-gram 统计不稳定；
- train-domain high-frequency 但 validation-domain low-frequency；
- high-frequency key 的 representation 对训练重复模式过拟合；
- longest-match 使低阶和高阶 context 的 coverage 发生竞争；
- f-gram Transformer 学到训练语料中的模板 shortcut；
- exact key 没有 hash collision，但存在 train/validation support mismatch。

此外，SCONE 论文使用不同的频率 cutoff。公开版本的实验报告了例如：

- 10M f-grams 对应约 21,956 的 cutoff；
- 1B f-grams 对应约 70 的 cutoff；
- 在固定 20M f-gram budget 下，最大 n 从 2 到 8 时 cutoff 由约 7 增加到约 108。

这说明“只维护高频”本身不是一个固定方法，而是一组 budget/threshold 选择。你们可以批判的是：

> frequency selection may prevent collision and cold-row waste, but the paper does not establish that its chosen frequency cutoff minimizes repeated-epoch train/validation gap.

### 8.3 SCONE 的关键对照

你们与 SCONE 的最有价值比较不是“我们的 hash 表比 SCONE 好/坏”，而是：

| 维度 | SCONE | 你们的研究 |
|---|---|---|
| key | train corpus 高频集合 | 当前模型实际命中的 n-gram |
| value learning | shared f-gram Transformer | table/gate dynamics |
| collision | 主要消除 | 显式测量 |
| frequency | 用于选 key | 用于解释 gap |
| repeated epochs | 不是核心分析 | 核心机制 |
| validation support | 通过固定集合间接处理 | 直接按 bucket 观测 |

如果 SCONE-style exact hot table 仍出现低频 gap，那么你们的结论会更强：gap 不只是 collision 问题，而是“重复 epoch + memory capacity + route-specific credit assignment”的问题。

## 9. TN-gram：消除 hash collision，但不自动消除长尾

论文：[Tensorizing Engram: Sharing Latents Across N-Gram Embeddings is Beneficial in LLMs](https://arxiv.org/abs/2606.08347)；[HTML](https://arxiv.org/html/2606.08347v1)

TN-gram 用共享 CP factors 和 order-absorption vectors 表示 n-gram memory，目标是：

- 不使用 Engram-style per-order hash table；
- 避免显式 hash collision；
- 让 nested n-gram 共享 latent structure；
- 让未在训练集出现的高阶组合也能通过因子得到表示。

### 9.1 它消除的风险

如果把“collision”严格定义为两个 raw n-gram 被 modulo 映射到同一 row，那么 TN-gram 没有这个问题。

### 9.2 它仍可能遇到的风险

TN-gram 不是每个 n-gram 一个自由向量，而是：

\[
\mathcal T(i_1,\ldots,i_n)
\approx
\sum_{r=1}^{R}
A_1(i_1,r)\cdots A_n(i_n,r).
\]

因此它把 collision 换成了低秩共享：

- 两个 n-gram 不共享一个 hash row，但可能共享同一组 factor；
- rare token factor 仍然可能更新不足；
- rare token tuple 的输出依赖多个 factor 的乘积；
- 高阶组合的学习信号可能被低阶 factor 的频率分布主导；
- rank 太低时，长尾 context 可能被压到相同的低维子空间；
- rank 太高时，优化和 memory cost 上升。

这不是 hash collision，而是**representation interference / low-rank aliasing**。

### 9.3 可以怎样批判

对 TN-gram 的批评应更精确：

> Tensorization removes explicit hash collisions, but it does not remove frequency-conditioned estimation error. A rare tuple can still be represented by factors dominated by frequent contexts, and the resulting train/validation gap may persist even when collision count is exactly zero.

这正好给出一个强实验矩阵：

```text
Engram hash
TN-gram tensorized
SCONE exact frequent
```

在相同训练数据、相同 gate 和相同 memory budget 下，分别画：

- n-gram frequency → gap；
- row load / factor usage → gap；
- train/val contribution；
- multi-epoch trajectory。

## 10. X-gram：最接近你们问题的后续工作

论文：[Beyond N-gram: Data-Aware X-gram Extraction for Efficient Embedding Parameter Scaling](https://arxiv.org/abs/2604.21724)；[HTML](https://arxiv.org/html/2604.21724v1)；[代码](https://github.com/Longyichen/X-gram)

X-gram 的论文摘要直接把以下现象列为动机：

- Zipfian under-training of the long tail；
- heterogeneous demand across layers；
- slot collapse；
- 固定 lookup scaling 带来 redundant embeddings。

它的策略包括：

- VIP reservation，为高频 token 保留 dedicated capacity；
- hybrid hashing，把 long tail 路由到 shared buckets；
- alias mixing；
- ShortConv 提取局部结构；
- depth-aware gating；
- sparse-aware learning-rate schedule。

### 10.1 这对你们意味着什么

X-gram 是强相关工作，但它的目标与您们不同：

- X-gram 主要优化 memory utilization、parameter efficiency 和平均 benchmark；
- 你们关注 repeated epochs 下的 route-specific train/validation gap。

因此你们可以批判 X-gram 的一个明确缺口：

> X-gram addresses long-tail under-training at the row-utilization level, but row utilization and generalization gap are not equivalent. A bucket may be frequently updated and still overfit to repeated training contexts.

需要验证：

- VIP rows 是否有更大的 train/val gap；
- shared tail bucket 是否降低 gap 还是仅仅降低参数浪费；
- alias mixing 是否把多个 context 的训练信号混合成正则化；
- ShortConv 是否放大或抑制低频 shortcut；
- depth-aware gate 是否在后续 epoch 继续偏爱已经过拟合的 head memory。

### 10.2 你们可以复用 X-gram 的概念

把你们的 bucket 结果对齐为三组：

```text
dedicated head
shared tail
novel / unseen
```

分别报告：

\[
\text{mean gap},
\quad
\text{frequency-weighted gap contribution},
\quad
\text{per-token gap contribution}.
\]

这会比简单的“低频 vs 高频”更接近 X-gram 的架构分区。

## 11. Infini-gram：不是 collision baseline，而是检索上界

论文：[Infini-gram](https://arxiv.org/abs/2401.17377)

Infini-gram 用 suffix array 做外部 corpus lookup，支持 variable/unbounded n，并通过 backoff 计算统计 n-gram 概率。

它没有 learned embedding row，因此不存在：

- embedding row 的 hash collision；
- row 的 sparse update；
- gate 对 embedding value 的错误 credit assignment。

但它仍有：

- train corpus vs validation corpus support mismatch；
- 罕见长匹配的估计方差；
- memorization / retrieval leakage；
- external index 与端到端模型的接口问题。

因此它适合做：

- 统计检索上界；
- exact context coverage 对照；
- variable-length n 的 oracle；

不适合直接作为 Engram/Over-Encoding 的同类替代。

## 12. 哪些算法最可能出现你们关注的 gap

### 12.1 高风险组：直接 hash + 可训练 table

优先级最高：

1. **Engram**
2. **Over-Encoding**
3. **LongCat N-gram Embedding**
4. **BLT hash n-gram embedding**

共同特征：

- fixed hash tables；
- repeated context 激活相同 physical row；
- 训练期间 row 参数不断更新；
- 多数方案没有 frequency-conditioned gate；
- 多数方案只报告平均 validation 指标。

### 12.2 中风险组：latent/tensor shared memory

包括：

- N-Grammer；
- TN-gram；
- X-gram 的 shared tail。

它们不一定发生 raw hash collision，但仍可能有：

- latent code collision；
- factor sharing；
- shared bucket interference；
- high-frequency factor domination；
- route-specific gate mismatch。

### 12.3 较低直接风险组：shared generator / exact hot set

SCONE 的直接 low-frequency-row 风险较低，因为 f-gram vector 由共享模型生成；但它仍有：

- train/validation support mismatch；
- high-frequency memorization；
- threshold sensitivity；
- longest-match selection bias；
- repeated epoch over-specialization。

## 13. 你们能否“批判”这些工作：可以，但要用三层证据

### 第一层：相关性

证明某方法真的使用了：

- n-gram memory；
- trainable lookup；
- fixed hash；
- frequency-selected key；
- gate 或 injection。

### 第二层：机制

证明该方法中存在你们定义的对象：

- n-gram count；
- hash row；
- collision partner；
- gate/value norm；
- route-specific train/val loss。

### 第三层：反事实干预

只有第三层才能说“导致 gap”：

- 去掉低频 keys；
- 替换 exact hot table；
- 改变 collision assignment；
- freeze table/gate；
- 打乱 epoch-specific mapping；
- 保持参数量和 FLOPs 不变。

只做第一层和第二层时，应写“exposes a potential failure mode”，不要写“proves the method fails”。

## 14. 最重要的混淆：低频、collision、novel 不是一回事

建议在所有图表中区分：

| 标签 | 定义 |
|---|---|
| frequent | train count 高 |
| rare | train count 低但非零 |
| novel | train count 为零、validation 中出现 |
| cold row | physical row 更新次数低 |
| overloaded row | 多个 distinct n-gram 映射到同一 row |
| high-mass collision | collision partner 的总 train count 高 |
| exact hot | 被精确表收录 |
| hashed tail | 进入共享 hash bucket |

一个 n-gram 可以同时是：

- rare；
- hashed；
- overloaded row 的成员；
- validation 中重复出现；

也可以是：

- frequent；
- hashed；
- 独占 row；
- 仍然由于训练重复而 overfit。

如果不把这些标签拆开，容易错误地把所有 gap 都叫作 collision。

## 15. 推荐的统一实验矩阵

### 15.1 架构维度

| 组 | Memory | collision | frequency selection |
|---|---|---:|---:|
| A | vanilla no n-gram | 无 | 无 |
| B | Over-Encoding / Engram hash | 有 | 无 |
| C | hash + frequency cutoff | 有 | 有 |
| D | exact frequent table | 无 | 有 |
| E | exact hot + hashed tail | 热区无，尾部有 | 有 |
| F | TN-gram | 无显式 hash | 全空间低秩 |
| G | SCONE-style shared generator | 无 key collision | 有 |

### 15.2 训练维度

对每组加入：

- 1 epoch；
- 2 epochs；
- 3 epochs；
- mixed epoch；
- frozen table；
- frozen gate；
- epoch-dependent hash；
- repeated vs non-repeated data。

### 15.3 必须控制

- 总参数；
- activated FLOPs；
- table bytes；
- gate/injection position；
- optimizer；
- learning-rate schedule；
- train/validation split；
- n-gram orders；
- 每个 token 的 memory read 数。

## 16. 最有价值的图

### 16.1 Frequency-conditioned gap

\[
x=\log(1+\operatorname{count}_{\mathrm{train}}(g)),
\quad
y=\operatorname{val\_loss}(g)-\operatorname{train\_loss}(g).
\]

分别画 unigram、bigram、trigram。

### 16.2 Collision-conditioned gap

\[
x=\operatorname{row\_load}(h(g)),
\quad
y=\Delta_g.
\]

同时用颜色表示 frequency bucket。

### 16.3 Two-dimensional heatmap

横轴：frequency bucket。  
纵轴：row load bucket。  
颜色：gap contribution。

这张图最能回答：

- gap 是低频本身造成的；
- collision load 造成的；
- 还是两者交互造成的。

### 16.4 Gate mismatch

对每个 bucket 画：

\[
\operatorname{mean}(\alpha_g),
\quad
\operatorname{mean}(\|v_g\|),
\quad
\operatorname{mean}(\alpha_g\|v_g\|),
\quad
\Delta_g.
\]

如果 gate norm 在 gap 之后仍然增大，说明 gate 可能继续奖励训练 shortcut。

### 16.5 Hot/cold route decomposition

借鉴 Engram-Nine 的 route-stratified evaluation，分别画：

- train loss hot；
- validation loss hot；
- train loss cold；
- validation loss cold；
- hot/cold gap；
- total contribution；
- per-token contribution。

## 17. Engram-Nine：最直接的先例，但证据等级要谨慎

论文：[A Collision-Free Hot-Tier Extension for Engram-Style Conditional Memory](https://arxiv.org/abs/2601.16531)；[HTML](https://arxiv.org/html/2601.16531v2)

Engram-Nine 用 MPHF 给 top-frequency n-gram 建 collision-free hot tier，cold tier 保留原始 multi-head hash。其预印本报告：

- 在严格 iso-parameter 下，去掉高频 collision 并不稳定改善 validation loss；
- 训练过程中 hot/cold 的优势会发生 flip；
- collision-free 配置可能更早出现 flip；
- collision 可能带来隐式正则化；
- gate 早期偏好 hot positions，但该偏好可能在 hot positions 后来 loss 更高时仍然持续。

这和你们的研究非常接近，但它不是最终定论。原因包括：

- 目前是预印本；
- 中等规模 nanoGPT；
- 训练步数和数据规模有限；
- hot set 是静态的；
- MPHF hot path 与 hashed cold path 的表示结构不同；
- 两个随机种子不足以覆盖大规模方差。

它最重要的价值是：**它已经说明“无碰撞不一定更好”，因此你们不能把 paper framing 写成简单的 collision elimination。**

更好的 framing 是：

> Collision-free memory isolates index precision, while our frequency-conditioned gap analysis isolates generalization under repeated updates. The two axes need not have the same optimum.

## 18. 对你们论文主张的建议表述

### 可以有把握地说

- 现有方法大量依赖 hash-based n-gram memory；
- 这些方法允许 distinct n-grams 共享 physical slots；
- n-gram 的 Zipf 长尾使不同 context 的训练信号严重不均衡；
- 仅报告 aggregate validation loss 无法判断哪些 frequency bucket 贡献了 gap；
- context-aware gate 能抑制部分噪声，但不能恢复 collision 前的独立身份；
- exact hot table 和 frequency cutoff 是合理的可测试干预；
- collision-free 不必然优于 collision-prone，因为共享可能提供 regularization。

### 需要实验后再说

- 低频 n-gram 是 gap 的唯一原因；
- hash collision 是 gap 的主要原因；
- 高频 exact table 一定改善 validation；
- SCONE/X-gram 一定不会出现 gap；
- gate 一定放大 rare-context overfit；
- 每 epoch 改变 hash 一定消除 gap。

## 19. 最推荐的论文贡献定位

最有潜力的定位是：

> **Frequency-conditioned generalization diagnostics for sparse n-gram memory.**

具体贡献可以是：

1. 发现 aggregate loss 之外的 frequency-conditioned gap；
2. 证明 gap 与 n-gram hit frequency 相关；
3. 分离 low-frequency memorization、hash collision 和 gate amplification；
4. 在相同参数/FLOPs 下比较 hash、exact-hot、shared-generator 或 tensorized memory；
5. 给出 repeated-epoch training 下的 memory-freezing 或 frequency-aware routing 建议。

不要把贡献过早写成：

> We show that hash collisions are the cause of overfitting.

更准确的是：

> We show that n-gram memory creates frequency-dependent generalization regimes, and that collision precision, update sharing, and gating interact nontrivially with repeated training.

## 20. 最小可执行批判实验

如果只做一组实验，建议采用以下四个模型：

```text
M0: no n-gram
M1: Engram/Over-Encoding hash
M2: frequency cutoff + hash tail
M3: exact hot + hashed tail
```

每个模型保持：

- 相同 n；
- 相同 table bytes；
- 相同 gate；
- 相同 optimizer；
- 相同训练数据和 epochs。

每个 step 记录：

- n-gram ID；
- train frequency bucket；
- hash row；
- row load；
- hot/exact/cold route；
- gate；
- value norm；
- gated value norm；
- token loss。

最终输出：

1. frequency bucket 的 train/val/gap；
2. row load bucket 的 train/val/gap；
3. frequency × row load heatmap；
4. gate × frequency 曲线；
5. epoch 1/2/3 的轨迹；
6. 总贡献与 per-token 贡献。

只要 M1→M2 降低 gap，而 M2→M3 没有额外收益，就说明主要问题是低频 memory activation，而非 collision。

如果 M1→M2 不明显，但 M2→M3 明显改善，则支持 collision 或共享 row interference。

如果 M3 的 gap 仍然存在，则支持更一般的结论：**精确 lookup 不能解决 repeated-epoch memory overfitting**。

## References

1. [Huang et al., 2025 — Over-Tokenized Transformer](https://arxiv.org/abs/2501.16975)
2. [Cheng et al., 2026 — Engram](https://arxiv.org/abs/2601.07372)
3. [Liu et al., 2026 — Scaling Embeddings Outperforms Scaling Experts](https://arxiv.org/abs/2601.21204)
4. [Pagnoni et al., 2025 — Byte Latent Transformer](https://arxiv.org/abs/2412.09871)
5. [Roy et al., 2022 — N-Grammer](https://arxiv.org/abs/2207.06366)
6. [Yu et al., 2025 — SCONE](https://arxiv.org/abs/2502.01637)
7. [Zhou et al., 2026 — TN-gram](https://arxiv.org/abs/2606.08347)
8. [Chen et al., 2026 — X-gram](https://arxiv.org/abs/2604.21724)
9. [Lin, 2026 — Engram-Nine](https://arxiv.org/abs/2601.16531)
10. [Liu et al., 2024 — Infini-gram](https://arxiv.org/abs/2401.17377)