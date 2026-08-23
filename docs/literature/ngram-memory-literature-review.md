# N-gram Memory 文献调研：精确高频表、Hash、BPE 与可变长度上下文

更新时间：2026-07-30

## 结论先行

“只维护高频 n-gram，并用无碰撞的精确表查找”是一个合理且已经被公开论文实现过的方向。最直接的对应工作是 Google 的 **SCONE**：

- 先在训练语料中发现频繁 n-gram，形成固定的 f-gram 集合；
- 原始 tokenizer 和输出词表保持不变；
- 每个被选中的 f-gram 都有自己的精确 key，不通过 modulo hash 与其他 n-gram 共用 slot；
- 训练期间不直接为数以亿计的 f-gram 独立反向传播，而是用一个较小的 f-gram Transformer 生成表示；
- 训练完成后把表示预计算为 key-value 表，并放在 GPU 之外；
- 推理时按最长匹配的 f-gram 做低成本查表，找不到时退回普通 token。

因此，问题不是“防碰撞是否数学上可行”。它当然可行。真正的设计权衡是：

1. **表只对哪些 key 精确？** 全部 \(V^n\) 不现实；只对语料中筛选出来的高频集合精确，则可行。
2. **这些 key 的向量如何学习？** 每个 key 独立训练会造成更新稀疏；SCONE 用共享的 f-gram 模型解决这一点。
3. **如何支持不同长度？** 可以用有限最大长度 \(K\) 的最长匹配，也可以用后缀数组等外部检索结构支持无界 n；但后者不是一个普通的、端到端可训练的 embedding table。
4. **高频是否等于高价值？** 不一定。频率是很好的工程筛选标准，但不是最优的信息量标准。

换句话说，你提出的方案可以更准确地表述为：

> 保留 base tokenizer；从训练语料中选择一个预算受限的高频、多长度 f-gram 集合；为该集合建立静态无碰撞 key-value memory；用最长匹配或 backoff 查找；用 gate 将查表结果注入 Transformer。

这不是 Engram 的原始路线。Engram 选择了固定大小、多头 hashing，再用 context-aware gate 缓解碰撞噪声；SCONE 则选择了精确的频繁 n-gram 集合，并把“如何学习这些稀疏向量”单独建模。

## 1. 先区分三种“防碰撞”

### 1.1 给整个组合空间建无碰撞表

如果 base vocabulary 为 \(V\)，n-gram 长度为 \(n\)，完整表需要：

\[
V^n \times d
\]

个参数，其中 \(d\) 是 embedding 维度。

例如 \(V=30{,}000\)、\(d=4096\)：

- 2-gram 有 \(9\times 10^8\) 个组合；
- 仅 2-gram embedding 就需要约 \(3.69\times 10^{12}\) 个参数；
- 3-gram 会再乘以 \(30{,}000\)。

此时“查找是否是 \(O(1)\)”没有解决主要问题，因为表本身无法放下。这个版本的 collision-free design 不可行，除非使用极强的低秩、张量分解或其他参数化假设。

### 1.2 给“语料中出现过的所有 n-gram”建无碰撞表

设语料中实际出现过的 distinct n-gram 数为 \(U_n\)。精确表的空间复杂度变成：

\[
O\left(\sum_n U_n d\right),
\]

而不是 \(O(V^n d)\)。

这已经是可工程化的方案，但 \(U_n\) 仍然会随着语料规模和 n 增长得非常快。长语料中，大量 n-gram 只出现一次或极少次数；把它们都放进表里通常既浪费存储，也没有足够训练信号。

### 1.3 只给频繁 n-gram 建无碰撞表

设选中的集合为：

\[
\mathcal{F}=\{g:\operatorname{count}(g)\geq c,\; |g|\leq K\},
\]

并且总预算限制为 \(|\mathcal F|\leq B\)。此时：

- 每个 key 可以有唯一的 value；
- 查询可以用 hash dictionary、sorted dictionary、B+ tree 或 minimal perfect hash；
- 未命中的 n-gram 退回更短的 n-gram 或普通 token；
- 表大小由 \(B\) 控制，而不是由 \(V^n\) 控制。

这正是 SCONE 的基本方向。需要注意：SCONE 论文描述的是精确 f-gram key-value store，使用 dense embedding matrix 加 hash dictionary，NVMe 场景使用 LMDB/B+ tree；它并没有把“minimal perfect hash”作为核心贡献。minimal perfect hash 可以作为一种实现，但不是必要条件。

## 2. SCONE：已经实现了你的核心想法

论文：[Scaling Embedding Layers in Language Models](https://arxiv.org/abs/2502.01637)；[HTML 版本](https://arxiv.org/html/2502.01637v3)

SCONE 的全称是 **Scalable, Contextualized, Offloaded, N-gram Embedding**。它与“直接扩大 tokenizer vocabulary”不同：

- base tokenizer 不变；
- output vocabulary 不变，因此 logits 计算不随 f-gram 数目扩大；
- f-gram 只作为 input-side 的额外表示；
- f-gram 表可以放到 host memory 或 NVMe；
- 推理时只为当前命中的 f-gram 取向量。

### 2.1 如何发现 f-gram：BPE-inspired，但不是重新训练 BPE tokenizer

SCONE 先定义最大长度 \(K\)，考虑：

\[
V_{\text{token}}^{[2,K]}
=
\bigcup_{n=2}^{K}V_{\text{token}}^n.
\]

它对训练语料做多次扫描：

1. 统计 2-gram；
2. 对后续 n-gram 设置最小频次阈值，论文描述的示例阈值为 5；
3. 利用前一阶 n-gram 的频次剪枝下一阶候选；
4. 对所有发现的 n-gram 按频率排序；
5. 选择 top f-grams，形成 \(\mathcal F\)。

这里的“BPE-style”只表示发现过程受到 BPE 的频率合并思想启发。它没有真的把 `a b c` 替换为一个新 token，也没有改变主模型的序列长度和 output vocabulary。SCONE 的方法更像：

```text
base tokens
    ├── ordinary token embedding
    └── longest matching frequent f-gram embedding
```

在 SCONE 的算法中，当前位置 \(i\) 会寻找以当前位置结尾的、最长的已收录 f-gram。若不存在，则只使用普通 token embedding。

### 2.2 为什么不直接对大表反向传播

如果每个 f-gram 都是一个独立 parameter，那么一个 key 收到梯度的次数大致等于它在训练序列中被命中的次数。词表越大，越多 embedding 进入长尾，导致：

- 很多向量几乎没有更新；
- 低频 key 的表示质量不稳定；
- 新的、相近的 n-gram 之间不能共享统计信号；
- 直接扩大表时，参数量增加得比有效训练信号快。

SCONE 的关键设计是把训练和推理解耦：

1. 训练时，用共享参数的 f-gram Transformer \(\mathcal A_{\text{f-gram}}\) 读取 f-gram 中的 token embeddings；
2. 该模型输出 f-gram 的 contextualized vector；
3. 主模型使用这个向量训练；
4. 训练结束后，对所有选中的 f-gram 离线计算向量；
5. 推理时用预计算的精确表 \(\mathcal F\)，不再运行 f-gram Transformer。

这解决的是“精确表的每个 row 都要独立学”的问题，但代价是训练阶段增加了一个模型和额外 FLOPs。SCONE 的卖点是：这些额外训练成本不必转化为推理时的 GPU 参数、FLOPs 和显存。

### 2.3 SCONE 的复杂度边界

对于固定的 \(K\)：

- 发现候选：约 \(K-1\) 次语料扫描；
- 查询：固定长度下可以看作 \(O(1)\) 的 key lookup；
- 实际向量读取：至少要付出 \(O(d)\) 的内存带宽成本；
- 存储：约 \(O(|\mathcal F|d)\) 加上 key/index 结构；
- 未命中：需要 longest-match 或 backoff 逻辑。

所以“\(O(1)\) lookup”只描述索引定位，不代表整个操作没有成本。对 embedding memory 来说，实际瓶颈通常是：

- host-to-device memory bandwidth；
- 随机访问造成的 cache miss；
- 多卡训练中的 sparse gradient aggregation；
- 多个候选长度带来的额外 key 构造和查找。

## 3. 为什么不总是用 minimal perfect hash

### 3.1 它确实可以做到唯一映射

对于一个提前知道、不会变化的 key 集合 \(\mathcal F\)，minimal perfect hash 可以建立：

\[
h:\mathcal F\rightarrow\{0,\ldots,|\mathcal F|-1\},
\]

且对 \(\mathcal F\) 内的 key 没有碰撞。这样每个 key 只占一个 embedding row。

对于静态推理表，这是完全合理的工程选择。它的优势是：

- 无 collision；
- 低额外索引空间；
- 常数时间查询；
- 适合离线构建后长期部署。

### 3.2 它没有解决四个更大的问题

#### 问题一：key 集合不是免费得到的

要构造 \(\mathcal F\)，必须先：

- 扫描训练语料；
- 统计频率；
- 选择阈值或 top-B；
- 决定不同 n 之间如何分配预算；
- 处理版本变化和数据增量。

minimal perfect hash 的构建本身可以是离线一次性成本，但它不是在线 \(O(1)\) 的过程。

#### 问题二：它只保证索引不碰撞，不保证向量学得好

如果一个 f-gram 只出现 5 次，给它一个独立 row 并不能创造训练信号。精确表把“别的 n-gram 的统计量也更新到这个 row”彻底切断了。

Hash 的碰撞有害的一面是语义混杂，但也有可能带来统计共享和正则化。SCONE 的处理方式不是接受碰撞，而是使用共享的 f-gram model，让不同 key 通过 token-level 参数共享。

#### 问题三：未登录 key 没有自然的表示

精确表只知道 \(\mathcal F\) 中的 key。对于：

- validation/test 中未在训练集合出现过的组合；
- 训练中出现次数低于阈值的组合；
- 领域迁移后出现的新组合；

必须设计 fallback。常见 fallback 包括：

- 更短的 suffix；
- 逐 token embedding；
- 子词/字符级 compositional encoder；
- hashed tail table。

这意味着实际系统通常需要“精确热表 + fallback”，而不是单一精确表。

#### 问题四：表更新会破坏静态索引

minimal perfect hash 最适合静态集合。如果持续加入新 n-gram，通常需要重建函数或增加额外层。在线训练和增量语料场景下，普通 hash dictionary、分层表或 immutable snapshot 往往更灵活。

因此，minimal perfect hash 适合：

- 训练完成后的 frozen inference table；
- key 集合稳定；
- 主要目标是压缩索引和消除 collision。

它不一定适合：

- 端到端训练期间持续加入 key；
- 频率随着数据流不断更新；
- 需要快速实验多个阈值和预算；
- 多卡训练中的动态 sparse update。

## 4. “n 为什么不直接无限延伸？”

### 4.1 固定 K 是工程约束，不是数学定理

固定 \(K\) 的好处是可控：

- 只需要扫描有限个 n；
- 每个位置最多处理有限个候选；
- 计算图和 kernel 容易实现；
- key 的最大长度、缓存和通信量有上界；
- 可以把不同 n 的容量和参数预算单独调节。

SCONE 采用的就是最大长度 \(K\) 的 finite f-gram 集合。其方法章节以 2 到 \(K\) 为定义，实验展示了 2 到 6-gram 的频率增长。这里的 \(K\) 是超参数，而不是认为自然语言在某个理论长度处终止。

Engram 也采用最大 n-gram order \(N\)，并对每个 order 和 hash head 建立 memory table。它的核心目标是固定预算下的条件 memory，而不是构造一个无界 n-gram 检索器。

### 4.2 继续增大 n 的实际问题

随着 n 增大：

1. distinct key 数快速增长；
2. 单个 key 的训练样本数下降；
3. 当前序列中命中长 key 的概率下降；
4. 更容易出现 train-only key；
5. 表容量和离线预计算成本增长；
6. 训练时每个 token 需要处理的候选长度变多；
7. 更长的 key 可能把重复文本、数据污染和记忆化一起带进模型。

因此，固定 \(K\) 本质上是一个 bias-coverage-storage trade-off。不是“n=3 在数学上最合理”，而是实验和系统约束下常常已经足够。

### 4.3 Infini-gram 给出了“无界 n”的另一种答案

论文：[Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377)；[HTML 版本](https://arxiv.org/html/2401.17377v4)

Infini-gram 不建立一个可训练的 embedding table，而是：

- 把整个 token corpus 建成 suffix array；
- 查询给定 prefix 的最长可匹配 suffix；
- 使用该 suffix 的计数估计下一个 token；
- 如果长上下文不存在，就 backoff 到更短上下文。

它实现的是外部、非参数化的统计检索。论文报告了：

- index 约 7 bytes/token；
- n-gram count 查询平均延迟低于 20 ms；
- 支持任意长度的 n；
- index 可以留在磁盘上。

它证明“无界 n 的查询”是可以做的，但不能直接推出“Transformer 内部应该使用一个无界、端到端训练的 embedding 表”。两者的接口不同：

| 方案 | memory 内容 | 是否端到端训练 | n | 主要成本 |
|---|---|---:|---:|---|
| Engram | learned hash embeddings | 是 | 有限 \(N\) | table、collision、通信 |
| SCONE | learned f-gram vectors | 训练时是；推理时 frozen | 有限 \(K\) | f-gram model 与离线缓存 |
| Infini-gram | corpus counts/index | 否 | 无界 | suffix-array 构建、检索、IO |
| TN-gram | tensorized learned function | 是 | 有限最大阶，参数随阶数线性增长 | tensor rank、表达能力和计算 |

## 5. 为什么“继续用 BPE”不完全等价

### 5.1 BPE 的目标是改变离散化

传统 BPE 反复执行：

1. 统计相邻 symbol pair；
2. 选择频率最高的 pair；
3. 把 pair merge 成新 symbol；
4. 在重新编码后的序列上继续 merge。

它的目标是生成一个新的 tokenizer vocabulary，并减少序列长度。

如果把 BPE 直接用于 over-encoding，会改变：

- token boundary；
- 序列长度；
- position id；
- attention pattern；
- output tokenization；
- 训练标签和解码接口。

### 5.2 f-gram memory 的目标是保留 base tokenizer

SCONE 和 Engram 这类方法不需要改变主序列。一个当前位置可以同时拥有：

\[
e_t
=
e_{\text{token}}(x_t)
+
e_{\text{f-gram}}(x_{t-k+1:t}),
\]

然后由 gate 决定 f-gram memory 对 hidden state 的影响。这样：

- output head 仍然只预测 base token；
- 不需要重新定义生成协议；
- f-gram memory 可以单独 offload；
- 多种长度可以作为辅助 memory，而不必参与 tokenizer segmentation。

所以，“BPE-style discovery”是发现频繁组合的办法；“BPE tokenization”是改变整个离散序列的办法。SCONE 采用前者，不是后者。

### 5.3 BPE 的贪心 merge 和最长匹配也不一样

BPE 的 merge 是全局词表构建过程；模型运行时只对已经训练好的 tokenizer 做确定性分词。f-gram memory 则是在每个位置动态判断：

- 是否存在该 suffix；
- 多个长度都存在时选哪个；
- 是否同时累加多个 order；
- 未命中时如何 backoff；
- 是否让 gate 读到所有候选而不是只读最长候选。

这些是 memory routing 问题，不仅仅是 tokenization 问题。

## 6. 频率筛选是否“数学上更合理”

频率筛选是一个很好的第一版，但“只保留高频”不等于“保留最有用”。

### 6.1 高频的优点

- 高频 row 有更多梯度或更多预计算样本；
- 查询命中率高，单位存储的覆盖率好；
- host memory cache 更容易获得高 hit rate；
- 频率统计简单、可复现、无需额外模型；
- 高阶候选可以通过低阶频次做有效剪枝。

### 6.2 高频的不足

高频组合可能是：

- 信息量很低的功能词组合；
- 语料中的格式模板；
- 只在训练域中常见、在验证域中不常见的模式；
- 与其他高频组合高度冗余的 key；
- 频繁但不需要独立 memory 的局部模式。

相反，一些中频组合可能具有很高的条件信息量，例如实体、API 名称、数学短语或领域术语。

更精确的候选评分可以考虑：

\[
\operatorname{score}(g)
=
\operatorname{count}(g)
\times
\operatorname{utility}(g),
\]

其中 utility 可以由以下指标估计：

- PMI 或 normalized PMI；
- 条件熵下降；
- token-level loss reduction；
- train/validation 上的预测增益；
- 每 byte 的 loss gain；
- 跨文档、跨 domain 的覆盖率；
- 与已有 f-gram 表示的去冗余程度。

实际可行的第一版仍然应该以 frequency threshold/top-B 为主，因为它最容易建立干净 baseline。更复杂的 utility selection 应该放在第二阶段。

## 7. TN-gram：另一条“无 collision 但不存完整表”的路线

论文：[Tensorizing Engram: Sharing Latents Across N-Gram Embeddings is Beneficial in LLMs](https://arxiv.org/abs/2606.08347)；[HTML 版本](https://arxiv.org/html/2606.08347v1)

TN-gram 的出发点与你的直觉一致：Engram 的独立 hash table 有两类缺陷：

1. 不同 n-gram 可能 collision；
2. `New York` 与 `New York City` 这种嵌套结构没有显式共享。

它用 CP decomposition 表示整个 n-gram tensor：

\[
\mathcal T(i_1,\ldots,i_n)
\approx
\sum_{r=1}^{R}
A_1(i_1,r)\cdots A_n(i_n,r),
\]

并使用跨 order 共享的因子和 order-absorption vectors。参数量从完整表的指数形式降低到近似：

\[
O(nVR+dR).
\]

它的性质是：

- 对 token tuple 的寻址没有 hash collision；
- 未直接见过的 tuple 也能通过共享因子得到表示；
- 参数量随 order 近似线性增加；
- 但它不再是“每个 key 一个完全独立的自由 embedding”；
- 表达能力受到 tensor rank \(R\) 限制；
- 查询需要若干因子查找和乘法，不是纯粹的一次 embedding row fetch。

因此 TN-gram 是第三种路线：

```text
full exact table       : 无 collision，但空间指数爆炸
frequency exact table  : 无 collision，但只覆盖选中的 key
hash table             : 固定空间，但 collision
tensorized memory      : 无显式 collision，靠低秩共享覆盖全空间
```

它适合回答“能不能不 collision 同时保留 compositional generalization”，但不一定替代你想要的“只保留高频 key”的工程方案。

## 8. N-Grammer 与 Hash Embeddings 的位置

### N-Grammer

论文：[N-Grammer: Augmenting Transformers with latent n-grams](https://arxiv.org/abs/2207.06366)

N-Grammer 先把 token embedding 通过 product quantization 映射到离散 latent code，再对 latent code 的 n-gram 做查表。它不是直接对原始 token id 做巨大 \(V^n\) 表，因此通过离散 codebook 压缩组合空间。

它说明了一个重要事实：n-gram memory 不一定必须基于原始 token tuple；可以基于共享的 latent representation。代价是增加 codebook、量化误差和额外的离散化设计。

### Hash Embeddings

论文：[Hash Embeddings for Efficient Word Representations](https://arxiv.org/abs/1709.03933)

Hash Embeddings 不是简单地让多个 token 共用同一个向量，而是为一个 token 取多个 hash slot，再用额外权重组合。这是 Engram 多头 hashing 的历史近邻：多次独立映射可以降低单一 collision 对最终表示的破坏，但不能让 collision 消失。

## 9. 对你的问题逐条回答

### Q1：为什么不直接用防碰撞 hash 表？

可以，前提是 key 集合被限制为有限的、离线确定的 \(\mathcal F\)。这就是 SCONE 的精确 f-gram memory 路线。

不可以的版本是：对全部 \(V^n\) 组合建立固定大小且完全无碰撞的表。它在存储上不可行。

### Q2：overencoding 的 n-gram 是固定大小吗？

Over-Encoding 论文的 embedding 结构使用有限的 n-gram orders，并通过 modulo hashing 让表大小独立于完整 \(V^n\)。它不是无界 n。

SCONE 明确把最大 f-gram 长度 \(K\) 作为方法参数，并在 2 到 6-gram 的候选发现上展示扩展趋势。Engram 同样使用最大 order \(N\)。

“固定最大 n”是计算和存储边界，不是语言建模的数学必然性。

### Q3：为什么不继续用 BPE？

如果“用 BPE”指的是从语料中发现高频组合，SCONE 已经这么做了。

如果“用 BPE”指的是把组合真正 merge 成新 token，那么它会改变主模型序列、位置、输出 vocabulary 和解码协议；这与 over-encoding memory 的目标不同。

### Q4：按照词频维护 n-gram 是否 \(O(1)\)？

运行时对一个已经选中的固定 key 做 lookup，可以是 expected \(O(1)\)。但完整系统还包括：

- 语料统计：至少线性扫描；
- key 选择：排序或 top-B；
- 可变长度匹配：需要最多 \(K\) 个候选或 trie/backoff；
- embedding 读取：\(O(d)\) 内存带宽；
- 训练更新：sparse gradient、聚合、通信；
- 增量更新：可能需要重建静态索引。

所以“查询是 \(O(1)\)”是正确但不完整的说法。

### Q5：只维护高频是否更合理？

作为第一个强 baseline，是的。它有明确的覆盖率、容量和命中率控制，也最接近 SCONE。

但最终优选标准不应该只有 count。建议比较：

1. top-B by frequency；
2. threshold by frequency；
3. frequency × PMI；
4. frequency × validation loss reduction；
5. per-byte utility；
6. exact hot set + hashed tail。

## 10. 推荐的实现路线

如果目标是验证“精确高频 memory 是否优于固定 hash memory”，建议不要一开始引入 minimal perfect hash、tensor network 和动态 BPE 三个变量。先做以下最小对照：

### 实验 A：Exact-Frequent Memory

- base tokenizer 不变；
- 统计 train-only 的 2-gram 和 3-gram；
- 设定 `min_count ∈ {2, 5, 10, 50}`；
- 设定总表预算 `B`；
- 用普通 immutable dictionary 或排序数组实现精确 key；
- 查找采用最长 suffix match；
- 未命中退回 unigram；
- 保持现有 gate、value projection 和插入层位置不变。

### 实验 B：Exact-Hot + Hashed-Tail

- top-B 的高频 key 使用精确表；
- 其余候选进入固定容量 hash table；
- 让 gate 或一个显式 source bit 区分 exact-hot 与 hashed-tail；
- 比较同样总参数预算下的效果。

这个混合方案通常比“所有 key 都精确”更实用：热数据没有 collision，尾部仍然保留有限容量和统计共享。

### 实验 C：Variable-Length Frequent Memory

不要先固定只用 2-gram 或只用 3-gram，而是：

- 统计所有 \(2\leq n\leq K\) 的候选；
- 用统一预算选 key；
- 每个位置选择最长命中的 f-gram；
- 另做一个“各 order 同时取值并由 gate 融合”的版本；
- 记录每个 order 的 hit coverage、平均 key length、每 token memory bytes。

### 必须记录的系统指标

除了 loss 和 downstream score，还应记录：

- unique key 数；
- 每个 key 的 count 分布；
- train/validation hit rate；
- exact hit rate 与 fallback rate；
- 平均匹配长度；
- 每 token 读取的 embedding bytes；
- cache hit rate；
- offline 构建时间；
- 表索引大小；
- 训练期间每个 row 的更新次数；
- OOD 文本上的 key coverage。

这些指标能把“模型质量差异”拆成 coverage、训练信号、collision 和系统带宽四个因素。

## 11. 统一比较表

| 方法 | key 集合 | collision | n | 训练方式 | 推理 memory | 主要优点 | 主要代价 |
|---|---|---|---|---|---|---|---|
| Over-Encoding | 所有 n-gram 经 modulo 映射 | 有 | 有限 | 直接训练 hash table | input-side table | 简单、可扩表、额外成本小 | collision，低频/未见组合共享不透明 |
| Engram | 所有局部 suffix n-gram 经多头 hash | 有，但多头缓解 | 有限 \(N\) | 直接训练 sparse tables | 可 offload | gate、MoE 互补、预取友好 | table collision、跨 order 不共享 |
| SCONE | 频繁 f-gram 精确集合 | 无 key collision | 有限 \(K\) | f-gram model 生成向量 | frozen exact table | 高频覆盖、精确 lookup、可 offload | 训练增加模型；尾部需 fallback |
| Infini-gram | corpus 中的所有可检索 substring | 无计数 collision | 无界 | 非参数 corpus index | suffix array/index | 无界 n、强检索能力 | 非端到端 embedding；构建和 IO 成本 |
| TN-gram | tensorized 全空间 | 无显式 hash collision | 有限最大 order | 端到端低秩参数化 | factor tables | 参数随 order 近似线性、跨 order 共享 | rank 限制表达能力，非单 row lookup |
| N-Grammer | latent code 的 n-gram | code lookup 可能共享 | 有限 | 离散 latent + embedding | latent n-gram table | 组合空间压缩、可泛化 | 量化和 codebook 设计复杂 |

## 12. 最终判断

你的直觉可以保留，而且应该把问题从“为什么大家不用防碰撞 hash”改写成：

> 在固定 memory budget 下，精确高频表、随机/多头 hash 表、以及共享低秩参数化，哪一种在 coverage、训练信号、泛化和系统带宽之间的 Pareto frontier 更好？

文献给出的答案不是单一结论：

- **SCONE** 证明了“高频 + 精确 key + offloaded table”是可行的；
- **Engram** 证明了“固定容量 hash + context-aware gate”在大规模 sparse model 中有很强的 scaling potential；
- **Infini-gram** 证明了“无界 n”更适合由外部 suffix-array 检索实现；
- **TN-gram** 证明了“避免 hash collision”也可以通过共享的低秩结构实现，而不必显式存储每个 key。

因此，最合理的下一步不是直接替换为完整 BPE tokenizer，而是先实现一个 **SCONE-style exact frequent f-gram baseline**，再在相同总参数和相同 gate 下与 Engram-style hash baseline 对比。这样可以单独测量“精确高频 key”本身的收益，而不会把 tokenizer、gate、n、表大小和训练方式同时混在一起。

## References

1. [Yu et al., 2025 — Scaling Embedding Layers in Language Models (SCONE)](https://arxiv.org/abs/2502.01637)
2. [Huang et al., 2025 — Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling](https://arxiv.org/abs/2501.16975)
3. [Cheng et al., 2026 — Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram)](https://arxiv.org/abs/2601.07372)
4. [Zhou et al., 2026 — Tensorizing Engram: Sharing Latents Across N-Gram Embeddings is Beneficial in LLMs](https://arxiv.org/abs/2606.08347)
5. [Liu et al., 2024/2025 — Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377)
6. [Roy et al., 2022 — N-Grammer: Augmenting Transformers with latent n-grams](https://arxiv.org/abs/2207.06366)
7. [Svenstrup et al., 2017 — Hash Embeddings for Efficient Word Representations](https://arxiv.org/abs/1709.03933)
8. [Pagnoni et al., 2025 — Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)
9. [Liu et al., 2026 — Scaling Embeddings Outperforms Scaling Experts in Language Models](https://arxiv.org/abs/2601.21204)