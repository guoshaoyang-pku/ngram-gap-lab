# SCONE vs Engram：架构选择、Infra 权衡、伪代码与未探索空间

更新时间：2026-07-30

## 1. 为什么 DeepSeek 选了 Engram 而不是 SCONE？

SCONE 确实在概念上更直接——"先统计高频组合，再精确建表，用共享模型生成向量"——但 Engram 对 DeepSeek 的场景有多项关键优势：

### 1.1 输入层 vs 中间层：这是根本分歧

SCONE 主要作用在输入 embedding 层。它替换的是：

```
token → embedding → 第一层 Transformer
```

变成：

```
token → embedding + longest-matched f-gram → 第一层 Transformer
```

Engram 直接插入中间层：

```
... → 第 k 层 hidden state → Engram memory injection → Attention → MoE → ...
```

论文实验显示 Engram 放在第 2 层和第 15 层。这对大型 MoE 的意义是：**Engram 把局部 pattern 的识别从早期 Transformer 块中卸载出去，让这些块腾出容量处理更复杂的推理。**

SCONE 让输入表示更好，但之后依然需要主干逐层传递和重建局部信息。Engram 直接在中间层把这块工作接管了。

### 1.2 训练成本：SCONE 需要额外训练一个 Transformer

SCONE 训练时需要：

```python
# 训练时
e = A_fgram( E(token_j), ..., E(token_i) )  # 额外跑一个 f-gram Transformer
# 推理时
e = F[ (token_j, ..., token_i) ]             # 查预计算表
```

这意味着训练时同时跑两个模型。SCONE 论文的实验设置中，f-gram 模型本身就有 1.8B 参数，且总训练 token 数减半（500B vs 1T）来平衡成本。

Engram 的训练不需要额外模型——它直接对 embedding table 反向传播，通过 table sharding + All-to-All 通信扩展到多卡。

### 1.3 推理时的确定性 prefetch 是 Engram 的核心 infra 优势

这是最关键的系统设计差异。Engram 的 lookup 地址完全由输入 token ID 决定，**在 forward pass 开始之前就已经知道**。因此：

```
时间线：
  t0: 已知所有 token IDs → 已知所有 n-gram hash 地址
  t1: 异步从 host memory 开始 prefetch 对应行
  t2: GPU 计算前面的 Transformer 层
  t3: prefetch 完成，数据已在 GPU
  t4: Engram 模块执行，数据已就绪
```

SCONE 的 longest-match 虽然也是确定性的，但查找过程需要先找到当前 token 对应的最长 f-gram（需要 dictionary/trie 查询），而且通常只放在输入层，没有中间层计算做 buffer 来隐藏延迟。

### 1.4 SCONE 的 train/inference 表示不一致

SCONE 训练时用 f-gram Transformer 生成向量，推理时用缓存的最终快照。如果缓存没有及时更新或 generator 的最终状态与 main model 的联合训练不够稳定，推理时的表示可能与训练时不同。

Engram 训练和推理时使用完全相同的 lookup 逻辑——训练时查表+反向传播，推理时查表（offload）。

### 1.5 SCONE 的 key 集合是 domain-dependent

SCONE 需要在训练语料上统计频率确定 f-gram 集合。换到新领域时，很多有价值的短语不在集合中，只能退回 token embedding。Engram 的 hash 是 vocabulary-agnostic——任何 suffix 都能映射到某个 hash slot，不需要重新统计。

### 1.6 DeepSeek 有 mHC 多分支架构，Engram 天然适配

DeepSeek 使用的是 Manifold-Constrained Hyper-Connections（M=4 分支），不是标准单流 Transformer。Engram 的设计中 embedding table 和 W_V 跨分支共享，每个分支有独立 W_K，这正好适配多分支架构。SCONE 的输入层替换方案没有这种分支级适配。

### 1.7 Sparsity Allocation

Engram 论文的核心贡献之一是在 MoE 和 memory 之间做参数分配，找到了 U 形 scaling law：把 20-25% 的参数预算从 MoE 搬给 memory，效果优于纯 MoE。SCONE 没有这个分析框架——它把 embedding scaling 作为独立的、与主干大小无关的 scaling axis。

### 1.8 Engram 对 SCONE 的明确引用

Engram 论文在 Related Work 中引用了 SCONE（Yu et al., 2025），但把它归类为 "input-layer n-gram embedding" 方向，与 Over-Encoding 和 BLT 并列。Engram 的核心叙事是：这些方法都只在输入层做 n-gram encoding，没有利用中间层 hidden state 做 context-aware gating，也没有解决 MoE 架构下的 memory 与 compute 的协同分配问题。

## 2. 关于 longest match 的 infra 问题

你担心的是对的。如果每次回传的 lookup 数量不一致（有时 1 个 2-gram，有时 1 个 3-gram，有时多个），会带来：

- 不规则的 memory read pattern；
- 动态的计算图；
- 难以做算子融合和 prefetch；
- 内存分配和释放的碎片化。

### SCONE 的 longest match：固定每次 1 个 f-gram

```python
def scone_forward(token_ids, fgram_set, fgram_cache_or_model):
    embeds = []
    for i, tid in enumerate(token_ids):
        # 找以 i 结尾的最长 f-gram
        j = i
        for L in range(max_len, 1, -1):
            if i - L + 1 >= 0:
                gram = tuple(token_ids[i-L+1:i+1])
                if gram in fgram_set:
                    j = i - L + 1
                    break
        if j == i:
            embeds.append(token_embed[tid])
        else:
            embeds.append(lookup(gram))
    return embeds
```

**优点：每次只查一个 f-gram + 一个 token embedding = 固定 2 次 memory read。**
**缺点：丢失其他阶的结构信息。**

### Engram：固定每次 (N-1) × K 次 lookup

```python
def engram_retrieve(token_ids, t, N, K):
    e = []
    for n in range(2, N+1):          # 2-gram, 3-gram, ...
        gram = token_ids[t-n+1:t+1]  # suffix
        for k in range(K):           # K hash heads
            idx = hash_k(gram, k) % M_nk
            e.append(E_nk[idx])
    return concat(e)                 # 固定维度: (N-1)*K*d
```

**每次固定 (N-1) × K 次 lookup，无论什么 context。**
**优点：计算图完全确定，可以融合、prefetch、batch。**
**缺点：memory traffic 随 N 和 K 线性增长，即使某些高阶 n-gram 没有信息量。**

### 折中方案：固定数量、动态选择 source

```python
# 固定 lookup (N-1) 次，但内容根据频率动态决定
for n in range(2, N+1):
    gram = token_ids[t-n+1:t+1]
    if gram in hot_set:
        e_n = hot_table[gram]           # 精确
    else:
        e_n = hash_table[hash(gram)]     # hash fallback
# 最终 concat(e_2, e_3, ...) 仍然是固定维度
```

这样每个 token 仍然有固定 (N-1) 个 lookup，但 hot n-gram 用精确 key，cold 用 hash。计算图形状不变，只是 lookup 的 source 不同。

## 3. Engram 如何适配 MoE 架构

### 3.1 整体结构

DeepSeek 使用的 backbone 是 **30 层 Transformer，每层包含 multi-head latent attention + MoE FFN，通过 Manifold-Constrained Hyper-Connections (M=4 branches) 连接。**

Engram 被插入到特定层的 Attention 之前：

```
输入 token IDs
  → token embedding + 第 0 层
  → 第 1 层: Attention → MoE
  → [Engram 模块 @ layer 2]
  → 第 2 层: Attention → MoE
  → ...
  → [Engram 模块 @ layer 15]
  → 第 15 层: Attention → MoE
  → ...
  → 第 30 层 → output
```

### 3.2 哪些是共享的，哪些是独立的

| 组件 | 共享范围 | 数量 |
|---|---|---|
| **Embedding tables** E_{n,k} | 所有分支共享 | 每 (n,k) 对 1 个 |
| **Value projection** W_V | 所有分支共享 | 1 个 |
| **Key projections** W_K^{(m)} | 每个分支独立 | M 个 |
| **Hash 函数** φ_{n,k} | 固定，非学习 | 每 (n,k) 对 1 个 |
| **Depthwise Conv** | 所有分支共享 | 1 个（在 concat 后） |

关键设计：W_V 和 M 个 W_K^{(m)} 可以被融合成**一次 FP8 矩阵乘法**，最大化 GPU 利用率。

### 3.3 同一个词的不同含义如何被编码

以 "right" 为例：

```text
"turn right at the corner"   → right = 方向
"you're absolutely right"    → right = 正确
```

在 Engram 中：

1. **Hash 层面**：两个 "right" 的 suffix n-gram 不同：
   - "turn right" 和 "absolutely right" 是不同的 2-gram
   - 它们可能 hash 到不同 slot，也可能 collision 到同一 slot

2. **即使 collision 到同一 slot，gate 可以区分**：
   ```
   α_t = σ( RMSNorm(h_t)^T · RMSNorm(W_K · e_t) / √d )
   ```
   h_t 已经聚合了全局上下文（"turn" 和 "absolutely" 的语义不同），所以 `RMSNorm(h_t)` 不同。即使 e_t 相同（collision），`RMSNorm(h_t)^T · RMSNorm(W_K e_t)` 也会不同，gate 值不同。

3. **多分支多角度**：M=4 个分支各有独立的 W_K^{(m)}，每个分支可以从不同角度判断 "这个 memory 对我当前这个分支的上下文是否相关"。

4. **ShortConv 提供局部平滑**：即使 gate 没法完美区分，depthwise causal conv 可以在相邻 token 之间平滑 gate 输出，减少单个位置的噪声。

**但关键限制是**：gate 只能调节 `α_t · v_t` 的强度，不能改变 v_t 本身的方向。如果两个 n-gram 真的 collision 到同一个 slot，它们的 v_t = W_V · e_t 是完全相同的向量。gate 可以说"这次少用一点"，但不能说"从 e_t 中提取 right-as-direction 的成分，丢弃 right-as-correct 的成分"。这个限制在我们之前的文档中已经讨论过。

## 4. 数学伪代码

### 4.1 Over-Encoding (OE)

```python
# 参数
V = 32000          # base vocabulary size
d = 4096           # model dim
N = 3              # max n-gram order
K = 4              # number of sub-tables
m = 12800000       # hash table size per sub-table
p = V              # base for p-ary encoding

# 表
E_0 = Param(V, d)                           # 1-gram: V×d
E_nk = Param(m, d // ((N-1)*K))             # sub-table: m × d/((N-1)K)
W_nk = Param(d, d // ((N-1)*K))             # projection: d × d/((N-1)K)

def oe_forward(token_ids):
    T = len(token_ids)
    embeds = []
    for i in range(T):
        e_i = E_0[token_ids[i]]                    # 1-gram
        for n in range(2, N+1):
            if i - n + 1 < 0: break
            # p-ary encoding: map n tokens to unique integer
            x_n = sum(token_ids[i-j] * (p**j) for j in range(n))
            for k in range(K):
                h = E_nk[x_n % (m + k)]             # modulo hash, each sub-table slightly different size
                e_i = e_i + W_nk @ h                # project and add
        e_i = e_i / ((N-1)*K + 1)                   # normalize
        embeds.append(e_i)
    return embeds

# 关键：所有 n-gram 无差别 hash，不区分频率
# 关键：直接相加，没有 gate
# 关键：只作用在输入层
```

### 4.2 Engram

```python
# 参数
V = 128000         # raw vocabulary
d = 4096           # model dim
N = 3              # max n-gram order
K = 2              # hash heads per order
M_nk = [prime_sizes]  # prime table sizes
M_branches = 4     # mHC branches

# 表
P = precompute_canonical_map(V)              # NFKC + lowercase → V' (~23% smaller)
E_nk = [Param(M_nk, d_mem) for n,k]         # embedding tables
W_V  = Param(d, (N-1)*K*d_mem)              # shared value projection
W_K  = [Param(d, (N-1)*K*d_mem) for _ in range(M_branches)]  # branch-specific key

# Conv
conv = DepthwiseCausalConv1d(kernel=4, dilation=N)

def engram_forward(h_prev, token_ids, t):
    # h_prev: (B, M_branches, d)  - previous layer hidden states per branch
    # Phase 1: Deterministic Retrieval
    e_t = []
    for n in range(2, N+1):
        if t - n + 1 < 0: break
        g = tuple(P[tid] for tid in token_ids[t-n+1:t+1])   # canonical suffix
        for k in range(K):
            z = multiplicative_xor_hash(g, seed=k) % M_nk[n][k]
            e_t.append(E_nk[n][k][z])
    e_t = concat(e_t)  # shape: (N-1)*K*d_mem

    # Phase 2: Branch-specific Gating
    v_t = W_V @ e_t    # shared value: d
    outputs = []
    for m in range(M_branches):
        k_t = W_K[m] @ e_t                                     # branch key: d
        alpha = sigmoid(dot(RMSNorm(h_prev[m]), RMSNorm(k_t)) / sqrt(d))
        u_t = alpha * v_t                                      # gated value: d
        outputs.append(u_t)

    # Phase 3: ShortConv + Residual
    U = stack(outputs)                                    # (B, M_branches, d)
    U_norm = RMSNorm(U)
    Y = silu(conv(U_norm)) + U                            # causal conv + residual
    h = h_prev + Y                                        # residual connection
    return h

# 关键：后缀 n-gram → multi-head hash → concat → gate → conv → residual
# 关键：embedding table 和 W_V 跨分支共享，W_K 独立
# 关键：Gate 用 hidden state 做 query，检索 memory 做 key/value
# 关键：只插入特定层（layer 2, 15）
```

### 4.3 SCONE

```python
# 参数
K = 6              # max f-gram length
S = 10_000_000     # number of f-grams
d = 4096

# 训练前：BPE-style discovery
def discover_fgrams(corpus, K, S):
    candidates = {}  # gram → count
    for n in range(2, K+1):
        for seq in corpus:
            for i in range(len(seq)-n+1):
                gram = tuple(seq[i:i+n])
                if n == 2 or gram[:n-1] in prev_set:
                    candidates[gram] = candidates.get(gram, 0) + 1
        # 只保留 freq >= 5 的，减少内存
        candidates = {g: c for g, c in candidates.items() if c >= 5}
        prev_set = set(candidates.keys())
    # 全序排序，取 top S
    sorted_grams = sorted(candidates.items(), key=lambda x: -x[1])
    return set(g for g, _ in sorted_grams[:S])

# 训练时
E_tok = Param(V, d)                              # token embedding
A_fgram = Transformer(num_layers=..., dim=d)      # f-gram transformer
A_main = Transformer(num_layers=..., dim=d)       # main transformer
D = Linear(d, V)                                  # output head

def scone_train(token_ids):
    T = len(token_ids)
    embeds = []
    for i in range(T):
        # longest match
        j = i
        for L in range(min(K, i+1), 1, -1):
            gram = tuple(token_ids[i-L+1:i+1])
            if gram in FGRAM_SET:
                j = i - L + 1
                break
        if j == i:
            e_i = E_tok[token_ids[i]]
        else:
            # 用 f-gram transformer 生成 contextualized embedding
            tok_embeds = [E_tok[tid] for tid in token_ids[j:i+1]]
            e_i = A_fgram(tok_embeds)[-1]         # 取最后一个 token 的输出
        embeds.append(e_i)
    h = A_main(embeds)
    logits = D(h)
    return logits

# 推理时：预计算所有 f-gram
def scone_precompute():
    cache = {}
    for gram in FGRAM_SET:
        tok_embeds = [E_tok[tid] for tid in gram]
        cache[gram] = A_fgram(tok_embeds)[-1].detach()  # 冻结
    return cache  # 存到 host RAM 或 NVMe

def scone_inference(token_ids, cache):
    embeds = []
    for i in range(len(token_ids)):
        j = i
        for L in range(min(K, i+1), 1, -1):
            gram = tuple(token_ids[i-L+1:i+1])
            if gram in cache:
                j = i - L + 1
                break
        if j == i:
            e_i = E_tok[token_ids[i]]
        else:
            e_i = cache[gram]                      # O(1) dictionary lookup
        embeds.append(e_i)
    h = A_main(embeds)
    return D(h)
```

### 4.4 TN-gram

```python
# 参数
V = 32000
d = 4096
N = 6              # max n-gram order
R = 128            # CP rank

# CP factors (shared across orders)
A = [Param(V, R) for _ in range(N)]          # token-position factors: V×R
w = [Param(R) for _ in range(N-2)]           # order-absorption: R

# Gate projection (same as Engram)
W_K = Param(d, d)
W_V = Param(d, d)

def tngram_retrieve(t, token_ids):
    e_t = []
    for n in range(2, N+1):
        if t - n + 1 < 0: break
        # CP decomposition for n-gram embedding
        e_n = 0
        for r in range(R):
            val = 1.0
            for j in range(n):
                val *= A[j][token_ids[t-n+1+j], r]   # outer product of factors
            if n > 2:
                val *= w[n-2][r]                      # order-absorption
            e_n += val * u_r                           # u_r: learned basis vector
        e_t.append(e_n)
    e_t = concat(e_t)  # (N-1)*d
    return e_t

# 后续 gate 和 injection 与 Engram 相同
# 关键：无 hash collision，无显式表
# 关键：参数 O(NVR + dR)，随 N 线性增长
# 关键：高阶 n-gram 可以通过低阶因子组合泛化
```

### 4.5 X-gram

```python
# 核心三组件：VIP routing, ShortConv, depth-aware gate

# VIP reservation
N_vip = ...          # number of VIP tokens
E_vip = Param(N_vip, d_vip)          # dedicated VIP table
E_hash = Param(M, d_hash)            # shared hash table

def xgram_retrieve(t, token_ids):
    tid = token_ids[t]
    if tid in VIP_SET:
        e_vip = E_vip[vip_idx[tid]]              # dedicated slot
    # alias mixing: 从 hash table 取多个 slot，混合
    e_hash = []
    for h in range(H):
        idx = hash_h(token_ids[t-k:t+1]) % M
        e_hash.append(E_hash[idx])
    e_hash = alias_mix(e_hash)                   # weighted combination
    return e_vip + e_hash

# ShortConv refinement
def xgram_extract(e):
    e = RMSNorm(e)
    # Multi-scale depthwise conv
    out = 0
    for k in [3, 5, 7]:
        out += silu(conv_k(e))
    return gate(e) * out + e

# Injection: attention value stream + inter-layer residual
def xgram_inject(h, v, delta):
    v_new = v + delta                          # 注入 attention value
    h_new = h + delta                          # 注入 inter-layer residual
    return h_new, v_new
```

## 5. 各种排列组合是否都被尝试过了？

**没有，远未探索完。** 设计空间至少包含以下维度，且每个维度有多个选择：

| 维度 | 已知选择 | 已发表论文覆盖 |
|---|---|---|
| **Key 选择** | hash (Engram, OE, BLT), frequency exact (SCONE), frequency VIP+hash (X-gram), tensorized (TN-gram), latent code (N-Grammer) | 各自独立，缺 head-to-head |
| **Key 来源** | raw token, canonical token, byte, latent code | 独立，缺对照 |
| **n 阶数** | 2, 3, 2-6, 2-8, 2-9 | 各自选，少统一对比 |
| **Hash 机制** | single, multi-head, modulo, XOR, polynomial rolling | 各自选，缺 collision rate 对比 |
| **多阶组合** | sum (OE), concat (Engram), longest match (SCONE), sum+residual (X-gram) | 各自选，缺消融 |
| **表示学习** | direct table (OE, Engram), shared generator (SCONE), CP tensor (TN-gram) | 各自选，SCONE vs Engram 无直接对比 |
| **Gate 机制** | none (OE, SCONE), scalar sigmoid (Engram), multi-branch (Engram mHC), depth-aware (X-gram), per-source | 很少交叉 |
| **Gate 输入** | hidden state only, hidden + frequency, hidden + row load, hidden + confidence | 几乎未探索 |
| **Injection 位置** | input layer (OE, SCONE, BLT), specific intermediate layers (Engram), attention value stream (X-gram), inter-layer residual (X-gram) | 各自选 |
| **Conv 后处理** | none (OE, SCONE), depthwise causal (Engram), multi-scale SwiGLU (X-gram) | 各自选 |
| **Backbone 适配** | dense (OE, SCONE), MoE (Engram), multi-branch (Engram), byte-level (BLT) | 各自选 |
| **Optimizer 策略** | same LR, different LR, freeze schedule, memory dropout, gate dropout | 几乎未探索 |
| **Offload 策略** | host RAM dense+dict, NVMe LMDB, PCIe prefetch, tiered cache | 各自选 |

### 明确未探索的组合

以下是**没有任何已发表论文直接测试过**的组合，但对你们的研究最有价值：

1. **SCONE key selection + Engram gate + Engram injection**：高频精确 key，中间层 gate 注入，不用 hash collision
2. **TN-gram + Engram gate**：张量分解表示，无 hash collision，但保留 gate 和中间层注入
3. **Frequency-aware gate**：gate 输入除了 hidden state，还包含 frequency bucket 或 row load
4. **Exact hot + hashed tail + per-source gate**：高频精确，低频散列，gate 区分 source
5. **Memory freeze schedule**：epoch 1 后冻结 memory，只保留 gate 可训练
6. **Epoch-dependent hash seed**：每个 epoch 改变 hash mapping，防止 persistent addressing
7. **Memory LR schedule**：不同 frequency bucket 使用不同 LR
8. **SCONE 的 f-gram generator + TN-gram 的 tensorized output**：共享生成器参数化，但输出是张量形式
9. **Per-order capacity allocation**：2-gram 60% 容量，3-gram 30%，4-gram 10%
10. **Collision-confidence gate**：gate 显式接收 row load 或 collision multiplicity

### 哪些组合可能最有效

根据已有证据和你们的现象：

1. **SCONE hot selection + Engram gate + intermediate injection**（第 1 优先级）
2. **Exact hot + small hash tail + separate gates**（第 2 优先级）
3. **TN-gram + Engram gate**（第 3 优先级，纯无碰撞，但需要验证 rank 是否足够）
4. **Frequency-aware gate**（低风险，高收益，可以直接加在现有 Engram 上）
5. **Memory LR / freeze schedule**（低风险，对你们的 gap 问题直接相关）

## 6. 对你们论文的最终建议

不要写成 "SCONE is better than Engram" 或 "Engram is better than SCONE"。两者解决的问题不同。

你们最有价值的贡献可以是：

> **在 repeated-epoch training 下，hash collision 和 frequency-conditioned memory learning 的交互关系。这个分析维度在 SCONE、Engram、OE、TN-gram 和 X-gram 中都没有被系统研究。**

建议的核心实验矩阵：

```text
M0: 无 n-gram
M1: Engram hash (collision-prone, all suffix)
M2: frequency cutoff + hash tail (frequency-aware, collision-prone)
M3: exact hot + hash tail (frequency-aware, collision-free hot)
M4: exact hot + Engram gate (SCONE selection + Engram injection)
M5: TN-gram + Engram gate (no collision, tensorized)
```

在相同参数量、相同 FLOPs、相同 epochs 下，报告：

- frequency-conditioned train/val/gap
- collision row load
- gate/value/gated-value norm
- hot/cold contribution
- 每 token 的 memory bytes
- 缓存命中率

判别逻辑：

- M3 的 gap 仍然存在 → 根因不是 collision，而是 repeated-epoch memory overfitting
- M3 的 gap 消失但 M2 的 gap 存在 → collision 是重要因素
- M3 反而比 M1 差 → collision 可能提供隐式正则化

## References

1. [Huang et al., 2025 — Over-Tokenized Transformer](https://arxiv.org/abs/2501.16975)
2. [Cheng et al., 2026 — Engram](https://arxiv.org/abs/2601.07372)
3. [Yu et al., 2025 — SCONE](https://arxiv.org/abs/2502.01637)
4. [Zhou et al., 2026 — TN-gram](https://arxiv.org/abs/2606.08347)
5. [Chen et al., 2026 — X-gram](https://arxiv.org/abs/2604.21724)
6. [Lin, 2026 — Engram-Nine](https://arxiv.org/abs/2601.16531)
7. [Xie et al., 2025 — Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880)