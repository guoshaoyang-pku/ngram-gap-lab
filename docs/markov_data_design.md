# Markovian Toy Data × ngram-gap-lab: 研究设计

## 1. 问题回顾

主项目研究的问题：
> 给 nanoGPT 加一个 n-gram 查表（bigram + trigram），在固定顺序多轮重放数据时，
> 为什么会产生 train/val gap？gap 的大小由什么因素决定？

核心发现：
> n-gram 信号绕过 attention 的程度 = gap 的大小
> （y > input > v, gap: 1.82 > 0.64 > 0.60）

## 2. 当前方法的局限

目前用的是真实网页文本（climbmix-400b），无法控制：
- 数据的"可记忆程度"（bigram 依赖有多强？）
- 数据中不同 n-gram 的频率分布
- 数据中是否存在长程依赖（超过 bigram/trigram）

我们不知道如果数据**完全由 bigram 决定**，gap 会不会消失？或者反过来，
如果数据**完全不依赖 bigram**，gap 还有没有？

## 3. Markovian Toy Data 的设计思路

### 3.1 数据生成模型（arXiv:2605.01199v1 Def 2.1）

```
转移矩阵: P = λI + (1-λ)1πᵀ
```

每个 token 的生成规则：
- 概率 λ：重复上一个 token
- 概率 1-λ：从平稳分布 π 中重新采样（与上一个 token 无关）

### 3.2 控制参数

| 参数 | 含义 | 研究用途 |
|------|------|----------|
| **λ** | 停留在当前 token 的概率 | **控制 bigram 依赖强度**：λ 越大，数据越由 bigram 决定 |
| **π₁** | 高频 token 的质量 | 控制数据的不均匀程度 |
| **δ** | 低频 token 之间的频率差异 | 模拟"有结构的低频模式"vs"完全随机噪声" |

### 3.3 为什么这个设计适合研究 gap？

**核心洞察**：这个 Markov 链是**一阶**的——下一个 token 只依赖于前一个 token。

这意味着：
- **bigram 信息是最优的**：知道前一个 token 就掌握了全部预测信息
- trigram 不需要——前两个 token 不提供额外信息
- 数据中不存在长程依赖

在这样的数据上，n-gram 表的"记忆能力"可以被精确量化：

```
理论最优预测准确率 = 能从 bigram 中提取的最大信息
                   = f(λ, π)

如果模型的 train/val gap > 理论最优能解释的部分
  → gap 来自过拟合（记忆了不该记的东西）
如果 gap ≈ 理论最优
  → gap 只是数据分布的反映，不是问题
```

## 4. 实验设计

### 实验 1：λ-sweep（核心实验）

固定 `π₁=0.3, δ=0.001, vocab=8192, seq_len=2048`

| λ | 含义 | 预测 |
|---|------|------|
| 0.0 | 完全随机，token 之间独立 | bigram 表毫无用处，三种注入 gap 应该接近 0 |
| 0.3 | 弱 bigram 依赖 | gap 开始出现，y 可能领先 |
| 0.6 | 中等 bigram 依赖 | gap 增大，但三种注入的差距可能不明显 |
| 0.9 | 强 bigram 依赖（接近真实文本） | 三种注入出现明显分层 (y > input > v) |
| 0.99 | 极强 bigram 依赖 | gap 应该最大，因为数据几乎完全可由 bigram 预测 |

**要验证的假设**：
> gap(y) - gap(v) 应该随 λ 单调递增。
> 即：数据越依赖 bigram，绕过 attention 的优势越大。

### 实验 2：π₁-sweep（频率分布的影响）

固定 `λ=0.9, δ=0.001`

| π₁ | 含义 | 预测 |
|----|------|------|
| 0.1 | 高频 token 只占 10% | 数据相对均匀，bigram 模式分散 |
| 0.3 | 高频 token 占 30%（类似真实文本） | 中等偏斜 |
| 0.5 | 高频 token 占 50% | 数据极度偏斜，少数 token 主导 |

**要验证的假设**：
> π₁ 越大，bigram 查表的"碰撞"越少（常用 token 组合更集中），
> gap 应该越小（因为表能更准确地记住高频模式）。

### 实验 3：表大小 ablation（控制碰撞率）

在上面的配置下，额外变化 `bigram_table_size`（当前默认 `vocab_size * 64`）。

| 表大小因子 | 碰撞程度 | 预测 |
|-----------|---------|------|
| ×16 | 大量碰撞 | gap 大（表不够用，频繁覆盖） |
| ×64（默认）| 中等碰撞 | 基准线 |
| ×256 | 很少碰撞 | gap 小（表几乎能记住所有 bigram） |

**要验证的假设**：
> 碰撞越多 → 表越倾向于覆盖旧记忆 → train 上表现差 → gap 缩小（但 train loss 也差）。
> 碰撞越少 → 表越能精确记忆 → train 上表现好 → 同时过拟合风险增大 → gap 可能反而更大。
> 存在一个最优表大小，在 train loss 和 gap 之间取得平衡。

## 5. 和主项目的接口

### 5.1 数据格式兼容

生成的数据与现有 `TokenizedShardDataset` 完全兼容：

```bash
# 生成 Markov 数据
python code/generate_markov_data.py \
    --lambda_val 0.9 --pi_1 0.3 --out_dir data/markov_l0.9

# 直接用现有代码训练（只需要改数据路径）
python code/train.py \
    --data_dir data/markov_l0.9/train \
    --train_shards 0,1,2 \
    --val_shards 0,1,2,3,4,5,6,7 \
    --injection_position y
```

### 5.2 需要改动的地方（TODO）

现有代码的 `train.py` 需要做以下修改才能跑 toy 数据：

1. **vocab_size**：现有默认 8192，toy 数据也是 8192 → 直接兼容
2. **数据目录结构**：现有代码期望 `data_dir/shard_XXXXX.bin`，toy 数据生成的就是这个格式 → 直接兼容
3. **不需要改 `train.py`**，只需要传正确的命令行参数即可

### 5.3 自动化 λ-sweep 脚本（TODO）

```bash
# 批量生成不同 λ 的数据并分别训练
for lambda_val in 0.0 0.3 0.6 0.9 0.99; do
    python code/generate_markov_data.py \
        --lambda_val $lambda_val --out_dir data/markov_l${lambda_val}
    python code/train.py \
        --data_dir data/markov_l${lambda_val}/train \
        --injection_position y --run_id "lambda_${lambda_val}_y"
    python code/train.py \
        --data_dir data/markov_l${lambda_val}/train \
        --injection_position input --run_id "lambda_${lambda_val}_input"
done
```

## 6. 理论预期（和 arXiv 论文的对话）

这篇论文的核心发现是：attention 在 Markov 数据上经历"聚焦-稀释"循环，
交替学习高频模式和均匀化注意力权重。

我们的实验提供了**互补视角**：
- 论文：attention 自己怎么学 Markov 结构
- 我们：如果绕过 attention（用 n-gram 表直接记忆），gap 如何变化

结合两者可以回答：
> 在真实训练中，attention 的"理解"和 n-gram 的"记忆"各自贡献了多少 gap？
> 通过 λ-sweep 分离两者的贡献。

如果实验发现 gap 主要来自 n-gram 记忆（而非 attention 的问题），
那意味着减少 train/val gap 的方向应该是**抑制 n-gram 表的记忆能力**，
而不是改进 attention 机制。

## 7. 最小可行实验（MVP）

如果只想先跑一个实验验证思路，建议：

```bash
# 1. 生成 3 组 λ 的数据
python code/generate_markov_data.py --lambda_val 0.3 --num_seqs_per_shard 1000 --out_dir data/markov_l0.3
python code/generate_markov_data.py --lambda_val 0.9 --num_seqs_per_shard 1000 --out_dir data/markov_l0.9
python code/generate_markov_data.py --lambda_val 0.99 --num_seqs_per_shard 1000 --out_dir data/markov_l0.99

# 2. 在每组数据上分别跑 3 种注入位置（共 9 个实验）
# 3. 画出 gap vs λ 的曲线，看是否单调递增
# 4. 如果单调递增 → 验证了核心假设 → 写 paper
```
