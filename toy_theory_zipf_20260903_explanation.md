# `toy_theory_zipf_20260903.py` 说明

源文件：

`/Users/harry/Desktop/phys_of_AI/forking/2026_9/tasks/l7_theory_zipf/toy_theory_zipf_20260903.py`

这份脚本只负责生成 token ID 文件。它不修改 `code/train.py`，不生成
bigram/trigram 标签，也不提前生成 embedding。它做的唯一理论性改变是：
把自然语言语料换成连续的、独立同分布（iid）的有限 Zipf token 流。

**函数名校对：当前版本脚本没有 `parse_frequencies` 函数。** 如果在旧笔记、旧
回答或旧的 `.pyc` 文件中看到这个名字，它不属于当前脚本。当前脚本第 66--70
行是 `VOCAB_SIZE`、`SUPPORT_SIZE`、`MODEL_N_LAYER`、`MODEL_N_HEAD` 和
`MODEL_N_EMBD` 五个基本设置；当前脚本真正定义的函数从 `shard_name` 开始，
完整函数列表和精确行号见第 6 节。

## 0. 先核对代码范围

当前源文件实际是 **576 行**，不是 400 多行。其结构是：

| 源代码行 | 内容 |
|---|---|
| 1--40 | shebang 和模块说明；声明无命令行参数、iid finite-Zipf 数据、输出文件，以及不预埋 n-gram 结构 |
| 42--49 | Python future 声明和 `hashlib`、`json`、`math`、`Path`、`numpy` 导入 |
| 52--156 | 日期、输出目录、模型/数据几何、Zipf 参数、分片规格和派生计数常量 |
| 159--572 | 10 个函数：生成分片、计算 oracle、写 metadata/contract、打印结果 |
| 575--576 | `if __name__ == "__main__"` 入口保护 |

因此，下面的说明对应的是当前这份 576 行源文件。此前只把 `generate()` 的前半段讲
清楚，没有把它后面的 metadata/contract 和文件末尾入口展开，是讲解不完整；不是
源代码只有 400 多行。

### 0.1 文件头与导入（第 1--49 行）

- 第 1 行 `#!/usr/bin/env python3`：在 Unix 系统中允许直接用 Python 解释器运行。
- 第 2--40 行模块 docstring：说明脚本只生成 token ID；train/validation 是独立
  的有限 Zipf iid 流；模型以后才从滑窗中感知 bigram/trigram；输出有哪些文件；
  不写 context、label、padding、SEP 或 transition matrix。
- 第 42 行 `from __future__ import annotations`：让类型注解延后求值，不改变数据。
- 第 44--47 行：分别导入文件摘要、JSON、数学检查、路径操作所需的标准库。
- 第 49 行：导入 `numpy`，用于概率、随机抽样、数组和 `.npz` 文件。

## 1. 与原始自然语言实验统一的基本设置

这些是从 `agents.md`、`README.md`、`code/train.py` 和
`docs/experiment-log.md` 核对出的模型/数据几何：

| 项目 | 本 toy 设置 |
|---|---:|
| vocab size | `8192` |
| 模型层数 / attention heads | `8 / 6` |
| 模型 embedding width | `768` |
| sequence length | `2048` |
| device batch size | `72` |
| 每个 device batch 的 loss token 数 | `72 × 2048 = 147,456` |
| train shard | `1` |
| train 每个完整 epoch | `337` 个 device batches |
| validation source pool | `2,3,4,5,6,7,8,9,10,6542`（完整写入） |
| validation pool 总 batch 数 | `9 × 337 + 284 = 3,317` 个 device batches |
| fixed validation eval | `4` 个 device batches（只决定每次评估读多少） |
| validation interval | `10` steps（训练端设置） |
| data mode | `fixed`，按顺序重复 train shard |
| data seed | `42`；validation 使用独立 seed |
| token 文件格式 | little-endian `uint16` token ID |

模型架构、embedding 宽度、optimizer、学习率和 n-gram 开关仍由
`code/train.py`/launcher 负责；本脚本不改它们。原始 tokenizer 的数据准备脚本
会在每行开头放 BOS；本 toy 为了保持连续 iid 流，刻意不插入 BOS/SEP/padding，
但仍保持同样的 vocab、sequence length、batch 和 shard/loader 几何。这是理论分布
差异，不是模型架构差异。

## 2. 完整 validation 数据与 fixed-val 评估必须区分

原始自然语言实验有一个很大的 validation **来源池**（与 train 使用不同的
shard；这里的“不重叠”是数据文件/样本流不重叠，不是 token ID 不能重复）：

```text
train source: shard 1
validation source pool: shards 2,3,4,5,6,7,8,9,10,6542
```

本 toy 实际写入与原始自然语言实验相同分片坐标和规模的完整 validation pool，
但其中的 token 内容由本脚本重新按 Zipf iid 分布抽样。train shard 1 有 337 个
device batches；validation 的 shard 2--10 各有 337 个 device
batches，shard 6542 有 284 个 device batches，共 `3,317` 个 device batches。每个 loader chunk 需要
`2048+1=2049` 个原始 token，因此：

$$
N_{\mathrm{train,raw}}
=337\times72\times(2048+1)
=49{,}716{,}936.
$$

$$
N_{\mathrm{val,pool}}
=9\times49{,}716{,}936
 +284\times72\times2049
=489{,}350{,}376.
$$

因此完整 validation pool 与 train 的比例是：

$$
\frac{N_{\mathrm{val,pool}}}{N_{\mathrm{train,raw}}}
=9.8427299703.
$$

因此，实际生成文件的 train:validation 比例是：

$$
N_{\mathrm{train,raw}}:N_{\mathrm{val,pool}}
=1:9.8427299703.
$$

实际参与 loss 的 token 不包括每个 chunk 最后那个 target-shaping token。完整
validation pool 的 loss token 数为：

$$
N_{\mathrm{train,loss}}
=337\times72\times2048
=49{,}692{,}672,
$$

$$
N_{\mathrm{val,pool,loss}}
=9\times337\times72\times2048
 +284\times72\times2048
=489{,}111{,}552.
$$

当前 `code/train.py` 默认 `val_batches=4`。启动时它从已写入的 validation
shard 2 开始缓存 4 个 batch，之后每次 fixed-val 评估复用这 4 个 batch；这只
是评估子集，不是 validation 文件的总大小：

$$
N_{\mathrm{val,raw}}^{\mathrm{fixed}}
=4\times72\times2049
=590{,}112,
\qquad
N_{\mathrm{val,loss}}^{\mathrm{fixed}}
=4\times72\times2048
=589{,}824.
$$

结论：脚本生成的 validation 文件总量是 `489,350,376`，与原始自然语言实验的
分片几何和规模一致，但 token 内容是重新抽样的 Zipf iid 流；每次默认 fixed-val
评估实际使用 `590,112` 个 raw token。两者不能混为一谈。若把 `val_batches` 改大
（必须在训练 setting 中显式记录），可以从同一完整 pool 读取更多 fixed batches，
不需要重新生成数据。

## 3. 数据的数学定义

脚本生成一个连续 token 流：

$$
x_1,x_2,\ldots,x_N,
\qquad x_t\overset{\mathrm{iid}}{\sim}P(x).
$$

词表和 Zipf 支撑大小都是 `K=8192`。按 rank $r$ 排列后：

$$
P(x_{(r)})
=\frac{r^{-\alpha}}{\sum_{j=1}^{K}j^{-\alpha}},
\qquad
r=1,\ldots,K,
\qquad
\alpha=\frac43.
$$

`alpha=4/3` 是理论设定，不是自然语言实验中从数据估计出的参数。它对应第
8 节理论中 continuation 尾部假设：

$$
p_{(r)}\propto r^{-\alpha}.
$$

由于这里每个 token 都是 iid，任意由模型滑窗得到的有限 context $c$ 都满足：

$$
P(y\mid c)=P(y).
$$

也就是说，context 不携带真实预测信息；它只让模型有机会记住训练集中的有限
采样误差。这个设计正好把“边际分布/支撑导致的频率核”和“模型是否使用 n-gram
表”分开。

模型打开 n-gram 后，才在每个 2049-token loader chunk 内构造滑窗 context：

$$
c_2=(x_{t-1},x_t),
\qquad
c_3=(x_{t-2},x_{t-1},x_t),
\qquad
y=x_{t+1}.
$$

脚本本身不写入这些 context，也不写入 continuation map。chunk 边界处不会把前一个
chunk 的 token 接到下一个 chunk；边界位置按 `train.py` 的 chunk 内索引规则处理。
脚本只生成连续 token ID，不能把这些边界处理误解成数据中预埋了 padding 或 SEP。

## 4. 为什么这能检验第 8 节的因子化

第 8 节的核心形式是：

$$
g_e(f)\approx a_e M(f),
\qquad
G_e\approx a_e Q.
$$

其中：

- $f$ 是 train 中某个 context 的命中次数；
- $M(f)$ 是由训练计数得到的 Good--Turing 缺失质量核；
- $a_e$ 是 fixed replay 完成 $e$ 个 pass 后的模型读出强度；
- $Q$ 是把各频率层按 validation token 质量加总后的固定量。

本 toy 的概率选择给每个 context 相同的 continuation 边际分布：

$$
p_c(y)=P(y).
$$

因此，给定 context 的 train continuation 计数仍然是从同一个 Zipf 分布抽取的
有限样本。理论中的 missing mass 可写成：

$$
P_0(f)=\sum_y p(y)(1-p(y))^f.
$$

脚本额外计算 singleton 代理：

$$
M(f)\approx
\mathbb{E}\left[\frac{N_1(f)}{f}\right]
=\sum_y p(y)(1-p(y))^{f-1}.
$$

如果训练后的 `gap(f)` 能近似等于同一个 $M(f)$ 乘一个随 replay 改变的
幅度，那么就支持“频率形状由数据核决定、epoch 变化由读出幅度决定”的因子化。
这里不预先指定某个 bigram 或 trigram 必须出现，也不直接把 gap 写进数据。

理想 Zipf 尾部下：

$$
M(f)\propto f^{-\beta},
\qquad
\beta=1-\frac1\alpha.
$$

本设定 $\alpha=4/3$，所以理想渐近指数是：

$$
\beta=1-\frac{3}{4}=\frac14.
$$

这是 missing-mass oracle 的理论预测，不是保证模型最终测得的 gap 指数一定
等于 `1/4`。有限词表、context 频率分布、表碰撞、backbone 读出和训练步数都
可能让实测指数偏离它。

## 5. embedding 是怎样处理的

`.bin` 文件中每个元素只是一个 token ID：

```text
uint16，占 2 bytes，取值 0..8191
```

文件中没有 768 维向量。模型读取 token ID 后，在 `code/train.py` 中执行类似：

$$
e_t=\mathrm{wte}[x_t]\in\mathbb{R}^{768}.
$$

因此：

- `8192` 是词表大小；
- `768` 是模型 embedding width；
- `uint16` 是磁盘存储类型；
- 生成器只产生 ID，不初始化或训练 embedding。

当 n-gram-on 时，模型再把查到的 bigram/trigram table 向量加到 input/wte
残差流；当 n-gram-off 时，同一批 ID 直接走普通 transformer。两组实验必须
复用同一目录和同一 `.bin` 文件。

## 6. 生成器每个函数的作用

以下行号以当前脚本为准；可用命令重新查看：

```bash
nl -ba tasks/l7_theory_zipf/toy_theory_zipf_20260903.py
```

### 6.1 常量区（第 52--156 行）

- `GENERATOR_DATE`、`SETTING_REVISION_DATE`：记录脚本和 setting 的版本日期。
- `OUTPUT_DIR`：固定输出目录，避免服务器命令行隐藏参数。
- `VOCAB_SIZE`、`SUPPORT_SIZE`：模型词表和 Zipf 支撑大小，均为 `8192`。
- `MODEL_N_LAYER`、`MODEL_N_HEAD`、`MODEL_N_EMBD`：只写入 metadata，证明 toy
  与主线模型几何一致。
- `SEQUENCE_LEN=2048`、`DEVICE_BATCH_SIZE=72`：决定 loader chunk 和 batch
  几何。
- `ZIPF_ALPHA=4/3`：唯一的理论分布参数。
- `TRAIN_SEED=42`、`VALIDATION_SEED_BASE`：train/validation 使用独立随机流。
- `TOKENS_PER_LOADER_CHUNK=2049`：一个 `(input,target)` 样本需要 2049 个原始 ID。
- `TRAIN_DEVICE_BATCHES_PER_EPOCH=337`：完整 train shard 的 epoch 长度。
- `FIXED_VALIDATION_DEVICE_BATCHES=4`：匹配 `code/train.py` 默认 `val_batches=4`；
  它只表示每次 fixed-val 评估缓存的 batch 数，不表示 validation 文件总量。
- `TOKENS_PER_DEVICE_BATCH=72×2048=147456`：一个 batch 的 loss token 数。
- `VAL_INTERVAL_STEPS=10`：记录主线评估节奏，生成器不执行训练。
- `FULL_SHARD_TOKENS`：计算 train shard 1 的 `49,716,936` 个原始 token。
- `ORIGINAL_VALIDATION_TAIL_DEVICE_BATCHES=284`：shard 6542 的 batch 数。
- `VALIDATION_SHARD_SPECS`：实际写出 shard 2--10（各 337 batches）和 6542
  （284 batches），总计 `489,350,376` raw tokens。
- `VALIDATION_DEVICE_BATCHES_FULL_POOL`：完整 validation pool 的 `3,317` 个
  device batches。
- `TRAIN_LOSS_TOKENS`、`VALIDATION_LOSS_TOKENS`：排除每个 chunk 的最后一个
  shaping token 后，分别得到 `49,692,672` 和 `489,111,552`。
- `FIXED_VALIDATION_RAW_TOKENS`、`FIXED_VALIDATION_LOSS_TOKENS`：默认每次
  fixed-val 评估使用的 `590,112` raw / `589,824` loss tokens。
- `ORIGINAL_VALIDATION_POOL_SHARDS`、`ORIGINAL_VALIDATION_POOL_TOKENS`：完整
  validation pool 的 shard ID 与 token 数；现在实际 materialize，而不是仅记录。
- `CHUNK_TOKENS`：每次最多抽样一百万个 token，避免一次性占用过多内存。
- `OVERWRITE=False`：已有产物时拒绝覆盖。
- `ORACLE_FREQUENCIES`：计算 oracle 的频率网格 `1,2,4,...,2^20`。

### 6.2 `shard_name`（第 159--161 行）

```python
shard_name(1) == "shard_00001.bin"
```

把整数 shard ID 格式化为 `train.py` 期望的五位文件名。命令行不会传 shard
文件名；文件名由这个函数统一生成。

### 6.3 `sha256`（第 164--173 行）

分块读取文件并计算 SHA-256。它不把整个大文件读进内存，最后把每个 shard 的
摘要写入 metadata，方便复制到服务器后核对数据是否改变。

### 6.4 `validate_setting`（第 176--210 行）

在写盘前检查：

1. `uint16` 能表示词表中的所有 ID；
2. support size 不超过 vocab size；
3. `alpha>1` 且为有限数；
4. sequence length、batch 数和 chunk 数为正；
5. train shard 必须是 `1`；
6. validation shard 必须完整覆盖 `2..10,6542`；
7. 每个文件长度必须整除 `2049`，不会在 loader 中产生半个样本。

### 6.5 `zipf_probabilities`（第 213--218 行）

构造长度为 `8192` 的概率数组：

1. `np.arange(1, SUPPORT_SIZE+1)` 生成 rank `1..8192`；
2. `ranks ** (-ZIPF_ALPHA)` 计算 $r^{-4/3}$；
3. 除以总和使概率和为 `1`；
4. 返回 `float64` 数组，供 CDF 抽样使用。

### 6.6 `sample_zipf`（第 221--233 行）

用逆 CDF 方法抽普通 token ID：

1. `rng.random(sample_count)` 产生均匀随机数；
2. `np.searchsorted(cdf, ...)` 找到对应的 Zipf rank；
3. 转成 `uint16`，得到磁盘格式的 token ID。

这里没有 context 循环、Markov 转移矩阵或 label 规则。

### 6.7 `write_iid_shard`（第 236--251 行）

打开一个 `.bin` 文件，循环调用 `sample_zipf`，直到写入精确的
`token_count`。每次最多写 `CHUNK_TOKENS=1,000,000` 个 ID，并用
little-endian `uint16` 写盘。因此 shard 内是连续 iid 流，而不是一组人为绑定的
二元或三元短句。

### 6.8 `loglog_slope`（第 254--269 行）

对正的、有限的 $(x,y)$ 点做 log-log 一次拟合。如果

$$
y\propto x^{-\beta},
$$

拟合出的直线斜率为 $-\beta$，函数返回正的 `beta`。它只用于 oracle 数值摘要，
不把模型训练出的 gap 当作理论输入。

### 6.9 `oracle_curves`（第 272--284 行）

在频率网格上计算两条理论曲线：

$$
P_0(f)=\sum_y p(y)(1-p(y))^f,
$$

和 singleton 代理：

$$
\sum_y p(y)(1-p(y))^{f-1}.
$$

结果写入 `marginal_oracle.npz`，用于后续把实测 context 统计和理论核比较。

### 6.10 `write_json`（第 287--289 行）

用固定缩进、排序后的 key 和结尾换行写 JSON，使 metadata 可读、可比较、可审计。

### 6.11 `shard_record`（第 292--306 行）

为一个 shard 生成记录：ID、文件名、角色、seed、raw token 数、loss token 数、
字节数、loader chunks 和 device batches。这样不打开二进制文件，也能检查 split
和 loader 几何。

### 6.12 `generate`（第 309--543 行）

总编排函数，顺序如下：

1. 调用 `validate_setting`；
2. 创建固定输出目录；
3. 检查是否会覆盖已有文件；
4. 计算 Zipf 概率和 CDF；
5. 用 seed `42` 写 train `shard_00001.bin`；
6. 用独立 validation seeds 写 validation `shard_00002.bin`--`shard_00010.bin`
   和 `shard_06542.bin`；
7. 计算并保存 `marginal_oracle.npz`；
8. 计算所有 shard 的 SHA-256；
9. 写 `metadata.json`、兼容副本 `meta.json` 和 `run_contract.json`。

`generate()` 的后半段不是另一段数据生成逻辑，而是在为刚才写出的数据写审计信息：

- **第 373--380 行**：创建 metadata 的基本字段，包括 schema 版本、实验名、生成器
  文件名、两个日期、输出目录和二进制格式。
- **第 381--387 行 `toy_distribution`**：记录数据确实是有限 Zipf iid；训练和验证
  使用同一边际分布，但使用独立随机数流。
- **第 388--397 行 `model_alignment`**：记录模型对齐信息：8 层、6 个 heads、
  768 维 embedding、8192 词表、2048 序列长度、72 的 device batch，以及磁盘上的
  `uint16`。这里仍然只是记录，不会创建 embedding。
- **第 398--432 行 `split_alignment`**：记录 train/validation 的 shard 列表、
  raw token 数、loss token 数、完整 validation pool 与 train 的比例、fixed-val
  每次使用的 4 个 batch、验证间隔等。这里的数值来自前面常量，不会再次生成数据。
- **第 433--442 行 `shards` 与 `oracle`**：`shards` 保存每个文件的计数和 seed；
  `oracle` 保存 `marginal_oracle.npz` 的文件名、频率网格、渐近/有限样本指数、
  熵以及两条曲线的公式说明。
- **第 443--452 行**：记录 context 频率应由生成后的 train shard 推导，并明确标记
  没有预埋 context、bigram、trigram、block、SEP、padding、transition matrix 或
  context-specific continuation map；同时保存每个 shard 的 SHA-256 摘要。
- **第 454--455 行**：把同一份 metadata 写成 `metadata.json` 和兼容副本 `meta.json`。
- **第 457--491 行 `contract.versioned_setting`**：把 vocab、模型几何、batch、
  fixed-val、Zipf 参数、seed、token 数和比例再次写入可机器读取的实验契约，便于训练
  或复核脚本直接检查。
- **第 492--510 行 `scientific_invariants`**：写入不可变科学约束，例如 iid 流、
  train/validation 随机流独立、两边 token ID 可以重复、context 在生成后统计、
  ngram on/off 共用同一数据、完整 validation pool 已实际写出。
- **第 511--539 行 `model_use`**：记录 `train.py` 应怎样使用这些文件：train 用 shard
  `1`，validation pool 用 `2..10,6542`；默认 fixed-val 从 shard 2 的前缀取 4 个
  batch，而不是把其他 validation 文件删掉。
- **第 540--543 行**：把 shard 哈希写入 `run_contract.json`，然后返回 metadata。

因此，这 235 行主要是“写 provenance/contract”，不是额外的数据结构；真正产生 token
的调用只有第 335--351 行的 `sample_zipf()` 和 `write_iid_shard()`。

### 6.13 `main`（第 546--572 行）

不解析任何命令行参数。第 548--552 行调用 `generate()`，并把文件已存在、setting
错误或系统写盘错误转换为简短的生成失败信息；第 553--567 行打印输出目录、完整
train/validation raw token 数、完整 validation-pool 比例和默认 fixed-val loss-token
比例；第 568--572 行打印词表、embedding width、sequence length 和 alpha，并说明
context frequency 要在下游统计。

第 575--576 行是 Python 入口保护：只有直接执行这个文件时才调用 `main()`；如果把
它作为模块导入，则不会自动生成数据。

## 7. 运行命令

所有参数都在 Python 文件顶部，正式生成命令不带参数：

```bash
cd /Users/harry/Desktop/phys_of_AI/forking/2026_9
```

```bash
python3 tasks/l7_theory_zipf/toy_theory_zipf_20260903.py
```

如果需要改变 vocab、alpha、seed、数据量或输出目录，必须修改并保存这个版本化
Python 文件；不能在服务器命令行临时传参。

## 8. 生成后的文件

新脚本会写入：

```text
tasks/l7_theory_zipf/results/inputs/
└── theory_zipf_iid_mainline_aligned_20260904/
    ├── shard_00001.bin       # train，49,716,936 raw tokens
    ├── shard_00002.bin       # validation，49,716,936 raw tokens
    ├── ...
    ├── shard_00010.bin       # validation，49,716,936 raw tokens
    ├── shard_06542.bin       # validation tail，41,897,952 raw tokens（284 batches）
    ├── marginal_oracle.npz
    ├── metadata.json
    ├── meta.json
    └── run_contract.json
```

`run_contract.json` 中的 `model_use.validation_shards` 和
`materialized_validation_shards` 均为 `2,3,4,5,6,7,8,9,10,6542`；
`fixed_eval_validation_shards` 写为 `2`，表示默认 fixed-val 的 4 个 batch
来自完整 pool 的 shard 2 前缀。完整 pool 已经真实写入这些文件；
`validation_batches=4` 只表示 `train.py` 默认从 pool 开头缓存 4 个 batch。

不会生成 `train_tokens.bin`、`val_tokens.bin`、context label、BOS、SEP、padding
或 transition matrix。现有 launcher 可沿用
`train_shards=1`、`val_shards=2,3,4,5,6,7,8,9,10,6542`；默认
`val_batches=4` 时只从完整 pool 开头取 4 个 batch，若在训练 setting 中把
`val_batches` 提高，仍可直接复用这些已生成的文件。
`train.py` 的 on/off 对比应复用这一目录；只改变模型是否开启 bigram+trigram，不能
为两组重新抽两套数据。

主线 gap 仍按同一 logged step 定义：

$$
\mathrm{gap}(s)
=\mathrm{fixed\_val\_loss}(s)
-\mathrm{online\_train\_loss}(s).
$$
