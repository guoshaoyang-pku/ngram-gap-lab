# `train.py` 中 bigram 操作的逐行代码讲解

这篇文档面向没有学过 Python、PyTorch 的读者。目标不是只描述 bigram 的概念，而是沿着程序真正的执行顺序，解释代码中的每一段语法、每个张量的形状，以及一次 bigram 查表和更新到底发生了什么。

本文只讨论当前实验配置：

```text
enable_bigram_ve = True
enable_unigram_ve = False
enable_trigram_ve = False
injection_position = "input"
n_layer = 8
n_embd = 768
vocab_size = 8192
```

对应源代码是 [`code/train.py`](code/train.py)。行号以当前文件为准。

---

## 0. 先给出最终结论

这份代码没有把两个 token 真正合并成一个新的 tokenizer token，也没有缩短序列。

它做的是：

```text
前一个 token ID + 当前 token ID
              ↓
          组成 bigram
              ↓
       哈希成两个表格行号
              ↓
       从两张表各取一个向量
              ↓
       384 维和 384 维拼接
              ↓
          得到 768 维向量
              ↓
  加到“当前 token”的输入表示上
              ↓
       Transformer 预测下一个 token
```

例如，模型使用 `(喜欢, 吃)` 的 bigram 向量增强“吃”的表示，再用它预测“苹果”。

用数学符号写就是：

\[
x_t^{(0)} = E_{\rm token}(w_t) + E_{\rm position}(t) + E_{\rm bigram}(w_{t-1},w_t).
\]

这里：

- `w_t` 是当前位置的 token；
- `w_{t-1}` 是前一个 token；
- `E_token` 是普通 token embedding；
- `E_position` 是位置 embedding；
- `E_bigram` 是这份代码额外查出来的 bigram 向量；
- `x_t^(0)` 是送入第一层 Transformer 的输入。

---

## 1. 阅读代码前需要知道的最少 Python 语法

### 1.1 变量赋值

```python
x = 3
```

意思是把数字 `3` 保存到名字 `x` 中。等号在这里不是数学上的“判断相等”，而是“把右边的结果交给左边的名字”。

### 1.2 函数调用

```python
f(x)
```

意思是调用函数 `f`，把 `x` 作为输入。

在 PyTorch 中，一个 embedding 表也可以像函数一样调用：

```python
vector = table[row_number]
```

代码实际写成：

```python
vector = table(row_number)
```

两者都可以理解成“从 `table` 的指定行取向量”。

### 1.3 `if`

```python
if condition:
    do_something()
```

意思是：只有 `condition` 为真时，才执行缩进的代码。

### 1.4 `for`

```python
for li in [1, 3, 5, 7]:
    do_something(li)
```

意思是依次令：

```text
li = 1
li = 3
li = 5
li = 7
```

并分别执行一次缩进部分。

### 1.5 列表、集合和字典

```python
[1, 3, 5, 7]       # 列表：有顺序
{1, 3, 5, 7}       # 集合：保存成员，不强调顺序
{"1": table_1}     # 字典：用 key 找 value
```

代码里的 `self.bigram_ves["1"]` 可以理解成：“取出名字为 `1` 的那套 bigram 表”。

### 1.6 `self`

`self` 表示“当前这个模型对象自己”。例如：

```python
self.bigram_table_size = 524288
```

意思是把 `bigram_table_size` 作为模型自身的一项长期属性保存下来，其他函数之后仍然可以访问它。

### 1.7 张量和形状

PyTorch 的 `Tensor` 可以理解成多维数字表。

输入 token 的形状写作：

```text
(B, T)
```

其中：

- `B` 是 batch size，一次并行处理多少条序列；
- `T` 是每条序列有多少个 token。

例如两条长度为 4 的序列：

```text
idx = [[ 1, 10, 20, 30],
       [ 1, 51, 62, 73]]
```

它的形状是 `(2, 4)`，即 `B=2, T=4`。

embedding 查表以后，每个 token 都会多出一个长度为 `N` 的向量，所以形状从 `(B,T)` 变成：

```text
(B, T, N)
```

当前 `N = n_embd = 768`。

### 1.8 下标和切片

```python
idx[:, :1]
```

方括号表示从张量中取一部分。逗号前后分别控制两个维度：

```text
idx[第一个维度, 第二个维度]
```

第一个 `:` 表示“batch 维全部保留”；第二个 `:1` 表示“序列维只取下标 1 之前的元素”，也就是第一个 token。

```python
idx[:, :-1]
```

第一个 `:` 仍然表示保留所有 batch；`:-1` 表示从开头取到倒数第一个元素之前，也就是去掉最后一个 token。

### 1.9 下标从 0 开始

Python 的第一个元素编号是 `0`，不是 `1`。

8 个 Transformer block 在代码中编号为：

```text
0, 1, 2, 3, 4, 5, 6, 7
```

所以代码中的 `layer 1,3,5,7`，如果用日常“第一层、第二层”的说法，对应的是第 2、4、6、8 层。

---

## 2. 配置：哪些 n-gram 被打开

源代码第 50-80 行定义配置：

```python
@dataclass
class Config:
    vocab_size: int = 8192
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 768
    sequence_len: int = 2048
    dropout: float = 0.0
    bias: bool = True
    enable_nanogpt_ngram_ve: bool = True
    enable_unigram_ve: bool = False
    enable_bigram_ve: bool = True
    enable_trigram_ve: bool = False
    enable_fourgram_ve: bool = False
    nanogpt_ngram_injection_position: str = "input"
```

逐行解释：

```python
@dataclass
```

这是 Python 的一个辅助标记，让 `Config` 更方便保存一组配置。它不执行 bigram 运算。

```python
class Config:
```

定义一种名为 `Config` 的配置对象。

```python
vocab_size: int = 8192
```

tokenizer 一共有 8192 个普通 token ID，合法编号通常是 `0` 到 `8191`。`: int` 表示希望这个配置是整数。

```python
n_layer: int = 8
```

Transformer 有 8 个 block。

```python
n_embd: int = 768
```

模型宽度是 768。实验里如果把 `N` 定义为模型宽度，那么这里的 `N` 就是 `n_embd`。每个普通 token 和每个最终 bigram 都用 768 个数表示。

```python
enable_nanogpt_ngram_ve: bool = True
```

打开整个 n-gram value embedding 系统。`bool` 表示布尔值，只能取 `True` 或 `False`。

```python
enable_unigram_ve: bool = False
enable_bigram_ve: bool = True
enable_trigram_ve: bool = False
```

只打开 bigram，关闭 unigram 和 trigram。

```python
nanogpt_ngram_injection_position: str = "input"
```

`str` 表示字符串。值为 `"input"` 表示把 n-gram 向量加在 Transformer 的最初输入上。

命令行参数第 701-702、717-723 行允许运行时覆盖这些默认值：

```python
parser.add_argument("--injection_position", default="input",
                    choices=["v", "y", "input"])
parser.add_argument("--enable_bigram", type=int, default=1)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--n_embd", type=int, default=768)
parser.add_argument("--vocab_size", type=int, default=8192)
```

例如命令行中的：

```bash
--enable_bigram 1 --injection_position input --n_embd 768
```

分别表示打开 bigram、在输入注入、模型宽度设为 768。

---

## 3. 为什么代码选择编号 1、3、5、7

源代码第 91-93 行：

```python
def has_ve(layer_idx: int, n_layer: int) -> bool:
    """Alternating VE layers (matches OPHIS convention)."""
    return layer_idx % 2 == (n_layer - 1) % 2
```

逐行解释：

```python
def has_ve(...):
```

`def` 表示定义函数。这个函数接收当前层编号 `layer_idx` 和总层数 `n_layer`，返回该编号是否被选中。

```python
-> bool
```

表示函数返回 `True` 或 `False`。

```python
%
```

百分号在这里是“取余数”。例如：

```text
7 % 2 = 1
6 % 2 = 0
```

当前 `n_layer=8`：

```text
(n_layer - 1) % 2
= (8 - 1) % 2
= 7 % 2
= 1
```

所以返回条件变成：

```python
layer_idx % 2 == 1
```

编号 `1,3,5,7` 除以 2 的余数都是 1，因此被选中。

源代码第 297 行：

```python
ngram_layers = sorted(i for i in range(config.n_layer)
                      if has_ve(i, config.n_layer))
```

可以把这一行展开成更容易读的伪代码：

```text
先建立空列表 ngram_layers
依次尝试 i = 0,1,2,3,4,5,6,7
如果 has_ve(i,8) 为真，就把 i 放进列表
最后把列表排序
```

结果是：

```python
ngram_layers = [1, 3, 5, 7]
```

这种隔层选择是代码沿用的 OPHIS 实验约定，不是 bigram 数学上必须只放在这些层。它减少表的套数、参数量和计算量。

还要特别注意：当前是 `input` 模式。这些编号主要用于创建四套不同的表；四套表的输出会在模型输入处先求和。当前模式不是运行到第 1、3、5、7 号 block 时才分别注入。

---

## 4. 哈希常数是什么

源代码第 102-107 行：

```python
_BASE_BIGRAM_PRIMES = [
    [(2654435761, 2246822519), (1013904223, 6291469)],
    [(374761393, 668265263), (3266489917, 104729)],
    [(1640531527, 97531), (48271, 40503)],
    [(16777619, 2166136261), (3432918353, 461845907)],
]
```

外层列表有 4 项，对应四套 bigram 表。每一项里面又有两对数字，对应两次哈希。

映射关系是：

```text
编号 1 的表：第 0 组常数
编号 3 的表：第 1 组常数
编号 5 的表：第 2 组常数
编号 7 的表：第 3 组常数
```

例如编号 1 的表使用：

```text
第一次哈希：(2654435761, 2246822519)
第二次哈希：(1013904223, 6291469)
```

这些大数不是训练出来的参数，只是固定的哈希常数。它们的作用是让不同 token 对尽量散落到不同表行，并让两次哈希的结果彼此不同。

第 123-138 行的 `expand_bigram_hash_primes` 用于层数更多、所需常数组数超过 4 时自动生成额外常数。当前只需要 4 组，因此第 124-125 行直接执行：

```python
if count <= len(base):
    return base[:count]
```

这里 `count=4`，`len(base)=4`，所以返回已有的四组常数；后面的扩展逻辑不会运行。

---

## 5. 模型初始化时如何建立 bigram 表

源代码第 303-317 行：

```python
self.bigram_ve_layers = (
    set(ngram_layers) if config.enable_nanogpt_ngram_ve and config.enable_bigram_ve else set()
)
self.bigram_table_size = config.vocab_size * 64
self.bigram_K = 2
half_dim = config.n_embd // 2
_bp = expand_bigram_hash_primes(_BASE_BIGRAM_PRIMES, len(ngram_layers))
self.bigram_hash_primes_per_layer = {}
self.bigram_ves = nn.ModuleDict()
for j, li in enumerate(sorted(self.bigram_ve_layers)):
    self.bigram_ves[str(li)] = nn.ModuleList([
        nn.Embedding(self.bigram_table_size, half_dim),
        nn.Embedding(self.bigram_table_size, config.n_embd - half_dim),
    ])
    self.bigram_hash_primes_per_layer[li] = _bp[j]
```

下面逐行拆解。

### 第 303-305 行：决定是否建立表

```python
self.bigram_ve_layers = (
    set(ngram_layers) if condition else set()
)
```

这是 Python 的简写，意思是：

```text
如果总开关和 bigram 开关都打开：
    bigram_ve_layers = {1,3,5,7}
否则：
    bigram_ve_layers = 空集合
```

`and` 表示两个条件都必须为真。

### 第 306 行：每张表有多少行

```python
self.bigram_table_size = config.vocab_size * 64
```

代入当前值：

```text
8192 × 64 = 524288
```

因此每张 bigram 子表有 524288 行。

为什么不直接给每一种 bigram 一行？因为可能的 token 对数量是：

```text
8192 × 8192 = 67,108,864
```

直接建立完整表会更大。因此代码建立较小的 524288 行表，并用哈希把大量可能的 bigram 映射进去。代价是不同 bigram 可能落到同一行，这叫哈希碰撞。

### 第 307 行：为什么是两张子表

```python
self.bigram_K = 2
```

`K=2` 表示每套 bigram 表实际包含两个子表，也会做两次哈希。

### 第 308 行：把 768 维切成两半

```python
half_dim = config.n_embd // 2
```

`//` 表示整数除法。当前：

```text
768 // 2 = 384
```

### 第 309 行：取得四组哈希常数

```python
_bp = expand_bigram_hash_primes(_BASE_BIGRAM_PRIMES, len(ngram_layers))
```

`len(ngram_layers)` 是列表长度，当前为 4。函数返回前面介绍的四组常数。

### 第 310-311 行：建立两个容器

```python
self.bigram_hash_primes_per_layer = {}
self.bigram_ves = nn.ModuleDict()
```

第一行建立普通字典，保存“每套表对应哪些哈希常数”。

第二行建立 PyTorch 的模块字典，保存真正可训练的 embedding 表。使用 `ModuleDict` 很重要，因为 PyTorch 会自动把其中的权重登记为模型参数，反向传播和优化器才能找到它们。

### 第 312 行：依次建立四套表

```python
for j, li in enumerate(sorted(self.bigram_ve_layers)):
```

`sorted(...)` 把集合排成 `[1,3,5,7]`。

`enumerate(...)` 同时给出“第几次循环”和“实际层编号”：

```text
j=0, li=1
j=1, li=3
j=2, li=5
j=3, li=7
```

### 第 313-316 行：每套表包含两个 embedding

```python
self.bigram_ves[str(li)] = nn.ModuleList([
    nn.Embedding(self.bigram_table_size, half_dim),
    nn.Embedding(self.bigram_table_size, config.n_embd - half_dim),
])
```

`str(li)` 把整数层号变成字符串，例如整数 `1` 变成字符串 `"1"`，因为 `ModuleDict` 的 key 使用字符串。

`nn.ModuleList` 是 PyTorch 管理一组子模块的列表。

`nn.Embedding(A,B)` 可以理解成一个形状为 `(A,B)` 的可学习数字表：

```text
第一张子表：nn.Embedding(524288,384)
第二张子表：nn.Embedding(524288,384)
```

因此每套表的结构是：

```text
bigram_ves[层号][0]：524288 行 × 384 列
bigram_ves[层号][1]：524288 行 × 384 列
```

### 第 317 行：保存该套表的哈希常数

```python
self.bigram_hash_primes_per_layer[li] = _bp[j]
```

例如第一次循环 `j=0, li=1`，所以把 `_bp[0]` 的两对常数交给编号 1 的表。

### 参数量提醒

一套表包含的数值个数是：

```text
524288 × 384 × 2 = 402,653,184
```

四套表总共是：

```text
402,653,184 × 4 = 1,610,612,736
```

这说明当前 bigram 表本身非常大。这里的“只选隔层表”会把表套数控制在 4；如果 8 层每层都建同样的表，表参数还会翻倍。

---

## 6. bigram 表如何随机初始化

源代码第 337-355 行：

```python
@torch.no_grad()
def init_weights(self):
    ...
    s = 3 ** 0.5 * self.config.n_embd ** -0.5
    ...
    for lvs in self.bigram_ves.values():
        for bve in lvs:
            torch.nn.init.uniform_(bve.weight, -s, s)
```

逐行解释：

```python
@torch.no_grad()
```

表示初始化权重时不记录梯度，因为初始化不是训练。

```python
s = 3 ** 0.5 * self.config.n_embd ** -0.5
```

`**` 表示乘方。这个公式等价于：

\[
s=\sqrt{3}/\sqrt{N}.
\]

当前 `N=768`，所以 `s=0.0625`。

```python
for lvs in self.bigram_ves.values():
```

依次取出四套表。

```python
for bve in lvs:
```

依次取出一套表里的两个子表。

```python
torch.nn.init.uniform_(bve.weight, -s, s)
```

把子表里的每个数字随机初始化在：

```text
[-0.0625, 0.0625]
```

之间。训练开始前，这些表还没有学到任何 bigram 含义。

---

## 7. 数据如何生成输入和预测答案

源代码第 565-613 行定义数据集。关键部分是：

```python
self.chunk_size = sequence_len + 1
...
chunk = np.array(buf[start:start + self.chunk_size], dtype=np.int64)
inp = torch.from_numpy(chunk[:-1])
tgt = torch.from_numpy(chunk[1:])
yield inp, tgt
```

### 为什么读取 `sequence_len + 1`

如果模型输入长度 `T=4`，需要 4 个输入 token 和它们各自的下一个 token 答案，所以原始块需要 5 个 token。

假设：

```text
chunk = [BOS, 我, 喜欢, 吃, 苹果]
```

### `chunk[:-1]`

`:-1` 表示去掉最后一个元素：

```text
inp = [BOS, 我, 喜欢, 吃]
```

### `chunk[1:]`

`1:` 表示去掉第一个元素，从下标 1 一直取到末尾：

```text
tgt = [我, 喜欢, 吃, 苹果]
```

把两者对齐：

```text
inp = [BOS, 我,   喜欢, 吃  ]
tgt = [我,  喜欢, 吃,   苹果]
```

所以每个输入位置的任务都是预测右边的下一个 token。

第 615-625 行把多条序列堆叠成 batch：

```python
yield (torch.stack(batch_inp).to(device),
       torch.stack(batch_tgt).to(device))
```

`torch.stack` 把许多形状为 `(T,)` 的单条序列叠成 `(B,T)`。

`.to(device)` 把数据移动到 CPU 或 GPU 上。

第 626-627 行：

```python
self._epoch += 1
# fixed mode: no shuffle, deterministic replay
```

`+= 1` 表示 epoch 计数加一。固定模式不打乱数据，所以第二个 epoch 会按相同顺序再次看到同一批 bigram。

---

## 8. 核心函数：逐行生成 input bigram residual

真正执行当前 `input` 注入的代码是第 364-394 行：

```python
def _compute_input_ngram_residual(self, idx):
    """Over-encoding: sum all enabled layers' n-gram values, no gate."""
    _B, T = idx.size()
    residual = None
    prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
    prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
    if self.config.enable_unigram_ve:
        ...
    if self.config.enable_bigram_ve:
        for li in sorted(self.bigram_ve_layers):
            lvs = self.bigram_ves[str(li)]
            primes = self.bigram_hash_primes_per_layer[li]
            idxs = [((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size
                    for p1, p2 in primes]
            bgve = torch.cat([lvs[k](idxs[k]) for k in range(self.bigram_K)], dim=-1)
            residual = bgve if residual is None else residual + bgve
    if self.config.enable_trigram_ve:
        ...
    if residual is None:
        residual = torch.zeros(...)
    return residual
```

下面逐行代入一个具体例子。

### 第 364 行：函数入口

```python
def _compute_input_ngram_residual(self, idx):
```

定义一个函数，输入 `idx` 是 token ID 张量，输出与 token embedding 相加的 n-gram residual。

函数名前面的下划线 `_` 是 Python 程序员常用约定，表示这是模型内部使用的辅助函数；它不会改变函数行为。

假设 batch 里只有一句话：

```text
idx = [[1, 10, 20, 30]]
       [BOS, 我, 喜欢, 吃]
```

形状是 `(B,T)=(1,4)`。

### 第 366 行：取得 batch 和序列长度

```python
_B, T = idx.size()
```

`idx.size()` 返回 `(1,4)`，所以：

```text
_B = 1
T  = 4
```

变量 `_B` 前的下划线表示后面不再使用这个值。

### 第 367 行：准备累加器

```python
residual = None
```

`None` 表示“现在还没有值”。后面第一次得到 bigram 向量时，直接把它放进 `residual`；之后再把其他表的向量加上去。

### 第 368 行：生成前一个 token

```python
prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
```

先分别计算：

```text
idx[:, :1]  = [[1]]
idx[:, :-1] = [[1, 10, 20]]
```

`torch.cat([...], dim=1)` 表示沿着第 1 号维度，也就是序列方向，把两段连接：

```text
prev_idx = [[1, 1, 10, 20]]
            [BOS, BOS, 我, 喜欢]
```

与原输入对齐：

```text
prev_idx = [BOS, BOS, 我,   喜欢]
idx      = [BOS, 我,   喜欢, 吃  ]
```

于是每一列表示一个 bigram：

```text
位置0：(BOS, BOS)
位置1：(BOS, 我)
位置2：(我, 喜欢)
位置3：(喜欢, 吃)
```

第一位本来没有前一个 token，因此代码复制第一位自己。数据以 `BOS` 对齐时，这一对就是 `(BOS,BOS)`。

### 第 369 行：生成前两个 token

```python
prev2_idx = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
```

这行是为 trigram 准备的。当前 `enable_trigram_ve=False`，所以虽然它被算出来，后面不会参与结果。理解当前 bigram 时可以先忽略它。

### 第 370-373 行：unigram 分支

```python
if self.config.enable_unigram_ve:
    ...
```

当前 unigram 开关是 `False`，所以整个缩进块跳过。

### 第 374 行：进入 bigram 分支

```python
if self.config.enable_bigram_ve:
```

当前开关是 `True`，所以执行下面的循环。

### 第 375 行：遍历四套表

```python
for li in sorted(self.bigram_ve_layers):
```

依次令：

```text
li=1, li=3, li=5, li=7
```

以下第 376-380 行会执行四次。

### 第 376 行：取当前这一套的两个子表

```python
lvs = self.bigram_ves[str(li)]
```

第一次循环 `li=1`，因此相当于：

```text
lvs = bigram_ves["1"]
```

其中：

```text
lvs[0] 是 524288 × 384 的第一张子表
lvs[1] 是 524288 × 384 的第二张子表
```

### 第 377 行：取当前表的两组哈希常数

```python
primes = self.bigram_hash_primes_per_layer[li]
```

当 `li=1` 时：

```text
primes = [(2654435761,2246822519),
          (1013904223,6291469)]
```

### 第 378 行：计算两个查表行号

```python
idxs = [
    ((prev_idx * p1) ^ (idx * p2)) % self.bigram_table_size
    for p1, p2 in primes
]
```

这是“列表推导式”，可以展开为：

```text
建立空列表 idxs
对 primes 中的每一对 (p1,p2)：
    计算 ((prev_idx × p1) XOR (idx × p2)) % 524288
    把结果加入 idxs
```

这里的乘法、异或和取余都会对 `(B,T)` 张量中的每一个位置分别执行。因此输出的每组行号仍然是 `(B,T)`。

`^` 不是乘方。在 Python 整数和整数张量中，它是按位异或 `XOR`。

对于位置 3 的 `(喜欢=20, 吃=30)`，编号 1 表的第一次哈希是：

```text
20 × 2654435761 = 53088715220
30 × 2246822519 = 67404675570
两者按位 XOR      = 16877107238
16877107238 % 524288 = 276518
```

第二次哈希是：

```text
20 × 1013904223 = 20278084460
30 × 6291469    = 188744070
两者按位 XOR     = 20198392554
20198392554 % 524288 = 197354
```

所以对这一位置：

```text
idxs[0] = 276518
idxs[1] = 197354
```

完整 batch 中，`idxs[0]` 和 `idxs[1]` 都是形状 `(B,T)` 的行号表，而不是单个数字。

### 第 379 行：查两张表并拼接

```python
bgve = torch.cat(
    [lvs[k](idxs[k]) for k in range(self.bigram_K)],
    dim=-1
)
```

先看：

```python
range(self.bigram_K)
```

因为 `K=2`，它产生 `k=0` 和 `k=1`。

于是列表推导式等价于：

```python
[lvs[0](idxs[0]), lvs[1](idxs[1])]
```

对于 `(喜欢,吃)`：

```text
lvs[0](276518) -> 从第一张子表第 276518 行取 384 维向量
lvs[1](197354) -> 从第二张子表第 197354 行取 384 维向量
```

对整个 batch 查表以后，两者形状分别是：

```text
(B,T,384)
(B,T,384)
```

`torch.cat(..., dim=-1)` 中的 `-1` 表示最后一个维度，也就是向量维度。拼接后：

```text
(B,T,384) + (B,T,384) -> (B,T,768)
```

注意：这里确实发生了“拼接”，但拼接的是两次哈希查出的两个半向量，不是把输入序列里的两个 token 合并成一个位置。

### 第 380 行：四套表之间求和

```python
residual = bgve if residual is None else residual + bgve
```

这是 Python 条件表达式，可以展开成：

```text
如果 residual 目前还是 None：
    residual = bgve
否则：
    residual = residual + bgve
```

四次循环的结果是：

```text
第一次 li=1：residual = bgve_1
第二次 li=3：residual = bgve_1 + bgve_3
第三次 li=5：residual = bgve_1 + bgve_3 + bgve_5
第四次 li=7：residual = bgve_1 + bgve_3 + bgve_5 + bgve_7
```

这里是逐元素相加，不是继续拼接。因此最终形状仍然是 `(B,T,768)`，不是 `(B,T,3072)`。

对于任意 bigram `(a,b)`，最终向量可以写成：

\[
E_{\rm bigram}(a,b)=e_1(a,b)+e_3(a,b)+e_5(a,b)+e_7(a,b).
\]

### 第 381-390 行：trigram 分支

当前 trigram 开关为 `False`，所以跳过。

### 第 391-393 行：没有 n-gram 时的备用结果

```python
if residual is None:
    residual = torch.zeros(...)
```

如果 unigram、bigram、trigram 全部关闭，`residual` 会一直是 `None`。这时建立一个全零张量，保证加到输入上也不会改变输入。

当前 bigram 开启，因此通常不会进入这个分支。

### 第 394 行：返回结果

```python
return residual
```

把形状为 `(B,T,768)` 的最终 n-gram residual 返回给主 `forward` 函数。

---

## 9. `forward` 中如何真正注入模型输入

源代码第 396-403 行：

```python
def forward(self, idx, targets=None):
    B, T = idx.size()
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
    x = self.transformer.wte(idx)
    x = x + self.transformer.wpe(pos)
    if self.config.nanogpt_ngram_injection_position == "input":
        x = x + self._compute_input_ngram_residual(idx)
    x = self.transformer.drop(x)
```

### 第 396 行：模型前向计算入口

```python
def forward(self, idx, targets=None):
```

`idx` 是输入 token ID，`targets` 是正确答案。`targets=None` 表示生成文本时可以不提供答案。

### 第 397 行：取得形状

```python
B, T = idx.size()
```

例如 `idx.shape=(72,2048)` 时：

```text
B=72
T=2048
```

### 第 398 行：建立位置编号

```python
pos = torch.arange(0, T, ...).unsqueeze(0)
```

`torch.arange(0,T)` 产生：

```text
[0,1,2,...,T-1]
```

`.unsqueeze(0)` 在最前面增加一个 batch 维，形状从 `(T,)` 变成 `(1,T)`。

### 第 399 行：普通 token embedding

```python
x = self.transformer.wte(idx)
```

`wte` 是普通词表 embedding。它把每个整数 token ID 查成一个 768 维向量：

```text
idx: (B,T)
x:   (B,T,768)
```

### 第 400 行：加位置 embedding

```python
x = x + self.transformer.wpe(pos)
```

`wpe` 根据位置 `0,1,2,...` 查出位置向量，逐位置加到 token 向量上。

### 第 401-402 行：注入 bigram

```python
if self.config.nanogpt_ngram_injection_position == "input":
    x = x + self._compute_input_ngram_residual(idx)
```

`==` 表示判断左右是否相等。当前配置确实是字符串 `"input"`，所以执行第 402 行。

对于位置 `t`：

```text
x[t]
= 普通 token 向量
+ 位置向量
+ bigram(prev_token, current_token) 向量
```

具体到示例的“吃”：

```text
x_吃
= wte(吃)
+ wpe(位置3)
+ bigram_residual(喜欢,吃)
```

然后这个 `x` 才被送进第一个 Transformer block。

### 第 403 行：dropout

```python
x = self.transformer.drop(x)
```

训练时可以随机屏蔽部分数值用于正则化。但当前 `dropout=0.0`，所以它不改变 `x`。

---

## 10. 为什么后面又计算了一遍 bigram

源代码第 404-434 行又出现：

```python
prev_idx = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
...
bigram_indices = {}
if self.config.enable_bigram_ve:
    for li in self.bigram_ve_layers:
        ...
        bigram_indices[li] = [...]
...
for i, block in enumerate(self.transformer.h):
    ...
    if self.config.enable_bigram_ve and i in self.bigram_ve_layers:
        ...
        bgve = torch.cat(...)
    x = block(x, ve=ve, bigram_ve=bgve, trigram_ve=tgve)
```

这段代码是为了同时支持另外两种注入位置 `v` 和 `y`。它为每个 block 准备 bigram 向量，并通过参数 `bigram_ve=bgve` 传进去。

但是 attention 的代码第 233-250 行只在以下条件使用它：

```python
if self.ngram_injection_position == "v":
    ...使用 bigram_ve...

if self.ngram_injection_position == "y":
    ...使用 bigram_ve...
```

当前值是 `"input"`，既不等于 `"v"`，也不等于 `"y"`。因此：

```text
第 404-434 行确实又查了一遍 bigram 表，
但这些第二遍查出的向量没有参与 loss，
也不会从这一条路径产生梯度。
```

当前 `input` 实验真正有效的路径只有：

```python
x = x + self._compute_input_ngram_residual(idx)
```

后面的第二遍查表是冗余计算，可以理解为同一个 `forward` 为 `v/y/input` 三种实验共用代码而留下的通用路径。理解当前实验时不要误以为 bigram 又在编号 1、3、5、7 的 block 内注入了一次。

---

## 11. Transformer 如何用它预测下一个 token

第 421-436 行把 `x` 依次送过 8 个 block：

```python
for i, block in enumerate(self.transformer.h):
    ...
    x = block(x, ...)
x = self.transformer.ln_f(x)
logits = self.lm_head(x)
```

`for` 循环让 `i` 依次为 `0` 到 `7`。每次把上一层输出交给下一层。

`ln_f` 是最后的归一化。

`lm_head` 把每个位置的 768 维表示变成 8192 个分数：

```text
logits.shape = (B,T,8192)
```

每个位置的 8192 个分数对应“下一个 token 是词表中每个 token”的倾向。

对于输入位置“吃”，模型使用包含 `(喜欢,吃)` bigram 信息的表示，输出对下一个 token 的分数。正确答案是“苹果”。

---

## 12. loss 如何计算

源代码第 437-441 行：

```python
if targets is not None:
    loss = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        targets.view(-1), ignore_index=-1, reduction="mean")
    return loss
```

### `if targets is not None`

如果训练时提供了正确答案，就计算 loss。

### `logits.float()`

把预测分数转换为常用的 32 位浮点数，增加 loss 计算的数值稳定性。

### `.view(-1, logits.size(-1))`

`logits.size(-1)` 是最后一维大小，即词表大小 8192。

`.view(-1,8192)` 把 batch 维和序列维摊平：

```text
(B,T,8192) -> (B×T,8192)
```

这里的 `-1` 表示让 PyTorch 根据总元素数自动推断这一维。

### `targets.view(-1)`

把答案从 `(B,T)` 摊平成 `(B×T,)`。

### `F.cross_entropy`

交叉熵比较每个位置的 8192 个预测分数和正确 token ID。正确 token 概率越低，loss 越大。

### `reduction="mean"`

把所有有效位置的 loss 求平均，最终返回一个标量。

---

## 13. 一次训练 step 如何更新 bigram 表

训练循环在第 794-812 行：

```python
for step in range(cfg.max_steps):
    optimizer.zero_grad()
    accum_loss = 0.0
    for _ in range(grad_accum):
        inp, tgt = next(train_iter)
        loss = model(inp, targets=tgt) / grad_accum
        loss.backward()
        accum_loss += loss.item()
    optimizer.step(lr_mult=lr_mult)
```

### `for step in range(cfg.max_steps)`

重复执行训练 step。`step` 从 0 开始计数。

### `optimizer.zero_grad()`

清除上一个训练 step 留下的梯度。PyTorch 默认会累加梯度，如果不清除，新旧 step 会错误混在一起。

### `inp, tgt = next(train_iter)`

从数据加载器取得下一个 batch 的输入和答案。

### `loss = model(inp, targets=tgt)`

执行前面完整的前向过程：

```text
构造 bigram
→ 哈希
→ 查表
→ 加到输入
→ Transformer
→ 预测
→ 计算交叉熵
```

### `loss.backward()`

反向传播。PyTorch 沿着参与 loss 的所有计算反向求导。

因为 input bigram 向量参与了：

```python
x = x + bigram_residual
```

所以梯度能一路返回到 bigram 表中被查到的那些行。

如果同一个 batch 内 `(喜欢,吃)` 出现多次，它们会查到相同的行。这些位置产生的梯度会在该行上相加。

### `optimizer.step(...)`

根据刚才的梯度真正修改参数。

因此“每一步更新”的精确定义是：

```text
一个训练 step 中先处理完整 batch，
所有 token 的梯度先累积，
然后 optimizer.step() 统一更新一次。
```

不是每读取一个 token 就立即改一次表。

当前默认配置：

```python
grad_accum = total_batch_size // (device_batch_size * sequence_len)
```

代入：

```text
147456 // (72 × 2048) = 1
```

所以默认情况下，一个 batch 正好对应一个 optimizer step。

---

## 14. 优化器具体怎样修改表

第 470-480 行先区分普通模型参数和 n-gram 参数：

```python
ngram_markers = ("value_embeds", "bigram_ves", "trigram_ves",
                 "ve_gate", "bigram_gate", "trigram_gate")
self.ngram_params = []
self.adam_params = []
for name, p in model.named_parameters():
    if any(m in name for m in ngram_markers):
        self.ngram_params.append((name, p))
    else:
        self.adam_params.append((name, p))
```

`model.named_parameters()` 依次给出模型中每个参数的名字 `name` 和数值张量 `p`。

`any(...)` 表示只要参数名字包含任意一个 n-gram 标记，就归入 `ngram_params`。

因此名字中包含 `bigram_ves` 的所有表权重使用 RMSProp；普通 Transformer 权重使用 AdamW。

bigram 表的更新函数在第 516-531 行：

```python
def _rmsprop_step(self, name, p, lr_t):
    g = p.grad
    if g is None:
        return
    ...
    exp_avg_sq.lerp_(g.square(), 1 - b2)
    bias2 = 1 - b2 ** step
    denom = (exp_avg_sq / bias2).sqrt() + 1e-10
    p.add_(g / denom, alpha=-lr_t)
```

这里：

- `p` 是 bigram 表参数；
- `p.grad` 是该表的梯度，保存为 `g`；
- `g.square()` 是梯度逐元素平方；
- `exp_avg_sq` 保存历史平方梯度的移动平均；
- `denom` 用历史梯度大小归一化当前更新；
- `lr_t` 是当前学习率；
- `p.add_(..., alpha=-lr_t)` 中负号表示沿着降低 loss 的方向更新。

可以粗略理解为：

\[
表格新值 = 表格旧值 - 学习率\times归一化后的梯度.
\]

第 527 行明确写着：

```python
# no weight decay on n-gram tables
```

所以 n-gram 表不使用 weight decay。

`nn.Embedding` 的反向传播只让本 batch 查到的行得到非零梯度。没有被访问的行参数值不会在这个 step 因梯度更新而改变；如果哈希碰撞导致多个 bigram 使用同一行，它们的梯度会混在该行中。

---

## 15. 用 `(喜欢,吃)` 完整走一遍程序

设 token ID：

```text
BOS=1, 我=10, 喜欢=20, 吃=30, 苹果=40
```

原始数据块：

```text
[1,10,20,30,40]
```

数据加载器生成：

```text
inp = [1,10,20,30]
tgt = [10,20,30,40]
```

forward 收到：

```text
idx = [1,10,20,30]
```

第 368 行生成：

```text
prev_idx = [1,1,10,20]
```

最后一个输入位置形成：

```text
(prev_idx[3], idx[3]) = (20,30) = (喜欢,吃)
```

对编号 1 的表进行两次哈希：

```text
h_1,0(20,30) = 276518
h_1,1(20,30) = 197354
```

查表：

```text
table_1_0[276518] -> 384维
table_1_1[197354] -> 384维
```

拼接：

```text
e_1(喜欢,吃) -> 768维
```

对编号 3、5、7 的表重复同样过程。它们使用不同哈希常数和独立权重，得到：

```text
e_3(喜欢,吃)
e_5(喜欢,吃)
e_7(喜欢,吃)
```

求和：

```text
r(喜欢,吃) = e_1 + e_3 + e_5 + e_7
```

输入注入：

```text
x_吃 = token_embedding(吃)
     + position_embedding(位置3)
     + r(喜欢,吃)
```

经过 8 个 Transformer block 后：

```text
lm_head(x_吃) -> 对8192个候选下一词的分数
```

该位置正确答案来自 `tgt[3]`：

```text
tgt[3] = 40 = 苹果
```

交叉熵根据模型给“苹果”的概率计算 loss。`loss.backward()` 把梯度传回刚才查到的八个子表行，也就是四套表乘两张子表。`optimizer.step()` 再更新这些行。

下一次再次出现 `(喜欢,吃)` 时，同样的 token ID 和固定哈希常数会产生相同的行号，所以会读到上次更新后的向量。

---

## 16. “频率”在这段机制中的准确含义

假设训练集中：

```text
(喜欢,吃) 出现 1000 次
(讨厌,吃) 出现 5 次
```

由于哈希函数固定，`(喜欢,吃)` 每次都会访问同一组表行，因此这些行会收到很多次来自该 bigram 的梯度。

固定顺序重放 3 个 epoch 时，同一批 bigram 会按相同顺序再次出现，所以它们对应的表行也会再次被更新。

但还需要记住两点：

1. 高频 bigram 的某个哈希行也可能和其他 bigram 碰撞，所以一张子表的一行不一定只承载一个 bigram。
2. 同一个 bigram 在四套表中使用不同常数，并且每套表又使用两次哈希，因此它的完整表示由多个行共同决定，能减轻单次哈希碰撞的影响。

因此频率不是代码显式维护的一个计数器。代码没有写：

```text
count[(喜欢,吃)] += 1
```

频率的影响来自：同一个 bigram 反复查到同一组参数行，于是这些参数反复参与 loss、反复接收梯度。

---

## 17. 最容易混淆的七件事

### 17.1 没有创建新的离散 token

`(喜欢,吃)` 没有被 tokenizer 改成一个新 ID，词表大小仍是 8192。

### 17.2 没有缩短序列

输入原来有 `T` 个位置，bigram 操作后仍然有 `T` 个位置。

### 17.3 bigram 向量附着在第二个 token 上

`(喜欢,吃)` 的向量加在“吃”这个位置，不是“喜欢”的位置。

### 17.4 没有使用预测答案

预测“苹果”时只构造 `(喜欢,吃)`，不会构造 `(吃,苹果)`，所以没有 label leakage。

### 17.5 两种“合并”必须区分

代码组合两个 token ID 来计算哈希 key；之后又拼接两张子表的 384 维向量。它没有把两个序列位置合并。

### 17.6 四套表是求和，不是拼接

每套内部：`384 + 384` 是沿维度拼接成 768。

四套之间：四个 768 维向量逐元素求和，结果仍是 768。

### 17.7 “奇数层注入”对 input 模式并不准确

代码编号 1、3、5、7 决定建立哪四套表。当前 `input` 模式先把四套表的输出相加，然后只在 Transformer 输入处注入一次。只有改成 `v` 或 `y` 时，后面的 block 内注入路径才真正使用相应层的 bigram 向量。

---

## 18. 最终执行流程图

```text
原始连续 token
[BOS, 我, 喜欢, 吃, 苹果]
              │
              ├─ 输入 inp = [BOS, 我, 喜欢, 吃]
              └─ 答案 tgt = [我, 喜欢, 吃, 苹果]

inp
 │
 ├─ 普通 token embedding ─────────────────────────────┐
 ├─ position embedding ───────────────────────────────┤
 │                                                    │
 └─ 右移一格得到 prev_idx                             │
      │                                               │
      └─ 逐位置组成 (prev,current)                    │
           │                                          │
           ├─ 编号1：两次哈希→两次查表→384+384=768 ─┐│
           ├─ 编号3：两次哈希→两次查表→384+384=768 ─┤│
           ├─ 编号5：两次哈希→两次查表→384+384=768 ─┤│
           └─ 编号7：两次哈希→两次查表→384+384=768 ─┘│
                            │                         │
                      四个768维向量求和               │
                            │                         │
                            └──────── 输入逐元素相加 ─┘
                                        │
                                8层 Transformer
                                        │
                               每个位置8192个分数
                                        │
                              与右移后的 tgt 算 loss
                                        │
                                  loss.backward()
                                        │
                         被访问的 bigram 表行获得梯度
                                        │
                                optimizer.step()
                                        │
                              RMSProp 更新这些表行
```

一句最精确的总结是：

> 代码把 `(前一个 token ID, 当前 token ID)` 当成一个可重复计算的地址，经两次哈希从两张可训练表中取出两个半向量；每套表内部拼接，四套表之间求和，然后把所得 768 维上下文向量加到当前 token 的输入 embedding 上。模型预测下一个 token 后，梯度会更新这次查到的表行。
