# clean-table-rework：n-gram 表架构重做（单表、无 4 层求和、无 mult 缩放）

> 状态：**已拍板（用户 2026-08-25）**。重扫所有 hash-table-size 实验。
> 本文档是重做后 table-size 轴的**新 SSOT 定义**，取代 plan-5 §P3 里基于
> 旧 4 层求和表架构的 table 轴设计。

---

## 0. TL;DR

- **问题**：现有 n-gram 表是"4 层独立表求和 + 每层 2-hash 拼接 + 表大小被
  `vocab × mult` 锁定"的历史冗余架构（继承自 v/y 注入时代）。用户要求回归
  **干净的 context→embedding 单表**：一个 context 映射到一个完整 embedding，
  像 `wte` 一样直接、无 hash 拼接、无 4 层求和。
- **新架构**（`--bigram_clean_table`）：`nn.Embedding(R, n_embd)` 单表，
  表大小 **R 任意设定**（不再强制 `vocab × mult`），注入仍是 `input` / wte
  over-encoding，单层。
- **动机**：现有 69 点 table 网格、mult=128 离群点、K/N 斜率、perfect-vs-
  碰撞的 2.17 倍，全部建立在"4 层求和 + 2-hash 拼接"框架上。换架构后
  collision 逻辑、K/N 曲线、perfect 对比全部改变，必须在**新框架下重扫 R**
  才能回答"clean 单表的 gap vs table size"。
- **不重做的部分**：注入点（input/v/y）差异、频率→gap、replay/epoch 机制、
  row-level 分析——这些不依赖"4 层 vs 单表"，结论仍有效。

---

## 1. 旧架构的问题（为什么重做）

### 1.1 现在的 input 注入实际是什么

`_compute_input_ngram_residual`（train.py 468-497）做的事：

1. 对同一个 context，**用 4 组不同 primes hash 到 4 个不同行**
   （`bigram_ve_layers = {1,3,5,7}`，每层 `_bp[j]` 不同）；
2. 从 **4 张独立 `nn.Embedding`**（`bigram_ves[str(li)]`）各取向量；
3. **每张表内部又是 2 个半维 embedding 拼接**（`bigram_K=2`，各 384 维，
   concat 成 768 维）；
4. 4 个 768 维向量**求和**成一个 residual，加到 `x = wte + wpe`。

所以"一个 context → 一个 embedding"在旧架构里其实是：
**context → 4 组不同 primes → 4 张表各 2 个半维 → 8 个半维向量 → 拼成 4 个
768 维 → 求和成 1 个**。

### 1.2 冗余清单

| 冗余 | 位置 | 后果 |
|---|---|---|
| **4 层求和** | `_compute_input_ngram_residual` 遍历 `bigram_ve_layers` | input 注入把 4 层求和后加在入口，**层与层区分失去功能意义**（v/y 注入时每层各注一份才需要每层独立表）；参数/显存 **4 倍** |
| **2-hash 拼接**（`bigram_K=2`） | `__init__` 349-352 | 每层表 = 2 个 `nn.Embedding(半维)` 拼接，不是一张 `nn.Embedding(n, n_embd)` |
| **表大小被 `vocab × mult` 锁定** | `bigram_table_size = vocab_size × table_mult` | 表行数必须 = 8192 × mult，无法任意设定 R；`mult` 只是便于命名的缩放，不是独立变量 |

### 1.3 为什么是历史遗留

- 4 层求和从 `has_ve`（交替层）+ `value_embeds` 继承，v/y 时代"每层各注一份"
  有意义（ResFormer 风格），input 时代求和后层区分无意义。
- 2-hash 拼接直接学自 deepseek-engram 的 multi-head hashing（"K 个 hash head
  各查 prime-size 表再 concat"，arXiv 2601.07372v1 §2.1）。但 engram 的
  multi-head 是为了**缓解碰撞**，而用户要的干净单表是**不要碰撞的 perfect 行号**
  （或单一 hash），与 multi-head 目的不同。

---

## 2. 新架构（clean 单表，SSOT）

### 2.1 模型

| 项 | 值 | 说明 |
|---|---|---|
| 表 | **单个 `nn.Embedding(R, n_embd)`** | 一个 context → 一行 → 一个完整向量，像 `wte(token_id)` 一样直接 |
| **表大小 R** | **任意设定** | 不再强制 `vocab × mult`；R 是自由参数，直接决定碰撞 |
| 注入 | `input` / wte over-encoding | 不变，`x = wte + clean_table(row_id)` |
| 层数 | **单层**（`bigram_ve_layers = {layer1}`） | 无 4 层求和 |
| hash | 单一 hash（或 perfect-map 行号） | 无 2-hash 拼接；`bigram_K = 1` |
| 行号来源 | 同一 context 映射同一行 | 碰撞与否由 R vs distinct-context 数决定 |

### 2.2 实现

- `train.py` 新增 `--bigram_clean_table R`（或复用 perfect-map 行号，R = distinct + 1 时零碰撞）：
  ```python
  # clean 单表：R 任意设定
  self.bigram_clean = nn.Embedding(R, config.n_embd)
  # 行号 = 单一 hash(prev, idx) % R   （或 perfect map）
  row = (prev * p1 ^ idx * p2) % R
  x = x + self.bigram_clean(row)   # input 注入，一次 lookup
  ```
- 改动 ~30 行，全部新参数默认关闭，**不碰已有 run 口径**。
- 沿用 `--save_final_model` / `--fixed_train_probe` / row-level 分析管线。

### 2.3 显存（相对旧架构）

| R | clean 单表参数（fp32） | 旧 4 层 mult=64（fp32） |
|---|---:|---:|
| 1M | 768M | 1610M（4×402.7M） |
| 3.54M（= distinct，零碰撞） | 2.7G | —（4 层 perfect 需 ~130G，放不下单卡） |

clean 单表在同 R 下比旧 4 层**小 4 倍**，且零碰撞锚点（R=N）可放进单卡。

---

## 3. 重扫实验设计（新 table 轴）

### 3.1 横轴：R（clean 单表行数），不再用 mult

| R | K/N（= R / 3.54M distinct） | 角色 |
|---|---:|---|
| 64K | 0.018 | 强碰撞 |
| 128K | 0.036 | |
| 256K | 0.072 | |
| 512K | 0.145 | |
| 1M | 0.283 | 接近旧默认 |
| 2M | 0.565 | |
| 3.54M | 1.0 | **零碰撞（R = distinct）** |
| （可选 5M/8M） | >1 | 超容量，验证 K>N 饱和 |

### 3.2 训练

- 复用 plan-5 §P3 的训练协议（L4 epoch、1000 步、seed 42 → 43/44）；
- bigram-only / trigram-only / both 三个 module；
- no-ngram 共享基线，不随 R 重跑；
- 每档产物：`table_occupancy.json`（若 hash 表）、fixed probe 日志、row-level。

### 3.3 与旧结论的关系

- 旧 69 点网格（4 层求和 + 2-hash）**降级为历史框架**，只作溯源；
- 新网格在 clean 单表框架下独立生成，两套**并列**展示；
- mult=128 离群点、K/N 斜率、perfect 2.17 倍等旧发现，在新框架下**重新检验**。

---

## 4. 科学问题（重扫要回答的）

1. **collision 是否是 gap 的必要条件**：R=N 零碰撞 vs R 很小时强碰撞，gap 差多少？
2. **clean 单表的 Δ vs R 曲线**：斜率、是否饱和、jamming（K~N）行为？
3. **4 层求和 vs 单层**：旧 4 层 gap 是单层 ~3.8 倍，是参数量的功劳还是求和架构的功劳？
4. **与 SCONE 对照**：SCONE 是 input-layer longest-match 精确表（无碰撞），
   clean 单表零碰撞版本正好是它的"可训练向量"版，可做文献锚点。

---

## 5. 引用与兼容

- 本文档是 **table 轴新 SSOT**；plan-5 §P3（旧 4 层表架构）降级为历史记录。
- 与 Engram（multi-head hash + gate）的关系：clean 单表**去掉** multi-head 与
  gate，是最简 over-encoding；这是刻意为之（研究 gap 需要无 gate 的纯记忆）。
- 与 SCONE（input-layer longest-match）的关系：注入位置相同，只是 SCONE 用
  longest-match 精确表，我们扫 R（含碰撞）或 R=N（零碰撞）。
