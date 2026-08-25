# Patch: agents.md §1.2 n-gram 表架构重做（clean 单表）

> 状态：**已应用（用户 2026-08-25 拍板）**。
> 关联：`docs/notes/method/clean-table-rework.md`（新 SSOT 全文）。

## 1. 这次改了什么

`agents.md` §1.2 的 `table size` 行由旧的

> **1M** | `vocab_size × 64 = 524,288` 行 × 2 个 hash embedding = **1,048,576**

改为 **clean 单表架构**（`nn.Embedding(R, n_embd)`，R 任意设定、单层、
无 2-hash 拼接），并标注旧 1M/4 层/2-hash 版本为历史框架。

## 2. 为什么改

用户（2026-08-25）指出旧架构有**冗余**：

1. **4 层求和**：input 注入把 4 层独立表求和后加在入口，层与层区分失去功能
   意义（v/y 时代才需要每层独立），参数/显存白增 4 倍；
2. **2-hash 拼接**：每层表 = 2 个半维 embedding 拼接，不是一张
   `nn.Embedding(n, n_embd)`；
3. **表大小被 `vocab × mult` 锁定**：R = 8192 × mult 是历史命名选择，不是
   数学必需；用户要求 R 任意设定以自由扫描碰撞。

用户拍板：**重扫所有 hash-table-size 实验**，在 clean 单表架构下给出更精细
结果。因此 agents.md 的 table 轴定义必须更新，否则新 run 会继续用旧架构。

## 3. 兼容性

- 旧 1M/4 层/2-hash 表仍是**历史 run 的 setting**（所有已跑 run 的口径不变，
  结论仍有效，但标注为历史框架）；
- 新 clean 单表是**新 run 的默认**（`--bigram_clean_table R`），不与旧 run 混用；
- 两套并行，重扫完成后在报告里并列展示。

## 4. 回滚

若用户改主意，`agents.md` §1.2 改回旧定义即可；`clean-table-rework.md` 保留
为历史记录，不删除（它是新框架的权威文档，回滚时需同步标注降级）。

## 5. 触发点

- 任何新 table-size run（含重扫）默认走 clean 单表架构；
- 引用旧 69 点网格、mult=128 离群点、perfect 2.17 倍等结论时，须标注
  `[HISTORICAL 4-LAYER FRAMEWORK]`。
