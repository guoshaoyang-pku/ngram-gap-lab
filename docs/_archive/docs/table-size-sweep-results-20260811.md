# Table-size sweep: 参数量 vs 低频涨落加权 (verified 2026-08-11)

> ⛔ **[DEPRECATED SETTING]** 本文档的实验建立在 `current shell` backbone 上，**不是本课题的极简 setting**。
> 结论方向可能正确，但所有数字都必须在极简 setting（vanilla nanoGPT + `input`/wte 注入 + table 1M + RMSProp 无动量）下重跑才能进主线。
> 极简 setting 定义见 `agents.md` §1；重跑队列见 `agents.md` §6.3。本文件仅供历史溯源。


**Question (from 2026-08-10 review):** gap 与 n-gram 表大小之间有什么定量关系？是参数量（capacity）主导，还是低频涨落加权（collision-induced fluctuation pooling）主导？

## 实验设计

- 数据：`t5_low`（vocab=2048，32,768 个 distinct 2-gram contexts，Zipf 型频率 r=1..256）
- 模型：`nanogpt_current_shell` 8 层 × 768，2000 步，seed 42（M=16/64 额外 seed 7 复测）
- **唯一变量**：`NGRAM_TABLE_MULT = NGRAM_EFFECTIVE_MULT = M`，物理行数 = 2048×M
  - 在 `toy/ws/train.py` 加入 env 旋钮（`NGRAM_TABLE_MULT` / `NGRAM_EFFECTIVE_MULT`），默认 64 与 CROSSOVER B baseline 字节一致
- 碰撞载荷：load = 32768 / (2048×M)
  - M=1 → load 16（16 个 context 挤 1 行）
  - M=16 → load 1.0（**恰好无碰撞**）
  - M=64/256 → 稀疏（load 0.25/0.06，行数超过 context 数，纯死参数）

## 结果

| M | load | headline gap | r=1 | r=2 | r=4 | r=8 | r=16..256 |
|---|---|---|---|---|---|---|---|
| 1  | 16.0 | 4.88 | 14.28 | 15.24 | 16.48 | 18.15 | ≈0 |
| 4  | 4.0  | 5.98 | 14.10 | 14.99 | 16.08 | 17.50 | ≈0 |
| 8  | 2.0  | 6.77 | 14.27 | 15.33 | 16.65 | 18.01 | ≈0 |
| 16 | 1.0  | 7.50±0.10 | 15.46 | 17.22 | 19.00 | 20.40 | ≈0 |
| 64 | 0.25 | 7.78±0.01 | 16.97 | 18.87 | 20.37 | 21.55 | ≈0 |
| 256| 0.06 | 7.27 | 17.01 | 18.62 | 19.92 | 21.01 | ≈0 |

图：`docs/figs/table_size_sweep.png`（A. headline gap vs M；B. 低频 bucket gap vs M）

## 结论

**1. 参数量不是主导**。把参数砍 64×（M=64→1），headline gap 只掉 2.9 nat（7.78→4.88），且**在无碰撞点 M=16 就饱和**——继续加死参数（M=64→256）gap 不再涨（甚至微降，7.78→7.27）。如果 capacity 主导，gap 应随参数持续上升。

**2. 效应发生在碰撞区（M<16）**。从 M=1→16（load 16→1），gap 从 4.88 单调涨到 7.50（+2.6 nat）。碰撞把低频 context 的涨落 pooled 到同一行，稀释了"诚实 val"下的 gap——这正是**低频涨落加权**的 signature。

**3. 但低频 bucket gap（r=1..8）本身对表大小极不敏感**。M=1 vs M=64，r=1 gap 只差 1.19×（14.28→16.97），r=8 只差 1.17×。也就是说：**gap 的形状（1/r 结构）不随表大小变**，变的只是整体幅度的一小部分。within-bucket slope 在所有 M 下都 ≈ +0.11（r=1→8 gap 递增），结构完全稳定。

**4. 与理论 MC 的偏差**：静态 count-table MC 预测无碰撞后 gap 应继续涨（M=16→256 再 +53%），但 GPU 实测饱和。说明真实模型在稀疏表下没有"免费午餐"——行越多，每行被训练的 update 越稀疏，抵消了 pooling 的减少。**参数量对 gap 的上限贡献在 M≈16（无碰撞点）就已封顶。**

## 一句话回答

> 表大小对 gap 的定量关系是：**碰撞区（M<16）内 gap 随表大小涨（低频涨落 pooling 稀释 gap），无碰撞后饱和，参数量再多也无用**。低频 bucket（r≤8）的 gap 幅度对表大小只有 ~1.2× 的弱依赖，gap 的主体结构（1/r 形状、within-bucket slope）完全由频率分布决定，不由表大小决定。

## 复现

- 代码：`toy/ws/train.py`（`NGRAM_TABLE_MULT`/`NGRAM_EFFECTIVE_MULT` env 旋钮）+ `toy/ws/table_size_patch.py`
- 启动：`toy/ws/run_table_sweep.sh`（round1: ts_m1/m4/m16/m64）+ `run_sweep2.sh`（round2: m8/m256/s7 复测）
- 分析：`toy/toy_analyze.py --run ts_mX`（source runs/ts_mX/env.sh 后）
- 运行目录：`/data3/guoshaoyang/ngram-gap-exp/toy/runs/ts_m*`；checkpoints 在 `toy/checkpoints/ts_m*`
