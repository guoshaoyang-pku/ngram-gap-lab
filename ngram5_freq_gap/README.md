# ngram5_freq_gap — 受控数据干预运行时

> **定位**：这不是主线 `code/train.py` 的竞争实现，而是**第四个实验维度**的运行时。
> 主线三个维度（注入点 / table 优化器 / epoch 长度）都动**模型侧**；本包固定极简 setting，
> 只动**数据侧**。它同时是 `code/run_minimal_matrix.sh` 与 `tasks/l4_synth_powerlaw/` 的共用 trainer。

---

## 1. 科学问题：coincidental gap 的频率分解

构造受控数据集，使 train / val 的 next-token 是**从同一直方图独立抽样**——
gap 因此是纯粹的**有限样本巧合**，没有分布漂移。理论预期 `gap(r) ≈ (K_eff − 1) / r`。

唯一自变量是 **`alpha`**（低频上采样强度）：

```
n_train(b) = round(r(b) · f_train · k(b))
n_val(b)   = round(r(b) · f_val)              # val 永不重采样
k(b)       = clip((r_ref / r(b))^alpha, k_min, k_max)
```

`alpha = 0` 是基线；`alpha > 0` 抬高低频桶的有效 `r_train`，**应当压缩 gap**。
这是对数据分布本身的**因果干预**，不是观察性关联。

## 2. 它符合极简 setting

所有 `CURRENT_*` 开关都被显式关闭（见 `cluster/run_on_cluster.sh`）：

| 极简项 | 本包设置 |
|---|---|
| backbone | `ARCH_VARIANT=nanogpt_original` + `NGRAM5_TRUNK=transformer` |
| position encoding | `POSITION_ENCODING=learned_abs`（无 RoPE） |
| normalization | `CURRENT_NORMALIZATION=layernorm`（无 RMSNorm） |
| current shell | `CURRENT_NGRAM_INJECTION_IMPL/ATTENTION_NORM/HEAD_GATE/LAYER_POOL/LOGIT_SOFTCAP/LINEAR_BIAS=none`，`RESIDUAL_PATH=plain` |
| Muon | `NANOGPT_MATRIX_OPTIMIZER=adamw`（无 Muon） |
| 注入点 | `NANOGPT_NGRAM_INJECTION_POSITION=input` |
| table 优化器 | `NGRAM_TABLE_OPTIMIZER=rmsprop`，`NGRAM_TABLE_BETAS=0.0,0.999` |

## 3. 主线两个 bug 在本包不存在

- **`table_betas[1]` 被覆盖**：本包**没有优化器实现**，β₂ 经环境变量透传，无此代码路径。
- **freq-bin 复用训练迭代器**：本包用**完全 materialized 的固定 batch 列表**
  （`trainer.py` `fixed_train_probe_batches`），probe 只读遍历，且用
  `itertools.chain(fixed_train_probe_batches, train_loader)` 保证不跳数据；
  batch 身份以 SHA256 记入 run contract。

⚠️ **口径差异（不可直接比较）**：本包 train probe 是**训练前抓取、全程不变的 2 个 batch**，
测的是「这批固定数据的记忆曲线」；主线是**滚动的独立诊断迭代器**，测的是「训练分布上的平均 train loss」。
两者各自自洽，但数值**不能互相引用**。

## 4. 文件职责

| 文件 | 行数 | 职责 |
|---|---:|---|
| `data_gen.py` | 1556 | ★ **不可替代资产**。受控数据集生成器：Mersenne 多项式 hash → 精确直方图扫描 → alpha 重采样 → 流式发射。支持任意 `--order`、`--val-source {train,test}`（巧合 gap vs 真 held-out）、`--emit-format bin`、token cache / fast scan / fast emit（全语料性能路径） |
| `trainer.py` | 1467 | 训练循环。**独有能力**：DDP、checkpoint/resume、全 batch trace、probe details npz、60+ 字段 run contract（含 batch SHA256）。**不含模型与优化器** |
| `lib.py` | 830 | 集群 canonical `lib.py` 的 **fork**，新增 `ngram5_blocks` data mode。⛔ **禁止合并回主线**——历史上该合并曾让集群 `lib.py` 不可导入 |
| `model.py` | 126 | 动态加载器：从仓库 `code/train.py`（或 launcher 同步副本）加载 `NanoGPT`，取出主线模型与优化器 |
| `hash_utils.py` | 80 | `data_gen.py` hash 的精确 torch 张量版（31-bit limb 避免 int64 溢出）。与主线 table 寻址 hash **数学上无关**，不可合并 |
| `gap_experiment.py` | — | 已提升为 `code/gap_experiment.py`，此处为兼容 re-export |
| `_gap_experiment_vendored.py` | 312 | 集群同步用的 vendored 副本（集群上没有 `code/`） |
| `make_smoke_data.py` | 154 | CPU smoke 数据 fixture |
| `resample_aligned_dataset.py` | 199 | 一次性数据集对齐工具，无 launcher 引用，**候选废弃** |
| `tests/` | 582 | ★ **全仓库唯一单元测试**（3 文件 22 个 test，纯 CPU）。保护 `data_gen.py` 的科学正确性：hash 确定性、torch/python hash 等价、alpha 重采样因子、train/val 独立抽样前提、精确索引 vs hash 碰撞回归 |
| `cluster/run_on_cluster.sh` | 131 | ★ **主力入口**。rsync → 带 contract 校验的数据集生成 → 启动 trainer |
| `cluster/run_on_cluster.sh` | 131 | ★ **主力入口**。rsync → 带 contract 校验的数据集生成 → 启动 trainer。full-163 全语料线已退役（脚本删除），数据坐标见 `docs/notes/data/full-corpus-full163.md` |

## 5. 已知阻塞项

### P0 · `model.py` 与主线模型绑定 ✅ 已修复（2026-08-24）

`model.py` 现在优先加载仓库内的 `code/train.py`；集群 launcher 将同一份文件
同步为 package 根目录的 `train.py`，作为远程副本入口。不存在 vanilla fallback、
current-shell fallback 或仓库外历史模型 fallback。

因此 ngram5 的 backbone、n-gram table、初始化和 `MixedOptimizer` 都来自同一份
主线实现；`run_contract.json` 会记录实际加载的 source path 与 optimizer 口径。

### P1 · full-163 线 ✅ 已退役（2026-08-23）

脚本已删除（从未跑完、依赖仓库外文件、非极简主线）。
完整数据集的坐标与生成参数登记在 `docs/notes/data/full-corpus-full163.md`，
未来做全语料仿真按那份文档 + `agents.md` §1 重建，不复活旧脚本。
## 6. 命名说明

包名里的 `ngram5` 是历史误称——launcher 实际跑 `--order 3`，数据集叫 `trigram_alpha*`。
改名 `controlled_ngram` 需同步 3 个调用点（`code/run_minimal_matrix.sh`、
`tasks/l4_synth_powerlaw/cluster/*.sh`、`code/tools/validate_fastgen.py`）与集群 rsync 路径，
暂缓。

## 7. 依赖图

```
code/run_minimal_matrix.sh ──────────────┐
tasks/l4_synth_powerlaw/cluster/*.sh ────┼──> ngram5_freq_gap/trainer.py
code/tools/validate_fastgen.py ──────────┘         │
                                                    ├──> lib.py ──> code/gap_experiment.py
                                                    ├──> model.py ──> code/train.py
                                                    └──> hash_utils.py

code/make_ngram_blocks.py ──(输出契约兼容)─────────> ngram5_blocks loader
```

`code/make_ngram_blocks.py`（262 行）是 `data_gen.py` 的 **alpha=0 特例**——
极简数据路径，无 Poisson、无 alpha、无频率控制。两者是主线/干预线的分工，不重复。

## 8. 运行

```bash
# 单 alpha 单 GPU（主力入口）
ALPHA=0.0 MAXSTEPS=1000 GPU=0 bash ngram5_freq_gap/cluster/run_on_cluster.sh

# CPU 单元测试（无需 GPU / tokenizer / pyarrow）
python -m pytest ngram5_freq_gap/tests -q
```
