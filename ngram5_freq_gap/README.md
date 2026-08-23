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
| `model.py` | 126 | 动态加载器：从集群 `train.py` 源码截取定义段并 exec，取出 `NanoGPTOriginal`。⚠️ 见下方 P0 |
| `hash_utils.py` | 80 | `data_gen.py` hash 的精确 torch 张量版（31-bit limb 避免 int64 溢出）。与主线 table 寻址 hash **数学上无关**，不可合并 |
| `gap_experiment.py` | — | 已提升为 `code/gap_experiment.py`，此处为兼容 re-export |
| `_gap_experiment_vendored.py` | 312 | 集群同步用的 vendored 副本（集群上没有 `code/`） |
| `make_smoke_data.py` | 154 | CPU smoke 数据 fixture |
| `resample_aligned_dataset.py` | 199 | 一次性数据集对齐工具，无 launcher 引用，**候选废弃** |
| `tests/` | 582 | ★ **全仓库唯一单元测试**（3 文件 22 个 test，纯 CPU）。保护 `data_gen.py` 的科学正确性：hash 确定性、torch/python hash 等价、alpha 重采样因子、train/val 独立抽样前提、精确索引 vs hash 碰撞回归 |
| `cluster/run_on_cluster.sh` | 131 | ★ **主力入口**。rsync → 带 contract 校验的数据集生成 → 启动 trainer |
| `cluster/run_big_continuous.sh`<br>`cluster/launch_ddp_train.sh`<br>`cluster/monitor_full163.sh` | 166 | full-163 全语料 4-GPU DDP 长跑（70000 步）。⚠️ **未完成线**，见下方 P1 |
| `CLUSTER_PATCH_GUIDE.md` | 100 | ⚠️ `[OUTDATED]`，行号已失效 |

## 5. 已知阻塞项

### P0 · `model.py` 的 fallback 是死路径（本包当前在本仓库跑不起来）

`model.py` 依次尝试仓库根 `train.py` 与集群 `/data3/guoshaoyang/ngram-gap-exp/train.py`，
CPU 回退指向 `nanogpt_gap_vanilla_control/`——**这三个在本仓库都不存在**。
`data_gen.py` 的 `_load_upstream_lib()` 同理找不到 upstream `lib.py`。

后果有两层：
1. 本地 `import model` 直接 `ModuleNotFoundError`；
2. 更严重的是，§2 那张「符合极简 setting」的表**只是环境变量声明，无法在仓库内验证**，
   因为真正的 backbone 代码不在这里。主线 `code/train.py` 是可直接审计的。

**修法**：把 fallback 改为主线 `code/train.py`。这一步同时消灭「两份 nanoGPT 实现」的隐患，
是把本包真正并入主线的关键动作。

### P1 · full-163 线未完成

closure 状态为 `DATA_GEN_RUNNING`，`meta.json` 从未产出；输入 `data_split.full163.json`
不在本仓库。相关 3 个脚本是**未验证代码**，已加 banner 标注，不要当作可信资产。

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
                                                    ├──> model.py ──> ⚠️ 集群 train.py（仓库外）
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
