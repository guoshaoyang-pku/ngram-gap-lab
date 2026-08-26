# 实验注册表（Experiment Registry）

> 维护时间：2026-08-25（git `2e1a747`）
> 本表回答三个问题：**① 哪些实验是当前注册的合法实验；② 每个实验做了什么统计、画了什么图；③ 核心代码在哪、来源是否唯一可追溯。**
> 权威数据源：主线 run 为 `data/runs_fixed/*_fixed/`；scaling 专属 run 为
> `data/runs_scaling/*_fixed/`。口径见 `agents.md` §1；数值细节见
> `experiment-log.md` 与 `experiment-lines.md`。

---

## 0. 当前合法 setting 基线（SSOT 摘要）

- **backbone**：vanilla nanoGPT 8L · 6H · 768D，vocab 8192，seq 2048，learned abs PE，LayerNorm，tied embedding，全 attention。
- **n-gram 表**：bigram + trigram（unigram/fourgram 关）；注入点 `input`（wte over-encoding）为主，v / y 仅作消融对照；**clean 单表** `nn.Embedding(R, 768)` 单层、单 hash、R 任意（`--bigram_clean_table R`）。
- **优化器**：表 RMSProp 无动量 betas `(0.0, 0.99)`、表 LR ×2；backbone AdamW `(0.8, 0.95)` wd 0.1、lr **0.0006**、**100-step `warmup_constant`**。这是 v5 主线：step 1–100 从 `1.5e-4` 线性升至 `6e-4`，之后恒定；不使用 cosine 或 warmdown。
- **数据**：`fixed` 模式固定顺序 epoch replay；train shard 1、val shard 不重叠；device batch 72、total batch 147,456 tokens；seed 42（43/44 复现）。
- **计算**：bf16（autocast）、**默认不 compile**；steps 1000 / 2000。
- **gap 定义（唯一）**：`val_loss − train_loss`，同一 step、同一 batch 口径，train 侧 = **当前训练 batch 的 online loss**（v3 起；v2 之前是 4-batch 诊断窗口，已废弃）。
- **评估节奏**：三层默认 freq=10（主）/ freq=50（只看曲线）/ `--val_steps 1000`（只要末端）。freq 跟随 val_steps 对齐。

---

## 1. 实验总表

| 实验大类 | 实验名称 | 实验 setting | 统计处理 / 画的图 | 状态 |
|---|---|---|---|---|
| **主线注入点** | M2 注入点消融 v10 | input / y / v / nogram 四臂；2000 步；val 每 10 步；v10 时代口径 | 注入点 loss/gap 曲线；频率 bin 双轴图；gap-vs-frequency 图；loss-gap-table-RMS 对齐 | ✅ 完成（历史口径，被 v2 取代） |
| **主线注入点** | M2-v2 注入点消融（β₂=0.99、表 LR ×2） | 同上四臂；2000 步；该波次使用 bf16+compile，且频率诊断不是 current-batch 口径 | 注入点 gap/loss 曲线；final gap 对比；频率 bin 双轴图（23 桶） | ⚠️ 历史波次；被 v5 的 no-compile、current-batch 契约取代 |
| **主线注入点** | M2-v3 注入点消融（current-batch 口径） | 同上四臂；freq-bin train 侧 = 当前训练 batch per-token loss（零额外 forward） | 同上 + freq-bin 的 train 侧与 online train_loss 完全同 batch 校验 | 🟡 三机队列运行中（`_v3_fixed` 部分回传） |
| **主线注入点** | M2-v4 注入点消融（历史 warmup/constant 口径） | 同四臂；2000 步；历史 run 的 schedule/表架构必须逐目录核验 | 仅作过程记录，不能作为 v5 结论 | ⚠️ 被 v5 取代 |
| **主线注入点** | M2-v5 注入点消融（当前权威） | input / y / v / nogram；2000 步；clean R=2²⁰；`lr=6e-4`、`warmup_constant(100)`；bf16、不 compile；train loss=当前 batch online loss；seed 42/43/44 | 原始 online loss/gap 点（只以 3 点均值连线）；频率 bin 双轴图；gap-vs-frequency；loss-gap-table-RMS 对齐 | ✅ 三 seed × 四臂完成；seed 42 input/y/v/nogram=5.741/3.640/2.014/0.245，s43=5.811/3.277/2.881/0.253，s44=5.515/3.439/2.723/0.253 |
| **剂量扫描** | M5 shard 大小扫描（dose） | train shards 0.25x–8x（12 档）；input 注入；2000 步 | gap vs dose（对数-对数幂律 α≈−2）；gap @2000 对比；epoch 分界标注 | ✅ 完成（v2 12 点齐） |
| **剂量扫描** | M5-v5 fixed-step 剂量扫描 | 0.25x、0.5x、0.75x、1.5x、2x、2.5x、3x、4x、5x、6x、8x；input；2000 步；v5 SSOT | 原始 online gap-vs-dose；双对数 dose 图；epoch 分界；频率分解 | ✅ 11/11 完成；gap 11.589（0.25x）→−0.077（8x） |
| **剂量扫描** | M6-v5 epoch 对齐批 | 固定 5 epoch；0.25x–4x；420–6700 steps；input；v5 SSOT | 固定 epoch 下的 online gap 对比、epoch 分界与剂量曲线 | ✅ 9/9 完成；0.75x/1x/2x/4x gap=4.511/4.503/3.357/2.089 |
| **表优化器** | M3 Table 优化器 · 1x epoch | RMSProp/AdamW/SGD；LR 剂量（×1/×2/×4）；β₂（0.98/0.99）；1x epoch | 优化器对比 loss/gap；LR 剂量曲线；β₂ 曲线 | ⚠️ 完成但 β₂ 结论待修正（B2 bug 直击） |
| **表优化器** | M4 Table 优化器 · 2x epoch | 同 M3，epoch 拉长 | 1x vs 2x 对比 | ⚠️ 同上（`b2_099` 0.64→2.00，+210%） |
| **表优化器** | M3b-v5 表学习率 × β₂ gate | clean 双表、input、1000 步；RMSProp/AdamW/SGD；scale 0.5–4；β₂ 0.95–0.999；以及 2000 步 seed 43/44 gate | 末端 loss/gap、完整曲线与健康区筛选 | ✅ 完成；正式中心点为 RMSProp `(0,0.99)`、表 LR ×2 |
| **表优化器** | M7 短 epoch × β₂ | 0.25x/0.5x 短 epoch；β₂=0.99 | per-epoch 台阶清晰度曲线 | ⚠️ 完成但图未按 `_fixed` 重生成 |
| **因果干预** | Causal interventions（v5） | input、1000 步；`reset_table` e1/e2、`mask_readout` e1、freeze-table e1、freeze-backbone e1；其余 v5 SSOT | 干预前后 online gap 与 effect size | ✅ 5/5 完成；reset-e1/e2/mask/freeze-table/freeze-backbone 均有权威端点 |
| **三轴 scaling** | S1 epoch 轴 · fixed-step（v5） | L1–L4 前缀（42/84/168/337 batches）；both 与 no-gram；1000 步；v5 SSOT | online gap vs epoch 长度；scaling 报告 | ✅ 8/8 完成；both/nogram 均为完整 L1–L4 曲线 |
| **三轴 scaling** | S1 epoch 轴 · fixed-epoch | 这是历史 compile 线；不纳入当前 v5 结论 | 仅保留历史溯源 | ⚠️ 不执行；若恢复该问题，须按 v5 单独重新注册 |
| **三轴 scaling** | S1 table size 轴（v5） | bigram/trigram 同步 clean R；18 个近对数均匀点，16K–2.347M；1000 步末端；v5 SSOT | gap-R、occupancy/collision 与频率加权负载 | ✅ 18/18 完成；只使用双表同步扩容结果 |
| **三轴 scaling** | S1 frequency 轴（v5） | L4 `epoch_batches=337`；bigram/trigram/both/nogram；1000 步；exact-frequency；v5 SSOT | G(E,f) 两因素模型检验 | ✅ 4/4 完成；gap=0.586/1.099/1.529/0.031（bigram/trigram/both/nogram） |
| **三轴 scaling** | S1 backbone safety（v5） | no-gram、8000 步、v5 SSOT | 无表 backbone 长训练 online gap | ✅ 8000/8000 完成；final gap=1.102 |
| **自然语言 5gram** | N1–N4 order=5 | 5gram context +trigram 注入 / 纯 transformer / LR ×1 / LR ×4 | per-bucket gap；gap-freq 图 | ✅ 完成（seed 42/43） |
| **受控数据干预** | ngram5 数据生成（alpha 上采样） | 固定极简 setting，只动数据侧 alpha | gap(r) ≈ (K_eff−1)/r 检验 | 🟡 独立包（`ngram5_freq_gap/`），与主线正交 |
| **toy 理论** | L1 查表记忆×replay | toy model | 记忆-gap 主矩阵 | ✅ 完成 |
| **toy 理论** | L2 Markov 精确 gap | 闭式解 | markov 三臂曲线 | ✅ 完成 |
| **toy 理论** | L3 单 context 采样律 | gap(r) 采样律 | 直接出图 | ✅ 完成 |
| **toy 理论** | L4 幂律合成数据 | 合成数据 + 真 harness | gap vs samples 幂律 | ✅ 完成 |
| **toy 理论** | L5 优化器伪影 | RMSProp v 锯齿 / 表容量 | 锯齿 / 容量对照 | ✅ 完成 |

> 状态图例：✅ 完成 · 🟡 进行中/待回传 · ⚠️ 完成但结论待修正/口径历史。

---

## 2. 统计处理清单（按产物类型）

| 统计 / 产物 | 说明 | 输出文件 |
|---|---|---|
| **online gap（主测量）** | `val_loss − train_loss`，同一 step、同一 batch；train 侧 = 当前训练 batch | `train_log.jsonl`（每 10 步） |
| **频率分桶 loss** | 按 exact 频率 23 桶；train 侧 v3 起 = 当前 batch per-token loss；val 侧 = fixed val batches | `freq_bin_loss.jsonl` |
| **exact-frequency** | 按 exact f 存 token count / distinct contexts / loss sum² / mean；`shared` 给 context-matched gap | `exact_freq_loss.jsonl` |
| **fixed train probe** | 独立 dataset 实例抓固定 train batches，SHA256 记账；仅诊断口径，不进主图 | `fixed_train_loss.jsonl` |
| **table 统计** | clean 单表按 branch 的 physical/logical rows R、distinct contexts K、occupancy、collision rate、singleton fraction、freq-weighted load；逻辑地址 2R 仅适用于历史 2-hash 表 | `table_occupancy.json`（`code/table_occupancy.py`） |
| **table 范数** | bigram/trigram 各层表 RMS 轨迹 | `table_norm.jsonl` |
| **因果干预** | reset / mask_readout / freeze 前后 gap | 干预对比图 |

---

## 3. 核心代码与「来源唯一可追溯」说明

### 3.1 核心代码位置

| 模块 | 文件 | 职责 |
|---|---|---|
| **训练主程序** | `code/train.py`（1715 行） | vanilla nanoGPT + n-gram 表；3 注入点；MixedOptimizer（表 RMSProp/AdamW/SGD + backbone AdamW）；fixed 数据模式；freq-bin / exact-freq / probe 统计；干预接线 |
| **频率索引** | `code/ngram_freq.py`（501 行） | `GlobalFrequencyIndex.build_from_chunks`，与模型 hash 逐位置一致 |
| **表 occupancy** | `code/table_occupancy.py`（251 行） | 行占用 / 碰撞 / 负载统计 |
| **replay/epoch 纯函数** | `code/gap_experiment.py`（312 行） | 主线与 ngram5 共用 |
| **数据准备** | `code/prepare_data.py` / `make_ngram_blocks.py` | token shards；受控 block 构造 |
| **集群启动器** | `code/cluster/*.sh` | `run_v5_clean.sh` 是单 run 契约；`run_v5_main_manifest.sh` 调度注入点、剂量、epoch、因果、probe、long 与 S1；`run_v5_table_grid.sh` 调度 18 点双表 R 网格。旧 launcher 仅供历史溯源 |
| **受控数据干预** | `ngram5_freq_gap/`（data_gen.py / trainer.py） | 第四维度：动数据不动模型 |
| **作图脚本** | `docs/plot_scripts/`（canonical：`gen_all_figures.py`、`gen_sweep_v2_figs.py`、`gen_shard_sweep_figs.py`、`gen_epoch_aligned_figs.py` 等） | 图进 `docs/figs/` 按实验线分目录 |
| **toy 线** | `tasks/l1..l5/` | 独立自包含，纯 numpy/torch |

### 3.2 注入与训练原理（一句话版）

- **注入**：`input` 为主——`x = wte(idx) + wpe(pos) + Σ ngram_ve`，n-gram 向量直接加在 token 嵌入上（over-encoding，不走 attention）；`y` 注入在 attention 输出后加回 residual；`v` 注入在 attention 的 V 上加（路径最间接）。
- **表架构**：新 SSOT 是 **clean 单表** `nn.Embedding(R, 768)`，一个 context → 一行 → 一个完整向量，单层、单 hash（`--bigram_clean_table R`，R 任意；与 perfect-map 组合 = 零碰撞锚点）。旧 4 层求和 + 2-hash 拼接（`vocab×mult` 表）已降级为历史框架。
- **优化器**：表与 backbone 分开——表 RMSProp 无动量 `(0.0, 0.99)`，表 LR ×2（实际 `0.0012`）；backbone AdamW。β₁=0 即无动量（`table_betas` bug 已修：`b2 = self.table_betas[1]`）。
- **训练**：`fixed` 模式 = 固定顺序 epoch replay（每轮从头重放同一 shard，L4=337 batches/epoch）；grad_accum 使 total batch = 147,456 tokens；bf16 autocast；默认不 compile。

### 3.3 来源是否唯一确定、可追溯

**结论：是，且是当前仓库的硬约束。** 依据：

1. **单一权威数据源**：主线为 `data/runs_fixed/*_fixed/`，scaling 为 `data/runs_scaling/*_fixed/`；`data/runs/` 与不带后缀副本因 freq-bin 诊断 bug（复用训练迭代器）已作废。两个已知 bug（freq-bin 白吃 train batch、`table_betas[1]` 被覆盖）已修复；v5 的 freq-bin train 侧进一步改为当前训练 batch 的逐 token loss。
2. **口径变更必须新 run_id**（P2）：v2→v3 的 freq-bin train 侧改动即因此新起 `_v3` 后缀；`_fixed` 后缀标记「修复后」run，`_v3` 标记「current-batch 口径」run，命名即口径。
3. **代码即契约**：launcher（`code/cluster/*.sh`）显式传全部关键参数（β₂=0.99、×2、bf16、不 compile），不依赖默认值；跨机跑前 `md5sum` 核对 `code/train.py`、`code/ngram_freq.py`、`code/cluster/*.sh`（§4.3）。
4. **文档权威性分级**：`agents.md` §1（SSOT）→ `experiment-lines.md`（全景）→ `experiment-log.md`（登记簿，含每次口径变更与勘误）→ `claims-ledger.md`（断言台账）。任何结论必须带 run_id / step / seed。
5. **历史框架显式隔离**：clean-table 重做后旧 4 层架构标记 `[HISTORICAL 4-LAYER FRAMEWORK]`，与新旧数据并列但不可混用；current shell / Muon / RoPE 系结论一律 `[DEPRECATED SETTING]`。

**唯一留白**：`experiment-log.md` 中 v10 时代（pre-v2）数值尚未回填为 `_fixed`（T1/T2/T3 待办）；v3 波次运行中，回传后需用 current-batch 口径重绘 freq-bin 图。这两处是「可追溯性」目前仅有的未闭合项。
