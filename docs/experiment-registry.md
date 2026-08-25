# 实验注册表（Experiment Registry）

> 维护时间：2026-08-25（git `2e1a747`）
> 本表回答三个问题：**① 哪些实验是当前注册的合法实验；② 每个实验做了什么统计、画了什么图；③ 核心代码在哪、来源是否唯一可追溯。**
> 权威数据源：`data/runs_fixed/*_fixed/`；口径见 `agents.md` §1；数值细节见 `experiment-log.md` 与 `experiment-lines.md`。

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
| **主线注入点** | M2-v2 注入点消融（新标准 β₂=0.99·×2） | 同上四臂；2000 步；bf16+compile（v2 波次）；seed 42 + s43/s44 复现 | 注入点 gap/loss 曲线（图 1，原始点+3 点平滑线，epoch 边界虚线标注）；final gap 对比；频率 bin 双轴图（23 桶） | ✅ 完成（当前权威注入点图） |
| **主线注入点** | M2-v3 注入点消融（current-batch 口径） | 同上四臂；freq-bin train 侧 = 当前训练 batch per-token loss（零额外 forward） | 同上 + freq-bin 的 train 侧与 online train_loss 完全同 batch 校验 | 🟡 三机队列运行中（`_v3_fixed` 部分回传） |
| **主线注入点** | M2-v4 注入点消融（历史 warmup/constant 口径） | 同四臂；2000 步；历史 run 的 schedule/表架构必须逐目录核验 | 仅作过程记录，不能作为 v5 结论 | ⚠️ 被 v5 取代 |
| **主线注入点** | M2-v5 注入点消融（当前权威候选） | 同四臂；2000 步；clean R=2²⁰；`lr=6e-4`、`warmup_constant(100)`；bf16 不 compile；seed 42，后续 43/44 | 注入点 gap/loss 曲线；频率 bin 双轴图；gap-vs-frequency；loss-gap-table-RMS 对齐 | 🟡 input 的 1000-step LR 筛选完成；y/v/nogram 对照运行中；全量队列待 optimizer gate |
| **剂量扫描** | M5 shard 大小扫描（dose） | train shards 0.25x–8x（12 档）；input 注入；2000 步 | gap vs dose（对数-对数幂律 α≈−2）；gap @2000 对比；epoch 分界标注 | ✅ 完成（v2 12 点齐） |
| **剂量扫描** | M5-v3 剂量重刷 | 同上 11 档；current-batch freq 口径 | 同 M5 + freq-bin current-batch | 🟡 三机队列运行中 |
| **剂量扫描** | M6 epoch 对齐批（e6，实际 5 epoch） | 对齐 epoch 数（5 epoch）；0.25x–3x | epoch 对齐 gap 曲线；nogram 对照 | ⚠️ 不完整：缺 4x–8x；命名 `_e6` 实为 5 epoch、无 LR schedule（勘误见 log §12） |
| **表优化器** | M3 Table 优化器 · 1x epoch | RMSProp/AdamW/SGD；LR 剂量（×1/×2/×4）；β₂（0.98/0.99）；1x epoch | 优化器对比 loss/gap；LR 剂量曲线；β₂ 曲线 | ⚠️ 完成但 β₂ 结论待修正（B2 bug 直击） |
| **表优化器** | M4 Table 优化器 · 2x epoch | 同 M3，epoch 拉长 | 1x vs 2x 对比 | ⚠️ 同上（`b2_099` 0.64→2.00，+210%） |
| **表优化器** | M3b 表学习率 × β₂ 消融深挖（附录） | 高表学习率体检；β₂×LR 消融 + 补点 | 附录报告 + figs（`docs/appendices/lr_beta_ablation/`） | 🟡 进行中 |
| **表优化器** | M7 短 epoch × β₂ | 0.25x/0.5x 短 epoch；β₂=0.99 | per-epoch 台阶清晰度曲线 | ⚠️ 完成但图未按 `_fixed` 重生成 |
| **因果干预** | Causal interventions（v2） | step 1000 干预：reset_e1 / reset_e2 / mask_readout / freeze_table / freeze_backbone | 干预前后 gap 对比（reset_e1 −75%、reset_e2 −98%、mask_readout −98%） | ✅ 完成（v2 波次） |
| **三轴 scaling** | S1 epoch 轴 · fixed-step | L1–L4 前缀（42/84/168/337 batches）× bigram/trigram/both/nogram；1000 步 | gap vs epoch 长度；scaling 报告 | ⚠️ 历史 compile 波次 48/48；no-compile 待重跑 |
| **三轴 scaling** | S1 epoch 轴 · fixed-epoch | 同 arms，6 epoch 对齐（L1=252…L4=2022 步） | 重播次数对齐下的 gap | ⚠️ 同上 |
| **三轴 scaling** | S1 table size 轴 | 旧：1M 逻辑地址 4 层架构（23+12×3 sizes）；**新：clean 单表 R 网格（64K–4M + perfect，13 点）** | 旧 34 点 K/N 曲线（锯齿）；**新 clean 单表 gap-R 曲线（光滑单调）**；相图（K/N vs gap）；perfect 零碰撞锚点；forking 图 | ✅ clean 单表网格已完成（新 SSOT）；旧框架标记历史 |
| **三轴 scaling** | S1 frequency 轴 | L4 + 1M × 4 arms；exact-frequency | G(E,f) 两因素模型检验 | ⚠️ 历史 compile 24/24；no-compile 待重跑 |
| **三轴 scaling** | S1 backbone safety | 长训 no-ngram（5000 步） | 无表 backbone 是否产生 gap（final +16.66，量级参考） | ✅ 完成（旧口径） |
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
| **table 统计** | per branch/layer/hash 的 physical rows R、逻辑地址 2R、distinct contexts K、occupancy、collision rate、singleton fraction、freq-weighted load | `table_occupancy.json`（`code/table_occupancy.py`） |
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
| **集群启动器** | `code/cluster/*.sh` | `run_rerun_v2.sh`（v2/v3 主线）、`run_injpos*.sh`、`run_table_opt*.sh`、`run_shard_sweep*.sh`、`run_scaling_*.sh`、`run_clean_table_grid.sh` |
| **受控数据干预** | `ngram5_freq_gap/`（data_gen.py / trainer.py） | 第四维度：动数据不动模型 |
| **作图脚本** | `docs/plot_scripts/`（canonical：`gen_all_figures.py`、`gen_sweep_v2_figs.py`、`gen_shard_sweep_figs.py`、`gen_epoch_aligned_figs.py` 等） | 图进 `docs/figs/` 按实验线分目录 |
| **toy 线** | `tasks/l1..l5/` | 独立自包含，纯 numpy/torch |

### 3.2 注入与训练原理（一句话版）

- **注入**：`input` 为主——`x = wte(idx) + wpe(pos) + Σ ngram_ve`，n-gram 向量直接加在 token 嵌入上（over-encoding，不走 attention）；`y` 注入在 attention 输出后加回 residual；`v` 注入在 attention 的 V 上加（路径最间接）。
- **表架构**：新 SSOT 是 **clean 单表** `nn.Embedding(R, 768)`，一个 context → 一行 → 一个完整向量，单层、单 hash（`--bigram_clean_table R`，R 任意；与 perfect-map 组合 = 零碰撞锚点）。旧 4 层求和 + 2-hash 拼接（`vocab×mult` 表）已降级为历史框架。
- **优化器**：表与 backbone 分开——表 RMSProp 无动量 `(0.0, 0.99)` 表 LR ×2（=0.008）；backbone AdamW。β₁=0 即无动量（`table_betas` bug 已修：`b2 = self.table_betas[1]`）。
- **训练**：`fixed` 模式 = 固定顺序 epoch replay（每轮从头重放同一 shard，L4=337 batches/epoch）；grad_accum 使 total batch = 147,456 tokens；bf16 autocast；默认不 compile。

### 3.3 来源是否唯一确定、可追溯

**结论：是，且是当前仓库的硬约束。** 依据：

1. **单一权威数据源**：只有 `data/runs_fixed/*_fixed/` 是合法数据；`data/runs/` 与不带后缀副本因 freq-bin 诊断 bug（复用训练迭代器）已作废并**彻底删除**（按「bug 不归档」原则）。两个已知 bug（freq-bin 白吃 train batch、`table_betas[1]` 被覆盖）已修复并有验证 smoke（`smoke_fixed_verify`）。
2. **口径变更必须新 run_id**（P2）：v2→v3 的 freq-bin train 侧改动即因此新起 `_v3` 后缀；`_fixed` 后缀标记「修复后」run，`_v3` 标记「current-batch 口径」run，命名即口径。
3. **代码即契约**：launcher（`code/cluster/*.sh`）显式传全部关键参数（β₂=0.99、×2、bf16、不 compile），不依赖默认值；跨机跑前 `md5sum` 核对 `code/train.py`、`code/ngram_freq.py`、`code/cluster/*.sh`（§4.3）。
4. **文档权威性分级**：`agents.md` §1（SSOT）→ `experiment-lines.md`（全景）→ `experiment-log.md`（登记簿，含每次口径变更与勘误）→ `claims-ledger.md`（断言台账）。任何结论必须带 run_id / step / seed。
5. **历史框架显式隔离**：clean-table 重做后旧 4 层架构标记 `[HISTORICAL 4-LAYER FRAMEWORK]`，与新旧数据并列但不可混用；current shell / Muon / RoPE 系结论一律 `[DEPRECATED SETTING]`。

**唯一留白**：`experiment-log.md` 中 v10 时代（pre-v2）数值尚未回填为 `_fixed`（T1/T2/T3 待办）；v3 波次运行中，回传后需用 current-batch 口径重绘 freq-bin 图。这两处是「可追溯性」目前仅有的未闭合项。
