# Toy Plan · P1/P2 因果干预极简重跑

> **主题**：在极简 setting（干净 vanilla nanoGPT + input 注入 + n-gram 表）下重跑
> `agents.md` §6.3「废弃结论，保留问题」队列的四条因果结论。
> 旧结论建立在 current-shell 污染 setting 上（见 `docs/_archive/docs/p12-causal-results.md`），
> 数字已废弃；这里在干净 setting 下重做，回答「gap 的产生与传导是否依赖 n-gram 表」。
>
> **状态**：📋 **部分执行中**（3/5 臂在集群运行），本文档为独立 plan，不自动触发实验。
> **与 plan-4 的关系**：plan-4 A4 用 2000 步 + `nglab1x_*_v2` 命名重刷同样 5 臂；
> 本 toy plan 用 1000 步 + `nglab1x_input_*` 命名（与 §15 登记口径一致）。
> **两份计划并行、互不替代**——本 toy plan 先出结果验证机制，plan-4 A4 走全量 v2 口径。

---

## 1. 目的

- 在极简 setting 下重跑四条因果干预，验证旧结论是否成立：
  - 全 table 行回滚（reset_table）→ 旧 −89%
  - 屏蔽 bigram/trigram readout（mask_readout）→ 旧 −89%
  - 冻结 table（freeze_table）→ 旧 −49%
  - 冻结 backbone（freeze_backbone）→ 旧 −54%
- 确认「gap 的产生与传导完全依赖 n-gram 表」（回滚/屏蔽 → 塌缩到 nogram 量级）。
- 全部为**内部对照**：同 setting 下，干预 vs 不干预的相对 gap 变化，与全局 SSOT 解耦。

## 2. Setting（与控制臂严格一致）

| 项 | 值 |
|---|---|
| 模型 | 8L · 6H · 768D vanilla nanoGPT（LayerNorm、learned abs，无 Muon/RoPE/RMSNorm） |
| 注入 | bigram + trigram input 注入，table 1M |
| 优化器 | table RMSProp(0.0,0.99) · ×2（**新 SSOT**，commit b54dd34）；backbone AdamW(0.8,0.95) lr 0.004 wd 0.1 |
| 数据 | shard 1 train / shard 2 val（fixed 顺序，seed 42） |
| 步数 | 1000 步（batch 72，total 147456） |
| 评估 | val 每 10 步 4 batch（v10 fixed-val） |
| dtype | fp32（与唯一真实控制臂 `vanilla_input_1000_seed42` 一致） |

> **⚠️ 控制臂口径问题**：唯一有真实产物的控制臂 `vanilla_input_1000_seed42`
> （+0.858）是**旧 SSOT（β₂=0.999/×1, fp32）**跑的；新 SSOT（0.99/×2）下控制臂
> 尚无真实产物。本 toy plan 的干预臂用新 SSOT，因此**需要补跑一个新 SSOT 控制臂**
> 才能做干净的相对对照（见 §4 待决问题）。

## 3. 干预臂矩阵（5 臂）

| # | run_id | 干预 | 触发 epoch | 复现旧结论 |
|---|---|---|---|---|
| 1 | `nglab1x_input_reset_e2` | 全 table 行回滚 init | e2 边界（~step674） | p1_reset_all_e2（−89%）|
| 2 | `nglab1x_input_reset_e1` | 全 table 行回滚 init | e1 边界（~step337） | p1_reset_all_e1（−13% 对照）|
| 3 | `nglab1x_input_mask_e1` | 屏蔽 bigram/trigram readout | e1 边界 | p2_readout_mask_e1（−89%）|
| 4 | `nglab1x_input_freeze_table_e1` | 冻结 table（保留 e1 内容）| e1 边界 | p2_freeze_table_e1（−49%）|
| 5 | `nglab1x_input_freeze_backbone_e1` | 冻结 backbone（仅 table 更新）| e1 边界 | p2_table_gate_only_e1（−54%）|

干预在 epoch 边界一次性触发（`--intervention_epoch`，0-indexed）。
实现：`code/train.py --intervention <type> --intervention_epoch <n>`（`apply_intervention()` 一次性闩锁）。
Launcher：`code/cluster/run_causal_minimal.sh <gpu> <arm>`。

## 4. 执行状态

| 臂 | 状态 | GPU | 备注 |
|---|---|---|---|
| `nglab1x_input_reset_e2` | 🔄 运行中 | 1 | 新 SSOT 0.99/×2 fp32 |
| `nglab1x_input_reset_e1` | 🔄 运行中 | 2 | 同上 |
| `nglab1x_input_mask_e1` | 🔄 运行中 | 3 | 同上，已过 e1 干预点 |
| `nglab1x_input_freeze_table_e1` | ⏸ 未启动 | — | 等 GPU 释放 |
| `nglab1x_input_freeze_backbone_e1` | ⏸ 未启动 | — | 等 GPU 释放 |
| 控制臂（新 SSOT） | ⏸ 未跑 | — | **待补**，见下 |

### 待决问题

1. **控制臂**：新 SSOT（0.99/×2）控制臂需要补跑（`nglab1x_input_v2` 同 setting 1000 步，
   或直接复用 plan-4 A1#1 `nglab1x_input_v2` 的前 1000 步）。否则干预臂只能对照
   旧 SSOT 的 +0.858（口径不一致，结论定性可用、定量有偏差）。
2. **§15 登记数字无产物支撑**：`docs/experiment-log.md` §15 表格里的 5 个臂
   （+0.054 / +0.351 / +0.058 / +0.601 / +0.780，标记 ✅ done 2026-08-24）
   **没有任何真实产物**（本地+集群均无 summary/train_log）。本 toy plan 跑出的真实结果
   将回填替换 §15。
3. **dtype**：控制臂是 fp32；若后续正式实验切 bf16（§16 声称但无产物），干预臂需同步切换。

## 5. 产物与验证

- 每个臂：`data/runs_fixed/nglab1x_input_{arm}/` 下 `train_log.jsonl` + `summary.json` + `train.log`
- 分析：6 臂 gap@1000 对比表 + 干预前后曲线（`code/plot_dtype_causal.py` 参考）
- 验证标准：
  - reset_e2 / mask_e1 → gap 塌缩到 nogram 量级（0.04~0.06）→ 复现 −89%
  - freeze_table → 中等降幅（旧 −49%）
  - freeze_backbone → 本 toy setting 预期弱于旧的 −54%（无 gate/reader 额外放大器）
- 回填：`docs/experiment-log.md` §15（真实数字替换登记值）

## 6. 不做

- 不改 `code/`（train.py 已含干预实现）
- 不动旧 `_fixed` / `data/runs/` 数据
- 不启动 plan-4 的 2000 步 v2 全量重刷（由 plan-4 自己管理）
- 不跑 current-shell / Muon / RoPE 相关任何实验
