# Plan 4 · 新标准全量重刷清单（供 review，不自动启动）

> **新标准（2026-08-24 拍板）**：β₂=0.99（无动量）· 表学习率 ×2（表实际 0.008）。
> **状态**：📋 **待用户 review**。确认后才由执行 agent 启动，本文档不自动触发任何实验。
> **执行器**：`code/cluster/run_rerun_v2.sh`（所有参数显式，不依赖默认值）。
> **与其他工作的关系**：`tasks/s1_scaling_three_axis` 是**独立的 scaling 测量实验线**
> （冻结在旧默认 β₂=0.99 · ×1），与本重刷并行、互不干扰，见其自身文档。

---

## 0. 重刷原则

1. **只重刷「新标准下会变化」的实验**——即带 n-gram 表、且旧配置是 β₂=0.999 或表学习率 ×1 的。
2. **不重刷**：无表对照（表参数无关，仍补 1 个同批对照）、表优化器消融中 β₂≠0.99 或 ×1 以外的扫描点（消融变量本身就是那些值）、已在跑的表大小扫描。
3. 新 run 统一 `..._v2` 后缀，写入 `data/runs_fixed/`；旧 `_fixed` 保留作历史对照，**不删不改**。
4. 所有重刷默认：seed 42，input 注入，2000 步，val 每 10 步，freq-bin 每 10 步，`--table_betas 0.0,0.99 --table_lr_scale 2.0`。

## 1. 重刷清单（Group A · 必做，33 个）

### A1 · 注入点消融（新标准基线，4 个）

| # | 新 run_id | 注入点 | train shards | val shards | steps | 旧对照 |
|---|---|---|---|---|---|---|
| 1 | `nglab1x_input_v2` | input | 1 | 2,3,4,5,6,7,8,9,10,6542 | 2000 | `nglab1x_v10_input_fixed`(1.87) |
| 2 | `nglab1x_y_v2` | y | 1 | 同上 | 2000 | `nglab1x_v10_y_fixed`(5.80) |
| 3 | `nglab1x_v_v2` | v | 1 | 同上 | 2000 | `nglab1x_v10_v_fixed`(5.45) |
| 4 | `nglab1x_nogram_v2` | input（关表）| 1 | 同上 | 2000 | `nglab1x_v10_nogram_fixed`(0.25) |

> nogram 理论上不依赖表参数，重刷一次是为了与新标准三臂同批可比。
> 命令：`run_rerun_v2.sh <gpu> <id> <train> <val> 2000 [--injection_position y/v]`，
> nogram 追加 `--enable_bigram 0 --enable_trigram 0`。

### A2 · shard 剂量扫描（8 个，2000 步）

| # | 新 run_id | train shards | val shards | 旧对照 |
|---|---|---|---|---|
| 5 | `nglab0_25x_input_v2` | 62 | 2,3,4,5,6,7,8,9,10,6542 | `nglab0_25x_input_fv_fixed`(9.69) |
| 6 | `nglab0_5x_input_v2` | 60 | 同上 | `nglab0_5x_input_fv_fixed`(3.99) |
| 7 | `nglab0_75x_input_v2` | 63 | 同上 | `nglab0_75x_input_fv_fixed`(2.44) |
| 8 | `nglab1_5x_input_v2` | 1,61 | 3,4,5,6,7,8,9,10,6542 | `nglab1_5x_input_fv_fixed`(0.97) |
| 9 | `nglab2x_input_v2` | 1,2 | 3,4,5,6,7,8,9,10,6542 | `nglab2x_input_v10_fv_fixed`(0.58) |
| 10 | `nglab2_5x_input_v2` | 1,2,64 | 4,5,6,7,8,9,10,6542 | `nglab2_5x_input_fv_v3_fixed`(0.81) |
| 11 | `nglab3x_input_v2` | 1,2,3 | 4,5,6,7,8,9,10,6542 | `nglab3x_input_fv_v3_fixed`(0.83) |
| 12 | `nglab4x_input_v2` | 1,2,3,4 | 5,6,7,8,9,10,6542 | `nglab4x_input_fv_v3_fixed`(1.62) |

### A3 · e6 对齐批（9 个，步数 = 旧值精确复现）

| # | 新 run_id | train shards | val shards | steps | 旧对照 |
|---|---|---|---|---|---|
| 13 | `nglab0_25x_e6_v2` | 62 | 2,3,4,5,6,7,8,9,10,6542 | 420 | `nglab0_25x_e6_fixed`(1.56) |
| 14 | `nglab0_5x_e6_v2` | 60 | 同上 | 840 | `nglab0_5x_e6_fixed`(1.47) |
| 15 | `nglab0_75x_e6_v2` | 63 | 同上 | 1260 | `nglab0_75x_e6_fixed`(2.68) |
| 16 | `nglab1x_e6_v2` | 1 | 同上 | 1685 | `nglab1x_e6_fixed`(1.39) |
| 17 | `nglab1_5x_e6_v2` | 1,61 | 3,4,5,6,7,8,9,10,6542 | 2525 | `nglab1_5x_e6_fixed`(4.33) |
| 18 | `nglab2x_e6_v2` | 1,2 | 3,4,5,6,7,8,9,10,6542 | 3350 | `nglab2x_e6_fixed`(1.43) |
| 19 | `nglab2_5x_e6_v2` | 1,2,64 | 4,5,6,7,8,9,10,6542 | 4190 | `nglab2_5x_e6_fixed`(1.38) |
| 20 | `nglab3x_e6_v2` | 1,2,3 | 4,5,6,7,8,9,10,6542 | 5000 | `nglab3x_e6_fixed`(2.99) |
| 21 | `nglab4x_e6_v2` | 1,2,3,4 | 5,6,7,8,9,10,6542 | 5000 | **新点**（补 M6 缺口）|

### A4 · 短 epoch × β₂ 家族（2 个，2000 步）

| # | 新 run_id | train shards | val shards | 旧对照 |
|---|---|---|---|---|
| 22 | `nglab025x_v2` | 62 | 2,3,4,5,6,7,8,9,10,6542 | `nglab025x_b2_099_fixed`(8.97, 旧×1) |
| 23 | `nglab05x_v2` | 60 | 同上 | `nglab05x_b2_099_fixed`(4.02, 旧×1) |

### A5 · 大 shard 端（3 个，2000 步）

| # | 新 run_id | train shards | val shards | 旧对照 |
|---|---|---|---|---|
| 24 | `nglab5x_input_v2` | 1,2,3,4,5 | 6,7,8,9,10,6542 | `nglab5x_input_fv_fixed`(0.01) |
| 25 | `nglab6x_input_v2` | 1,2,3,4,5,6 | 7,8,9,10,6542 | `nglab6x_input_fv_fixed`(−0.09) |
| 26 | `nglab8x_input_v2` | 1,2,3,4,5,6,7,8 | 9,10,6542 | `nglab8x_input_fv_fixed`(−0.07) |

### A6 · 因果干预五臂（5 个，1000 步，新标准重刷）

| # | 新 run_id | 干预 | 触发 | 说明 |
|---|---|---|---|---|
| 27 | `nglab1x_reset_e1_v2` | reset_table | epoch 1（0-indexed）开始 | 擦表 |
| 28 | `nglab1x_reset_e2_v2` | reset_table | epoch 2 开始 | 擦表 |
| 29 | `nglab1x_mask_e1_v2` | mask_readout | epoch 1 开始 | 屏蔽读出 |
| 30 | `nglab1x_freeze_table_e1_v2` | freeze_table | epoch 1 开始 | 冻结表 |
| 31 | `nglab1x_freeze_backbone_e1_v2` | freeze_backbone | epoch 1 开始 | 冻结 backbone |

> 追加参数 `--intervention <type> --intervention_epoch <n>`；控制臂 = #1（`nglab1x_input_v2`）。

### A7 · 固定 train probe（ρ 测量，2 个，2000 步）

| # | 新 run_id | train shards | 说明 |
|---|---|---|---|
| 32 | `nglab1x_input_rho_v2` | 1 | `--fixed_train_probe 4 --probe_eval_interval 10` |
| 33 | `nglab2x_input_rho_v2` | 1,2 | 同上 |

> 两因素模型验证入口（plan-3 T9）；probe 用独立迭代器，不消费训练流（防 B1 复发）。

## 2. 补充项（Group B · 建议做，5 个）

| # | 新 run_id | 内容 | steps |
|---|---|---|---|
| 34 | `nglab1x_input_v2_s43` | #1 的 seed 43 复现 | 2000 |
| 35 | `nglab1x_input_v2_s44` | #1 的 seed 44 复现 | 2000 |
| 36 | `nglab1x_nogram_long_v2` | **长 no-ngram 基线**（缩小数据量前的归因保险，plan-3 T8）| **8000** |
| 37 | `nglab1x_table_mult_128_v2` | 表大小扫描补 128（×2 口径，若需要）| 1000 |
| 38 | `nglab1x_table_mult_256_v2` | 表大小扫描补 256（同上）| 1000 |

> #34/35 给主线设置误差棒（现在所有结论都是单 seed）。
> #36 若之后要做「缩小单 epoch 数据量」，这是必需的归因基线。

## 3. 不重刷（Group C · 明确排除）

| 家族 | 为什么不重刷 |
|---|---|
| 表优化器消融（adamw/sgd、β₂=0.98/0.9999/0.99999、×4 扫描）| 消融变量本身就是那些值；附录已结论「×2 以上不健康」 |
| 已在集群跑的表大小扫描（M=16/32 等）| 正在跑；注意它冻结在 ×1，是否要 ×2 版本见待决问题 2 |
| `s1_scaling_three_axis` 全部 | 独立实验线，冻结在旧默认，由自己的计划管理 |
| 附录补点（`nglab{1,2}x_opt_rmsprop_b2_099_lr1`）| 正在跑，跑完作为 ×1 对照归档 |

## 4. 资源与排期估算

- **Group A**：33 个 run ≈ **42 GPU-小时**（26 个 2000 步 + 5 个 1000 步 + e6 平均 ~2200 步）。
- **Group B**：5 个 ≈ **14 GPU-小时**（含 1 个 8000 步长跑 ≈5.5 小时）。
- 合计 ≈ **56 GPU-小时**。360-1 全空（8 卡），A 约 6 小时，A+B 约半天。
- 调度建议：A1+A2+A6+A7 优先（结论主干）→ A3 → A4/A5。

## 5. 执行方式（给执行 agent）

1. **先同步代码**：`rsync code/train.py code/cluster/run_rerun_v2.sh` 到目标机并 `md5sum` 核对。
2. 每个 run 一条命令，例如：
   ```bash
   bash code/cluster/run_rerun_v2.sh 0 nglab1x_input_v2 1 2,3,4,5,6,7,8,9,10,6542 2000
   ```
   因果臂追加 `--intervention reset_table --intervention_epoch 2`；
   probe 臂追加 `--fixed_train_probe 4 --probe_eval_interval 10`。
3. **跑中**：每 30 分钟检查存活（`ps` + `train.log tail`）；失败的记录原因，不盲目重跑。
4. **跑完**：逐条核对 `summary.json`（`table_betas=[0,0.99]`、`table_lr_scale=2.0`），
   回填 `docs/experiment-log.md` 新 section「v2 波次」，并在 `experiment-lines.md` 标状态。
5. **不做**：不改 `code/`，不动旧 `_fixed` 数据，不启动 `s1_scaling` 相关。

## 6. 待用户确认的问题

1. **A3 的 e6 步数**精确复现旧值（420/1685/2525…）——确认还是取整？（建议精确复现，保证同口径可比。）
2. **表大小扫描**当前冻结在 ×1。新标准定为 ×2 后，是否要**另起一轮 ×2 的表大小扫描**？
3. **A7 的 ρ 测量**要不要加 per-frequency-bin 版本？
4. Group B #36 的长 no-ngram（8000 步）确认纳入？
5. 5x/6x/8x 的 **e6 点**（旧数据也没有）要不要一并补？
