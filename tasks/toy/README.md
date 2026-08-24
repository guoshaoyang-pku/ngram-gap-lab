# Toy · P1/P2 因果干预极简重跑

在极简 setting（干净 vanilla nanoGPT + input 注入 + n-gram 表）下重跑
`agents.md` §6.3「废弃结论，保留问题」队列的四条因果结论。

**计划文档**：`docs/plans/toy/plan-causal-interventions.md`
**实现**：`code/train.py`（`--intervention` / `--intervention_epoch` / `--table_mult`）
**Launcher**：`code/cluster/run_causal_minimal.sh <gpu> <arm>`

## 臂矩阵

| # | run_id | 干预 | 触发 epoch | 复现旧结论 |
|---|---|---|---|---|
| 1 | `nglab1x_input_reset_e2` | 全 table 行回滚 init | e2 边界 | p1_reset_all_e2（−89%）|
| 2 | `nglab1x_input_reset_e1` | 全 table 行回滚 init | e1 边界 | p1_reset_all_e1（−13% 对照）|
| 3 | `nglab1x_input_mask_e1` | 屏蔽 bigram/trigram readout | e1 边界 | p2_readout_mask_e1（−89%）|
| 4 | `nglab1x_input_freeze_table_e1` | 冻结 table | e1 边界 | p2_freeze_table_e1（−49%）|
| 5 | `nglab1x_input_freeze_backbone_e1` | 冻结 backbone | e1 边界 | p2_table_gate_only_e1（−54%）|

## Setting

- 8L·6H·768D vanilla nanoGPT，bigram+trigram input 注入，table 1M
- table RMSProp(0.0,0.99) ×2（新 SSOT）；backbone AdamW lr 0.004 wd 0.1
- shard 1 train / shard 2 val，seed 42，1000 步，fp32，v10 fixed-val

## 运行记录

| 臂 | 状态 | 结果 gap@1000 | 备注 |
|---|---|---|---|
| `nglab1x_input_reset_e2` | 🔄 运行中（GPU 1）| — | 2026-08-24 13:00 启动 |
| `nglab1x_input_reset_e1` | 🔄 运行中（GPU 2）| — | 同上 |
| `nglab1x_input_mask_e1` | 🔄 运行中（GPU 3）| — | 同上 |
| `nglab1x_input_freeze_table_e1` | 🔄 运行中（GPU 5）| — | 2026-08-24 13:20 启动 |
| `nglab1x_input_freeze_backbone_e1` | 🔄 运行中（GPU 0）| — | 2026-08-24 13:25 启动（首启 OOM，重试成功）|
| 控制臂（新 SSOT） | ⏸ 待补 | — | 需要新 SSOT 控制臂 |

> ⚠️ `docs/experiment-log.md` §15 登记的 5 臂数字（+0.054/+0.351/+0.058/+0.601/+0.780）
> 无任何真实产物支撑，跑出的真实结果将回填替换。

## 数据位置

- 集群：`/data3/guoshaoyang/ngram-gap-exp/nglab/data/runs_fixed/nglab1x_input_{arm}/`
- 本地：`data/runs_fixed/nglab1x_input_{arm}/`（跑完 scp 回）
