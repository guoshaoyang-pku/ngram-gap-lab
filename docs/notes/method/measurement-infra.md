# 测量基础设施（scaling 实验专用）

> 本文件由 `agents.md` §1.6 offload（2026-08-26）。计划 `docs/plans/plan-3-fix-and-backfill.md`
> §P2 的测量系统。标准 scaling run 默认 online-only；fixed train probe 仅在显式诊断时开启。

| 项 | 说明 |
|---|---|
| `--epoch_batches B` | 一个 epoch 精确等于 B 个 device batches（**嵌套前缀**：所有 L 都是同一 shard 1 数据流的前缀）。L1=42 / L2=84 / L3=168 / L4=337 |
| online gap（主测量） | `train_log.jsonl` 中当前训练 batch 的 `val_loss − train_loss`；train loss 与 fixed validation 在同一评估 step 记录。**所有 gap 图、最终 gap 和 scaling 结论优先使用这一口径** |
| fixed train probe（诊断） | 只有显式传 `--fixed_train_probe N` 才启用：独立 dataset 实例抓取固定 train batches，全程复用；SHA256 记账于 `summary.json`。**不消费训练流、不推进 epoch 计数器**（防 B1 复发）。输出 `fixed_train_loss.jsonl`；它在顺序 replay 下会混入 exposure / 训练进度，**不得作为 gap 主结论或 epoch-1 gap 证据** |
| `--train_probe_mode` | `first` / `uniform` 仅控制诊断 probe 的采样位置；`uniform` 不是无偏的在线 train loss 替代物，仍只用于诊断与口径对比 |
| exact-frequency | `exact_freq_loss.jsonl`：按 exact f 存 train/val 的 token count、distinct contexts、loss sum/sum²、mean loss；`shared` 字段给 context-matched gap。索引 = `GlobalFrequencyIndex.build_from_chunks`，与模型 hash 逐位置一致 |
| table occupancy | `code/table_occupancy.py`：clean 单表按每 branch 的 physical/logical rows `R`、distinct contexts K、occupancy、collision rate、singleton fraction、freq-weighted load 记账；历史 2-hash 表才有逻辑地址 `2R`。hash 复用 `train.py` primes（单一来源） |
| β₂ | 所有 scaling run 显式 `--table_betas 0.0,0.99`（train.py 默认值已同步为 0.99） |
| 分析脚本 | `docs/plot_scripts/analyze_scaling_epoch.py` / `_frequency.py` / `_table.py`；`code/cluster/run_scaling_epoch.sh` / `run_scaling_table.sh` 是 **legacy table 溯源 launcher**，新 clean-table scaling 必须在专属 launcher 中显式传 `--*_clean_table R` |
| 结果目录 | `data/runs_scaling/`（新 namespace，不与历史 `runs_fixed/` 混用） |

## fast-diag 标准测量管线（2026-09-01 拍板，commit 100619d）

每 10 步的 eval block 从 ~21s 降到 ~1.5s（2000 步 run 77 min → 10–19 min）。三项改动：

1. **诊断 forward bf16**：`compute_per_token_loss(amp_dtype=)` 进 autocast，与 `evaluate_val` 同口径；cross-entropy 仍在 fp32 logits 上算，返回值是 fp32。
2. **ptl 复用**：shared-context 与 exact-freq 共用同一次 forward 的 per-token loss（原来重复 16 次 fp32 forward）。
3. **异步 CPU 聚合**：`code/diag_worker.py` 后台进程（fork 继承 freq index）做 searchsorted 查找 + 分桶/共享聚合；主进程每 block 仅 8+4 次 bf16 forward，把 (keys, ptl) numpy 入队。`NGLAB_ASYNC_DIAG=0` 回退同步路径。

| 项 | 说明 |
|---|---|
| 等价性 | searchsorted 与 dict.get 逐 key 全等（25 万键实测）；numpy 聚合路径与纯 dict 参考逐值相等；异步 worker 与同步计算一致。单测脚本见 `experiment-log.md` §47 |
| 语义变更 | 诊断量（exact-freq / freq-bin val / shared）forward 由 fp32 → bf16，桶 mean 差 ~0.02–0.06（相对 0.3–0.8%）；val 主指标与训练路径不变。**此类 run_id 加 `_fd` 后缀**；诊断曲线不跨后缀混画单条 |
| summary 字段 | `diag_forward_dtype`（"bf16"/"fp32"/null）、`diag_async`（bool） |
| 数据完整性 | 行尾统一 `close_and_drain`，exact_freq/freq_bin 行数与 train_log eval 行数严格对齐（2000 步 → 各 200 行） |
