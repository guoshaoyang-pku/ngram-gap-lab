# Plan 5 · S1 三轴 scaling 正式实验交接

> **目的**：在唯一极简 setting 下，正式验证 epoch length、exact context
> frequency、table size 三条 scaling。  
> **交接对象**：`ngram-gap-lab` 中的 agents。  
> **当前状态**：seed 42 正式 full grid 与 **seed 43/44 三 seed 复现（epoch /
> table / frequency）均已完成**，261/261 个 run 通过 QC；H1–H4 猜想检验已
> 回填（附录报告 §7）：ΔG 方向 seed-stable、trigram table 幂律无饱和、
> 两因素 β 可辨识（cv 4–13%）、模块交互不可合并。仍缺 frequency 的
> epoch-dependent fit 与最终 scaling 定律提升。
> **前置提交**：`1f59ffc fix: decouple frequency diagnostics from fixed probes`

## 1. 先看哪里：极简报告与当前证据

极简主报告：

- `docs/report/index.html`
- 浏览器本地入口：`file:///Users/guoshaoyang/Desktop/workdir/ngram-gap-lab/docs/report/index.html`

三轴 scaling 附录：

- `docs/appendices/s1_scaling_three_axis/report.md`
- `docs/appendices/s1_scaling_three_axis/report.html`
- 图片目录：`docs/appendices/s1_scaling_three_axis/figs/`
- 任务目录：`tasks/s1_scaling_three_axis/`

本地当前有 15 个结果目录：

- 8 个历史 `pilot_*` / safety 目录，只作溯源；
- 7 个当前标准 `basic_*` 锚点，seed 42；
- 后续正式结果必须进入带 `_fixed` 后缀的目录，不能混用历史目录。

当前 basic 锚点的主要观察：

| 轴 | 当前结果 | 只能支持的结论 |
|---|---|---|
| epoch | L1/both `+2.385`，L4/both `+2.341`；no-ngram `+0.085/-0.008` | 新标准下 1000 步已出现明显 forking；L1/L4 还不能宣称有 scaling |
| table | 1M/bigram `+0.801`，16K/bigram `+0.016`，1M/trigram `+0.815` | table 变小后 gap 显著下降，方向支持 collision/locality 候选解释 |
| frequency | both 与 1M bigram 的 gap 随 exact `f` 下降 | 仅 observational consistency，不是 `f` 的因果证明 |
| backbone | L1 no-ngram 5000-step safety 最近到 4000 步，gap `+13.236` | 不能把 backbone-only gap 假设为恒为零 |

这些数值都来自单 seed、1000-step 基础 QC，不是最终 scaling 定律。

## 2. SSOT：所有新 run 的冻结 setting

除当前实验变量外，必须保持：

- vanilla nanoGPT：8L / 6H / 768D；
- learned absolute position、LayerNorm、tied embedding；
- n-gram 为 `input` / wte over-encoding；
- 模块臂：`bigram-only`、`trigram-only`、`both`、`no-ngram`；
- table 默认 1M logical addresses：每个 n-gram、每层两个 hash 总计
  `2R = 1,048,576`，单 hash physical rows `R = 524,288`；
- table optimizer：RMSProp，无动量，显式
  `--table_betas 0.0,0.99`；
- backbone optimizer：现有 AdamW `(0.8,0.95)`；
- natural corpus，train / val shard 严格不重叠；
- seed 42 先跑，之后 seed 43/44 复现；
- bf16 autocast + `torch.compile`；
- 主 gap：同一 fixed probe 上的 `fixed_val_loss - fixed_train_loss`；
- 不得引入 current shell、Muon、RoPE、RMSNorm、fourgram 或额外架构变体。

新代码已将评估 cadence 分开：

- fixed probe：基础 QC 每 25 步；正式网格按正式 contract；
- wide frequency bucket：每 100 步；
- exact-frequency：每 100 步；
- epoch 边界和最终 step 额外触发；
- exact-frequency 不得再绑定在 fixed-probe 的每次触发上。

本次 cadence 修正已写入：

- `code/train.py`
- `tasks/s1_scaling_three_axis/code/train.py`
- `tasks/s1_scaling_three_axis/launchers/run_scaling_basic.sh`
- `tasks/s1_scaling_three_axis/launchers/run_scaling_basic_table.sh`

每个新 run 的 `summary.json` 必须包含：

- `table_betas = [0.0, 0.99]`；
- `table_lr_scale`；
- `compute_dtype = "bf16"`；
- `torch_compile = true`；
- `fixed_train_probe_sha256`；
- `epoch_batches`；
- `exact_freq_eval_interval`；
- 实际 git commit 或等价代码指纹。

## 3. 执行顺序

### P0 · 先完成安全检查

1. 检查 5000-step backbone safety 是否完成；同步最终
   `summary.json` 和 `fixed_train_loss.jsonl`。
2. 不因已有 `+13.236` 预设最终结论；先确认是否 NaN、是否 OOM、是否
   训练稳定，以及该 run 的 setting / cadence 是否属于当前标准。
3. 运行：

```bash
python tasks/s1_scaling_three_axis/analysis/test_scaling_measurement.py
python -m py_compile \
  tasks/s1_scaling_three_axis/code/train.py \
  tasks/s1_scaling_three_axis/code/ngram_freq.py \
  tasks/s1_scaling_three_axis/code/table_occupancy.py
```

4. 任何新 run 开始前，登记到 `docs/experiment-log.md` 和
   `docs/experiment-lines.md`，记录 run id、机器、GPU、setting、数据 prefix、
   seed 和目标。

### P1 · Epoch length：两种对齐都跑

使用 shard 1 的嵌套前缀：

| label | `epoch_batches` | 相对长度 |
|---|---:|---:|
| L1 | 42 | 1/8 |
| L2 | 84 | 1/4 |
| L3 | 168 | 1/2 |
| L4 | 337 | 1 |

固定步数：

- 1000 steps；
- L1–L4；
- `bigram-only`、`trigram-only`、`both`、`no-ngram`；
- step-anchored LR；
- 主量为 `ΔG_module = G_module - G_no-ngram`。

固定 epoch：

- 训练 6 个完整 epoch；
- 读取 epoch 3 与 epoch 6 截面；
- `--lr_schedule_epochs 6`；
- 所有 L 使用同一 epoch-anchored LR 轨迹；
- 保留真实的 epoch boundary，不用近似 step 替代。

分析要求：

- 同时画 `gap vs step` 和 `gap vs epoch`；
- fixed-step 与 fixed-epoch 分开；
- 不只报告 aggregate raw gap，必须报告 no-ngram-adjusted `ΔG`；
- 若所有 `ΔG > 0`，才拟合有效幂律；否则保留原空间趋势与单调性；
- 至少 seed 42/43/44 后再写 scaling 结论。

### P2 · Exact-frequency：先做形状，再做拟合

优先使用 epoch grid 的 L4、1M table：

- `bigram-only`、`trigram-only`、`both`、`no-ngram`；
- fixed-step 和 fixed-epoch 的预注册截面；
- exact `f`，禁止用宽 bucket 几何中点替代。

纳入标准：

- train / val 都有该 exact `f`；
- 每个 split 至少 1,024 tokens；
- 至少 32 个 distinct contexts；
- 不满足标准的 `f` 与原因写入 fit manifest；
- `f=0` 只报告 val loss，不定义 gap；
- token-marginal 与 context-matched gap 分开报告。

拟合：

```text
G_l,E(f) = A_l f^(-beta_l)
           [1 - exp(-c_l,E f^(gamma_l))]
```

比较：

- M0：纯有效幂律；
- M1：`c_l,E = E * c_l`；
- M2：每个 epoch 自由 `c_l,E`；
- 原 gap 空间的加权 robust fit 为主；
- log-log 只作可视化；
- 报告 profile likelihood / 参数可辨识性；
- 自然语料下最终措辞只能是 observational consistency。

`both` 不得强行拟合单一频率公式：

- 分别按 `f_bigram` / `f_trigram` 做 marginal；
- 计算 `I = ΔG_both - ΔG_bigram - ΔG_trigram`；
- interaction 显著时，增加交互项或明确停止单公式解释。

### P3 · Table size：只向更小规模

横轴固定为每个 n-gram、每层两个 hash 的总 logical addresses `2R`：

| `table_mult` | physical `R` | logical `2R` |
|---:|---:|---:|
| 64 | 524,288 | 1,048,576 |
| 56 | 458,752 | 917,504 |
| 48 | 393,216 | 786,432 |
| 40 | 327,680 | 655,360 |
| 36 | 294,912 | 589,824 |
| 32 | 262,144 | 524,288 |
| 28 | 229,376 | 458,752 |
| 24 | 196,608 | 393,216 |
| 20 | 163,840 | 327,680 |
| 18 | 147,456 | 294,912 |
| 16 | 131,072 | 262,144 |
| 14 | 114,688 | 229,376 |
| 12 | 98,304 | 196,608 |
| 10 | 81,920 | 163,840 |
| 9 | 73,728 | 147,456 |
| 8 | 65,536 | 131,072 |
| 7 | 57,344 | 114,688 |
| 6 | 49,152 | 98,304 |
| 5 | 40,960 | 81,920 |
| 4 | 32,768 | 65,536 |
| 3 | 24,576 | 49,152 |
| 2 | 16,384 | 32,768 |
| 1 | 8,192 | 16,384 |

训练：

- L4；
- 1000 steps；
- bigram-only / trigram-only / both；
- seed 42 后补 seed 43/44；
- 1M 默认点复用合格的 epoch-grid L4 run；
- no-ngram 共享基线，不随 table size 重跑。

每个 table run 的正式测量产物包括：

- `table_occupancy.json`；
- fixed train/validation probe 日志；原始 7 个 table size 的 21 个 dense run
  每 10 步记录，另 37 个加密 sparse run 只在最终 step 1000 记录；exact-frequency 只在
  独立 frequency 轴 run 中记录；
- 实际参数量、physical rows、logical addresses；
- collision rate、singleton fraction、mean/p95 co-occupants；
- frequency-weighted row load。

比较：

1. `ΔG vs logical addresses`；
2. `ΔG vs measured collision metric`；
3. `ΔG vs singleton context fraction`；
4. `ΔG(R,f)`；
5. bigram / trigram / both 分面；
6. power-law、collision-aware saturation、非参数单调/spline；
7. seed-bootstrap 预测误差与信息准则。

不要只依据理论 `K/R`；row hash 必须复用模型实现，并由
`test_occupancy_hash_matches_model_hash` 验证。

## 4. QC gate 与停止条件

正式 full grid 只有在以下条件均满足后才能用于报告：

- iterator、epoch counter、fixed probe 不互相污染；
- fixed probe token hash 跨 run 一致；
- train / val 零重叠；
- beta2 实际读取 `table_betas[1]`；
- frequency 轴 run 的 exact-frequency key 与模型逐位置一致；
- table occupancy 的 numpy / torch hash 一致；
- 无 NaN/OOM/异常 loss；
- 每个实验变量有对应 no-ngram 或共享 baseline；
- 至少 seed 42/43/44；
- 所有 run 有 manifest、代码 commit、机器/GPU、数据 prefix。

### seed 42 正式网格回填（2026-08-24）

- epoch full grid：32/32 完成；`epoch_batches` 为 42/84/168/337；
- table full grid + 两轮加密取点：69/69 完成，覆盖
  `table_mult=64,56,48,40,36,32,28,24,20,18,16,14,12,10,9,8,7,6,5,4,3,2,1`；
- frequency axis：8/8 完成，exact-frequency 每 100 步；
- `bb_safety_L1_nogram_5000` 完成，final fixed gap 为 +16.66 @5000，
  但该 run 是旧 cadence + fp32/no compile，只作量级参考；
- 正式 109 个 `_fixed` run 的统一 QC 全部通过：无 NaN/坏行；dense run 为
  10 步 validation/probe cadence，48 个 table sparse run 只在最终 step
  1000 监测；bf16 + compile，probe hash 一致；
- 三轴图和摘要位于 `docs/appendices/s1_scaling_three_axis/figs/`，
  频率拟合和排除 manifest 位于 `figs/fit_manifest.json`；
- **三 seed 复现完成（2026-08-25）**：seed 43/44 的 epoch（32×2）、table
  加密（36×2）、frequency（8×2）共 152 个新 run 全部完成并通过 QC，三 seed
  正式 run 合计 261 个；H1–H4 判定见附录报告 §7 与实验登记 §17.2；
- **尚未完成**：frequency 的 epoch-dependent fit（epoch 3/6 截面）、跨 seed
  profile-likelihood，以及把三轴结果提升为主报告 scaling 定律。

### table 加密取点（2026-08-24）

table 原始 7 个规模只有 7 个横坐标，双对数图难以判断中间形状。为保持
最终 gap 口径不变，同时避免重复计算完整训练曲线，第一轮新增
`table_mult=48,24,12,6,3` 五个规模；每个规模跑 bigram/trigram/both 三个
module，共 15 个 run。第二轮继续新增
`table_mult=56,40,36,28,20,18,14,10,9,7,5` 十一个规模，
每个只跑 bigram/trigram，共 22 个 run。两轮均使用 `MONITOR=sparse`：
`--val_steps 1000 --probe_eval_interval 1000 --table_norm_interval 1000`，
因此每个 run 只保存最终 fixed train/val/gap 和 occupancy，不把中间点补画
成曲线。原始 21 个 dense run 与两轮合计 48 个 sparse run 合并为 69 个最终
table 点（bigram/trigram/both 各 23 点）；图 `table_gap_vs_2R.png`
和 `table_gap_vs_collision.png` 均使用这 69 个点。

若 QC 不通过：

1. 停止扩展网格；
2. 在 `docs/experiment-log.md` 记录失败原因；
3. 新建修正版 run id，不覆盖旧结果；
4. 更新本 plan 的状态和报告 claim boundary。

## 5. 交付物

代码：

- `tasks/s1_scaling_three_axis/code/train.py`
- `tasks/s1_scaling_three_axis/code/ngram_freq.py`
- `tasks/s1_scaling_three_axis/code/table_occupancy.py`
- `tasks/s1_scaling_three_axis/launchers/`
- `tasks/s1_scaling_three_axis/analysis/`

结果：

- `data/runs_scaling/<run_id>_fixed/`
- `summary.json`
- `fixed_train_loss.jsonl`
- `exact_freq_loss.jsonl`
- `table_occupancy.json`
- run manifest 与 probe hash

分析：

- `docs/plot_scripts/analyze_scaling_epoch.py`
- `docs/plot_scripts/analyze_scaling_frequency.py`
- `docs/plot_scripts/analyze_scaling_table.py`
- 每个图一条可重建命令；
- 所有排除的 frequency bin 与原因落入 manifest。

文档：

- 回填 `docs/appendices/s1_scaling_three_axis/report.md`；
- 更新 `docs/experiment-log.md`；
- 更新 `docs/experiment-lines.md`；
- 只有三 seed 和 QC 完成后，才更新 `docs/report/index.html` 的 scaling 章节；
- 不覆盖历史版本，不把 basic QC 误写成最终定律。

## 6. Agent 交接规则

接手后按以下顺序推进：

1. 先读 `agents.md`、本文件和
   `docs/appendices/s1_scaling_three_axis/report.md`；
2. 先完成 P0 和 pilot gate，不直接开 31-run full grid；
3. 所有远端运行先做 setting/data/hash manifest；
4. 每完成一组 run 立即同步、画图、回填日志；
5. 发现 setting 不一致时，停止汇总，不要通过改名或脚本过滤掩盖；
6. 不删除历史 run，不覆盖旧报告；
7. 每个逻辑阶段单独 commit；
8. 未经用户明确授权，不执行 `git push`。

**当前结论边界**：seed 42 的图形和探索性频率拟合已完成，但至少 seed
43/44、跨 seed uncertainty、frequency 的 epoch-dependent fit 和
table saturation 仍未完成。因此不能把当前结果写成已确认的三轴 scaling
定律；主报告 `docs/report/index.html` 仍不应更新。