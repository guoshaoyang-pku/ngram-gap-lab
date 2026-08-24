# 实验线索引（Experiment Lines）

> 本文件回答一个问题：**「某个实验在哪、归哪条线、图在哪、结论可不可信」**。
> 数值细节看 `experiment-log.md`；断言状态看 `claims-ledger.md`；setting 定义看 `../agents.md` §1。

---

## ⚠️ 权威数据源

**`data/runs_fixed/` 里带 `_fixed` 后缀的 run 是唯一权威数据。**

`data/runs/`（118 MB，64 个 run）与 `runs_fixed/` 里不带后缀的陈旧副本
**已于 2026-08-23 彻底删除**——按仓库原则，确认是 bug 污染的内容不归档、直接清除。
同时删除了由这些数据生成的全部图（`docs/_archive/figs_history/`，6 MB）。

### 两个 bug（代码已修复）

| bug | 症状 | 修复 |
|---|---|---|
| freq-bin 诊断复用训练迭代器 | 每次 freq eval 白吃 5 个 train batch，这些 batch 永不参与优化，且 epoch 计数虚高。**所有** launcher 都传了 `--freq_index`，故**全部历史 run 受影响** | `code/train.py` 已改用独立的 `freq_train_ds` 迭代器 |
| `table_betas[1]` 被静默覆盖 | `_table_rmsprop_step` 里 `b2` 错取 `self.ngram_beta2`，导致**所有显式传 `--table_betas` 的 rmsprop run 实际都跑 b2=0.999**，`--table_betas` 完全失效 | `code/train.py:537` 已改为 `b2 = self.table_betas[1]` |

详见 `notes/method/freq-bin-train-iter-bug.md`。
修复后的验证证据保留在 `data/runs_fixed/smoke_fixed_verify/`（400 步 smoke）。

### 修正幅度不是噪声

| run | pre-fix | `_fixed` | 变化 |
|---|---:|---:|---|
| `nglab1x_v10_input` | 1.931 | **1.867** | −3% |
| `nglab1x_v10_y` | 5.049 | 5.804 | +15% |
| `nglab1x_v10_v` | 5.041 | 5.450 | +8% |
| `nglab1x_v10_nogram` | 0.231 | 0.245 | +6% |
| `nglab0_25x_input_fv` | 12.99 | 9.69 | −25% |
| `nglab1x_opt_rmsprop_2x` | 2.376 | **4.674** | **+97%** |
| `nglab2x_opt_rmsprop_2x_b2_099` | 0.643 | **1.995** | **+210%** |
| `nglab1x_opt_sgd_09` | −0.002 | +0.073 | 符号翻转 |

⚠️ **`experiment-log.md` 全文数值仍是 pre-fix 的**，与本表矛盾。
回填是 `docs/notes/plans/plan-3-fix-and-backfill.md` 的 T1 任务。

## 主线（nanoGPT）

| # | 线名 | 科学问题 | run_id 前缀 | launcher | 作图脚本 | 图目录 | log § | 状态 |
|---|---|---|---|---|---|---|---|---|
| **M1** | 注入点消融 v50 | v/y/input 哪个产生 gap | `nglab_{v,y,input,nogram}` | `run_injpos.sh` | `gen_injpos_plot.py` | `figs/main/injpos_ablation.html` | §1–3 | 🗄️ 过时（val 每 50 步），被 M2 取代 |
| **M2** | **注入点消融 v10（博客主线）** | 同上，2000 步 + val 每 10 步 + 无表对照 | `nglab1x_v10_*_fixed` | `run_injpos_parallel.sh` | **`gen_all_figures.py`**（canonical）<br>`build_injpos_data_json.py`<br>`build_blog_clone_v10.py` | `figs/main/` | §8 | ✅ 完成 |
| **M3** | Table 优化器 · 1x epoch | RMSProp/AdamW/SGD 谁写得快；LR 剂量；β₂ | `nglab1x_opt_*_fixed` | `run_table_opt.sh` | `analyze_table_opt.py` | `figs/table_opt/` | §9 §9a §9b §9d | ⚠️ 完成但**结论待修正**（β₂ bug 直击此线） |
| **M4** | Table 优化器 · 2x epoch | 同 M3，epoch 拉长看 β₂/LR 是否改变 | `nglab2x_opt_*_fixed` | `run_table_opt_2x.sh` | `analyze_table_opt_2x.py`<br>`analyze_table_opt_1x_vs_2x.py` | `figs/table_opt/` | §9c | ⚠️ 同上（`b2_099` 从 0.64→2.00） |
| **M5** | shard 大小扫描 | 「epoch shard 越大 gap 越小」是否成立 | `nglab{0_25x…8x}_input_fv*_fixed` | `run_shard_sweep{,_v2,_360}.sh` | `gen_shard_sweep_figs.py` | `figs/epoch_scale/` | §4 §6 §7 §10 | ✅ 完成（12 点齐） |
| **M6** | epoch 对齐批（e6） | 对齐 epoch 数后 M5 的单调关系是否消失 | `nglab*_e6_fixed` | `launch_360_*.sh` | `gen_epoch_aligned_figs.py`<br>`gen_nogram_vs_epochaligned_figs.py` | `figs/epoch_scale/` | §12 | ❌ **不完整**：仅 0.25x–3x，缺 4x/5x/6x/8x |
| **M7** | 短 epoch × β₂ | β₂ 是否改变 per-epoch 台阶清晰度 | `nglab{025x,05x}_b2_099` | `run_epoch_short_b2.sh` | `gen_short_epoch_b2_figs.py` | `figs/short_epoch_b2/` | §11(B) | ⚠️ 完成但图未按 `_fixed` 重生成 |

**M6 的缺口值得单独提**：§12 结论「对齐 epoch 后单调关系消失」目前只有 8/12 个点支撑，
而缺失的恰是最能证伪的大 shard 端（4x/5x/6x/8x）。要么补跑，要么在结论里显式限定覆盖范围。

## Toy 线

完整说明见 **`../tasks/README.md`**。

| # | 线名 | 脚本目录 | 结果目录 |
|---|---|---|---|
| L1 | Lookup-Table 记忆 × Replay | `tasks/l1_lookup_replay/` | `tasks/l1_lookup_replay/results/` |
| L2 | Markov 链精确 gap 闭式解 | `tasks/l2_markov_exact/` | `tasks/l2_markov_exact/results/` |
| L3 | 单 context 采样律 gap(r) | `tasks/l3_sampling_law/` | 直接出图 |
| L4 | 幂律合成数据 × 真 harness | `tasks/l4_synth_powerlaw/` | 360-2 远端 + 本地汇总 JSON |
| L5 | 优化器伪影（RMSProp v 锯齿 / 表容量） | `tasks/l5_optimizer_artifact/` | `tasks/l5_optimizer_artifact/results/` |

另有两条**跑在真 harness 上的 toy 线**（图在 `figs/toy/`，脚本在 OPHIS toy 工作区，未迁）：

| # | 线名 | run_id | 作图脚本 | log § |
|---|---|---|---|---|
| T1 | toy β 扫描 / 台阶溯源 | `t5b_*` | `gen_within_epoch_figs.py` | §11(A) |
| T2 | toy 严格 Zipf | `t5z_zipf_s4{2,3,4}` | `gen_zipf_experiment_figs.py`<br>`analyze_zipf_gap.py` | §13 |

## 独立包

**`ngram5_freq_gap/`** —— **第四个实验维度：受控数据干预**（固定极简 setting，只动数据侧）。
唯一自变量 `alpha`（低频上采样强度），检验 `gap(r) ≈ (K_eff−1)/r`。
它与主线的「注入点 / table 优化器 / epoch 长度」三个维度**正交**，不是竞争实现。
定位、极简 setting 逐项核对、以及 P0/P1 阻塞项见 **`../ngram5_freq_gap/README.md`**。

⚠️ **口径隔离**：该包的 train probe 是「训练前抓取、全程不变的 2 个 batch」，
主线是「滚动的独立诊断迭代器」。两者各自自洽但**数值不可互相引用**。

---

## 已知待办

完整的可执行方案见 **`notes/plans/plan-3-fix-and-backfill.md`**（面向 agent 阵列，含验收标准）。
摘要：

| # | 任务 | 优先级 | 状态 |
|---|---|---|---|
| T1 | 回填 `experiment-log.md`：pre-fix 数值**直接覆盖**为 `_fixed`，不保留旧值 | P0 | 待办 |
| T2 | 修正 β₂ 记录（标注无效 + 删无支撑结论，**不补跑实验**，用户已明确不重要） | P0 | 待办 |
| T3 | 用 `_fixed` 数据重生成全部图 | P0 | 待办 |
| T4 | 修 `experiment-log.md` 结构（§10 重复、§5 缺失、90 行逐字重复） | P1 | 待办 |
| T5 | 修 `ngram5_freq_gap/model.py` 的死 fallback → 指向 `code/train.py` | P1 | 待办 |
| T6 | 补 M6 缺口（4x/5x/6x/8x）或显式限定结论覆盖范围 | P1 | 待用户拍板 |
| T7 | full-163 线：脚本已删除，数据坐标入库 `docs/notes/data/full-corpus-full163.md` | — | ✅ 已解决 |
| T8 | 长时程 no-ngram 对照（**缩小数据量前的保险**，不做缩小可缓跑） | P2 | 待办 |
| T9 | 固定 train 采样集合的 loss 曲线（测 ρ，与 T8 可并行） | P2 | 待办 |
| T10 | 缩小单 epoch 数据量以放大 gap（**唯一必须等 T8 的动作**） | P2 | 等 T8 |
| T11 | 工程整理：`RUNS_DIR` 环境变量化、抽 `table_opt_common.py`、包改名 | P3 | 随手做 |
| T12 | 干预机制已接线（CLI + 主循环 + `run_causal_minimal.sh`） | P1 | 代码就绪，待 GPU 实跑 |
