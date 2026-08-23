# 实验线索引（Experiment Lines）

> 本文件回答一个问题：**「某个实验在哪、归哪条线、图在哪、结论可不可信」**。
> 数值细节看 `experiment-log.md`；断言状态看 `claims-ledger.md`；setting 定义看 `../agents.md` §1。

---

## ⚠️ 权威数据源

| | |
|---|---|
| **权威数据** | `data/runs_fixed/`，run_id 带 **`_fixed`** 后缀 |
| 过时数据 | `data/runs/`（以及 `runs_fixed/` 里**不带** `_fixed` 后缀的对照副本） |

原因：`code/train.py` 曾有一个 freq-bin 诊断复用训练迭代器的 bug —— 每次 freq eval 额外
吃掉 5 个 train batch，这些 batch 永不参与优化，且 epoch 计数虚高。**所有** launcher 都传了
`--freq_index`，所以**全部历史 run 受影响**。详见 `notes/method/freq-bin-train-iter-bug.md`。

同一次修复还带出第二个 bug：`_table_rmsprop_step` 里 `b2` 错取 `self.ngram_beta2` 而非
`self.table_betas[1]`，导致**所有显式传 `--table_betas` 的 rmsprop arm，旧 run 实际都跑的 b2=0.999**，
`--table_betas` 被静默忽略。

修正幅度不是噪声级别：

| run | 旧 gap | `_fixed` gap | 变化 |
|---|---:|---:|---|
| `nglab1x_v10_input` | 1.931 | **1.867** | −3% |
| `nglab1x_v10_y` | 5.049 | 5.804 | +15% |
| `nglab1x_v10_v` | 5.041 | 5.450 | +8% |
| `nglab1x_v10_nogram` | 0.231 | 0.245 | +6% |
| `nglab0_25x_input_fv` | 12.99 | 9.69 | −25% |
| `nglab1x_opt_rmsprop_2x` | 2.376 | **4.674** | **+97%** |
| `nglab2x_opt_rmsprop_2x_b2_099` | 0.643 | **1.995** | **+210%** |
| `nglab1x_opt_sgd_09` | −0.002 | +0.073 | 符号翻转 |

**当前状态**：所有 14 个作图脚本已指向 `runs_fixed` + `_fixed`；
但 `experiment-log.md` 全文数值**仍是 pre-fix 的**，两者互相矛盾，回填是待办第一项。

---

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

完整说明见 **`../code/toy/README.md`**。

| # | 线名 | 脚本目录 | 结果目录 |
|---|---|---|---|
| L1 | Lookup-Table 记忆 × Replay | `code/toy/l1_lookup_replay/` | `data/toy_results/l1_lookup_replay/` |
| L2 | Markov 链精确 gap 闭式解 | `code/toy/l2_markov_exact/` | `data/toy_results/l2_markov_exact/` |
| L3 | 单 context 采样律 gap(r) | `code/toy/l3_sampling_law/` | 直接出图 |
| L4 | 幂律合成数据 × 真 harness | `code/toy/l4_synth_powerlaw/` | 360-2 远端 + 本地汇总 JSON |
| L5 | 优化器伪影（RMSProp v 锯齿 / 表容量） | `code/toy/l5_optimizer_artifact/` | `data/toy_results/l5_optimizer_artifact/` |

另有两条**跑在真 harness 上的 toy 线**（图在 `figs/toy/`，脚本在 OPHIS toy 工作区，未迁）：

| # | 线名 | run_id | 作图脚本 | log § |
|---|---|---|---|---|
| T1 | toy β 扫描 / 台阶溯源 | `t5b_*` | `gen_within_epoch_figs.py` | §11(A) |
| T2 | toy 严格 Zipf | `t5z_zipf_s4{2,3,4}` | `gen_zipf_experiment_figs.py`<br>`analyze_zipf_gap.py` | §13 |

## 独立包

**`ngram5_freq_gap/`** —— order-5 / trigram controlled 实验的**独立训练包**（vendored）。
按 `alpha`（低频上采样系数）组织，自带 `trainer.py` / `data_gen.py` / `lib.py` 与 5 个 launcher。
它与主线的「注入点 / table 优化器 / epoch 长度」三个维度**正交**，
且 `experiment-log.md` 全文未登记它、`data/runs*` 里无对应 run。

---

## 已知待办

1. **回填 `experiment-log.md`**：用 `_fixed` 数据替换全部 gap 数值，旧值保留为 `(pre-fix: X)`。
2. **重写 §9c / §9d 的 β₂ 结论**：原结论「β₂ 影响 ≤ ±0.4，远小于 LR 效应」建立在
   `--table_betas` 被静默忽略的 run 上，必须用 `_fixed` 数据重新判定。
3. **修 `experiment-log.md` 编号**：§10 出现两次（「基础统计归档」与「shard 扫描」是两个不同实验）；
   §5 缺失；§10 内约 90 行逐字重复。
4. **补 M6 缺口**或显式限定结论覆盖范围。
5. **合并 `gen_all_figures_v10.py` 进 `gen_all_figures.py`**：后者是前者功能超集
   （27 vs 22 函数，多出幂律拟合 / 曲线去噪 / step-slice），加 `--out-dir` 即可，省 43K 重复代码。
6. **抽 `table_opt_common.py`**：三个 `analyze_table_opt*.py` 约 40% 的行重复（三个入口保留，问题确实不同）。
7. **统一 `RUNS_DIR` 为环境变量**：现在 14 个脚本各自硬编码，下次再出 bug 又要改 14 处。
