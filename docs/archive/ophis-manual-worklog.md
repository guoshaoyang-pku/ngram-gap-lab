file:///Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap/docs/ngram-gap-mechanism-guide.html

> 🗄️ **[ARCHIVE]** 本文档来自已弃用的 `OPHIS_gap` 仓库，仅供历史溯源。
> 其中部分结论建立在 `current shell` / Muon / RoPE 等非极简 setting 上，引用前请对照 `agents.md` §6。


路径为 /Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap
我们的核心汇报文档为：/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap/docs/ngram-gap-mechanism-guide.html
这个是我的同伴做的实验：/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap/docs/ngram-gap-regime-bridge.html
实验很多可以作为参考

## 8月7日 — 合成 transition pilot 完成（360-2，4 runs）
- **任务**：`docs/synthetic-transition-task-design.md` 的 order=5 合成任务，A/B 两方案 × 注入 on/off。
- **数据**（360-2 `ngram5_data/synth_{A,B}_*/`）：vocab 8192、SEP 8191、order 5、block_len 7、
  4096 contexts、freq scale 8；train 57.09M / val 456.69M tokens；条件熵 Bayes 参考
  A=2.567 / B=4.387 nats（`transition_matrix.npz` 逐 context 精确熵）。
- **运行**（seed42，2000 步，val interval 10，探针 100–2000）：4 runs 全 rc=0；
  GPU0/1 两波并行；注入 run 约 171s/个（2.88B 参数含表，VRAM 16.7GB），对照约 144s（64.6M）。
- **结果**（val 构造块探针 target CE − H，step 2000）：注入 A +0.07 / B +0.06（贴着 Bayes）；
  对照 A +1.55 / B +0.90，且 excess 随 r(c) 单调下降（A 低频 ~8–12 → 高频 1.4；B ~4–6 → 0.8）。
  结论：注入能消除 gap；对照 gap 在完全受控的转移律上复现；B（lowrank 共享结构）对照优于 A（私有稀疏记忆）。
- **图/数据**：`docs/figs/fig_synth_excess_vs_freq.svg`、`fig_synth_excess_vs_step.svg`、
  `synth_{A,B}_summary.json`；分析脚本 `ngram5_freq_gap/analyze_synth.py`。
- **过程坑（已修，本地为准）**：生成器 cumsum+bisect 提速 ~400×（bit-identical）；
  360-2 trainer.py 缺 contexts 分支（已同步本地版）；`freq_index.keys.numel()` 打印崩溃已修；
  no-gram 控制组 `NANOGPT_NGRAM_INJECTION_IMPL` 必须为 nanogpt + `NANOGPT_ENABLE_NGRAM_VE=0`。
- **注意**：低频格 n=2–8 噪声大；对照 2000 步未收敛，excess 是上界；`allgram_frequency_decomposition.jsonl`
  是另一种 per-frequency 口径（branch=exact_context），未用于本报告主表。

## 8月7日（续）— epoch 对齐批（历史记录；当前仍等待 4x–8x canonical `_e6` 收口）
- **任务**（用户）：§10 的 step 对齐 sweep（gap@2000）里「shard 越大 gap 越小」
  混杂了「大 shard 只走了更少 epoch」。用户要求**对齐 epoch 数量**再看。
- **做法**：12 个 shard 大小全部训到 ~6 个 epoch，`--lr_schedule_epochs 6`
  把 LR 锚定到 epoch（所有 run 共享同一条 LR-vs-epoch 轨迹，排除 §10 ⚠️ 的 LR 拉伸混杂）。
  0.25x–1.5x 跑 360-2（14:26–15:47）；2x–3x 跑 360-1（16:17–17:12，360-2 首跑 OOM 后重跑）；
  4x–8x 跑 ophis-gpu（13:50 启动，8x 预计 ~22:30）。
- **结果（0.25x–3x 已完成）**：gap@6pass（epoch 7 首个 eval，lr=0.05）
  = 0.25x +1.09 / 0.5x +1.42 / 0.75x +0.85 / 1x +1.91 / 1.5x +0.91 / 2x +0.80 /
  2.5x +0.93 / 3x +2.14 —— **单调递减消失**，在 +0.8~+2.1 带内非单调波动。
  对比 step 对齐（gap@2000 = +13 → +0.5 → ~0）：单调递减主要来自「大 shard 重播轮数更少」
  （8x 在 2000 步内仅 ~2 epoch），而非 shard 大小本身；同重播轮数 + 同 LR-per-epoch 下，
  每 epoch 的重放 gap 大致相当。0.25x 在 6 pass 只有 +1.1（vs 36 pass 时 +13.0）→
  小 shard 的巨额 gap 是重播次数累积出来的。表 norm 随 shard 增大（tri 0.06→0.13），
  gap 却不随表大小单调增长。
- 图/数据：`ngram-gap-lab/docs/figs_epoch_scale/gap_vs_shard_size_epoch_aligned.png`、
  `gap_vs_epoch_curves.png`、`epoch_aligned_train_val_gap.png`；
  登记 `ngram-gap-lab/docs/experiment-log.md` §12；脚本 `docs/plot_scripts/gen_epoch_aligned_figs.py`。

## 8月6日（epoch 长度缩放实验 · ngram-gap-lab）
- （本会话续）用户重申：标准实验（blog 主线）validation 从 50 步改 10 步；并怀疑 table 优化器（RMSProp 无动量）学习慢/滞后，要试其他优化器。
  - 已把 OPHIS 规范配置改为 v10：`tools/cluster/run_injpos_ablation.sh`、`run_injpos_sanity.sh` 的 `VAL_LOSS_INTERVAL_STEPS=50→10`；`plans/plan-1-gap-formation.md` §3.1a、`docs/injpos-experiment-log.md` §6 同步。
  - v10 标准重跑（nglab1x_v10_v/y/input/nogram，2000 步）由 ngram-gap-lab 并行 agent 推进中（截至 20:35：v 在 GPU2 ~step220，input/y/nogram 未启动）；blog 数据 `injpos_ablation_data.json` 待 v10 input 完成后用 `build_injpos_data_json.py` 重建（自动变 10 步档）。
  - 优化器消融已实现：ngram-gap-lab `code/train.py` 新增 `--table_optimizer {rmsprop,adamw,sgd} --table_lr_scale --table_betas`（默认不变）；launcher `code/cluster/run_table_opt.sh`（4 arms：rmsprop_2x / adamw_090999 / adamw_080950 / sgd_09，input·1x·1000 步·v10）；已在 `ngram-gap-lab/docs/experiment-log.md` 登记 §9（planned）。待用户确认后占 GPU 3/4/6。

- 用户要求：标准实验（blog 主线）validation 间隔从 50 步改为 10 步；把训练集扩到 2x（epoch 1 变两倍）跑到 2000 步；原实验也跑到 2000 步对照；再把 epoch 长度减半看现象。
- 发现：`/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab`（blog 指向的干净复现代码）已有一个并行 agent 在做 v10/2000 步的 v/y/input/nogram 消融（`run_injpos_parallel.sh`，GPU 0/5/6/7，2026-08-06 19:56 启动）；2x 数据集（`nglab2x_*`，v50/2000 步）已由 `run_train2x.sh` 完成（gap@2000：v 1.169 / y 3.101 / input 0.687，见 `ngram-gap-lab/docs/experiment-log.md` §4）。
- 我补的两个实验（已登记 `ngram-gap-lab/docs/experiment-log.md` §6/§7，launcher `code/cluster/run_epoch_scale_v10.sh`）：
  - `nglab2x_input_v10`：2x 训练集 + v10 细曲线（与 v10 标准可逐点对齐）。
  - `nglab0_5x_input`：半 epoch（`shard_00060.bin` = shard_00001 前 12132 行，~168 steps/epoch）+ v10。
- **val-fix（20:35）**：用户指出 val 不能从数据流里 sequential 抽（v10 下 200 次 eval × 4 批会让 val 曲线沿 val 集滑动）。已修 `ngram-gap-lab/code/train.py`：启动时一次性捕获 `fixed_val_batches`（val loss）与 `fixed_freq_val_batches`（val 侧 freq-bin），每次 eval 复用同一批 val 数据；train 仍是唯一移动队列（消费主队列）。
  - 首跑 `nglab2x_input_v10` / `nglab0_5x_input`（移动窗 val）已停（数据保留 `data/runs/` 下），以 `nglab2x_input_v10_fv` / `nglab0_5x_input_fv`（fixed-val）重跑，GPU 5/7，20:41 启动，预计 ~22:00 完成。
  - 注意：并行 agent 的 `nglab1x_v10_v`（GPU2，20:24 启动）与 `nglab2x_input_fine`（GPU1，v50）也是旧逻辑（移动窗 val），其后新启动的 run 会自动用 fixed-val 代码。
  - **结果（22:30 回填）**：0.5x/1x/2x 三跑均完成（fixed-val，v10，2000 步），gap@2000 = **+4.95 / +1.96 / +0.50**；观测 epoch 长 ~110–120 / ~230 / ~450 steps。剂量关系：epoch 越短 gap 越早越强（0.5x 的 gap 是 2x 的 ~10 倍）。图 `ngram-gap-lab/docs/figs_epoch_scale/epoch_scale_train_val_gap.png`，数据见 `ngram-gap-lab/docs/experiment-log.md` §6/§7。
- **shard 大小扫描（8/7 彻夜批，12 点全部完成）**：gap@2000 = 0.25x **+12.99** / 0.5x +4.95 / 0.75x +2.12 / 1x +1.96 / 1.5x +0.87 / 2x +0.50 / 2.5x −0.03 / 3x +0.11 / 4x −0.03 / 5x −0.05 / 6x −0.11 / 8x +0.03 —— **假设成立：shard 越大 gap@2000 单调越小，0.25–2x 近似幂律（log-log 斜率 ≈ −1.5~−2），≥2.5x 在 2000 步内 gap≈0（大 shard 只是延迟 gap 出现，见 epoch 坐标图）**。2.5x/3x/4x 首跑 val 与 train 重叠已修正为 `_v2`（val 从最后一个 train shard 之后开始，10:11 图已用 v2 重跑）。⚠️ 2.5x v2 延长跑（3200 步）train 在 epoch 3–5 停滞 ~3.8、epoch 6 才掉到 3.31，最终 gap +0.69 不可与 3x +1.94 / 4x +1.55 直接比——主因是延长 run 的 LR 调度按 max_steps 拉伸（step 2000 时 lr_mult 0.60 vs ≤2x 的 0.05），已补 `_v3`（max_steps=2000 同 LR 调度）并完成：2.5x/3x/4x 公平 gap@2000 = **+0.04/+0.03/−0.04 ≈ 0**，与延长 run 一致 → 主剂量曲线对 LR 稳健（同 LR 下 2.5x train@2000 3.49 vs v2 3.80，LR 拉伸确实拖慢 train 但 gap 不变）；2.5x seed43 长跑探针完成：final gap **+0.74**（s42 为 +0.69），epoch 2-5 同样停滞 → 停滞是「延长 run 的 LR 拉伸 × 数据」的确定性现象、跨 seed 复现，2000 步预算内（主剂量曲线）无异常。图 `ngram-gap-lab/docs/figs_epoch_scale/dose_response_gap2000.png`、`gap_vs_epochs.png`、`sweep_train_val_gap.png`；日志 §10。
- 数据准备：`data/tokenized/shard_00060.bin`、`data/freq_index_train0_5x.npz`（半 epoch 频率索引）已在集群建好。
- 备注：本仓库 `tools/cluster/epoch_scale/` 的 OPHIS 方案 launcher 因 GPU 被占 OOM 且与 ngram-gap-lab 重复，已删除；集群 `ngram-gap-exp/lib.py` 恢复为 20260805 干净备份（此前被 ngram5 补丁破坏成不可导入）。

## 8月4日
- **toy v5 2×2 干净证明完成（12 runs = 3 seeds × 4 象限），guide 已新增 §16。**
- 设计（`toy/toy5_data_gen.py`）：vocab=2048，32768 个 2-gram 键；low 模式 30720 键 r<16（train/val next 独立）+ 1024 共享键；high 模式全部 r=64 共享 next；块结构 [a,b,y,S] 让附带上下文在 train/val 分布一致。模型 = baseline_current 同款（bigram+trigram VE、8×768、RMSProp 表、seed 42/43/44）。
- 结果（headline gap = val−train，3-seed 均值±std）：
  - 两个都有（on_low）：**6.92 ± 0.43**（train 0.004 / val 6.80）
  - 去掉 n-gram 表（off_low）：2.12 ± 0.003（train 2.58 / val 4.69）——backbone 固有过拟合底噪
  - 去掉低频键（on_high）：0.84 ± 0.26，但**核心键 gap ≈ 0.001**（0.84 全是附带位置效应）
  - 都去掉（off_high）：0.064 ± 0.000（干净基准）
- 最关键：per-r 精确分桶呈**阶梯函数**——on_low 在 r<16 处 gap 15–20、r≥16 处 ≤0.005，断崖恰在 r=16 共享阈值。表只在低频键上制造 gap。
- 图：`docs/figs/t5_2x2_headline.svg`（四象限柱状）、`t5_step_function.svg`（per-r 阶梯）、`t5_trajectories.svg`、`t5_loss_curves.svg`；数据 `toy/run_meta_table_t5.json`。
- 运行记录：ophis-gpu `/data3/guoshaoyang/ngram-gap-exp/toy/`，runs/t5_*（12 个，均 done）；分析 `toy_analyze.py --run` 12 个并行完成。
- 过程踩坑：上一轮遗留的旧启动器（high 4000 steps 版）曾并发抢写 run_meta/启动 4000-step run，已杀掉 `t5_on_high_s43/44` 旧实例并以 2500 steps 重跑；toy5_launch.sh 的 wait_gpu 已修（只统计 running 的 GPU）。所有 12 个 run 最终均为正确配置（low 2000 / high 2500 steps）。
- **P0：svbird F 频率遮罩复现 wave1 完成（2026-08-04 凌晨，seed42）**：7 arms 全部从同一 step-337 checkpoint fork（run_exp.sh 新增 p0_freqmask_* 系列；mode=custom value-readout 置零，未改 train.py）。结果（svbird 口径，last100 = mean(val−train) over 901–1000）：no-mask 1.096；B 1–5 → 1.002（−8.6%）；T 1–5 → 0.867（−20.9%）；B+T 1–5 → 0.778（−29.0%，非加性未复现，本地近似可加）；B 6–200 → 1.017（−7.2%）；B+T 0–200 → 0.155（−85.9%，val 反升 4.55→4.37）；B+T 0–1000 → 0.087（−92.1%，val 4.73 性能代价）。val 侧 B/T 覆盖率 0–200: 43.0%/88.6%（svbird 42.1%/89.0%）、0–1000: 73.0%/97.0%（svbird 72.5%/97.0%）。图 docs/figs/p0_freqmask_*.{svg,png}、docs/interactive/fig_p0_freqmask.html；guide §15.5。
- **P0 注意（口径诚实性）**：fork 组 source run 带 fixed probe，probe loader 消耗全局 RNG → source 的 epoch-1 数据流与 baseline_current 不同（step-337 train 4.95 vs 5.13），fork no-mask last100 1.096 ≠ baseline 0.788。7 arms 同源同流，组内配对干净。实现为 readout 前向置零（masked_fill，被遮行等价冻结）；grad-keep（forward-only）变体待 wave2 补。
- **P0 wave2 完成（2026-08-04 03:00）**：s43/s44 同源 checkpoint + 关键 arm 3-seed 补齐 + grad-keep 变体 + ≥1001。3-seed 汇总（svbird 口径，last100 mean±std）：no-mask 1.068±0.112；B+T 1–5 → 0.781±0.032（−26.5%±5.2%，三 seed −29/−30/−20.5%）；B+T 0–200 → 0.155±0.004（−85.4%±1.2%，三 seed val 全部改善 4.55/4.60/4.52→4.37/4.39/4.39）；B+T 0–1000 → 0.087（−92.1%）；B+T ≥1001 → 1.095（−0.1%，val 恶化 4.65，纯性能代价）。grad-keep 变体（train.py 加 NGRAM_FREQUENCY_READOUT_MASK_GRAD_KEEP：readout 前向置零但梯度照常流向 table/gate）0.153 ≈ masked_fill 0.155——结论对实现细节稳健。guide §15.5 已更新为 3-seed 表。
- **P0 wave3 完成（2026-08-04 03:45，全部 seed42）**：comb/rowzero/shard/replay 四组边界检验，结果已写入 guide §15.6：
  - comb（只留中频段 201–1000）：last100 gap 0.148 ≈ 单 0–200（0.155），但 val 4.54 > 4.37 → 低频遮罩减 gap 与高频遮罩性能代价近似可加（复现 svbird F 结论）。
  - rowzero freq-peak/large（e1 边界按 gap_contribution 选桶清零+冻结）：1.096/1.108，**无效**——e1 边界可测到的高贡献桶（峰值仅 +0.019 nats）不是 e3 gap 的载体；支持「滞后行」机制（p1 reset@e2 −89% vs e1 −13%）。
  - replay4/replay6（3→4→6 epochs）：gap 1.07 → 3.47 → 6.17，train 3.17→1.78→0.43、val 4.55→5.34→6.62——**重播次数是干净的剂量反应**。
  - shard2/shard3（2×/3× 数据仍 3 epochs）：train 塌到 0.45/0.09、val 6.88/8.16、gap 6.45/8.09——训练预算越大塌缩越彻底（与 no-mask 差异含步数因素，no-ngram 对照运行中）。
  - 图：docs/figs/p0_freqmask_conditions.svg（shard/replay 全轨迹）；guide §15.6。
- **no-ngram 对照（历史记录：当时运行中，后续已完成）**：replay6_nongram / shard2_nongram / shard3_nongram（CURRENT_NGRAM_INJECTION_IMPL=none）用于区分 shard/replay 的 gap 爆炸是表还是 backbone 驱动，结果已汇入 guide §15.6。
- **toy 燃料剂量扫描（历史记录，后续已完成）**：toy5f 的 high16 与低频占比扫描已完成，结果已汇入 guide §16.10。**P0 wave2 也已完成**：s43/s44 checkpoint source + ≥1001 高频对照和 grad-keep 结果见 guide §15.5。原始启动坑仍保留作复现记录：`NGRAM_GLOBAL_FREQUENCY_CUSTOM_RANGES` 的 `1001+` 语法会被 `split("+")` 拆坏，改用显式上界。
- **统一口径 compact 表**：tools/compact_table.py 生成 runs/compact_table.csv（25 个核心 run 的 last100/e2/e3/final gap + train_337/687，svbird 口径），本地副本 docs/figs/p0_compact_table.csv。
- **toy 级因果干预（2026-08-04 夜，on_low 上断表的三个口子）**：
  - e1 后屏蔽 readout → 2.12±0.003 = **逐位等于 off_low**（表贡献的 4.8 nats 全走 readout 通道）
  - e1 后冻结表 → 2.49±0.16（−64%）；e2 边界全表回滚一次 → 7.90（无效：toy 重播 29 epoch，回滚后重新累积；与真实 P1 的 −89% 差异在于真实 e3 只有 313 步）
  - 对应真实数据 P1/P2（−49%/−89%）互证；图 `docs/figs/t5_interventions.svg`
- **toy 干预补充：每 epoch 换 hash（exp4 模拟，3 seeds）→ 1.96 ± 0.09（−72%，train 3.26 背不动）**：表行每 epoch 被打散，无法稳定累积错误记忆；对应真实 exp4 −93%。reset@e2 3 seeds 7.80 ± 0.10（无效，已确认）。
- **off_low 长训对照（8000 步 = 104 epoch，3 seeds）**：headline 只到 3.36（train 地板 2.49 背不动），但 r=1 低频键 gap 涨到 20.7（超过 on_low）——低频键过拟合在无表时同样存在且加重，表的作用是让 train 完全塌陷（0.004）→ gap 翻倍以上。结论：去低频 = 根治，去表 = 显著缓解但不根治。
- 历史待办已更新：svbird F 频率遮罩复现（P0，baseline_current 框架 + 3 seeds）已完成；BPE 方向目前只有 token-remapping proxy，真正 BPE tokenizer / Engram / Over-Encoding 直测仍未做。

## 8月3日
- 评估了 svbird 线上页面（https://svbird.github.io/ngram-gap-regime-bridge/，A–G 全部，重点 E/F）：下载其数据文件复算 18 个 F 数字全部一致；E/F 方向与本地 P1/P2 全部一致（readout mask ≈0.06、gate 无效、reset 特异性、低频 readout 遮罩有效、高频遮罩=性能代价）。
- 结论：符合理论预期、无反例，结论收敛；新维度「按频率遮 readout」本地没有，需统一框架复现。
- guide 已新增 §12.5（对照）+ §15（评估与统一复现计划）；当前 P0/P1 已完成，P2 为可选补强，P3–P5 deferred，见 guide §15.3。
- svbird run 数据在 svbird 侧 remote_training_runs/，本地/集群无副本；复现时以 baseline_current 框架 + 3 seeds 重跑。

## 7月22日
我们发现了一个现象。在有ngram组建的nanogpt中，epoch2，epoch3的时候，train loss会阶梯下降，然后val loss会阶梯上升。形成gap。

我们希望解释：
我们希望解释，这个是什么导致的。目前我们的猜测就是ngram对于比较稀有的token过拟合，见到重复的训练样本之后，train就会学到shortcut，从而导致过拟合。
我们还希望能预测什么样的setting可以加剧或者消除gap

目前的机理解释：
目前我们已经做了一些验证实验。有一条逻辑的通路。首先我们看了一下，ngram的贡献是以v的形式，会有一个ngram的v项目，乘上一个gate作为模长，添加到attention的v上面。确实我们观测到新的epoch这个norm会有增加，可能意味着ngram项观测到了数据集分布的变化，发现增加ngram的norm可以因为押对一些重复的样本来获得优势，导致对应的norm增大。但是这个ngram context下的train和val的分布不一样，对于稀有的ngram这个gap会比较大，就会导致overfit和val的翘起。

写文章的角度：
这个研究可以给出一些启示。比如说overencoding的项目在后训练的时候，这些overencoding的项目可能应该被冻结，因为后训练SFT或者RL可能会有多个epoch，重复训练可能会导致overfit。

关于repo的使用方式：参考的分支为 windows/gap windows/bottom-up-gap-decomposition
我们的分支为：shaoyang/nanogpt-gap-explain

目前的todo：
1. 文档有很多地方要改，有些是画图和显示的问题，有些是需要补充观测量，甚至补充实验。
目前发现 5k+重复次数的ngram 的 gap不明显，这个是最重要的现象。

图 14.7：Gap contribution = frac × (val - train)。这是回答"哪个 bucket 对 val 翘起贡献最大"的关键图。 Novel bucket（红，粗线）从 epoch 2 开始就领先，最终贡献 ~1.5。 之后是 21-50、6-10、51-100 这些低中频 bucket。 5k+ 高频 bucket 的 gap contribution 几乎为 0（因为它在 train 和 val 上都拟合得不好）。

首先这里的作图，y轴的最低点应该是0，遇到负数就往下超出一些，横轴线一定要是0.


现在这里的hit是什么含义？
我觉得应该是bigram或者trigram的表被命中的次数吧。在train里面单个epoch被命中的次数吧。
然后我们要先做出这个的分布。
做一个分布图。然后也做一个累积分布图（从低频到高频），更直观展现出平均每个token的重复次数。


然后 5k+ 1k~5k，除了展示总贡献，还要展示平均每个token的贡献。万一是因为罕见token的数量就是很大呢。

然后后面的几个图片，都是做成多图叠加可以切换的。一张图展示train loss，一张图展示 val loss，一张图展示gap。上面还可以选择是看 per token，还是总贡献。


2. 还有一个可以验证的东西。就是我们看到远程（11.9 关键发现：Gate 与 Gap 的时间错位）
这里，gate norm和table norm的增加基本上和gap的发生是同时的。
【注意这些都在windows的那两个分支里面】
对于delay的情况，我们可以去看 gate norm，table norm和gated rms的这个情况。

B. Vanilla nanoGPT → Bottom-up Current-shell Gap
从干净负例逐步加入新增机制；旧 bridge 到 current core 仍无目标 gap，切到完整 current shell 后得到本地正例。

Vanilla nanoGPT
ARCH_VARIANT=nanogpt_original; no n-gram value tables

+ full n-gram value tables
unigram + bigram + trigram; table AdamW
no target gap
branch: table RMSProp
nanoGPT + full n-gram; table RMSProp

这个setting，我们发现epoch2没有发生fork，epoch3才产生fork。这里要把norm的图片，和后面的loss按照频率的分布的图片都画一下。



+ Muon matrix optimizer
nanoGPT + full n-gram + matrix Muon

观测到 delayed gap。让我们把norm和loss的图片也都画一下。我们看一下趋势是否是对齐的。对齐了就好解释了。






## 7月29日

7月29日周会
https://guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide/#chapter-7
现象
[图片]
目前的解释：
首先，应该是ngram导致了这个问题。我们作出ngram的 norm和steps的关系，发现趋势是一致的，所以应该就是ngram导致了gap。去掉ngram之后 gap 显著降低，但不是数学意义上的严格零；同 setting 的 no-ngram 对照 final gap 约为 0.125±0.001，见 guide §15.6。
[图片]
然后我们对于ngram（bigram trigram）context不同的token统计loss，我们预期罕见的ngram，train会overfit，val会翘起。所以作出gap和频率之间的关系，可以看到确实罕见的contextgap更明显地产生了。
[图片]
（启示）我们使用ngram应该避免过拟合低频context。我们把ngram hash table换成over-tokenization，对于token进行bpe合并，做一个不碰撞的hash表只over-encode高频的token组合。应该会有比较好的效果（还没试）


「关于这里，有几个问题比较大」在线文档里面的几张图片是假的。
存个档。/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap/docs/ngram-gap-mechanism-guide-0728.html

现在的版本应该可以删掉很多图片和section。
1. 实验设置这一章可以全部保留。
2. Gap现象和因果链合并到一起，只展示那个图片。
3. 关键证据和消融矩阵和诊断图暂时都删掉。
4. §7.9 关键发现：Gate 与 Gap 的时间错位。从第七章到这里这张复合的图，之前的图全部都删掉。这张复合的图片保留。
5. 后面的实验都要和这个复合图是同一个setting。最重要的是“trigram-context 命中频次分桶：gap (val-train)“这里val train要分别写出来。都以train的频率为准。按理说是有1～10的选项的。现在只有11-20开始，很奇怪。然后也提供pertoken和总贡献的选项。train val gap的选项。其他的图片就不需要了。【简记的话就是所有的gap图都要自动带上val和train的曲线，这样才清晰】
6. hit count记录一下train的一个epoch里面各种context重复的次数。分别对于train val统计，uni，bigram，trigram统计，也是复合图。注意统计的频率定义仅在train上。
6. muon实验之类的都暂时先删掉。除了我点名的图片和章节，其他都先删掉。
7. 所有的图片（包括其他文档的图片），svg和png放到 /Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap/docs/figs 这个里面。子html放到另外的文件夹里面。


文章可以说的点：

1. 逻辑，核心的observable应该是频率和gap的关系。
  1. 绘制出gap和词频的关系，发现确实有关系
  2. 我们的干预方法就是可以手动去掉低频词表，或许gap就可以降低。从而给出启示（bpe来限制低频分量）
  3. 验证：用bpe做overencoding之后，效果确实好很多
2. 逻辑，gap出现是因为训练集里面多epoch有重复，模型学会了在训练的早期去memorize一些pattern
  1. 没有ngram就没有gap——已经验证
  2. 两个epoch的数据混起来训练，也会有gap——已经验证
  3. Froze ngram+gate，没有gap——郭绍阳验证，vbird证伪（可以理解）
  4. 每一个epoch的ngram查询有独立的hash错位，导致同一个context无法稳定查到同一行，预期显著降低 gap——已验证（guide §10 exp4，NGRAM_HASH_RESEED_PER_EPOCH=1，双 shell −93%，2026-07-30）
  5. BPE 杜绝低频 context，可能降低 gap——真正 BPE 未验证；当前只有 token-remapping proxy（gap −67%）


## 7月30日（晚）— 统一 setting 到 current-shell baseline

之前 §2 现象图用的是 `20260720-bottomup-min-gap`（current shell, 32768/12），§7.9/§9 用的是 `20260725e_nano_rmsprop_fixv3`（nanogpt_original, gap 仅 0.688）——两套 setting 不一致，且 §9 频次图 trigram 因 hash 碰撞只有 11-20 起的桶。

**本次**：在 ophis-gpu (GPU 1) 补跑了一个干净的 current-shell baseline run —— `baseline_current`（current shell + current-style grouping + bigram+trigram RMSProp, seed42, 1000 step, vocab=8192, n_layer=8, epoch 边界 [338,687], epoch3 gap 从 ~0.5 扩大到 1.33）。该 run 同时记录了逐 step norm（reader_compact）+ allgram 频次分解（allgram_frequency_decomposition.jsonl）+ 频次探针（global_frequency_probe_observations.jsonl）。

现在 §2 现象图、§7.9 loss↔norm 双轴图、§9 频次分桶 + 命中频次分布图**全部统一到 baseline_current 同一 run**。
- 频次用精确离线索引（GlobalNgramFrequencyIndex），桶细分为 novel/1/2/3/4/5/6-10/11-20/…/5k+，trigram 现在所有桶都有数据（不再有 hash 碰撞下限问题）。
- 关键发现：trigram 的 val novel 占 ~30%（大量 val context 从未在 train 出现），是 val 翘起的主要来源；bigram val novel 仅 ~4%。
- run_exp.sh 新增 `baseline_current` case（NGRAM_GLOBAL_FREQUENCY_MODE=baseline，无干预）。
- artifacts 在 `remote_training_runs/baseline_current/`，构建脚本 `tools/build_freq_figures.py` / `build_loss_norm_json.py` / `build_phenomenon_svg.py`，渲染 `tools/render_*`。
- §10 干预实验仍用原 B.1（nanogpt_original）+ current-shell 两套（exp4/exp5），未动。
- 浏览器 tab-list bridge 不稳定无法目视，已用 node DOM-shim 验证三张交互图 JS 渲染无误（loss-norm 5 条线、gap_by_freq 30 条线、hitcount 15 柱+1 累积线）。





## 8月2日 — P1/P2 因果拆解第一波 + exp4/5/6 统一 setting 验证

### P1/P2 因果 run（8 干预 + 1 同 obs 控制，全部 baseline_current setting，seed42，1000 step）
- 控制（同 obs，`runs/baseline_current`）：final gap@1000 = 1.096（原基线无 direct-kernel obs = 1.330，obs 探针有 ~±15% 扰动，见下）
- **P1 historical row-state**：e2 边界全行回滚 → final gap 0.121（−89%）；e1 边界全行回滚 → 0.956（−13%）；ref 行回滚 0.992、随机行回滚 1.126（特异性弱，主要是 scale）
- **P2 write vs reader/backbone**：冻结 table@e1 → 0.559（−49%）；冻结 reader/backbone@e1 → 0.507（−54%）；冻结 gate → 0.865（−21%）；屏蔽 readout@e1 → 0.116（−89%）
- 结论：历史行内容是必要的（e2 回滚 −89%）；readout 通道必要（−89%）；table write 与 reader/backbone 各贡献约一半
- 图表：`docs/figs/fig_p12_gap_curves.svg`、`docs/interactive/fig_p12_causal.html`、数据 `fig_p12_data.json`；脚本 `tools/analyze_p12.py`、`tools/build_p12_figures.py`
- 集群：`runs/p1_*`、`runs/p2_*`；run_exp.sh 新增 baseline_current/p1_*/p2_* case（在集群，未同步本地）
- 备注：direct-kernel/history-ablation obs 的 in-place 表写入引入 run-to-run 不确定（同配置重跑 final gap 0.956 vs 0.828），效应量远大于噪声；重跑 `p1_reset_all_e2_rerun`/`p2_readout_mask_e1_rerun` 确认中

### exp4/5/6 统一 setting 验证（从已有 run 提取，口径 val−raw_train）
- exp4 hash reseed/epoch：final gap 0.282 vs baseline 1.330（−79%）→ **hash 错位验证通过**
- exp5 低频 gate 清零（bigram/trigram 1-200）：final gap 1.553，未消除——解释：novel（count=0）桶不在 1-200 范围内，val 翘起主体是 novel，干预打偏
- exp6（精确重复次数分桶，1700 step）：1000 step 处 gap 1.502，与 baseline 同量级（口径差异来自 survey 探针与模式）

### 待办（下一步）
- [ ] 等重跑结果确认后更新 p12-causal-results.md
- [ ] BPE/over-encoding 干预（新实验，需实现 over-encode 高频组合，或并入 toy 数据集频率桶验证）
- [x] toy 数据集（历史记录；生成器、sanity run、2×2、剂量和干预均已完成，结果见 guide §16）
- [ ] 用户检查 P1/P2 图 → 反馈修图


## 8月2日（第二轮）— exp7 诊断 + 3-seed 稳健性 + toy 交付

### exp7：over-encoding 代理验证失败（已诊断，结论写死）
- `exp7_overencode_th200`（低频 gate 清零 bigram/trigram 0-200，含 novel）final gap **1.280**；`th1000`（0-1000）**1.686**；控制 1.096 → **干预无效，反而略增**
- mask 生效确认：`train.py` `_add_value_residual` 中 `value_gate.masked_fill`（train/eval forward 均调用）；`train.log` 显示 ranges 含 novel（lower=0 补丁 `train.py.bak_20260802_exp7`）
- 分桶诊断（step1000，probe 口径）：bigram ≤200 gap 1.40→1.66→2.10（控制→th200→th1000）；trigram ≤200 1.07→1.34→1.72；未 mask 桶也升
- 机制：gate 输入侧（hidden state）不含频率信息，不是频率干预的有效位点；强制清零后模型把低频 train 位置记忆转移给 backbone（train 3.38→2.98），val 不受益（4.47→4.66）→ gap 维持/扩大
- 对照：readout 整体屏蔽（−89%）有效，因为从传导口整体断开记忆通道；频率结构在表/行内容（P1 e2 回滚 −89%、toy gap(r)），不在 gate
- **「真正 BPE 合并高频组合/低频共享行」仍属未验证项**（需新实验，等用户确认方向）

### 3-seed 稳健性（seed42/43/44）
- `p1_reset_all_e2`：0.121 / 0.123 / 0.122（+rerun 0.123）— 3-seed 波动 < ±0.01
- `p2_readout_mask_e1`：0.116 / 0.121 / 0.126（+rerun 0.119）— 3-seed 波动 < ±0.01
- 控制组 3-seed：seed42 1.096、seed43 1.530、seed44 0.997 — 控制本身训练动力学敏感（0.997–1.530），干预塌缩行为 seed 不变
- `p2_freeze_both_e1`：0.572 ≈ 仅冻结 table（0.559），gate 冻结不叠加

### exp4/5/6 统一 setting 验证（baseline_current 口径）
- exp4 hash reseed/epoch：0.282（−79%）✅；exp5 1-200 无效（novel 不在范围）；exp6 @1000 gap 1.502 同量级

### toy 数据集交付（subagent Bohr）
- 15/15 runs 复现末 epoch gap（6.5–8.7 nats）；Spearman ≤ −0.9（13/15 = −1.0）；r1/r5120 ≥63×；novel 解释 val 翘起 96–100%
- 边界：reshuffle 不减少 gap（per-context 频率不变）；字面 Pearson(gap,log r) 仅 n2 达标 → 建议 Spearman/log-log ρ
- 产物：`docs/toy-dataset-results.md`、`docs/figs/toy_*`（48）、`toy/`（生成器/launch/analyze/run_meta_table.json）；集群大文件未同步（toy/ws、runs、checkpoints、cache、data）

### 待办
- [ ] 用户检查 P1/P2 图（fig_p12_gap_curves.svg / interactive）与 toy 判据（ρ 形式、novel trivial、reshuffle）
- [x] baseline_current_s44 完成（gap 0.997），3-seed 控制表已补齐；run_summary.json 由 `tools/analyze_p12.py` 生成
- [ ] BPE 真正实现方向等用户确认
- [ ] P1/P2 第二波（可选）：epoch-lag matched probe、frequency-stratified row zero


## 8月4日（夜跑）— P0 wave3 收尾 + no-ngram 对照 + toy same-order 公平考卷 + 剂量扫描

### P0 wave3 完成 + no-ngram 对照（guide §15.6，baseline_current，seed42）
- wave3 8 arms：comb（只留 201–1000）0.148 ≈ 单遮 0–200 → 低频遮罩 gap 缩减与高频性能代价近似可加；rowzero freq-peak/large 1.096/1.108 无效（e1 边界贡献峰值仅 +0.019 nats → 支持「滞后行」机制：e2 写入 e3 重读）；replay4/6 → 3.47/6.17；shard2/3 → 6.45/8.09
- **no-ngram 对照（同一框架最干净因果点）**：replay6/shard2/shard3 关掉 ngram 注入（CURRENT_NGRAM_INJECTION_IMPL=none）→ last100 gap **0.056 / 0.029 / 0.020**（−99%~−99.8%），train 停在 5.2、val 也停在 5.2 → gap 爆炸全部由表贡献，backbone 本身不产生 replay 特异 val 翘起
- 图：`docs/figs/p0_freqmask_conditions.svg`（新增灰色虚线 no-ngram 轨迹）、交互 `docs/interactive/fig_p0_freqmask.html`（汇总表加 3 行 no-ngram）

### t5f 燃料剂量扫描（guide §16.10，shuffled-val，12 runs 完成）
- 巧合低频键占比 0/25/50/75/93.75% → **per-context (seen) gap 单调：0.005 → 0.86 → 2.36 → 5.33 → 7.08**；headline（shuffled val）被顺序混杂抹平 ~7（novel 上下文噪音）——口径选错会得出「低频没关系」的错误结论
- 图：`docs/figs/t5f_dose.svg`（左 clean dose 单调 / 右 headline 平线对照）

### toy same-order 公平考卷（guide §16.9，12 runs 已完成训练，分析中）
- **混杂修复**：原 v5 shuffled val 的 headline gap 6.87 全来自 train/val 顺序不同的附带上下文；`--val-order same` 变体 = val 与 train 同流、仅巧合键换答案 → 附带上下文完全一致，headline 即干净口径
- 生成器验证：low_same 差异恰为 65536 个巧合答案位（全在 y 位置）、high16_same（frac=0）差异 0
- 训练结果：on_low train 0.004/val 4.04（gap ≈ 4.0）、off_low train 2.58/val 3.97（gap ≈ 1.4）→ 表让 train 塌缩、val 卡住
- **同一批又补了 same-order frac 剂量（t5sf，9 runs：25/50/75% × 3 seeds）**，把 headline 口径剂量曲线也钉死（原 t5f 的 headline 被顺序混杂）

### 纪律记录
- wait_gpu 不能以 run_meta status 判占用（train 完成不写 done，需 analyze 才写）→ 只信进程级 /proc CUDA_VISIBLE_DEVICES + 显存 <1000MiB 双检查
- toy runs 很快（15ms/step，2080 步 32s）；t5f 分析每 run ~2min，12 runs 后台 ~25min；分析完自动删 checkpoint

### 8月4日（夜跑续）— t5s 公平考卷分析完成 + 真实数据去低频对拍启动
- **t5s same-order 2×2 分析完成（12 runs）**：on×low **3.94±0.23**、off×low 1.40±0.002、on×high16 0.000、off×high16 0.000（train=val=3.56）→ 只有「表+低频键」同时在场才有 gap（guide §16.9）；t5sf 同序剂量 25/50/75% headline=0.40/0.98/2.56（+low 3.94）完美单调（§16.10 表新增行）
- 图：`docs/figs/t5s_2x2.svg`（补丁：cache 名 t5_ 前缀）、交互 `docs/interactive/fig_t5s_2x2.html`；compact 数据由集群 `toy/t5s_dump3.py`（按 per-run run_meta 聚合，修复 runs_meta_all.json 丢字段问题）生成
- **真实数据去低频（BPE 代理）**：`tools/remap_rare_tokens.py` fixpoint 重映射完成 T=3000/T=8000（rare survivors 0.002% 行）；prescan freqidx 完成；bigram 唯一上下文 4.03M→1.84M（−54%）→0.54M（−87%），trigram 22.8M→14.2M→5.8M
- **nofreq 对拍已启动**（`launch_nofreq.sh`，6 runs = t3000/t8000 × seed42/43/44，baseline_current 同 setting，1000 步，injection=current）：GPU 0/1/3/4 先跑 4 个，另 2 个等 GPU；完成后从 run_summary.json 取 final_gap/epoch 边界填 guide §16.11
- guide §16.11 新增（历史记录：当时为运行中占位，后续已填入结果）；plan-1 §4.2 同步

### 8月4日（夜跑续2）— nofreq_t8000 结果出炉：真实数据去低频 gap −67%
- **nofreq_t8000（3 seeds，baseline_current 同 setting）**：epoch-3 final gap = 0.44/0.42/0.34（mean **0.40±0.05**）vs baseline_current 1.10/1.53/1.00（mean 1.21±0.28）→ **−67%**；final train 3.00 vs 3.31、final val 3.40 vs 4.52
- **epoch 边界几乎一致**（nofreq [337,685] vs baseline [338,687]）→ 同 setting 公平对拍成立（排除重播轮次混杂）
- 图：`docs/figs/nofreq_gap.svg`（本地生成修中文豆腐块）+ 交互 `docs/interactive/fig_nofreq.html`；guide §16.11 已填结果
- **t3000（中间剂量）重跑中**：首轮 3 runs 全挂（triton/gcc InductorError，4 并发编译冲突，exit_code=0 是 tee 假象）；失败目录归档 `runs/_fail_t3000_s*_inductor_compile`，已清理重跑（GPU 0/1/3）
- run_meta：`tools/write_nofreq_meta.py` 写 nofreq_t8000 三 runs（seed/setting/epoch 边界/gap）

### 8月4日（夜跑续3）— nofreq_t3000 完成：真实数据剂量单调 1.21→0.69→0.40
- **t3000（3 seeds）**：epoch-3 final gap = 0.88/0.71/0.47（mean **0.69±0.20**，−43%）；epoch 边界 [338,686] 与 baseline 一致
- **剂量单调**：baseline 1.21±0.28 → T=3000 0.69±0.20（bigram 键 −54%）→ T=8000 0.40±0.05（bigram 键 −87%）→ toy 剂量在真实数据复现
- 图 `docs/figs/nofreq_gap.svg` 更新为 3 组；guide §16.11 表格补 t3000 行；run_meta 由 `tools/finalize_nofreq.sh` 写（analyze_p12 + epoch 边界提取）
- **长时程对照（历史记录：当时运行中，后续已完成）**：baseline_long / nofreq_t3000_long / nofreq_t8000_long（2000 步，seed42，GPU 4/5/6）
- 清理：t5f_high16 三个 20G checkpoint 已删（分析已完成）；确认 nofreq runs 不覆盖 run_artifacts/epoch1_fork_checkpoint_s*.pt（EPOCH1_FORK_CHECKPOINT_PATH 未设置 → 写各自 run 目录）
- SSH 不稳定：长连接 ~3min 被断（port 50002 reset），改用短命令 + `login=false` 规避

### 8月4日（夜跑续4）— 长时程对照完成：激进去除稳健、部分去除只推迟
- **2000 步（≈6 epochs，seed42）**：baseline_long final gap **1.07**（= 1000 步水平，epoch3 后平台）；nofreq_t3000_long **1.23**（部分去除只推迟，剩余中低频键继续喂表、反超 baseline）；nofreq_t8000_long **0.28**（激进去除稳健）
- 解读：去除强度必须大到键空间塌缩（−87%）才能在更长重播下保持抑制；与 toy §16.8 off_low_long 同构（燃料越多 gap 越长）
- guide §16.11 新增长时程表；plan-1 §4.2 同步；run_meta 已写（finalize_nofreq.sh）

### 8月4日（夜跑续5）— P0 comb/rowzero 补 3 seeds + 论文统一表
- **comb（0–200+≥1001）3 seeds**：last100 gap = **0.149±0.002**（0.148/0.147/0.151）≈ 单遮 0–200（0.155±0.004）→ 高频遮罩只加「绝对 loss 代价」，低频遮罩的 gap 缩减保持（可加性 3-seed 确认）
- **rowzero freq-peak/large 3 seeds**：**1.175±0.090 / 1.179±0.087**（s42 1.096/1.108、s43 1.157/1.152、s44 1.273/1.277）→ 干净负结果确认（与 no-mask 1.068±0.112 无差异），支持「滞后行」机制
- 补跑方式：run_exp.sh 新增 6 个 case（comb/rowzero × s43/s44，fork 从 run_artifacts 对应 seed 的 epoch1 checkpoint；rowzero 需 `FIXED_PROBE_NGRAM_FREQUENCY_CAPTURE=1`，初版漏了 → ValueError 后修复重跑）；p0_analyze.py/p0_summary.py ARMS/GROUPS 补 s43/s44
- **论文统一表**：tools/compact_table.py LABELS 加入 nofreq（t3000/t8000 × 3 seeds）、long（2000 步 × 3 配置）、P0 补跑 6 runs → runs/compact_table.csv 已重生成，本地 `docs/figs/p0_compact_table.csv` 同步
- P0 图本地重生成（build_p0_figures.py，p0_data 拉取更新后的 summary/table）

### 8月4日（夜跑续6）— P1/P2 因果 3-seed 补跑完成（10 runs）+ 长时程图
- **P1/P2 3-seed 补跑（10 runs = 5 干预 × s43/s44，baseline_current 同 setting，1000 步，全部完成并分析）**：freeze table@e1 0.559/0.656/0.564（mean **0.593±0.054**，−50%）、freeze reader/backbone（table-gate-only）0.507/0.580/0.374（mean **0.487±0.105**，−59%）、freeze gate 0.865/1.234/0.838（mean 0.979±0.23，**−19% 弱**）、ref 行回滚 0.992/1.391/1.076、rand 行回滚 1.126/1.051/1.332（**跨 seed 均 null**）
- 结论升级：全表级干预（回滚/冻结表/冻结 reader/屏蔽 readout）跨 seed 稳健（−50%~−89%，两个 −89% 干预 std≤0.005）；**小规模行回滚 null 排除「回滚操作本身」artifact** → 历史效应是「全表内容」级的；gate 是次要放大器（3-seed 确认）
- guide §12.1 新增 3-seed 汇总表、§12.4 稳健性段落更新、§16.7/§17 数字更新（−49%→−50%）；compact_table.py LABELS +6 runs → compact_table.csv 重生成并同步本地 `docs/figs/p0_compact_table.csv`；run_meta 已写（tools/write_p1p2_meta.py，epoch 边界 [338,687]）
- **长时程去低频图（图 16.9）**：`docs/figs/nofreq_gap_long.svg/png` + 交互 `docs/interactive/fig_nofreq_long.html`（2000 步曲线 + epoch 边界虚线；曲线数据从集群 64MB obcurves 提取 train/val 两序列后拉回本地，避免拉大文件）
- **同 setting 干净去表对照（baseline_nongram，3 seeds，guide §15.6）**：baseline_current 同配置（1000 步）只关 ngram 注入（injection=none）→ final gap 0.124/0.124/0.126（mean **0.125±0.001**，−90%），train 停在 5.21、val 停在 5.33——主 setting 下 gap 全部由表贡献；run_exp.sh +3 case、compact_table +3 行、run_meta 已写
- 运行细节：run_exp.sh 新增 10 个 case（备份 `.bak_20260804_p1p2`）；3 波并行（GPU 1/5/6/7，每 run ~20min）；gate_freeze 首启因与 wave3a 争 GPU 6/7 显存失败，改为等 wave3a 完成后再启；wave3b 等待条件从 run_summary.json 改为 `[artifact_package] wrote`（train.py 不写 run_summary，需 analyze_p12 后补）

### 8月4日 — guide 新增 §15.0：svbird F 实验本身（人话完整版）
- 用户要求把 svbird 页面 F 部分写成 guide 里独立的重要实验章节：新增 `§15.0 svbird F 实验本身：它到底做了什么`（chap 15 开头，h3 裸小节与全书一致）
- 内容：一句话版本 → 为什么做（观测→因果）→ 同源对照原理（step-337 fork、组内严格配对、v3/20260729 只比形状）→ 主组数字表（3.779 → 1–5 联合 2.802/−25.9% → 0–200 0.514 → 0–1000 0.232）→ 20260803 组 epoch-2 数字 → 两个结论（低频 readout 直接贡献、高频遮罩=性能代价）+ 近似可加 → 证明/未证明边界 → 与本地三线收敛（P0 −86%/−92%、去表 −90%、去低频 −67%）
- 附带：plan-1 §5.1 章节状态表补 §15.0/§15.1-15.4/§15.5-15.6 行

### 8月4日 — 新增干净故事页 ngram-gap-story.html
- 用户要求：一个「假装完全不懂」也能读的完整故事，尽量简洁，综合双方实验、以本地结果为主
- 新建 `docs/ngram-gap-story.html`（17KB，零 JS 依赖）：TL;DR → 现象（p0_freqmask_curves）→ 三个嫌疑犯 → 玩具 2×2（t5s_2x2）+ 剂量（t5f_dose）→ 真实去低频（nofreq_gap{,_long}）→ 去表 −90% → 因果拆解（p12_3seed_bars）→ 合作方 svbird F 对照表 → 因果链图 + 口径
- guide 顶部加「想先看干净故事？」横幅链接；plan-1 §5.1 加故事页行

## 2026-08-08 收口核对（当前状态附录）

- Plan 1/2 的核心实验和 Phase 5 formal 可以标记完成；本线剩余主要是文档、claim ceiling 和论文 Introduction/Background 整理。
- vanilla graft 的 v4 formal 已有 9 个 primary arms、QC PASS；`SOURCE_MANIFEST`/README 中旧的 “v4 pending” 已按正式报告改成历史说明。
- `ngram-gap-lab` 的 1x v10 四行台账已回填为 done；epoch-aligned 4x–8x canonical `_e6` 仍缺结果，不能用 step-aligned `*_input_fv` 替代。
- 旧时间序列中的“未跑/运行中”条目保留为历史记录，不代表当前状态；当前判断见 `docs/closure-status-20260808.md`。
- BPE/Over-Encoding 相关实验目前是 token-remapping proxy，不是真正 BPE tokenizer 或 Engram/Over-Encoding 直接重训。
- Phase 5 的 exact packed-batch retention 为 null、packed-row retention 为 invalid/null；不要把它写成 exposure-2 retention 或跨模型 optimizer-state 必要性证据。
- full-163 远端 data generation 正在运行，`meta.json` 出现前不启动对应 DDP；已有 `ddp_smoke_20260808-171552` 使用旧 aligned-v2 数据，只算 preflight，不算 full-163 结果；当前不重复启动数据生成任务。
- 逐条可写入正文的表述见 `docs/claims-ledger-20260808.md`。
