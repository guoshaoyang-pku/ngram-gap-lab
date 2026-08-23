# P1/P2 因果拆解结果（2026-08-02）

> ⛔ **[DEPRECATED SETTING]** 本文档的实验建立在 `current shell` backbone 上，**不是本课题的极简 setting**。
> 结论方向可能正确，但所有数字都必须在极简 setting（vanilla nanoGPT + `input`/wte 注入 + table 1M + RMSProp 无动量）下重跑才能进主线。
> 极简 setting 定义见 `agents.md` §1；重跑队列见 `agents.md` §6.3。本文件仅供历史溯源。


> 统一 setting：`baseline_current`（current shell + current-style grouping + bigram/trigram RMSProp，seed42，1000 step，vocab=8192，n_layer=8，epoch 边界 [338,687]）。
> 所有 run 均在集群 ophis-gpu 的 `/data3/guoshaoyang/ngram-gap-exp/runs/` 下；干预在 epoch 边界激活（step 338 = e1→e2，step 687 = e2→e3）。
> gap 口径与 §7.9 一致：`val_loss − raw_train_loss`（val 每 50 步采样）。
> 观测口径：与 baseline_current 相同 setting + 完整 theory obs（direct kernel / row history / history ablation），观测本身为 no_grad 探针。

## run 矩阵与结果

| run | 干预（激活点） | final gap@1000 | peak gap e3 | vs 同 obs 控制 |
|---|---|---|---|---|
| `baseline_current`（同 obs 控制） | 无 | 1.096 | 1.096 | — |
| `baseline_current_orig_backup`（原基线） | 无（无 direct-kernel obs） | 1.330 | 1.330 | （obs 口径差异，见 §备注） |
| `p1_reset_all_e1` | e1 边界全 table 行回滚到 init（restore_initial, selector=all） | 0.956 | 0.980 | −13% |
| `p1_reset_all_e2` | e2 边界全 table 行回滚到 init | 0.121 | 0.610 | **−89%** |
| `p1_reset_ref_e1` | e1 边界仅 seen_train_probe 引用的行回滚 | 0.992 | 0.992 | −9% |
| `p1_reset_rand_e1` | e1 边界随机等量行回滚（对照） | 1.126 | 1.126 | +3% |
| `p2_freeze_table_e1` | e1 边界冻结 bigram/trigram table（保留 e1 内容） | 0.559 | 0.559 | −49% |
| `p2_table_gate_only_e1` | e1 边界冻结 reader/backbone（仅 table+gate 更新，97 个张量冻结） | 0.507 | 0.507 | **−54%** |
| `p2_gate_freeze_e1` | e1 边界冻结 gate（7 个张量） | 0.865 | 0.865 | −21% |
| `p2_readout_mask_e1` | e1 边界屏蔽 bigram/trigram readout（NGRAM_BRANCH_MASK） | 0.116 | 0.135 | **−89%** |

epoch bpb 摘要：控制 e1=1.820 / e2=1.585；`p1_reset_all_e2` e1=1.781 / e2=1.566；`p2_readout_mask_e1` e1=1.786 / e2=1.810（屏蔽 readout 后 e2 val 变差，符合预期）。

## 因果结论

**P1 · historical row-state 是必要的**
- e2 边界把全部 table 行回滚到 init 后，e3 gap 从 1.10 → 0.12（−89%）。说明 e3 的大 gap 依赖 e1/e2 累积的 table 行内容；行被擦掉后即使 e3 重新写入 313 步，也无法重建同量级 gap。
- e1 边界回滚只有 −13%（e2 会重新写入，e3 时行历史又恢复）；ref 行回滚（−9%）与随机行回滚（+3%）差异很小，说明「回滚少量行」的效应主要是行 norm/scale 层面的，特异性证据弱——但 e2 全量回滚的 −89% 是内容层面的强证据。

**P2 · table write 与 reader/backbone 各贡献约一半**
- 冻结 table（保留 e1 内容，backbone 继续训练）：gap 1.10 → 0.56（−49%）。持续写 table 不是 gap 的全部来源；已有行内容 + backbone/reader 放大可单独撑起约一半。
- 冻结 reader/backbone（仅 table+gate 更新）：gap 1.10 → 0.51（−54%）。table 写入单独也只能撑起约一半。
- 两者合起来说明：gap = 历史行内容/写入 × reader/backbone 放大，二者缺一不可（各 ~50%）。
- 屏蔽 readout（e1 后 bigram/trigram 完全不注入）：gap → 0.12（−89%）。n-gram readout 通道是 gap 的必要传导口（对应 F2 injection/reader）。
- 冻结 gate：−21%，gate 是次要放大器，不是必要条件。

## 备注（可复现性口径）

- **obs 探针扰动**：direct kernel / history ablation 的有限差分会对 table 做 in-place 写回，引入轻微 run-to-run 不确定（同配置重跑 final gap 波动 ~±15%，e1 bpb 波动 ~0.1）。原因待查（疑似 in-place index_add_ + fused kernel 的非确定性归约）。所有干预 run 与控制 run 使用同一 obs 配置，比较内部一致；效应量（−49%~−89%）远大于该噪声。
- 重跑确认（同配置）：
  - `p1_reset_all_e1_rerun` final gap 0.828（首跑 0.956，−13%~−24%）
  - `p1_reset_all_e2_rerun` final gap 0.123（首跑 0.121）— 高度可复现
  - `p2_readout_mask_e1_rerun` final gap 0.119（首跑 0.116）— 高度可复现
- 原 baseline（`baseline_current_orig_backup`，无 direct-kernel obs）final gap 1.330；同 obs 控制 1.096。已有图表（§2/§7.9/§9）基于原 baseline 数据，口径说明见 plan-1。

## 产物
- `docs/figs/fig_p12_data.json`、`docs/figs/fig_p12_gap_curves.svg`、`docs/interactive/fig_p12_causal.html`
- 集群原始数据：`runs/{p1_*,p2_*}/observable_curves.obcurves.json`、`run_summary.json`
- 构建脚本：`tools/analyze_p12.py`、`tools/build_p12_figures.py`（run_exp.sh 新 case 在集群，未同步到本地仓库）

---

## 第二波（2026-08-02 追加）：3-seed 稳健性 + freeze_both + exp7 诊断

### 3-seed 稳健性（关键干预，seed42/43/44）

| run | seed42 | seed43 | seed44 | rerun | 区间 |
|---|---|---|---|---|---|
| `baseline_current`（控制） | 1.096 | 1.530 | 0.997 | — | 1.096–1.530 |
| `p1_reset_all_e2` | 0.121 | 0.123 | 0.122 | 0.123 | 0.121–0.123 |
| `p2_readout_mask_e1` | 0.116 | 0.121 | 0.126 | 0.119 | 0.116–0.126 |

- 两个 −89% 关键干预在 3 seed 下均 < ±0.01 波动，稳健。
- 控制组本身跨 seed 波动大（0.997–1.530，训练动力学敏感），但干预后塌缩到 ~0.12 的行为是 seed 不变的。

### p2_freeze_both_e1（table + gate 一起冻结）

- final gap 0.572 ≈ p2_freeze_table_e1（0.559），gate 冻结不叠加 → gate 是次要放大器（与 p2_gate_freeze_e1 −21% 一致）。

### exp7：over-encoding 干预验证（gate 按频率清零，含 novel）

| run | 干预 | final gap@1000 | vs 控制 |
|---|---|---|---|
| `baseline_current` | 无 | 1.096 | — |
| `exp7_overencode_th200` | gate 清零 bigram/trigram 低频 (0,200]，含 novel | 1.280 | +17% |
| `exp7_overencode_th1000` | gate 清零 (0,1000] | 1.686 | +54% |

**干预未消除 gap，反而略增。** 代码确认 mask 在 train/eval forward 中正确作用于 gate 输出
（`value_gate.masked_fill`，`train.py` `_add_value_residual`）；`train.log` 确认 ranges 加载含 novel（lower=0）。

**分桶诊断（global_frequency_probe，step 1000，口径 val−train）**：

| 桶（train 频次） | 控制 gap | exp7 th200 gap | exp7 th1000 gap |
|---|---|---|---|
| bigram ≤200（masked） | +1.40 | +1.66 | +2.10 |
| bigram >200（unmasked） | +0.68 | +0.98 | +1.28 |
| trigram ≤200（masked） | +1.07 | +1.34 | +1.72 |
| trigram >200（unmasked） | +0.04 | +0.29 | +0.41 |

末步 loss 对照：控制 train 3.376 / val 4.471；exp7 th200 train 3.227 / val 4.506；exp7 th1000 train 2.977 / val 4.663。

**机制解释（为什么 gate 侧频率剪枝无效）**：
1. gate 是学出来的，输入侧（hidden state）不含频率信息——它不是频率干预的有效位点；它本就会自行抑制部分低频注入。
2. 强制清零低频 gate 后，固定顺序多 epoch replay 下模型把这些 train 位置的记忆**转移给 backbone**：
   train loss 不升反降（3.38 → 2.98），而 val 不受益（val 低频上下文本来就靠行内容弱帮助，清零后连这点帮助也消失，val 4.47 → 4.66）→ gap 维持或扩大。
3. 与 P1/P2 对照：把**整个 readout 通道**关掉（`p2_readout_mask_e1`，−89%）有效，因为那是从传导口整体断开记忆通道；
   按频率在 gate 上局部剪枝无效，因为频率结构存在于**表/行内容与 train 数据统计**（P1 e2 回滚 −89%、toy gap(r) 单调），不在 gate 里。
4. 结论：BPE/over-encoding 的 gate-zero 代理**不是有效干预**；「真正 BPE 合并高频组合/低频共享行」仍属未验证项
   （需要新实验：低频 context 合并到共享行或直接改 tokenizer）。有效方向是表/行层面频率结构（P1、toy）+ epoch hash 错位（exp4）。
