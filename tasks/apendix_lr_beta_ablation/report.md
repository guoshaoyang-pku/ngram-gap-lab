# 附录 · 表学习率 × β₂ 消融（M3/M4 表优化器线的深挖）

> **实验线**：T-table-opt（表优化器消融）的子专题
> **状态**：🟡 进行中 —— 两个补点（β₂=0.99 · LR×1）在 ophis-gpu 跑（2026-08-24 10:36 启动，预计 ~50 min）
> **数据源**：`data/runs_fixed/*_fixed/`（post-fix 权威数据）；补点跑完经 `extract_data.py --remote` 拉回
> **代码**：`extract_data.py`（数据提取）+ `make_figures.py`（7 张图）；跑法见文末

---

## 0. 这个专题回答什么问题

1. **β₂ 消融**：表优化器（RMSProp，无动量 β₁=0）的 β₂ ∈ {0.98, 0.99, 0.999, 0.9999, 0.99999} 对 gap 的影响。
2. **表学习率消融**：`table_lr_scale` ∈ {1, 2, 4}（表实际学习率 0.004 / 0.008 / 0.016，**backbone 固定 0.004**）。
3. **两者的交互**：β₂ 的效应是否依赖表学习率？（用户从曲线簇观察到的关键线索）
4. **体检**：高表学习率是否真的合适？（val loss 曲线簇审查 —— 用户提出"我们之前可能没仔细 debug 过 LR"）

## 1. 核心发现（按时间顺序，含自我修正）

### F1 · β₂ 越小、gap 越大（方向确定）

所有 16 个 β₂ 专题 run（post-fix，配置真实生效）一致显示：**β₂ 降低 → 终点 gap 增大**。
例如 2-epoch shard · LR×2：β₂=0.98 → 2.10，β₂=0.99999 → 0.72（差 3 倍）。

**机制解释**：β₂ 控制梯度平方 EMA 的窗口。窗口短（β₂ 小）→ 稀疏行（低频 n-gram）的
梯度被更激进地更新 → 记忆更快 → gap 更大。β₂ 本质上是**低频行的学习率杠杆**。

### F2 · 表学习率越大、gap 越大，但伴随训练崩坏（用户怀疑被证实）

| 表 LR | val 最低 | val 结束 | train 结束 | 判定 |
|---|---|---|---|---|
| ×1（0.004） | 4.49 | 4.61 | 2.74 | ✅ 健康（1x shard） |
| ×2（0.008） | 4.75 | **7.05** | 1.55 | ⚠️ val 崩坏 |
| ×4（0.016） | 4.82 | **7.01** | 1.51 | ⚠️ val 峰值 10.8 |

（1x shard · β₂=0.99；2-epoch shard 同模式：×1 健康 3.52，×2 涨到 4.25，×4 峰值 10.8。）

**表学习率 ×2 以上 = 表写入过猛 = 过拟合**。这些高 LR run 的"大 gap"不能用来做 β₂ 比较——
因为训练本身已经坏了。

### F3 · 修正一个先前的表面结论（关键）

曲线簇上"β₂ 的曲线簇在表 LR×4 时几乎合并"，最初可能被读成"β₂ 在高表学习率下不重要"。
**这是误读**，正确解读是：

- 表学习率 ×4 把训练搞崩了（val → 10.8），所有 β₂ 都烂在一起，
  β₂ 的差异被"训练崩坏"这个更大的破坏淹没；
- β₂ 压差数据：1x · LR×2 时 β₂ 差 +1.01（+22%）；1x · **LR×4** 时只剩 +0.34（+6%）；
  2-epoch · LR×2 时 +0.36（+21%）；2-epoch · **LR×4** 时 +0.12（≈0）。
- 所以：**不是 β₂ 不重要，而是高表学习率的实验不可信**。表学习率 ×1 是唯一健康的设置。

### F4 · 推论：主线 β₂ 默认值的选择

- 主线表学习率 = ×1（0.004），这是唯一健康的表学习率。
- β₂=0.999（默认）与 β₂=0.99 的对照点（主线配置下）**正在补跑**（10:36 启动）。
- 已有侧面证据（短 epoch 家族，表学习率 ×1）：β₂=0.999 vs 0.99 的 gap 差异很小
  （0.25x：9.69 vs 8.97；0.5x：3.99 vs 4.02）。初步看 β₂ 0.99 与 0.999 差异不大。
- **待补点完成后**才能下主线结论；届时若差异 <20%，保持默认 0.999 即可（不动默认值）。

## 2. 全部图片（`figs/`，一图一变量）

每张图**只变一个变量**（其余固定项写在图副标题），每张图三个面板：**train loss / val loss / gap**，
虚线为 epoch 边界。A 组变表学习率，B 组变 β₂，C 组看交互。

| 文件 | 固定 | 变化 |
|---|---|---|
| `fig_lr_sweep_b2_099_1x.svg` | 1x shard · β₂=0.99 | 表学习率 ×1/×2/×4 |
| `fig_lr_sweep_b2_099_2ep.svg` | 2-epoch shard · β₂=0.99 | 表学习率 ×1/×2/×4 |
| `fig_lr_sweep_b2_0999_1x.svg` | 1x shard · β₂=0.999 | 表学习率 ×1/×2/×4 |
| `fig_lr_sweep_b2_0999_2ep.svg` | 2-epoch shard · β₂=0.999 | 表学习率 ×1/×2/×4 |
| `fig_b2_sweep_1x_lr2.svg` | 1x shard · 表学习率 ×2 | β₂ 0.98/0.99/0.999 |
| `fig_b2_sweep_2ep_lr2.svg` | 2-epoch shard · 表学习率 ×2 | β₂ 0.98→0.99999 |
| `fig_b2_sweep_1x_lr4.svg` | 1x shard · 表学习率 ×4 | β₂ 0.98/0.99/0.999 |
| `fig_b2_sweep_2ep_lr4.svg` | 2-epoch shard · 表学习率 ×4 | β₂ 0.98→0.9999 |
| `fig_beta2_spread_vs_lr.svg` | — | β₂ 压差（0.98−0.999）随表学习率衰减 |

## 3. 实验清单

### 已有（`data/runs_fixed/`，全部 _fixed 权威）

| 家族 | run | β₂ | 表 LR | 终点 gap | val 健康 |
|---|---|---|---|---|---|
| 1x·LR×2 | `nglab1x_opt_rmsprop_2x_b2_098_fixed` | 0.98 | 2 | 5.68 | ⚠️ |
| 1x·LR×2 | `nglab1x_opt_rmsprop_2x_b2_099_fixed` | 0.99 | 2 | 5.50 | ⚠️ |
| 1x·LR×2 | `nglab1x_opt_rmsprop_2x_fixed` | 0.999 | 2 | 4.67 | ⚠️ |
| 1x·LR×4 | `nglab1x_opt_rmsprop_4x_b2_{098,099}_fixed` + `4x_fixed` | 0.98/0.99/0.999 | 4 | 5.60/5.50/5.26 | ⚠️ val 峰值 10.7 |
| 2ep·LR×2 | `nglab2x_opt_rmsprop_2x_b2_{098,099}_fixed` 等 5 点 | 0.98→0.99999 | 2 | 2.10→0.72 | ✅ |
| 2ep·LR×4 | `nglab2x_opt_rmsprop_4x_b2_*_fixed` | 0.98→0.9999 | 4 | 2.22→2.10 | ⚠️ 峰值 10.7 |
| 基线 | `nglab1x_v10_input_fixed` / `nglab2x_input_v10_fv_fixed` / `nglab1x_v10_nogram_fixed` | 0.999 | 1 | 1.87/0.58/0.25 | ✅ |
| 短epoch | `nglab025x_b2_099_fixed` / `nglab05x_b2_099_fixed` 等 4 点 | 0.99/0.999 | 1 | 见 F4 | ✅ |

### 本附录新跑（2026-08-24）

| run | 设置 | GPU | 状态 |
|---|---|---|---|
| `nglab1x_opt_rmsprop_b2_099_lr1` | β₂=0.99 · 表 LR×1 · 1x shard · 2000 步 | GPU5 | 🏃 跑 |
| `nglab2x_opt_rmsprop_b2_099_lr1` | β₂=0.99 · 表 LR×1 · 2-epoch shard · 2000 步 | GPU7 | 🏃 跑 |

启动命令（集群 `/tmp/run_b2_099_lr1.sh`）：极简 setting + `--table_lr_scale 1.0 --table_betas 0.0,0.99`，
其余与 `run_injpos.sh` 完全一致（10 步间隔、fixed val batches、freq-bin eval）。

## 4. 对主线的行动建议

1. **表学习率保持 ×1**（0.004）。×2/×4 是训练崩坏设置，未来实验不再使用；
   此前基于它们的"β₂ 效应"读数按 F3 重新解释。
2. β₂ 默认值待补点：若 β₂=0.99 vs 0.999（主线配置）差异 <20%，**保持默认 0.999**（少改动）；
   若明显更小且无副作用，才切换。
3. `experiment-log.md` §9c/§9d 的 β₂ 段落待本附录补点结论后重写（plan-3 T2 的输入）。

## 5. 复现

```bash
cd ngram-gap-lab
.venv/bin/python tasks/apendix_lr_beta_ablation/extract_data.py            # 提取已有数据
.venv/bin/python tasks/apendix_lr_beta_ablation/extract_data.py --remote ophis-gpu  # 补点完成后拉回
.venv/bin/python tasks/apendix_lr_beta_ablation/make_figures.py            # 重画全部图
.venv/bin/python tasks/apendix_lr_beta_ablation/build_report.py            # 重建 report.html
```
