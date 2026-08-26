# 废弃清单（不得进入主线）

> 本文件由 `agents.md` §6 offload（2026-08-26）。**判定标准是规则，保留在 agents.md §6**；
> 这里是具体的废弃项台账与「废弃结论，保留问题」。历史溯源专用，不作结论来源。

## 具体废弃项

| 内容 | 位置 | 原因 |
|---|---|---|
| `baseline_current` / `exp6_freqdecomp_current` | OPHIS `remote_training_runs/` | current shell，是同事的错误 setting；全库 11 个作图脚本 + 4 份结果文档都指向它 |
| `ngram-gap-regime-bridge.html` | OPHIS `docs/` | 同事的 current-shell 实验 |
| `ongoing_experiment/` 全目录 | OPHIS | 含 `exp4_hashreseed_current.log` 等 |
| `figB*` 系列图（~40 SVG） | OPHIS `docs/figs/`、`docs/interactive/` | Muon / nofork / rmsprop_freeze 专用 |
| `nanogpt_gap_causal/`、`nanogpt_gap_onset_source/`、`nanogpt_gap_vanilla_graft/`、`remote_gap_snapshot/` | OPHIS | **隐性污染**：不叫 current shell，但 parent 模型带 RoPE + RMSNorm |
| `ngram5_*_aligned_v2_final` | OPHIS `remote_training_runs/` | `run_contract.json` 显示 MuonAdamW + betas `(0.5, 0.999)` + val interval 50 |

## 「废弃结论，保留问题」

以下结论**方向可能是对的、问题很有价值，但数字全部建在 current shell 上，必须在极简 setting 重跑**：

| 问题 | 原结论（DEPRECATED SETTING） | 存档位置 |
|---|---|---|
| 表内容是不是 gap 的载体 | e2 边界 table 全量 reset → gap −89% | `docs/_archive/docs/p12-causal-results.md` |
| readout 通道是不是必要 | 屏蔽 readout → −89% | 同上 |
| table write vs backbone 各占多少 | 冻结 table −49%；冻结 backbone −54% | 同上 |
| 表大小是不是主导变量 | M=16 无碰撞点后 gap 饱和 → 不是参数量，是碰撞区低频涨落加权 | `docs/_archive/docs/table-size-sweep-results-20260811.md` |

重跑这些是当前最高价值的实验队列。
