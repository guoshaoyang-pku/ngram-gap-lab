# OPHIS_gap 迁移记录

> 本文件由 `agents.md` §7 offload（2026-08-26）。一次性迁移记录，仅历史溯源。
> `OPHIS_gap` 已弃用，本仓库是唯一开发仓库（`agents.md` §5）。

## 已迁入本仓库（2026-08-23）

| 迁入位置 | 来源 | 内容 |
|---|---|---|
| `docs/notes/theory/` | `docs/theory_notes/` + `markov-unigram-exact-gap-20260811.md` | 5 篇纯理论推导，零 backbone 依赖 |
| `docs/notes/literature/` | `docs/literature/` + 4 篇顶层长综述 | 9 个文件，含 arXiv 复核过的 `references.bib` 与可直接进论文的 related work |
| `docs/notes/method/` | sawtooth 审计、合成任务设计、排除台账 | 方法论与踩坑 |
| `docs/plans/` | `plans/` | plan-1 机制总纲（§3.1a 是极简 setting 的原始定义）、plan-2 文献故事线 |
| `docs/claims-ledger.md` | `docs/claims-ledger-20260808.md` | C1–C9 断言台账 |
| `docs/_archive/docs/` | closure-status、p12-causal、table-size-sweep、injpos-log、manual 工作日志 | 历史溯源 |
| `tasks/l1..l5/` | `toy/` + `toy/results/` | 9 个纯 numpy/torch 脚本 + 结果，全库唯一零 current-shell 污染的代码 |
| `code/tools/` | `tools/` | 语料熵计算、生成器等价性校验 |
| `docs/figs/theory/` | `docs/figs/` 中的 markov / gap_vs_samples / synth 系列 | 理论图 |
| `tasks/*/results/` | `toy/results/` | L1 主矩阵、L2 三个 markov 臂、L5 五臂对照 |
| `data/injpos_*.json` | `remote_training_runs/` | injpos obs summary + 2000 步延长数据 |

## 未迁移（留在 OPHIS_gap，只读溯源）

- 2.0 GB 的 `remote_training_runs/`（其中 `ngram5_trigram_full_trace/` 单目录 1.7 GB 是一次性调试 trace）
- 所有 current shell / Muon / RoPE 系源码目录与结果
- `ngram5_freq_gap/` 代码（本仓库已有同一份，仅 `lib.py` 存在无逻辑差异的格式差异）
- injpos v/y/input 的 `train.log`（本仓库 `nglab1x_v10_*` 已是更新波次）

⚠️ 已知冲突：OPHIS 的 `injpos_ablation_data.json`（292 KB）与本仓库 `data/injpos_ablation_data.json`（40 KB）不是同一波次。`claims-ledger.md` 记录：`input/train_log.jsonl` 旧波次 gap 为 1.9615，canonical `summary.json` / `train.log` 为 1.9308，**不能混用**。引用 injpos 数字时以本仓库 `summary.json` 为准。
