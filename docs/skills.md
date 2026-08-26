# ngram-gap-lab 技能注册表

本页登记本项目常用的 Agent skills，作为仓库内的导航副本。技能的完整行为定义仍以
`.agents/skills/` 下对应的 `SKILL.md` 为准；项目的实验规则与极简 setting 以
`agents.md` 为准。

## 技能总览

| 技能 | 主要用途 | 什么时候使用 |
|---|---|---|
| `ngram-gap-settings` | 定义、审计或修改训练 setting | 新 baseline、ablation、launcher 审计，以及跑实验前核对极简契约 |
| `ngram-gap-experiment-registration` | 登记、交接和回填实验 | 创建 `run_id`、准备 GPU handoff、记录运行状态、把结果变成可追溯 evidence |
| `ngram-gap-plotting` | 生成和审查证据可追溯图表 | loss/gap、frequency、table occupancy、scaling 图，以及将图嵌入报告前 |
| `ngram-gap-rerun-v4` | 按当前 v5 clean-table 标准执行重跑 | 注入点、剂量、table optimizer/LR/β₂、因果和 table-size 重跑；名称虽为 legacy，setting 不是 legacy |

## 各技能职责

### `ngram-gap-settings`

它负责回答“这次实验到底应该用什么 setting”，并检查是否只改变了当前要检验的
一个变量。核心检查包括：

- 以 `agents.md` §1 的极简 setting 为 SSOT；
- 显式写出 clean single table 的 `R`，非 table-size 实验默认
  `R_bigram = R_trigram = 2^20`；
- 锁定 `input` 注入、vanilla nanoGPT、LayerNorm、learned absolute position、
  RMSProp `(0.0, 0.99)`、table LR scale `2.0`；
- 显式传 `--lr_schedule warmup_constant --warmup_steps 100`，不引入 warmdown；
- 保持主测量为同一 logged step 的
  `online gap = fixed validation loss - online current-training-batch loss`；
- 发现模型、测量语义或架构变化时，要求使用新的 `run_id`。

它不负责登记结果，也不负责从历史数字倒推出 setting。

本地定义：`.agents/skills/ngram-gap-settings/SKILL.md`。

### `ngram-gap-experiment-registration`

它负责把一个实验变成可执行、可审计的记录，而不是只写一段口头 setting。主要工作
包括：

- 在占 GPU 前，将实验登记为 `planned`；
- 建立一一对应的 `run_id`、结果目录、登记表行和详细 section；
- 记录科学问题、可证伪比较、owner、机器/GPU、seed、终点和完整命令；
- 写清 train/validation/frequency-index 路径，并证明 train/val 不重叠；
- 多机实验前记录 git commit 和关键代码的 `md5sum`；
- 按 `planned → running → done` 或 `stalled` 更新生命周期；
- 完成后核验 `summary.json`、`train_log.jsonl`、最终 step、seed、指标和诊断产物；
- 向 plotting 技能交接 run ID、artifact 路径、比较关系、指标定义、seed 数和 evidence 状态。

权威主线结果放在 `data/runs_fixed/<run_id>_fixed/`；scaling 结果使用明确的
`data/runs_scaling/` namespace，不能覆盖已完成 run。

本地定义：`.agents/skills/ngram-gap-experiment-registration/SKILL.md`。

### `ngram-gap-plotting`

它负责把已记录的实验产物转成证据可追溯的图，并防止把无效数据画成主结论。主要工作
包括：

- 只使用符合 `agents.md` 规定的 canonical run 和 `_fixed` 产物；
- 先核对 run 的 config、final step、seed、表架构和测量语义；
- 从 JSONL 或其他记录产物读取数值，不手填结果数字；
- 使用主口径 online gap，不用 `fixed_train_loss.jsonl` 替代；
- 将脚本放在 `docs/plot_scripts/`，将生成图放在对应的 `docs/figs/<line>/`；
- 在 caption 或相邻说明中标出 run ID、step/endpoint、seed 数、指标定义和变化项；
- 对 dense curve、final-only endpoint、novel frequency context 和历史/deprecated
  artifact 做清晰区分；
- 生成后检查图例、坐标轴、引用路径、HTML/SVG 嵌入和 `git diff --check`。

它不负责凭 prose 或历史数字制造图，也不负责未经授权修改或发布 blog。

本地定义：`.agents/skills/ngram-gap-plotting/SKILL.md`。

### `ngram-gap-rerun-v4`

这是一个 legacy-named launcher skill，但它服务的是当前 v5 clean-table 标准，不应被
误解为允许使用旧表架构。它适合需要实际重跑或检查重跑条件的任务，包括：

- injection arms；
- dose scans；
- table optimizer、table LR、β₂ ablations；
- causal runs；
- table-size sweeps。

它应锁定当前标准：backbone LR `0.0006`、`warmup_constant`、100-step warmup、
table betas `0.0,0.99`、table LR scale `2.0`、bf16、no compile 和 clean-table
容量显式传参。它还负责同步代码、核对 hash、登记/回填 run，并拒绝 legacy table。

该技能的名称保留是为了兼容既有工作流；新实验仍必须遵守 `agents.md`，并先经过
setting 审计与 experiment registration。当前运行环境提供此技能，但本仓库目前没有
对应的 `.agents/skills/ngram-gap-rerun-v4/` 本地目录。

## 推荐调用顺序

有实际计算的实验按以下顺序执行：

```text
ngram-gap-settings
        ↓
ngram-gap-experiment-registration
        ↓
ngram-gap-rerun-v4（需要启动/重跑时）
        ↓
ngram-gap-experiment-registration（running → done 回填）
        ↓
ngram-gap-plotting（需要图表时）
```

只做 setting 检查时不必启动后续技能；只做历史结果叙述时也不应把
`ngram-gap-plotting` 当作数据来源。任何结论都必须回到具体 `run_id`、logged step、
seed 数和 `docs/claims-ledger.md` 的 evidence 状态。

## 相关入口

- 项目规则与 setting SSOT：[`agents.md`](../agents.md)
- 实验线全景：[`experiment-lines.md`](experiment-lines.md)
- 实验登记簿：[`experiment-log.md`](experiment-log.md)
- 断言台账：[`claims-ledger.md`](claims-ledger.md)
- 作图规范：[`plot_scripts/README.md`](plot_scripts/README.md)
- 技能源码目录：[`../.agents/skills/`](../.agents/skills/)