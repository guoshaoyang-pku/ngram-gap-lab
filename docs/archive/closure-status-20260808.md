# OPHIS gap closure status

> 🗄️ **[ARCHIVE]** 本文档来自已弃用的 `OPHIS_gap` 仓库，仅供历史溯源。
> 其中部分结论建立在 `current shell` / Muon / RoPE 等非极简 setting 上，引用前请对照 `agents.md` §6。


更新时间：2026-08-08  
审计口径：只把有实际证据或明确下一道 gate 的工作算作项目状态；旧 queue、被排除 run、历史 pending 文案不自动算作 backlog。

## 总结

当前有 4 条仍需动作的收口线，另有 1 条 bottom-up shell 线应先做“归档或复活”的决策。vanilla graft 已由 9-run formal QC 关闭。本轮没有启动新的 GPU 训练或修改远程状态。

| 线 | 状态 | 已核实事实 | 下一道 gate |
|---|---|---|---|
| `ngram-gap-lab` 清洁复现 | `WAITING_EXTERNAL` | `nglab1x_v10_{v,y,input,nogram}` 的 logs 和台账都已回填为 done；epoch-aligned 4x–8x canonical `_e6` 仍没有完整本地结果 | 确认或补齐 4x–8x `_e6`，处理 input 旧波次 metadata 冲突，再重生成报告图表 |
| OPHIS Plan 1/2 主线 | `DOCS_PENDING` | 核心机制、P0/P1/P2、toy、shard/replay、beta scan 和 phase5 证据已经有记录；claims ledger 已建立，Plan 1 §10/§11 和 Plan 2 Introduction/Background 尚未完全收口 | 以现有 claims ledger 为准，更新旧 manual 段落，完成论文 Introduction/Background；单独决定是否做 Engram 直接实验 |
| `ngram5_freq_gap` full-163 | `DATA_GEN_RUNNING / PREFLIGHT_DONE` | 本地 smoke 只有 20 steps、order=3、vanilla fallback；远端 full-163 data generation 已在运行；另有一个 4-GPU 120-step preflight 已完成，但 contract 指向旧的 `aligned_v2_20260806` 数据，不是 full-163 | 只监控生成日志并等待 `meta.json`；用 full-163 数据目录重新做 DDP smoke，再决定正式长跑 |
| vanilla control | `QUEUED_UNRUN` | queue 和固定 validation identity 流程已准备，本地没有正式 run 结果 | 若仍需要外部 validity control，先跑 smoke+1000；否则明确 deferred |
| vanilla graft | `DONE_FORMAL` | formal report 有 9 runs（disabled/frozen/full × seeds 42–44），QC `PASS`；paired onset 和 full-vs-disabled/frozen contrasts 已完成 | 不再启动；引用 formal report，并把 source manifest/README 的 v4 pending 改为历史说明 |
| bottom-up shell ablation | `DECISION_ONLY` | positive control 加 14 个 one-factor runner 都已实现，但没有正式结果；这条线可能已被更简洁的 baseline/P1/P2 证据取代 | 明确写入 archived/superseded，或重新授权后才启动正例和 14 个 ablation |

## 已完成，不再作为待收口项目

- Phase 5 optimizer semantics：21/21 planned runs complete，独立 formal analyzer 21/21 valid，不能再按“待跑实验”统计。
- vanilla graft formal：9/9 arms complete，QC `PASS`，不能再按“待跑实验”统计；正式结果见 `nanogpt_gap_causal/reports/analysis_vanilla_graft_formal_20260716_v3/VANILLA_GRAFT_FORMAL_ANALYSIS.md`。
- 注入点 observable 重跑：`injpos_obs_summary.json` 已覆盖 input/y/v 三组，step 10–1000 的 norm/grad 点和 gap 曲线已有本地交付物，不再按“进行中/排队”统计。
- Plan 1 的 P0/P1/P2、toy 2×2/dose、BPE/remapping、Zipf 和主要 beta/optimizer 扫描已有结果。
- 历史 Phase34 中标记为 `CANCELLED / DO NOT LAUNCH` 的 queue 只保留为排除记录，不恢复执行。

主线 claim ceiling：当前 BPE/Over-Encoding 证据是 token-remapping/本地等价代理，不是真正 BPE tokenizer 或 Engram/Over-Encoding 直接重训；正文应写成 proxy。Phase 5 的 exact packed-batch retention 为 null、packed-row retention 为 invalid/null，因此不能声称已经证明 exposure-2 retention 或跨模型的 optimizer-state 必要性。

## 推荐执行顺序

1. 先收 `ngram-gap-lab`：1x v10 ledger 已关闭；剩余是获取/确认 4x–8x epoch-aligned `_e6` 状态并处理 input 旧波次 metadata 冲突。
2. 同步收 OPHIS 文档：沿用现有 claims ledger，明确“已验证、代理验证、未做”的边界。
3. full-163 当前已有远端 data generation 在运行；等待 `meta.json` 后用新数据目录做 DDP smoke/长跑 gate，不重复启动生成任务，也不把旧 aligned-v2 preflight 写成 full-163 结果。
4. vanilla control 和 bottom-up shell 保留为显式决策门，不自动消耗 GPU；vanilla graft 已关闭。

## 交付风险

当前 OPHIS 主 checkout 有 14 个 tracked files 被修改，并有大量未跟踪 figures、runs 和辅助脚本；`ngram-gap-lab` 也有独立的 dirty worktree。本报告只新增状态文档，没有对这些既有改动做 reset、stash、批量 stage 或覆盖。

本地环境缺少 PyTorch/pytest，因此本次自动化只适合做文件、日志、JSON、shell 和 Python 语法级门禁；CUDA 结果必须保留远端证据。

## 远端只读核验

2026-08-08 21:17（CST）通过 `ophis-gpu` 只读检查确认：

- `/data3/guoshaoyang/ngram-gap-exp/ngram5_data/trigram_exact_alpha0.0_full163_20260808/meta.json` 尚不存在。
- `ngram5_data/gen_full163.log` 已存在，data generation PID `4131043` 仍在运行（本次快照 elapsed `04:12:10`），命令为 order=3、alpha=0.0、5,000,000 buckets、doc_len=2048。
- `runs/ngram5_big/ddp_smoke_20260808-171552/summary.json` 和 `v2/summary.json` 均为 `status=complete`，但其 contract 的 `source_dataset` 为旧的 `trigram_exact_alpha0.0_aligned_v2_20260806`；这只是 DDP/model/resume preflight，不是 full-163 证据。
- 当前核验时没有匹配到正在运行的 `trainer.py`/DDP 进程；不能把旧 smoke 记成 full-163 已启动或已完成。
- 本次没有启动、停止、重启或修改任何远端任务。

## 本轮自动门禁

- `ngram5_freq_gap` Python 文件：`py_compile` 通过；四个 shell launcher：`bash -n` 通过。
- vanilla control/graft queue JSON：全部解析通过。
- vanilla graft unittest：19 个用例运行，9 个通过、10 个因本地没有 PyTorch 跳过。
- vanilla control unittest：未能导入，原因是本地没有 PyTorch；这不是 control 实验结果。
- `git diff --check` 仍被已有 `nanogpt_gap_causal/lib.py` 修改中的 CRLF/尾随空白触发；本轮没有重格式化或覆盖该文件。
