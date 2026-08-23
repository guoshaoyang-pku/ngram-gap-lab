# Claims ledger

更新时间：2026-08-08。这个文件把当前可以写进正文的结论、证据范围和禁止越界的表述分开。

| ID | 当前表述 | 状态 | 证据范围 / ceiling |
|---|---|---|---|
| C1 | 在已验证的 `baseline_input` 与 matched vanilla-graft 协议内，n-gram value memory 足以形成 train/validation gap；这些协议不需要把 current shell 或 Muon 当作该现象的必要条件。 | `SUPPORTED_LOCAL` | 由 baseline/no-ngram、注入点、toy 和 9-run vanilla-graft formal 对照支持；不能推广为所有 plain NanoGPTOriginal、tokenizer、模型或训练协议都不需要其它组件。 |
| C2 | gap 的频率/重播效应可通过频率分桶、去低频代理、P0 mask、toy dose 和 epoch-aligned shard/replay 对照得到因果支持。 | `SUPPORTED_LOCAL` | 主要为当前 setting 的多 seed 结果；epoch-lag matched probe 和 frequency-stratified row zero 仍是可选补强，不应伪称已完成。 |
| C3 | token-remapping 去低频实验支持“减少被 n-gram memory 覆盖的键空间会削弱 gap”。 | `PROXY_ONLY` | 这是 BPE/over-encoding 的本地代理，不是真正 BPE tokenizer 或 Engram/Over-Encoding 直接重训；正文必须写 proxy。 |
| C4 | optimizer update semantics 会改变 gap 的跃迁形状和幅度；在本协议内，dense RMSProp state 不是形成大 gap 的必要条件。 | `SUPPORTED_LOCAL_WITH_CEILING` | Phase 5 为 21/21 runs、独立 QC 通过；exact packed-batch retention 为 null，packed-row retention 为 invalid/null，不能声称已证明 exposure-2 retention 或跨模型的 state 必要性。 |
| C5 | full-163 order-3 frequency experiment 已经完成。 | `UNSUPPORTED` | 远端 data generation 仍在运行，`meta.json` 尚未出现；已有的 4-GPU 120-step preflight 使用旧 aligned-v2 数据，只能证明 DDP/model/resume 路径。 |
| C6 | Engram/Over-Encoding 本身已经被当前实验直接验证。 | `UNRUN` | 目前只有机制对照和本地代理；是否启动 direct matched experiment 仍是独立决策。 |
| C7 | matched vanilla Transformer + hashed n-gram graft 在 3 seeds 下重现了明显 gap，且 full 相对 disabled/frozen 的 onset 和 final contrasts 通过 formal QC。 | `SUPPORTED_LOCAL_WITH_CEILING` | formal report 为 9 runs、QC PASS；参数量和 runtime 不匹配，结论只覆盖该 paired protocol，不是对所有 vanilla/Engram 设计的普遍证明。 |
| C8 | bottom-up shell ablation 已经证明各 current-shell 组件的必要性。 | `UNRUN` | positive control + 14 个 ablation 只有 runner，没有正式结果；建议先归档/标记 superseded。 |
| C9 | n-gram 表大小（参数量）对 gap 的定量关系：碰撞区（M<16, load>1）内 gap 随表大小单调涨，无碰撞点 M=16 后饱和，继续加参数量不再增大 gap。 | `SUPPORTED_LOCAL` | 2026-08-11 t5_low table-size sweep（M=1/4/8/16/64/256，seed 42，M=16/64 加 seed 7 复测）：headline gap 4.88→7.50→7.78→7.27；M=16 恰好 load=1.0 无碰撞。结论限当前协议（current shell + bigram/trigram RMSProp, 2000 step, t5_low），未覆盖 hash-salt、collision-free dictionary、overlap 直方图（P6 剩余项）。 |

## 台账冲突

- `ngram-gap-lab` 的 `nglab1x_v10_{v,y,input,nogram}` canonical logs 和总表现已 DONE；`input/train_log.jsonl` 仍保留旧波次 gap 1.9615，不能与 canonical `summary.json`/`train.log` 的 1.9308 混用。
- epoch-aligned canonical `nglab4x_e6`、`nglab5x_e6`、`nglab6x_e6`、`nglab8x_e6` 目前缺失；现有 `*_input_fv` 属于另一条 step-sweep。

## 写作规则

正文优先使用 `SUPPORTED_LOCAL` 的窄表述；`PROXY_ONLY` 必须显式标记 proxy；`UNRUN` 只能写成未验证或待决策；任何跨模型、跨 tokenizer、跨训练预算的普遍化都需要新的证据。
