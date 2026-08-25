# ngram-gap-lab

Minimal clean reproduction of **n-gram value memory induced replay-specific train/val gap** on vanilla nanoGPT.

> **Agents / contributors start here: [`agents.md`](agents.md).** Codex workflows
> for settings, experiment registration, and plotting are versioned with the
> repository in [`.agents/skills/`](.agents/skills/).
> It holds the working principles, the single source of truth for the minimal setting,
> cluster + storage coordinates, and the deprecation list.

## What this is

A standalone, clean repo that isolates one phenomenon: when a vanilla nanoGPT is augmented with a
trainable n-gram value table (bigram + trigram) and trained on a small dataset with fixed-order
multi-epoch replay, a train/val gap emerges. **1000 steps is enough to see the fork.**

This repo reproduces that gap with the **simplest possible setting** — no current shell, no Muon,
no RoPE, no RMSNorm, no momentum on the table optimizer.

## Minimal setting (SSOT: `agents.md` §1)

| | |
|---|---|
| backbone | vanilla nanoGPT — 8L · 6H · 768D, vocab 8192, learned abs position, LayerNorm, tied embeddings |
| n-gram module | bigram + trigram **clean single tables**, **`input` / wte injection** (over-encoding) |
| table size | clean-table `R_bigram = R_trigram = 2^20 = 1,048,576` outside a table-size experiment; never use legacy `table_mult` |
| table optimizer | **RMSProp, no momentum**, betas `(0.0, 0.99)` |
| backbone optimizer | AdamW `(0.8, 0.95)`, lr 0.004, wd 0.1, **constant LR** |
| data | fixed-order epoch replay, seed 42, **1000 default** / 2000 extended steps |
| evaluation | online current-batch train loss; fixed validation batches; validation + freq-bin eval every **10 steps** for full curves |

`v` (pre-attention value residual) and `y` (post-attention residual) injection exist **only as ablations**.

## Key finding

The gap depends on **whether the n-gram signal can effectively reach the output** without being
drowned by attention mixing.

| Injection | Style | Gap @2000 steps | Run |
|---|---|---|---|
| **`input`** | over-encoding (Engram / SCONE mainstream) | **1.93** | `nglab1x_v10_input` |
| `y` | post-attention residual (ResFormer variant) | 5.05 | `nglab1x_v10_y` |
| `v` | value residual (pre-attention, drowned by V) | 5.04 | `nglab1x_v10_v` |
| — | no n-gram (negative control) | 0.23 | `nglab1x_v10_nogram` |

## Repo structure

```
ngram-gap-lab/
├── agents.md              # ★ working principles + minimal setting SSOT + clusters + workspace
├── .agents/skills/         # ★ versioned Codex workflows for setting / registration / plotting
├── code/
│   ├── train.py           # vanilla nanoGPT + n-gram table + 3 injection points
│   ├── ngram_freq.py      # per-frequency-bin loss statistics
│   ├── cluster/           # canonical launcher index + historical launchers + env setup
│   ├── tools/              # corpus entropy, generator equivalence checks
├── tasks/                  # self-contained toy / mathematical validation tasks
├── ngram5_freq_gap/       # order-5 / trigram controlled experiment package
├── docs/
│   ├── plan.md            # phenomenon definition, ablation variables, experiment queue
│   ├── experiment-lines.md # experiment-line index + authoritative sources
│   ├── experiment-log.md  # ★ the single experiment registry
│   ├── claims-ledger.md   # ★ claim ledger (C1–C9, SUPPORTED / PROXY_ONLY / UNRUN)
│   ├── notes/             # theory, literature, method, and data notes
│   ├── plans/             # plan-1 mechanism, plan-2 literature story
│   ├── _archive/          # historical docs (incl. deprecated current-shell results)
│   ├── plot_scripts/      # figure generation
│   └── figs/              # figures (figs/theory/ for toy & theory)
└── data/                  # gitignored: tokenized shards, freq index, and run outputs
```

## Quick start

### On a GPU cluster

```bash
cd /path/to/ngram-gap-lab
bash code/cluster/setup_env.sh                 # env (reuses existing torch)
bash code/cluster/run_baseline.sh 0 <run_id>
```

`run_baseline.sh` is the new main-line entry point: input injection, LR 0.004,
RMSProp `(0.0, 0.99)`, table LR ×2, **constant** LR, bf16, 1000 steps, online
train loss, and fixed validation. Non-table-size experiments lock both clean
table capacities to `2^20`; a table-size line must name its alternative R
values in a dedicated launcher. The legacy injection-point launcher remains
for historical provenance, not as a new clean-table default.

### Local CPU smoke test

```bash
pip install torch numpy
python code/train.py --run_id smoke --injection_position input --steps 10 \
  --data_dir /path/to/tokenized --device_batch_size 4 --total_batch_size 8192 \
  --n_layer 1 --n_head 1 --n_embd 16 --sequence_len 32 --dtype fp32 \
  --bigram_clean_table 64 --trigram_clean_table 64
```

This CPU command is a non-evidence smoke test only; it deliberately changes
the model and table sizes from the research contract.

### Frequency index

```bash
.venv/bin/python code/ngram_freq.py --data_dir <tokenized> \
  --train_shards 1 --vocab_size 8192 --out data/runs_fixed/<run_id>_fixed/freq_index.npz
```

## Public report

The authoritative write-up is the 9-chapter minimal-mainline version at
[ngram-gap-mechanism-guide](https://guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide/).
The old chapter 0–19 full version is **not** maintained — see `agents.md` §2.

## Relationship to the predecessor codebase

This repo is a clean extraction from a larger predecessor codebase, whose `train.py` was 6966 lines
with current shell, Muon, RoPE, and RMSNorm. Those components are not part of the minimal evidence
base; the theory notes, literature, methodology, toy scripts, and figures needed here have been
migrated into this repository (see `agents.md` §7).

## License

Apache-2.0 (inheriting from the nanoGPT upstream).
