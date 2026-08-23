# ngram-gap-lab

Minimal clean reproduction of **n-gram value memory induced replay-specific train/val gap** on vanilla nanoGPT.

> **Agents / contributors start here: [`agents.md`](agents.md).**
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
| n-gram module | bigram + trigram value table, **`input` / wte injection** (over-encoding) |
| table size | **1M** (`vocab_size × 64 = 524,288` rows × 2 hash embeddings) — default, never changed |
| table optimizer | **RMSProp, no momentum**, betas `(0.0, 0.999)` (alternative β₂ = `0.99`) |
| backbone optimizer | AdamW `(0.8, 0.95)`, lr 0.004, wd 0.1 |
| data | fixed-order epoch replay, seed 42, 1000 / 2000 steps |
| evaluation | validation + freq-bin eval every **10 steps**, **fixed** validation batches |

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
├── code/
│   ├── train.py           # vanilla nanoGPT + n-gram table + 3 injection points (<1000 lines)
│   ├── ngram_freq.py      # per-frequency-bin loss statistics
│   ├── cluster/           # cluster launchers + env setup
│   ├── toy/               # pure numpy/torch theory scripts (no backbone dependency)
│   └── tools/             # corpus entropy, generator equivalence checks
├── ngram5_freq_gap/       # order-5 / trigram controlled experiment package
├── docs/
│   ├── plan.md            # phenomenon definition, ablation variables, experiment queue
│   ├── experiment-log.md  # ★ the single experiment registry
│   ├── claims-ledger.md   # ★ claim ledger (C1–C9, SUPPORTED / PROXY_ONLY / UNRUN)
│   ├── theory/            # unigram gap, power law, Markov, real long-tail correction
│   ├── literature/        # paper notes + related work + references.bib
│   ├── method/            # methodology and pitfalls (sawtooth audit, synthetic task design)
│   ├── plans/             # plan-1 mechanism, plan-2 literature story
│   ├── archive/           # historical docs (incl. deprecated current-shell results)
│   ├── plot_scripts/      # figure generation
│   └── figs/              # figures (figs/theory/ for toy & theory)
└── data/                  # gitignored: tokenized shards, freq index, run outputs, toy results
```

## Quick start

### On ophis-gpu

```bash
cd /data3/guoshaoyang/ngram-gap-lab
bash code/cluster/setup_env.sh                 # env (reuses existing torch)
bash code/cluster/run_injpos.sh 0 2000         # v/y/input/nogram ablation, 2000 steps
```

### Local CPU smoke test

```bash
pip install torch numpy
python code/train.py --run_id smoke --injection_position input --steps 10 \
  --data_dir /path/to/tokenized --device_batch_size 4 --total_batch_size 8192
```

### Frequency index

```bash
.venv/bin/python code/ngram_freq.py --data_dir <tokenized> \
  --train_shards 1 --vocab_size 8192 --out data/runs/<run_id>/freq_index.npz
```

## Public report

The authoritative write-up is the 9-chapter minimal-mainline version at
[ngram-gap-mechanism-guide](https://guoshaoyang-pku.github.io/blogs/ngram-gap-mechanism-guide/).
The old chapter 0–19 full version in the deprecated OPHIS repo is **not** maintained — see `agents.md` §2.

## Relationship to OPHIS

This repo is a clean extraction from the larger OPHIS codebase, whose `train.py` was 6966 lines with
current shell, Muon, RoPE, RMSNorm — all proven unnecessary for the gap phenomenon. **OPHIS_gap is
deprecated as of 2026-08-23**; theory notes, literature, methodology, toy scripts and figures have been
migrated here (see `agents.md` §7).

## License

Apache-2.0 (inheriting from OPHIS/nanoGPT upstream).
