# ngram-gap-lab

Minimal clean reproduction of **n-gram value memory induced replay-specific train/val gap** on vanilla nanoGPT.

## What this is

A standalone, clean repo that isolates one phenomenon: when a vanilla nanoGPT is augmented with a trainable n-gram value table (bigram + trigram) and trained on a small dataset with fixed-order multi-epoch replay, a train/val gap emerges. This repo reproduces that gap with the **simplest possible setting** — no current shell, no Muon, no RoPE, no RMSNorm.

## Key finding

The gap depends only on **whether n-gram signal can effectively reach the output** (not get drowned by attention mixing). Three injection points:

| Injection | Style | Gap @1000 steps | Mechanism |
|---|---|---|---|
| `input` | over-encoding (Engram/SCONE mainstream) | 0.64 | n-gram value added to wte output, no attention |
| `y` | post-attention residual (ResFormer variant) | 1.82 | n-gram value added after attention, no mixing |
| `v` | value residual (ResFormer, pre-attention) | 0.60 | signal drowned by V (only 6.5% of V norm) |

## Repo structure

```
ngram-gap-lab/
├── code/                 # training code (decoupled from docs)
│   ├── train.py          # <1000 lines: nanoGPT + n-gram table + 3 injection points
│   ├── ngram_freq.py     # clean per-frequency-bin loss statistics
│   └── cluster/          # cluster launchers + env setup
├── docs/                 # experiment plans, logs, plots
│   ├── plan.md           # standard setting + todo
│   ├── experiment-log.md # experiment registry
│   └── plot_scripts/     # figure generation scripts
└── data/                 # gitignored: tokenized shards, run outputs
    ├── tokenized/         # reuse existing shards (not re-generated)
    └── runs/              # training logs, checkpoints, freq stats
```

## Quick start

### On ophis-gpu cluster

```bash
# 1. clone
cd /data3/guoshaoyang
git clone git@github.com:guoshaoyang-pku/ngram-gap-lab.git
cd ngram-gap-lab

# 2. setup env (reuses torch from existing venv)
bash code/cluster/setup_env.sh

# 3. run v/y/input ablation (serial, ~6 min/run on 1 H200)
bash code/cluster/run_injpos.sh 0 1000

# 4. doubled training-size replay experiment (three GPUs, 2000 steps)
bash code/cluster/run_train2x.sh 5,6,7 2000

# 5. build frequency index + per-bin loss stats
.venv/bin/python code/ngram_freq.py \
  --data_dir /data3/guoshaoyang/ngram-gap-exp/data \
  --train_shards 1 --vocab_size 32768 \
  --out data/runs/nglab_input/freq_index.npz
```

### Local (CPU smoke test)

```bash
pip install torch numpy
python code/train.py --run_id smoke --injection_position input --steps 10 \
  --data_dir /path/to/tokenized --device_batch_size 4 --total_batch_size 8192
```

## Standard setting (baseline_input)

See `docs/plan.md` for full spec. Core config:
- vanilla nanoGPT (8 layer, 6 head, 768 dim, vocab 32768)
- bigram + trigram n-gram value table (hash + embedding + gate)
- injection point: `input` (over-encoding to wte)
- optimizer: mixed (RMSProp for table, AdamW for backbone)
- table betas: (0.0, 0.999) — no momentum, β₂ persists across epochs
- fixed-order epoch replay, seed 42, 1000 steps

The doubled-training-size experiment uses train shards `1,2`, runs for 2000
steps, and writes complete `train_log.jsonl`, `table_norm.jsonl`,
`freq_bin_loss.jsonl`, `summary.json`, and `train.log` files under
`data/runs/nglab2x_{v,y,input}/`.

## Relationship to OPHIS

This repo is a clean extraction from the larger [OPHIS](https://github.com/dyu056/vibeautoresearch) codebase. The full OPHIS `train.py` is 6966 lines with current shell, Muon, RoPE, etc. — all proven unnecessary for the gap phenomenon. This repo keeps only the ~800 lines that matter.

## License

Apache-2.0 (inheriting from OPHIS/nanoGPT upstream).
