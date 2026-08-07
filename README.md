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

## Markov Toy Data Experiments

### Motivation

Real text (climbmix) cannot control how "memorizable" the data is. To isolate the
role of data statistics in the n-gram train/val gap, we replace real text with
a synthetic first-order Markov chain (arXiv:2605.01199v1, Definition 2.1).

### Data Generation

Transition matrix `P = λI + (1-λ)1πᵀ`:
- With probability λ: the next token repeats the current one
- With probability 1-λ: sample a new token from stationary distribution π

π has one high-frequency token (mass π₁) and many low-frequency tokens with
small frequency perturbations (δ). λ directly controls bigram-dependency strength.

Usage:
```bash
python code/generate_markov_data.py \
    --lambda_val 0.8 --pi_1 0.3 \
    --num_seqs_per_shard 22000 --num_train_shards 1 --num_val_shards 10 \
    --out_dir data/markov_toy
```

Output: uint16 `.bin` shard files compatible with the existing `TokenizedShardDataset`.

### Code Changes (branch `hrzhao`)

1. `code/generate_markov_data.py` — Markov chain data generator.
   - `build_stationary_distribution()`: auto-calculates δ to prevent negative probabilities.
   - `generate_sequences()`: semi-vectorized across all sequences per position.

2. `code/train.py` — n-gram tables disabled by default.
   - `enable_bigram_ve: bool = False`, `enable_trigram_ve: bool = False` (Config)
   - `--enable_bigram 0`, `--enable_trigram 0` (argparse defaults)
   - `val_interval_steps` and `--val_interval` both changed from 50 to 5.
   - N-gram tables are now opt-in: pass `--enable_bigram 1 --enable_trigram 1`.

### Standard Configuration

```
Model:  vanilla nanoGPT (8 layers, 6 heads, d_model=768)
Vocab:  8192 tokens
SeqLen: 2048
Data:   1 train shard (~22,000 sequences), 10 val shards
        λ=0.8, π₁=0.3, δ=auto (~8.5e-6)
Steps:  1000 (batch_size=72, total_batch=147,456, grad_accum=1)
        ~306 steps/epoch → ~3.3 epochs per run
Optim:  AdamW lr=0.004, betas=(0.8, 0.95), weight_decay=0.1
        (ngram tables: RMSProp, betas=(0.0, 0.999))
Eval:   every 5 steps, 4 batches
```

### Results (H200, seed=42)

| Experiment | train loss | val loss | gap (val - train) |
|---|---|---|---|
| Pure nanoGPT (no ngram) | 1.83 | 1.85 | **+0.02** |
| + ngram, injection=v | 1.81 | 1.85 | **+0.04** |
| + ngram, injection=y | 1.60 | 1.98 | **+0.38** |
| + ngram, injection=input | 1.57 | 2.06 | **+0.49** |

Key findings:
- Pure nanoGPT on Markov data shows **no gap** (gap ≈ 0.02 across all 4 epochs).
- Adding ngram tables produces a significant gap, confirming that the gap is
  caused by the ngram memory channel, not by data artifacts.
- Injection position ordering on Markov data: **input > y ≫ v**, different from
  the original climbmix ordering (y ≫ input > v).
- v position produces almost no gap (0.04) — on bigram-only data, attention
  inherently handles the dependency, and the ngram signal added inside V is
  drowned by the attention mixing.
- Gap grows with each epoch (epoch 1: ~0, epoch 2: +0.04, epoch 3: +0.22,
  epoch 4: +0.39), confirming the multi-epoch replay mechanism.
- The theoretical loss floor on Markov data is higher than real text (~1.85
  for pure nanoGPT), because 20% of transitions are inherently unpredictable
  random jumps. The ngram table can push train loss below this floor (1.57)
  by memorizing specific train-set transitions — direct evidence of overfitting.

### Gap by Epoch (ngram-input)

| Epoch | Steps | avg gap |
|-------|-------|---------|
| 1 | 1–305 | ~0.00 |
| 2 | 310–610 | +0.04 |
| 3 | 615–915 | +0.22 |
| 4 | 920–1000 | +0.39 |

### Interpretation

The Markov chain isolates bigram dependency as the sole structure in the data.
With λ=0.8, attention alone can handle the prediction task (gap ≈ 0). The ngram
table introduces a gradient shortcut — it learns faster than attention and
crowds it out of the training signal. The resulting gap reveals the tension
between memorization (ngram table) and generalization (attention). The gap's
magnitude is controlled by the injection position, which determines how much
the memorized signal is mixed by attention before reaching the output.

### Planned Experiments

1. **λ-sweep** (0.0, 0.3, 0.6, 0.9, 0.99): test whether gap magnitude scales
   with bigram-dependency strength.
2. **π₁-sweep** (0.1, 0.3, 0.5): test whether frequency skew affects gap
   (higher π₁ → fewer collisions → potentially smaller gap).
3. **Table-size ablation** (×16, ×64, ×256): test the collision rate vs.
   memorization precision trade-off.

### Files

```
code/generate_markov_data.py   — Markov data generator
docs/markov_data_design.md     — Full research design document
docs/figs/                     — Experiment plots
```

## License

Apache-2.0 (inheriting from OPHIS/nanoGPT upstream).
