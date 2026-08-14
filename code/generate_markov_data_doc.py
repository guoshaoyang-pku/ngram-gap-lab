"""Generate Markov toy data with a SHARED deterministic content map.

Minimal modification of the original generator (P = λI + (1-λ)1πᵀ):
  - vocab reduced to 256 shared "common words", used by ALL sequences
  - a GLOBAL deterministic pair map g(prev, cur) -> next (one dictionary
    shared by every sequence) is mixed in with tunable probability

Generation, for each step t >= 2 (x_1 ~ Markov step):
  with prob content_prob:  x_{t+1} = g(x_{t-1}, x_t)   deterministic content
  with prob 1-content_prob: original Markov step
                            (stay w.p. λ, else jump to π)

The shared map gives the n-gram table "hot" pairs (each pair appears in
every sequence -> massive exposure per epoch) whose transitions it can
memorize; the backbone can also learn the shared map, but slower.  The
content fraction is a tunable knob; start small (0.1).

Metadata saved next to each shard:
  shard_XXXXX_gmap.npy : the global map, shape (vocab_size, vocab_size)

Usage:
  python code/generate_markov_data_doc.py
"""

import argparse
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Stationary distribution π (kept from the original generator)
# ---------------------------------------------------------------------------

def build_stationary_distribution(vocab_size, pi_1, delta):
    """π_0 = pi_1; π_i = (1-pi_1)/(d-1) + c_i·δ for i >= 1."""
    d = vocab_size
    base = (1.0 - pi_1) / (d - 1)
    if delta is None:
        delta = base / 10.0

    half = (d - 1) // 2
    c = np.zeros(d, dtype=np.float64)
    for i in range(1, d):
        idx = i - 1
        if idx < half:
            c[i] = 1.0
        elif idx < 2 * half:
            c[i] = -1.0
    total_c = np.sum(c[1:])
    if abs(total_c) > 1e-10:
        c[1] -= total_c

    pi = np.zeros(d, dtype=np.float64)
    pi[0] = pi_1
    for i in range(1, d):
        pi[i] = base + c[i] * delta
    pi = np.maximum(pi, 1e-12)
    pi = pi / np.sum(pi)
    return pi


def build_global_map(vocab_size, rng):
    """One shared deterministic dictionary g(a, b) -> next token.

    Outputs are uniform over the whole vocabulary, so un-memorized content
    looks random (entropy ln(V)) and memorized content is exactly predicted.
    """
    return rng.integers(0, vocab_size, size=(vocab_size, vocab_size)).astype(np.int32)


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def generate_sequences(num_seqs, sequence_len, pi, lambda_val, content_prob,
                       g_map, seed=None):
    """Semi-vectorized generation mixing the shared map and the Markov law.

    x_1 ~ Uniform(V); x_2 follows the Markov law; for j >= 3:
      r < content_prob        -> x_j = g(x_{j-2}, x_{j-1})
      otherwise (Markov step) -> stay on x_{j-1} w.p. λ, else jump to π

    Returns (num_seqs, sequence_len + 1) uint16 token ids.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    d = len(pi)
    total_len = sequence_len + 1

    data = np.zeros((num_seqs, total_len), dtype=np.int32)
    data[:, 0] = rng.integers(0, d, size=num_seqs)

    # Pre-generate randomness for every position
    jump_values = rng.choice(d, size=(num_seqs, total_len), p=pi)
    rand_vals = rng.random(size=(num_seqs, total_len))
    branch_r = rng.random(size=(num_seqs, total_len))  # content vs Markov

    # j = 1: plain Markov step (no pair available yet)
    stay = rand_vals[:, 1] < lambda_val
    data[:, 1] = np.where(stay, data[:, 0], jump_values[:, 1])

    for j in range(2, total_len):
        # shared-map content branch: key (x_{j-2}, x_{j-1})
        next_content = g_map[data[:, j - 2], data[:, j - 1]]
        # Markov background branch: stay w.p. λ, else jump to π
        stay = rand_vals[:, j] < lambda_val
        next_markov = np.where(stay, data[:, j - 1], jump_values[:, j])
        data[:, j] = np.where(branch_r[:, j] < content_prob, next_content, next_markov)

    return data.astype(np.uint16)


# ---------------------------------------------------------------------------
# Shard writing
# ---------------------------------------------------------------------------

def write_shard(data_sequences, filepath):
    flat = data_sequences.ravel().astype(np.uint16)
    flat.tofile(filepath)
    return len(flat)


def generate_shards(num_seqs_per_shard, num_shards, sequence_len, pi,
                    lambda_val, content_prob, g_map, out_dir, prefix="shard",
                    seed_offset=0):
    os.makedirs(out_dir, exist_ok=True)
    metadata = []

    for shard_idx in range(num_shards):
        seed = seed_offset + shard_idx * 1000 + 42
        data = generate_sequences(num_seqs_per_shard, sequence_len, pi,
                                  lambda_val, content_prob, g_map, seed=seed)
        filepath = os.path.join(out_dir, f"{prefix}_{shard_idx:05d}.bin")
        n_tokens = write_shard(data, filepath)
        np.save(os.path.join(out_dir, f"{prefix}_{shard_idx:05d}_gmap.npy"), g_map)
        metadata.append({
            "shard_id": shard_idx,
            "filepath": filepath,
            "num_tokens": n_tokens,
            "num_seqs": num_seqs_per_shard,
            "seed": seed,
        })
        print(f"  Wrote {filepath}: {n_tokens:,} tokens ({num_seqs_per_shard} sequences)")

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Markov toy data: original λ-Markov background "
                    "plus a SHARED deterministic pair map (tunable fraction)")

    # Data parameters
    parser.add_argument("--vocab_size", type=int, default=256,
                        help="Vocabulary size; all tokens are shared common words")
    parser.add_argument("--sequence_len", type=int, default=2048,
                        help="Sequence length s (context window)")
    parser.add_argument("--lambda_val", type=float, default=0.8,
                        help="λ for the background Markov law (stay probability)")
    parser.add_argument("--pi_1", type=float, default=0.3,
                        help="π₁: probability mass of the high-frequency token 0")
    parser.add_argument("--content_prob", type=float, default=0.1,
                        help="Fraction of steps following the shared deterministic "
                             "pair map (start small)")
    parser.add_argument("--delta", type=float, default=None,
                        help="δ: frequency perturbation among low-freq tokens")

    # Shard parameters
    parser.add_argument("--num_seqs_per_shard", type=int, default=22000,
                        help="Number of sequences per shard file")
    parser.add_argument("--num_train_shards", type=int, default=1,
                        help="Number of training shards")
    parser.add_argument("--num_val_shards", type=int, default=10,
                        help="Number of validation shards")

    # Output
    parser.add_argument("--out_dir", type=str, default="data/markov_doc",
                        help="Output directory for generated data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")

    args = parser.parse_args()

    # Validate
    assert 0 <= args.lambda_val <= 1, "lambda_val must be in [0,1]"
    assert 0 <= args.content_prob <= 1, "content_prob must be in [0,1]"
    assert 0 < args.pi_1 < 1, "pi_1 must be in (0,1)"
    assert args.vocab_size >= 2, "vocab_size must be >= 2"

    pi = build_stationary_distribution(args.vocab_size, args.pi_1, args.delta)
    g_map = build_global_map(args.vocab_size, np.random.default_rng(args.seed))

    print("=" * 70)
    print("Markov Toy Data Generator — shared deterministic pair map")
    print("=" * 70)
    print(f"  Vocab size:      {args.vocab_size} (all shared common words)")
    print(f"  Sequence length: {args.sequence_len}")
    print(f"  λ (background):  {args.lambda_val}")
    print(f"  π₁ (high-freq):  {args.pi_1}")
    print(f"  content_prob:    {args.content_prob} (fraction of deterministic steps)")
    print(f"  Train shards:    {args.num_train_shards}")
    print(f"  Val shards:      {args.num_val_shards}")
    print(f"  Seqs per shard:  {args.num_seqs_per_shard}")
    print(f"  Output dir:      {args.out_dir}")
    print()

    print("Generating training shards...")
    train_meta = generate_shards(
        args.num_seqs_per_shard, args.num_train_shards,
        args.sequence_len, pi, args.lambda_val, args.content_prob, g_map,
        os.path.join(args.out_dir, "train"),
        prefix="shard", seed_offset=args.seed)

    print()
    print("Generating validation shards...")
    val_meta = generate_shards(
        args.num_seqs_per_shard, args.num_val_shards,
        args.sequence_len, pi, args.lambda_val, args.content_prob, g_map,
        os.path.join(args.out_dir, "val"),
        prefix="shard", seed_offset=args.seed + 10000)

    total_train_tokens = sum(m["num_tokens"] for m in train_meta)
    total_val_tokens = sum(m["num_tokens"] for m in val_meta)

    config = {
        "description": "Markov background (λI + (1-λ)1πᵀ) plus a SHARED "
                       "deterministic pair map g(a,b) with tunable fraction",
        "generation_params": {
            "vocab_size": args.vocab_size,
            "sequence_len": args.sequence_len,
            "lambda_val": args.lambda_val,
            "pi_1": args.pi_1,
            "content_prob": args.content_prob,
            "delta": args.delta,
            "num_seqs_per_shard": args.num_seqs_per_shard,
            "num_train_shards": args.num_train_shards,
            "num_val_shards": args.num_val_shards,
            "seed": args.seed,
        },
        "transition_rule": "x_{t+1} = g(x_{t-1}, x_t) w.p. content_prob; "
                           "otherwise stay w.p. λ or jump to π",
        "train": {
            "num_shards": args.num_train_shards,
            "total_tokens": total_train_tokens,
            "per_shard": train_meta,
        },
        "val": {
            "num_shards": args.num_val_shards,
            "total_tokens": total_val_tokens,
            "per_shard": val_meta,
        },
    }

    config_path = os.path.join(args.out_dir, "data_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    print()
    print(f"Done! Generated {total_train_tokens + total_val_tokens:,} tokens total")
    print(f"  Train: {total_train_tokens:,} tokens across {args.num_train_shards} shards")
    print(f"  Val:   {total_val_tokens:,} tokens across {args.num_val_shards} shards")
    print(f"  Config: {config_path}")


if __name__ == "__main__":
    main()
