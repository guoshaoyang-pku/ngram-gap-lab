"""
Generate synthetic Markovian data with PER-SEQUENCE stickiness.

Each sequence k has its own identity = its stickiness λ_k ~ U(λ_low, λ_high):

  P_k = λ_k·I + (1-λ_k)·1πᵀ
    * With probability λ_k: stay on the same token
    * With probability 1-λ_k: jump to a token sampled from stationary π

Unlike the plain generator (single global λ), train sequences are no longer
exchangeable: the n-gram table can memorize the empirical stickiness of a
sequence on re-read (epoch boundary), while the backbone must infer λ_k
from context — mimicking document-specific statistics in natural language.

The per-sequence λ_k values are saved as shard_XXXXX_lambda.npy for analysis.

Usage:
  python code/generate_markov_data_perseq.py \
      --vocab_size 8192 \
      --sequence_len 2048 \
      --lambda_low 0.4 \
      --lambda_high 0.95 \
      --pi_1 0.3 \
      --num_seqs_per_shard 22000 \
      --num_train_shards 1 \
      --num_val_shards 10 \
      --out_dir data/markov_perseq
"""

import argparse
import json
import os
import numpy as np

# ---------------------------------------------------------------------------
# Stationary distribution π (Eq. 1 from the paper)
# ---------------------------------------------------------------------------

def build_stationary_distribution(vocab_size, pi_1, delta):
    """
    Build the stationary distribution π.

    π_0 = pi_1  (high-frequency token, index 0)
    For i >= 1: π_i = (1 - pi_1)/(d-1) + c_i * delta
    where c_i are perturbation coefficients that sum to 0.

    If delta is None, it's auto-set to base/10 to ensure a small but
    meaningful perturbation that won't make any token probability negative.
    """
    d = vocab_size
    base = (1.0 - pi_1) / (d - 1)

    if delta is None:
        delta = base / 10.0
        print(f"  delta auto-set to base/10 = {delta:.2e} (base={base:.2e})")

    if delta >= base:
        print(f"  WARNING: delta ({delta:.2e}) >= base ({base:.2e}), "
              f"negative-perturbation tokens will be clamped to ~0!")
        print(f"  This means ~half the vocab will never appear. "
              f"Consider delta < {base:.2e}")

    # Perturbation coefficients that sum to 0 (alternating +1/-1)
    half = (d - 1) // 2
    c = np.zeros(d, dtype=np.float64)
    for i in range(1, d):
        idx = i - 1
        if idx < half:
            c[i] = 1.0
        elif idx < 2 * half:
            c[i] = -1.0
        # remaining (if d-1 is odd) get c=0

    # Ensure sum of c_i = 0 (within floating point)
    total_c = np.sum(c[1:])
    if abs(total_c) > 1e-10:
        c[1] -= total_c

    pi = np.zeros(d, dtype=np.float64)
    pi[0] = pi_1
    for i in range(1, d):
        pi[i] = base + c[i] * delta

    # Clamp to ensure all probabilities are positive
    num_clamped = int(np.sum(pi[1:] < 1e-12))
    if num_clamped > 0:
        print(f"  WARNING: {num_clamped}/{d-1} low-freq tokens clamped to ~0 (delta too large)")

    pi = np.maximum(pi, 1e-12)
    pi = pi / np.sum(pi)  # re-normalize
    actual_pi1 = pi[0]
    if abs(actual_pi1 - pi_1) > 0.01:
        print(f"  NOTE: π[0] adjusted from {pi_1:.4f} to {actual_pi1:.4f} due to renormalization")

    return pi


# ---------------------------------------------------------------------------
# Transition matrix P = λI + (1-λ)1πᵀ (Eq. 2 from the paper)
# ---------------------------------------------------------------------------

def sample_next_token(current_token, pi, lambda_val):
    """
    Sample the next token from the Markov chain transition.

    P = λI + (1-λ)1πᵀ means:
      - With probability λ: stay at current_token
      - With probability 1-λ: sample from stationary distribution π (independent of current)

    Note: this is a simplified transition that makes the chain mix very quickly.
    A more realistic transition would be P = λI + (1-λ)Q where Q is a full matrix,
    but this form is analytically tractable and suffices for studying bigram dependency.
    """
    if np.random.random() < lambda_val:
        return current_token
    else:
        return np.random.choice(len(pi), p=pi)


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def generate_sequences(num_seqs, sequence_len, pi, lambda_vals, seed=None):
    """
    Generate sequences with PER-SEQUENCE stickiness:
      x_1 ~ Uniform(V)
      x_{j+1} ~ P_k(x_j) = lambda_k * I + (1 - lambda_k) * 1*pi^T

    Each sequence k gets its own lambda_k (the "sequence identity").
    The backbone cannot infer lambda_k from a single token; the n-gram
    table can memorize the empirical stickiness per sequence on re-read.

    Semi-vectorized: loops over positions but vectorized across sequences.

    lambda_vals: array of shape (num_seqs,) with one stickiness per sequence.
    Returns an array of shape (num_seqs, sequence_len + 1) of token IDs.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    lambda_vals = np.asarray(lambda_vals, dtype=np.float64).reshape(-1)
    assert lambda_vals.shape[0] == num_seqs, \
        f"lambda_vals must have one entry per sequence, got {lambda_vals.shape}"
    d = len(pi)
    total_len = sequence_len + 1

    data = np.zeros((num_seqs, total_len), dtype=np.int32)

    # Position 0: uniform random over vocab
    data[:, 0] = rng.integers(0, d, size=num_seqs)

    # Pre-generate "jump values" for all positions (sampled from pi)
    jump_values = rng.choice(d, size=(num_seqs, total_len), p=pi)

    # Pre-generate random thresholds for stay/jump decision at each position
    rand_vals = rng.random(size=(num_seqs, total_len))

    # Fill each position: if random < lambda_k, stay (copy previous);
    # otherwise, jump (use the pre-sampled value from pi)
    for j in range(1, total_len):
        stay = rand_vals[:, j] < lambda_vals  # (num_seqs,) per-sequence stickiness
        data[:, j] = np.where(stay, data[:, j-1], jump_values[:, j])

    return data.astype(np.uint16)


# ---------------------------------------------------------------------------
# Write to shard format (compatible with TokenizedShardDataset)
# ---------------------------------------------------------------------------

def write_shard(data_sequences, filepath):
    """
    Write sequences as a flat uint16 binary file.

    TokenizedShardDataset reads these as:
      chunk_size = sequence_len + 1
      input = chunk[:-1], target = chunk[-1]

    So we just flatten all sequences and write them contiguously.
    """
    flat = data_sequences.ravel()  # flatten to 1D
    flat = flat.astype(np.uint16)
    flat.tofile(filepath)
    return len(flat)


def generate_shards(num_seqs_per_shard, num_shards, sequence_len, pi, lambda_range,
                    out_dir, prefix="shard", seed_offset=0):
    """
    Generate multiple shard files, each sequence with its own lambda_k
    sampled from lambda_range = (lambda_low, lambda_high).
    The per-sequence lambda values are saved as a .npy file next to each
    shard for downstream analysis.
    Returns list of (shard_id, num_tokens) tuples.
    """
    os.makedirs(out_dir, exist_ok=True)
    metadata = []
    lambda_low, lambda_high = lambda_range

    for shard_idx in range(num_shards):
        seed = seed_offset + shard_idx * 1000 + 42
        rng = np.random.default_rng(seed)
        # Per-sequence stickiness — the "sequence identity"
        lambda_per_seq = rng.uniform(lambda_low, lambda_high, size=num_seqs_per_shard)
        data = generate_sequences(num_seqs_per_shard, sequence_len, pi, lambda_per_seq, seed=seed)
        filepath = os.path.join(out_dir, f"{prefix}_{shard_idx:05d}.bin")
        n_tokens = write_shard(data, filepath)
        # Save per-sequence lambda alongside the shard
        np.save(os.path.join(out_dir, f"{prefix}_{shard_idx:05d}_lambda.npy"), lambda_per_seq)
        metadata.append({
            "shard_id": shard_idx,
            "filepath": filepath,
            "num_tokens": n_tokens,
            "num_seqs": num_seqs_per_shard,
            "seed": seed,
            "lambda_mean": float(lambda_per_seq.mean()),
            "lambda_min": float(lambda_per_seq.min()),
            "lambda_max": float(lambda_per_seq.max()),
        })
        print(f"  Wrote {filepath}: {n_tokens:,} tokens ({num_seqs_per_shard} sequences, "
              f"lambda mean={lambda_per_seq.mean():.3f})")

    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Markovian toy data with PER-SEQUENCE stickiness "
                    "(each sequence has its own lambda_k, the 'sequence identity')")

    # Data parameters
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="Vocabulary size d")
    parser.add_argument("--sequence_len", type=int, default=2048,
                        help="Sequence length s (context window)")
    parser.add_argument("--lambda_low", type=float, default=0.4,
                        help="λ_low: lower bound of per-sequence stay probability")
    parser.add_argument("--lambda_high", type=float, default=0.95,
                        help="λ_high: upper bound of per-sequence stay probability")
    parser.add_argument("--pi_1", type=float, default=0.3,
                        help="π₁: probability mass of the high-frequency token (0-1)")
    parser.add_argument("--delta", type=float, default=None,
                        help="δ: frequency perturbation among low-freq tokens. "
                             "Default: auto-set to base/10 for safety")

    # Shard parameters
    parser.add_argument("--num_seqs_per_shard", type=int, default=22000,
                        help="Number of sequences per shard file")
    parser.add_argument("--num_train_shards", type=int, default=1,
                        help="Number of training shards")
    parser.add_argument("--num_val_shards", type=int, default=10,
                        help="Number of validation shards")

    # Output
    parser.add_argument("--out_dir", type=str, default="data/markov_toy",
                        help="Output directory for generated data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")

    args = parser.parse_args()

    # Validate
    assert 0 <= args.lambda_low < args.lambda_high <= 1, \
        f"need 0 <= lambda_low < lambda_high <= 1, got ({args.lambda_low}, {args.lambda_high})"
    assert 0 < args.pi_1 < 1, f"pi_1 must be in (0,1), got {args.pi_1}"
    assert args.vocab_size >= 2, "vocab_size must be >= 2"

    # Build stationary distribution
    pi = build_stationary_distribution(args.vocab_size, args.pi_1, args.delta)
    lambda_range = (args.lambda_low, args.lambda_high)

    # Print summary
    print("=" * 70)
    print("Markovian Toy Data Generator — per-sequence stickiness")
    print("=" * 70)
    print(f"  Vocab size:      {args.vocab_size}")
    print(f"  Sequence length: {args.sequence_len}")
    print(f"  λ range:         [{args.lambda_low}, {args.lambda_high}] (per-sequence uniform)")
    print(f"  π₁ (high-freq):  {args.pi_1}")
    print(f"  δ (perturbation):{args.delta if args.delta is not None else 'auto'}")
    print(f"  Train shards:    {args.num_train_shards}")
    print(f"  Val shards:      {args.num_val_shards}")
    print(f"  Seqs per shard:  {args.num_seqs_per_shard}")
    print(f"  Output dir:      {args.out_dir}")
    print()
    print(f"  Stationary distribution (first 10 tokens):")
    for i in range(min(10, args.vocab_size)):
        print(f"    token {i:4d}: π = {pi[i]:.6f}")
    if args.vocab_size > 10:
        print(f"    ... (total {args.vocab_size} tokens)")
    print()
    print(f"  Interpretation:")
    print(f"    - Each sequence has its own λ_k ~ U({args.lambda_low}, {args.lambda_high}): "
          f"its identity")
    print(f"    - Token 0 appears {pi[0]*100:.1f}% of the time in equilibrium")
    print(f"    - Bigram dependency strength varies per sequence ∝ λ_k")
    print()

    # Generate train shards
    print("Generating training shards...")
    train_meta = generate_shards(
        args.num_seqs_per_shard, args.num_train_shards,
        args.sequence_len, pi, lambda_range,
        os.path.join(args.out_dir, "train"),
        prefix="shard", seed_offset=args.seed)

    print()
    print("Generating validation shards...")
    val_meta = generate_shards(
        args.num_seqs_per_shard, args.num_val_shards,
        args.sequence_len, pi, lambda_range,
        os.path.join(args.out_dir, "val"),
        prefix="shard", seed_offset=args.seed + 10000)

    # Save metadata
    total_train_tokens = sum(m["num_tokens"] for m in train_meta)
    total_val_tokens = sum(m["num_tokens"] for m in val_meta)

    config = {
        "description": "Markovian toy data with per-sequence stickiness "
                       "(sequence identity = its own lambda_k)",
        "generation_params": {
            "vocab_size": args.vocab_size,
            "sequence_len": args.sequence_len,
            "lambda_low": args.lambda_low,
            "lambda_high": args.lambda_high,
            "lambda_sampling": "per-sequence uniform",
            "pi_1": args.pi_1,
            "delta": args.delta,
            "num_seqs_per_shard": args.num_seqs_per_shard,
            "num_train_shards": args.num_train_shards,
            "num_val_shards": args.num_val_shards,
            "seed": args.seed,
        },
        "stationary_distribution_first10": pi[:10].tolist(),
        "transition_matrix": "P_k = λ_k·I + (1-λ_k)·1πᵀ,  λ_k ~ U(lambda_low, lambda_high) per sequence",
        "theoretical_properties": {
            "ergodic": True,
            "irreducible": args.pi_1 < 1.0,
            "mixing_time_fast": True,  # independent resampling mixes in O(1) steps
            "bigram_dependency_strength_range": [args.lambda_low, args.lambda_high],
        },
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
    print()
    print("To use with train_perseq.py:")
    print(f"  python code/train_perseq.py \\")
    print(f"    --data_dir {args.out_dir}/train \\")
    print(f"    --train_shards {','.join(str(i) for i in range(args.num_train_shards))} \\")
    print(f"    --val_shards {','.join(str(i) for i in range(args.num_val_shards))} \\")
    print(f"    --vocab_size {args.vocab_size} \\")
    print(f"    --injection_position input")


if __name__ == "__main__":
    main()
