"""
Generate synthetic Markovian data based on arXiv:2605.01199v1 Definition 2.1.

Data generation model:
  - Vocab V = {0, 1, ..., d-1}, where d = vocab_size
  - Token 0 is the "high-frequency" token (mass π₁)
  - Tokens 1..d-1 are "low-frequency" tokens with small frequency perturbations
  - Transition matrix: P = λI + (1-λ)1πᵀ
    * With probability λ: stay on the same token
    * With probability 1-λ: jump to a token sampled from stationary distribution π
  - Each sequence: x_1 ~ Uniform(V), then x_{j+1} ~ P_{x_j} for j=1..s
  - Input = (x_1, ..., x_s), target = x_{s+1}

Key control parameters:
  - λ (lambda): how predictable the data is. High λ = strong bigram dependency
  - π₁ (pi_1): probability mass of the high-frequency token
  - δ (delta): perturbation to break symmetry among low-frequency tokens

Output format:
  - shard_XXXXX.bin files (uint16), compatible with TokenizedShardDataset
  - Each chunk = sequence_len + 1 tokens: [x_1..x_s, y]
  - Additional metadata saved as JSON for experiment tracking

Usage:
  python code/generate_markov_data.py \
      --vocab_size 8192 \
      --sequence_len 2048 \
      --lambda_val 0.9 \
      --pi_1 0.3 \
      --delta 0.001 \
      --num_seqs_per_shard 5000 \
      --num_train_shards 3 \
      --num_val_shards 8 \
      --out_dir data/markov_lambda0.9
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

    This creates one dominant token (index 0) and many rare tokens
    with slight frequency differences among them (useful for studying
    how models differentiate between rare patterns).
    """
    d = vocab_size
    base = (1.0 - pi_1) / (d - 1)  # uniform share for low-freq tokens

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
    pi = np.maximum(pi, 1e-12)
    pi = pi / np.sum(pi)  # re-normalize

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

def generate_sequences(num_seqs, sequence_len, pi, lambda_val, seed=None):
    """
    Generate sequences according to Definition 2.1:
      x_1 ~ Uniform(V)
      x_{j+1} ~ P_{x_j}  for j = 1, ..., sequence_len

    Returns an array of shape (num_seqs, sequence_len + 1) of token IDs.
    The first sequence_len columns are input, the last column is target.
    """
    if seed is not None:
        np.random.seed(seed)

    d = len(pi)
    data = np.zeros((num_seqs, sequence_len + 1), dtype=np.uint16)

    for i in range(num_seqs):
        # Uniform first token
        data[i, 0] = np.random.randint(0, d)
        # Sample remaining tokens from Markov chain
        for j in range(1, sequence_len + 1):
            data[i, j] = sample_next_token(data[i, j-1], pi, lambda_val)

    return data


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


def generate_shards(num_seqs_per_shard, num_shards, sequence_len, pi, lambda_val,
                    out_dir, prefix="shard", seed_offset=0):
    """
    Generate multiple shard files.
    Returns list of (shard_id, num_tokens) tuples.
    """
    os.makedirs(out_dir, exist_ok=True)
    metadata = []

    for shard_idx in range(num_shards):
        seed = seed_offset + shard_idx * 1000 + 42
        data = generate_sequences(num_seqs_per_shard, sequence_len, pi, lambda_val, seed=seed)
        filepath = os.path.join(out_dir, f"{prefix}_{shard_idx:05d}.bin")
        n_tokens = write_shard(data, filepath)
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
        description="Generate Markovian toy data (arXiv:2605.01199v1 Def 2.1)")

    # Data parameters
    parser.add_argument("--vocab_size", type=int, default=8192,
                        help="Vocabulary size d")
    parser.add_argument("--sequence_len", type=int, default=2048,
                        help="Sequence length s (context window)")
    parser.add_argument("--lambda_val", type=float, default=0.9,
                        help="λ: stay-on-same-token probability (0-1). "
                             "Higher = stronger bigram dependency = more memorizable")
    parser.add_argument("--pi_1", type=float, default=0.3,
                        help="π₁: probability mass of the high-frequency token (0-1)")
    parser.add_argument("--delta", type=float, default=0.001,
                        help="δ: frequency perturbation among low-freq tokens")

    # Shard parameters
    parser.add_argument("--num_seqs_per_shard", type=int, default=5000,
                        help="Number of sequences per shard file")
    parser.add_argument("--num_train_shards", type=int, default=3,
                        help="Number of training shards")
    parser.add_argument("--num_val_shards", type=int, default=8,
                        help="Number of validation shards")

    # Output
    parser.add_argument("--out_dir", type=str, default="data/markov_toy",
                        help="Output directory for generated data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")

    args = parser.parse_args()

    # Validate
    assert 0 <= args.lambda_val <= 1, f"lambda_val must be in [0,1], got {args.lambda_val}"
    assert 0 < args.pi_1 < 1, f"pi_1 must be in (0,1), got {args.pi_1}"
    assert args.vocab_size >= 2, "vocab_size must be >= 2"

    # Build stationary distribution
    pi = build_stationary_distribution(args.vocab_size, args.pi_1, args.delta)

    # Print summary
    print("=" * 70)
    print("Markovian Toy Data Generator (arXiv:2605.01199v1 Def 2.1)")
    print("=" * 70)
    print(f"  Vocab size:      {args.vocab_size}")
    print(f"  Sequence length: {args.sequence_len}")
    print(f"  λ (stay prob):   {args.lambda_val}")
    print(f"  π₁ (high-freq):  {args.pi_1}")
    print(f"  δ (perturbation):{args.delta}")
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
    print(f"    - λ={args.lambda_val:.2f}: each token repeats itself with "
          f"{args.lambda_val*100:.0f}% probability")
    print(f"    - Token 0 appears {pi[0]*100:.1f}% of the time in equilibrium")
    print(f"    - Effective bigram dependency strength ∝ λ")
    print()

    # Generate train shards
    print("Generating training shards...")
    train_meta = generate_shards(
        args.num_seqs_per_shard, args.num_train_shards,
        args.sequence_len, pi, args.lambda_val,
        os.path.join(args.out_dir, "train"),
        prefix="shard", seed_offset=args.seed)

    print()
    print("Generating validation shards...")
    val_meta = generate_shards(
        args.num_seqs_per_shard, args.num_val_shards,
        args.sequence_len, pi, args.lambda_val,
        os.path.join(args.out_dir, "val"),
        prefix="shard", seed_offset=args.seed + 10000)

    # Save metadata
    total_train_tokens = sum(m["num_tokens"] for m in train_meta)
    total_val_tokens = sum(m["num_tokens"] for m in val_meta)

    config = {
        "description": "Markovian toy data (arXiv:2605.01199v1 Def 2.1)",
        "generation_params": {
            "vocab_size": args.vocab_size,
            "sequence_len": args.sequence_len,
            "lambda_val": args.lambda_val,
            "pi_1": args.pi_1,
            "delta": args.delta,
            "num_seqs_per_shard": args.num_seqs_per_shard,
            "num_train_shards": args.num_train_shards,
            "num_val_shards": args.num_val_shards,
            "seed": args.seed,
        },
        "stationary_distribution_first10": pi[:10].tolist(),
        "transition_matrix": "P = λI + (1-λ)1πᵀ",
        "theoretical_properties": {
            "ergodic": True,
            "irreducible": args.pi_1 < 1.0,
            "mixing_time_fast": True,  # independent resampling mixes in O(1) steps
            "bigram_dependency_strength": args.lambda_val,
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
    print("To use with train.py:")
    print(f"  python code/train.py \\")
    print(f"    --data_dir {args.out_dir}/train \\")
    print(f"    --train_shards {','.join(str(i) for i in range(args.num_train_shards))} \\")
    print(f"    --val_shards {','.join(str(i) for i in range(args.num_val_shards))} \\")
    print(f"    --vocab_size {args.vocab_size} \\")
    print(f"    --injection_position input")


if __name__ == "__main__":
    main()
