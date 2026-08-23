#!/usr/bin/env python3
"""Clean power-law gap toy: fine frequency buckets + probabilistic rule + honest val.

Goal: reproduce a *clean* double-log-linear gap(r) ~ (K_eff-1)/r on the real
harness, matching the ideal count-table learner (MC-verified slope -1.00).

Differences from synthetic_transition_gen.py (synth pilot):
  1. Fine exact buckets: r in {1,2,4,...,1024} x 128 contexts each (scale=1),
     so every bucket has enough contexts for low-noise per-bucket means.
  2. Val is *context-uniform*: every context gets the same number of fresh
     val draws (VAL_REPS), independent of r -> the val probe hits each context
     ~equally -> per-bucket sample counts are clean (not token-weighted).
  3. Scheme A (sparse_restart, private+global, K_eff ~ 8) only.

Count-table prediction (see docs/theory_notes/toy-gap-frequency-distributions.md):
  gap(r) = val CE - train CE ~= (K_eff - 1)/r   ->  log-log slope -1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from bisect import bisect
from collections import Counter
from itertools import accumulate
from pathlib import Path

import numpy as np

# (r, n_contexts) -- exact counts (scale=1), equal n per bucket.
FINE_BUCKETS = [
    (1, 128), (2, 128), (4, 128), (8, 128), (16, 128), (32, 128),
    (64, 128), (128, 128), (256, 128), (512, 128), (1024, 128),
]
VAL_REPS = 8  # fresh val draws per context (context-uniform)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("probability weights must have positive mass")
    return [value / total for value in values]


def make_contexts(
    rng: random.Random, count: int, context_len: int, hub_size: int
) -> list[tuple[int, ...]]:
    contexts = []
    seen = set()
    while len(contexts) < count:
        context = tuple(rng.randrange(1, hub_size + 1) for _ in range(context_len))
        if context not in seen:
            seen.add(context)
            contexts.append(context)
    return contexts


def sparse_restart_distributions(
    contexts: list[tuple[int, ...]],
    vocab: int,
    sep: int,
    rng: random.Random,
    hub_size: int,
    support_size: int,
    restart: float,
) -> list[list[float]]:
    targets = list(range(hub_size + 1, sep))
    base = normalize([1.0 / ((token + 1) ** 1.05) for token in range(len(targets))])
    distributions = []
    for _context in contexts:
        support = rng.sample(targets, support_size)
        support_weights = normalize(
            [1.0 / (index + 1) for index in range(support_size)]
        )
        weights = [restart * base[index] for index in range(len(targets))]
        for token, weight in zip(support, support_weights):
            weights[token - hub_size - 1] += (1.0 - restart) * weight
        distributions.append(normalize(weights))
    return distributions


def blocks_for_contexts(
    contexts: list[tuple[int, ...]],
    frequencies: list[int],
    distributions: list[list[float]],
    vocab: int,
    sep: int,
    rng: random.Random,
    hub_size: int,
    repetitions: int,
) -> list[tuple[int, ...]]:
    """repetitions = number of draws PER CONTEXT (context-uniform val) or per
    occurrence (train). For train, repetitions=1 -> each context appears
    exactly `frequency` times. For val, repetitions=VAL_REPS -> context-uniform."""
    blocks = []
    targets = list(range(hub_size + 1, sep))
    for context, frequency, weights in zip(contexts, frequencies, distributions):
        cum = list(accumulate(weights))
        total = cum[-1]
        n_draws = frequency * repetitions
        for _ in range(n_draws):
            target = targets[bisect(cum, rng.random() * total)]
            blocks.append((*context, target, sep))
    return blocks


def align_blocks(tokens: list[int], block_len: int, row_stride: int, sep: int) -> list[int]:
    if len(tokens) % block_len:
        raise ValueError("designed token stream must contain complete blocks")
    remainder = len(tokens) % row_stride
    if remainder:
        tokens.extend([sep] * (row_stride - remainder))
    return tokens


def exact_counts(
    tokens: list[int], order: int, context_max_token: int
) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    for index in range(order, len(tokens)):
        context = tuple(tokens[index - order:index])
        if all(1 <= token <= context_max_token for token in context):
            counts[context] += 1
    return counts


def generate(args: argparse.Namespace) -> dict:
    vocab = args.vocab
    sep = vocab - 1
    block_len = args.context_len + 2
    row_stride = (args.sequence_len + 1) // block_len * block_len
    rng = random.Random(args.seed)
    total_contexts = sum(n for _, n in FINE_BUCKETS)
    contexts = make_contexts(rng, total_contexts, args.context_len, args.hub_size)
    frequencies = []
    for r, n in FINE_BUCKETS:
        frequencies.extend([r] * n)
    distributions = sparse_restart_distributions(
        contexts, vocab, sep, rng, args.hub_size, args.support_size, args.restart
    )
    train_blocks = blocks_for_contexts(
        contexts, frequencies, distributions, vocab, sep, rng, args.hub_size, 1
    )
    val_blocks = blocks_for_contexts(
        contexts, frequencies, distributions, vocab, sep, rng, args.hub_size,
        VAL_REPS,
    )
    rng.shuffle(train_blocks)
    rng.shuffle(val_blocks)
    train_tokens = [token for block in train_blocks for token in block]
    val_tokens = [token for block in val_blocks for token in block]
    train_tokens = align_blocks(train_tokens, block_len, row_stride, sep)
    val_tokens = align_blocks(val_tokens, block_len, row_stride, sep)
    train_counts = exact_counts(train_tokens, args.context_len, args.hub_size)
    context_set = set(contexts)
    missing = sorted(context_set - set(train_counts))
    if missing:
        raise ValueError(f"exact train count missing for {len(missing)} contexts")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_tokens.txt"
    val_path = out_dir / "val_tokens.txt"
    train_path.write_text(" ".join(map(str, train_tokens)) + "\n")
    val_path.write_text(" ".join(map(str, val_tokens)) + "\n")
    exact_contexts = np.asarray(sorted(train_counts), dtype=np.int32)
    values = np.asarray(
        [train_counts[context] for context in sorted(train_counts)],
        dtype=np.int64,
    )
    np.savez(
        out_dir / "exact_ngram_counts.npz",
        contexts=exact_contexts,
        counts=values,
    )
    transition_contexts = np.asarray(contexts, dtype=np.int32)
    transition_probabilities = np.asarray(distributions, dtype=np.float32)
    np.savez(
        out_dir / "transition_matrix.npz",
        contexts=transition_contexts,
        probabilities=transition_probabilities,
        target_tokens=np.arange(args.hub_size + 1, sep, dtype=np.int32),
    )
    row_entropy = -np.sum(
        transition_probabilities
        * np.log(np.maximum(transition_probabilities, 1e-30)),
        axis=1,
    )
    # K_eff per context = exp(H) (participation count)
    k_eff = np.exp(row_entropy)
    frequency_by_context = np.asarray(
        [train_counts[context] for context in contexts],
        dtype=np.float64,
    )
    bayes_loss = float(
        np.sum(frequency_by_context * row_entropy) / np.sum(frequency_by_context)
    )
    contexts_json = {
        " ".join(map(str, context)): {
            "train_frequency": int(train_counts[context]),
            "scheme": args.scheme,
        }
        for context in contexts
    }
    (out_dir / "contexts.json").write_text(
        json.dumps(contexts_json, sort_keys=True, separators=(",", ":")) + "\n"
    )
    metadata = {
        "schema_version": 2,
        "experiment": "synthetic_powerlaw_gap",
        "vocab": vocab,
        "order": args.context_len,
        "context_len": args.context_len,
        "block_len": block_len,
        "loader_row_stride": row_stride,
        "sep_token": sep,
        "scheme": args.scheme,
        "seed": args.seed,
        "num_contexts": total_contexts,
        "fine_buckets": {str(r): n for r, n in FINE_BUCKETS},
        "val_reps_per_context": VAL_REPS,
        "val_mode": "honest_fresh_samples_context_uniform",
        "frequency_definition": "exact_train_epoch_context_count",
        "frequency_source_split": "train",
        "frequency_key_type": "exact_context",
        "frequency_index_format": "context_matrix_v1",
        "hash_bucket_occupancy_diagnostic": False,
        "distribution_definition": "known_conditional_transition_matrix",
        "bayes_loss_available": True,
        "transition_matrix_file": "transition_matrix.npz",
        "transition_matrix_shape": list(transition_probabilities.shape),
        "frequency_weighted_bayes_ce": bayes_loss,
        "mean_k_eff_exp_h": float(np.mean(k_eff)),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "n_distinct_exact_contexts": len(train_counts),
        "train_frequency_quantiles": {
            f"q{quantile}": int(np.percentile(values, quantile))
            for quantile in (0, 50, 90, 99, 100)
        },
        "checksum_train": checksum(train_path),
        "checksum_val": checksum(val_path),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    # meta.json: the loader/trainer read this name (vocab/sep/block/order +
    # the frequency contract). Same content + vocab_size for compatibility.
    loader_meta = dict(metadata)
    loader_meta["vocab_size"] = int(metadata["vocab"])
    (out_dir / "meta.json").write_text(
        json.dumps(loader_meta, indent=2, sort_keys=True) + "\n"
    )
    (out_dir / "run_contract.json").write_text(
        json.dumps(
            {
                "experiment": "synthetic_powerlaw_gap",
                "scheme": args.scheme,
                "conditional_distribution": "sparse_restart_private_plus_global",
                "parameters": vars(args),
                "frequency_contract": metadata,
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scheme", default="sparse_restart")
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--context-len", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=2048)
    parser.add_argument("--hub-size", type=int, default=256)
    parser.add_argument("--support-size", type=int, default=8)
    parser.add_argument("--restart", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260807)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()
