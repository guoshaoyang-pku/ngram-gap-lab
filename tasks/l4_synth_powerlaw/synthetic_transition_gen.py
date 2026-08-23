#!/usr/bin/env python3
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


DEFAULT_PROFILE = [
    (1, 1024),
    (2, 768),
    (4, 512),
    (8, 384),
    (16, 256),
    (64, 128),
    (256, 64),
    (1024, 32),
]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packed_context(context: tuple[int, ...], vocab: int) -> int:
    key = 0
    for token in context:
        key = key * vocab + token
    return key


def normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("probability weights must have positive mass")
    return [value / total for value in values]


def sample_from_weights(
    rng: random.Random, values: list[int], weights: list[float]
) -> int:
    return rng.choices(values, weights=weights, k=1)[0]


def zipf_weights(vocab: int, exponent: float) -> list[float]:
    return [1.0 / ((token + 1) ** exponent) for token in range(vocab)]


def sample_dirichlet(
    rng: random.Random, size: int, concentration: float
) -> list[float]:
    values = [rng.gammavariate(concentration, 1.0) for _ in range(size)]
    return normalize(values)


def make_contexts(
    rng: random.Random,
    count: int,
    context_len: int,
    hub_size: int,
) -> list[tuple[int, ...]]:
    contexts = []
    seen = set()
    while len(contexts) < count:
        context = tuple(rng.randrange(1, hub_size + 1) for _ in range(context_len))
        if context not in seen:
            seen.add(context)
            contexts.append(context)
    return contexts


def assign_frequencies(
    profile: list[tuple[int, int]], count: int, scale: int
) -> list[int]:
    values = []
    for frequency, number in profile:
        values.extend([frequency * scale] * number)
    if len(values) < count:
        values.extend([values[-1]] * (count - len(values)))
    return values[:count]


def sparse_restart_distributions(
    contexts: list[tuple[int, ...]],
    vocab: int,
    sep: int,
    rng: random.Random,
    hub_size: int,
    support_size: int,
    restart: float,
    zipf_exponent: float,
) -> list[list[float]]:
    targets = list(range(hub_size + 1, sep))
    base = normalize(zipf_weights(len(targets), zipf_exponent))
    distributions = []
    for context in contexts:
        support = rng.sample(targets, support_size)
        support_weights = normalize(
            [1.0 / (index + 1) for index in range(support_size)]
        )
        weights = [restart * base[index] for index in range(len(targets))]
        for token, weight in zip(support, support_weights):
            weights[token - hub_size - 1] += (1.0 - restart) * weight
        distributions.append(normalize(weights))
    return distributions


def lowrank_sparse_distributions(
    contexts: list[tuple[int, ...]],
    vocab: int,
    sep: int,
    rng: random.Random,
    hub_size: int,
    topics: int,
    topic_concentration: float,
    private_mass: float,
    private_support: int,
    zipf_exponent: float,
) -> list[list[float]]:
    targets = list(range(hub_size + 1, sep))
    topic_distributions = [
        normalize(
            [
                (1.0 / ((token + 1) ** zipf_exponent))
                * (0.25 + rng.random())
                for token in range(len(targets))
            ]
        )
        for _ in range(topics)
    ]
    distributions = []
    for context in contexts:
        mixture = sample_dirichlet(rng, topics, topic_concentration)
        shared = [
            sum(mixture[topic] * topic_distributions[topic][index]
                for topic in range(topics))
            for index in range(len(targets))
        ]
        private_tokens = rng.sample(targets, private_support)
        private = [0.0] * len(targets)
        private_weights = normalize(
            [1.0 / (index + 1) for index in range(private_support)]
        )
        for token, weight in zip(private_tokens, private_weights):
            private[token - hub_size - 1] = weight
        distributions.append(
            normalize(
                [
                    (1.0 - private_mass) * shared[index]
                    + private_mass * private[index]
                    for index in range(len(targets))
                ]
            )
        )
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
) -> tuple[list[tuple[int, ...]], Counter[tuple[int, ...]]]:
    # Precompute the cumulative distribution once per context. This is
    # bit-identical to random.choices(weights=...) but ~100x faster, because
    # random.choices rebuilds the cumulative weights on every call.
    blocks = []
    counts: Counter[tuple[int, ...]] = Counter()
    targets = list(range(hub_size + 1, sep))
    for context, frequency, weights in zip(contexts, frequencies, distributions):
        cum = list(accumulate(weights))
        total = cum[-1]
        for _ in range(frequency * repetitions):
            target = targets[bisect(cum, rng.random() * total)]
            blocks.append((*context, target, sep))
            counts[context] += 1
    return blocks, counts


def align_blocks(tokens: list[int], block_len: int, row_stride: int, sep: int) -> list[int]:
    if len(tokens) % block_len:
        raise ValueError("designed token stream must contain complete blocks")
    remainder = len(tokens) % row_stride
    if remainder:
        tokens.extend([sep] * (row_stride - remainder))
    return tokens


def exact_counts(
    tokens: list[int],
    order: int,
    context_max_token: int,
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
    contexts = make_contexts(
        rng, args.num_contexts, args.context_len, args.hub_size
    )
    frequencies = assign_frequencies(DEFAULT_PROFILE, args.num_contexts, args.frequency_scale)
    if args.scheme == "sparse_restart":
        distributions = sparse_restart_distributions(
            contexts, vocab, sep, rng, args.hub_size, args.support_size, args.restart,
            args.zipf_exponent,
        )
    else:
        distributions = lowrank_sparse_distributions(
            contexts, vocab, sep, rng, args.hub_size, args.topics, args.topic_concentration,
            args.private_mass, args.private_support, args.zipf_exponent,
        )
    train_blocks, _ = blocks_for_contexts(
        contexts, frequencies, distributions, vocab, sep, rng, args.hub_size, 1
    )
    val_blocks, _ = blocks_for_contexts(
        contexts, frequencies, distributions, vocab, sep, rng, args.hub_size,
        args.val_repetitions,
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
    frequency_by_context = np.asarray(
        [train_counts[context] for context in contexts],
        dtype=np.float64,
    )
    bayes_loss = float(
        np.sum(frequency_by_context * row_entropy)
        / np.sum(frequency_by_context)
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
        "schema_version": 1,
        "vocab": vocab,
        "order": args.context_len,
        "context_len": args.context_len,
        "block_len": block_len,
        "loader_row_stride": row_stride,
        "sep_token": sep,
        "scheme": args.scheme,
        "seed": args.seed,
        "num_contexts": args.num_contexts,
        "frequency_definition": "exact_train_epoch_context_count",
        "frequency_source_split": "train",
        "frequency_key_type": "exact_context",
        "frequency_index_format": "context_matrix_v1",
        "frequency_index_format": "context_matrix_v1",
        "hash_bucket_occupancy_diagnostic": False,
        "distribution_definition": "known_conditional_transition_matrix",
        "bayes_loss_available": True,
        "transition_matrix_file": "transition_matrix.npz",
        "transition_matrix_shape": list(transition_probabilities.shape),
        "frequency_weighted_bayes_ce": bayes_loss,
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
    (out_dir / "run_contract.json").write_text(
        json.dumps(
            {
                "experiment": "synthetic_transition_task",
                "scheme": args.scheme,
                "conditional_distribution": (
                    "sparse_restart" if args.scheme == "sparse_restart"
                    else "lowrank_shared_plus_private_sparse"
                ),
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
    parser.add_argument(
        "--scheme", choices=["sparse_restart", "lowrank_sparse"], required=True
    )
    parser.add_argument("--vocab", type=int, default=8192)
    parser.add_argument("--context-len", type=int, default=5)
    parser.add_argument("--sequence-len", type=int, default=2048)
    parser.add_argument("--num-contexts", type=int, default=4096)
    parser.add_argument("--hub-size", type=int, default=256)
    parser.add_argument("--frequency-scale", type=int, default=8)
    parser.add_argument("--val-repetitions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--support-size", type=int, default=8)
    parser.add_argument("--restart", type=float, default=0.10)
    parser.add_argument("--topics", type=int, default=32)
    parser.add_argument("--topic-concentration", type=float, default=0.25)
    parser.add_argument("--private-mass", type=float, default=0.50)
    parser.add_argument("--private-support", type=int, default=4)
    parser.add_argument("--zipf-exponent", type=float, default=1.05)
    generate(parser.parse_args())


if __name__ == "__main__":
    main()