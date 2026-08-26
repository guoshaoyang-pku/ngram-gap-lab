#!/usr/bin/env python3
"""5-gram frequency-controlled dataset generator.

Builds a coincidental-gap dataset from real NLP tokens (climbmix shard 1,
~63M BPE tokens, vocab 8192):

  - context = 5 real tokens, target = 1 next token (order-5 n-gram context).
  - 5-gram context is hashed (polynomial rolling hash) into ``bucket_count``
    buckets (default 1,000,000).  Collisions are allowed and intentional:
    the bucket defines the "frequency class" and the next-token histogram
    is aggregated over all colliding 5-grams.
  - For each bucket ``b`` with natural occurrence count ``r(b)`` and next-token
    histogram ``h(b)``, we resample train/val next-tokens by drawing
    independently from ``h(b)`` as a multinomial distribution:

        n_train(b) = round(r(b) * f_train * k(b))
        n_val(b)   = round(r(b) * f_val)          # val keeps natural dist
        k(b)       = clip((r_ref / r(b))^alpha, k_min, k_max)

    Low-frequency buckets (small ``r``) get ``k > 1`` (up-sampling); high-
    frequency buckets get ``k < 1`` (down-sampling).  ``alpha=0`` => ``k=1``
    everywhere (baseline, no resampling).  Val is never resampled so it stays
    a faithful sample of the true language distribution.

  - Coincidental gap: train and val next-tokens for the same bucket are
    *independent* draws from the same histogram.  For low-frequency buckets
    the two samples disagree on the next-token distribution => large gap.
    For high-frequency buckets the law of large numbers makes them agree =>
    small gap.  Up-sampling a low-frequency bucket raises its effective
    ``r_train`` and should compress the gap.

  - Output token stream uses 7-token blocks ``[c1,c2,c3,c4,c5, next, SEP]``
    with ``SEP = vocab-1`` separating blocks, matching the toy5 convention.
    The model trains on the full packed sequence with standard next-token CE;
    the gap analysis only measures loss at the ``next`` position (position 5
    of each block, 0-indexed).

The tokenizer and parquet shards are read via the local ``lib.py`` adapter so
the generator and trainer use the same cache and tokenizer contract.

CLI:
  python data_gen.py --out-dir data/alpha0.5 --alpha 0.5
  python data_gen.py --out-dir data/pilot    --max-tokens 10000000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Lazily import the upstream tokenizer/parquet plumbing only when we actually
# need to read real shards.  Pure-logic helpers (hashing, factor computation,
# sampling, block-stream assembly) do not depend on it, so unit tests can run
# without pyarrow/rustbpe installed.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_upstream_lib():
    os.environ.setdefault(
        "AUTORESEARCH_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "autoresearch"),
    )
    os.environ.setdefault(
        "DATA_DIR_OVERRIDE",
        os.path.join(os.environ["AUTORESEARCH_CACHE_DIR"], "data"),
    )
    # Resolve by path so an unrelated module on PYTHONPATH cannot change the
    # tokenizer or shard-split contract.
    candidates = [
        Path(__file__).resolve().parent / "lib.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "_ngram5_upstream_lib", candidate
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load {candidate}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError(
        "cannot locate upstream lib.py (tried: "
        + ", ".join(str(c) for c in candidates)
        + ")"
    )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

# 64-bit polynomial rolling hash over 5 tokens.  We use two independent
# accumulators and fold to one value, then mod bucket_count.
#
# IMPORTANT: the modulus must fit in signed int64 so the same hash can be
# computed in Python and PyTorch.  A direct 61-bit multiplication still
# overflows int64, so ``hash_utils.hash5_bucket_tensor`` uses exact limb
# multiplication before the Mersenne reduction.
_HASH_BASE1 = 0x9E3779B97F4A7C15 % ((1 << 61) - 1)
_HASH_BASE2 = 0xC2B2AE3D27D4EB4F % ((1 << 61) - 1)
_HASH_MOD   = (1 << 61) - 1  # Mersenne prime, fits in int64
_HASH_INIT1 = 0x517CC1B727220A95 % _HASH_MOD
_HASH_INIT2 = 0x6C62272E07BB0142 % _HASH_MOD
_HASH_MASK61 = _HASH_MOD  # also used as a bit mask (all-ones in low 61 bits)


def hash_n(tokens: list[int]) -> int:
    """Hash an n-token context to a 61-bit integer (no modulo by bucket here).

    Works for any n (trigram n=3, 5-gram n=5, etc.).  Computed modulo
    2^61-1 (Mersenne prime) so the identical computation can be reproduced
    in a PyTorch int64 tensor by trainer._bucket_id_tensor.
    """
    h1 = _HASH_INIT1
    h2 = _HASH_INIT2
    for t in tokens:
        h1 = (h1 * _HASH_BASE1 + t) % _HASH_MOD
        h2 = (h2 * _HASH_BASE2 + t) % _HASH_MOD
    h2_shifted = ((h2 << 37) | (h2 >> 24)) & _HASH_MASK61
    h = (h1 ^ h2_shifted) & _HASH_MASK61
    return h

# Backwards-compatible alias.
hash5 = hash_n


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

# Optional per-shard raw-token cache.  When set (by generate()), the token
# stream iterators tokenize each parquet shard once, persist a flat uint16
# ``.npy`` plus a uint64 offsets array, and reuse it on later passes.  This
# avoids re-running the BPE encoder on every counting/emission pass, which is
# the dominant cost for the full 163-shard corpus.
_TOKEN_CACHE_DIR: str | None = None


def _iter_split_token_streams(tokenizer, max_tokens: int | None, split: str):
    """Yield token-id lists (one per document, BOS-prepended) from a split.

    Uses the per-shard token cache when ``_TOKEN_CACHE_DIR`` is set.
    Stops after at least ``max_tokens`` tokens have been yielded when set.
    """
    lib = _load_upstream_lib()
    bos = tokenizer.get_bos_token_id()
    parquet_paths = lib.split_parquet_files(split)
    missing = [p for p in parquet_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {split} shards. Run code/prepare_data.py first: {missing}"
        )
    emitted = 0
    cache_dir = Path(_TOKEN_CACHE_DIR) if _TOKEN_CACHE_DIR else None
    for filepath in parquet_paths:
        tokens_per_doc = None
        if cache_dir is not None:
            shard_id = int(Path(filepath).stem.split("_")[-1])
            npy_path = cache_dir / f"shard_{shard_id:05d}.npy"
            off_path = cache_dir / f"shard_{shard_id:05d}.offsets.npy"
            if npy_path.is_file() and off_path.is_file():
                flat = np.fromfile(npy_path, dtype=np.uint16)
                offsets = np.fromfile(off_path, dtype=np.uint64)
                for start, end in zip(offsets[:-1], offsets[1:]):
                    doc_tokens = flat[start:end].tolist()
                    emitted += len(doc_tokens)
                    if max_tokens is not None and emitted > max_tokens:
                        return
                    yield doc_tokens
                continue
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(filepath)
        doc_lists: list[list[int]] = []
        for rg_idx in range(pf.num_row_groups):
            batch = pf.read_row_group(rg_idx).column("text").to_pylist()
            if batch:
                doc_lists.extend(tokenizer.encode(batch, prepend=bos))
        if cache_dir is not None and doc_lists:
            shard_id = int(Path(filepath).stem.split("_")[-1])
            npy_path = cache_dir / f"shard_{shard_id:05d}.npy"
            off_path = cache_dir / f"shard_{shard_id:05d}.offsets.npy"
            cache_dir.mkdir(parents=True, exist_ok=True)
            lengths = np.asarray([len(d) for d in doc_lists], dtype=np.uint64)
            offsets = np.concatenate(
                [np.zeros(1, dtype=np.uint64), np.cumsum(lengths)]
            )
            flat = np.concatenate(
                [np.asarray(d, dtype=np.uint16) for d in doc_lists]
            )
            flat.tofile(npy_path)
            offsets.tofile(off_path)
        for tokens in doc_lists:
            emitted += len(tokens)
            if max_tokens is not None and emitted > max_tokens:
                return
            yield tokens


def iter_train_token_streams(tokenizer, max_tokens: int | None):
    """Yield token-id lists (one per document, BOS-prepended) from train split."""
    yield from _iter_split_token_streams(tokenizer, max_tokens, split="train")


def iter_test_token_streams(tokenizer, max_tokens: int | None):
    """Yield token-id lists (one per document, BOS-prepended) from test split."""
    yield from _iter_split_token_streams(tokenizer, max_tokens, split="test")


def scan_histogram(tokenizer, bucket_count: int,
                   max_tokens: int | None,
                   order: int = 5) -> tuple[dict[int, Counter], list[int]]:
    """Scan the train corpus and build the per-bucket next-token histogram.

    ``order`` is the n-gram context length (3 = trigram, 5 = 5-gram).
    For each position ``i >= order``, the context is ``tokens[i-order:i]``
    and the target is ``tokens[i]``.

    Returns:
      hist: {bucket_id: Counter({next_token: count})}
      bucket_r: list[int] of length ``bucket_count`` with total occurrences
                per bucket (kept separately for fast median/quantile).
    """
    hist: dict[int, Counter] = defaultdict(Counter)
    bucket_r = [0] * bucket_count
    n_contexts = 0
    for tokens in iter_train_token_streams(tokenizer, max_tokens):
        if len(tokens) < order + 1:
            continue
        for i in range(order, len(tokens)):
            ctx = tokens[i - order:i]
            nxt = tokens[i]
            b = hash_n(ctx) % bucket_count
            hist[b][nxt] += 1
            bucket_r[b] += 1
            n_contexts += 1
    print(f"[scan] contexts={n_contexts:,}  non-empty buckets="
          f"{len(hist):,}  / {bucket_count:,}", flush=True)
    return hist, bucket_r


def scan_exact_histogram(
    tokenizer, max_tokens: int | None, order: int = 5
) -> dict[tuple[int, ...], Counter]:
    """Scan the complete train source and count exact context occurrences.

    This is the frequency index used by the experiment.  It deliberately
    keeps the context tuple as the key; hash buckets are not involved.
    """
    hist: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    n_contexts = 0
    for tokens in iter_train_token_streams(tokenizer, max_tokens):
        if len(tokens) < order + 1:
            continue
        for i in range(order, len(tokens)):
            context = tuple(tokens[i - order:i])
            hist[context][tokens[i]] += 1
            n_contexts += 1
    print(
        f"[scan:exact] contexts={n_contexts:,}  distinct_contexts={len(hist):,}",
        flush=True,
    )
    return hist


def scan_exact_counts_packed(
    tokenizer, max_tokens: int | None, order: int, vocab: int
) -> dict:
    """Count exact contexts with packed keys when the key fits in int64."""
    counts = Counter()
    total = 0
    for tokens in iter_train_token_streams(tokenizer, max_tokens):
        if len(tokens) < order + 1:
            continue
        for index in range(order, len(tokens)):
            key = _context_key(tokens, index, order, vocab)
            counts[key] += 1
            total += 1
    print(
        f"[scan:exact-packed] contexts={total:,}  "
        f"distinct_contexts={len(counts):,}",
        flush=True,
    )
    return counts


def _context_key(
    tokens: list[int], end: int, order: int, vocab: int
) -> int | tuple[int, ...]:
    context = tuple(int(token) for token in tokens[end - order:end])
    if order <= 3 and vocab**order <= (1 << 63) - 1:
        return _encode_context(context, vocab)
    return context


def _packed_context(tokens: list[int], end: int, order: int, vocab: int):
    return _context_key(tokens, end, order, vocab)


def _factor_for_count(
    count: int, *, alpha: float, r_ref: float, k_min: float, k_max: float
) -> float:
    k = (r_ref / count) ** alpha if alpha != 0 else 1.0
    return min(k_max, max(k_min, k))


def emit_exact_packed_stream(
    tokenizer,
    counts: Counter[int],
    *,
    role: str,
    alpha: float,
    r_ref: float,
    k_min: float,
    k_max: float,
    f_train: float,
    f_val: float,
    doc_len: int,
    sep_token: int,
    max_tokens: int | None,
    rng: random.Random,
    order: int,
    vocab: int,
    output_path: Path,
    emit_format: str = "txt",
) -> tuple[int, int]:
    """Stream exact-context blocks using the loader's block-aligned row stride.

    ``emit_format``: "txt" (legacy space-separated text), "bin" (flat uint16
    file next to the text path, for the numpy-mmap loader on huge corpora),
    or "both".
    """
    lam_fraction = f_train if role == "train" else f_val
    block_count = 0
    token_count = 0
    pending: list[str] = []
    write_txt = emit_format in ("txt", "both")
    write_bin = emit_format in ("bin", "both")
    bin_handle = None
    txt_handle = None
    bin_chunks: list[np.ndarray] = []
    bin_written = 0
    if write_bin:
        bin_handle = output_path.with_suffix(".bin").open("wb")
    if write_txt:
        txt_handle = output_path.open("w", encoding="utf-8")
    try:
        for tokens in iter_train_token_streams(tokenizer, max_tokens):
            if len(tokens) < order + 1:
                continue
            for index in range(order, len(tokens)):
                key = _packed_context(tokens, index, order, vocab)
                count = counts.get(key, 0)
                if count <= 0:
                    continue
                lam = lam_fraction * _factor_for_count(
                    count, alpha=alpha, r_ref=r_ref, k_min=k_min, k_max=k_max
                )
                copies = _poisson_draw(lam, rng)
                if copies <= 0:
                    continue
                block = [*tokens[index - order:index], tokens[index], sep_token]
                block_count += copies
                token_count += copies * len(block)
                if write_bin:
                    for _ in range(copies):
                        bin_chunks.append(np.asarray(block, dtype=np.uint16))
                    if len(bin_chunks) >= 16384:
                        arr = np.concatenate(bin_chunks)
                        arr.tofile(bin_handle)
                        bin_written += arr.size
                        bin_chunks.clear()
                if write_txt:
                    encoded = " ".join(map(str, block))
                    for _ in range(copies):
                        pending.append(encoded)
                    if len(pending) >= 8192:
                        txt_handle.write(" ".join(pending) + " ")
                        pending.clear()
        # Production rows use MAX_SEQ_LEN=2048, so the loader consumes 2045
        # block-aligned tokens and pads the remaining four input positions.
        # Keep the small synthetic-test contract (doc_len != 2048) intact.
        row_stride = (
            (doc_len + 1) // (order + 2) * (order + 2)
            if doc_len == 2048
            else doc_len
        )
        if write_bin:
            if bin_chunks:
                arr = np.concatenate(bin_chunks)
                arr.tofile(bin_handle)
                bin_written += arr.size
                bin_chunks.clear()
            remainder = bin_written % row_stride
            if remainder:
                pad = np.full(row_stride - remainder, sep_token, dtype=np.uint16)
                pad.tofile(bin_handle)
                bin_written += pad.size
            token_count = bin_written
        if write_txt:
            if pending:
                txt_handle.write(" ".join(pending) + " ")
                pending.clear()
            remainder = token_count % row_stride
            if remainder:
                txt_handle.write(" ".join([str(sep_token)] * (row_stride - remainder)))
                token_count += row_stride - remainder
            txt_handle.write("\n")
    finally:
        if bin_handle is not None:
            bin_handle.close()
        if txt_handle is not None:
            txt_handle.close()
    print(
        f"[emit:exact-packed:{role}] blocks={block_count:,} "
        f"tokens={token_count:,}",
        flush=True,
    )
    return block_count, token_count


def emit_unseen_val_stream(
    tokenizer,
    *,
    val_frac: float,
    doc_len: int,
    sep_token: int,
    max_tokens: int | None,
    rng: random.Random,
    order: int,
    vocab: int,
    output_path: Path,
    emit_format: str = "txt",
) -> tuple[int, int]:
    """Emit a small held-out val block stream from the TEST split.

    Test-split contexts are unseen in the train frequency index, so every
    occurrence is emitted with ``Poisson(val_frac)`` copies (factor k=1).
    This gives a small validation stream of genuinely unseen data, matching
    the "val loss on a small unseen segment" requirement of continuous runs.
    """
    block_count = 0
    token_count = 0
    pending: list[str] = []
    write_txt = emit_format in ("txt", "both")
    write_bin = emit_format in ("bin", "both")
    bin_handle = None
    txt_handle = None
    bin_chunks: list[np.ndarray] = []
    bin_written = 0
    if write_bin:
        bin_handle = output_path.with_suffix(".bin").open("wb")
    if write_txt:
        txt_handle = output_path.open("w", encoding="utf-8")
    try:
        for tokens in iter_test_token_streams(tokenizer, max_tokens):
            if len(tokens) < order + 1:
                continue
            for index in range(order, len(tokens)):
                copies = _poisson_draw(val_frac, rng)
                if copies <= 0:
                    continue
                block = [*tokens[index - order:index], tokens[index], sep_token]
                block_count += copies
                token_count += copies * len(block)
                if write_bin:
                    for _ in range(copies):
                        bin_chunks.append(np.asarray(block, dtype=np.uint16))
                    if len(bin_chunks) >= 16384:
                        arr = np.concatenate(bin_chunks)
                        arr.tofile(bin_handle)
                        bin_written += arr.size
                        bin_chunks.clear()
                if write_txt:
                    encoded = " ".join(map(str, block))
                    for _ in range(copies):
                        pending.append(encoded)
                    if len(pending) >= 8192:
                        txt_handle.write(" ".join(pending) + " ")
                        pending.clear()
        row_stride = (
            (doc_len + 1) // (order + 2) * (order + 2)
            if doc_len == 2048
            else doc_len
        )
        if write_bin:
            if bin_chunks:
                arr = np.concatenate(bin_chunks)
                arr.tofile(bin_handle)
                bin_written += arr.size
                bin_chunks.clear()
            remainder = bin_written % row_stride
            if remainder:
                pad = np.full(row_stride - remainder, sep_token, dtype=np.uint16)
                pad.tofile(bin_handle)
                bin_written += pad.size
            token_count = bin_written
        if write_txt:
            if pending:
                txt_handle.write(" ".join(pending) + " ")
                pending.clear()
            remainder = token_count % row_stride
            if remainder:
                txt_handle.write(" ".join([str(sep_token)] * (row_stride - remainder)))
                token_count += row_stride - remainder
            txt_handle.write("\n")
    finally:
        if bin_handle is not None:
            bin_handle.close()
        if txt_handle is not None:
            txt_handle.close()
    print(
        f"[emit:unseen-val:{'test'}] blocks={block_count:,} "
        f"tokens={token_count:,}",
        flush=True,
    )
    return block_count, token_count


def compute_context_factors(
    context_counts: dict[tuple[int, ...], int],
    *,
    alpha: float,
    r_ref: float,
    k_min: float,
    k_max: float,
) -> dict[tuple[int, ...], float]:
    """Compute sampling factors from exact train context frequencies."""
    factors = {}
    for context, count in context_counts.items():
        if count <= 0:
            factors[context] = 1.0
            continue
        k = (r_ref / count) ** alpha if alpha != 0 else 1.0
        factors[context] = min(k_max, max(k_min, k))
    return factors


def sample_exact_splits(
    exact_hist: dict[tuple[int, ...], Counter],
    factors: dict[tuple[int, ...], float],
    *,
    f_train: float,
    f_val: float,
) -> dict[tuple[int, ...], dict]:
    """Create target counts keyed by exact context, never by hash bucket."""
    splits = {}
    for context, histogram in exact_hist.items():
        count = sum(histogram.values())
        if count <= 0:
            continue
        splits[context] = {
            "r": count,
            "k": factors[context],
            "n_train_target": int(round(count * f_train * factors[context])),
            "n_val_target": int(round(count * f_val)),
            "next_hist_topk": histogram.most_common(8),
            "n_distinct_next": len(histogram),
        }
    return splits


# ---------------------------------------------------------------------------
# Factor + sampling
# ---------------------------------------------------------------------------

def compute_factors(bucket_r: list[int], *, alpha: float, r_ref: float,
                    k_min: float, k_max: float) -> list[float]:
    """k(b) = clip((r_ref / r(b))^alpha, k_min, k_max); r(b)==0 -> 1.0."""
    out = [1.0] * len(bucket_r)
    for b, r in enumerate(bucket_r):
        if r <= 0:
            continue
        ratio = r_ref / r
        k = ratio ** alpha if alpha != 0 else 1.0
        if k < k_min:
            k = k_min
        elif k > k_max:
            k = k_max
        out[b] = k
    return out


def _multinomial_draw(hist: Counter, n: int, rng: random.Random) -> list[int]:
    """Draw n tokens from the empirical distribution ``hist``.

    Uses the alias-method-free, weighted-choice approach: expand to a list
    only when n is small relative to the histogram total; otherwise sample
    by cumulative-sum binary search for efficiency.
    """
    if n <= 0:
        return []
    total = sum(hist.values())
    if total == 0:
        return []
    items = list(hist.items())
    if len(items) == 1:
        return [items[0][0]] * n
    # For very small n, direct weighted choice is fine and fast.
    if n <= 64:
        # cumulative weights
        cum = []
        s = 0
        for tok, c in items:
            s += c
            cum.append(s)
        out = []
        for _ in range(n):
            x = rng.randrange(total)
            # linear scan is fine for small item counts; most next-token
            # histograms have few entries per bucket.
            for idx, c in enumerate(cum):
                if x < c:
                    out.append(items[idx][0])
                    break
            else:
                out.append(items[-1][0])
        return out
    # Larger n: build cumulative array + binary search.
    tokens = [t for t, _ in items]
    cum = []
    s = 0
    for _, c in items:
        s += c
        cum.append(s)
    out = []
    import bisect
    for _ in range(n):
        x = rng.randrange(total)
        idx = bisect.bisect_right(cum, x)
        out.append(tokens[idx])
    return out


def sample_splits(hist: dict[int, Counter], bucket_r: list[int],
                  factors: list[float], *, f_train: float, f_val: float,
                  rng_train: random.Random, rng_val: random.Random,
                  ) -> dict[int, dict]:
    """For each non-empty bucket, decide target train/val sample counts.

    Returns ``{bucket: {r, k, n_train_target, n_val_target}}``.

    NOTE: This only decides the *target counts* per bucket.  The actual
    sampling of which real occurrences to keep/drop/replicate happens in
    ``scan_and_emit``, which re-scans the corpus and applies a per-occurrence
    replication draw so the emitted blocks carry real 5-gram contexts.
    """
    out: dict[int, dict] = {}
    for b, h in hist.items():
        r = bucket_r[b]
        k = factors[b]
        n_train = int(round(r * f_train * k))
        n_val = int(round(r * f_val))
        if n_train == 0 and n_val == 0:
            continue
        out[b] = {
            "r": r,
            "k": k,
            "n_train_target": n_train,
            "n_val_target": n_val,
        }
    return out


# ---------------------------------------------------------------------------
# Pass 2: re-scan corpus and emit real n-gram blocks with replication
# ---------------------------------------------------------------------------

# Block layout: [c0, ..., c_{order-1}, next, SEP]  (order + 2 tokens)
# The SEP is also reused for padding the stream to a multiple of DOC_LEN.
# BLOCK_LEN is order-dependent; set at generate() time.


def scan_and_emit(tokenizer, splits: dict[int, dict], bucket_count: int,
                  *, role: str, f_train: float, f_val: float,
                  doc_len: int, sep_token: int, max_tokens: int | None,
                  rng: random.Random, order: int = 5) -> tuple[list[int], list[dict]]:
    """Second pass: re-scan the corpus and emit real n-gram blocks.

    For each real occurrence ``(ctx, next)`` with bucket ``b``:
      - We decide how many copies to emit into the requested role's stream
        using a Poisson draw with mean ``lambda(b)``:
            train: lambda = n_train_target(b) / r(b)   (= f_train * k(b))
            val:   lambda = n_val_target(b)   / r(b)   (= f_val)
        This naturally handles up-sampling (lambda > 1 => some occurrences
        emit multiple copies) and down-sampling (lambda < 1 => some
        occurrences emit zero copies).  The emitted copies carry the *real*
        n-gram context, so the model and the n-gram injection tables see
        authentic language.

    The alternative role (e.g. val when emitting train) is simply skipped
    here — each role is emitted by its own ``scan_and_emit`` call so the
    two streams are independent draws (the coincidental-gap source).

    Returns (tokens, block_meta) where block_meta records per block:
      {bucket, next, block_start, block_len}.
    """
    block_len = order + 2  # [c0..c_{order-1}, next, SEP]
    target_key = "n_train_target" if role == "train" else "n_val_target"
    # Precompute lambda per bucket (mean copies per occurrence).
    lambdas: dict[int, float] = {}
    for b, d in splits.items():
        r = d["r"]
        if r <= 0:
            continue
        lambdas[b] = d[target_key] / r

    tokens: list[int] = []
    meta: list[dict] = []
    n_emitted = 0
    for tk in iter_train_token_streams(tokenizer, max_tokens):
        if len(tk) < order + 1:
            continue
        for i in range(order, len(tk)):
            ctx = tk[i - order:i]
            nxt = tk[i]
            b = hash_n(ctx) % bucket_count
            lam = lambdas.get(b)
            if lam is None:
                continue
            # Poisson-draw the number of copies (0,1,2,...).
            copies = _poisson_draw(lam, rng)
            for _ in range(copies):
                start = len(tokens)
                tokens.extend(ctx)
                tokens.append(nxt)
                tokens.append(sep_token)
                meta.append({
                    "bucket": b,
                    "next": nxt,
                    "block_start": start,
                    "block_len": block_len,
                })
                n_emitted += 1
    # Pad to a multiple of doc_len with SEP.
    rem = len(tokens) % doc_len
    if rem:
        tokens.extend([sep_token] * (doc_len - rem))
    print(f"[emit:{role}] blocks={n_emitted:,}  tokens={len(tokens):,}",
          flush=True)
    return tokens, meta


def scan_and_emit_exact(
    tokenizer,
    splits: dict[tuple[int, ...], dict],
    *,
    role: str,
    f_train: float,
    f_val: float,
    doc_len: int,
    sep_token: int,
    max_tokens: int | None,
    rng: random.Random,
    order: int = 5,
) -> tuple[list[int], list[dict]]:
    """Emit blocks using exact-context factors and real contexts."""
    block_len = order + 2
    target_key = "n_train_target" if role == "train" else "n_val_target"
    lambdas = {
        context: values[target_key] / values["r"]
        for context, values in splits.items()
        if values["r"] > 0
    }
    tokens = []
    metadata = []
    for token_list in iter_train_token_streams(tokenizer, max_tokens):
        if len(token_list) < order + 1:
            continue
        for index in range(order, len(token_list)):
            context = tuple(token_list[index - order:index])
            copies = _poisson_draw(lambdas.get(context, 0.0), rng)
            for _ in range(copies):
                start = len(tokens)
                tokens.extend(context)
                tokens.extend((token_list[index], sep_token))
                metadata.append({
                    "context": list(context),
                    "next": token_list[index],
                    "block_start": start,
                    "block_len": block_len,
                })
    remainder = len(tokens) % doc_len
    if remainder:
        tokens.extend([sep_token] * (doc_len - remainder))
    print(
        f"[emit:exact:{role}] blocks={len(metadata):,}  tokens={len(tokens):,}",
        flush=True,
    )
    return tokens, metadata


def _poisson_draw(lam: float, rng: random.Random) -> int:
    """Draw a non-negative integer from Poisson(lam).

    For lam < 30 we use Knuth's algorithm; for larger lam we fall back to a
    normal approximation (lam + sqrt(lam)*z, clamped >= 0).  This is used
    only for replication counts so exactness is not critical.
    """
    if lam <= 0:
        return 0
    if lam < 30:
        # Knuth
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= L:
                return k - 1
    # normal approximation
    z = rng.gauss(0.0, 1.0)
    val = int(round(lam + math.sqrt(lam) * z))
    return max(0, val)


# ---------------------------------------------------------------------------
# Stats / metadata
# ---------------------------------------------------------------------------

def _quantiles(sorted_vals: list[int], qs: list[float]) -> list[int]:
    if not sorted_vals:
        return [0] * len(qs)
    n = len(sorted_vals)
    out = []
    for q in qs:
        idx = max(0, min(n - 1, int(round((n - 1) * q))))
        out.append(sorted_vals[idx])
    return out


def _write_fivegram_counts_npz(out_dir: Path, hist: dict[int, Counter],
                                bucket_r: list[int], bucket_count: int,
                                vocab: int) -> None:
    """Write ``fivegram_counts.npz`` for the cluster's GlobalNgramFrequencyIndex.

    The cluster index keys n-grams by a single integer computed as
    ``((c0*V + c1)*V + c2)*V + c3)*V + c4`` (base-V encoding of the context).
    But our histogram is keyed by *hash bucket* (hash5(c0..c4) % bucket_count),
    NOT by the full 5-gram context — we intentionally collapsed distinct
    5-grams into 1M buckets.  So we cannot recover the full base-V key.

    Instead, we emit the frequency index keyed by the *bucket id* itself,
    and the cluster's fivegram branch lookup must compute the same bucket id
    (hash5 % bucket_count) at query time.  This is a deliberate divergence
    from bigram/trigram (which use the full base-V key) because 5-grams would
    overflow int64 at vocab=8192 (8192^5 = 3.6e19 > 2^63).

    Keys = sorted non-empty bucket ids (int64), counts = r(b) (int64).
    """
    import numpy as np
    nonempty = [(b, bucket_r[b]) for b in range(bucket_count) if bucket_r[b] > 0]
    nonempty.sort(key=lambda x: x[0])
    if not nonempty:
        raise RuntimeError("no non-empty buckets; cannot write frequency index")
    keys = np.array([b for b, _ in nonempty], dtype=np.int64)
    counts = np.array([c for _, c in nonempty], dtype=np.int64)
    np.savez(out_dir / "fivegram_counts.npz", keys=keys, counts=counts)
    print(f"[fivegram_counts] wrote {len(nonempty):,} bucket entries to "
          f"{out_dir / 'fivegram_counts.npz'}", flush=True)


def _encode_context(context: tuple[int, ...], vocab: int) -> int:
    key = 0
    for token in context:
        key = key * vocab + int(token)
    if key > (1 << 63) - 1:
        raise ValueError(
            "exact context key exceeds signed int64; use order<=3 at vocab=8192 "
            "or add a packed multiword index before generating this dataset"
        )
    return key


def _write_exact_counts_npz(
    out_dir: Path,
    exact_hist: dict[tuple[int, ...], Counter],
    vocab: int,
) -> None:
    import numpy as np

    if exact_hist and len(next(iter(exact_hist))) > 3:
        contexts = np.asarray(list(exact_hist), dtype=np.int32)
        counts = np.asarray(
            [sum(hist.values()) for hist in exact_hist.values()],
            dtype=np.int64,
        )
        np.savez(
            out_dir / "exact_ngram_counts.npz",
            contexts=contexts,
            counts=counts,
        )
        np.savez(
            out_dir / "fivegram_counts.npz",
            contexts=contexts,
            counts=counts,
        )
        print(
            f"[exact_counts] wrote {len(contexts):,} context-matrix entries to "
            f"{out_dir / 'exact_ngram_counts.npz'}",
            flush=True,
        )
        return

    rows = sorted(
        (_encode_context(context, vocab), sum(hist.values()), context)
        for context, hist in exact_hist.items()
    )
    keys = np.asarray([row[0] for row in rows], dtype=np.int64)
    counts = np.asarray([row[1] for row in rows], dtype=np.int64)
    np.savez(out_dir / "exact_ngram_counts.npz", keys=keys, counts=counts)
    np.savez(out_dir / "fivegram_counts.npz", keys=keys, counts=counts)
    (out_dir / "exact_ngram_contexts.json").write_text(
        json.dumps(
            {
                " ".join(map(str, context)): {
                    "frequency": int(count),
                    "next_hist_topk": exact_hist[context].most_common(8),
                    "n_distinct_next": len(exact_hist[context]),
                }
                for _, count, context in rows
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    print(
        f"[exact_counts] wrote {len(rows):,} context entries to "
        f"{out_dir / 'exact_ngram_counts.npz'}",
        flush=True,
    )


def _write_packed_exact_counts_npz(
    out_dir: Path, exact_counts: dict
) -> None:
    import numpy as np

    if exact_counts and isinstance(next(iter(exact_counts)), tuple):
        contexts = np.asarray(list(exact_counts), dtype=np.int32)
        counts = np.asarray(
            [exact_counts[tuple(context)] for context in contexts],
            dtype=np.int64,
        )
        np.savez(
            out_dir / "exact_ngram_counts.npz",
            contexts=contexts,
            counts=counts,
        )
        np.savez(
            out_dir / "fivegram_counts.npz",
            contexts=contexts,
            counts=counts,
        )
        (out_dir / "exact_ngram_contexts.json").write_text(
            json.dumps(
                {
                    "format": "exact_context_matrix",
                    "frequency_definition": "exact_train_epoch_context_count",
                    "source": "exact_ngram_counts.npz",
                    "n_contexts": len(contexts),
                    "order": int(contexts.shape[1]),
                },
                sort_keys=True,
            )
            + "\n"
        )
        print(
            f"[exact_counts] wrote {len(contexts):,} context-matrix entries to "
            f"{out_dir / 'exact_ngram_counts.npz'}",
            flush=True,
        )
        return

    keys = np.asarray(sorted(exact_counts), dtype=np.int64)
    counts = np.asarray([exact_counts[key] for key in keys], dtype=np.int64)
    np.savez(out_dir / "exact_ngram_counts.npz", keys=keys, counts=counts)
    np.savez(out_dir / "fivegram_counts.npz", keys=keys, counts=counts)
    if len(keys) <= 1_000_000:
        (out_dir / "exact_ngram_contexts.json").write_text(
            json.dumps(
                {
                    str(int(key)): {
                        "frequency": int(exact_counts[key]),
                        "frequency_definition": "exact_train_epoch_context_count",
                    }
                    for key in keys.tolist()
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    else:
        (out_dir / "exact_ngram_contexts.json").write_text(
            json.dumps(
                {
                    "format": "packed_exact_context_index",
                    "frequency_definition": "exact_train_epoch_context_count",
                    "source": "exact_ngram_counts.npz",
                    "n_contexts": len(keys),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(
        f"[exact_counts] wrote {len(keys):,} packed context entries to "
        f"{out_dir / 'exact_ngram_counts.npz'}",
        flush=True,
    )


def build_metadata(*, vocab: int, bucket_count: int, alpha: float, r_ref: float,
                   k_min: float, k_max: float, f_train: float, f_val: float,
                   dataset_seed: int, doc_len: int, sep_token: int,
                   exact_counts: dict[int, int],
                   splits: dict[int, dict],
                   train_tokens_len: int, val_tokens_len: int,
                   max_tokens: int | None, order: int = 5) -> dict:
    nonempty = [r for r in exact_counts.values() if r > 0]
    nonempty.sort()
    n_buckets = len(nonempty)
    total_r = sum(nonempty)
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    qv = _quantiles(nonempty, qs)
    # effective r_train distribution after factor
    eff_r_train = [d["n_train_target"] for d in splits.values()]
    eff_r_train.sort()
    qv_train = _quantiles(eff_r_train, qs)
    return {
        "schema_version": 1,
        "order": order,
        "context_len": order,
        "block_len": order + 2,
        "vocab": vocab,
        "sep_token": sep_token,
        "doc_len": doc_len,
        "bucket_count": bucket_count,
        "alpha": alpha,
        "r_ref": r_ref,
        "k_min": k_min,
        "k_max": k_max,
        "f_train": f_train,
        "f_val": f_val,
        "dataset_seed": dataset_seed,
        "max_tokens_scanned": max_tokens,
        "n_nonempty_buckets": 0,
        "n_distinct_exact_contexts": n_buckets,
        "total_contexts": total_r,
        "bucket_r_quantiles": {},
        "exact_context_frequency_quantiles": {
            f"q{int(q*100)}": v for q, v in zip(qs, qv)
        },
        "eff_r_train_quantiles": {f"q{int(q*100)}": v for q, v in zip(qs, qv_train)},
        "frequency_definition": "exact_train_epoch_context_count",
        "frequency_source_split": "train",
        "frequency_key_type": "exact_context",
        "hash_bucket_occupancy_diagnostic": False,
        "frequency_index_scope": (
            "complete upstream train epoch before controlled block resampling"
        ),
        "train_tokens": train_tokens_len,
        "val_tokens": val_tokens_len,
        "loader_row_stride": (
            (doc_len + 1) // (order + 2) * (order + 2)
            if doc_len == 2048
            else doc_len
        ),
        "loader_rows_train": train_tokens_len // (
            (
                (doc_len + 1) // (order + 2) * (order + 2)
                if doc_len == 2048
                else doc_len
            )
        ),
        "loader_rows_val": val_tokens_len // (
            (
                (doc_len + 1) // (order + 2) * (order + 2)
                if doc_len == 2048
                else doc_len
            )
        ),
        "stream_padding": "global_block_aligned_row_stride",
        "block_alignment_version": 2,
        "loader_selection": _load_upstream_lib().describe_shard_selection(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Vectorized fast paths for full-corpus scale (alpha=0, max_tokens=None,
# --emit-format bin).  Reference-equivalent: same per-document context
# windows, same packed keys, same Poisson resampling distribution.
# ---------------------------------------------------------------------------

def _iter_split_shard_flat(split: str):
    """Yield ``(shard_id, flat_uint16_tokens, doc_offsets)`` for cached shards."""
    lib = _load_upstream_lib()
    parquet_paths = lib.split_parquet_files(split)
    cache_dir = Path(_TOKEN_CACHE_DIR)
    if cache_dir is None:
        raise RuntimeError("fast path requires --token-cache")
    for filepath in parquet_paths:
        shard_id = int(Path(filepath).stem.split("_")[-1])
        npy_path = cache_dir / f"shard_{shard_id:05d}.npy"
        off_path = cache_dir / f"shard_{shard_id:05d}.offsets.npy"
        if not (npy_path.is_file() and off_path.is_file()):
            raise FileNotFoundError(f"fast path: missing cached shard {npy_path}")
        flat = np.fromfile(npy_path, dtype=np.uint16)
        offsets = np.fromfile(off_path, dtype=np.uint64)
        yield shard_id, flat, offsets


def _context_window_mask(flat: np.ndarray, offsets: np.ndarray, order: int):
    """Mask + next-token positions for per-document context windows.

    A window of ``order`` context tokens starting at ``j`` is valid iff both
    ``j`` and ``j + order`` lie inside the same document (no cross-document
    contexts), matching the reference ``for i in range(order, len(doc))``
    scan with ``j = i - order`` and next token at ``j + order``.
    """
    n = len(flat)
    if n < order + 1:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64)
    starts = offsets[:-1].astype(np.int64)
    ends = offsets[1:].astype(np.int64)
    lengths = ends - starts
    j = np.arange(n - order, dtype=np.int64)
    doc_idx = np.searchsorted(ends, j, side="right")
    doc_idx = np.clip(doc_idx, 0, len(starts) - 1)
    rel = j - starts[doc_idx]
    valid = (rel >= 0) & (rel + order < lengths[doc_idx])
    return valid, (j + order)[valid]


def _pack_context_keys(ctx: np.ndarray, vocab: int, order: int) -> np.ndarray:
    weights = np.asarray(
        [vocab ** (order - 1 - k) for k in range(order)], dtype=np.int64
    )
    return (ctx.astype(np.int64) * weights).sum(axis=1)


def _poisson_cdf(lam: float, max_k: int = 24) -> np.ndarray:
    probs = np.asarray(
        [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(max_k)],
        dtype=np.float64,
    )
    cdf = np.cumsum(probs)
    return np.minimum(cdf / cdf[-1], 1.0)


def scan_exact_counts_packed_fast(
    tokenizer, max_tokens: int | None, order: int, vocab: int
) -> dict:
    """Vectorized exact-context scan (per-shard numpy unique + pair merge)."""
    assert max_tokens is None, "fast scan requires max_tokens=None"
    use_context_matrix = order > 3 or vocab**order > (1 << 63) - 1
    per_shard_keys: list[np.ndarray] = []
    per_shard_counts: list[np.ndarray] = []
    total = 0
    for _, flat, offsets in _iter_split_shard_flat("train"):
        if len(flat) < order + 1:
            continue
        valid, _ = _context_window_mask(flat, offsets, order)
        if not valid.any():
            continue
        win = np.lib.stride_tricks.sliding_window_view(
            flat[:-1].astype(np.int64), order
        )
        contexts = win[valid].astype(np.int32, copy=False)
        if use_context_matrix:
            ucontexts, ucounts = np.unique(contexts, axis=0, return_counts=True)
            per_shard_keys.append(ucontexts)
        else:
            keys = _pack_context_keys(contexts, vocab, order)
            ukeys, ucounts = np.unique(keys, return_counts=True)
            per_shard_keys.append(ukeys)
        per_shard_counts.append(ucounts)
        total += int(valid.sum())
    counts: Counter[int] = Counter()
    if per_shard_keys and use_context_matrix:
        contexts_all = np.concatenate(per_shard_keys)
        counts_all = np.concatenate(per_shard_counts)
        order_idx = np.lexsort(
            tuple(contexts_all[:, column] for column in reversed(range(order)))
        )
        contexts_sorted = contexts_all[order_idx]
        counts_sorted = counts_all[order_idx]
        boundaries = np.concatenate(
            [[0], np.flatnonzero(np.any(
                contexts_sorted[1:] != contexts_sorted[:-1], axis=1
            )) + 1]
        )
        unique_contexts = contexts_sorted[boundaries]
        unique_counts = np.add.reduceat(counts_sorted, boundaries)
        counts = {
            tuple(context.tolist()): int(count)
            for context, count in zip(unique_contexts, unique_counts)
        }
        del contexts_all, counts_all, contexts_sorted, counts_sorted
    elif per_shard_keys:
        keys_all = np.concatenate(per_shard_keys)
        cnts_all = np.concatenate(per_shard_counts)
        order_idx = np.argsort(keys_all, kind="stable")
        k = keys_all[order_idx]
        c = cnts_all[order_idx]
        boundaries = np.concatenate(
            [[0], np.flatnonzero(k[1:] != k[:-1]) + 1]
        )
        ukeys = k[boundaries]
        ucounts = np.add.reduceat(c, boundaries)
        counts.update(dict(zip(ukeys.tolist(), ucounts.tolist())))
        del keys_all, cnts_all, k, c, order_idx, boundaries
    print(
        f"[scan:exact-packed:fast] contexts={total:,}  "
        f"distinct_contexts={len(counts):,}",
        flush=True,
    )
    return counts


def _emit_shard_blocks_fast(
    flat: np.ndarray, offsets: np.ndarray, *, order: int, vocab: int,
    lam: float, cdf: np.ndarray, rng_gen, sep_token: int,
) -> np.ndarray | None:
    """Vectorized poisson-resampled blocks for one shard (uint16 flat)."""
    if len(flat) < order + 1:
        return None
    valid, nxt_pos = _context_window_mask(flat, offsets, order)
    if not valid.any():
        return None
    t = flat.astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(t[:-1], order)
    ctx = win[valid]
    nxt = t[nxt_pos]
    n = len(ctx)
    r = rng_gen.random(n)
    copies = np.searchsorted(cdf, r).astype(np.int64)
    keep = copies > 0
    nk = int(keep.sum())
    if nk == 0:
        return None
    blocks = np.concatenate([ctx[keep], nxt[keep, None]], axis=1)
    blocks = np.repeat(blocks, copies[keep], axis=0)
    out = np.empty((len(blocks), order + 2), dtype=np.uint16)
    out[:, :order + 1] = blocks.astype(np.uint16)
    out[:, order + 1] = sep_token
    return out.reshape(-1)


def emit_exact_packed_stream_fast(
    tokenizer, counts, *, role: str, alpha: float, r_ref: float,
    k_min: float, k_max: float, f_train: float, f_val: float,
    doc_len: int, sep_token: int, max_tokens: int | None,
    rng, order: int, vocab: int, output_path: Path, emit_format: str,
    dataset_seed: int,
) -> tuple[int, int]:
    """Fast vectorized emit for alpha=0 (factor k=1), bin output, no max_tokens."""
    assert alpha == 0.0, "fast emit requires alpha=0 (factor k=1)"
    assert max_tokens is None, "fast emit requires max_tokens=None"
    assert emit_format == "bin", "fast emit requires --emit-format bin"
    lam = f_train if role == "train" else f_val
    cdf = _poisson_cdf(lam)
    seed = (dataset_seed ^ 0x4444) if role == "train" else (dataset_seed ^ 0x5555)
    rng_gen = np.random.default_rng(seed)
    row_stride = (doc_len + 1) // (order + 2) * (order + 2) if doc_len == 2048 else doc_len
    bin_handle = output_path.with_suffix(".bin").open("wb")
    bin_written = 0
    block_count = 0
    try:
        for _, flat, offsets in _iter_split_shard_flat("train"):
            chunk = _emit_shard_blocks_fast(
                flat, offsets, order=order, vocab=vocab, lam=lam,
                cdf=cdf, rng_gen=rng_gen, sep_token=sep_token,
            )
            if chunk is None:
                continue
            chunk.tofile(bin_handle)
            bin_written += chunk.size
            block_count += len(chunk) // (order + 2)
        remainder = bin_written % row_stride
        if remainder:
            pad = np.full(row_stride - remainder, sep_token, dtype=np.uint16)
            pad.tofile(bin_handle)
            bin_written += pad.size
    finally:
        bin_handle.close()
    token_count = bin_written
    print(
        f"[emit:exact-packed:{role}:fast] blocks={block_count:,} "
        f"tokens={token_count:,}",
        flush=True,
    )
    return block_count, token_count


def emit_unseen_val_stream_fast(
    tokenizer, *, val_frac: float, doc_len: int, sep_token: int,
    max_tokens: int | None, rng, order: int, vocab: int,
    output_path: Path, emit_format: str, dataset_seed: int,
) -> tuple[int, int]:
    """Fast vectorized emit of the unseen test-split val stream (Poisson val_frac)."""
    assert max_tokens is None, "fast val emit requires max_tokens=None"
    assert emit_format == "bin", "fast val emit requires --emit-format bin"
    cdf = _poisson_cdf(val_frac)
    rng_gen = np.random.default_rng(dataset_seed ^ 0x5555)
    row_stride = (doc_len + 1) // (order + 2) * (order + 2) if doc_len == 2048 else doc_len
    bin_handle = output_path.with_suffix(".bin").open("wb")
    bin_written = 0
    block_count = 0
    try:
        for _, flat, offsets in _iter_split_shard_flat("test"):
            chunk = _emit_shard_blocks_fast(
                flat, offsets, order=order, vocab=vocab, lam=val_frac,
                cdf=cdf, rng_gen=rng_gen, sep_token=sep_token,
            )
            if chunk is None:
                continue
            chunk.tofile(bin_handle)
            bin_written += chunk.size
            block_count += len(chunk) // (order + 2)
        remainder = bin_written % row_stride
        if remainder:
            pad = np.full(row_stride - remainder, sep_token, dtype=np.uint16)
            pad.tofile(bin_handle)
            bin_written += pad.size
    finally:
        bin_handle.close()
    token_count = bin_written
    print(
        f"[emit:unseen-val:fast] blocks={block_count:,} "
        f"tokens={token_count:,}",
        flush=True,
    )
    return block_count, token_count


def generate(out_dir: Path, *, alpha: float, bucket_count: int,
             f_train: float, f_val: float, k_min: float, k_max: float,
             r_ref_mode: str, r_ref_fixed: float | None,
             dataset_seed: int, doc_len: int, max_tokens: int | None,
             tokenizer_dir: str | None, order: int = 5,
             val_source: str = "train", val_frac: float | None = None,
             emit_format: str = "txt",
             token_cache_dir: str | None = None,
             use_fast_scan: bool = False,
             use_fast_emit: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    global _TOKEN_CACHE_DIR
    _TOKEN_CACHE_DIR = token_cache_dir

    lib = _load_upstream_lib()
    tokenizer = lib.Tokenizer.from_directory(tokenizer_dir) if tokenizer_dir else lib.Tokenizer.from_directory()
    vocab = tokenizer.get_vocab_size()
    sep_token = vocab - 1

    print(f"[data_gen] order={order}  vocab={vocab}  sep={sep_token}  alpha={alpha}  "
          f"buckets={bucket_count:,}  max_tokens={max_tokens}", flush=True)

    if use_fast_scan:
        exact_packed_counts = scan_exact_counts_packed_fast(
            tokenizer, max_tokens, order=order, vocab=vocab
        )
    else:
        exact_packed_counts = scan_exact_counts_packed(
            tokenizer, max_tokens, order=order, vocab=vocab
        )
    exact_counts = exact_packed_counts

    # r_ref
    nonempty = sorted(exact_counts.values())
    if r_ref_mode == "median":
        r_ref = float(nonempty[len(nonempty) // 2]) if nonempty else 1.0
    elif r_ref_mode == "mean":
        r_ref = (sum(nonempty) / len(nonempty)) if nonempty else 1.0
    elif r_ref_mode == "fixed":
        assert r_ref_fixed is not None and r_ref_fixed > 0
        r_ref = float(r_ref_fixed)
    else:
        raise ValueError(f"unknown r_ref_mode {r_ref_mode!r}")
    print(f"[data_gen] r_ref={r_ref:.2f}  (mode={r_ref_mode})", flush=True)

    factors = {
        key: _factor_for_count(
            count, alpha=alpha, r_ref=r_ref, k_min=k_min, k_max=k_max
        )
        for key, count in exact_counts.items()
    }
    # factor distribution summary
    fk = sorted(factors.values())
    fk_q = _quantiles(fk, [0.0, 0.1, 0.5, 0.9, 1.0])
    print(f"[data_gen] factor k quantiles (min,p10,p50,p90,max) = {fk_q}",
          flush=True)

    rng_train = random.Random(dataset_seed ^ 0x1111)
    rng_val = random.Random(dataset_seed ^ 0x2222)
    rng_emit_tr = random.Random(dataset_seed ^ 0x4444)
    rng_emit_va = random.Random(dataset_seed ^ 0x5555)

    train_token_count = 0
    val_token_count = 0
    if use_fast_emit:
        train_blocks, train_token_count = emit_exact_packed_stream_fast(
            tokenizer, exact_counts, role="train", alpha=alpha, r_ref=r_ref,
            k_min=k_min, k_max=k_max, f_train=f_train, f_val=f_val,
            doc_len=doc_len, sep_token=sep_token, max_tokens=max_tokens,
            rng=rng_emit_tr, order=order, vocab=vocab,
            output_path=out_dir / "train_tokens.txt",
            emit_format=emit_format, dataset_seed=dataset_seed,
        )
    else:
        train_blocks, train_token_count = emit_exact_packed_stream(
            tokenizer, exact_counts, role="train", alpha=alpha, r_ref=r_ref,
            k_min=k_min, k_max=k_max, f_train=f_train, f_val=f_val,
            doc_len=doc_len, sep_token=sep_token, max_tokens=max_tokens,
            rng=rng_emit_tr, order=order, vocab=vocab,
            output_path=out_dir / "train_tokens.txt",
            emit_format=emit_format,
        )
    if val_source == "test":
        if use_fast_emit:
            val_blocks, val_token_count = emit_unseen_val_stream_fast(
                tokenizer,
                val_frac=val_frac if val_frac is not None else 0.02,
                doc_len=doc_len, sep_token=sep_token, max_tokens=max_tokens,
                rng=rng_emit_va, order=order, vocab=vocab,
                output_path=out_dir / "val_tokens.txt",
                emit_format=emit_format, dataset_seed=dataset_seed,
            )
        else:
            val_blocks, val_token_count = emit_unseen_val_stream(
                tokenizer,
                val_frac=val_frac if val_frac is not None else 0.02,
                doc_len=doc_len, sep_token=sep_token, max_tokens=max_tokens,
                rng=rng_emit_va, order=order, vocab=vocab,
                output_path=out_dir / "val_tokens.txt",
                emit_format=emit_format,
            )
    else:
        if use_fast_emit:
            val_blocks, val_token_count = emit_exact_packed_stream_fast(
                tokenizer, exact_counts, role="val", alpha=alpha, r_ref=r_ref,
                k_min=k_min, k_max=k_max, f_train=f_train, f_val=f_val,
                doc_len=doc_len, sep_token=sep_token, max_tokens=max_tokens,
                rng=rng_emit_va, order=order, vocab=vocab,
                output_path=out_dir / "val_tokens.txt",
                emit_format=emit_format, dataset_seed=dataset_seed,
            )
        else:
            val_blocks, val_token_count = emit_exact_packed_stream(
                tokenizer, exact_counts, role="val", alpha=alpha, r_ref=r_ref,
                k_min=k_min, k_max=k_max, f_train=f_train, f_val=f_val,
                doc_len=doc_len, sep_token=sep_token, max_tokens=max_tokens,
                rng=rng_emit_va, order=order, vocab=vocab,
                output_path=out_dir / "val_tokens.txt",
                emit_format=emit_format,
            )

    _write_packed_exact_counts_npz(out_dir, exact_counts)
    if len(exact_counts) <= 1_000_000:
        contexts_payload = {
            (
                " ".join(map(str, key))
                if isinstance(key, tuple)
                else str(int(key))
            ): {
                "r": int(count),
                "k": float(factors[key]),
                "n_train_target": round(count * f_train * factors[key]),
                "n_val_target": round(count * f_val),
                "n_train_actual": None,
                "n_val_actual": None,
                "next_hist_topk": [],
                "n_distinct_next": None,
                "frequency_definition": "exact_train_epoch_context_count",
            }
            for key, count in exact_counts.items()
        }
        (out_dir / "contexts.json").write_text(
            json.dumps(contexts_payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
    else:
        (out_dir / "contexts.json").write_text(
            json.dumps(
                {
                    "format": "packed_exact_context_index",
                    "frequency_definition": "exact_train_epoch_context_count",
                    "source": "exact_ngram_counts.npz",
                    "n_contexts": len(exact_counts),
                },
                sort_keys=True,
            )
            + "\n"
        )

    exact_q = _quantiles(nonempty, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    effective_counts = sorted(
        round(count * f_train * factors[key])
        for key, count in exact_counts.items()
    )
    effective_q = _quantiles(effective_counts, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    meta = build_metadata(
        vocab=vocab, bucket_count=bucket_count, alpha=alpha, r_ref=r_ref,
        k_min=k_min, k_max=k_max, f_train=f_train, f_val=f_val,
        dataset_seed=dataset_seed, doc_len=doc_len, sep_token=sep_token,
        exact_counts=exact_counts,
        splits={
            key: {
                "n_train_target": round(
                    count * f_train * factors[key]
                )
            }
            for key, count in exact_counts.items()
        },
        train_tokens_len=train_token_count, val_tokens_len=val_token_count,
        max_tokens=max_tokens, order=order,
    )
    meta.update({
        "exact_effective_train_frequency_quantiles": {
            f"q{int(q * 100)}": value
            for q, value in zip(
                [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
                effective_q,
            )
        },
        "val_source": val_source,
        "emit_format": emit_format,
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    # metadata.json: companion to fivegram_counts.npz, expected by
    # GlobalNgramFrequencyIndex.__init__.
    (out_dir / "metadata.json").write_text(json.dumps({
        "vocab_size": vocab,
        "order": order,
        "bucket_count": bucket_count,
        "n_contexts": int(sum(exact_counts.values())),
        "n_distinct_contexts": int(len(exact_counts)),
        "frequency_definition": "exact_train_epoch_context_count",
        "frequency_source_split": "train",
        "frequency_key_type": "exact_context",
        "frequency_index_scope": (
            "complete upstream train epoch before controlled block resampling"
        ),
        "hash_bucket_occupancy_diagnostic": False,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in meta.items()
                      if k not in ("loader_selection",)}, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="exponent for (r_ref/r)^alpha; 0 => no resampling")
    ap.add_argument("--bucket-count", type=int, default=5_000_000,
                    help="hash bucket count (default 5M)")
    ap.add_argument("--order", type=int, default=5,
                    help="n-gram context length (3=trigram, 5=5-gram)")
    ap.add_argument("--f-train", type=float, default=0.8)
    ap.add_argument("--f-val", type=float, default=0.2)
    ap.add_argument("--k-min", type=float, default=0.25)
    ap.add_argument("--k-max", type=float, default=8.0)
    ap.add_argument("--r-ref-mode", choices=["median", "mean", "fixed"],
                    default="median")
    ap.add_argument("--r-ref-fixed", type=float, default=None)
    ap.add_argument("--dataset-seed", type=int, default=20260805)
    ap.add_argument("--doc-len", type=int, default=2048,
                    help="packed row length (default 2048 = upstream MAX_SEQ_LEN)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap scanned tokens (pilot); default = full shard 1")
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--val-source", choices=["train", "test"], default="train",
                    help="val source: 'train' = coincidental-gap val from the "
                         "train split (legacy); 'test' = held-out unseen val "
                         "from the test split")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="Poisson emission fraction for --val-source test "
                         "(default 0.02); ignored for train-source val")
    ap.add_argument("--emit-format", choices=["txt", "bin", "both"],
                    default="txt",
                    help="'txt' legacy space-separated stream (default); 'bin' "
                         "flat uint16 for the numpy-mmap loader on huge corpora; "
                         "'both'")
    ap.add_argument("--token-cache", default=None,
                    help="dir for per-shard raw-token cache (avoids re-tokenizing "
                         "on every pass; required for the full multi-shard corpus)")
    ap.add_argument("--fast-scan", action="store_true",
                    help="use vectorized per-shard numpy scan (requires --token-cache, max_tokens=None)")
    ap.add_argument("--fast-emit", action="store_true",
                    help="use vectorized numpy emit (requires alpha=0, --emit-format bin, max_tokens=None)")
    args = ap.parse_args()

    if not 0.0 <= args.f_train <= 1.0 or not 0.0 <= args.f_val <= 1.0:
        ap.error("f_train and f_val must be in [0,1]")
    # Note: f_train + f_val may exceed 1 when alpha > 0 because up-sampling
    # multiplies the train draw by k(b) > 1.  The val draw uses the natural
    # f_val fraction only, so the two streams are independent Poisson draws
    # over the same corpus (the coincidental-gap source).
    if args.k_min <= 0 or args.k_max < args.k_min:
        ap.error("k_min must be > 0 and k_max >= k_min")

    generate(Path(args.out_dir),
             alpha=args.alpha, bucket_count=args.bucket_count,
             f_train=args.f_train, f_val=args.f_val,
             k_min=args.k_min, k_max=args.k_max,
             r_ref_mode=args.r_ref_mode, r_ref_fixed=args.r_ref_fixed,
             dataset_seed=args.dataset_seed, doc_len=args.doc_len,
             max_tokens=args.max_tokens, tokenizer_dir=args.tokenizer_dir,
             order=args.order, val_source=args.val_source,
             val_frac=args.val_frac, emit_format=args.emit_format,
             token_cache_dir=args.token_cache,
             use_fast_scan=args.fast_scan, use_fast_emit=args.fast_emit)


if __name__ == "__main__":
    main()
