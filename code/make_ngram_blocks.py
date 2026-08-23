#!/usr/bin/env python3
"""Minimal n-gram block dataset generator (no Poisson, no resampling).

Builds a plain sliding-window block stream from pre-tokenized shard .bin
files (produced by code/prepare_data.py):

    block = [c0, c1, ..., c_{order-1}, next, SEP]   (len = order + 2)

Every real occurrence is emitted exactly once (no Poisson, no alpha, no
frequency control).  This is the "极简" data path the standard experiment
lives on, in contrast to ngram5_freq_gap/data_gen.py which resamples.

Output (compatible with the ngram5_blocks loader + trainer probes):
  <out_dir>/train_tokens.bin   flat uint16 token stream (block-aligned)
  <out_dir>/val_tokens.bin     flat uint16 token stream (block-aligned)
  <out_dir>/meta.json          loader contract (vocab/order/block_len/sep)
  <out_dir>/metadata.json      companion for ExactNgramIndex
  <out_dir>/exact_ngram_counts.npz  packed train context keys + counts
  <out_dir>/fivegram_counts.npz     alias for backward compat

Usage:
  python code/make_ngram_blocks.py \
    --data-dir data/tokenized \
    --train-shards 1 \
    --val-shards 2,3,4,5,6,7,8,9,10,6542 \
    --out-dir data/ngram5_minimal_order5 \
    --order 5
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def _load_shard(data_dir: Path, shard_id: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a shard as (flat_tokens, offsets|None).

    Supports two on-disk layouts:
      - ``shard_<id>.npy`` + ``shard_<id>.offsets.npy`` (cluster token cache,
        flat uint16 stream + uint64 per-document offsets)
      - ``shard_<id>.bin`` (code/prepare_data.py packed rows, flat uint16)
    Returns ``offsets=None`` when only the flat stream exists (no document
    boundaries -> the whole shard is treated as one document).
    """
    npy_path = data_dir / f"shard_{shard_id:05d}.npy"
    off_path = data_dir / f"shard_{shard_id:05d}.offsets.npy"
    if npy_path.exists():
        flat = np.fromfile(npy_path, dtype=np.uint16)
        offsets = None
        if off_path.exists():
            offsets = np.fromfile(off_path, dtype=np.uint64)
        return flat, offsets
    bin_path = data_dir / f"shard_{shard_id:05d}.bin"
    if bin_path.exists():
        return np.memmap(bin_path, dtype=np.uint16, mode="r"), None
    raise FileNotFoundError(f"Missing shard: {bin_path} (or {npy_path})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="dir with shard_<id>.bin files (code/prepare_data.py output)")
    ap.add_argument("--train-shards", required=True, help="comma-separated shard ids")
    ap.add_argument("--val-shards", required=True, help="comma-separated shard ids")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--order", type=int, default=5,
                    help="n-gram context length (5 = five context tokens)")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--sep-token", type=int, default=8191,
                    help="separator token id (default vocab-1)")
    ap.add_argument("--doc-len", type=int, default=2048,
                    help="packed row length (2048 = upstream MAX_SEQ_LEN)")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="limit to the first N documents per shard (0 = all; pilot)")
    ap.add_argument("--max-val-docs", type=int, default=0,
                    help="limit val emission to first N documents (0 = all)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sep_token = args.sep_token
    vocab = args.vocab
    order = args.order
    doc_len = args.doc_len
    block_len = order + 2
    row_stride = (doc_len + 1) // block_len * block_len

    train_ids = [int(x) for x in args.train_shards.split(",") if x.strip()]
    val_ids = [int(x) for x in args.val_shards.split(",") if x.strip()]

    def emit(ids: list[int], out_name: str,
             accumulate_counts: bool) -> tuple[int, int, Counter[int] | None]:
        chunks: list[np.ndarray] = []
        written = 0
        blocks = 0
        counts: Counter[int] = Counter() if accumulate_counts else None
        out_path = out_dir / out_name
        max_docs = args.max_docs if accumulate_counts else args.max_val_docs
        with out_path.open("wb") as fh:
            for sid in ids:
                flat, offsets = _load_shard(data_dir, sid)
                # Emit within each document only (no cross-document contexts),
                # matching data_gen's per-document sliding windows.
                if offsets is not None:
                    doc_range = range(len(offsets) - 1)
                    if max_docs > 0:
                        doc_range = range(min(max_docs, len(offsets) - 1))
                    for d in doc_range:
                        lo = int(offsets[d]); hi = int(offsets[d + 1])
                        if hi - lo <= order:
                            continue
                        for start in range(lo, hi - order):
                            ctx = flat[start:start + order]
                            nxt = int(flat[start + order])
                            block = np.empty(block_len, dtype=np.uint16)
                            block[:order] = ctx
                            block[order] = nxt
                            block[order + 1] = sep_token
                            chunks.append(block)
                            blocks += 1
                            if counts is not None:
                                counts[tuple(int(t) for t in ctx)] += 1
                            if len(chunks) >= 16384:
                                arr = np.concatenate(chunks)
                                arr.tofile(fh)
                                written += arr.size
                                chunks.clear()
                else:
                    # flat stream with no document boundaries: one document
                    lo = 0; hi = len(flat)
                    if hi - lo > order:
                        for start in range(lo, hi - order):
                            ctx = flat[start:start + order]
                            nxt = int(flat[start + order])
                            block = np.empty(block_len, dtype=np.uint16)
                            block[:order] = ctx
                            block[order] = nxt
                            block[order + 1] = sep_token
                            chunks.append(block)
                            blocks += 1
                            if counts is not None:
                                counts[tuple(int(t) for t in ctx)] += 1
                            if len(chunks) >= 16384:
                                arr = np.concatenate(chunks)
                                arr.tofile(fh)
                                written += arr.size
                                chunks.clear()
            # flush tail
            if chunks:
                arr = np.concatenate(chunks)
                arr.tofile(fh)
                written += arr.size
                chunks.clear()
            # pad to row_stride boundary with SEP
            remainder = written % row_stride
            if remainder:
                pad = np.full(row_stride - remainder, sep_token, dtype=np.uint16)
                pad.tofile(fh)
                written += pad.size
        return blocks, written, counts

    train_blocks, train_tokens, exact_counts = emit(
        train_ids, "train_tokens.bin", accumulate_counts=True)
    val_blocks, val_tokens, _ = emit(
        val_ids, "val_tokens.bin", accumulate_counts=False)

    print(f"[make_ngram_blocks] order={order} block_len={block_len} "
          f"train_blocks={train_blocks:,} train_tokens={train_tokens:,} "
          f"val_blocks={val_blocks:,} val_tokens={val_tokens:,}")

    # ---- meta.json (loader contract) ----
    meta = {
        "schema_version": 1,
        "order": order,
        "context_len": order,
        "block_len": block_len,
        "vocab": vocab,
        "sep_token": sep_token,
        "doc_len": doc_len,
        "loader_row_stride": row_stride,
        "loader_rows_train": train_tokens // row_stride,
        "loader_rows_val": val_tokens // row_stride,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "n_nonempty_buckets": len(exact_counts),
        "stream_padding": "global_block_aligned_row_stride",
        "block_alignment_version": 2,
        "frequency_definition": "exact_train_epoch_context_count",
        "frequency_source_split": "train",
        "frequency_key_type": "exact_context",
        "hash_bucket_occupancy_diagnostic": False,
        "resampling": "none",
        "alpha": 0.0,
        "source": "code/make_ngram_blocks.py (sliding window, one copy per event)",
        "train_shards": train_ids,
        "val_shards": val_ids,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # ---- metadata.json (companion for ExactNgramIndex) ----
    nonempty = sorted(exact_counts.values())
    (out_dir / "metadata.json").write_text(json.dumps({
        "vocab_size": vocab,
        "order": order,
        "bucket_count": 0,
        "n_contexts": int(sum(exact_counts.values())),
        "n_distinct_contexts": int(len(exact_counts)),
        "frequency_definition": "exact_train_epoch_context_count",
        "frequency_source_split": "train",
        "frequency_key_type": "exact_context",
        "frequency_index_scope": (
            "complete upstream train epoch before controlled block resampling"
        ),
        "hash_bucket_occupancy_diagnostic": False,
        "min_count": min(nonempty) if nonempty else 0,
        "max_count": max(nonempty) if nonempty else 0,
        "total_contexts": int(sum(exact_counts.values())),
    }, indent=2, sort_keys=True) + "\n")

    # ---- exact_ngram_counts.npz (unpacked context tuples + counts) ----
    # Store the full context token tuples (shape (n, order)) so the trainer's
    # ExactNgramIndex can look up exact frequencies without packing into an
    # int64 key (which overflows for order=5, vocab=8192: 8192^5 > int64 max).
    # Avoid a Python-level sort of millions of tuple keys: build the arrays
    # directly from the Counter (dict iteration order is fine; ExactNgramIndex
    # uses a dict lookup, not sorted keys).
    n_ctx = len(exact_counts)
    ctx_arr = np.empty((n_ctx, order), dtype=np.int32)
    counts_arr = np.empty(n_ctx, dtype=np.int64)
    for i, (ctx, c) in enumerate(exact_counts.items()):
        ctx_arr[i] = ctx
        counts_arr[i] = c
    np.savez(out_dir / "exact_ngram_counts.npz",
             contexts=ctx_arr, counts=counts_arr)
    np.savez(out_dir / "fivegram_counts.npz",
             contexts=ctx_arr, counts=counts_arr)
    print(f"[make_ngram_blocks] wrote {n_ctx:,} distinct context keys")
    print(f"[make_ngram_blocks] context matrix shape {ctx_arr.shape} (order={order}, vocab={vocab})")

    # ---- contexts.json (small convenience, only if few) ----
    if n_ctx <= 1_000_000:
        (out_dir / "contexts.json").write_text(
            json.dumps({
                ",".join(str(t) for t in ctx): {
                    "r": int(c),
                    "frequency_definition": "exact_train_epoch_context_count",
                }
                for ctx, c in exact_counts.items()
            }, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
