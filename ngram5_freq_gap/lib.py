# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
"""Runtime utilities: tokenizer wrapper, dataloader, and BPB evaluation."""

import os
import json
import math
import pickle
import random
import copy

import torch

# pyarrow is only needed by the parquet-based loaders (fixed/epoch_shuffle/...).
# The ngram5_blocks loader does not use it, so we defer the import to keep
# lib importable in environments without pyarrow (e.g. local CPU smoke tests).
pq = None
def _ensure_pyarrow():
    global pq
    if pq is None:
        import pyarrow.parquet as _pq
        pq = _pq
    return pq

from gap_experiment import (
    batch_epoch_reshuffle_indices,
    batch_fixed_shuffle_indices,
    dataset_equivalents,
    epoch_indices,
    epoch_reshuffle_indices,
    interleaved_replay_offsets,
    ordered_replay_offsets,
    overlap_row_mapping,
    replay_order_annotations,
    replacement_indices,
    shuffle_buffer_stream,
)

MAX_SEQ_LEN = 2048
TIME_BUDGET = 300
EVAL_TOKENS = 40 * 524288

# H200 adapter: Recursive's upstream baseline expects /data, while our prepared
# Karpathy shards/tokenizer live under ~/.cache/autoresearch when /data is
# absent OR when AUTORESEARCH_CACHE_DIR is set (cluster convention).
_CACHE_DIR = os.environ.get(
    "AUTORESEARCH_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "autoresearch"),
)
_USE_UPSTREAM_DATA_DIR = os.path.isdir("/data") and not os.environ.get("AUTORESEARCH_CACHE_DIR", "").strip()
DATA_DIR = "/data" if _USE_UPSTREAM_DATA_DIR else os.path.join(_CACHE_DIR, "data")
DEFAULT_TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizer") if _USE_UPSTREAM_DATA_DIR else os.path.join(_CACHE_DIR, "tokenizer")
FIXED_TOKENIZER_DIR = os.environ.get("FIXED_TOKENIZER_DIR", "").strip()
TOKENIZER_DIR = FIXED_TOKENIZER_DIR or DEFAULT_TOKENIZER_DIR
# data_split.json lives in the repo root (parent of ngram5_freq_gap/ on the
# cluster), not in this directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(PROJECT_ROOT) if os.path.basename(PROJECT_ROOT) == "ngram5_freq_gap" else PROJECT_ROOT
DATA_SPLIT_PATH = os.path.join(_REPO_ROOT, "data_split.json")
if not os.path.exists(DATA_SPLIT_PATH):
    # fallback: cluster convention, data_split.json next to train.py
    DATA_SPLIT_PATH = "/data3/guoshaoyang/ngram-gap-exp/data_split.json"
BOS_TOKEN = "<|reserved_0|>"


def _shard_filename(index):
    return f"shard_{index:05d}.parquet"


def _normalize_shard_ids(values, split_name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"data_split.json must define non-empty list '{split_name}'")
    seen = set()
    ids = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid shard id in '{split_name}': {value!r}")
        if value in seen:
            raise ValueError(f"Duplicate shard id in '{split_name}': {value}")
        seen.add(value)
        ids.append(value)
    return ids


def load_data_split():
    with open(DATA_SPLIT_PATH, "r", encoding="utf-8") as f:
        split = json.load(f)
    train_ids = _normalize_shard_ids(split.get("train"), "train")
    test_ids = _normalize_shard_ids(split.get("test"), "test")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"Shard ids cannot appear in both train and test: {overlap}")
    return {"train": train_ids, "test": test_ids}


def _canonical_split(split):
    if split == "train":
        return "train"
    if split in ("val", "test"):
        return "test"
    raise ValueError(f"Unknown split: {split!r}")


def split_parquet_files(split):
    split_ids = load_data_split()[_canonical_split(split)]
    return [os.path.join(DATA_DIR, _shard_filename(index)) for index in split_ids]


def describe_shard_selection():
    split = load_data_split()
    return {
        "data_split_path": DATA_SPLIT_PATH,
        "data_dir": DATA_DIR,
        "tokenizer_dir": TOKENIZER_DIR,
        "fixed_tokenizer_dir": FIXED_TOKENIZER_DIR,
        "train_ids": split["train"],
        "test_ids": split["test"],
        "train_parquet_files": split_parquet_files("train"),
        "test_parquet_files": split_parquet_files("test"),
    }


def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


TRAIN_DATA_MODES = {
    "fixed",
    "epoch_shuffle",
    "epoch_reshuffle",
    "replacement",
    "shuffle_buffer",
    "interleaved_replay",
    "batch_fixed_shuffle",
    "batch_epoch_reshuffle",
    "ngram5_blocks",
}
BATCH_ORDER_DATA_MODES = {"batch_fixed_shuffle", "batch_epoch_reshuffle"}


def _load_documents(split):
    """Materialize a split for exact global shuffling or uniform replacement."""
    documents = []
    _pq = _ensure_pyarrow()
    for filepath in split_parquet_files(split):
        pf = _pq.ParquetFile(filepath)
        parquet_row = 0
        for rg_idx in range(pf.num_row_groups):
            batch = pf.read_row_group(rg_idx).column("text").to_pylist()
            documents.extend(
                (
                    text,
                    {"source_path": filepath, "parquet_row": parquet_row + local_index},
                )
                for local_index, text in enumerate(batch)
            )
            parquet_row += len(batch)
    if not documents:
        raise ValueError(f"No documents found for split {split!r}")
    return documents


def _document_batches(
    split,
    tokenizer_batch_size=128,
    with_metadata=False,
    data_mode="fixed",
    data_seed=42,
    shuffle_buffer_size=8192,
):
    parquet_paths = split_parquet_files(split)
    missing = [p for p in parquet_paths if not os.path.exists(p)]
    assert not missing, f"Missing {split} shards. Run prepare.py first: {missing}"
    if data_mode not in TRAIN_DATA_MODES:
        raise ValueError(f"Unknown data mode {data_mode!r}; expected one of {sorted(TRAIN_DATA_MODES)}")
    epoch = 1

    if data_mode in {"epoch_shuffle", "epoch_reshuffle", "replacement", "shuffle_buffer"}:
        documents = _load_documents(split)
        num_documents = len(documents)
        emitted_documents = 0
        replacement_rng = random.Random(data_seed)
        shuffle_stream = (
            shuffle_buffer_stream(num_documents, shuffle_buffer_size, random.Random(data_seed))
            if data_mode == "shuffle_buffer"
            else None
        )
        while True:
            if data_mode in {"epoch_shuffle", "epoch_reshuffle"}:
                # epoch_reshuffle keeps epoch 1 identical to the fixed-order arm,
                # then changes only the repeated passes for a clean intervention.
                indices = (
                    epoch_reshuffle_indices(num_documents, data_seed, epoch)
                    if data_mode == "epoch_reshuffle"
                    else epoch_indices(num_documents, data_seed, epoch)
                )
            else:
                indices = (
                    [next(shuffle_stream) for _ in range(tokenizer_batch_size)]
                    if data_mode == "shuffle_buffer"
                    else replacement_indices(num_documents, tokenizer_batch_size, replacement_rng)
                )

            for start in range(0, len(indices), tokenizer_batch_size):
                index_batch = indices[start:start + tokenizer_batch_size]
                output_batch = [documents[index] for index in index_batch]
                texts = [text for text, _metadata in output_batch]
                metadata = [
                    {
                        **item_metadata,
                        "sampling_mode": data_mode,
                        "sample_epoch": 0 if data_mode in {"replacement", "shuffle_buffer"} else epoch,
                        "shuffle_buffer_size": shuffle_buffer_size if data_mode == "shuffle_buffer" else 0,
                        "documents_sampled_before": emitted_documents + local_index,
                        "dataset_equivalents": dataset_equivalents(
                            emitted_documents + local_index, num_documents
                        ),
                    }
                    for local_index, (_text, item_metadata) in enumerate(output_batch)
                ]
                output_epoch = 0 if data_mode in {"replacement", "shuffle_buffer"} else epoch
                if with_metadata:
                    yield texts, metadata, output_epoch
                else:
                    yield texts, output_epoch
                emitted_documents += len(index_batch)

            if data_mode in {"epoch_shuffle", "epoch_reshuffle"}:
                epoch += 1
            continue

    while True:
        _pq = _ensure_pyarrow()
        for filepath in parquet_paths:
            pf = _pq.ParquetFile(filepath)
            parquet_row = 0
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    texts = batch[i:i+tokenizer_batch_size]
                    metadata = [
                        {
                            "source_path": filepath,
                            "parquet_row": parquet_row + i + local_index,
                            "sampling_mode": data_mode,
                            "sample_epoch": epoch,
                        }
                        for local_index in range(len(texts))
                    ]
                    if with_metadata:
                        yield texts, metadata, epoch
                    else:
                        yield texts, epoch
                parquet_row += len(batch)
        epoch += 1


def _clone_packed_batch(batch, return_metadata):
    if return_metadata:
        inputs, targets, epoch, metadata = batch
        return inputs.detach().clone(), targets.detach().clone(), epoch, copy.deepcopy(metadata)
    inputs, targets, epoch = batch
    return inputs.detach().clone(), targets.detach().clone(), epoch


# ---------------------------------------------------------------------------
# ngram5_blocks data mode: read pre-tokenised 5-gram block streams
# ---------------------------------------------------------------------------

# Block layout produced by ngram5_freq_gap/data_gen.py:
#   [c0, c1, c2, c3, c4, next, SEP]  (7 tokens)
# The loader packs these into T+1 rows (inputs = row[:-1], targets = row[1:])
# and, when return_metadata=True, annotates the target position of each block
# with the bucket id so the frequency-gap decomposition can key on it.
# Block layout and target offset are read from the dataset's meta.json
# (order-dependent: block_len = order + 2, target_offset = order).

_NGRAM5_BLOCK_LEN = 7
_NGRAM5_TARGET_OFFSET = 5


def _ngram5_block_dataloader(B, T, split, *, return_metadata=False, data_seed=42):
    """Yield packed batches from a pre-tokenised n-gram block stream.

    Reads ``$NGRAM5_DATA_DIR/{train,val}_tokens.txt`` (space-separated ints)
    and ``$NGRAM5_DATA_DIR/meta.json`` (for vocab + sep_token + block_len +
    order).  Each row of the batch is a contiguous slice of length T+1,
    block-aligned (rows start at multiples of block_len so blocks never
    straddle a row).

    Yields ``(inputs, targets, epoch)`` or, when return_metadata=True,
    ``(inputs, targets, epoch, metadata)`` where metadata is a list (len B)
    of lists of per-segment dicts compatible with the upstream packing
    metadata schema, plus a ``bucket`` field on target segments.
    """
    import os as _os
    data_dir = _os.environ.get("NGRAM5_DATA_DIR", "").strip()
    if not data_dir:
        raise RuntimeError(
            "TRAIN_DATA_MODE=ngram5_blocks requires NGRAM5_DATA_DIR to point "
            "at the directory produced by ngram5_freq_gap/data_gen.py"
        )
    data_dir = _os.path.abspath(data_dir)
    split_name = "train" if split in ("train",) else "val"
    tokens_path = _os.path.join(data_dir, f"{split_name}_tokens.txt")
    meta_path = _os.path.join(data_dir, "meta.json")
    bin_path = _os.path.join(data_dir, f"{split_name}_tokens.bin")
    if not _os.path.exists(tokens_path) and not _os.path.exists(bin_path):
        raise FileNotFoundError(
            f"ngram5_blocks: missing {tokens_path} (or {bin_path})"
        )
    if not _os.path.exists(meta_path):
        raise FileNotFoundError(f"ngram5_blocks: missing {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    sep_token = int(meta["sep_token"])
    block_len = int(meta["block_len"])  # order + 2
    order = int(meta["order"])
    target_offset = order  # index of `next` within a block (0-based)

    # Load token stream once.  Prefer the flat uint16 ``.bin`` written by
    # data_gen (--emit-format bin) for huge corpora: a numpy memmap keeps
    # resident memory to the pages actually touched, and row access is
    # O(1).  Fall back to the legacy space-separated ``.txt`` stream.
    tokens_is_array = False
    if _os.path.exists(bin_path):
        import numpy as _np
        tokens = _np.memmap(bin_path, dtype=_np.uint16, mode="r")
        tokens_is_array = True
    else:
        with open(tokens_path, "r", encoding="utf-8") as fh:
            tokens = [int(x) for x in fh.read().split()]
    n_tokens = len(tokens)
    if n_tokens < T + 1:
        raise ValueError(
            f"ngram5_blocks: token stream too short ({n_tokens} < {T + 1})"
        )

    row_capacity = T + 1
    blocks_per_row = row_capacity // block_len
    if blocks_per_row == 0:
        raise ValueError(f"seq_len {T} too small for block_len {block_len}")
    # Row starts must be block-aligned so no block straddles a row boundary.
    # We advance by row_capacity tokens per row; because the stream is itself
    # block-aligned (blocks are contiguous + SEP-padded to doc_len), row
    # boundaries at multiples of row_capacity are block-aligned when
    # row_capacity is a multiple of block_len.  When it is not, we still start
    # rows at multiples of block_len (advancing by row_capacity rounded down
    # to a block boundary) and pad the tail.
    row_stride = (row_capacity // block_len) * block_len
    n_rows = n_tokens // row_stride
    if n_rows == 0:
        raise ValueError(
            f"ngram5_blocks: not enough tokens for one row ({n_tokens} < {row_stride})"
        )

    rank = int(_os.environ.get("NGRAM5_RANK", "0"))
    world_size = int(_os.environ.get("NGRAM5_WORLD_SIZE", "1"))
    # Per-rank rng: ranks see disjoint row subsets with different shuffles.
    # NGRAM5_BLOCK_SHUFFLE=0 disables the per-epoch row shuffle entirely
    # (deterministic fixed-order replay: rows visited in the same order every
    # epoch).  Default 1 preserves the existing per-epoch shuffle behaviour.
    shuffle_rows = _os.environ.get("NGRAM5_BLOCK_SHUFFLE", "1") != "0"
    rng = random.Random(data_seed + rank)
    epoch = 1
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    while True:
        if shuffle_rows:
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
        else:
            row_order = list(range(n_rows))
        if world_size > 1:
            row_order = row_order[rank::world_size]
        for batch_start in range(0, len(row_order), B):
            batch_rows = row_order[batch_start:batch_start + B]
            if len(batch_rows) < B:
                # wrap around to fill the last partial batch
                extra = rng.sample(row_order, B - len(batch_rows))
                batch_rows = batch_rows + extra
            metadata = [[] for _ in range(B)] if return_metadata else None
            for i, row_idx in enumerate(batch_rows):
                start = row_idx * row_stride
                end = start + row_stride
                if tokens_is_array:
                    row = tokens[start:end].tolist()
                else:
                    row = tokens[start:end]
                # pad tail with SEP if the stride didn't fill row_capacity
                if len(row) < row_capacity:
                    row = row + [sep_token] * (row_capacity - len(row))
                row_buffer[i] = torch.tensor(row, dtype=torch.long)
                if return_metadata:
                    for j in range(blocks_per_row):
                        block_start_in_row = j * block_len
                        tgt_pos_in_targets = block_start_in_row + target_offset - 1
                        if tgt_pos_in_targets < T:
                            # Keep the raw context; the trainer hashes it.
                            ctx_n = row[block_start_in_row:block_start_in_row + order]
                            bucket = 0
                            # NOTE: this decodes the *bucket id* from the
                            # context tokens using hash5 % bucket_count, but
                            # we do not have hash5 here without importing
                            # data_gen.  Instead we record the raw context
                            # tokens; the frequency decomposition will hash
                            # them at query time.  We store bucket=None and
                            # let the trainer compute it.
                            segment = {
                                "row_start": block_start_in_row,
                                "row_end": block_start_in_row + block_len,
                                "bucket": None,
                                "context": list(ctx_n),
                                "next": row[block_start_in_row + target_offset],
                                "is_target_position": True,
                                "target_position_in_targets": tgt_pos_in_targets,
                                "source_path": tokens_path,
                                "sample_epoch": epoch,
                                "sampling_mode": "ngram5_blocks",
                            }
                            if order == 5:
                                segment["context5"] = list(ctx_n)
                            metadata[i].append(segment)
            inputs = row_buffer[:, :-1].clone()
            targets = row_buffer[:, 1:].clone()
            if return_metadata:
                yield inputs, targets, epoch, metadata
            else:
                yield inputs, targets, epoch
        epoch += 1


def _annotate_batch_order_metadata(metadata, data_mode, epoch, source_epoch, source_batch_index, schedule_index):
    annotated = copy.deepcopy(metadata)
    for row_segments in annotated:
        for segment in row_segments:
            segment.update(
                {
                    "sampling_mode": data_mode,
                    "sample_epoch": epoch,
                    "source_epoch": source_epoch,
                    "source_batch_index": source_batch_index,
                    "scheduled_batch_index": schedule_index,
                }
            )
    return annotated


def _batch_order_dataloader(
    tokenizer,
    B,
    T,
    split,
    *,
    buffer_size,
    return_metadata,
    data_mode,
    data_seed,
    batch_overlap,
    replay_new_steps,
    replay_steps,
    shuffle_buffer_size,
):
    """Shuffle already-packed batches while preserving their contents exactly."""
    base_loader = make_dataloader(
        tokenizer,
        B,
        T,
        split,
        buffer_size=buffer_size,
        return_metadata=return_metadata,
        data_mode="fixed",
        data_seed=data_seed,
        batch_overlap=batch_overlap,
        replay_new_steps=replay_new_steps,
        replay_steps=replay_steps,
        shuffle_buffer_size=shuffle_buffer_size,
    )

    source_epoch = None
    packed_batches = []
    while True:
        batch = next(base_loader)
        batch_epoch = batch[2]
        if source_epoch is None:
            source_epoch = batch_epoch
        elif batch_epoch != source_epoch:
            break
        packed_batches.append(_clone_packed_batch(batch, return_metadata))

    if not packed_batches:
        raise ValueError("No packed batches were available for batch-order shuffling")

    num_batches = len(packed_batches)
    epoch = 1
    while True:
        if data_mode == "batch_fixed_shuffle":
            order = batch_fixed_shuffle_indices(num_batches, data_seed)
        elif data_mode == "batch_epoch_reshuffle":
            order = batch_epoch_reshuffle_indices(num_batches, data_seed, epoch)
        else:
            raise ValueError(f"Unsupported batch-order data mode {data_mode!r}")

        for schedule_index, source_batch_index in enumerate(order):
            cached = packed_batches[source_batch_index]
            if return_metadata:
                inputs, targets, cached_source_epoch, metadata = cached
                yield (
                    inputs,
                    targets,
                    epoch,
                    _annotate_batch_order_metadata(
                        metadata,
                        data_mode,
                        epoch,
                        cached_source_epoch,
                        source_batch_index,
                        schedule_index,
                    ),
                )
            else:
                inputs, targets, _cached_source_epoch = cached
                yield inputs, targets, epoch
        epoch += 1


def make_dataloader(
    tokenizer,
    B,
    T,
    split,
    buffer_size=1000,
    return_metadata=False,
    data_mode="fixed",
    data_seed=42,
    batch_overlap=0.0,
    replay_new_steps=50,
    replay_steps=50,
    replay_order="original",
    replay_cyclic_offset=0,
    replay_order_seed=0,
    shuffle_buffer_size=8192,
):
    """BOS-aligned dataloader with best-fit packing. Every row starts with BOS;
    documents are packed best-fit, cropping the shortest doc when nothing fits."""
    assert split in ["train", "val", "test"]
    if not 0.0 <= batch_overlap <= 1.0:
        raise ValueError("batch_overlap must be in [0, 1]")
    if data_mode in BATCH_ORDER_DATA_MODES:
        yield from _batch_order_dataloader(
            tokenizer,
            B,
            T,
            split,
            buffer_size=buffer_size,
            return_metadata=return_metadata,
            data_mode=data_mode,
            data_seed=data_seed,
            batch_overlap=batch_overlap,
            replay_new_steps=replay_new_steps,
            replay_steps=replay_steps,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        return
    if data_mode == "ngram5_blocks":
        yield from _ngram5_block_dataloader(
            B,
            T,
            split,
            return_metadata=return_metadata,
            data_seed=data_seed,
        )
        return
    if data_mode == "interleaved_replay":
        interleaved_replay_offsets(replay_new_steps, replay_steps)
        base_loader = make_dataloader(
            tokenizer,
            B,
            T,
            split,
            buffer_size=buffer_size,
            return_metadata=return_metadata,
            data_mode="fixed",
            data_seed=data_seed,
            batch_overlap=batch_overlap,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        cycle = 1
        pending_batch = None
        while True:
            cached_batches = []
            source_epoch = None
            for block_offset in range(replay_new_steps):
                batch = pending_batch if pending_batch is not None else next(base_loader)
                pending_batch = None
                batch_epoch = batch[2]
                if source_epoch is None:
                    source_epoch = batch_epoch
                elif batch_epoch != source_epoch:
                    pending_batch = batch
                    break
                if return_metadata:
                    inputs, targets, epoch, metadata = batch
                    new_metadata = copy.deepcopy(metadata)
                    for segments in new_metadata:
                        for segment in segments:
                            segment.update(
                                {
                                    "sampling_mode": data_mode,
                                    "interleaved_phase": "new",
                                    "interleaved_cycle": cycle,
                                    "interleaved_block_offset": block_offset,
                                }
                            )
                    cached_batches.append(
                        (inputs.detach().clone(), targets.detach().clone(), epoch, copy.deepcopy(new_metadata))
                    )
                    yield inputs, targets, epoch, new_metadata
                else:
                    inputs, targets, epoch = batch
                    cached_batches.append((inputs.detach().clone(), targets.detach().clone(), epoch))
                    yield inputs, targets, epoch

            replay_order_manifest = ordered_replay_offsets(
                len(cached_batches),
                replay_steps,
                order=replay_order,
                cyclic_offset=replay_cyclic_offset,
                seed=replay_order_seed,
            )
            replay_offsets = replay_order_manifest["source_offsets"]
            for replay_offset, source_offset in enumerate(replay_offsets):
                cached = cached_batches[source_offset]
                order_annotations = replay_order_annotations(
                    replay_order_manifest,
                    replay_offset,
                    source_offset,
                )
                if return_metadata:
                    inputs, targets, epoch, metadata = cached
                    replay_metadata = copy.deepcopy(metadata)
                    for segments in replay_metadata:
                        for segment in segments:
                            segment.update(
                                {
                                    "sampling_mode": data_mode,
                                    "interleaved_phase": "replay",
                                    "interleaved_cycle": cycle,
                                    **order_annotations,
                                }
                            )
                    yield inputs, targets, epoch, replay_metadata
                else:
                    inputs, targets, epoch = cached
                    yield inputs, targets, epoch
            cycle += 1
        return

    row_capacity = T + 1
    batches = _document_batches(
        split,
        with_metadata=return_metadata,
        data_mode=data_mode,
        data_seed=data_seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1
    overlap_rng = random.Random(data_seed + 10_000_019)
    previous_row_buffer = None
    previous_batch_metadata = None

    def refill_buffer():
        nonlocal epoch
        if return_metadata:
            doc_batch, metadata_batch, epoch = next(batches)
        else:
            doc_batch, epoch = next(batches)
            metadata_batch = [None] * len(doc_batch)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(
            {"tokens": tokens, "metadata": metadata}
            for tokens, metadata in zip(token_lists, metadata_batch)
        )

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        batch_metadata = [[] for _ in range(B)] if return_metadata else None
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, entry in enumerate(doc_buffer):
                    doc_len = len(entry["tokens"])
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    entry = doc_buffer.pop(best_idx)
                    doc = entry["tokens"]
                    row_start = pos
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]["tokens"]))
                    entry = doc_buffer.pop(shortest_idx)
                    doc = entry["tokens"]
                    row_start = pos
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
                if return_metadata:
                    used_tokens = pos - row_start
                    batch_metadata[row_idx].append(
                        {
                            **entry["metadata"],
                            "row_start": row_start,
                            "row_end": pos,
                            "document_tokens_used": used_tokens,
                            "document_tokens_total": len(doc),
                            "cropped": used_tokens < len(doc),
                        }
                    )

        replay_targets = set()
        if previous_row_buffer is not None and batch_overlap > 0.0:
            for target_row, source_row in overlap_row_mapping(B, batch_overlap, overlap_rng):
                row_buffer[target_row].copy_(previous_row_buffer[source_row])
                replay_targets.add(target_row)
                if return_metadata:
                    batch_metadata[target_row] = copy.deepcopy(previous_batch_metadata[source_row])

        if return_metadata:
            for row_idx, segments in enumerate(batch_metadata):
                replayed = row_idx in replay_targets
                for segment in segments:
                    segment["replayed_from_previous_batch"] = replayed

        if batch_overlap > 0.0:
            previous_row_buffer = row_buffer.clone()
            if return_metadata:
                previous_batch_metadata = copy.deepcopy(batch_metadata)

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        if return_metadata:
            yield inputs, targets, epoch, batch_metadata
        else:
            yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """Bits per byte: vocab-size-independent metric. Sums per-token
    cross-entropy (nats) and target byte lengths, converts nats/byte to
    bits/byte; special tokens (byte length 0) are excluded."""
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
