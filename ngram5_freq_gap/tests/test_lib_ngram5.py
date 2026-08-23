"""Tests for the ngram5_blocks data mode in lib.py.

These tests verify the block-aligned packing, SEP padding, epoch cycling,
and metadata schema of the _ngram5_block_dataloader, using a synthetic
token stream written to a temp directory (no real BPE tokenizer or parquet
shards required).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
_NGRAM5_ROOT = _HERE.parent  # tests/ -> ngram5_freq_gap/
sys.path.insert(0, str(_NGRAM5_ROOT))

from lib import make_dataloader, _NGRAM5_BLOCK_LEN, _NGRAM5_TARGET_OFFSET


def _write_synthetic_dataset(tmp_path: Path, *, n_blocks=50, doc_len=64,
                              vocab=8192, sep=None):
    """Write a minimal valid 5-gram block dataset to tmp_path."""
    sep = sep if sep is not None else vocab - 1
    # Each block: [c0,c1,c2,c3,c4, next, SEP] = 7 tokens
    blocks = []
    for i in range(n_blocks):
        c = [i % 256, (i * 7) % 256, (i * 13) % 256, (i * 17) % 256, (i * 19) % 256]
        nxt = (i * 23) % 1000
        blocks.extend(c + [nxt, sep])
    # pad to multiple of doc_len with sep
    rem = len(blocks) % doc_len
    if rem:
        blocks.extend([sep] * (doc_len - rem))
    train = " ".join(map(str, blocks))
    val = " ".join(map(str, blocks[:len(blocks) // 2 + doc_len]))
    # ensure val is multiple of doc_len
    val_blocks = [int(x) for x in val.split()]
    rem = len(val_blocks) % doc_len
    if rem:
        val_blocks.extend([sep] * (doc_len - rem))
    val = " ".join(map(str, val_blocks))

    (tmp_path / "train_tokens.txt").write_text(train + "\n")
    (tmp_path / "val_tokens.txt").write_text(val + "\n")
    meta = {
        "schema_version": 1, "order": 5, "context_len": 5,
        "block_len": _NGRAM5_BLOCK_LEN, "vocab": vocab, "sep_token": sep,
        "doc_len": doc_len, "bucket_count": 1000, "alpha": 0.0,
        "r_ref": 1.0, "k_min": 1.0, "k_max": 1.0, "f_train": 0.8, "f_val": 0.2,
        "dataset_seed": 42, "max_tokens_scanned": None,
        "n_nonempty_buckets": 50, "total_contexts": 50,
        "bucket_r_quantiles": {}, "eff_r_train_quantiles": {},
        "train_tokens": len(blocks), "val_tokens": len(val_blocks),
        "train_docs": len(blocks) // doc_len,
        "val_docs": len(val_blocks) // doc_len,
        "loader_selection": {"test": True},
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    # fivegram_counts.npz + metadata.json (needed by FivegramIndex, not by loader)
    import numpy as np
    keys = np.array(sorted(range(1000)), dtype=np.int64)
    counts = np.array([1] * 1000, dtype=np.int64)
    np.savez(tmp_path / "fivegram_counts.npz", keys=keys, counts=counts)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "vocab_size": vocab, "order": 5, "bucket_count": 1000,
        "n_contexts": 1000, "n_distinct_contexts": 50,
    }))
    return tmp_path


def test_ngram5_blocks_loads_batch(tmp_path, monkeypatch):
    _write_synthetic_dataset(tmp_path)
    monkeypatch.setenv("NGRAM5_DATA_DIR", str(tmp_path))
    loader = make_dataloader(None, B=2, T=32, split="train",
                             data_mode="ngram5_blocks", data_seed=42)
    x, y, ep = next(loader)
    assert x.shape == (2, 32)
    assert y.shape == (2, 32)
    assert ep == 1
    # inputs/targets are shifted by 1
    assert torch.equal(x[0, 1:], y[0, :-1])


def test_ngram5_blocks_epoch_cycles(tmp_path, monkeypatch):
    """When we exhaust all rows, epoch should increment."""
    _write_synthetic_dataset(tmp_path, n_blocks=10, doc_len=64)
    monkeypatch.setenv("NGRAM5_DATA_DIR", str(tmp_path))
    loader = make_dataloader(None, B=4, T=64, split="train",
                             data_mode="ngram5_blocks", data_seed=42)
    epochs_seen = set()
    for _ in range(20):
        _, _, ep = next(loader)
        epochs_seen.add(ep)
    # with only ~10 rows / 4 batch, we should cycle through epochs
    assert len(epochs_seen) > 1


def test_ngram5_blocks_block_alignment(tmp_path, monkeypatch):
    """Each row should start at a block boundary — the first token of each
    row should be a context token (c0), not a SEP or a next-token."""
    _write_synthetic_dataset(tmp_path, n_blocks=100, doc_len=64)
    monkeypatch.setenv("NGRAM5_DATA_DIR", str(tmp_path))
    loader = make_dataloader(None, B=1, T=63, split="train",
                             data_mode="ngram5_blocks", data_seed=42)
    x, y, ep = next(loader)
    # row_capacity = 64, block_len = 7, blocks_per_row = 9 (63//7=9, 9*7=63)
    # stride = 63, so each row is 63 tokens = 9 blocks
    # the first token of the row (x[0,0]) should be a context c0 in [0,255]
    assert 0 <= int(x[0, 0]) < 256


def test_ngram5_blocks_metadata_schema(tmp_path, monkeypatch):
    """When return_metadata=True, the loader yields per-block metadata."""
    _write_synthetic_dataset(tmp_path, n_blocks=100, doc_len=64)
    monkeypatch.setenv("NGRAM5_DATA_DIR", str(tmp_path))
    loader = make_dataloader(None, B=2, T=63, split="train",
                             data_mode="ngram5_blocks", data_seed=42,
                             return_metadata=True)
    x, y, ep, meta = next(loader)
    assert len(meta) == 2  # B=2
    for row_meta in meta:
        for seg in row_meta:
            assert "row_start" in seg
            assert "row_end" in seg
            assert "context5" in seg
            assert "next" in seg
            assert "is_target_position" in seg
            assert "sample_epoch" in seg
            assert "sampling_mode" in seg
            assert seg["sampling_mode"] == "ngram5_blocks"


def test_ngram5_blocks_requires_data_dir(monkeypatch):
    """Without NGRAM5_DATA_DIR, the loader should raise."""
    monkeypatch.delenv("NGRAM5_DATA_DIR", raising=False)
    with pytest.raises(RuntimeError, match="NGRAM5_DATA_DIR"):
        loader = make_dataloader(None, B=2, T=32, split="train",
                                 data_mode="ngram5_blocks", data_seed=42)
        next(loader)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
