#!/usr/bin/env python3
"""Build the exact chunk-aligned frequency index for the theory-Zipf run.

The setting is intentionally encoded here rather than passed as command-line
arguments.  The index counts the same 2049-token chunks and first-position
context convention used by code/train.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = (
    ROOT / "tasks" / "l7_theory_zipf" / "results" / "inputs"
    / "theory_zipf_iid_mainline_aligned_20260904"
)
OUT_PATH = DATA_DIR / "freq_index.npz"
VOCAB_SIZE = 8192
SEQUENCE_LEN = 2048
CHUNK_SIZE = SEQUENCE_LEN + 1
TRAIN_SHARD_IDS = [1]


def main() -> None:
    """Build and save one exact frequency index without CLI parameters."""
    if OUT_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing index: {OUT_PATH}")
    shard_path = DATA_DIR / "shard_00001.bin"
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    raw_tokens = shard_path.stat().st_size // np.dtype("<u2").itemsize
    n_chunks = raw_tokens // CHUNK_SIZE
    if raw_tokens % CHUNK_SIZE:
        raise ValueError("train shard is not aligned to sequence_len + 1")
    # This is the same chunk-local context convention as
    # code/ngram_freq.py: context windows never cross a 2049-token chunk.
    tokens = np.memmap(shard_path, dtype="<u2", mode="r").reshape(
        n_chunks, CHUNK_SIZE
    ).astype(np.int64, copy=False)
    cur = tokens[:, :SEQUENCE_LEN]
    prev = np.concatenate([tokens[:, :1], tokens[:, :SEQUENCE_LEN - 1]], axis=1)
    prev2 = np.concatenate([tokens[:, :2], tokens[:, :SEQUENCE_LEN - 2]], axis=1)
    bigram_keys = (prev * VOCAB_SIZE + cur).reshape(-1)
    trigram_keys = (
        prev2 * (VOCAB_SIZE * VOCAB_SIZE) + prev * VOCAB_SIZE + cur
    ).reshape(-1)
    bigram_unique, bigram_counts = np.unique(bigram_keys, return_counts=True)
    trigram_unique, trigram_counts = np.unique(trigram_keys, return_counts=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        bigram_keys=bigram_unique.astype(np.int64, copy=False),
        bigram_counts=bigram_counts.astype(np.int32, copy=False),
        trigram_keys=trigram_unique.astype(np.int64, copy=False),
        trigram_counts=trigram_counts.astype(np.int32, copy=False),
        vocab_size=np.array([VOCAB_SIZE], dtype=np.int64),
    )
    print(f"[freq-index] output={OUT_PATH}")
    print(f"[freq-index] raw_tokens={raw_tokens} chunks={n_chunks}")
    print(f"[freq-index] bigram_contexts={len(bigram_unique)}")
    print(f"[freq-index] trigram_contexts={len(trigram_unique)}")


if __name__ == "__main__":
    main()
