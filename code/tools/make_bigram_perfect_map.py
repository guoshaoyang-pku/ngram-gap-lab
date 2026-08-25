#!/usr/bin/env python
"""Build a collision-free (perfect) bigram context -> row map for --bigram_perfect_map.

Scans the SAME train stream as TokenizedShardDataset (fixed shard order,
chunk = sequence_len+1 uint16 tokens, self-pair boundary at each chunk head,
matching `prev_idx = cat([idx[:, :1], idx[:, :-1]])` in train.py) and assigns
every distinct packed bigram context (prev * vocab + cur) a consecutive row
id. Contexts never seen in the train stream are mapped to `n_distinct`, the
shared OOV/UNK row used by the perfect table at val time.

Optionally scans val shards to report the OOV token fraction of the map.

Usage:
  python code/tools/make_bigram_perfect_map.py \
      --data_dir data/tokenized --train_shards 1 \
      --val_shards 2,3 --out data/bigram_perfect_map_s1.npz
"""
import argparse
import hashlib
import os

import numpy as np


def _shard_packed(data_dir, sid, chunk_size, vocab):
    """Vectorized packed bigram contexts for one shard (train.py chunk口径)."""
    path = os.path.join(data_dir, f"shard_{sid:05d}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing shard: {path}")
    buf = np.memmap(path, dtype=np.uint16, mode="r")
    n = len(buf) // chunk_size
    arr = np.asarray(buf[: n * chunk_size]).reshape(n, chunk_size).astype(np.int64)
    cur = arr[:, :-1]                       # inp tokens (B, T)
    prev = np.empty_like(cur)
    prev[:, 0] = cur[:, 0]                  # self-pair at chunk head (train.py口径)
    prev[:, 1:] = cur[:, :-1]
    return (prev * vocab + cur).ravel(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--train_shards", default="1")
    ap.add_argument("--val_shards", default="")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--sequence_len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    chunk_size = args.sequence_len + 1
    v2 = args.vocab_size * args.vocab_size
    train_shards = [int(x) for x in args.train_shards.split(",") if x.strip()]
    val_shards = [int(x) for x in args.val_shards.split(",") if x.strip()]

    visited = np.zeros(v2, dtype=bool)
    total_chunks = 0
    for sid in train_shards:
        packed, n = _shard_packed(args.data_dir, sid, chunk_size, args.vocab_size)
        visited[packed] = True
        total_chunks += n
        print(f"[map] train shard {sid}: {n} chunks, {packed.size:,} contexts scanned")

    rows = np.flatnonzero(visited)
    n_distinct = int(rows.size)
    row_map = np.full(v2, n_distinct, dtype=np.int32)  # OOV -> n_distinct (UNK row)
    row_map[rows] = np.arange(n_distinct, dtype=np.int32)
    sha = hashlib.sha256(row_map.tobytes()).hexdigest()[:16]
    print(f"[map] distinct bigram contexts: {n_distinct:,} over {total_chunks} chunks")

    oov_rate = None
    if val_shards:
        n_oov = n_tot = 0
        for sid in val_shards:
            packed, _ = _shard_packed(args.data_dir, sid, chunk_size, args.vocab_size)
            n_oov += int((~visited[packed]).sum())
            n_tot += packed.size
        oov_rate = n_oov / max(1, n_tot)
        print(f"[map] val OOV token fraction: {oov_rate:.6f} ({n_oov:,}/{n_tot:,}) "
              f"over shards {val_shards}")

    np.savez(
        args.out,
        map=row_map,
        n_distinct=n_distinct,
        vocab_size=args.vocab_size,
        sequence_len=args.sequence_len,
        train_shards=train_shards,
        val_shards=val_shards,
        val_oov_rate=oov_rate if oov_rate is not None else -1.0,
        map_sha256=sha,
        boundary="self-pair",
    )
    print(f"[map] wrote {args.out} (sha256[:16]={sha}, rows={n_distinct + 1:,} incl UNK)")


if __name__ == "__main__":
    main()
