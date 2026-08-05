"""ngram-gap-lab · code/ngram_freq.py

Clean per-frequency-bin loss statistics for n-gram gap analysis.

This module implements a from-scratch frequency index and per-bin loss
accumulator, avoiding the legacy NgramHitTracker format incompatibility
(scalar-hash `keys` vs tuple `ctx_keys`) that blocked the old codebase.

Two pieces:
  1. GlobalFrequencyIndex  : offline scan of train tokens -> per-context hit count.
  2. FreqBinLossAccumulator: online accumulation of per-bin loss + token count,
     during train/val evaluation, using the same hash as the model's n-gram
     table (so bucket assignment matches the table row the model reads).

Buckets: novel (hit=0 in train), 1, 2, 3, 4, 5, 6-10, 11-20, 21-50,
         51-100, 101-200, 201-500, 501-1k, 1k-5k, 5k+.

Usage:
  # 1. build index once (offline)
  idx = GlobalFrequencyIndex.build(train_tokens, vocab_size)
  idx.save("data/runs/myrun/freq_index.npz")

  # 2. during eval, accumulate per-bin losses
  acc = FreqBinLossAccumulator(idx, n_head=6, head_dim=128)
  for inp, tgt in val_batches:
      per_token_loss = compute_per_token_loss(model, inp, tgt)  # (B, T)
      acc.update(inp, per_token_loss, branch="trigram")
  stats = acc.summary()   # dict of bucket -> {frac, mean_loss, total_contrib}
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Same hash prime families as train.py (must match for bucket assignment to
# match the table row the model reads).
_BASE_BIGRAM_PRIMES = [
    [(2654435761, 2246822519), (1013904223, 6291469)],
    [(374761393, 668265263), (3266489917, 104729)],
    [(1640531527, 97531), (48271, 40503)],
    [(16777619, 2166136261), (3432918353, 461845907)],
]
_BASE_TRIGRAM_PRIMES = [
    (16777619, 2166136261, 3432918353, 461845907, 2654435769, 1540483477),
    (3405403843, 2654435761, 2246822519, 1013904223, 6291469, 374761393),
    (668265263, 3266489917, 104729, 1640531527, 97531, 48271),
]

# Standard bucket edges (hit-count ranges). Novel = hit 0.
BUCKET_EDGES = [
    (0, 0, "novel"),
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 4, "4"),
    (5, 5, "5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, 200, "101-200"),
    (201, 500, "201-500"),
    (501, 1000, "501-1k"),
    (1001, 5000, "1k-5k"),
    (5001, 10**9, "5k+"),
]


def _bucket_label(hit_count: int) -> str:
    for lo, hi, label in BUCKET_EDGES:
        if lo <= hit_count <= hi:
            return label
    return "5k+"


def _all_bucket_labels() -> list:
    return [label for _, _, label in BUCKET_EDGES]


# ---------------------------------------------------------------------------
# 1. Global frequency index (offline scan of train tokens)
# ---------------------------------------------------------------------------


class GlobalFrequencyIndex:
    """Per-context (bigram/trigram) hit count from one train epoch.

    Uses layer-0 hash primes to assign a scalar row id per context. The index
    maps row_id -> hit_count. Because the model uses K=2 multi-hash, we index
    by the *first* hash only (layer 0, table 0) — this is sufficient for
    bucketing since all K hashes share the same hit-count distribution by
    construction (they're applied to the same token context).

    Stored format: npz with keys
      bigram_rows (int64), bigram_counts (int32),
      trigram_rows (int64), trigram_counts (int32)
    """

    def __init__(self, bigram: dict, trigram: dict, vocab_size: int):
        self.bigram = bigram        # row -> count
        self.trigram = trigram
        self.vocab_size = vocab_size
        self.bigram_table_size = vocab_size * 64
        self.trigram_table_size = vocab_size * 64

    @classmethod
    def build(cls, train_tokens: np.ndarray, vocab_size: int) -> "GlobalFrequencyIndex":
        """Scan train tokens (1D uint16/uint32 array) once, count all bigram/trigram contexts."""
        tokens = train_tokens.astype(np.int64)
        N = len(tokens)
        # bigram: (prev, cur) -> hash with layer-0 primes, table 0
        bp = _BASE_BIGRAM_PRIMES[0][0]  # (p1, p2)
        bts = vocab_size * 64
        prev = tokens[:-1]
        cur = tokens[1:]
        b_rows = ((prev * bp[0]) ^ (cur * bp[1])) % bts
        # trigram: (prev2, prev, cur) -> hash with layer-0 primes, table 0
        tp = _BASE_TRIGRAM_PRIMES[0]  # (p0..p5)
        tts = vocab_size * 64
        prev2 = tokens[:-2]
        prev1 = tokens[1:-1]
        cur_t = tokens[2:]
        t_rows = ((prev2 * tp[0]) ^ (prev1 * tp[1]) ^ (cur_t * tp[2])) % tts
        # count unique rows
        b_unique, b_counts = np.unique(b_rows, return_counts=True)
        t_unique, t_counts = np.unique(t_rows, return_counts=True)
        bigram = {int(r): int(c) for r, c in zip(b_unique, b_counts)}
        trigram = {int(r): int(c) for r, c in zip(t_unique, t_counts)}
        return cls(bigram, trigram, vocab_size)

    @classmethod
    def build_from_shards(cls, data_dir: str, shard_ids: list, vocab_size: int) -> "GlobalFrequencyIndex":
        """Concatenate all train shards and build the index."""
        chunks = []
        for sid in shard_ids:
            path = os.path.join(data_dir, f"shard_{sid:05d}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing shard: {path}")
            buf = np.memmap(path, dtype=np.uint16, mode="r")
            chunks.append(np.array(buf))
        all_tokens = np.concatenate(chunks)
        return cls.build(all_tokens, vocab_size)

    def save(self, path: str):
        b_rows = np.array(list(self.bigram.keys()), dtype=np.int64)
        b_counts = np.array(list(self.bigram.values()), dtype=np.int32)
        t_rows = np.array(list(self.trigram.keys()), dtype=np.int64)
        t_counts = np.array(list(self.trigram.values()), dtype=np.int32)
        np.savez(path, bigram_rows=b_rows, bigram_counts=b_counts,
                 trigram_rows=t_rows, trigram_counts=t_counts,
                 vocab_size=np.array([self.vocab_size]))

    @classmethod
    def load(cls, path: str) -> "GlobalFrequencyIndex":
        d = np.load(path)
        bigram = {int(r): int(c) for r, c in zip(d["bigram_rows"], d["bigram_counts"])}
        trigram = {int(r): int(c) for r, c in zip(d["trigram_rows"], d["trigram_counts"])}
        vocab_size = int(d["vocab_size"][0])
        return cls(bigram, trigram, vocab_size)

    def hit_count(self, branch: str, row: int) -> int:
        """Lookup hit count for a given (branch, row). Returns 0 if novel."""
        tbl = self.bigram if branch == "bigram" else self.trigram
        return tbl.get(int(row), 0)

    def hit_count_tensor(self, branch: str, rows: torch.Tensor) -> torch.Tensor:
        """Vectorized hit count lookup. rows: (B,T) long tensor -> (B,T) int32."""
        tbl = self.bigram if branch == "bigram" else self.trigram
        rows_np = rows.detach().cpu().numpy().astype(np.int64)
        # build lookup array (sparse -> dense via dict)
        # for speed, use np.vectorize with dict.get
        get = np.vectorize(lambda r: tbl.get(int(r), 0))
        counts = get(rows_np).astype(np.int32)
        return torch.from_numpy(counts).to(rows.device)


# ---------------------------------------------------------------------------
# 2. Per-frequency-bin loss accumulator (online, during eval)
# ---------------------------------------------------------------------------


class FreqBinLossAccumulator:
    """Accumulates per-token loss grouped by n-gram context hit-count bucket.

    For each batch (inp, tgt, per_token_loss):
      1. Compute the n-gram context row for each position (using layer-0 hash).
      2. Lookup hit count from GlobalFrequencyIndex.
      3. Map hit count -> bucket label.
      4. Accumulate loss_sum and token_count per bucket.

    Final summary gives per-bucket:
      frac          = token_count / total_tokens
      mean_loss     = loss_sum / token_count
      total_contrib = frac * mean_loss  (contribution to overall loss)
    """

    def __init__(self, freq_index: GlobalFrequencyIndex, vocab_size: int,
                 branch: str = "trigram"):
        self.freq_index = freq_index
        self.vocab_size = vocab_size
        self.branch = branch  # "bigram" or "trigram"
        self.table_size = (vocab_size * 64)
        self._loss_sum = {label: 0.0 for label in _all_bucket_labels()}
        self._token_count = {label: 0 for label in _all_bucket_labels()}
        self._total_tokens = 0

    def _compute_rows(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute layer-0 table-0 hash rows for each position. Returns (B,T) long."""
        B, T = inp.size()
        if self.branch == "bigram":
            bp = _BASE_BIGRAM_PRIMES[0][0]
            prev = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
            rows = ((prev.long() * bp[0]) ^ (inp.long() * bp[1])) % self.table_size
        else:  # trigram
            tp = _BASE_TRIGRAM_PRIMES[0]
            prev2 = torch.cat([inp[:, :2], inp[:, :-2]], dim=1)
            prev1 = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
            rows = ((prev2.long() * tp[0]) ^ (prev1.long() * tp[1]) ^ (inp.long() * tp[2])) % self.table_size
        return rows

    def update(self, inp: torch.Tensor, per_token_loss: torch.Tensor):
        """Accumulate one batch. inp: (B,T), per_token_loss: (B,T) float."""
        rows = self._compute_rows(inp)  # (B,T) long
        hits = self.freq_index.hit_count_tensor(self.branch, rows)  # (B,T) int32
        # map to bucket labels
        hits_np = hits.cpu().numpy().astype(np.int64)
        loss_np = per_token_loss.detach().cpu().numpy().astype(np.float64)
        # flatten
        hits_flat = hits_np.ravel()
        loss_flat = loss_np.ravel()
        for lo, hi, label in BUCKET_EDGES:
            mask = (hits_flat >= lo) & (hits_flat <= hi)
            n = int(mask.sum())
            if n > 0:
                self._loss_sum[label] += float(loss_flat[mask].sum())
                self._token_count[label] += n
        self._total_tokens += loss_flat.size

    def summary(self) -> dict:
        out = {}
        for label in _all_bucket_labels():
            n = self._token_count[label]
            if n > 0:
                mean_loss = self._loss_sum[label] / n
                frac = n / max(1, self._total_tokens)
                out[label] = {
                    "token_count": n,
                    "frac": frac,
                    "mean_loss": mean_loss,
                    "total_contrib": frac * mean_loss,
                }
            else:
                out[label] = {"token_count": 0, "frac": 0.0, "mean_loss": 0.0, "total_contrib": 0.0}
        return out

    def save_json(self, path: str, meta: Optional[dict] = None):
        d = {"branch": self.branch, "total_tokens": self._total_tokens}
        if meta:
            d["meta"] = meta
        d["buckets"] = self.summary()
        with open(path, "w") as f:
            json.dump(d, f, indent=2)


# ---------------------------------------------------------------------------
# 3. Convenience: compute per-token loss from model
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_per_token_loss(model, inp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """Returns (B, T) per-token cross-entropy loss (no reduction)."""
    logits = model(inp)  # (B, T, V)
    loss = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        tgt.view(-1), ignore_index=-1, reduction="none").view(tgt.size())
    return loss


# ---------------------------------------------------------------------------
# 4. CLI: build index from shards
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_shards", required=True, help="comma-separated shard ids")
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--out", required=True, help="output npz path")
    args = parser.parse_args()
    shard_ids = [int(x) for x in args.train_shards.split(",") if x.strip()]
    idx = GlobalFrequencyIndex.build_from_shards(args.data_dir, shard_ids, args.vocab_size)
    idx.save(args.out)
    print(f"[ngram_freq] saved index to {args.out}")
    print(f"[ngram_freq] bigram unique rows: {len(idx.bigram)}")
    print(f"[ngram_freq] trigram unique rows: {len(idx.trigram)}")
