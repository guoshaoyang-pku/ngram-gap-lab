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
    (6, 8, "6-8"),
    (9, 12, "9-12"),
    (13, 20, "13-20"),
    (21, 30, "21-30"),
    (31, 50, "31-50"),
    (51, 75, "51-75"),
    (76, 100, "76-100"),
    (101, 150, "101-150"),
    (151, 200, "151-200"),
    (201, 300, "201-300"),
    (301, 500, "301-500"),
    (501, 750, "501-750"),
    (751, 1000, "751-1k"),
    (1001, 2000, "1k-2k"),
    (2001, 5000, "2k-5k"),
    (5001, 10000, "5k-10k"),
    (10001, 10**9, "10k+"),
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

    Uses the actual token tuple (not the hash table row) as the key, so that
    novel contexts (never seen in train) are correctly identified. This is
    critical: the hash table has only vocab*64 rows, but the true context
    space is vocab^2 (bigram) or vocab^3 (trigram), so hash collisions would
    mask novel contexts if we used row ids as keys.

    Key encoding:
      bigram:  prev * vocab_size + cur           (range: vocab^2)
      trigram: prev2 * vocab^2 + prev * vocab + cur  (range: vocab^3)

    Stored format: npz with keys
      bigram_keys (int64), bigram_counts (int32),
      trigram_keys (int64), trigram_counts (int32),
      vocab_size (int64)
    """

    def __init__(self, bigram: dict, trigram: dict, vocab_size: int):
        self.bigram = bigram        # context_key -> count
        self.trigram = trigram
        self.vocab_size = vocab_size

    @classmethod
    def build(cls, train_tokens: np.ndarray, vocab_size: int) -> "GlobalFrequencyIndex":
        """Scan train tokens (1D uint16/uint32 array) once, count all bigram/trigram contexts."""
        tokens = train_tokens.astype(np.int64)
        N = len(tokens)
        # bigram: (prev, cur) -> prev * vocab + cur
        prev = tokens[:-1]
        cur = tokens[1:]
        b_keys = prev * vocab_size + cur
        # trigram: (prev2, prev, cur) -> prev2 * vocab^2 + prev * vocab + cur
        prev2 = tokens[:-2]
        prev1 = tokens[1:-1]
        cur_t = tokens[2:]
        t_keys = prev2 * (vocab_size * vocab_size) + prev1 * vocab_size + cur_t
        # count unique keys
        b_unique, b_counts = np.unique(b_keys, return_counts=True)
        t_unique, t_counts = np.unique(t_keys, return_counts=True)
        bigram = {int(k): int(c) for k, c in zip(b_unique, b_counts)}
        trigram = {int(k): int(c) for k, c in zip(t_unique, t_counts)}
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

    @classmethod
    def build_from_chunks(cls, data_dir: str, shard_ids: list, vocab_size: int,
                          n_chunks: int, chunk_size: int) -> "GlobalFrequencyIndex":
        """Build the index from the FIRST `n_chunks` training chunks.

        Each chunk is processed independently with the model's exact context
        semantics (first / first-two positions repeated, never crossing a
        chunk boundary), so the index counts exactly the contexts the model
        actually sees in one epoch.  This fixes the old build_from_shards
        behaviour, which concatenated raw tokens and therefore counted
        cross-chunk pairs the model never sees.

        Chunk layout matches TokenizedShardDataset: each chunk is
        `chunk_size = sequence_len + 1` packed tokens; position t predicts
        token t+1, and the n-gram context at position t is built from
        chunk[max(0, t-1)] (bigram) / chunk[max(0, t-2)], chunk[max(0, t-1)]
        (trigram) plus the current token chunk[t].
        """
        bigram: dict = {}
        trigram: dict = {}
        v2 = vocab_size * vocab_size
        seen = 0
        T = chunk_size - 1
        for sid in shard_ids:
            if seen >= n_chunks:
                break
            path = os.path.join(data_dir, f"shard_{sid:05d}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing shard: {path}")
            buf = np.memmap(path, dtype=np.uint16, mode="r")
            n = len(buf) // chunk_size
            take = min(n, n_chunks - seen)
            tokens = np.array(buf[:take * chunk_size], dtype=np.int64).reshape(take, chunk_size)
            cur = tokens[:, :T]                                        # (M, T)
            prev = np.concatenate([tokens[:, :1], tokens[:, :T - 1]], axis=1)   # (M, T)
            prev2 = np.concatenate([tokens[:, :2], tokens[:, :T - 2]], axis=1)  # (M, T)
            b_keys = prev * vocab_size + cur
            t_keys = prev2 * v2 + prev * vocab_size + cur
            for b_u, b_c in zip(*np.unique(b_keys, return_counts=True)):
                key = int(b_u)
                bigram[key] = bigram.get(key, 0) + int(b_c)
            for t_u, t_c in zip(*np.unique(t_keys, return_counts=True)):
                key = int(t_u)
                trigram[key] = trigram.get(key, 0) + int(t_c)
            seen += take
        return cls(bigram, trigram, vocab_size)

    def save(self, path: str):
        b_keys = np.array(list(self.bigram.keys()), dtype=np.int64)
        b_counts = np.array(list(self.bigram.values()), dtype=np.int32)
        t_keys = np.array(list(self.trigram.keys()), dtype=np.int64)
        t_counts = np.array(list(self.trigram.values()), dtype=np.int32)
        np.savez(path, bigram_keys=b_keys, bigram_counts=b_counts,
                 trigram_keys=t_keys, trigram_counts=t_counts,
                 vocab_size=np.array([self.vocab_size]))

    @classmethod
    def load(cls, path: str) -> "GlobalFrequencyIndex":
        d = np.load(path)
        bigram = {int(k): int(c) for k, c in zip(d["bigram_keys"], d["bigram_counts"])}
        trigram = {int(k): int(c) for k, c in zip(d["trigram_keys"], d["trigram_counts"])}
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
        self._loss_sum = {label: 0.0 for label in _all_bucket_labels()}
        self._token_count = {label: 0 for label in _all_bucket_labels()}
        self._total_tokens = 0

    def _compute_keys(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute context keys for each position. Returns (B,T) long.

        Key encoding (must match GlobalFrequencyIndex.build):
          bigram:  prev * vocab + cur
          trigram: prev2 * vocab^2 + prev * vocab + cur
        """
        B, T = inp.size()
        if self.branch == "bigram":
            prev = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
            keys = prev.long() * self.vocab_size + inp.long()
        else:  # trigram
            prev2 = torch.cat([inp[:, :2], inp[:, :-2]], dim=1)
            prev1 = torch.cat([inp[:, :1], inp[:, :-1]], dim=1)
            keys = (prev2.long() * (self.vocab_size * self.vocab_size)
                    + prev1.long() * self.vocab_size + inp.long())
        return keys

    def update(self, inp: torch.Tensor, per_token_loss: torch.Tensor):
        """Accumulate one batch. inp: (B,T), per_token_loss: (B,T) float."""
        keys = self._compute_keys(inp)  # (B,T) long
        hits = self.freq_index.hit_count_tensor(self.branch, keys)  # (B,T) int32
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


class ExactFreqLossAccumulator:
    """Accumulate per-exact-frequency loss statistics (no wide buckets).

    For each position we compute the exact n-gram context key (identical to
    the model's hashing input), look up its train hit count f in the
    GlobalFrequencyIndex, and accumulate token count / loss sum / loss^2 sum
    keyed by that exact f.  f=0 (novel context) is reported separately with
    only its mean loss (no gap is defined for novel contexts).

    Provides both marginal summaries (token-weighted mean loss per f) and the
    raw sufficient statistics needed for context-matched gap analysis.
    """

    def __init__(self, freq_index: GlobalFrequencyIndex, vocab_size: int,
                 branch: str = "trigram"):
        self.freq_index = freq_index
        self.vocab_size = vocab_size
        self.branch = branch
        # exact f -> dict(token_count, loss_sum, loss_sq_sum, distinct_keys)
        self._stats: dict = {}
        self._total_tokens = 0

    def _compute_keys(self, inp: torch.Tensor) -> torch.Tensor:
        """Context key per position, matching FreqBinLossAccumulator semantics."""
        return FreqBinLossAccumulator._compute_keys(self, inp)

    def update(self, inp: torch.Tensor, per_token_loss: torch.Tensor):
        """Accumulate one batch. inp: (B,T) token ids, per_token_loss: (B,T).

        Vectorized accumulation: counts / loss sums use np.bincount per exact
        f; distinct-context counts use a single lexsort over (f, key) so the
        whole batch is handled in numpy, not per-token Python loops.
        """
        keys = self._compute_keys(inp)                    # (B,T) long
        key_np = keys.cpu().numpy().ravel()
        loss_np = per_token_loss.detach().cpu().numpy().astype(np.float64).ravel()
        counts = self.freq_index.bigram if self.branch == "bigram" else self.freq_index.trigram
        get = np.vectorize(lambda r: counts.get(int(r), 0))
        freq_np = get(key_np).astype(np.int64)
        self._total_tokens += len(loss_np)

        # distinct (f, key) pairs -> distinct contexts per f
        order = np.lexsort((key_np, freq_np))
        sf = freq_np[order]
        sk = key_np[order]
        # mark start of each new (f, key) group: length len(sf)+1 sentinel
        is_start = np.ones(len(sf) + 1, dtype=bool)
        is_start[1:-1] = (sf[1:] != sf[:-1]) | (sk[1:] != sk[:-1])
        is_start[-1] = False
        # per-f token counts and distinct-context counts
        f_groups, starts = np.unique(sf, return_index=True)
        token_counts = np.add.reduceat(np.ones(len(sf), dtype=np.int64),
                                       starts) if len(sf) else np.array([], dtype=np.int64)
        # distinct contexts per f = count of starts within [starts[i], starts[i+1])
        dc_counts = np.add.reduceat(is_start.astype(np.int64)[:-1],
                                    starts) if len(sf) else np.array([], dtype=np.int64)
        # per-f loss sums / sq sums via bincount (f may be large; use sparse dict path)
        max_f = int(freq_np.max()) if len(freq_np) else 0
        loss_sums = np.bincount(freq_np, weights=loss_np, minlength=max_f + 1)
        loss_sq_sums = np.bincount(freq_np, weights=loss_np * loss_np, minlength=max_f + 1)
        for f, tc, dc in zip(f_groups.tolist(), token_counts.tolist(), dc_counts.tolist()):
            st = self._stats.setdefault(int(f), {
                "token_count": 0, "distinct_contexts": 0,
                "loss_sum": 0.0, "loss_sq_sum": 0.0,
            })
            st["token_count"] += int(tc)
            st["distinct_contexts"] += int(dc)
            st["loss_sum"] += float(loss_sums[int(f)])
            st["loss_sq_sum"] += float(loss_sq_sums[int(f)])

    def summary(self, min_contexts: int = 0) -> dict:
        """Return per-exact-f summary dict (f -> stats with mean loss)."""
        out = {}
        for f, st in sorted(self._stats.items()):
            n = st["token_count"]
            if n <= 0:
                continue
            entry = {
                "f": int(f),
                "token_count": n,
                "distinct_contexts": st["distinct_contexts"],
                "mean_loss": st["loss_sum"] / n,
                "loss_sum": st["loss_sum"],
                "loss_sq_sum": st["loss_sq_sum"],
            }
            if min_contexts and st["distinct_contexts"] < min_contexts:
                entry["excluded"] = f"distinct_contexts < {min_contexts}"
            out[int(f)] = entry
        return out

    def clean(self) -> dict:
        """Return JSON-serializable per-f dict."""
        return {int(f): {k: v for k, v in st.items()} for f, st in self._stats.items()}


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
