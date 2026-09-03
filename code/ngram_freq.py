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
import zlib
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


def _context_key(tokens: np.ndarray, position: int, branch: str,
                 vocab_size: int) -> int:
    """Return the raw-corpus context key at one input/loss position."""
    if branch == "bigram":
        return int(tokens[position - 1]) * vocab_size + int(tokens[position])
    if branch == "trigram":
        v2 = vocab_size * vocab_size
        return (int(tokens[position - 2]) * v2
                + int(tokens[position - 1]) * vocab_size
                + int(tokens[position]))
    raise ValueError(f"unknown n-gram branch: {branch}")


def _stable_group_seed(seed: int, source: str, branch: str, bucket: str) -> int:
    """Derive a reproducible independent reservoir RNG for one sample group."""
    label = f"{seed}:{source}:{branch}:{bucket}".encode("utf-8")
    return (int(seed) + zlib.crc32(label)) % (2 ** 32)


def _merge_priority_reservoir(reservoir: list, priorities: np.ndarray,
                             candidates: list, limit: int) -> None:
    """Merge local candidates chosen by iid random priority into a top-k reservoir.

    Retaining the globally smallest iid priorities is an exact uniform sample
    without replacement.  Each shard block contributes only its local top-k,
    because no other candidate ranked below those k can enter the global top-k.
    """
    if limit <= 0 or not candidates:
        return
    reservoir.extend((float(priority), candidate)
                     for priority, candidate in zip(priorities, candidates))
    reservoir.sort(key=lambda item: item[0])
    del reservoir[limit:]


def build_fixed_gram_manifest(
    data_dir: str,
    train_shards: list,
    val_shards: list,
    freq_index: "GlobalFrequencyIndex",
    vocab_size: int,
    sequence_len: int,
    samples_per_bucket: int = 100,
    seed: int = 42,
    scan_chunks_per_block: int = 256,
) -> dict:
    """Build a reproducible bucket-stratified occurrence manifest.

    The scan deliberately bypasses ``TokenizedShardDataset.iter_batches``.
    Each candidate is a real loss position in a complete raw shard chunk;
    only the first ``branch`` context positions are excluded because the
    model would otherwise use its synthetic left padding context.
    """
    if samples_per_bucket < 0:
        raise ValueError("samples_per_bucket must be non-negative")
    chunk_size = sequence_len + 1
    groups = {
        source: {
            branch: {
                bucket: {
                    "candidate_count": 0,
                    "selected_count": 0,
                    "samples": [],
                }
                for bucket in _all_bucket_labels()
            }
            for branch in ("bigram", "trigram")
        }
        for source in ("train", "val")
    }
    if scan_chunks_per_block <= 0:
        raise ValueError("scan_chunks_per_block must be positive")
    rngs = {
        (source, branch, bucket): np.random.default_rng(
            _stable_group_seed(seed, source, branch, bucket))
        for source in groups
        for branch in ("bigram", "trigram")
        for bucket in _all_bucket_labels()
    }
    reservoirs = {key: [] for key in rngs}
    bucket_upper_bounds = np.array([hi for _, hi, _ in BUCKET_EDGES], dtype=np.int64)

    shard_map = {
        "train": [int(x) for x in train_shards],
        "val": [int(x) for x in val_shards],
    }
    for source, shard_ids in shard_map.items():
        for shard_id in shard_ids:
            path = os.path.join(data_dir, f"shard_{shard_id:05d}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing shard: {path}")
            buf = np.memmap(path, dtype=np.uint16, mode="r")
            n_chunks = len(buf) // chunk_size
            for block_first_chunk in range(0, n_chunks, scan_chunks_per_block):
                block_chunk_count = min(scan_chunks_per_block, n_chunks - block_first_chunk)
                raw = buf[
                    block_first_chunk * chunk_size:
                    (block_first_chunk + block_chunk_count) * chunk_size
                ].reshape(block_chunk_count, chunk_size)
                inp = np.asarray(raw[:, :-1], dtype=np.int64)
                for branch, first_position in (("bigram", 1), ("trigram", 2)):
                    if branch == "bigram":
                        keys = inp[:, :-1] * vocab_size + inp[:, 1:]
                    else:
                        keys = (
                            inp[:, :-2] * (vocab_size * vocab_size)
                            + inp[:, 1:-1] * vocab_size
                            + inp[:, 2:]
                        )
                    hits = freq_index.hit_count_array(branch, keys)
                    bucket_ids = np.searchsorted(
                        bucket_upper_bounds, hits, side="left")
                    bucket_ids = np.minimum(bucket_ids, len(BUCKET_EDGES) - 1)
                    flat_keys = keys.ravel()
                    flat_hits = hits.ravel()
                    flat_bucket_ids = bucket_ids.ravel()
                    positions_per_chunk = keys.shape[1]
                    for bucket_id, (_, _, bucket) in enumerate(BUCKET_EDGES):
                        local_indices = np.flatnonzero(flat_bucket_ids == bucket_id)
                        if not len(local_indices):
                            continue
                        group = groups[source][branch][bucket]
                        group["candidate_count"] += int(len(local_indices))
                        local_limit = min(samples_per_bucket, len(local_indices))
                        if not local_limit:
                            continue
                        local_priorities = rngs[(source, branch, bucket)].random(
                            len(local_indices))
                        if local_limit < len(local_indices):
                            selected = np.argpartition(
                                local_priorities, local_limit - 1)[:local_limit]
                        else:
                            selected = np.arange(len(local_indices))
                        selected_indices = local_indices[selected]
                        candidate_rows = selected_indices // positions_per_chunk
                        candidate_positions = (
                            selected_indices % positions_per_chunk + first_position)
                        candidates = [
                            {
                                "source": source,
                                "branch": branch,
                                "bucket": bucket,
                                "shard": shard_id,
                                "chunk_start": int(
                                    (block_first_chunk + row) * chunk_size),
                                "position": int(position),
                                "context_key": int(flat_keys[index]),
                                "hit_count": int(flat_hits[index]),
                            }
                            for index, row, position in zip(
                                selected_indices, candidate_rows, candidate_positions)
                        ]
                        _merge_priority_reservoir(
                            reservoirs[(source, branch, bucket)],
                            local_priorities[selected], candidates, samples_per_bucket)

    for source in groups:
        for branch in groups[source]:
            for bucket in groups[source][branch]:
                group = groups[source][branch][bucket]
                group["samples"] = [candidate for _, candidate in
                                    reservoirs[(source, branch, bucket)]]
                group["samples"].sort(
                    key=lambda x: (x["shard"], x["chunk_start"], x["position"]))
                group["selected_count"] = len(group["samples"])
    return {
        "format_version": 1,
        "seed": int(seed),
        "samples_per_bucket": int(samples_per_bucket),
        "sequence_len": int(sequence_len),
        "vocab_size": int(vocab_size),
        "train_shards": [int(x) for x in train_shards],
        "val_shards": [int(x) for x in val_shards],
        "selection": (
            "reservoir-sampled token occurrences from complete shard chunks; "
            "bucket lookup uses the train frequency index"),
        "groups": groups,
    }


def save_fixed_gram_manifest(manifest: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_fixed_gram_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fixed_gram_manifest_matches(
    manifest: dict,
    train_shards: list,
    val_shards: list,
    vocab_size: int,
    sequence_len: int,
    samples_per_bucket: int,
    seed: int,
) -> bool:
    """Check the run-defining fields before reusing a manifest."""
    return (
        manifest.get("format_version") == 1
        and int(manifest.get("seed", -1)) == int(seed)
        and int(manifest.get("samples_per_bucket", -1)) == int(samples_per_bucket)
        and int(manifest.get("sequence_len", -1)) == int(sequence_len)
        and int(manifest.get("vocab_size", -1)) == int(vocab_size)
        and [int(x) for x in manifest.get("train_shards", [])] == [int(x) for x in train_shards]
        and [int(x) for x in manifest.get("val_shards", [])] == [int(x) for x in val_shards]
        and isinstance(manifest.get("groups"), dict)
    )


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
        self._lookup_cache = {}

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

    def _sorted_lookup(self, branch: str) -> tuple[np.ndarray, np.ndarray]:
        """Build a sorted raw-key lookup once for vectorized manifest scans."""
        cached = self._lookup_cache.get(branch)
        if cached is not None:
            return cached
        table = self.bigram if branch == "bigram" else self.trigram
        keys = np.fromiter(table.keys(), dtype=np.int64, count=len(table))
        counts = np.fromiter(table.values(), dtype=np.int32, count=len(table))
        if len(keys) > 1 and np.any(keys[1:] < keys[:-1]):
            order = np.argsort(keys)
            keys = keys[order]
            counts = counts[order]
        self._lookup_cache[branch] = (keys, counts)
        return keys, counts

    def hit_count_array(self, branch: str, rows: np.ndarray) -> np.ndarray:
        """Lookup an ndarray of raw context keys without a torch round trip."""
        flat = rows.astype(np.int64, copy=False).ravel()
        keys, counts = self._sorted_lookup(branch)
        if not len(keys):
            return np.zeros(rows.shape, dtype=np.int32)
        locations = np.searchsorted(keys, flat)
        valid = locations < len(keys)
        matched = np.zeros(flat.shape, dtype=np.int32)
        matched[valid] = counts[locations[valid]]
        valid[valid] &= keys[locations[valid]] == flat[valid]
        matched[~valid] = 0
        return matched.reshape(rows.shape)

    def hit_count_tensor(self, branch: str, rows: torch.Tensor) -> torch.Tensor:
        """Vectorized hit count lookup. rows: (B,T) long tensor -> (B,T) int32."""
        rows_np = rows.detach().cpu().numpy().astype(np.int64)
        counts = self.hit_count_array(branch, rows_np)
        return torch.from_numpy(counts).to(rows.device)


class ExactFrequencyMask:
    """GPU-resident exact-context frequency mask used during model forward.

    Unlike :class:`GlobalFrequencyIndex`, this class keeps the sorted arrays
    from ``freq_index.npz`` directly instead of expanding roughly 20 million
    trigram keys into a Python dictionary.  Lookup uses ``torch.searchsorted``
    on the model device, so the same mask path is affordable on every writer
    and validation batch.

    ``threshold`` has three representations:

    - ``None``: no-mask reference; lookup still runs, then every context is
      active.  This intentionally keeps its runtime representative.
    - non-negative integer: contexts with train hit count ``<= threshold``
      are inactive.
    - ``"all"``: every bigram and trigram context is inactive.
    """

    def __init__(self, bigram_keys: torch.Tensor, bigram_counts: torch.Tensor,
                 trigram_keys: torch.Tensor, trigram_counts: torch.Tensor,
                 vocab_size: int, threshold: int | None | str,
                 source_path: str = ""):
        if threshold != "all" and threshold is not None and int(threshold) < 0:
            raise ValueError("frequency-mask threshold must be non-negative, none, or all")
        self.vocab_size = int(vocab_size)
        self.threshold = threshold
        self.source_path = source_path
        self.bigram_keys = bigram_keys.to(dtype=torch.int64)
        self.bigram_counts = bigram_counts.to(dtype=torch.int32)
        self.trigram_keys = trigram_keys.to(dtype=torch.int64)
        self.trigram_counts = trigram_counts.to(dtype=torch.int32)
        self._validate_branch("bigram", self.bigram_keys, self.bigram_counts)
        self._validate_branch("trigram", self.trigram_keys, self.trigram_counts)

    @staticmethod
    def _validate_branch(branch: str, keys: torch.Tensor,
                         counts: torch.Tensor) -> None:
        if keys.ndim != 1 or counts.ndim != 1 or keys.numel() != counts.numel():
            raise ValueError(f"invalid {branch} frequency arrays")
        if keys.numel() > 1 and bool(torch.any(keys[1:] <= keys[:-1]).item()):
            raise ValueError(f"{branch} frequency keys must be strictly increasing")
        if counts.numel() and bool(torch.any(counts <= 0).item()):
            raise ValueError(f"{branch} frequency counts must be positive")

    @classmethod
    def load(cls, path: str, device: torch.device,
             threshold: int | None | str) -> "ExactFrequencyMask":
        with np.load(path) as data:
            vocab_size = int(np.asarray(data["vocab_size"]).reshape(-1)[0])
            arrays = [
                torch.from_numpy(np.asarray(data[name])).to(device=device)
                for name in (
                    "bigram_keys", "bigram_counts",
                    "trigram_keys", "trigram_counts",
                )
            ]
        return cls(*arrays, vocab_size=vocab_size, threshold=threshold,
                   source_path=os.path.abspath(path))

    @staticmethod
    def _lookup(keys: torch.Tensor, counts: torch.Tensor,
                raw_context_keys: torch.Tensor) -> torch.Tensor:
        if keys.numel() == 0:
            return torch.zeros_like(raw_context_keys, dtype=torch.int32)
        locations = torch.searchsorted(keys, raw_context_keys)
        in_bounds = locations < keys.numel()
        safe_locations = locations.clamp(max=keys.numel() - 1)
        exact = in_bounds & (keys[safe_locations] == raw_context_keys)
        return torch.where(
            exact,
            counts[safe_locations],
            torch.zeros_like(raw_context_keys, dtype=torch.int32),
        )

    def activity_masks(self, idx: torch.Tensor, prev_idx: torch.Tensor,
                       prev2_idx: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return active bigram/trigram positions using model chunk contexts."""
        if self.threshold == "all":
            inactive = torch.zeros_like(idx, dtype=torch.bool)
            return inactive, inactive
        bigram_raw = prev_idx * self.vocab_size + idx
        trigram_raw = (
            prev2_idx * (self.vocab_size * self.vocab_size)
            + prev_idx * self.vocab_size + idx
        )
        bigram_hits = self._lookup(
            self.bigram_keys, self.bigram_counts, bigram_raw)
        trigram_hits = self._lookup(
            self.trigram_keys, self.trigram_counts, trigram_raw)
        if self.threshold is None:
            # The lookups above are deliberately retained for a representative
            # no-mask runtime measurement.
            return torch.ones_like(idx, dtype=torch.bool), torch.ones_like(
                idx, dtype=torch.bool)
        threshold = int(self.threshold)
        return bigram_hits > threshold, trigram_hits > threshold

    def statistics(self) -> dict:
        """Describe train-index mass removed by this cumulative threshold."""
        result = {}
        for branch in ("bigram", "trigram"):
            counts = getattr(self, f"{branch}_counts").detach().cpu().numpy()
            total_occurrences = int(counts.astype(np.int64).sum())
            if self.threshold is None:
                selected = np.zeros(counts.shape, dtype=bool)
            elif self.threshold == "all":
                selected = np.ones(counts.shape, dtype=bool)
            else:
                selected = counts <= int(self.threshold)
            masked_occurrences = int(counts[selected].astype(np.int64).sum())
            result[branch] = {
                "unique_contexts": int(counts.size),
                "max_hit_count": int(counts.max()) if counts.size else 0,
                "total_occurrences": total_occurrences,
                "masked_unique_contexts": int(selected.sum()),
                "masked_occurrences": masked_occurrences,
                "masked_occurrence_fraction": (
                    masked_occurrences / total_occurrences
                    if total_occurrences else 0.0
                ),
            }
        return result


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


def fixed_gram_gap_summary(evaluation: dict) -> dict:
    """Combine independent train/val fixed-gram means into the requested gap."""
    train = evaluation.get("train", {})
    val = evaluation.get("val", {})
    out = {}
    for branch in ("bigram", "trigram"):
        out[branch] = {}
        for bucket in _all_bucket_labels():
            tr = train.get(branch, {}).get(bucket, {})
            va = val.get(branch, {}).get(bucket, {})
            tr_count = int(tr.get("sample_count", 0))
            va_count = int(va.get("sample_count", 0))
            tr_mean = tr.get("mean_loss")
            va_mean = va.get("mean_loss")
            gap = None if tr_mean is None or va_mean is None else float(va_mean - tr_mean)
            out[branch][bucket] = {
                "sample_count": min(tr_count, va_count),
                "train_sample_count": tr_count,
                "val_sample_count": va_count,
                "train_mean_loss": tr_mean,
                "val_mean_loss": va_mean,
                "gap_contribution": gap,
            }
    return out


def fixed_gram_overall_loss(source_summary: dict) -> Optional[float]:
    """Average the selected occurrences across both branches for audit only."""
    total = 0.0
    count = 0
    for branch in ("bigram", "trigram"):
        for bucket in _all_bucket_labels():
            stats = source_summary.get(branch, {}).get(bucket, {})
            n = int(stats.get("sample_count", 0))
            mean = stats.get("mean_loss")
            if n and mean is not None:
                total += n * float(mean)
                count += n
    return total / count if count else None


class FixedGramProbe:
    """Evaluate the same manifest-selected token occurrences at each checkpoint."""

    def __init__(self, manifest: dict, data_dir: str, sequence_len: int,
                 device: torch.device, device_batch_size: int = 4):
        self.manifest = manifest
        self.data_dir = data_dir
        self.sequence_len = int(sequence_len)
        self.chunk_size = self.sequence_len + 1
        self.device = device
        self.device_batch_size = max(1, int(device_batch_size))
        self._chunks = {}
        self._refs = {}
        self._prepare()

    def _prepare(self) -> None:
        groups = self.manifest.get("groups", {})
        for source in ("train", "val"):
            for branch in ("bigram", "trigram"):
                for bucket in _all_bucket_labels():
                    group = groups.get(source, {}).get(branch, {}).get(bucket, {})
                    samples = group.get("samples", [])
                    for sample in samples:
                        chunk_key = (
                            source, int(sample["shard"]), int(sample["chunk_start"]))
                        self._refs.setdefault(chunk_key, []).append(
                            (branch, bucket, int(sample["position"])))
        for source, shard, chunk_start in self._refs:
            path = os.path.join(self.data_dir, f"shard_{shard:05d}.bin")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing shard: {path}")
            buf = np.memmap(path, dtype=np.uint16, mode="r")
            raw = np.asarray(buf[chunk_start:chunk_start + self.chunk_size], dtype=np.uint16)
            if len(raw) != self.chunk_size:
                raise ValueError(
                    f"incomplete fixed-gram chunk: {path}:{chunk_start}")
            self._chunks[(source, shard, chunk_start)] = np.array(raw, copy=True)

    def manifest_stats(self) -> dict:
        return {
            "unique_chunks": len(self._chunks),
            "sample_count": {
                source: {
                    branch: {
                        bucket: int(self.manifest["groups"][source][branch][bucket]
                                   .get("selected_count", 0))
                        for bucket in _all_bucket_labels()
                    }
                    for branch in ("bigram", "trigram")
                }
                for source in ("train", "val")
            },
        }

    def evaluate(self, model) -> dict:
        """Return source-separated selected-token means; never touches a train iterator."""
        sums = {
            source: {
                branch: {bucket: 0.0 for bucket in _all_bucket_labels()}
                for branch in ("bigram", "trigram")
            }
            for source in ("train", "val")
        }
        counts = {
            source: {
                branch: {bucket: 0 for bucket in _all_bucket_labels()}
                for branch in ("bigram", "trigram")
            }
            for source in ("train", "val")
        }
        was_training = model.training
        model.eval()
        chunk_items = list(self._chunks.items())
        with torch.no_grad():
            for offset in range(0, len(chunk_items), self.device_batch_size):
                batch_items = chunk_items[offset:offset + self.device_batch_size]
                raw = np.stack([item[1] for item in batch_items], axis=0)
                inp = torch.from_numpy(raw[:, :-1].astype(np.int64)).to(self.device)
                tgt = torch.from_numpy(raw[:, 1:].astype(np.int64)).to(self.device)
                losses = compute_per_token_loss(model, inp, tgt).detach().cpu().numpy()
                for row, (chunk_key, _) in enumerate(batch_items):
                    source = chunk_key[0]
                    for branch, bucket, position in self._refs[chunk_key]:
                        sums[source][branch][bucket] += float(losses[row, position])
                        counts[source][branch][bucket] += 1
        if was_training:
            model.train()
        result = {
            source: {
                branch: {
                    bucket: {
                        "sample_count": counts[source][branch][bucket],
                        "mean_loss": (
                            sums[source][branch][bucket] / counts[source][branch][bucket]
                            if counts[source][branch][bucket] else None),
                    }
                    for bucket in _all_bucket_labels()
                }
                for branch in ("bigram", "trigram")
            }
            for source in ("train", "val")
        }
        return result


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
