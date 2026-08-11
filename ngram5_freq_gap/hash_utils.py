"""Exact tensor implementation of the n-gram dataset hash."""

from __future__ import annotations

import torch


M61 = (1 << 61) - 1
BASE1 = 0x9E3779B97F4A7C15 % M61
BASE2 = 0xC2B2AE3D27D4EB4F % M61
INIT1 = 0x517CC1B727220A95 % M61
INIT2 = 0x6C62272E07BB0142 % M61

_MASK31 = (1 << 31) - 1
_MASK30 = (1 << 30) - 1
_MASK24 = (1 << 24) - 1
_MASK37 = (1 << 37) - 1


def _reduce_m61(value: torch.Tensor) -> torch.Tensor:
    """Reduce a non-negative signed-int64 tensor modulo ``2**61 - 1``."""
    value = (value & M61) + (value >> 61)
    value = (value & M61) + (value >> 61)
    return torch.where(value >= M61, value - M61, value)


def _mul_mod_m61(value: torch.Tensor, constant: int) -> torch.Tensor:
    """Multiply two 61-bit values exactly without signed-int64 overflow.

    Splitting at 31 bits keeps every intermediate below ``2**63``.  For
    ``M = 2**61 - 1``, ``2**61 == 1 (mod M)`` and ``2**62 == 2 (mod M)``.
    """
    lo = value & _MASK31
    hi = value >> 31
    const_lo = constant & _MASK31
    const_hi = constant >> 31
    cross = lo * const_hi + hi * const_lo
    combined = (
        lo * const_lo
        + ((cross & _MASK30) << 31)
        + (cross >> 30)
        + 2 * (hi * const_hi)
    )
    return _reduce_m61(combined)


def hash_bucket_tensor(*context_tokens: torch.Tensor, bucket_count: int) -> torch.Tensor:
    """Return ``data_gen.hash_n(context) % bucket_count`` exactly.

    Accepts any number of context token tensors (trigram: 3, 5-gram: 5, ...).
    All tensors must have identical shapes.
    """
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    if len(context_tokens) < 1:
        raise ValueError("at least one context token tensor required")
    shape = context_tokens[0].shape
    if any(t.shape != shape for t in context_tokens[1:]):
        raise ValueError("all context tensors must have identical shapes")

    h1 = torch.full_like(context_tokens[0], INIT1, dtype=torch.int64)
    h2 = torch.full_like(context_tokens[0], INIT2, dtype=torch.int64)
    for token in context_tokens:
        token64 = token.to(torch.int64)
        h1 = _reduce_m61(_mul_mod_m61(h1, BASE1) + token64)
        h2 = _reduce_m61(_mul_mod_m61(h2, BASE2) + token64)

    high = (h2 >> 24) & _MASK37
    low = h2 & _MASK24
    rotated = (low << 37) | high
    return ((h1 ^ rotated) & M61) % bucket_count


# Backwards-compatible alias.
def hash5_bucket_tensor(c0, c1, c2, c3, c4, bucket_count: int) -> torch.Tensor:
    """Return ``data_gen.hash5(context) % bucket_count`` exactly (5-gram)."""
    return hash_bucket_tensor(c0, c1, c2, c3, c4, bucket_count=bucket_count)


__all__ = ["hash_bucket_tensor", "hash5_bucket_tensor"]
