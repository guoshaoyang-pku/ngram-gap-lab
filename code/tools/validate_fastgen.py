#!/usr/bin/env python3
"""Validate vectorized fast scan/emit vs reference per-document loops.

Runs on the cluster venv.  Requires AUTORESEARCH_CACHE_DIR to point at the
full163 cache-home.  Uses cached shards only (no re-tokenization).

  A. fast scan == reference per-doc scan (exact, shard 00000)
  B. fast emit == reference per-occurrence emit on a truncated window with the
     same numpy Generator stream (byte-identical blocks)
  C. block structure well-formed on the full shard
  D. generate() CLI wiring on a 3-shard subset (npz == fast scan counts, meta,
     loader smoke)
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "token_cache_full163"
TRAIN_SHARDS = [0, 1, 2]
ORDER = 3
VOCAB = 8192
SEP = 8191
DATASET_SEED = 20260808
LAM_TRAIN = 0.8
dg = None


def load_flat(shard_id):
    flat = np.fromfile(CACHE / f"shard_{shard_id:05d}.npy", dtype=np.uint16)
    offsets = np.fromfile(CACHE / f"shard_{shard_id:05d}.offsets.npy", dtype=np.uint64)
    return flat, offsets


def ref_scan_docs(flat, offsets):
    """Reference per-document scan (mirrors data_gen.scan_exact_counts_packed)."""
    counts = Counter()
    total = 0
    for start, end in zip(offsets[:-1], offsets[1:]):
        doc = flat[start:end].tolist()
        if len(doc) < ORDER + 1:
            continue
        for i in range(ORDER, len(doc)):
            key = 0
            for tok in doc[i - ORDER:i]:
                key = key * VOCAB + int(tok)
            counts[key] += 1
            total += 1
    return counts, total


def fast_scan_shards(shard_ids):
    per_shard_keys, per_shard_counts = [], []
    total = 0
    for sid in shard_ids:
        flat, offsets = load_flat(sid)
        valid, _ = dg._context_window_mask(flat, offsets, ORDER)
        if not valid.any():
            continue
        win = np.lib.stride_tricks.sliding_window_view(
            flat[:-1].astype(np.int64), ORDER
        )
        keys = dg._pack_context_keys(win[valid], VOCAB, ORDER)
        ukeys, ucounts = np.unique(keys, return_counts=True)
        per_shard_keys.append(ukeys)
        per_shard_counts.append(ucounts)
        total += int(valid.sum())
    keys_all = np.concatenate(per_shard_keys)
    cnts_all = np.concatenate(per_shard_counts)
    order_idx = np.argsort(keys_all, kind="stable")
    k = keys_all[order_idx]
    c = cnts_all[order_idx]
    boundaries = np.concatenate([[0], np.flatnonzero(k[1:] != k[:-1]) + 1])
    counts = Counter(
        dict(zip(k[boundaries].tolist(), np.add.reduceat(c, boundaries).tolist()))
    )
    return counts, total


def ref_emit_window(flat, offsets, rng_gen, max_occ):
    """Reference per-occurrence emit over the first max_occ valid occurrences."""
    cdf = dg._poisson_cdf(LAM_TRAIN)
    valid, nxt_pos = dg._context_window_mask(flat, offsets, ORDER)
    n = int(valid.sum())
    n = min(n, max_occ)
    t = flat.astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(t[:-1], ORDER)
    ctx_all = win[valid][:n]
    nxt_all = t[nxt_pos[:n]]
    r = rng_gen.random(n)
    copies = np.searchsorted(cdf, r)
    out = []
    for j in range(n):
        ck = int(copies[j])
        if ck <= 0:
            continue
        blk = list(ctx_all[j]) + [int(nxt_all[j]), SEP]
        out.extend(blk * ck)
    return out


def fast_emit_window(flat, offsets, rng_gen, max_occ):
    """Fast vectorized emit, truncated to the first max_occ occurrences."""
    cdf = dg._poisson_cdf(LAM_TRAIN)
    valid, nxt_pos = dg._context_window_mask(flat, offsets, ORDER)
    n = int(valid.sum())
    n = min(n, max_occ)
    t = flat.astype(np.int64)
    win = np.lib.stride_tricks.sliding_window_view(t[:-1], ORDER)
    ctx = win[valid][:n]
    nxt = t[nxt_pos[:n]]
    r = rng_gen.random(n)
    copies = np.searchsorted(cdf, r).astype(np.int64)
    keep = copies > 0
    blocks = np.concatenate([ctx[keep], nxt[keep, None]], axis=1)
    blocks = np.repeat(blocks, copies[keep], axis=0)
    out = np.empty((len(blocks), ORDER + 2), dtype=np.uint16)
    out[:, :ORDER + 1] = blocks.astype(np.uint16)
    out[:, ORDER + 1] = SEP
    return out.reshape(-1).tolist()


def main():
    global ROOT, CACHE, dg
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(os.environ.get("NGLAB_ROOT", ROOT)),
    )
    parser.add_argument(
        "--token-cache",
        type=Path,
        default=None,
        help="directory containing cached shard_*.npy and offsets files",
    )
    parser.add_argument(
        "--cache-home",
        type=Path,
        default=None,
        help="upstream cache home containing the parquet data and tokenizer",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="tokenizer directory used by the generate() smoke",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="temporary output directory for the generate() smoke",
    )
    args = parser.parse_args()
    ROOT = args.repo_root.resolve()
    CACHE = (
        args.token_cache.resolve()
        if args.token_cache is not None
        else Path(os.environ.get("NGLAB_TOKEN_CACHE", CACHE)).resolve()
    )
    cache_home = (
        args.cache_home.resolve()
        if args.cache_home is not None
        else Path(os.environ.get("AUTORESEARCH_CACHE_DIR", "")).resolve()
    )
    if not CACHE.is_dir():
        raise SystemExit(f"token cache does not exist: {CACHE}")
    if not cache_home.is_dir():
        raise SystemExit(
            "cache home is required; pass --cache-home or set AUTORESEARCH_CACHE_DIR"
        )
    os.environ["AUTORESEARCH_CACHE_DIR"] = str(cache_home)
    sys.path.insert(0, str(ROOT / "ngram5_freq_gap"))
    import data_gen as dg_module

    dg = dg_module
    tokenizer_dir = (
        args.tokenizer_dir.resolve()
        if args.tokenizer_dir is not None
        else Path(os.environ.get("NGLAB_TOKENIZER_DIR", cache_home / "tokenizer"))
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else ROOT / "data" / "validation" / "fastgen"
    )

    # ---- A. scan equivalence on shard 00000 ----
    flat0, off0 = load_flat(0)
    ref_counts, ref_total = ref_scan_docs(flat0, off0)
    fast_counts, fast_total = fast_scan_shards([0])
    assert ref_total == fast_total, f"scan total mismatch {ref_total} vs {fast_total}"
    assert ref_counts == fast_counts, "scan counts mismatch!"
    print(f"[A] scan OK shard0: total={ref_total:,} distinct={len(ref_counts):,} (exact)")

    # ---- B. emit equivalence on a truncated window (same RNG stream) ----
    seed = DATASET_SEED ^ 0x4444
    MAXOCC = 200_000
    ref_out = ref_emit_window(flat0, off0, np.random.default_rng(seed), MAXOCC)
    fast_out = fast_emit_window(flat0, off0, np.random.default_rng(seed), MAXOCC)
    assert len(ref_out) == len(fast_out), f"emit len {len(ref_out)} vs {len(fast_out)}"
    assert ref_out == fast_out, "emit bytes mismatch!"
    print(f"[B] emit OK: {len(ref_out):,} tokens in window, byte-identical")

    # ---- C. block structure on full shard (fast full output) ----
    chunk = dg._emit_shard_blocks_fast(
        flat0, off0, order=ORDER, vocab=VOCAB, lam=LAM_TRAIN,
        cdf=dg._poisson_cdf(LAM_TRAIN),
        rng_gen=np.random.default_rng(seed), sep_token=SEP,
    )
    bl = ORDER + 2
    n_blocks = len(chunk) // bl
    mat = chunk[: n_blocks * bl].reshape(-1, bl)
    assert np.all(mat[:, -1] == SEP), "SEP position wrong"
    assert np.all(mat < VOCAB), "token out of vocab"
    assert len(chunk) % bl == 0, "flat not block-aligned"
    print(f"[C] block structure OK: {n_blocks:,} blocks, block-aligned")

    # ---- D. CLI-level generate() on 3-shard subset ----
    def split3(split):
        ids = TRAIN_SHARDS if split == "train" else [6542]
        base = os.path.join(str(cache_home), "data")
        return [os.path.join(base, f"shard_{i:05d}.parquet") for i in ids]

    _orig_load = dg._load_upstream_lib

    def _patched_load():
        mod = _orig_load()
        mod.split_parquet_files = split3
        return mod

    dg._load_upstream_lib = _patched_load
    if out_dir.exists():
        shutil.rmtree(out_dir)
    dg.generate(
        out_dir,
        alpha=0.0, bucket_count=5_000_000, f_train=0.8, f_val=0.2,
        k_min=0.25, k_max=8.0, r_ref_mode="median", r_ref_fixed=None,
        dataset_seed=DATASET_SEED, doc_len=2048, max_tokens=None,
        tokenizer_dir=str(tokenizer_dir),
        order=ORDER, val_source="train", val_frac=0.02, emit_format="bin",
        token_cache_dir=str(CACHE), use_fast_scan=True, use_fast_emit=True,
    )
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["frequency_definition"] == "exact_train_epoch_context_count"
    npz = np.load(out_dir / "exact_ngram_counts.npz")
    npz_counts = dict(zip(npz["keys"].tolist(), npz["counts"].tolist()))
    fast3, fast3_total = fast_scan_shards(TRAIN_SHARDS)
    assert npz_counts == fast3, "CLI npz != fast scan counts"
    assert meta["train_tokens"] % 2045 == 0, "train stream not row-stride aligned"
    assert meta["val_tokens"] % 2045 == 0, "val stream not row-stride aligned"
    print(f"[D] CLI generate OK: train_tokens={meta['train_tokens']:,} "
          f"val_tokens={meta['val_tokens']:,} distinct={len(npz_counts):,}")

    # loader smoke
    os.environ["NGRAM5_DATA_DIR"] = str(out_dir)
    from lib import _ngram5_block_dataloader  # noqa: E402
    loader = _ngram5_block_dataloader(2, 2048, "train", return_metadata=False, data_seed=42)
    for _ in range(2):
        x, y, ep = next(loader)
        assert x.shape == (2, 2048) and y.shape == (2, 2048), x.shape
        assert x.dtype == torch.long
    print("[D] loader smoke OK (2 batches of shape (2,2048))")
    print("ALL VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
