#!/usr/bin/env python3
"""X3 (corpus_rbar_freq_v1): corpus-side support width r(f) statistics. Zero GPU.

Scientific question (docs/experiment-log.md section 29): for an exact n-gram
context with train hit count f, how does the support width r(f) -- the number
of distinct next-tokens observed after that context -- grow with f?  This
decides in which frequency window the sampling-law analytic-region condition
f*P(y) >> 1 can hold, i.e. whether the (K-1)/f gap law applies.

Semantics (must match training exactly):
  * The shard is read as fixed chunks of `chunk_size = sequence_len + 1`
    tokens; the trailing partial chunk is dropped (same as
    GlobalFrequencyIndex.build_from_chunks / TokenizedShardDataset).
  * At input position t (t = 0..T-1, T = chunk_size - 1) the model's context
    is built from chunk[max(0,t-1)] (bigram) or chunk[max(0,t-2)],
    chunk[max(0,t-1)] (trigram) PLUS the current token chunk[t]; the
    prediction target is chunk[t+1].  Contexts never cross chunk boundaries.
  * Key encoding matches code/ngram_freq.py:
      bigram:  prev * V + cur            (range < V^2)
      trigram: prev2 * V^2 + prev * V + cur   (range < V^3)

Per-context quantities:
  f   = occurrence count of the context (one epoch of shard 1)
  r   = number of DISTINCT continuation tokens observed at those occurrences
  s1  = number of continuations seen exactly once (singletons)
  mgt = s1 / f   (Good-Turing missing-mass estimate: probability that the
        NEXT draw is a continuation never seen in these f draws)

Outputs (written to --out):
  rbar_by_exact_f_<branch>.npz : f_values, ctx_count, mean_r, median_r,
                                 mean_mgt  (per exact f)
  rbar_bins_<branch>.json      : log-spaced f-bin stats (mean/median r,
                                 r quantiles, mean mgt, analytic-region
                                 coverage fractions at several eps)
  summary.json                 : inputs + md5, token/context counts, and a
                                 cross-check of recounted f against
                                 data/freq_index.npz (which was built by raw
                                 concatenation and therefore includes a small
                                 number of cross-chunk pairs; differences are
                                 expected and reported, not hidden).

Everything is recomputable from shard_00001.bin + freq_index.npz alone.
CPU-only, numpy-only, deterministic.
"""

import argparse
import hashlib
import json
import os

import numpy as np


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def context_and_target_arrays(tokens: np.ndarray, chunk_size: int, vocab: int):
    """Return (bigram_keys, trigram_keys, targets) for every training position.

    tokens: 1-D int64 array of the whole shard (len divisible by chunk_size
    after truncation).  Vectorised over all chunks at once, identical to
    GlobalFrequencyIndex.build_from_chunks.
    """
    mat = tokens.reshape(-1, chunk_size)
    T = chunk_size - 1
    cur = mat[:, :T]
    prev = np.concatenate([mat[:, :1], mat[:, : T - 1]], axis=1)
    prev2 = np.concatenate([mat[:, :2], mat[:, : T - 2]], axis=1)
    b_keys = prev * vocab + cur
    v2 = vocab * vocab
    t_keys = prev2 * v2 + prev * vocab + cur
    targets = mat[:, 1:]
    return b_keys.ravel(), t_keys.ravel(), targets.ravel()


def per_context_stats(keys: np.ndarray, targets: np.ndarray, vocab: int):
    """Compute per-context f, r, s1 from (key, continuation) occurrences.

    Returns (ctx_keys, f, r, s1) sorted by ctx_keys.
    """
    # Encode (key, target) pairs into one int64: key < V^3 = 5.5e11,
    # pair < V^3 * V = 4.5e15 << 2^63.
    pairs = keys * vocab + targets
    uniq_pairs, pair_counts = np.unique(pairs, return_counts=True)
    ctx_of_pair = uniq_pairs // vocab
    # Group pairs by context.
    ctx_keys, start_idx = np.unique(ctx_of_pair, return_index=True)
    f = np.add.reduceat(pair_counts, start_idx)
    r = np.diff(np.append(start_idx, len(ctx_of_pair)))
    s1 = np.add.reduceat((pair_counts == 1).astype(np.int64), start_idx)
    return ctx_keys, f, r.astype(np.int64), s1


def log_bin_edges(fmax: int, n_bins: int = 32):
    edges = np.logspace(0, np.log10(fmax), n_bins)
    edges = np.unique(np.round(edges).astype(np.int64))
    if edges[0] != 1:
        edges = np.insert(edges, 0, 1)
    if edges[-1] <= fmax:
        edges = np.append(edges, fmax + 1)
    return edges


def bin_stats(f: np.ndarray, r: np.ndarray, s1: np.ndarray, edges: np.ndarray):
    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, len(edges) - 2)
    out = []
    qs = [10, 25, 50, 75, 90]
    for b in range(len(edges) - 1):
        m = idx == b
        n_ctx = int(m.sum())
        if n_ctx == 0:
            continue
        fb, rb, mb = f[m], r[m], s1[m] / f[m]
        rec = {
            "f_lo": int(edges[b]),
            "f_hi": int(edges[b + 1]),
            "contexts": n_ctx,
            "occurrences": int(fb.sum()),
            "mean_f": float(fb.mean()),
            "median_f": float(np.median(fb)),
            "mean_r": float(rb.mean()),
            "median_r": float(np.median(rb)),
            **{f"r_p{q}": float(np.percentile(rb, q)) for q in qs},
            "mean_mgt_missing": float(mb.mean()),
            "occ_frac_mgt_lt_0.01": float((mb < 0.01).mean()),
            "occ_frac_mgt_lt_0.05": float((mb < 0.05).mean()),
            "occ_frac_mgt_lt_0.10": float((mb < 0.10).mean()),
            "w_occ_frac_mgt_lt_0.01": float((mb < 0.01).dot(fb) / fb.sum()),
            "w_occ_frac_mgt_lt_0.05": float((mb < 0.05).dot(fb) / fb.sum()),
            "w_occ_frac_mgt_lt_0.10": float((mb < 0.10).dot(fb) / fb.sum()),
        }
        out.append(rec)
    return out


def crosscheck_against_npz(ctx_keys: np.ndarray, f: np.ndarray,
                           npz_path: str, branch: str, vocab: int):
    """Compare recounted f with data/freq_index.npz (raw-concatenation index)."""
    d = np.load(npz_path)
    nk = "bigram_keys" if branch == "bigram" else "trigram_keys"
    nc = "bigram_counts" if branch == "bigram" else "trigram_counts"
    ref_keys, ref_counts = d[nk], d[nc]
    order = np.argsort(ref_keys)
    ref_keys, ref_counts = ref_keys[order], ref_counts[order]
    pos = np.searchsorted(ref_keys, ctx_keys)
    pos_clip = np.clip(pos, 0, len(ref_keys) - 1)
    matched = ref_keys[pos_clip] == ctx_keys
    diff = f[matched] - ref_counts[pos_clip[matched]]
    return {
        "npz_path": os.path.basename(npz_path),
        "npz_entries": int(len(ref_keys)),
        "recounted_contexts": int(len(ctx_keys)),
        "matched_contexts": int(matched.sum()),
        "total_count_recounted": int(f.sum()),
        "total_count_npz_matched": int(ref_counts[pos_clip[matched]].sum()),
        "contexts_with_diff": int((diff != 0).sum()),
        "max_abs_diff": int(np.abs(diff).max()) if len(diff) else 0,
        "note": ("freq_index.npz was built by raw concatenation (build_from_shards) "
                 "and includes rare cross-chunk pairs; the recount above uses "
                 "build_from_chunks chunk semantics, matching training exactly."),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--shard", default="data/tokenized/shard_00001.bin")
    ap.add_argument("--freq_index", default="data/freq_index.npz")
    ap.add_argument("--vocab_size", type=int, default=8192)
    ap.add_argument("--chunk_size", type=int, default=2049,
                    help="sequence_len + 1")
    ap.add_argument("--out", default="data/runs_fixed/corpus_rbar_freq_v1_fixed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    buf = np.memmap(args.shard, dtype=np.uint16, mode="r")
    n_chunks = len(buf) // args.chunk_size
    print(f"[x3] shard={args.shard} md5={md5_of(args.shard)}")
    print(f"[x3] tokens={len(buf)} chunks={n_chunks} "
          f"(drop tail {len(buf) - n_chunks * args.chunk_size})")
    tokens = np.array(buf[: n_chunks * args.chunk_size], dtype=np.int64)

    b_keys, t_keys, targets = context_and_target_arrays(
        tokens, args.chunk_size, args.vocab_size)
    assert len(b_keys) == len(t_keys) == len(targets) == n_chunks * (args.chunk_size - 1)

    summary = {
        "run_id": "corpus_rbar_freq_v1",
        "spec": "docs/experiment-log.md section 29",
        "shard": os.path.abspath(args.shard),
        "shard_md5": md5_of(args.shard),
        "freq_index_md5": md5_of(args.freq_index),
        "vocab_size": args.vocab_size,
        "chunk_size": args.chunk_size,
        "n_chunks": int(n_chunks),
        "n_tokens_used": int(n_chunks * args.chunk_size),
        "n_positions": int(len(b_keys)),
        "definition": {
            "f": "context occurrence count in one epoch of shard 1 (chunk semantics)",
            "r": "distinct continuation tokens observed at those occurrences",
            "s1": "continuations observed exactly once",
            "mgt_missing": "s1 / f (Good-Turing missing mass)",
            "analytic_region_proxy": ("fraction of contexts (and f-weighted "
                                      "occurrences) with mgt_missing < eps"),
        },
    }

    for branch, keys in (("bigram", b_keys), ("trigram", t_keys)):
        print(f"[x3] {branch}: computing per-context stats ...")
        ctx_keys, f, r, s1 = per_context_stats(keys, targets, args.vocab_size)
        print(f"[x3] {branch}: {len(ctx_keys)} contexts, f_max={f.max()}, "
              f"mean_r={r.mean():.3f}")
        np.savez_compressed(
            os.path.join(args.out, f"rbar_by_exact_f_{branch}.npz"),
            f_values=f, ctx_keys=ctx_keys, r=r, s1=s1)
        edges = log_bin_edges(int(f.max()))
        bins = bin_stats(f, r, s1, edges)
        with open(os.path.join(args.out, f"rbar_bins_{branch}.json"), "w") as fh:
            json.dump({"branch": branch, "bins": bins}, fh, indent=2)
        cc = crosscheck_against_npz(ctx_keys, f, args.freq_index, branch,
                                    args.vocab_size)
        summary[f"crosscheck_{branch}"] = cc
        print(f"[x3] {branch}: npz cross-check -> "
              f"{cc['contexts_with_diff']} differing contexts "
              f"(max |diff| = {cc['max_abs_diff']})")

    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[x3] wrote results to {args.out}")


if __name__ == "__main__":
    main()
