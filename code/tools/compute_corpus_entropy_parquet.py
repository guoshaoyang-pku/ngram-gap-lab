#!/usr/bin/env python3
"""Bigram/trigram entropy of a parquet text corpus using the fixed tokenizer.

Tokenizes every document (with BOS prepended, matching the dataloader) and
reuses compute_ngram_entropy.split_entropy.  The "doc" variant excludes
n-grams containing BOS (within-document), the "full" variant is the
BOS-separated packed stream.

Usage:
  python3 compute_corpus_entropy_parquet.py SHARD.parquet [SHARD2.parquet ...] \
      --tokenizer-pkl TOKENIZER.pkl --vocab 8192 -o out.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq

from compute_ngram_entropy import split_entropy

BOS_TOKEN = "<|reserved_0|>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parquet_files", nargs="+")
    ap.add_argument("--tokenizer-pkl", required=True)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    with open(args.tokenizer_pkl, "rb") as f:
        enc = pickle.load(f)
    bos = enc.encode_single_token(BOS_TOKEN)
    print(f"[corpus] vocab={enc.n_vocab} bos={bos}", file=sys.stderr)

    parts: list[np.ndarray] = []
    n_docs = 0
    for path in args.parquet_files:
        tbl = pq.read_table(path)
        texts = tbl.column("text").to_pylist()
        for i in range(0, len(texts), 512):
            ids_batch = enc.encode_ordinary_batch(texts[i : i + 512], num_threads=8)
            for ids in ids_batch:
                parts.append(np.array([bos] + ids, dtype=np.int32))
                n_docs += 1
        print(f"[corpus] {path}: {len(texts):,} docs done", file=sys.stderr, flush=True)

    tokens = np.concatenate(parts)
    print(f"[corpus] {n_docs:,} docs, {len(tokens):,} tokens", file=sys.stderr, flush=True)

    result = split_entropy(tokens, args.vocab, bos)
    result["n_docs"] = n_docs
    out = {"vocab": args.vocab, "bos": int(bos), "splits": {"train": result}}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"[corpus] wrote {args.output}", file=sys.stderr)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
