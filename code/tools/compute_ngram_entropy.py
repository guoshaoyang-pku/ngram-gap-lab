#!/usr/bin/env python3
"""Empirical bigram / trigram entropy over space-separated token-id files.

Computes MLE (maximum-likelihood) entropies of a token corpus:

  H1         = H(X)                          unigram entropy
  H2_joint   = H(X, Y)                       bigram joint
  H2_cond    = H(Y | X)                      bigram conditional (bigram entropy)
  H3_joint   = H(X, Y, Z)                    trigram joint
  H3_cond    = H(Z | X, Y)                   trigram conditional (trigram entropy)

All nats (base e) and bits (base 2) are reported, together with perplexity
exp(H_cond) / 2^H_cond and distinct-context/pair/triple counts.

For each split, two variants are reported:
  - "full":  the raw token sequence (transitions that cross the SEP document
             separator are included);
  - "doc":   only n-grams whose tokens are all != SEP (document-internal).

Usage:
  python3 compute_ngram_entropy.py --vocab 8192 --sep 8191 \
      TRAIN_TOKENS.txt [VAL_TOKENS.txt ...] [-o out.json]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load_tokens(path: str, dtype=np.int32) -> np.ndarray:
    """Stream a space-separated ASCII int file into one int32 array."""
    chunks = []
    carry = b""
    with open(path, "rb") as f:
        while True:
            data = f.read(64 << 20)  # 64 MB
            if not data:
                break
            data = carry + data
            sp = data.rfind(b" ")
            if sp < 0:
                carry = data
                continue
            carry = data[sp + 1 :]
            chunk = np.fromstring(data[:sp], dtype=dtype, sep=" ")
            chunks.append(chunk)
    if carry.strip():
        chunks.append(np.fromstring(carry, dtype=dtype, sep=" "))
    if not chunks:
        return np.empty(0, dtype=dtype)
    return np.concatenate(chunks)


def _cond_entropy(u: np.ndarray, c: np.ndarray, ctx_sum: np.ndarray,
                  div: int, n: int) -> tuple[float, float]:
    """Return (joint_entropy, conditional_entropy) for one n-gram order."""
    c = c.astype(np.float64)
    p = c / n
    joint = -float(np.sum(p * np.log(p)))
    ctx = ctx_sum[u // div]          # total count of the context of each n-gram
    cond = -float(np.sum(p * np.log(c / ctx)))
    return joint, cond


def split_entropy(tokens: np.ndarray, vocab: int, sep: int) -> dict:
    n = len(tokens)
    out: dict = {"n_tokens": int(n)}

    # ---- unigram -----------------------------------------------------
    u1 = np.bincount(tokens, minlength=vocab)
    nz1 = u1[u1 > 0].astype(np.float64)
    p1 = nz1 / n
    h1 = -float(np.sum(p1 * np.log(p1)))

    # ---- bigram (key = prev * V + cur) --------------------------------
    prev = tokens[:-1].astype(np.int64)
    cur = tokens[1:]
    k2 = prev * vocab + cur
    u2, c2 = np.unique(k2, return_counts=True)
    ctx2 = np.bincount(u2 // vocab, weights=c2.astype(np.float64),
                       minlength=vocab)
    h2_joint, h2_cond = _cond_entropy(u2, c2, ctx2, vocab, n)

    # ---- trigram (key = (prev2 * V + prev) * V + cur) -----------------
    k3 = (tokens[:-2].astype(np.int64) * vocab + tokens[1:-1]) * vocab + \
        tokens[2:]
    u3, c3 = np.unique(k3, return_counts=True)
    ctx3 = np.bincount(u3 // vocab, weights=c3.astype(np.float64),
                       minlength=vocab * vocab)
    h3_joint, h3_cond = _cond_entropy(u3, c3, ctx3, vocab, n)

    out.update({
        "unigram_entropy": h1,
        "bigram_joint": h2_joint,
        "bigram_cond": h2_cond,
        "trigram_joint": h3_joint,
        "trigram_cond": h3_cond,
        "n_distinct_bigrams": int(len(u2)),
        "n_distinct_bigram_contexts": int(np.count_nonzero(ctx2)),
        "n_distinct_trigrams": int(len(u3)),
        "n_distinct_trigram_contexts": int(np.count_nonzero(ctx3)),
        "sep_count": int(np.count_nonzero(tokens == sep)),
    })

    # ---- document-internal variant (no SEP in any position) ----------
    m1 = tokens != sep
    u1b = np.bincount(tokens[m1], minlength=vocab)
    nz1b = u1b[u1b > 0].astype(np.float64)
    n1b = int(np.sum(nz1b))
    p1b = nz1b / n1b
    h1b = -float(np.sum(p1b * np.log(p1b)))

    mb2 = (prev != sep) & (cur != sep)
    k2b = k2[mb2]
    u2b, c2b = np.unique(k2b, return_counts=True)
    ctx2b = np.bincount(u2b // vocab, weights=c2b.astype(np.float64),
                        minlength=vocab)
    n2b = int(np.sum(c2b))
    h2b_joint, h2b_cond = _cond_entropy(u2b, c2b, ctx2b, vocab, n2b)

    mb3 = (tokens[:-2] != sep) & (tokens[1:-1] != sep) & (tokens[2:] != sep)
    k3b = k3[mb3]
    u3b, c3b = np.unique(k3b, return_counts=True)
    ctx3b = np.bincount(u3b // vocab, weights=c3b.astype(np.float64),
                        minlength=vocab * vocab)
    n3b = int(np.sum(c3b))
    h3b_joint, h3b_cond = _cond_entropy(u3b, c3b, ctx3b, vocab, n3b)

    out["doc"] = {
        "n_unigram_tokens": n1b,
        "n_bigram_ngrams": n2b,
        "n_trigram_ngrams": n3b,
        "unigram_entropy": h1b,
        "bigram_joint": h2b_joint,
        "bigram_cond": h2b_cond,
        "trigram_joint": h3b_joint,
        "trigram_cond": h3b_cond,
        "n_distinct_bigrams": int(len(u2b)),
        "n_distinct_bigram_contexts": int(np.count_nonzero(ctx2b)),
        "n_distinct_trigrams": int(len(u3b)),
        "n_distinct_trigram_contexts": int(np.count_nonzero(ctx3b)),
    }

    # ---- bits / perplexity / reference max ----------------------------
    ln2 = np.log(2.0)
    for variant in ("full", "doc"):
        d = out if variant == "full" else out["doc"]
        d["bigram_cond_bits"] = d["bigram_cond"] / ln2
        d["trigram_cond_bits"] = d["trigram_cond"] / ln2
        d["bigram_cond_perplexity"] = float(np.exp(d["bigram_cond"]))
        d["trigram_cond_perplexity"] = float(np.exp(d["trigram_cond"]))
    out["max_entropy_nats"] = float(np.log(vocab))
    out["max_entropy_bits"] = float(np.log2(vocab))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("token_files", nargs="+", help="space-separated token id files")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--sep", type=int, default=8191,
                    help="document-separator token id (used for the doc variant)")
    ap.add_argument("-o", "--output", default=None, help="write results JSON")
    args = ap.parse_args()

    results: dict = {
        "vocab": args.vocab,
        "sep": args.sep,
        "splits": {},
    }
    for path in args.token_files:
        name = path.rsplit("/", 1)[-1].replace("_tokens.txt", "")
        print(f"[entropy] loading {path} ...", file=sys.stderr, flush=True)
        tokens = load_tokens(path)
        print(f"[entropy] {len(tokens):,} tokens", file=sys.stderr, flush=True)
        results["splits"][name] = split_entropy(tokens, args.vocab, args.sep)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"[entropy] wrote {args.output}", file=sys.stderr)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
