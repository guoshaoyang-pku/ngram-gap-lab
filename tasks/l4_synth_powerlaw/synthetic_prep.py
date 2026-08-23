#!/usr/bin/env python3
"""Make a synthetic_transition generator output compatible with the ngram5 trainer.

The generator writes ``metadata.json``/``exact_ngram_counts.npz`` (context_matrix_v1).
The ngram5 trainer additionally needs ``meta.json`` and a few metadata fields
(``vocab_size``, ``n_nonempty_buckets``, ``total_contexts``). This script adds them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    d = Path(args.data_dir)
    md_path = d / "metadata.json"
    md = json.loads(md_path.read_text())

    vocab = int(md["vocab"])
    order = int(md["order"])
    sep = int(md["sep_token"])
    block_len = int(md["block_len"])

    md.setdefault("vocab_size", vocab)
    md.setdefault("n_nonempty_buckets", 0)
    md.setdefault("hash_bucket_occupancy_diagnostic", False)
    md["frequency_definition"] = "exact_train_epoch_context_count"
    md["frequency_source_split"] = "train"
    md["frequency_key_type"] = "exact_context"
    md["frequency_index_format"] = "context_matrix_v1"

    with np.load(d / "exact_ngram_counts.npz") as z:
        assert {"contexts", "counts"} <= set(z.keys()), (
            "exact_ngram_counts.npz must contain contexts+counts"
        )
        ctx = z["contexts"]
        counts = z["counts"]
        assert ctx.ndim == 2 and ctx.shape[1] == order, (
            f"contexts shape {ctx.shape} != (N,{order})"
        )
        total_contexts = int(counts.sum())
    md["total_contexts"] = total_contexts

    meta_json = {
        "schema_version": 2,
        "vocab": vocab,
        "vocab_size": vocab,
        "sep_token": sep,
        "order": order,
        "context_len": int(md.get("context_len", order)),
        "block_len": block_len,
        "frequency_definition": md["frequency_definition"],
        "frequency_source_split": md["frequency_source_split"],
        "frequency_key_type": md["frequency_key_type"],
        "hash_bucket_occupancy_diagnostic": False,
        "train_tokens": int(md["train_tokens"]),
        "val_tokens": int(md["val_tokens"]),
        "total_contexts": total_contexts,
        "n_distinct_exact_contexts": int(md["n_distinct_exact_contexts"]),
        "n_nonempty_buckets": 0,
    }
    (d / "meta.json").write_text(json.dumps(meta_json, indent=2) + "\n")
    md_path.write_text(json.dumps(md, indent=2, sort_keys=True) + "\n")
    print(
        f"[prep] {d}: meta.json written; distinct={ctx.shape[0]} "
        f"total_occ={total_contexts} order={order} block_len={block_len}"
    )


if __name__ == "__main__":
    main()
