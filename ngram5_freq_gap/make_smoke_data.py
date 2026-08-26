"""Generate a tiny synthetic 5-gram dataset on disk for trainer smoke tests.

Writes a dataset into ngram5_freq_gap/data/smoke/ that mirrors what
data_gen.py would produce from real BPE tokens, but using a synthetic
token stream so we can test the trainer on CPU without the real tokenizer
  or parquet shards.  Uses the collision-free exact-context data path.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import importlib.util
spec = importlib.util.spec_from_file_location("data_gen", _HERE / "data_gen.py")
data_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_gen)

VOCAB = 8192
SEP = VOCAB - 1
BUCKET_COUNT = 2000
DOC_LEN = 128  # small for smoke


def _synthetic_token_stream():
    """Yield synthetic token docs that exercise several 5-gram buckets."""
    rng = random.Random(0)
    # 50 distinct 5-gram contexts, each repeated a controlled number of times.
    freqs = [2, 5, 10, 50, 200] * 10
    docs = []
    for r in freqs:
        ctx = [rng.randint(0, 255) for _ in range(5)]
        nxt = rng.randint(0, 999)
        # build a doc: ctx + nxt + SEP, repeated r times
        doc = []
        for _ in range(r):
            doc.extend(ctx)
            doc.append(nxt)
            doc.append(SEP)
        docs.append(doc)
    return docs


def main():
    out = _HERE / "data" / "smoke"
    out.mkdir(parents=True, exist_ok=True)

    docs = _synthetic_token_stream()
    # Monkeypatch iter_train_token_streams to yield our synthetic docs.
    data_gen.iter_train_token_streams = lambda tokenizer, max_tokens: iter(docs)

    exact_hist = data_gen.scan_exact_histogram(None, max_tokens=None, order=5)
    exact_counts = {
        context: sum(next_hist.values())
        for context, next_hist in exact_hist.items()
    }
    factors = data_gen.compute_context_factors(
        exact_counts, alpha=0.5, r_ref=10.0, k_min=0.25, k_max=8.0
    )
    splits = data_gen.sample_exact_splits(
        exact_hist, factors, f_train=0.8, f_val=0.2
    )
    train_tokens, train_meta = data_gen.scan_and_emit_exact(
        None, splits, role="train", f_train=0.8, f_val=0.2,
        doc_len=DOC_LEN, sep_token=SEP, max_tokens=None,
        rng=random.Random(7), order=5)
    val_tokens, val_meta = data_gen.scan_and_emit_exact(
        None, splits, role="val", f_train=0.8, f_val=0.2,
        doc_len=DOC_LEN, sep_token=SEP, max_tokens=None,
        rng=random.Random(8), order=5)

    (out / "train_tokens.txt").write_text(" ".join(map(str, train_tokens)) + "\n")
    (out / "val_tokens.txt").write_text(" ".join(map(str, val_tokens)) + "\n")

    # contexts.json
    contexts = {}
    actual_n_train = Counter()
    actual_n_val = Counter()
    for m in train_meta:
        actual_n_train[tuple(m["context"])] += 1
    for m in val_meta:
        actual_n_val[tuple(m["context"])] += 1
    for context, d in splits.items():
        contexts[" ".join(map(str, context))] = {
            "r": d["r"],
            "k": d["k"],
            "n_train_target": d["n_train_target"],
            "n_val_target": d["n_val_target"],
            "n_train_actual": actual_n_train.get(context, 0),
            "n_val_actual": actual_n_val.get(context, 0),
            "next_hist_topk": d["next_hist_topk"],
            "n_distinct_next": d["n_distinct_next"],
            "frequency_definition": "exact_train_epoch_context_count",
        }
    (out / "contexts.json").write_text(
        json.dumps(contexts, sort_keys=True, separators=(",", ":")) + "\n")

    # exact_ngram_counts.npz + metadata.json
    data_gen._write_exact_counts_npz(out, exact_hist, VOCAB)
    (out / "metadata.json").write_text(json.dumps({
        "vocab_size": VOCAB,
        "order": 5,
        "bucket_count": BUCKET_COUNT,
        "n_contexts": int(sum(exact_counts.values())),
        "n_distinct_contexts": int(len(exact_hist)),
        "frequency_definition": "exact_train_epoch_context_count",
    }, indent=2, sort_keys=True) + "\n")

    # meta.json
    nonempty = sorted(r for r in exact_counts.values() if r > 0)
    meta = {
        "schema_version": 1,
        "order": 5,
        "context_len": 5,
        "block_len": 7,
        "vocab": VOCAB,
        "sep_token": SEP,
        "doc_len": DOC_LEN,
        "bucket_count": BUCKET_COUNT,
        "alpha": 0.5,
        "r_ref": 10.0,
        "k_min": 0.25,
        "k_max": 8.0,
        "f_train": 0.8,
        "f_val": 0.2,
        "dataset_seed": 0,
        "max_tokens_scanned": None,
        "n_nonempty_buckets": 0,
        "n_distinct_exact_contexts": len(exact_hist),
        "total_contexts": sum(nonempty),
        "bucket_r_quantiles": {},
        "exact_context_frequency_quantiles": {"q0": nonempty[0], "q50": nonempty[len(nonempty)//2], "q100": nonempty[-1]},
        "eff_r_train_quantiles": {"q0": 0, "q50": 0, "q100": 0},
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_docs": len(train_tokens) // DOC_LEN,
        "val_docs": len(val_tokens) // DOC_LEN,
        "frequency_definition": "exact_train_epoch_context_count",
        "frequency_source_split": "train",
        "frequency_key_type": "exact_context",
        "loader_selection": {"smoke": True},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[smoke] wrote {out}")
    print(f"[smoke] train_tokens={len(train_tokens)} val_tokens={len(val_tokens)} "
          f"train_docs={meta['train_docs']} val_docs={meta['val_docs']}")


if __name__ == "__main__":
    main()
