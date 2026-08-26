"""Lightweight unit tests for ngram5_freq_gap/data_gen.py.

These tests exercise the pure-Python pieces (hashing, factor computation,
Poisson replication, real-context block emission, fivegram_counts.npz) on a
synthetic token corpus, without requiring the real BPE tokenizer or parquet
shards.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import random
from collections import Counter
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
_MOD = _HERE.parent / "data_gen.py"
spec = importlib.util.spec_from_file_location("ngram5_data_gen", _MOD)
data_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_gen)

sys.path.insert(0, str(_HERE.parent))
from hash_utils import hash5_bucket_tensor


# ---------------------------------------------------------------------------
# hash5
# ---------------------------------------------------------------------------

def test_hash5_is_deterministic_and_bucketed():
    ctx = [10, 20, 30, 40, 50]
    h1 = data_gen.hash5(ctx)
    h2 = data_gen.hash5(ctx)
    assert h1 == h2
    b = h1 % 1_000_000
    assert 0 <= b < 1_000_000


def test_hash5_distinct_contexts_usually_distinct_buckets():
    """Sanity: distinct 5-grams should usually hash to distinct buckets
    (collisions allowed but not the common case)."""
    buckets = set()
    for i in range(1000):
        ctx = [i, i + 1, i + 2, i + 3, i + 4]
        buckets.add(data_gen.hash5(ctx) % 1_000_000)
    # expect close to 1000 distinct buckets
    assert len(buckets) > 900


def test_tensor_hash_matches_python_without_int64_overflow():
    rng = random.Random(20260805)
    contexts = [[rng.randrange(8192) for _ in range(5)] for _ in range(1000)]
    columns = [torch.tensor([ctx[i] for ctx in contexts]) for i in range(5)]
    actual = hash5_bucket_tensor(*columns, bucket_count=1_000_000).tolist()
    expected = [data_gen.hash5(ctx) % 1_000_000 for ctx in contexts]
    assert actual == expected


# ---------------------------------------------------------------------------
# compute_factors
# ---------------------------------------------------------------------------

def test_factors_alpha_zero_is_one():
    bucket_r = [0, 1, 5, 50, 500]
    f = data_gen.compute_factors(bucket_r, alpha=0.0, r_ref=10.0,
                                 k_min=0.25, k_max=8.0)
    assert f == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_factors_upsample_low_freq_downsample_high_freq():
    bucket_r = [0, 1, 5, 50, 500]
    f = data_gen.compute_factors(bucket_r, alpha=0.5, r_ref=10.0,
                                 k_min=0.25, k_max=8.0)
    assert f[1] > 1.0   # r=1 < r_ref=10 -> up
    assert f[4] < 1.0   # r=500 > r_ref=10 -> down
    assert f[0] == 1.0  # r=0 -> no data
    assert abs(f[1] - (10.0 ** 0.5)) < 1e-9
    assert f[4] == 0.25  # clipped


def test_factors_clipped_at_k_max():
    bucket_r = [0, 1]
    f = data_gen.compute_factors(bucket_r, alpha=0.5, r_ref=1000.0,
                                 k_min=0.25, k_max=8.0)
    assert f[1] == 8.0


# ---------------------------------------------------------------------------
# _poisson_draw
# ---------------------------------------------------------------------------

def test_poisson_draw_mean_converges():
    """Over many draws, the sample mean should be close to lambda."""
    rng = random.Random(0)
    lam = 2.5
    draws = [data_gen._poisson_draw(lam, rng) for _ in range(20000)]
    mean = sum(draws) / len(draws)
    assert abs(mean - lam) < 0.1


def test_poisson_draw_zero_lambda_returns_zero():
    assert data_gen._poisson_draw(0.0, random.Random(0)) == 0


def test_poisson_draw_large_lambda_uses_normal_approx():
    """lam >= 30 falls back to normal approx; just check it returns >= 0."""
    rng = random.Random(0)
    for _ in range(100):
        v = data_gen._poisson_draw(100.0, rng)
        assert v >= 0


# ---------------------------------------------------------------------------
# sample_splits (target counts only)
# ---------------------------------------------------------------------------

def test_sample_splits_target_counts_match_factor():
    bucket_r = [0, 2, 1000]
    factors = [1.0, 2.0, 0.5]
    hist = {1: Counter({11: 1, 22: 1}), 2: Counter({33: 800, 44: 200})}
    splits = data_gen.sample_splits(hist, bucket_r, factors,
                                    f_train=0.8, f_val=0.2,
                                    rng_train=random.Random(1),
                                    rng_val=random.Random(2))
    # bucket 1: r=2, k=2 -> n_train_target = round(2*0.8*2) = 3
    assert splits[1]["n_train_target"] == 3
    assert splits[1]["n_val_target"] == round(2 * 0.2)  # = 0
    # bucket 2: r=1000, k=0.5 -> n_train_target = round(1000*0.8*0.5) = 400
    assert splits[2]["n_train_target"] == 400
    assert splits[2]["n_val_target"] == 200


# ---------------------------------------------------------------------------
# scan_and_emit (real 5-gram context + Poisson replication)
# ---------------------------------------------------------------------------

def test_scan_and_emit_emits_real_contexts(monkeypatch):
    """Blocks emitted by scan_and_emit carry the real 5-gram context."""
    docs = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 9, 6, 7, 8, 9, 10, 11],
    ]
    monkeypatch.setattr(data_gen, "iter_train_token_streams",
                        lambda tokenizer, max_tokens: iter(docs))
    hist, bucket_r = data_gen.scan_histogram(None, bucket_count=100, max_tokens=None, order=3)
    factors = data_gen.compute_factors(bucket_r, alpha=0.0, r_ref=1.0,
                                       k_min=1.0, k_max=1.0)
    splits = data_gen.sample_splits(hist, bucket_r, factors,
                                    f_train=1.0, f_val=0.0,
                                    rng_train=random.Random(1),
                                    rng_val=random.Random(2))
    tokens, meta = data_gen.scan_and_emit(
        None, splits, 100, role="train", f_train=1.0, f_val=0.0,
        doc_len=64, sep_token=9999, max_tokens=None, rng=random.Random(7), order=3)
    order = 3
    block_len = order + 2  # 5
    assert len(tokens) % block_len == 0 or len(tokens) % 64 == 0
    # verify at least one block's context matches a real trigram from docs
    real_contexts = set()
    for tk in docs:
        for i in range(order, len(tk)):
            real_contexts.add(tuple(tk[i - order:i]))
    found_real = False
    for m in meta:
        ctx = tuple(tokens[m["block_start"]:m["block_start"] + order])
        if ctx in real_contexts:
            found_real = True
            break
    assert found_real, "no emitted block carries a real trigram context"


def test_scan_and_emit_upsampling_produces_more_blocks(monkeypatch):
    """With k > 1 (up-sampling), train should emit more blocks than val."""
    docs = [[0, 1, 2, 3, 4, 5] * 20]  # one context [0,1,2,3,4]->5, r=20
    monkeypatch.setattr(data_gen, "iter_train_token_streams",
                        lambda tokenizer, max_tokens: iter(docs))
    hist, bucket_r = data_gen.scan_histogram(None, bucket_count=100, max_tokens=None, order=3)
    # alpha > 0 with r_ref >> r => strong up-sampling
    factors = data_gen.compute_factors(bucket_r, alpha=0.5, r_ref=100.0,
                                       k_min=1.0, k_max=8.0)
    splits = data_gen.sample_splits(hist, bucket_r, factors,
                                    f_train=0.8, f_val=0.2,
                                    rng_train=random.Random(1),
                                    rng_val=random.Random(2))
    tr_tokens, tr_meta = data_gen.scan_and_emit(
        None, splits, 100, role="train", f_train=0.8, f_val=0.2, order=3,
        doc_len=64, sep_token=9999, max_tokens=None, rng=random.Random(7))
    va_tokens, va_meta = data_gen.scan_and_emit(
        None, splits, 100, role="val", f_train=0.8, f_val=0.2, order=3,
        doc_len=64, sep_token=9999, max_tokens=None, rng=random.Random(8))
    # train should have (on expectation) more blocks than val because k>1
    assert len(tr_meta) >= len(va_meta)


def test_scan_and_emit_alpha_zero_train_val_independent(monkeypatch):
    """With alpha=0 (k=1), train and val are independent Poisson(0.8)/(0.2)
    draws over the same corpus — the coincidental-gap source."""
    docs = [[0, 1, 2, 3, 4, 5] * 50]  # r=50 for context [0,1,2,3,4]
    monkeypatch.setattr(data_gen, "iter_train_token_streams",
                        lambda tokenizer, max_tokens: iter(docs))
    hist, bucket_r = data_gen.scan_histogram(None, bucket_count=100, max_tokens=None, order=3)
    factors = data_gen.compute_factors(bucket_r, alpha=0.0, r_ref=1.0,
                                       k_min=1.0, k_max=1.0)
    splits = data_gen.sample_splits(hist, bucket_r, factors,
                                    f_train=0.8, f_val=0.2,
                                    rng_train=random.Random(1),
                                    rng_val=random.Random(2))
    tr_tokens, tr_meta = data_gen.scan_and_emit(
        None, splits, 100, role="train", f_train=0.8, f_val=0.2, order=3,
        doc_len=64, sep_token=9999, max_tokens=None, rng=random.Random(7))
    va_tokens, va_meta = data_gen.scan_and_emit(
        None, splits, 100, role="val", f_train=0.8, f_val=0.2, order=3,
        doc_len=64, sep_token=9999, max_tokens=None, rng=random.Random(8))
    # both should have emitted blocks (Poisson means 40 and 10)
    assert len(tr_meta) > 0
    assert len(va_meta) > 0
    # train should be roughly 4x val (mean ratio)
    assert len(tr_meta) > len(va_meta)


# ---------------------------------------------------------------------------
# scan_histogram on synthetic tokens
# ---------------------------------------------------------------------------

def test_scan_histogram_on_synthetic_tokens(monkeypatch):
    docs = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ]
    monkeypatch.setattr(data_gen, "iter_train_token_streams",
                        lambda tokenizer, max_tokens: iter(docs))
    hist, bucket_r = data_gen.scan_histogram(None, bucket_count=100, max_tokens=None, order=3)
    assert sum(bucket_r) == 10  # 5 trigram contexts × 2 docs (order=3, 8 tokens/doc)
    b = data_gen.hash_n([0, 1, 2]) % 100  # first trigram context
    assert hist[b][3] == 2  # appears twice, next=3


def test_exact_histogram_ignores_hash_collisions(monkeypatch):
    docs = [[0, 1, 2, 9, 3, 4, 5]]
    monkeypatch.setattr(
        data_gen,
        "iter_train_token_streams",
        lambda tokenizer, max_tokens: iter(docs),
    )
    exact = data_gen.scan_exact_histogram(None, max_tokens=None, order=3)
    assert exact[(0, 1, 2)][9] == 1
    assert exact[(1, 2, 9)][3] == 1
    assert sum(sum(hist.values()) for hist in exact.values()) == 4


def test_exact_index_npz_matches_bruteforce_counter(tmp_path):
    import numpy as np

    exact = {
        (1, 2, 3): Counter({4: 2}),
        (2, 3, 4): Counter({5: 1}),
    }
    data_gen._write_exact_counts_npz(tmp_path, exact, vocab=100)
    with np.load(tmp_path / "exact_ngram_counts.npz") as data:
        keys = data["keys"].tolist()
        counts = data["counts"].tolist()
    expected = sorted(
        (data_gen._encode_context(context, 100), sum(hist.values()))
        for context, hist in exact.items()
    )
    assert list(zip(keys, counts)) == expected


def test_exact_index_npz_uses_context_matrix_for_order5(tmp_path):
    import numpy as np

    exact = {
        (1, 2, 3, 4, 5): Counter({6: 2}),
        (2, 3, 4, 5, 6): Counter({7: 1}),
    }
    data_gen._write_exact_counts_npz(tmp_path, exact, vocab=8192)
    with np.load(tmp_path / "exact_ngram_counts.npz") as data:
        assert "contexts" in data
        assert "keys" not in data
        assert data["contexts"].shape == (2, 5)
        assert data["counts"].tolist() == [2, 1]


# ---------------------------------------------------------------------------
# Full generate() on synthetic corpus
# ---------------------------------------------------------------------------

def test_generate_end_to_end_on_synthetic(monkeypatch, tmp_path):
    docs = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 9, 6, 7, 8, 9, 10, 11],
    ]

    class FakeTokenizer:
        def get_bos_token_id(self):
            return 0
        def get_vocab_size(self):
            return 8192

    class FakeLib:
        Tokenizer = type("T", (), {
            "from_directory": staticmethod(lambda d=None: FakeTokenizer()),
        })
        @staticmethod
        def describe_shard_selection():
            return {"fake": True}
        @staticmethod
        def split_parquet_files(split):
            return []
        MAX_SEQ_LEN = 2048

    monkeypatch.setattr(data_gen, "_load_upstream_lib", lambda: FakeLib)
    monkeypatch.setattr(data_gen, "iter_train_token_streams",
                        lambda tokenizer, max_tokens: iter(docs))

    out = tmp_path / "alpha0.5"
    data_gen.generate(
        out, alpha=0.5, bucket_count=1000,
        f_train=0.8, f_val=0.2, k_min=0.25, k_max=8.0,
        r_ref_mode="median", r_ref_fixed=None,
        dataset_seed=123, doc_len=64, max_tokens=None,
        tokenizer_dir=None,
    )

    assert (out / "meta.json").exists()
    assert (out / "contexts.json").exists()
    assert (out / "train_tokens.txt").exists()
    assert (out / "val_tokens.txt").exists()
    assert (out / "fivegram_counts.npz").exists()
    assert (out / "exact_ngram_counts.npz").exists()
    assert (out / "exact_ngram_contexts.json").exists()
    assert (out / "metadata.json").exists()

    meta = json.loads((out / "meta.json").read_text())
    assert meta["order"] == 5
    assert meta["vocab"] == 8192
    assert meta["block_len"] == 7
    assert meta["n_distinct_exact_contexts"] > 0
    assert meta["train_tokens"] > 0

    md = json.loads((out / "metadata.json").read_text())
    assert md["vocab_size"] == 8192
    assert md["order"] == 5

    # exact_ngram_counts.npz loads and is sorted
    import numpy as np
    z = np.load(out / "exact_ngram_counts.npz")
    contexts = z["contexts"]
    counts = z["counts"]
    assert contexts.shape == (meta["n_distinct_exact_contexts"], 5)
    assert counts.sum() == meta["total_contexts"]

    # token streams parse as ints and are multiples of doc_len
    tr = [int(x) for x in (out / "train_tokens.txt").read_text().split()]
    va = [int(x) for x in (out / "val_tokens.txt").read_text().split()]
    assert len(tr) % 64 == 0
    assert len(va) % 64 == 0
    assert all(0 <= t < 8192 for t in tr)

    assert meta["frequency_definition"] == "exact_train_epoch_context_count"
    # contexts.json has exact-context field names
    ctx = json.loads((out / "contexts.json").read_text())
    for context, d in ctx.items():
        assert "n_train_target" in d
        assert "n_val_target" in d
        assert "n_train_actual" in d
        assert "n_val_actual" in d
        assert "next_hist_topk" in d
        assert d["frequency_definition"] == "exact_train_epoch_context_count"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
