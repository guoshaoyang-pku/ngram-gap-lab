from __future__ import annotations

from collections import Counter

import torch

from trainer import ExactNgramIndex


def test_exact_index_distinguishes_colliding_hash_rows(tmp_path):
    import json
    import numpy as np

    data_dir = tmp_path
    (data_dir / "metadata.json").write_text(
        json.dumps(
            {
                "vocab_size": 10,
                "order": 3,
                "frequency_definition": "exact_train_epoch_context_count",
            }
        )
    )
    keys = np.asarray([12, 123], dtype=np.int64)
    counts = np.asarray([7, 2], dtype=np.int64)
    np.savez(data_dir / "exact_ngram_counts.npz", keys=keys, counts=counts)

    index = ExactNgramIndex(data_dir, torch.device("cpu"), order=3)
    result = index.lookup_frequency(
        torch.tensor([[0]]),
        torch.tensor([[1]]),
        torch.tensor([[2]]),
    )
    assert result.item() == 7
    result = index.lookup_frequency(
        torch.tensor([[1]]),
        torch.tensor([[2]]),
        torch.tensor([[3]]),
    )
    assert result.item() == 2


def test_exact_index_returns_zero_for_novel_validation_context(tmp_path):
    import json
    import numpy as np

    data_dir = tmp_path
    (data_dir / "metadata.json").write_text(
        json.dumps(
            {
                "vocab_size": 10,
                "order": 3,
                "frequency_definition": "exact_train_epoch_context_count",
            }
        )
    )
    np.savez(
        data_dir / "exact_ngram_counts.npz",
        keys=np.asarray([123], dtype=np.int64),
        counts=np.asarray([2], dtype=np.int64),
    )
    index = ExactNgramIndex(data_dir, torch.device("cpu"), order=3)
    result = index.lookup_frequency(
        torch.tensor([[9]]),
        torch.tensor([[8]]),
        torch.tensor([[7]]),
    )
    assert result.item() == 0