import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from train import TokenizedShardDataset  # noqa: E402


class LogicalBatchOrderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        chunks = np.concatenate([
            np.full(3, chunk_id, dtype=np.uint16)
            for chunk_id in range(12)
        ])
        chunks.tofile(self.data_dir / "shard_00001.bin")

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, device_batch_size, mode, seed=101):
        return TokenizedShardDataset(
            str(self.data_dir), [1], sequence_len=2,
            device_batch_size=device_batch_size, seed=seed,
            train_order=mode, logical_batch_size=4,
        )

    def test_paired_modes_share_first_epoch_then_diverge(self):
        frozen = self.dataset(4, "frozen_permutation")
        reshuffle = self.dataset(4, "epoch_reshuffle")
        self.assertEqual(frozen.logical_batch_order(0), reshuffle.logical_batch_order(0))
        self.assertEqual(frozen.logical_batch_order(0), frozen.logical_batch_order(1))
        self.assertNotEqual(reshuffle.logical_batch_order(0), reshuffle.logical_batch_order(1))
        for epoch in range(3):
            self.assertEqual(sorted(reshuffle.logical_batch_order(epoch)), [0, 1, 2])

    def test_order_seed_is_reproducible_and_independent(self):
        first = self.dataset(4, "epoch_reshuffle", seed=101)
        second = self.dataset(4, "epoch_reshuffle", seed=101)
        third = self.dataset(4, "epoch_reshuffle", seed=202)
        self.assertEqual(first.logical_batch_order(2), second.logical_batch_order(2))
        self.assertNotEqual(first.logical_batch_order(2), third.logical_batch_order(2))

    def test_sequential_then_reshuffle_preserves_original_first_epoch(self):
        sequential = self.dataset(4, "sequential", seed=42)
        reshuffle = self.dataset(4, "sequential_then_reshuffle", seed=101)
        self.assertEqual(reshuffle.logical_batch_order(0), [0, 1, 2])
        self.assertEqual(
            reshuffle.logical_batch_order(0), sequential.logical_batch_order(0)
        )
        self.assertNotEqual(reshuffle.logical_batch_order(1), [0, 1, 2])
        self.assertNotEqual(
            reshuffle.logical_batch_order(1), reshuffle.logical_batch_order(2)
        )

    def test_device_microbatching_preserves_logical_stream(self):
        def first_epoch_stream(device_batch_size):
            dataset = self.dataset(device_batch_size, "epoch_reshuffle")
            iterator = dataset.iter_batches(torch.device("cpu"))
            batches = 12 // device_batch_size
            return [
                int(token)
                for _ in range(batches)
                for token in next(iterator)[0][:, 0]
            ]

        self.assertEqual(first_epoch_stream(4), first_epoch_stream(2))

    def test_logical_batch_geometry_is_strict(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            TokenizedShardDataset(
                str(self.data_dir), [1], sequence_len=2,
                device_batch_size=3, seed=101,
                train_order="epoch_reshuffle", logical_batch_size=4,
            )


if __name__ == "__main__":
    unittest.main()
