import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ngram_freq import (  # noqa: E402
    FixedGramProbe,
    GlobalFrequencyIndex,
    _all_bucket_labels,
    _bucket_label,
    build_fixed_gram_manifest,
    fixed_gram_gap_summary,
)
from train import evaluate_freq_bins  # noqa: E402


class ZeroLogitModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, inp):
        return torch.zeros(
            inp.size(0), inp.size(1), self.vocab_size, device=inp.device)


class FixedGramProbeTest(unittest.TestCase):
    def _manifest(self, seed=42):
        tmp = tempfile.TemporaryDirectory()
        self.tmps = getattr(self, "tmps", [])
        self.tmps.append(tmp)
        data_dir = Path(tmp.name)
        # Twenty complete chunks provide enough repeated occurrences for the
        # reservoir sampler to make a seed-dependent choice.
        train = np.tile(np.array([1, 2, 1, 2, 1, 2], dtype=np.uint16), 20)
        val = np.tile(np.array([1, 2, 3, 4, 5, 6], dtype=np.uint16), 20)
        train.tofile(data_dir / "shard_00001.bin")
        val.tofile(data_dir / "shard_00002.bin")
        index = GlobalFrequencyIndex.build(train, vocab_size=10)
        manifest = build_fixed_gram_manifest(
            str(data_dir), [1], [2], index, 10, sequence_len=5,
            samples_per_bucket=2, seed=seed)
        return data_dir, index, manifest

    def tearDown(self):
        for tmp in getattr(self, "tmps", []):
            tmp.cleanup()

    def test_manifest_contexts_buckets_and_valid_positions(self):
        _, index, manifest = self._manifest()
        for source in ("train", "val"):
            for branch in ("bigram", "trigram"):
                first = 1 if branch == "bigram" else 2
                for bucket in _all_bucket_labels():
                    group = manifest["groups"][source][branch][bucket]
                    self.assertEqual(
                        group["selected_count"],
                        min(2, group["candidate_count"]),
                    )
                    for sample in group["samples"]:
                        self.assertGreaterEqual(sample["position"], first)
                        self.assertLess(sample["position"], 5)
                        self.assertEqual(
                            sample["hit_count"],
                            index.hit_count(branch, sample["context_key"]),
                        )
                        self.assertEqual(sample["bucket"], _bucket_label(sample["hit_count"]))
                        self.assertIn(source, ("train", "val"))

    def test_manifest_seed_is_reproducible(self):
        _, _, first = self._manifest(seed=7)
        _, _, second = self._manifest(seed=7)
        self.assertEqual(first, second)

        _, _, third = self._manifest(seed=8)
        first_samples = first["groups"]["train"]["bigram"]
        third_samples = third["groups"]["train"]["bigram"]
        self.assertTrue(any(
            first_samples[b]["samples"] != third_samples[b]["samples"]
            for b in _all_bucket_labels()
            if first_samples[b]["candidate_count"] > 2
        ))

    def test_probe_extracts_only_manifest_positions(self):
        data_dir, _, manifest = self._manifest()
        probe = FixedGramProbe(
            manifest, str(data_dir), sequence_len=5,
            device=torch.device("cpu"), device_batch_size=3)
        model = ZeroLogitModel(10)
        evaluation = probe.evaluate(model)
        expected = math.log(10.0)
        for source in ("train", "val"):
            for branch in ("bigram", "trigram"):
                for bucket in _all_bucket_labels():
                    stats = evaluation[source][branch][bucket]
                    manifest_stats = manifest["groups"][source][branch][bucket]
                    self.assertEqual(stats["sample_count"], manifest_stats["selected_count"])
                    if stats["sample_count"]:
                        self.assertAlmostEqual(stats["mean_loss"], expected, places=6)
        self.assertTrue(model.training)

    def test_gap_is_unweighted_mean_loss_difference(self):
        evaluation = {
            "train": {"bigram": {"1": {"sample_count": 2, "mean_loss": 2.0}},
                       "trigram": {}},
            "val": {"bigram": {"1": {"sample_count": 2, "mean_loss": 3.5}},
                     "trigram": {}},
        }
        result = fixed_gram_gap_summary(evaluation)
        self.assertEqual(result["bigram"]["1"]["gap_contribution"], 1.5)
        self.assertEqual(result["bigram"]["1"]["sample_count"], 2)
        self.assertEqual(result["bigram"]["1"]["train_sample_count"], 2)
        self.assertEqual(result["bigram"]["1"]["val_sample_count"], 2)

    def test_legacy_evaluator_reads_exact_batch_count(self):
        class CountingLoader:
            def __init__(self):
                self.count = 0

            def __next__(self):
                self.count += 1
                inp = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
                return inp, inp.clone()

        tokens = np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.uint16)
        index = GlobalFrequencyIndex.build(tokens, vocab_size=10)
        loader = CountingLoader()
        evaluate_freq_bins(ZeroLogitModel(10), loader, index, n_batches=4, vocab_size=10)
        self.assertEqual(loader.count, 4)


if __name__ == "__main__":
    unittest.main()
