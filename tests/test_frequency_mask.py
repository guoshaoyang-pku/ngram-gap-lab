import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from ngram_freq import ExactFrequencyMask
from train import Config, NanoGPT, online_gap_sample_reason


def toy_mask(threshold):
    return ExactFrequencyMask(
        bigram_keys=torch.tensor([9, 10, 19, 28]),
        bigram_counts=torch.tensor([1, 2, 3, 4]),
        trigram_keys=torch.tensor([73, 83, 138, 156]),
        trigram_counts=torch.tensor([1, 3, 2, 4]),
        vocab_size=8,
        threshold=threshold,
    )


class ExactFrequencyMaskTests(unittest.TestCase):
    def test_cumulative_threshold_masks_both_branches(self):
        idx = torch.tensor([[1, 2, 3, 4]])
        prev = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2 = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        bigram, trigram = toy_mask(2).activity_masks(idx, prev, prev2)
        expected = torch.tensor([[False, False, True, True]])
        self.assertTrue(torch.equal(bigram, expected))
        self.assertTrue(torch.equal(trigram, expected))

    def test_none_and_all_are_explicit_endpoints(self):
        idx = torch.tensor([[1, 2, 3, 4]])
        prev = torch.cat([idx[:, :1], idx[:, :-1]], dim=1)
        prev2 = torch.cat([idx[:, :2], idx[:, :-2]], dim=1)
        none_masks = toy_mask(None).activity_masks(idx, prev, prev2)
        all_masks = toy_mask("all").activity_masks(idx, prev, prev2)
        self.assertTrue(all(bool(mask.all()) for mask in none_masks))
        self.assertTrue(all(not bool(mask.any()) for mask in all_masks))

    def test_none_path_is_numerically_identical_and_all_has_no_table_gradient(self):
        config = Config(
            vocab_size=8, n_layer=1, n_head=1, n_embd=8,
            sequence_len=4, dropout=0.0,
        )
        torch.manual_seed(3)
        reference = NanoGPT(config)
        reference.init_weights()
        masked = NanoGPT(config)
        masked.load_state_dict(reference.state_dict())
        masked.set_exact_frequency_mask(toy_mask(None))
        idx = torch.tensor([[1, 2, 3, 4]])
        self.assertTrue(torch.equal(reference(idx), masked(idx)))

        all_masked = NanoGPT(config)
        all_masked.load_state_dict(reference.state_dict())
        all_masked.set_exact_frequency_mask(toy_mask("all"))
        targets = torch.tensor([[2, 3, 4, 5]])
        all_masked(idx, targets=targets).backward()
        table_grads = [
            parameter.grad
            for name, parameter in all_masked.named_parameters()
            if "bigram_ves" in name or "trigram_ves" in name
        ]
        self.assertTrue(table_grads)
        self.assertTrue(all(grad is not None for grad in table_grads))
        self.assertTrue(all(int(torch.count_nonzero(grad)) == 0 for grad in table_grads))

    def test_sparse_epoch_boundary_schedule(self):
        sampled = {
            step
            for step in range(1, 1012)
            if online_gap_sample_reason(step, 337, 1011, 50, (-1, 0, 1))
        }
        self.assertTrue({336, 337, 338, 673, 674, 675, 1011}.issubset(sampled))
        self.assertEqual(
            online_gap_sample_reason(337, 337, 1011, 50, (-1, 0, 1)),
            "epoch_boundary_+0",
        )


if __name__ == "__main__":
    unittest.main()
