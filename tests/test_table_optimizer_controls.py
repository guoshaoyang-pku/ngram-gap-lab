import unittest

import torch
import torch.nn as nn

from train import MixedOptimizer


class TinyMixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bigram_ves = nn.Parameter(torch.tensor([1.0]))
        self.backbone = nn.Parameter(torch.tensor([1.0]))


class TableOptimizerControlTests(unittest.TestCase):
    def test_table_lr_scale_only_changes_ngram_update(self):
        model = TinyMixedModel()
        optimizer = MixedOptimizer(
            model, lr=0.1, ngram_betas=(0.0, 0.99), adam_betas=(0.8, 0.95),
            weight_decay=0.0, ngram_lr_scale=0.5,
        )
        model.bigram_ves.grad = torch.ones_like(model.bigram_ves)
        model.backbone.grad = torch.ones_like(model.backbone)

        optimizer.step()

        self.assertAlmostEqual(model.bigram_ves.item(), 0.95, places=6)
        self.assertAlmostEqual(model.backbone.item(), 0.9, places=6)
        self.assertEqual(optimizer.ngram_beta2, 0.99)

    def test_zero_table_lr_freezes_only_ngram_parameters(self):
        model = TinyMixedModel()
        optimizer = MixedOptimizer(
            model, lr=0.1, ngram_betas=(0.0, 0.99), adam_betas=(0.8, 0.95),
            weight_decay=0.0, ngram_lr_scale=0.0,
        )
        model.bigram_ves.grad = torch.ones_like(model.bigram_ves)
        model.backbone.grad = torch.ones_like(model.backbone)

        optimizer.step()

        self.assertAlmostEqual(model.bigram_ves.item(), 1.0, places=6)
        self.assertAlmostEqual(model.backbone.item(), 0.9, places=6)

    def test_full_optimizer_state_round_trip_matches_continuation(self):
        first = TinyMixedModel()
        optimizer = MixedOptimizer(
            first, lr=0.1, ngram_betas=(0.0, 0.99), adam_betas=(0.8, 0.95),
            weight_decay=0.0, ngram_lr_scale=0.5,
        )
        first.bigram_ves.grad = torch.tensor([0.25])
        first.backbone.grad = torch.tensor([-0.5])
        optimizer.step(lr_mult=0.7)

        resumed = TinyMixedModel()
        resumed.load_state_dict(first.state_dict())
        resumed_optimizer = MixedOptimizer(
            resumed, lr=0.1, ngram_betas=(0.0, 0.99), adam_betas=(0.8, 0.95),
            weight_decay=0.0, ngram_lr_scale=0.5,
        )
        resumed_optimizer.load_state_dict(optimizer.state_dict())

        for model in (first, resumed):
            model.bigram_ves.grad = torch.tensor([-0.75])
            model.backbone.grad = torch.tensor([0.125])
        optimizer.step(lr_mult=0.4)
        resumed_optimizer.step(lr_mult=0.4)

        self.assertTrue(torch.equal(first.bigram_ves, resumed.bigram_ves))
        self.assertTrue(torch.equal(first.backbone, resumed.backbone))
        self.assertEqual(optimizer.rms_steps, resumed_optimizer.rms_steps)
        self.assertEqual(optimizer.adam_steps, resumed_optimizer.adam_steps)


if __name__ == "__main__":
    unittest.main()
