import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from train import (
    fixed_gram_sample_reason,
    fixed_probe_center_steps,
    frequency_sample_reason,
)


class FrequencySamplingTests(unittest.TestCase):
    def test_mid_epoch_probe_centers_are_one_based_writer_steps(self):
        self.assertEqual(
            fixed_probe_center_steps(337, 1000, 168),
            [169, 506, 843],
        )

    def test_sparse_epoch_and_probe_windows_include_their_centers(self):
        centers = fixed_probe_center_steps(337, 1000, 168)
        expected = {
            327: "epoch_dense", 332: "epoch_dense", 337: "epoch_dense",
            342: "epoch_dense", 347: "epoch_dense",
            159: "probe_dense", 164: "probe_dense", 169: "probe_dense",
            174: "probe_dense", 179: "probe_dense",
        }
        for step, reason in expected.items():
            self.assertEqual(
                frequency_sample_reason(step, 337, 50, 10, 5, 1000, centers, 10, 5),
                reason,
            )

    def test_regular_interval_has_priority(self):
        centers = fixed_probe_center_steps(337, 1000, 168)
        self.assertEqual(
            frequency_sample_reason(500, 337, 50, 10, 5, 1000, centers, 10, 5),
            "interval",
        )

    def test_fixed_gram_epoch_relative_schedule(self):
        offsets = (-10, -5, -1, 0, 1, 5, 10)
        sampled = {
            step
            for step in range(1, 1001)
            if fixed_gram_sample_reason(step, 337, 1000, 0, offsets) is not None
        }
        self.assertEqual(
            sampled,
            {
                328, 333, 337, 338, 339, 343, 348,
                665, 670, 674, 675, 676, 680, 685,
                1000,
            },
        )

    def test_online_epoch_window_can_sample_every_step(self):
        sampled = {
            step
            for step in range(300, 370)
            if frequency_sample_reason(step, 337, 50, 20, 1, 1000, [], 0, 1)
        }
        self.assertTrue(set(range(317, 358)).issubset(sampled))


if __name__ == "__main__":
    unittest.main()
