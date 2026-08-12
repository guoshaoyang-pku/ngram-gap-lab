import unittest

from train import fixed_probe_center_steps, frequency_sample_reason


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


if __name__ == "__main__":
    unittest.main()
