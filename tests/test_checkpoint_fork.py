import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class CheckpointForkTests(unittest.TestCase):
    def test_common_prefix_resumes_into_two_complete_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "data"
            runs = base / "runs"
            data.mkdir()
            runs.mkdir()
            rng = np.random.default_rng(7)
            for shard_id, chunks in ((1, 4), (2, 12)):
                rng.integers(
                    0, 32, size=chunks * 9, dtype=np.uint16
                ).tofile(data / f"shard_{shard_id:05d}.bin")

            common = [
                sys.executable, str(ROOT / "code" / "train.py"),
                "--steps", "6", "--seed", "42", "--data_seed", "42",
                "--order_seed", "101", "--data_dir", str(data),
                "--out_dir", str(runs), "--train_shards", "1",
                "--val_shards", "2", "--device_batch_size", "2",
                "--total_batch_size", "16", "--val_interval", "2",
                "--val_batches", "1", "--table_norm_interval", "1",
                "--lr", "0.001", "--n_layer", "1", "--n_head", "1",
                "--n_embd", "8", "--vocab_size", "32",
                "--sequence_len", "8", "--fixed_probe_batches", "0",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(
                common + [
                    "--run_id", "prefix", "--train_order", "sequential",
                    "--stop_after_step", "2", "--save_checkpoint_step", "2",
                ],
                check=True, capture_output=True, text=True, env=environment,
            )
            checkpoint = runs / "prefix" / "checkpoint_step_000002.pt"
            for run_id, mode in (
                ("sequential", "sequential"),
                ("reshuffle", "sequential_then_reshuffle"),
            ):
                subprocess.run(
                    common + [
                        "--run_id", run_id, "--train_order", mode,
                        "--resume_checkpoint", str(checkpoint),
                    ],
                    check=True, capture_output=True, text=True, env=environment,
                )

            def read_rows(run_id):
                path = runs / run_id / "online_loss.jsonl"
                return [json.loads(line) for line in path.read_text().splitlines() if line]

            sequential = read_rows("sequential")
            reshuffle = read_rows("reshuffle")
            self.assertEqual(len(sequential), 6)
            self.assertEqual(sequential[:2], reshuffle[:2])
            self.assertEqual(
                [row["logical_batch_id"] for row in sequential[:4]], [0, 1, 0, 1]
            )
            self.assertEqual(
                [row["logical_batch_id"] for row in reshuffle[:2]], [0, 1]
            )
            self.assertNotEqual(
                [row["logical_batch_id"] for row in reshuffle[2:4]], [0, 1]
            )
            metadata = [
                json.loads((runs / run_id / "frequency_measurement_meta.json").read_text())
                for run_id in ("sequential", "reshuffle")
            ]
            hashes = {
                row["checkpoint_resume"]["shared_parameter_state_sha256"]
                for row in metadata
            }
            self.assertEqual(len(hashes), 1)
            self.assertTrue(next(iter(hashes)))

            subprocess.run(
                common + [
                    "--run_id", "random_prefix",
                    "--train_order", "frozen_permutation",
                    "--stop_after_step", "2", "--save_checkpoint_step", "2",
                ],
                check=True, capture_output=True, text=True, env=environment,
            )
            random_checkpoint = (
                runs / "random_prefix" / "checkpoint_step_000002.pt"
            )
            for run_id, mode in (
                ("random_frozen", "frozen_permutation"),
                ("random_reshuffle", "epoch_reshuffle"),
            ):
                subprocess.run(
                    common + [
                        "--run_id", run_id, "--train_order", mode,
                        "--resume_checkpoint", str(random_checkpoint),
                    ],
                    check=True, capture_output=True, text=True, env=environment,
                )

            random_frozen = read_rows("random_frozen")
            random_reshuffle = read_rows("random_reshuffle")
            self.assertEqual(random_frozen[:2], random_reshuffle[:2])
            self.assertEqual(
                [row["logical_batch_id"] for row in random_frozen[:2]],
                [row["logical_batch_id"] for row in random_reshuffle[:2]],
            )
            self.assertEqual(
                [row["logical_batch_id"] for row in random_frozen[:2]],
                [row["logical_batch_id"] for row in random_frozen[2:4]],
            )
            random_metadata = [
                json.loads(
                    (runs / run_id / "frequency_measurement_meta.json").read_text()
                )
                for run_id in ("random_frozen", "random_reshuffle")
            ]
            random_hashes = {
                row["checkpoint_resume"]["shared_parameter_state_sha256"]
                for row in random_metadata
            }
            self.assertEqual(len(random_hashes), 1)
            self.assertEqual(
                [row["train_order"]["mode"] for row in random_metadata],
                ["frozen_permutation", "epoch_reshuffle"],
            )


if __name__ == "__main__":
    unittest.main()
