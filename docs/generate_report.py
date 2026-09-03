#!/usr/bin/env python3
"""Generate the single consolidated interactive report under docs/.

The report combines frozen historical plots with the validated RMSProp Stage 1,
Stage 2A, and strict Stage 3R 2x2 order-control conditions. It is written atomically only
after every input passes the setting, optimizer, completion, manifest, and
replay-edge checks used during the experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import fmean


BUCKETS = (
    "novel", "1", "2", "3", "4", "5", "6-10", "11-20", "21-50",
    "51-100", "101-200", "201-500", "501-1k", "1k-5k", "5k+",
)
COLORS = (
    "#E91E63", "#F44336", "#FF5722", "#FF9800", "#FFC107", "#FFEB3B",
    "#CDDC39", "#8BC34A", "#4CAF50", "#009688", "#00BCD4", "#03A9F4",
    "#2196F3", "#3F51B5", "#673AB7",
)
STAGE1_CONDITIONS = (
    (0.990, 0.5, "nglab_rms_b0990_lr050_s1"),
    (0.990, 1.0, "nglab_rms_b0990_lr100_s1"),
    (0.990, 1.5, "nglab_rms_b0990_lr150_s1"),
    (0.995, 0.5, "nglab_rms_b0995_lr050_s1"),
    (0.995, 1.0, "nglab_rms_b0995_lr100_s1"),
    (0.995, 1.5, "nglab_rms_b0995_lr150_s1"),
    (0.999, 0.5, "nglab_rms_b0999_lr050_s1"),
    (0.999, 1.0, "nglab_baseline_input_midprobe_sparse_20260812"),
    (0.999, 1.5, "nglab_rms_b0999_lr150_s1"),
)
STAGE2A_CONDITIONS = (
    (0.500, 0.250, "nglab_s2a_b0500_lr0250_s42"),
    (0.500, 0.500, "nglab_s2a_b0500_lr0500_s42"),
    (0.500, 1.000, "nglab_s2a_b0500_lr1000_s42"),
    (0.900, 0.250, "nglab_s2a_b0900_lr0250_s42"),
    (0.900, 0.500, "nglab_s2a_b0900_lr0500_s42"),
    (0.900, 1.000, "nglab_s2a_b0900_lr1000_s42"),
    (0.999, 0.000, "nglab_s2a_b0999_lr0000_s42_r1"),
    (0.999, 0.125, "nglab_s2a_b0999_lr0125_s42"),
    (0.999, 0.250, "nglab_s2a_b0999_lr0250_s42"),
    (0.999, 0.375, "nglab_s2a_b0999_lr0375_s42"),
    (0.999, 0.625, "nglab_s2a_b0999_lr0625_s42"),
    (0.999, 0.750, "nglab_s2a_b0999_lr0750_s42"),
    (0.999, 0.875, "nglab_s2a_b0999_lr0875_s42"),
)
STAGE3R_CONDITIONS = (
    ("original", "sequential", "nglab_s3r2_sequential_s42"),
    ("original", "sequential_then_reshuffle", "nglab_s3r2_reshuffle_s42_p101"),
    ("random", "frozen_permutation", "nglab_s3r3_random_frozen_s42_p101"),
    ("random", "epoch_reshuffle", "nglab_s3r3_random_reshuffle_s42_p101"),
)
FREQUENCY_MASK_CONDITIONS = (
    ("none", None, "nglab_freqmask_none_s42"),
    ("0", 0, "nglab_freqmask_x000000_s42"),
    ("1", 1, "nglab_freqmask_x000001_s42"),
    ("2", 2, "nglab_freqmask_x000002_s42"),
    ("5", 5, "nglab_freqmask_x000005_s42"),
    ("6", 6, "nglab_freqmask_x000006_s42"),
    ("7", 7, "nglab_freqmask_x000007_s42"),
    ("8", 8, "nglab_freqmask_x000008_s42"),
    ("9", 9, "nglab_freqmask_x000009_s42"),
    ("10", 10, "nglab_freqmask_x000010_s42"),
    ("11", 11, "nglab_freqmask_x000011_s42"),
    ("12", 12, "nglab_freqmask_x000012_s42"),
    ("13", 13, "nglab_freqmask_x000013_s42"),
    ("14", 14, "nglab_freqmask_x000014_s42"),
    ("15", 15, "nglab_freqmask_x000015_s42"),
    ("16", 16, "nglab_freqmask_x000016_s42"),
    ("17", 17, "nglab_freqmask_x000017_s42"),
    ("18", 18, "nglab_freqmask_x000018_s42"),
    ("19", 19, "nglab_freqmask_x000019_s42"),
    ("20", 20, "nglab_freqmask_x000020_s42"),
    ("25", 25, "nglab_freqmask_x000025_s42"),
    ("30", 30, "nglab_freqmask_x000030_s42"),
    ("35", 35, "nglab_freqmask_x000035_s42"),
    ("40", 40, "nglab_freqmask_x000040_s42"),
    ("45", 45, "nglab_freqmask_x000045_s42"),
    ("50", 50, "nglab_freqmask_x000050_s42"),
    ("60", 60, "nglab_freqmask_x000060_s42"),
    ("70", 70, "nglab_freqmask_x000070_s42"),
    ("80", 80, "nglab_freqmask_x000080_s42"),
    ("90", 90, "nglab_freqmask_x000090_s42"),
    ("100", 100, "nglab_freqmask_x000100_s42"),
    ("110", 110, "nglab_freqmask_x000110_s42"),
    ("120", 120, "nglab_freqmask_x000120_s42"),
    ("130", 130, "nglab_freqmask_x000130_s42"),
    ("140", 140, "nglab_freqmask_x000140_s42"),
    ("150", 150, "nglab_freqmask_x000150_s42"),
    ("160", 160, "nglab_freqmask_x000160_s42"),
    ("170", 170, "nglab_freqmask_x000170_s42"),
    ("180", 180, "nglab_freqmask_x000180_s42"),
    ("190", 190, "nglab_freqmask_x000190_s42"),
    ("200", 200, "nglab_freqmask_x000200_s42"),
    ("210", 210, "nglab_freqmask_x000210_s42"),
    ("500", 500, "nglab_freqmask_x000500_s42"),
    ("1k", 1000, "nglab_freqmask_x001000_s42"),
    ("2k", 2000, "nglab_freqmask_x002000_s42"),
    ("5k", 5000, "nglab_freqmask_x005000_s42"),
    ("20k", 20000, "nglab_freqmask_x020000_s42"),
    ("100k", 100000, "nglab_freqmask_x100000_s42"),
    ("all", "all", "nglab_freqmask_all_s42"),
)
FREQUENCY_MASK_BRIDGE_RUN_ID = "nglab_freqmask_none_s42_h200360"
FREQUENCY_MASK_INDEX_SHA256 = (
    "763a5548f75a7e326370610112fa58f61bcaa37d5a558c33caa3c7482007673d"
)
STAGE1_DEFAULT_RUN_ID = "nglab_baseline_input_midprobe_sparse_20260812"
STAGE2A_DEFAULT_RUN_ID = "nglab_s2a_b0999_lr0375_s42"
STAGE3R_DEFAULT_RUN_ID = "nglab_s3r2_reshuffle_s42_p101"
LEGACY_BASELINE_RUN = STAGE1_DEFAULT_RUN_ID
LOW_BUCKETS = ("1", "2", "3", "4", "5", "6-10", "11-20")
HIGH_BUCKETS = ("21-50", "51-100", "101-200", "201-500", "501-1k", "1k-5k", "5k+")
REQUIRED_RUN_FILES = (
    "summary.json",
    "online_loss.jsonl",
    "train_log.jsonl",
    "table_norm.jsonl",
    "online_frequency_gap_contribution.jsonl",
    "fixed_probe_frequency_gap_contribution.jsonl",
    "fixed_gram_frequency_gap_contribution.jsonl",
    "frequency_measurement_meta.json",
    "fixed_gram_probe_manifest.json",
)
STAGE3R_REQUIRED_FILES = (
    "summary.json",
    "online_loss.jsonl",
    "train_log.jsonl",
    "table_norm.jsonl",
    "online_frequency_gap_contribution.jsonl",
    "fixed_gram_frequency_gap_contribution.jsonl",
    "frequency_measurement_meta.json",
    "fixed_gram_probe_manifest.json",
    "train_batch_order.json",
)
FREQUENCY_MASK_REQUIRED_FILES = (
    "summary.json",
    "online_loss.jsonl",
    "online_gap.jsonl",
    "train_log.jsonl",
    "table_norm.jsonl",
    "frequency_measurement_meta.json",
    "train_batch_order.json",
    "runtime.txt",
    "train.log",
    "job_meta.txt",
)
FREQUENCY_MASK_EPOCH_ENDS = ((1, 337), (2, 674), (3, 1011))
STAGE3R_FIXED_GRAM_STEPS = (
    328, 333, 337, 338, 339, 343, 348,
    665, 670, 674, 675, 676, 680, 685, 1000,
)
HISTORICAL_CHARTS = {
    "injection_gap", "injection_loss", "table_norm", "input_alignment",
    "frequency_bins", "hitcount_distribution", "gap_vs_frequency_log",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_key_values(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def frequency_series(records: list[dict], train_key: str, val_key: str) -> dict:
    output = {
        branch: {
            bucket: {
                "steps": [], "epochs": [], "reasons": [],
                "train_loss": [], "val_loss": [], "train_frac": [],
                "val_frac": [], "train_contribution": [],
                "val_contribution": [], "gap_contribution": [],
                "mean_loss_gap": [],
            }
            for bucket in BUCKETS
        }
        for branch in ("bigram", "trigram")
    }
    for row in records:
        for branch in output:
            for bucket in BUCKETS:
                train = row[train_key][branch][bucket]
                val = row[val_key][branch][bucket]
                destination = output[branch][bucket]
                destination["steps"].append(row["step"])
                destination["epochs"].append(row["epoch"])
                destination["reasons"].append(row["reason"])
                destination["train_loss"].append(train["mean_loss"])
                destination["val_loss"].append(val["mean_loss"])
                destination["train_frac"].append(train["frac"])
                destination["val_frac"].append(val["frac"])
                destination["train_contribution"].append(train["total_contrib"])
                destination["val_contribution"].append(val["total_contrib"])
                destination["gap_contribution"].append(
                    row["gap_contribution"][branch][bucket]["contribution"]
                )
                destination["mean_loss_gap"].append(
                    row["gap_contribution"][branch][bucket]["mean_loss_gap"]
                )
    return output


def fixed_gram_series(records: list[dict]) -> dict:
    output = {
        branch: {
            bucket: {
                "steps": [], "epochs": [], "reasons": [],
                "train_loss": [], "val_loss": [], "train_sample_count": [],
                "val_sample_count": [], "gap_contribution": [],
                "mean_loss_gap": [],
            }
            for bucket in BUCKETS
        }
        for branch in ("bigram", "trigram")
    }
    for row in records:
        for branch in output:
            for bucket in BUCKETS:
                stats = row.get("branches", {}).get(branch, {}).get(bucket, {})
                destination = output[branch][bucket]
                destination["steps"].append(row.get("step"))
                destination["epochs"].append(row.get("epoch"))
                destination["reasons"].append(row.get("reason", ""))
                destination["train_loss"].append(stats.get("train_mean_loss"))
                destination["val_loss"].append(stats.get("val_mean_loss"))
                destination["train_sample_count"].append(
                    stats.get("train_sample_count", 0)
                )
                destination["val_sample_count"].append(
                    stats.get("val_sample_count", 0)
                )
                gap = stats.get("gap_contribution")
                destination["gap_contribution"].append(gap)
                destination["mean_loss_gap"].append(gap)
    return output


def validate_optimizer(
    run_id: str, beta2: float, lr_scale: float, summary: dict, meta: dict
) -> dict:
    optimizer = meta.get("optimizer", {}).get("ngram_table", {})
    if run_id == LEGACY_BASELINE_RUN and not optimizer:
        config = summary.get("config", {})
        require(
            config.get("ngram_table_betas") == [0.0, beta2],
            f"{run_id}: legacy beta metadata does not match matrix",
        )
        require(
            config.get("nanogpt_adam_lr") == 0.004 and lr_scale == 1.0,
            f"{run_id}: legacy center LR does not match Stage 1",
        )
        optimizer = {
            "name": "bias_corrected_rmsprop_no_momentum",
            "beta2": beta2,
            "lr_scale": lr_scale,
            "weight_decay": 0.0,
            "eps": 1e-10,
        }
    require(
        optimizer.get("name") == "bias_corrected_rmsprop_no_momentum",
        f"{run_id}: unexpected table optimizer",
    )
    require(
        abs(optimizer.get("beta2", -1.0) - beta2) < 1e-12,
        f"{run_id}: beta2 metadata does not match matrix",
    )
    require(
        abs(optimizer.get("lr_scale", -1.0) - lr_scale) < 1e-12,
        f"{run_id}: LR-scale metadata does not match matrix",
    )
    require(
        optimizer.get("weight_decay") == 0.0,
        f"{run_id}: table weight decay is not zero",
    )
    require(
        optimizer.get("eps") == 1e-10,
        f"{run_id}: table epsilon differs from Stage 1 setting",
    )
    return optimizer


def validate_fixed_setting(run_id: str, summary: dict, meta: dict) -> None:
    config = summary.get("config", {})
    expected_config = {
        "nanogpt_ngram_injection_position": "input",
        "enable_unigram_ve": False,
        "enable_bigram_ve": True,
        "enable_trigram_ve": True,
        "seed": 42,
        "max_steps": 1000,
        "nanogpt_adam_lr": 0.004,
        "adam_betas": [0.8, 0.95],
        "weight_decay": 0.1,
        "train_shards": [1],
    }
    for key, expected in expected_config.items():
        require(
            config.get(key) == expected,
            f"{run_id}: config {key} does not match the fixed setting",
        )

    backbone = meta.get("optimizer", {}).get("backbone")
    if backbone:
        require(backbone.get("name") == "adamw", f"{run_id}: backbone is not AdamW")
        require(backbone.get("betas") == [0.8, 0.95], f"{run_id}: backbone betas differ")
        require(backbone.get("base_lr") == 0.004, f"{run_id}: backbone LR differs")
        require(backbone.get("weight_decay") == 0.1, f"{run_id}: backbone WD differs")

    fixed_gram = meta.get("fixed_gram_probe", {})
    require(
        fixed_gram.get("samples_per_bucket") == 100,
        f"{run_id}: fixed-gram samples per bucket differs",
    )
    require(fixed_gram.get("seed") == 42, f"{run_id}: fixed-gram seed differs")
    counts = fixed_gram.get("stats", {}).get("sample_count", {})
    for split in ("train", "val"):
        for branch in ("bigram", "trigram"):
            branch_counts = counts.get(split, {}).get(branch, {})
            require(set(branch_counts) == set(BUCKETS), f"{run_id}: incomplete fixed-gram buckets")
            for bucket, count in branch_counts.items():
                expected = 0 if split == "train" and bucket == "novel" else 100
                require(
                    count == expected,
                    f"{run_id}: unexpected {split} {branch} {bucket} sample count",
                )


def value_at(records: list[dict], step: int, run_id: str) -> dict:
    rows = [row for row in records if row.get("step") == step]
    require(bool(rows), f"{run_id}: missing fixed-gram checkpoint at step {step}")
    return rows[-1]


def mean_bucket_gap(row: dict, branch: str, buckets: tuple[str, ...]) -> float:
    values = [row["branches"][branch][bucket]["gap_contribution"] for bucket in buckets]
    require(all(value is not None for value in values), f"{branch}: incomplete bucket values")
    return fmean(values)


def edge_metrics(run: dict) -> list[dict]:
    epoch_steps = run["meta"].get("sampling", {}).get("estimated_steps_per_epoch")
    require(
        isinstance(epoch_steps, int) and epoch_steps > 10,
        f"{run['run_id']}: missing epoch geometry",
    )
    metrics = []
    for edge in range(epoch_steps, 1000, epoch_steps):
        before = value_at(run["fixed_gram"], edge - 5, run["run_id"])
        after = value_at(run["fixed_gram"], edge + 5, run["run_id"])
        row = {"edge": edge}
        for branch in ("bigram", "trigram"):
            low = mean_bucket_gap(after, branch, LOW_BUCKETS) - mean_bucket_gap(
                before, branch, LOW_BUCKETS
            )
            high = mean_bucket_gap(after, branch, HIGH_BUCKETS) - mean_bucket_gap(
                before, branch, HIGH_BUCKETS
            )
            row[f"{branch}_low"] = low
            row[f"{branch}_high"] = high
            row[f"{branch}_tilt"] = low - high
        metrics.append(row)
    require(
        [row["edge"] for row in metrics] == [337, 674],
        f"{run['run_id']}: expected replay edges 337 and 674",
    )
    return metrics


def load_stage_run(
    runs_root: Path, beta2: float, lr_scale: float, run_id: str
) -> dict:
    run_dir = runs_root / run_id
    for filename in REQUIRED_RUN_FILES:
        path = run_dir / filename
        require(
            path.is_file() and path.stat().st_size > 0,
            f"{run_id}: missing or empty {filename}",
        )
    summary = read_json(run_dir / "summary.json")
    meta = read_json(run_dir / "frequency_measurement_meta.json")
    require(meta.get("run_id") == run_id, f"{run_id}: metadata run ID differs")
    validate_fixed_setting(run_id, summary, meta)
    optimizer = validate_optimizer(run_id, beta2, lr_scale, summary, meta)
    online_loss = read_jsonl(run_dir / "online_loss.jsonl")
    train_log = read_jsonl(run_dir / "train_log.jsonl")
    table_norm = read_jsonl(run_dir / "table_norm.jsonl")
    online_frequency = read_jsonl(run_dir / "online_frequency_gap_contribution.jsonl")
    fixed_frequency = read_jsonl(run_dir / "fixed_probe_frequency_gap_contribution.jsonl")
    fixed_gram = read_jsonl(run_dir / "fixed_gram_frequency_gap_contribution.jsonl")
    series = {
        "train_log": train_log,
        "online_loss": online_loss,
        "table_norm": table_norm,
        "online_frequency": online_frequency,
        "fixed_probe_frequency": fixed_frequency,
        "fixed_gram_frequency": fixed_gram,
    }
    for name, rows in series.items():
        require(bool(rows), f"{run_id}: {name} contains no records")
        require(rows[-1].get("step") == 1000, f"{run_id}: {name} does not reach step 1000")
    manifest_sha256 = hashlib.sha256(
        (run_dir / "fixed_gram_probe_manifest.json").read_bytes()
    ).hexdigest()
    run = {
        "run_id": run_id,
        "beta2": beta2,
        "lr_scale": lr_scale,
        "summary": summary,
        "meta": meta,
        "optimizer": optimizer,
        "manifest_sha256": manifest_sha256,
        "online_loss": online_loss,
        "train_log": train_log,
        "table_norm": table_norm,
        "online_frequency": online_frequency,
        "fixed_frequency": fixed_frequency,
        "fixed_gram": fixed_gram,
        "payload": {
            "runId": run_id,
            "beta2": beta2,
            "lrScale": lr_scale,
            "tableBaseLr": 0.004 * lr_scale,
            "optimizer": optimizer,
            "meta": meta,
            "onlineLoss": online_loss,
            "validation": train_log,
            "frequency": {
                "online": frequency_series(online_frequency, "train_writer", "online_val"),
                "fixed": frequency_series(fixed_frequency, "train_probe", "val_probe"),
                "fixed_gram": fixed_gram_series(fixed_gram),
            },
            "fixedReads": [
                {"step": row["step"], "epoch": row["epoch"], "reason": row["reason"]}
                for row in fixed_frequency
            ],
        },
    }
    run["edges"] = edge_metrics(run)
    return run


def load_stage3r_run(
    runs_root: Path, comparison_group: str, order_mode: str, run_id: str
) -> dict:
    run_dir = runs_root / run_id
    for filename in STAGE3R_REQUIRED_FILES:
        path = run_dir / filename
        require(
            path.is_file() and path.stat().st_size > 0,
            f"{run_id}: missing or empty {filename}",
        )
    require(
        not (run_dir / "fixed_probe_frequency_gap_contribution.jsonl").exists(),
        f"{run_id}: Stage 3R must not contain fixed-probe output",
    )

    summary = read_json(run_dir / "summary.json")
    meta = read_json(run_dir / "frequency_measurement_meta.json")
    order = read_json(run_dir / "train_batch_order.json")
    require(meta.get("run_id") == run_id, f"{run_id}: metadata run ID differs")
    validate_fixed_setting(run_id, summary, meta)
    optimizer = validate_optimizer(run_id, 0.999, 1.0, summary, meta)
    require(
        meta.get("fixed_probe") == {"enabled": False},
        f"{run_id}: fixed probe is not explicitly disabled",
    )
    order_meta = meta.get("train_order", {})
    require(comparison_group in {"original", "random"}, "bad Stage 3R group")
    require(
        order_mode in {
            "frozen_permutation", "epoch_reshuffle",
            "sequential", "sequential_then_reshuffle",
        },
        "bad Stage 3R mode",
    )
    require(order_meta.get("mode") == order_mode, f"{run_id}: train-order mode differs")
    require(order_meta.get("seed") == 101, f"{run_id}: order seed differs")
    require(order_meta.get("logical_batch_size") == 72, f"{run_id}: logical batch differs")
    require(order_meta.get("batches_per_epoch") == 337, f"{run_id}: epoch geometry differs")
    require(order.get("mode") == order_mode and order.get("seed") == 101, f"{run_id}: order file metadata differs")
    config = summary.get("config", {})
    if comparison_group == "original":
        require(
            order_mode in {"sequential", "sequential_then_reshuffle"},
            f"{run_id}: original-order group uses a random-order mode",
        )
    else:
        require(
            order_mode in {"frozen_permutation", "epoch_reshuffle"},
            f"{run_id}: random-order group uses an original-order mode",
        )
    require(config.get("data_seed") == 42, f"{run_id}: data seed differs from baseline")
    require(config.get("order_seed") == 101, f"{run_id}: independent order seed differs")
    resume = meta.get("checkpoint_resume", {})
    require(resume.get("completed_prefix_step") == 337, f"{run_id}: bad fork step")
    require(
        isinstance(resume.get("shared_parameter_state_sha256"), str)
        and len(resume["shared_parameter_state_sha256"]) == 64,
        f"{run_id}: missing shared parameter-state hash",
    )
    epochs = order.get("epochs", [])
    require(len(epochs) == 3, f"{run_id}: expected three recorded epoch orders")
    hashes = [epoch.get("sha256") for epoch in epochs]
    require(hashes == order_meta.get("epoch_sha256"), f"{run_id}: order hashes disagree")
    for epoch_index, epoch in enumerate(epochs, 1):
        values = epoch.get("order", [])
        require(epoch.get("epoch") == epoch_index, f"{run_id}: bad order epoch index")
        require(sorted(values) == list(range(337)), f"{run_id}: epoch order is not a permutation")
    sequential_order = list(range(337))
    if order_mode == "sequential":
        require(
            all(epoch["order"] == sequential_order for epoch in epochs),
            f"{run_id}: sequential order changed across epochs",
        )
    elif order_mode == "sequential_then_reshuffle":
        require(epochs[0]["order"] == sequential_order, f"{run_id}: epoch 1 is not sequential")
        require(len(set(hashes)) == 3, f"{run_id}: reshuffle did not change epochs 2/3")
    elif order_mode == "frozen_permutation":
        require(len(set(hashes)) == 1, f"{run_id}: frozen permutation changed across epochs")
        require(epochs[0]["order"] != sequential_order, f"{run_id}: frozen order is sequential")
    else:
        require(len(set(hashes)) == 3, f"{run_id}: epoch reshuffle did not change every epoch")
        require(epochs[0]["order"] != sequential_order, f"{run_id}: random epoch 1 is sequential")

    online_loss = read_jsonl(run_dir / "online_loss.jsonl")
    train_log = read_jsonl(run_dir / "train_log.jsonl")
    table_norm = read_jsonl(run_dir / "table_norm.jsonl")
    online_frequency = read_jsonl(run_dir / "online_frequency_gap_contribution.jsonl")
    fixed_gram = read_jsonl(run_dir / "fixed_gram_frequency_gap_contribution.jsonl")
    series = {
        "online_loss": online_loss,
        "train_log": train_log,
        "table_norm": table_norm,
        "online_frequency": online_frequency,
        "fixed_gram_frequency": fixed_gram,
    }
    for name, rows in series.items():
        require(bool(rows), f"{run_id}: {name} contains no records")
        require(rows[-1].get("step") == 1000, f"{run_id}: {name} does not reach step 1000")
    require(
        [row.get("step") for row in fixed_gram] == list(STAGE3R_FIXED_GRAM_STEPS),
        f"{run_id}: Stage 3R fixed-gram checkpoints differ",
    )
    online_steps = {row.get("step") for row in online_frequency}
    for epoch_start in (338, 675):
        expected = set(range(epoch_start - 20, epoch_start + 20))
        require(expected <= online_steps, f"{run_id}: incomplete online edge window at {epoch_start}")

    manifest_sha256 = hashlib.sha256(
        (run_dir / "fixed_gram_probe_manifest.json").read_bytes()
    ).hexdigest()
    payload = {
        "runId": run_id,
        "comparisonGroup": comparison_group,
        "orderMode": order_mode,
        "orderSeed": order_meta["seed"],
        "orderHashes": hashes,
        "beta2": 0.999,
        "lrScale": 1.0,
        "tableBaseLr": 0.004,
        "optimizer": optimizer,
        "meta": meta,
        "onlineLoss": online_loss,
        "validation": train_log,
        "onlineGap": [
            {
                "step": row["step"],
                "epoch": row["epoch"],
                "logicalBatchId": row.get("logical_batch_id"),
                "gap": row["online_val_loss"] - row["train_writer_loss"],
                "trainLoss": row["train_writer_loss"],
                "validationLoss": row["online_val_loss"],
                "reason": row["reason"],
            }
            for row in online_frequency
        ],
        "frequency": {
            "online": frequency_series(online_frequency, "train_writer", "online_val"),
            "fixed_gram": fixed_gram_series(fixed_gram),
        },
        "fixedReads": [],
    }
    return {
        "run_id": run_id,
        "comparison_group": comparison_group,
        "order_mode": order_mode,
        "summary": summary,
        "meta": meta,
        "optimizer": optimizer,
        "order": order,
        "order_hashes": hashes,
        "manifest_sha256": manifest_sha256,
        "online_loss": online_loss,
        "train_log": train_log,
        "table_norm": table_norm,
        "online_frequency": online_frequency,
        "fixed_gram": fixed_gram,
        "payload": payload,
    }


def load_frequency_mask_run(
    runs_root: Path, label: str, threshold: int | str | None, run_id: str
) -> dict:
    run_dir = runs_root / run_id
    for filename in FREQUENCY_MASK_REQUIRED_FILES:
        path = run_dir / filename
        require(
            path.is_file() and path.stat().st_size > 0,
            f"{run_id}: missing or empty {filename}",
        )
    for forbidden in (
        "fixed_probe_frequency_gap_contribution.jsonl",
        "fixed_gram_frequency_gap_contribution.jsonl",
        "online_frequency_gap_contribution.jsonl",
    ):
        require(
            not (run_dir / forbidden).exists(),
            f"{run_id}: frequency-mask sweep must not contain {forbidden}",
        )

    summary = read_json(run_dir / "summary.json")
    meta = read_json(run_dir / "frequency_measurement_meta.json")
    order = read_json(run_dir / "train_batch_order.json")
    require(summary.get("run_id") == run_id, f"{run_id}: summary run ID differs")
    require(meta.get("run_id") == run_id, f"{run_id}: metadata run ID differs")
    require(summary.get("steps") == 1011, f"{run_id}: summary does not reach step 1011")

    config = summary.get("config", {})
    expected_config = {
        "nanogpt_ngram_injection_position": "input",
        "enable_unigram_ve": False,
        "enable_bigram_ve": True,
        "enable_trigram_ve": True,
        "seed": 42,
        "max_steps": 1011,
        "nanogpt_adam_lr": 0.004,
        "adam_betas": [0.8, 0.95],
        "weight_decay": 0.1,
        "train_shards": [1],
        "data_mode": "fixed",
        "data_seed": 42,
        "order_seed": 42,
        "train_order": "sequential",
    }
    for key, expected in expected_config.items():
        require(
            config.get(key) == expected,
            f"{run_id}: config {key} does not match the frequency-mask setting",
        )
    optimizer = validate_optimizer(run_id, 0.999, 1.0, summary, meta)
    backbone = meta.get("optimizer", {}).get("backbone", {})
    require(
        backbone == {
            "name": "adamw", "betas": [0.8, 0.95],
            "base_lr": 0.004, "weight_decay": 0.1,
        },
        f"{run_id}: backbone optimizer metadata differs",
    )

    mask = meta.get("ngram_frequency_mask", {})
    summary_mask = summary.get("ngram_frequency_mask", {})
    expected_mode = "none" if threshold is None else str(threshold)
    require(mask == summary_mask, f"{run_id}: summary and metadata masks differ")
    require(mask.get("enabled") is True, f"{run_id}: mask facility is not enabled")
    require(mask.get("mode") == expected_mode, f"{run_id}: mask mode differs")
    require(mask.get("threshold") == threshold, f"{run_id}: mask threshold differs")
    require(
        mask.get("comparison") == "mask exact-context train hit count <= threshold",
        f"{run_id}: mask comparison rule differs",
    )
    require(mask.get("branches") == ["bigram", "trigram"], f"{run_id}: mask branches differ")
    require(mask.get("train_and_validation_forward") is True, f"{run_id}: mask is not applied to both forwards")
    require(mask.get("renormalize_remaining_residual") is False, f"{run_id}: masked residual is renormalized")
    require(mask.get("index_sha256") == FREQUENCY_MASK_INDEX_SHA256, f"{run_id}: frequency-index SHA differs")

    mask_statistics = {}
    for branch in ("bigram", "trigram"):
        stats = mask.get("index_statistics", {}).get(branch, {})
        unique_contexts = stats.get("unique_contexts")
        max_hit_count = stats.get("max_hit_count")
        total_occurrences = stats.get("total_occurrences")
        masked_unique = stats.get("masked_unique_contexts")
        masked_occurrences = stats.get("masked_occurrences")
        fraction = stats.get("masked_occurrence_fraction")
        require(
            all(isinstance(value, int) and value >= 0 for value in (
                unique_contexts, max_hit_count, total_occurrences,
                masked_unique, masked_occurrences,
            )),
            f"{run_id}: invalid {branch} mask counts",
        )
        require(unique_contexts > 0 and total_occurrences > 0, f"{run_id}: empty {branch} index")
        require(max_hit_count > 0, f"{run_id}: invalid {branch} maximum hit count")
        require(0 <= masked_unique <= unique_contexts, f"{run_id}: invalid {branch} masked contexts")
        require(0 <= masked_occurrences <= total_occurrences, f"{run_id}: invalid {branch} masked occurrences")
        require(
            isinstance(fraction, (int, float)) and math.isfinite(fraction)
            and abs(fraction - masked_occurrences / total_occurrences) < 1e-12,
            f"{run_id}: inconsistent {branch} masked fraction",
        )
        if threshold is None:
            require(masked_unique == masked_occurrences == 0 and fraction == 0.0, f"{run_id}: none masks {branch} contexts")
        elif threshold == "all":
            require(masked_unique == unique_contexts and masked_occurrences == total_occurrences and fraction == 1.0, f"{run_id}: all does not mask every {branch} context")
        mask_statistics[branch] = {
            "uniqueContexts": unique_contexts,
            "maxHitCount": max_hit_count,
            "totalOccurrences": total_occurrences,
            "maskedUniqueContexts": masked_unique,
            "maskedOccurrences": masked_occurrences,
            "maskedOccurrenceFraction": fraction,
        }

    require(meta.get("fixed_probe") == {"enabled": False}, f"{run_id}: fixed probe is not disabled")
    require(meta.get("fixed_gram_probe") is None, f"{run_id}: fixed-gram probe must be disabled")
    require(meta.get("legacy_freq_eval") is False, f"{run_id}: legacy frequency evaluation is enabled")
    sampling = meta.get("sampling", {})
    require(sampling.get("estimated_steps_per_epoch") == 337, f"{run_id}: epoch geometry differs")
    require(
        sampling.get("online_gap") == {
            "enabled": True,
            "base_interval": 50,
            "epoch_end_offsets": [-1, 0, 1],
            "val_batches": 1,
            "metric": "online_val_loss - train_writer_loss",
            "timing": "pre_optimizer_step",
        },
        f"{run_id}: online-gap sampling metadata differs",
    )
    require(
        sampling.get("fixed_gram") == {
            "mode": "shared_online", "base_interval": 50,
            "epoch_relative_steps": [], "include_final": True,
        },
        f"{run_id}: fixed-gram sampling must remain inactive",
    )

    order_meta = meta.get("train_order", {})
    require(order_meta.get("mode") == "sequential" and order_meta.get("seed") == 42, f"{run_id}: train-order metadata differs")
    require(order_meta.get("batches_per_epoch") == 337, f"{run_id}: train-order geometry differs")
    require(order.get("mode") == "sequential" and order.get("seed") == 42, f"{run_id}: train-order file differs")
    epochs = order.get("epochs", [])
    require(len(epochs) == 3, f"{run_id}: expected three train-order epochs")
    sequential = list(range(337))
    require(all(row.get("order") == sequential for row in epochs), f"{run_id}: batch order is not fixed sequential replay")

    online_loss = read_jsonl(run_dir / "online_loss.jsonl")
    online_gap = read_jsonl(run_dir / "online_gap.jsonl")
    train_log = read_jsonl(run_dir / "train_log.jsonl")
    table_norm = read_jsonl(run_dir / "table_norm.jsonl")
    require([row.get("step") for row in online_loss] == list(range(1, 1012)), f"{run_id}: online loss is incomplete")
    require(train_log and train_log[-1].get("step") == 1011, f"{run_id}: train log does not reach 1011")
    require(table_norm and table_norm[-1].get("step") == 1010, f"{run_id}: table norm does not reach 1010")
    expected_gap_steps = sorted(
        set(range(50, 1001, 50)) | {336, 337, 338, 673, 674, 675, 1010, 1011}
    )
    require([row.get("step") for row in online_gap] == expected_gap_steps, f"{run_id}: online-gap checkpoints differ")
    epoch_gaps = {}
    for epoch, step in FREQUENCY_MASK_EPOCH_ENDS:
        rows = [row for row in online_gap if row.get("step") == step]
        require(len(rows) == 1, f"{run_id}: missing epoch-{epoch} end gap")
        row = rows[0]
        require(row.get("epoch") == epoch, f"{run_id}: epoch-{epoch} end label differs")
        require(row.get("reason") == "epoch_boundary_+0", f"{run_id}: epoch-{epoch} end reason differs")
        require(row.get("parameter_state") == "pre_optimizer_step", f"{run_id}: epoch-{epoch} gap timing differs")
        gap = row.get("gap")
        require(isinstance(gap, (int, float)) and math.isfinite(gap), f"{run_id}: invalid epoch-{epoch} gap")
        require(abs(gap - (row["online_val_loss"] - row["train_writer_loss"])) < 1e-12, f"{run_id}: epoch-{epoch} gap arithmetic differs")
        epoch_gaps[str(epoch)] = gap

    runtime = read_key_values(run_dir / "runtime.txt")
    job_meta = read_key_values(run_dir / "job_meta.txt")
    wall_seconds = float(runtime.get("wall_s", "nan"))
    require(math.isfinite(wall_seconds) and wall_seconds > 0, f"{run_id}: invalid wall time")
    require(job_meta.get("run_id") == run_id, f"{run_id}: job metadata run ID differs")
    require(job_meta.get("threshold") == expected_mode, f"{run_id}: job metadata threshold differs")
    train_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    require("[nglab] DONE" in train_text, f"{run_id}: train log lacks DONE marker")
    require(not any(token in train_text for token in ("Traceback", "CUDA out of memory", "NaN")), f"{run_id}: train log contains an error marker")

    return {
        "label": label,
        "threshold": threshold,
        "runId": run_id,
        "epochGaps": epoch_gaps,
        "maskStatistics": mask_statistics,
        "wallSeconds": wall_seconds,
        "host": job_meta.get("host", "unknown"),
        "optimizer": optimizer,
        "indexSha256": mask["index_sha256"],
        "orderHashes": order_meta.get("epoch_sha256"),
    }


def build_frequency_mask_analysis(runs: list[dict], bridge: dict) -> dict:
    require(len(runs) == 49, "frequency-mask sweep requires exactly 49 conditions")
    require(
        [(run["label"], run["threshold"], run["runId"]) for run in runs]
        == list(FREQUENCY_MASK_CONDITIONS),
        "frequency-mask conditions are not in the documented order",
    )
    require(len({run["runId"] for run in runs}) == 49, "frequency-mask run IDs are not unique")
    require(bridge["runId"] == FREQUENCY_MASK_BRIDGE_RUN_ID, "frequency-mask bridge run differs")
    require(bridge["threshold"] is None, "frequency-mask bridge is not a none condition")
    require(
        len({run["indexSha256"] for run in runs + [bridge]}) == 1,
        "frequency-mask runs do not share one exact-frequency index",
    )
    require(
        len({tuple(run["orderHashes"]) for run in runs + [bridge]}) == 1,
        "frequency-mask runs do not share one replay order",
    )
    for branch in ("bigram", "trigram"):
        require(
            len({run["maskStatistics"][branch]["maxHitCount"] for run in runs + [bridge]}) == 1,
            f"{branch}: maximum hit count differs across runs",
        )
        masked_occurrences = [run["maskStatistics"][branch]["maskedOccurrences"] for run in runs]
        masked_contexts = [run["maskStatistics"][branch]["maskedUniqueContexts"] for run in runs]
        require(masked_occurrences == sorted(masked_occurrences), f"{branch}: masked occurrence counts are not monotone")
        require(masked_contexts == sorted(masked_contexts), f"{branch}: masked context counts are not monotone")

    primary = runs[0]
    bridge_delta = {
        epoch: bridge["epochGaps"][epoch] - primary["epochGaps"][epoch]
        for epoch in ("1", "2", "3")
    }
    all_equivalent_threshold = max(
        runs[-1]["maskStatistics"][branch]["maxHitCount"]
        for branch in ("bigram", "trigram")
    )
    require(
        all_equivalent_threshold > max(
            run["threshold"] for run in runs if isinstance(run["threshold"], int)
        ),
        "all sentinel does not extend the numeric threshold range",
    )
    points = []
    for ordinal, run in enumerate(runs):
        function_x = (
            None if run["threshold"] is None
            else all_equivalent_threshold if run["threshold"] == "all"
            else run["threshold"]
        )
        points.append({
            "ordinal": ordinal,
            "label": run["label"],
            "threshold": run["threshold"],
            "functionX": function_x,
            "runId": run["runId"],
            "epochGaps": run["epochGaps"],
            "maskedOccurrenceFraction": {
                branch: run["maskStatistics"][branch]["maskedOccurrenceFraction"]
                for branch in ("bigram", "trigram")
            },
            "wallSeconds": run["wallSeconds"],
            "host": run["host"],
        })
    return {
        "metric": "online validation loss - writer train loss",
        "timing": "pre_optimizer_step",
        "maskRule": "bigram and trigram exact contexts with train hit count <= x contribute no output and receive no update",
        "epochEnds": [
            {"epoch": epoch, "step": step} for epoch, step in FREQUENCY_MASK_EPOCH_ENDS
        ],
        "indexSha256": FREQUENCY_MASK_INDEX_SHA256,
        "axis": {
            "defaultScale": "log1p",
            "allEquivalentThreshold": all_equivalent_threshold,
            "noneControlX": 0,
        },
        "points": points,
        "bridgeAudit": {
            "primaryRunId": primary["runId"],
            "primaryHost": primary["host"],
            "bridgeRunId": bridge["runId"],
            "bridgeHost": bridge["host"],
            "primaryEpochGaps": primary["epochGaps"],
            "bridgeEpochGaps": bridge["epochGaps"],
            "bridgeMinusPrimary": bridge_delta,
            "maxAbsoluteDelta": max(abs(value) for value in bridge_delta.values()),
        },
    }


def build_stage3r_analysis(runs: list[dict]) -> dict:
    prefix_series = (
        "online_loss", "train_log", "table_norm", "online_frequency", "fixed_gram"
    )
    require(len(runs) == 4, "Stage 3R requires a strict 2x2 order-control matrix")
    expected_modes = {
        "original": {"sequential", "sequential_then_reshuffle"},
        "random": {"frozen_permutation", "epoch_reshuffle"},
    }
    pairing_audits = {}
    for group, modes in expected_modes.items():
        pair = [run for run in runs if run["comparison_group"] == group]
        require(len(pair) == 2, f"Stage 3R {group} pair is incomplete")
        by_mode = {run["order_mode"]: run for run in pair}
        require(set(by_mode) == modes, f"Stage 3R {group} modes differ")
        left, right = pair
        require(
            left["order_hashes"][0] == right["order_hashes"][0],
            f"Stage 3R {group} epoch-1 orders do not match",
        )
        shared_hashes = {
            run["meta"]["checkpoint_resume"]["shared_parameter_state_sha256"]
            for run in pair
        }
        require(
            len(shared_hashes) == 1,
            f"Stage 3R {group} branches do not share one parameter state",
        )
        for name in prefix_series:
            inherited = [
                [row for row in run[name] if row["step"] <= 337]
                for run in pair
            ]
            require(
                inherited[0] == inherited[1],
                f"Stage 3R {group} inherited {name} prefix is not identical",
            )
        pairing_audits[group] = {
            "epoch1OrderHash": left["order_hashes"][0],
            "sharedParameterStateSha256": next(iter(shared_hashes)),
            "forkStep": 337,
            "prefixRowsIdentical": True,
        }
    require(
        pairing_audits["original"]["epoch1OrderHash"]
        != pairing_audits["random"]["epoch1OrderHash"],
        "Stage 3R original and random epoch-1 orders unexpectedly match",
    )
    require(
        pairing_audits["original"]["sharedParameterStateSha256"]
        != pairing_audits["random"]["sharedParameterStateSha256"],
        "Stage 3R distinct epoch-1 orders unexpectedly produced one parameter state",
    )

    epoch_starts = (338, 675)
    online_rows = []
    fixed_gram_rows = []
    for run in runs:
        online_by_step = {row["step"]: row for row in run["online_frequency"]}
        fixed_by_step = {row["step"]: row for row in run["fixed_gram"]}
        for start in epoch_starts:
            gaps = {
                step: online_by_step[step]["online_val_loss"]
                - online_by_step[step]["train_writer_loss"]
                for step in range(start - 10, start + 10)
            }
            immediate = gaps[start] - gaps[start - 1]
            window_jump = fmean(gaps[step] for step in range(start, start + 10)) - fmean(
                gaps[step] for step in range(start - 10, start)
            )
            online_rows.append(
                {
                    "runId": run["run_id"],
                    "comparisonGroup": run["comparison_group"],
                    "orderMode": run["order_mode"],
                    "epochStart": start,
                    "immediateJump": immediate,
                    "mean10Jump": window_jump,
                }
            )
            before = fixed_by_step[start - 5]
            after = fixed_by_step[start + 5]
            fixed_gram_rows.append(
                {
                    "runId": run["run_id"],
                    "comparisonGroup": run["comparison_group"],
                    "orderMode": run["order_mode"],
                    "epochStart": start,
                    "relativeBefore": -5,
                    "relativeAfter": 5,
                    "branches": {
                        branch: {
                            bucket: after["branches"][branch][bucket]["gap_contribution"]
                            - before["branches"][branch][bucket]["gap_contribution"]
                            if before["branches"][branch][bucket]["gap_contribution"] is not None
                            and after["branches"][branch][bucket]["gap_contribution"] is not None
                            else None
                            for bucket in BUCKETS
                        }
                        for branch in ("bigram", "trigram")
                    },
                }
            )
    return {
        "epochStarts": list(epoch_starts),
        "onlineEdgeMetrics": online_rows,
        "fixedGramEdgeChanges": fixed_gram_rows,
        "pairingAudit": pairing_audits,
    }


def build_lr_analysis(runs: list[dict]) -> dict:
    analysis_betas = (0.500, 0.900, 0.999)
    probe_windows = {
        "first": {"before": 164, "read_start": 169, "read_end": 172, "after": 174},
        "second": {"before": 501, "read_start": 506, "read_end": 509, "after": 511},
    }
    groups = {}
    for beta2 in analysis_betas:
        selected = sorted(
            (
                run for run in runs
                if abs(run["beta2"] - beta2) < 1e-12 and run["lr_scale"] <= 1.0
            ),
            key=lambda run: run["lr_scale"],
        )
        expected_count = 9 if beta2 == 0.999 else 3
        require(
            len(selected) == expected_count,
            f"beta2={beta2}: expected {expected_count} LR-analysis points",
        )
        require(
            len({run["lr_scale"] for run in selected}) == expected_count,
            f"beta2={beta2}: duplicate LR-analysis point",
        )
        points = []
        for run in selected:
            online_row = value_at(run["online_frequency"], 674, run["run_id"])
            online_gap = online_row["online_val_loss"] - online_row["train_writer_loss"]
            for branch in ("bigram", "trigram"):
                decomposed_gap = sum(
                    online_row["gap_contribution"][branch][bucket]["contribution"]
                    for bucket in BUCKETS
                )
                require(
                    abs(decomposed_gap - online_gap) < 1e-6,
                    f"{run['run_id']}: online {branch} contributions do not sum to gap",
                )

            probe_changes = {}
            for name, window in probe_windows.items():
                before = value_at(run["fixed_frequency"], window["before"], run["run_id"])
                after = value_at(run["fixed_frequency"], window["after"], run["run_id"])
                for checkpoint in (before, after):
                    checkpoint_gap = checkpoint["fixed_val_loss"] - checkpoint["fixed_train_loss"]
                    for branch in ("bigram", "trigram"):
                        decomposed_gap = sum(
                            checkpoint["gap_contribution"][branch][bucket]["contribution"]
                            for bucket in BUCKETS
                        )
                        require(
                            abs(decomposed_gap - checkpoint_gap) < 1e-6,
                            f"{run['run_id']}: fixed {branch} contributions do not sum to gap",
                        )
                probe_changes[name] = {
                    branch: {
                        bucket: {
                            "before": before["gap_contribution"][branch][bucket]["contribution"],
                            "after": after["gap_contribution"][branch][bucket]["contribution"],
                            "delta": (
                                after["gap_contribution"][branch][bucket]["contribution"]
                                - before["gap_contribution"][branch][bucket]["contribution"]
                            ),
                        }
                        for bucket in BUCKETS
                    }
                    for branch in ("bigram", "trigram")
                }
            points.append(
                {
                    "runId": run["run_id"],
                    "lrScale": run["lr_scale"],
                    "tableBaseLr": 0.004 * run["lr_scale"],
                    "onlineGapStep674": online_gap,
                    "probeRead": probe_changes,
                }
            )
        groups[f"{beta2:.3f}"] = {"beta2": beta2, "points": points}
    return {"epoch2Step": 674, "probeWindows": probe_windows, "groups": groups}


def load_historical_snapshot(path: Path) -> dict:
    require(path.is_file() and path.stat().st_size > 0, f"missing historical snapshot {path}")
    snapshot = read_json(path)
    require(snapshot.get("schema_version") == 1, "unsupported historical snapshot schema")
    charts = snapshot.get("charts", {})
    require(set(charts) == HISTORICAL_CHARTS, "historical snapshot must contain seven charts")
    expected_trace_counts = {
        "injection_gap": 3,
        "injection_loss": 6,
        "table_norm": 2,
        "input_alignment": 3,
    }
    for name, count in expected_trace_counts.items():
        require(len(charts[name].get("traces", [])) == count, f"{name}: expected {count} traces")
    for name in ("frequency_bins", "hitcount_distribution", "gap_vs_frequency_log"):
        series = charts[name].get("series", {})
        require(set(series) == {"bigram", "trigram"}, f"{name}: missing branch")
        require(
            len(series["bigram"]) == len(BUCKETS) and len(series["trigram"]) == len(BUCKETS),
            f"{name}: incomplete bucket coverage",
        )
    return snapshot


def table_cell(value: str, class_name: str = "") -> str:
    class_attribute = f' class="{class_name}"' if class_name else ""
    return f"<td{class_attribute}>{html.escape(value)}</td>"


def format_lr_scale(value: float) -> str:
    return f"{value:.3f}"


def format_table_lr(value: float) -> str:
    return f"{0.004 * value:.6f}"


def render_tables(runs: list[dict]) -> tuple[str, str]:
    convergence_rows = []
    edge_rows = []
    for run in runs:
        final = run["train_log"][-1]
        rms_values = [
            value for key, value in run["table_norm"][-1].items() if key.endswith(".rms")
        ]
        require(bool(rms_values), f"{run['run_id']}: missing final table RMS")
        convergence_rows.append(
            f'<tr tabindex="0" data-run-id="{html.escape(run["run_id"])}">'
            + table_cell(f"{run['beta2']:.3f}")
            + table_cell(format_lr_scale(run["lr_scale"]))
            + table_cell(format_table_lr(run["lr_scale"]))
            + table_cell(f"{final['train_loss']:.4f}")
            + table_cell(f"{final['val_loss']:.4f}")
            + table_cell(
                f"{final['gap']:+.4f}",
                "positive" if final["gap"] > 0 else "negative",
            )
            + table_cell(f"{fmean(rms_values):.4f}")
            + table_cell(run["run_id"])
            + "</tr>"
        )
        for metric in run["edges"]:
            cells = [
                table_cell(f"{run['beta2']:.3f}"),
                table_cell(format_lr_scale(run["lr_scale"])),
                table_cell(str(metric["edge"])),
            ]
            for name in (
                "bigram_low", "bigram_high", "bigram_tilt",
                "trigram_low", "trigram_high", "trigram_tilt",
            ):
                value = metric[name]
                cells.append(
                    table_cell(
                        f"{value:+.4f}",
                        "positive" if value > 0 else "negative" if value < 0 else "",
                    )
                )
            edge_rows.append(
                f'<tr tabindex="0" data-run-id="{html.escape(run["run_id"])}">'
                + "".join(cells)
                + "</tr>"
            )
    return "".join(convergence_rows), "".join(edge_rows)


def render_run_options(runs: list[dict], default_run_id: str) -> str:
    options = []
    for run in runs:
        selected = " selected" if run["run_id"] == default_run_id else ""
        label = (
            f"beta2={run['beta2']:.3f} · LR x{format_lr_scale(run['lr_scale'])}"
            f" · {run['run_id']}"
        )
        options.append(
            f'<option value="{html.escape(run["run_id"])}"{selected}>{html.escape(label)}</option>'
        )
    return "".join(options)


def stage3r_label(comparison_group: str, order_mode: str) -> str:
    labels = {
        ("original", "sequential"): "epoch 1 original order · no shuffle",
        ("original", "sequential_then_reshuffle"): "epoch 1 original order · shuffle",
        ("random", "frozen_permutation"): "epoch 1 random order · no shuffle",
        ("random", "epoch_reshuffle"): "epoch 1 random order · shuffle",
    }
    return labels[(comparison_group, order_mode)]


def render_stage3r_options(runs: list[dict]) -> str:
    return "".join(
        f'<option value="{html.escape(run["run_id"])}"'
        f'{" selected" if run["run_id"] == STAGE3R_DEFAULT_RUN_ID else ""}>'
        f'{html.escape(stage3r_label(run["comparison_group"], run["order_mode"]))} · {html.escape(run["run_id"])}</option>'
        for run in runs
    )


def render_stage3r_loss_toggles(runs: list[dict]) -> str:
    return "".join(
        f'<label class="inline-toggle" for="stage3r-loss-toggle-{index}">'
        f'<input type="checkbox" id="stage3r-loss-toggle-{index}" '
        f'class="stage3r-loss-toggle" data-run-id="{html.escape(run["run_id"])}" checked> '
        f'{html.escape(stage3r_label(run["comparison_group"], run["order_mode"]))}</label>'
        for index, run in enumerate(runs)
    )


def render_stage3r_tables(runs: list[dict], analysis: dict) -> tuple[str, str]:
    convergence_rows = []
    for run in runs:
        final = run["train_log"][-1]
        rms_values = [
            value for key, value in run["table_norm"][-1].items() if key.endswith(".rms")
        ]
        require(bool(rms_values), f"{run['run_id']}: missing final table RMS")
        convergence_rows.append(
            f'<tr tabindex="0" data-run-id="{html.escape(run["run_id"])}">'
            + table_cell(stage3r_label(run["comparison_group"], run["order_mode"]))
            + table_cell(f"{final['train_loss']:.4f}")
            + table_cell(f"{final['val_loss']:.4f}")
            + table_cell(
                f"{final['gap']:+.4f}",
                "positive" if final["gap"] > 0 else "negative",
            )
            + table_cell(f"{fmean(rms_values):.4f}")
            + table_cell(run["order_hashes"][0][:12] + "…")
            + table_cell(run["run_id"])
            + "</tr>"
        )

    fixed_by_key = {
        (row["runId"], row["epochStart"]): row
        for row in analysis["fixedGramEdgeChanges"]
    }
    edge_rows = []
    for metric in analysis["onlineEdgeMetrics"]:
        fixed = fixed_by_key[(metric["runId"], metric["epochStart"])]
        cells = [
            table_cell(stage3r_label(metric["comparisonGroup"], metric["orderMode"])),
            table_cell(str(metric["epochStart"])),
            table_cell(
                f"{metric['immediateJump']:+.4f}",
                "positive" if metric["immediateJump"] > 0 else "negative",
            ),
            table_cell(
                f"{metric['mean10Jump']:+.4f}",
                "positive" if metric["mean10Jump"] > 0 else "negative",
            ),
        ]
        for branch in ("bigram", "trigram"):
            values = fixed["branches"][branch]
            low = fmean(values[bucket] for bucket in LOW_BUCKETS)
            high = fmean(values[bucket] for bucket in HIGH_BUCKETS)
            for value in (low, high, low - high):
                cells.append(
                    table_cell(
                        f"{value:+.4f}",
                        "positive" if value > 0 else "negative" if value < 0 else "",
                    )
                )
        edge_rows.append(
            f'<tr tabindex="0" data-run-id="{html.escape(metric["runId"])}">'
            + "".join(cells)
            + "</tr>"
        )
    return "".join(convergence_rows), "".join(edge_rows)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>n-gram gap 实验总报告</title>
<style>
:root { color-scheme: light; --ink:#202124; --muted:#5f6368; --line:#d9dde3; --panel:#fff; --accent:#3949ab; }
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; margin:0; background:#f7f8fa; color:var(--ink); }
header, main { width:min(1320px,calc(100% - 32px)); margin:0 auto; }
header { padding:28px 0 14px; }
h1 { margin:0 0 8px; font-size:1.65rem; }
h2 { margin:0 0 10px; font-size:1.32rem; }
h3 { margin:22px 0 8px; font-size:1.05rem; }
p { line-height:1.55; }
code { background:#eef0f4; padding:.1rem .3rem; border-radius:3px; }
.note { color:var(--muted); font-size:.9rem; max-width:1120px; }
.toc { position:sticky; top:0; z-index:10; background:rgba(247,248,250,.96); border-block:1px solid var(--line); backdrop-filter:blur(8px); }
.toc-inner { width:min(1320px,calc(100% - 32px)); margin:0 auto; display:flex; gap:8px; overflow-x:auto; padding:9px 0; }
.toc a { color:#334; text-decoration:none; white-space:nowrap; border:1px solid var(--line); border-radius:999px; padding:6px 12px; background:white; }
.chapter { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:20px; margin:22px 0; scroll-margin-top:62px; box-shadow:0 2px 8px rgba(32,33,36,.04); }
.chart { width:100%; min-height:440px; }
.chart-scroll { width:100%; overflow-x:auto; }
.chart-scroll .chart { min-width:960px; }
.controls { display:flex; flex-wrap:wrap; gap:12px; align-items:end; margin:14px 0 8px; }
.controls label { display:grid; gap:4px; min-width:0; max-width:100%; font-size:.88rem; color:var(--muted); }
.controls .inline-toggle { display:flex; align-items:center; gap:6px; padding:7px 9px; border:1px solid var(--line); border-radius:5px; background:#fafbfc; color:var(--ink); }
.inline-toggle input { margin:0; }
button, select, input[type="number"] { font:inherit; padding:6px 9px; border:1px solid #b7bdc7; border-radius:5px; background:white; color:var(--ink); }
select { max-width:100%; }
button { cursor:pointer; }
button.active { background:#30343b; color:white; border-color:#30343b; }
.table-wrap { width:100%; overflow-x:auto; }
.result-table { border-collapse:collapse; width:100%; font-size:.86rem; margin:8px 0 14px; }
.result-table th,.result-table td { border:1px solid #d2d6dc; padding:6px 8px; text-align:right; white-space:nowrap; }
.result-table th { background:#f0f2f5; font-weight:600; }
.result-table td:last-child { text-align:left; }
.result-table tbody tr { cursor:pointer; }
.result-table tbody tr:hover,.result-table tbody tr.selected { background:#eef2ff; }
.positive { color:#ab3b16; font-weight:600; }
.negative { color:#1565a8; font-weight:600; }
.run-meta { padding:8px 10px; background:#f4f6fb; border-left:3px solid var(--accent); font-size:.9rem; margin:8px 0; overflow-wrap:anywhere; }
footer { width:min(1320px,calc(100% - 32px)); margin:20px auto 32px; color:var(--muted); font-size:.85rem; }
@media (max-width:780px) { header,main,.toc-inner,footer { width:min(100% - 20px,1320px); } .chapter { padding:14px; } .chart { min-height:380px; } .controls label { flex:1 1 100%; width:100%; } .controls select { width:100%; min-width:0; } }
</style>
</head>
<body>
<header>
  <h1>n-gram gap 实验总报告</h1>
  <p class="note">一个入口汇总历史注入点消融、历史频率机制、RMSProp Stage 1/2A、Stage 3R reshuffle 与累计频率遮罩 sweep。历史图来自冻结数据快照；当前实验图表由本地 <code>data/runs</code> 原始输出严格校验后生成。</p>
</header>
<nav class="toc" aria-label="章节导航"><div class="toc-inner">
  <a href="#historical-injection">1. 历史注入点消融</a>
  <a href="#historical-frequency">2. 历史频率机制</a>
  <a href="#rmsprop-stage1-results">3. RMSProp Stage 1</a>
  <a href="#rmsprop-stage2a-results">4. RMSProp Stage 2A</a>
  <a href="#reshuffle-stage3r-results">5. Stage 3R reshuffle</a>
  <a href="#frequency-mask-sweep">6. 累计频率遮罩</a>
</div></nav>
<main>
<section class="chapter" id="historical-injection">
  <h2>1. 历史注入点消融</h2>
  <p class="note">v / y / input 三种注入点，vanilla nanoGPT + bigram/trigram + seed 42 + 1000 steps。竖线标记 fixed replay 边界。</p>
  <div id="historical-gap-chart" class="chart"></div>
  <div id="historical-loss-chart" class="chart"></div>
  <div id="historical-table-norm-chart" class="chart"></div>
  <div id="historical-input-alignment-chart" class="chart"></div>
</section>

<section class="chapter" id="historical-frequency">
  <h2>2. 历史频率机制</h2>
  <p class="note">历史 input run 的 train-frequency bucket 分解、命中频次分布和末态 gap。全部保留 hover 与分支切换。</p>
  <div class="controls">
    <label>context<select id="historical-frequency-branch"><option value="bigram">bigram</option><option value="trigram">trigram</option></select></label>
    <label>曲线<select id="historical-frequency-view"><option value="gap">gap (val - train)</option><option value="loss">train / val loss</option></select></label>
    <label>统计量<select id="historical-frequency-metric"><option value="per_token">per-token</option><option value="total">fraction x loss</option></select></label>
  </div>
  <div id="historical-frequency-chart" class="chart"></div>
  <div class="controls">
    <label>分布 context<select id="historical-distribution-branch"><option value="bigram">bigram</option><option value="trigram">trigram</option></select></label>
  </div>
  <div id="historical-hitcount-chart" class="chart"></div>
  <div class="controls">
    <label>log-frequency context<select id="historical-log-branch"><option value="both">both</option><option value="bigram">bigram</option><option value="trigram">trigram</option></select></label>
  </div>
  <div id="historical-log-gap-chart" class="chart"></div>
</section>

<section class="chapter" id="rmsprop-stage1-results">
  <h2>3. RMSProp Stage 1: beta2 x table learning rate</h2>
  <p class="note">所有条件使用 input injection、bigram + trigram、seed 42、one-shard fixed replay、1000 steps 和同一 fixed-gram manifest。仅改变 table RMSProp <code>beta2</code> 与 table LR scale。</p>
  <h3>Final online convergence</h3>
  <div class="table-wrap"><table class="result-table" id="stage-convergence-table">
    <thead><tr><th>table beta2</th><th>LR scale</th><th>table base LR</th><th>final train loss</th><th>final validation loss</th><th>final global gap</th><th>final mean table RMS</th><th>run</th></tr></thead>
    <tbody>__CONVERGENCE_ROWS__</tbody>
  </table></div>
  <h3>Fixed-gram bucket shape at replay edges</h3>
  <p class="note">每个 delta 为 <code>edge+5 - edge-5</code>。低频均值覆盖 1 至 11-20，高频均值覆盖 21-50 至 5k+；tilt = low - high。</p>
  <div class="table-wrap"><table class="result-table" id="stage-edge-table">
    <thead><tr><th>table beta2</th><th>LR scale</th><th>replay edge</th><th>bigram low Δ</th><th>bigram high Δ</th><th>bigram tilt</th><th>trigram low Δ</th><th>trigram high Δ</th><th>trigram tilt</th></tr></thead>
    <tbody>__EDGE_ROWS__</tbody>
  </table></div>
  <h3 id="stage-run-explorer">Condition explorer</h3>
  <div class="controls">
    <label>condition<select id="stage-run-select">__STAGE1_RUN_OPTIONS__</select></label>
    <label>online loss stride<select id="stage-loss-stride"><option value="1">1 step</option><option value="5">5 steps</option><option value="10" selected>10 steps</option><option value="25">25 steps</option><option value="50">50 steps</option></select></label>
    <label>frequency source<select id="stage-source"><option value="fixed_gram" selected>fixed gram sample</option><option value="online">online writer batch</option><option value="fixed">fixed batch probe</option></select></label>
    <label>context<select id="stage-branch"><option value="bigram">bigram</option><option value="trigram">trigram</option></select></label>
    <span><button type="button" class="experiment-metric" data-prefix="stage" data-metric="loss">per-token loss</button> <button type="button" class="experiment-metric active" data-prefix="stage" data-metric="gap">mean per-token gap</button></span>
  </div>
  <div id="stage-run-meta" class="run-meta"></div>
  <div id="stage-loss-chart" class="chart"></div>
  <div id="stage-frequency-chart" class="chart"></div>
  <p id="stage-status" class="note"></p>
</section>

<section class="chapter" id="rmsprop-stage2a-results">
  <h2>4. RMSProp Stage 2A: wider beta2 and low-LR sweep</h2>
  <p class="note">固定 Stage 1 setting，将 <code>beta2</code> 扩展到 0.5/0.9，并细分 LR scale 0 至 1。这里列出 13 个新增条件；Stage 1 已存在的参考条件不重复计数。所有运行使用同一 fixed-gram manifest。</p>
  <h3>Learning-rate response at fixed beta2</h3>
  <p class="note">第一张图使用 epoch 2 末尾 step 674 的 online gap：<code>online validation loss - writer train loss</code>，对应第 674 步 optimizer update 前的 online observable。后两张图使用 post-update fixed probe 的 bucket gap contribution，分别计算第一次阅读前后 step 164→174、第二次阅读前后 step 501→511 的变化；阅读窗口本身为 169–172 和 506–509。</p>
  <div class="controls">
    <label>fixed beta2<select id="stage2a-analysis-beta"><option value="0.999" selected>0.999</option><option value="0.900">0.900</option><option value="0.500">0.500</option></select></label>
    <label>probe context<select id="stage2a-analysis-branch"><option value="bigram">bigram</option><option value="trigram">trigram</option></select></label>
  </div>
  <div id="stage2a-gap-lr-chart" class="chart"></div>
  <p id="stage2a-gap-lr-fit" class="note"></p>
  <div id="stage2a-probe1-lr-freq-chart" class="chart"></div>
  <div id="stage2a-probe2-lr-freq-chart" class="chart"></div>
  <h3>Final online convergence</h3>
  <div class="table-wrap"><table class="result-table" id="stage2a-convergence-table">
    <thead><tr><th>table beta2</th><th>LR scale</th><th>table base LR</th><th>final train loss</th><th>final validation loss</th><th>final global gap</th><th>final mean table RMS</th><th>run</th></tr></thead>
    <tbody>__STAGE2A_CONVERGENCE_ROWS__</tbody>
  </table></div>
  <h3>Fixed-gram bucket shape at replay edges</h3>
  <p class="note">每个 delta 为 <code>edge+5 - edge-5</code>。低频均值覆盖 1 至 11-20，高频均值覆盖 21-50 至 5k+；tilt = low - high。</p>
  <div class="table-wrap"><table class="result-table" id="stage2a-edge-table">
    <thead><tr><th>table beta2</th><th>LR scale</th><th>replay edge</th><th>bigram low Δ</th><th>bigram high Δ</th><th>bigram tilt</th><th>trigram low Δ</th><th>trigram high Δ</th><th>trigram tilt</th></tr></thead>
    <tbody>__STAGE2A_EDGE_ROWS__</tbody>
  </table></div>
  <h3 id="stage2a-run-explorer">Condition explorer</h3>
  <div class="controls">
    <label>condition<select id="stage2a-run-select">__STAGE2A_RUN_OPTIONS__</select></label>
    <label>online loss stride<select id="stage2a-loss-stride"><option value="1">1 step</option><option value="5">5 steps</option><option value="10" selected>10 steps</option><option value="25">25 steps</option><option value="50">50 steps</option></select></label>
    <label>frequency source<select id="stage2a-source"><option value="fixed_gram" selected>fixed gram sample</option><option value="online">online writer batch</option><option value="fixed">fixed batch probe</option></select></label>
    <label>context<select id="stage2a-branch"><option value="bigram">bigram</option><option value="trigram">trigram</option></select></label>
    <span><button type="button" class="experiment-metric" data-prefix="stage2a" data-metric="loss">per-token loss</button> <button type="button" class="experiment-metric active" data-prefix="stage2a" data-metric="gap">mean per-token gap</button></span>
  </div>
  <div id="stage2a-run-meta" class="run-meta"></div>
  <div id="stage2a-loss-chart" class="chart"></div>
  <div id="stage2a-frequency-chart" class="chart"></div>
  <p id="stage2a-status" class="note"></p>
</section>

<section class="chapter" id="reshuffle-stage3r-results">
  <h2>5. Stage 3R: epoch-1 order × later-epoch shuffle</h2>
  <p class="note">严格 2×2 对照：epoch 1 使用原始 <code>0…336</code> 顺序或 seed-101 随机顺序；之后的 epoch 固定复用 epoch-1 顺序，或每个 epoch 重新 shuffle。model/data seed 均为 42，order seed 为 101，其他设置与 baseline 相同。每一行的 shuffle/no-shuffle 两分支都从对应的 step-337 完整 post-update checkpoint 分叉；fixed probe 均关闭。</p>
  <p id="stage3r-pairing-audit" class="note"></p>
  <h3>Final online convergence</h3>
  <div class="table-wrap"><table class="result-table" id="stage3r-convergence-table">
    <thead><tr><th>order strategy</th><th>final train loss</th><th>final validation loss</th><th>final global gap</th><th>final mean table RMS</th><th>epoch-1 order hash</th><th>run</th></tr></thead>
    <tbody>__STAGE3R_CONVERGENCE_ROWS__</tbody>
  </table></div>
  <h3>Replay-edge response</h3>
  <p class="note">online immediate jump = <code>G(start)-G(start-1)</code>；online mean-10 jump = 新 epoch 前后各 10 步均值之差。fixed-gram delta 使用 <code>r=+5 minus r=-5</code>；low/high/tilt 的 bucket 定义与前两阶段一致。</p>
  <div class="table-wrap"><table class="result-table" id="stage3r-edge-table">
    <thead><tr><th>order strategy</th><th>epoch start</th><th>online immediate Δ</th><th>online mean-10 Δ</th><th>bigram low Δ</th><th>bigram high Δ</th><th>bigram tilt</th><th>trigram low Δ</th><th>trigram high Δ</th><th>trigram tilt</th></tr></thead>
    <tbody>__STAGE3R_EDGE_ROWS__</tbody>
  </table></div>
  <h3>Complete online loss</h3>
  <p class="note">每组实验有独立开关。writer train loss 展示全部 1000 个 optimizer step，不做 stride 或平滑；validation loss 为每 50 step 的四个 validation batch 均值。同一种 epoch-1 顺序下，shuffle/no-shuffle 的 step 1–337 曲线严格重合。</p>
  <div class="controls" id="stage3r-loss-toggles">__STAGE3R_LOSS_TOGGLES__</div>
  <div id="stage3r-full-online-loss-chart" class="chart"></div>
  <h3 id="stage3r-run-explorer">Condition explorer</h3>
  <div class="controls">
    <label>condition<select id="stage3r-run-select">__STAGE3R_RUN_OPTIONS__</select></label>
    <label>online loss stride<select id="stage3r-loss-stride"><option value="1">1 step</option><option value="5">5 steps</option><option value="10" selected>10 steps</option><option value="25">25 steps</option><option value="50">50 steps</option></select></label>
    <label>frequency source<select id="stage3r-source"><option value="online" selected>online writer batch</option><option value="fixed_gram">fixed gram sample</option></select></label>
    <label>context<select id="stage3r-branch"><option value="bigram">bigram</option><option value="trigram">trigram</option></select></label>
    <label class="inline-toggle" for="stage3r-dense-sampling"><input type="checkbox" id="stage3r-dense-sampling" checked> 显示边界密集采样；关闭后仅保留 50-step 常规点和 epoch-start 点</label>
    <span><button type="button" class="experiment-metric" data-prefix="stage3r" data-metric="loss">per-token loss</button> <button type="button" class="experiment-metric active" data-prefix="stage3r" data-metric="gap">mean per-token gap</button></span>
  </div>
  <div id="stage3r-run-meta" class="run-meta"></div>
  <div id="stage3r-loss-chart" class="chart"></div>
  <div id="stage3r-global-gap-chart" class="chart"></div>
  <div id="stage3r-frequency-chart" class="chart"></div>
  <p id="stage3r-status" class="note"></p>
</section>

<section class="chapter" id="frequency-mask-sweep">
  <h2>6. Cumulative exact-frequency masking sweep</h2>
  <p class="note">在最小 input-injection setting 上同时遮罩 bigram 和 trigram：训练语料精确命中频次 <code>freq ≤ x</code> 的 context 不产生 n-gram 输出，也不更新 table。每个条件固定 sequential replay 三个 epoch、seed 42；关闭 fixed probe、fixed-gram sample 与频率分桶评估。三条曲线分别取 epoch 末尾 step 337、674、1011 的 pre-update online gap（moving validation loss − writer train loss）。主曲线使用连续数值坐标；<code>all</code> 等效放在联合最大命中频次 <code>x=195,964</code>，而 <code>none</code> 仅作为 <code>x=0</code> 处的空心对照点，不参与函数连线。</p>
  <div class="controls">
    <label>curve mode<select id="frequency-mask-y-view"><option value="absolute" selected>3 lines: epoch 1 / 2 / 3 end</option><option value="increment">2 lines: epoch 2−1 / epoch 3−1</option></select></label>
    <label>show measured thresholds from 0 to x<input type="number" id="frequency-mask-x-max" min="0" step="1" value="210" inputmode="numeric"></label>
    <label>x-axis scale<select id="frequency-mask-x-scale"><option value="log1p" selected>continuous log10(x + 1)</option><option value="linear">continuous linear x</option></select></label>
  </div>
  <div class="chart-scroll"><div id="frequency-mask-gap-chart" class="chart"></div></div>
  <p id="frequency-mask-view-status" class="note"></p>
  <p id="frequency-mask-bridge-audit" class="note"></p>
</section>
</main>
<footer>Generated deterministically by <code>docs/generate_report.py</code>. Historical data: <code>docs/data/historical-figures.json</code>. Current experiment data: <code>data/runs/</code>.</footer>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
const historical = __HISTORICAL_DATA__;
const experimentReports = {stage:__STAGE1_DATA__,stage2a:__STAGE2A_DATA__,stage3r:__STAGE3R_DATA__};
const lrAnalysis = __LR_ANALYSIS_DATA__;
const stage3rAnalysis = __STAGE3R_ANALYSIS_DATA__;
const frequencyMaskAnalysis = __FREQUENCY_MASK_ANALYSIS_DATA__;
const buckets = __BUCKETS__;
const bucketColors = __COLORS__;
const plotConfig = {responsive:true, displaylogo:false};
const clone = value => JSON.parse(JSON.stringify(value));
const replayShapes = [337,686].map(step => ({type:"line",x0:step,x1:step,y0:0,y1:1,yref:"paper",line:{color:"#aaa",dash:"dot",width:1}}));

function renderHistoricalInjection() {
  Plotly.newPlot("historical-gap-chart", clone(historical.charts.injection_gap.traces), {title:"Train/validation gap (validation - train)",xaxis:{title:"step"},yaxis:{title:"gap",zeroline:true},shapes:replayShapes,margin:{l:65,r:30,t:50,b:52}}, plotConfig);
  Plotly.newPlot("historical-loss-chart", clone(historical.charts.injection_loss.traces), {title:"Train/validation loss",xaxis:{title:"step"},yaxis:{title:"loss"},shapes:replayShapes,margin:{l:65,r:30,t:50,b:52}}, plotConfig);
  const norms = clone(historical.charts.table_norm.traces).map((trace,index) => ({...trace,mode:"lines",line:{color:index===0?"#2196F3":"#F44336",width:2}}));
  Plotly.newPlot("historical-table-norm-chart", norms, {title:"N-gram table parameter RMS (input run)",xaxis:{title:"step"},yaxis:{title:"RMS"},shapes:replayShapes,margin:{l:65,r:30,t:50,b:52}}, plotConfig);
  const alignment = clone(historical.charts.input_alignment.traces).map((trace,index) => ({...trace,mode:"lines",yaxis:index===2?"y2":"y",line:{color:["#4CAF50","#FF9800","#9C27B0"][index],width:index===0?1.5:2,dash:index===0?"dash":"solid"}}));
  Plotly.newPlot("historical-input-alignment-chart", alignment, {title:"Loss + gap alignment (input run)",xaxis:{title:"step"},yaxis:{title:"loss"},yaxis2:{title:"gap",side:"right",overlaying:"y"},shapes:replayShapes,margin:{l:65,r:65,t:50,b:52}}, plotConfig);
}

function renderHistoricalFrequency() {
  const branch = document.getElementById("historical-frequency-branch").value;
  const view = document.getElementById("historical-frequency-view").value;
  const metric = document.getElementById("historical-frequency-metric").value;
  const data = historical.charts.frequency_bins.series[branch];
  const traces = [];
  buckets.forEach((bucket,index) => {
    const row = data[bucket];
    if (!row || !row.steps.length) return;
    const gap = row.val_loss.map((value,j) => value-row.train_loss[j]);
    if (view === "gap") {
      traces.push({x:row.steps,y:gap,mode:"lines+markers",name:bucket,line:{color:bucketColors[index],width:2},hovertemplate:`bucket=${bucket}<br>step=%{x}<br>gap=%{y:.5f}<extra></extra>`});
    } else if (metric === "total") {
      traces.push({x:row.steps,y:row.val_loss.map((value,j)=>value*row.val_frac[j]),mode:"lines+markers",name:bucket+" (val)",line:{color:bucketColors[index],width:2},hovertemplate:`bucket=${bucket}<br>step=%{x}<br>val fraction x loss=%{y:.5f}<extra></extra>`});
      traces.push({x:row.steps,y:row.train_loss.map((value,j)=>value*row.train_frac[j]),mode:"lines",name:bucket+" (train)",visible:"legendonly",line:{color:bucketColors[index],width:1,dash:"dash"}});
    } else {
      traces.push({x:row.steps,y:row.val_loss,mode:"lines+markers",name:bucket+" (val)",line:{color:bucketColors[index],width:2}});
      traces.push({x:row.steps,y:row.train_loss,mode:"lines",name:bucket+" (train)",visible:"legendonly",line:{color:bucketColors[index],width:1,dash:"dash"}});
    }
  });
  const title = `Historical frequency bins · ${branch} · ${view === "gap" ? "gap (val - train)" : metric === "total" ? "fraction x loss" : "per-token loss"}`;
  Plotly.react("historical-frequency-chart", traces, {title,xaxis:{title:"step"},yaxis:{title:view==="gap"?"gap":metric==="total"?"fraction x loss":"loss",zeroline:view==="gap"},shapes:replayShapes,legend:{x:.01,y:.99,font:{size:9}},margin:{l:70,r:30,t:50,b:54}}, plotConfig);
}

function renderHistoricalDistribution() {
  const branch = document.getElementById("historical-distribution-branch").value;
  const rows = historical.charts.hitcount_distribution.series[branch];
  const custom = rows.map(row => [row.train_loss,row.val_loss,row.gap]);
  const traces = [
    {x:rows.map(row=>row.bucket),y:rows.map(row=>row.train_frac),customdata:custom,type:"bar",name:"train fraction",marker:{color:"#2d6f9f"},hovertemplate:"bucket=%{x}<br>train fraction=%{y:.2%}<br>train loss=%{customdata[0]:.3f}<br>val loss=%{customdata[1]:.3f}<br>gap=%{customdata[2]:.3f}<extra></extra>"},
    {x:rows.map(row=>row.bucket),y:rows.map(row=>row.val_frac),customdata:custom,type:"bar",name:"validation fraction",marker:{color:"#c4493d"},hovertemplate:"bucket=%{x}<br>validation fraction=%{y:.2%}<br>train loss=%{customdata[0]:.3f}<br>val loss=%{customdata[1]:.3f}<br>gap=%{customdata[2]:.3f}<extra></extra>"},
    {x:rows.map(row=>row.bucket),y:rows.map(row=>row.gap),mode:"lines+markers",name:"final gap",line:{color:"#353d79",width:3},yaxis:"y2",hovertemplate:"bucket=%{x}<br>final gap=%{y:.3f}<extra></extra>"}
  ];
  Plotly.react("historical-hitcount-chart", traces, {title:`Hit-count distribution + final gap · ${branch}`,barmode:"group",xaxis:{title:"hit-count bucket",type:"category",categoryorder:"array",categoryarray:buckets,tickangle:-42},yaxis:{title:"token fraction",rangemode:"tozero"},yaxis2:{title:"final gap",side:"right",overlaying:"y",zeroline:true},margin:{l:70,r:85,t:50,b:90}}, plotConfig);
}

function historicalLogTrace(branch) {
  const rows = historical.charts.gap_vs_frequency_log.series[branch];
  const color = branch === "bigram" ? "#353d79" : "#c4493d";
  return {x:rows.map(row=>row.x),y:rows.map(row=>row.gap),customdata:rows.map(row=>[row.bucket,row.x_low,row.x_high,row.train_frac,row.val_frac,row.train_loss,row.val_loss]),mode:"lines+markers",name:branch,line:{color,width:2.5},marker:{color,size:8},error_x:{type:"data",symmetric:false,array:rows.map(row=>row.x_high-row.x),arrayminus:rows.map(row=>row.x-row.x_low),color,thickness:1.2,width:4},hovertemplate:"bucket=%{customdata[0]}<br>frequency range=%{customdata[1]}-%{customdata[2]}<br>gap=%{y:.3f}<br>train fraction=%{customdata[3]:.2%}<br>validation fraction=%{customdata[4]:.2%}<br>train loss=%{customdata[5]:.3f}<br>validation loss=%{customdata[6]:.3f}<extra></extra>"};
}

function renderHistoricalLog() {
  const branch = document.getElementById("historical-log-branch").value;
  const traces = [historicalLogTrace("bigram"),historicalLogTrace("trigram")];
  traces[0].visible = branch === "trigram" ? "legendonly" : true;
  traces[1].visible = branch === "bigram" ? "legendonly" : true;
  Plotly.react("historical-log-gap-chart", traces, {title:"Final per-bucket gap vs training hit count",xaxis:{title:"training hit count (log scale)",type:"log",dtick:1},yaxis:{title:"final gap",zeroline:true},margin:{l:70,r:30,t:50,b:65}}, plotConfig);
}

function linearFit(points) {
  const n=points.length,meanX=points.reduce((sum,row)=>sum+row.lrScale,0)/n,meanY=points.reduce((sum,row)=>sum+row.onlineGapStep674,0)/n;
  const denominator=points.reduce((sum,row)=>sum+(row.lrScale-meanX)**2,0);
  const slope=points.reduce((sum,row)=>sum+(row.lrScale-meanX)*(row.onlineGapStep674-meanY),0)/denominator;
  const intercept=meanY-slope*meanX;
  const residual=points.reduce((sum,row)=>sum+(row.onlineGapStep674-(slope*row.lrScale+intercept))**2,0);
  const total=points.reduce((sum,row)=>sum+(row.onlineGapStep674-meanY)**2,0);
  return {slope,intercept,r2:total===0?1:1-residual/total};
}
function currentLrAnalysisGroup() { return lrAnalysis.groups[document.getElementById("stage2a-analysis-beta").value]; }
function renderGapLrAnalysis() {
  const group=currentLrAnalysisGroup(),points=group.points,fit=linearFit(points);
  const minimum=points[0].lrScale,maximum=points.at(-1).lrScale,fitX=[minimum,maximum],fitY=fitX.map(x=>fit.slope*x+fit.intercept);
  Plotly.react("stage2a-gap-lr-chart",[
    {x:points.map(row=>row.lrScale),y:points.map(row=>row.onlineGapStep674),customdata:points.map(row=>[row.runId,row.tableBaseLr]),mode:"lines+markers",name:"observed online gap",line:{color:"#3949ab",width:2},marker:{size:8},hovertemplate:"LR scale=%{x:.3f}<br>table LR=%{customdata[1]:.6f}<br>online val - train=%{y:.6f}<br>%{customdata[0]}<extra></extra>"},
    {x:fitX,y:fitY,mode:"lines",name:"linear least-squares fit",line:{color:"#d84315",width:2,dash:"dash"},hovertemplate:"linear fit=%{y:.6f}<extra></extra>"}
  ],{title:`Epoch-2 online gap vs table LR · beta2=${group.beta2.toFixed(3)} · step ${lrAnalysis.epoch2Step}`,xaxis:{title:"table LR scale",rangemode:"tozero"},yaxis:{title:"online validation loss - writer train loss",zeroline:true},margin:{l:80,r:30,t:55,b:58},legend:{x:.02,y:.98}},plotConfig);
  const caution=points.length<5?"（点数较少，仅作描述）":"（仅作经验描述，不代表优化动力学中的固定公式）";
  document.getElementById("stage2a-gap-lr-fit").textContent=`线性拟合：gap = ${fit.slope.toFixed(4)} × LR scale ${fit.intercept>=0?"+":"-"} ${Math.abs(fit.intercept).toFixed(4)}，R² = ${fit.r2.toFixed(4)} ${caution}`;
}
function renderProbeLrFrequency(readName,chartId,ordinal) {
  const group=currentLrAnalysisGroup(),branch=document.getElementById("stage2a-analysis-branch").value,window=lrAnalysis.probeWindows[readName];
  const traces=buckets.map((bucket,index)=>({
    x:group.points.map(row=>row.lrScale),
    y:group.points.map(row=>row.probeRead[readName][branch][bucket].delta),
    customdata:group.points.map(row=>[row.runId,row.tableBaseLr,row.probeRead[readName][branch][bucket].before,row.probeRead[readName][branch][bucket].after]),
    mode:"lines+markers",name:bucket,line:{color:bucketColors[index],width:1.8},marker:{size:6},
    hovertemplate:`bucket=${bucket}<br>LR scale=%{x:.3f}<br>table LR=%{customdata[1]:.6f}<br>before=%{customdata[2]:+.6f}<br>after=%{customdata[3]:+.6f}<br>after - before=%{y:+.6f}<br>%{customdata[0]}<extra></extra>`
  }));
  Plotly.react(chartId,traces,{title:`${ordinal} fixed-probe read · ${branch} · gap-contribution change (${window.before} → ${window.after}) · beta2=${group.beta2.toFixed(3)}`,xaxis:{title:"table LR scale",rangemode:"tozero"},yaxis:{title:"Δ bucket gap contribution (after - before)",zeroline:true},margin:{l:82,r:30,t:55,b:58},legend:{x:.01,y:.99,font:{size:9}}},plotConfig);
}
function renderLrAnalysis() {
  renderGapLrAnalysis();
  renderProbeLrFrequency("first","stage2a-probe1-lr-freq-chart","First");
  renderProbeLrFrequency("second","stage2a-probe2-lr-freq-chart","Second");
}

function stage3rStrategyLabel(report) {
  return {
    "original:sequential":"epoch 1 original order · no shuffle",
    "original:sequential_then_reshuffle":"epoch 1 original order · shuffle",
    "random:frozen_permutation":"epoch 1 random order · no shuffle",
    "random:epoch_reshuffle":"epoch 1 random order · shuffle"
  }[`${report.comparisonGroup}:${report.orderMode}`];
}
function stage3rReports() {
  const rank={"original:sequential":0,"original:sequential_then_reshuffle":1,"random:frozen_permutation":2,"random:epoch_reshuffle":3};
  return Object.values(experimentReports.stage3r).sort((a,b)=>rank[`${a.comparisonGroup}:${a.orderMode}`]-rank[`${b.comparisonGroup}:${b.orderMode}`]);
}
const stage3rColors={"original:sequential":"#2962ff","original:sequential_then_reshuffle":"#d84315","random:frozen_permutation":"#00897b","random:epoch_reshuffle":"#8e24aa"};
function stage3rColor(report) { return stage3rColors[`${report.comparisonGroup}:${report.orderMode}`]; }
function renderStage3rFullOnlineLoss() {
  const traces=[];
  const selected=new Set([...document.querySelectorAll(".stage3r-loss-toggle:checked")].map(input=>input.dataset.runId));
  stage3rReports().filter(report=>selected.has(report.runId)).forEach(report=>{
    const color=stage3rColor(report),label=stage3rStrategyLabel(report);
    traces.push({x:report.onlineLoss.map(row=>row.step),y:report.onlineLoss.map(row=>row.train_writer_loss),customdata:report.onlineLoss.map(row=>[report.runId,row.epoch]),mode:"lines",name:`${label} · writer train`,line:{color,width:1.35},hovertemplate:"step=%{x} (epoch %{customdata[1]})<br>writer train loss=%{y:.6f}<br>%{customdata[0]}<extra></extra>"});
    traces.push({x:report.validation.map(row=>row.step),y:report.validation.map(row=>row.val_loss),customdata:report.validation.map(row=>[report.runId,row.epoch]),mode:"lines+markers",name:`${label} · validation`,line:{color,width:1.8,dash:"dash"},marker:{color,size:5},hovertemplate:"step=%{x} (epoch %{customdata[1]})<br>validation loss=%{y:.6f}<br>%{customdata[0]}<extra></extra>"});
  });
  const shapes=[337.5,674.5].map(step=>({type:"line",x0:step,x1:step,y0:0,y1:1,yref:"paper",line:{color:"#666",dash:"dot",width:1.4}}));
  const annotations=traces.length?[]:[{xref:"paper",yref:"paper",x:.5,y:.5,text:"请至少选择一组实验",showarrow:false,font:{size:16,color:"#777"}}];
  Plotly.react("stage3r-full-online-loss-chart",traces,{title:"Complete online loss · four order-control runs",xaxis:{title:"optimizer step"},yaxis:{title:"cross-entropy loss"},shapes,annotations,margin:{l:70,r:220,t:55,b:55},legend:{x:1.02,y:1,xanchor:"left",font:{size:9}}},plotConfig);
}
function renderStage3rAudit() {
  const original=stage3rAnalysis.pairingAudit.original,random=stage3rAnalysis.pairingAudit.random;
  document.getElementById("stage3r-pairing-audit").textContent=`严格配对审计：两套 pair 的 step 1–337 五类日志均逐行完全相同。原始顺序 pair 共享参数状态 ${original.sharedParameterStateSha256.slice(0,12)}…；随机顺序 pair 共享参数状态 ${random.sharedParameterStateSha256.slice(0,12)}…。两套实验均保持 data seed 42、order seed 101，shuffle 只从 step 338 起生效。`;
}

function renderFrequencyMaskSweep() {
  const points=frequencyMaskAnalysis.points;
  const xInput=document.getElementById("frequency-mask-x-max");
  const parsedMax=Number(xInput.value);
  const xMax=Number.isFinite(parsedMax)&&parsedMax>=0?Math.floor(parsedMax):210;
  const functionPoints=points.filter(row=>row.functionX!==null&&row.functionX<=xMax);
  const noneControl=points.find(row=>row.functionX===null);
  const scale=document.getElementById("frequency-mask-x-scale").value;
  const view=document.getElementById("frequency-mask-y-view").value;
  const project=x=>scale==="log1p"?Math.log10(x+1):x;
  const definitions=view==="absolute"?[
    {name:"epoch 1 end · step 337",color:"#00897b",value:row=>row.epochGaps["1"],target:"1"},
    {name:"epoch 2 end · step 674",color:"#3949ab",value:row=>row.epochGaps["2"],target:"2"},
    {name:"epoch 3 end · step 1011",color:"#d84315",value:row=>row.epochGaps["3"],target:"3"},
  ]:[
    {name:"epoch 2 end − epoch 1 end",color:"#5e35b1",value:row=>row.epochGaps["2"]-row.epochGaps["1"],target:"2"},
    {name:"epoch 3 end − epoch 1 end",color:"#ef6c00",value:row=>row.epochGaps["3"]-row.epochGaps["1"],target:"3"},
  ];
  const traces=definitions.map(definition=>({
    x:functionPoints.map(row=>project(row.functionX)),
    y:functionPoints.map(definition.value),
    customdata:functionPoints.map(row=>[
      row.label,
      row.functionX,
      row.runId,
      row.maskedOccurrenceFraction.bigram,
      row.maskedOccurrenceFraction.trigram,
      row.wallSeconds/60,
      row.host,
      row.epochGaps["1"],
      row.epochGaps[definition.target],
    ]),
    mode:"lines+markers",
    name:definition.name,
    line:{color:definition.color,width:2.2},
    marker:{color:definition.color,size:7},
    hovertemplate:`condition=%{customdata[0]}<br>continuous x=%{customdata[1]:,.0f}<br>${view==="absolute"?"epoch-end online gap":"displayed gap increment"}=%{y:+.6f}<br>epoch 1 end=%{customdata[7]:+.6f}<br>target epoch end=%{customdata[8]:+.6f}<br>masked bigram occurrences=%{customdata[3]:.2%}<br>masked trigram occurrences=%{customdata[4]:.2%}<br>wall time=%{customdata[5]:.2f} min<br>host=%{customdata[6]}<br>%{customdata[2]}<extra></extra>`,
  }));
  traces.push({
    x:definitions.map(()=>project(frequencyMaskAnalysis.axis.noneControlX)),
    y:definitions.map(definition=>definition.value(noneControl)),
    customdata:definitions.map(definition=>[definition.name,noneControl.runId,noneControl.host]),
    mode:"markers",name:"none control (not connected)",
    marker:{symbol:"diamond-open",size:11,color:"#424242",line:{width:2,color:"#424242"}},
    hovertemplate:`none control at x=0<br>%{customdata[0]}<br>${view==="absolute"?"online gap":"gap increment"}=%{y:+.6f}<br>host=%{customdata[2]}<br>%{customdata[1]}<extra></extra>`,
  });
  const tickThresholds=[0,1,2,5,10,20,50,100,200,500,1000,5000,20000,100000,frequencyMaskAnalysis.axis.allEquivalentThreshold].filter(x=>x<=xMax);
  const thresholdLabel=x=>x===frequencyMaskAnalysis.axis.allEquivalentThreshold?`all (${x.toLocaleString()})`:x>=1000?`${x/1000}k`:String(x);
  const xaxis=scale==="log1p"?{
    title:"continuous mask threshold x · log10(x + 1) display scale",
    type:"linear",range:[0,project(Math.max(xMax,1))],tickmode:"array",tickvals:tickThresholds.map(project),ticktext:tickThresholds.map(thresholdLabel),tickangle:-35,
  }:{
    title:"continuous mask threshold x",type:"linear",range:[0,Math.max(xMax,1)],tickformat:"~s",
  };
  const title=view==="absolute"?"Epoch-end online gap":"Replay-added online gap relative to epoch 1 end";
  const yTitle=view==="absolute"?"online validation loss - writer train loss":"epoch-end gap increment from epoch 1";
  Plotly.react("frequency-mask-gap-chart",traces,{
    title:`${title} as a function of continuous mask threshold x · ${scale==="log1p"?"log10(x + 1) view":"linear view"}`,
    xaxis,
    yaxis:{title:yTitle,zeroline:true},
    legend:{orientation:"h",x:0,y:1.16,xanchor:"left",yanchor:"top"},
    margin:{l:82,r:30,t:90,b:88},
  },plotConfig);
  const measuredMaximum=functionPoints.length?functionPoints.at(-1).functionX:null;
  document.getElementById("frequency-mask-view-status").textContent=
    `显示 ${functionPoints.length} 个正式数值条件，输入范围 0–${xMax}，当前最大已测 x=${measuredMaximum??"none"}。折线只连接范围内已测点，点间线段是视觉引导，不代表在未采样整数上的观测值。`;
  const audit=frequencyMaskAnalysis.bridgeAudit;
  const format=value=>`${value>=0?"+":""}${value.toFixed(4)}`;
  document.getElementById("frequency-mask-bridge-audit").textContent=
    `环境桥接审计（不进入 49 点主曲线）：${audit.primaryRunId} @ ${audit.primaryHost} 的 epoch 1/2/3 gap = ${[1,2,3].map(epoch=>format(audit.primaryEpochGaps[String(epoch)])).join(" / ")}；${audit.bridgeRunId} @ ${audit.bridgeHost} = ${[1,2,3].map(epoch=>format(audit.bridgeEpochGaps[String(epoch)])).join(" / ")}。最大绝对差为 ${audit.maxAbsoluteDelta.toFixed(4)}；两者使用相同 seed、batch order、optimizer 与 frequency-index SHA。`;
}

const experimentMetrics = {stage:"gap",stage2a:"gap",stage3r:"gap"};
function currentExperimentReport(prefix) { return experimentReports[prefix][document.getElementById(`${prefix}-run-select`).value]; }
function formatScale(value) { return Number(value).toFixed(3); }
function stageEpochShapes(report) {
  const steps = report.meta?.sampling?.estimated_steps_per_epoch || 0;
  const maximum = Math.max(...report.onlineLoss.map(row=>row.step),0);
  const shapes = [];
  for (let step=steps; step<maximum; step+=steps) shapes.push({type:"line",x0:step,x1:step,y0:0,y1:1,yref:"paper",line:{color:"#aaa",dash:"dot",width:1}});
  return shapes;
}
function stageFixedProbeShapes(report) {
  const epochSteps=report.meta?.sampling?.estimated_steps_per_epoch||0;
  const probeBatches=report.meta?.fixed_probe?.train_batches||0;
  const gradAccum=report.meta?.geometry?.grad_accum||1;
  const offset=report.meta?.fixed_probe?.train_offset_optimizer_steps||0;
  const maximum=Math.max(...report.onlineLoss.map(row=>row.step),0);
  if (!epochSteps||!probeBatches) return [];
  const width=Math.ceil(probeBatches/gradAccum),shapes=[];
  for (let start=1;start<=maximum;start+=epochSteps) shapes.push({type:"rect",x0:start+offset-.5,x1:Math.min(maximum,start+offset+width-1)+.5,y0:0,y1:1,yref:"paper",layer:"below",fillcolor:"rgba(33,150,243,.12)",line:{width:0}});
  return shapes;
}
function thin(points,stride) { return points.filter(row=>row.step%stride===0||row.step===points.at(-1)?.step); }
function stage3rDenseSamplingEnabled() { return document.getElementById("stage3r-dense-sampling").checked; }
function stage3rKeepSparseSample(report,step) {
  const epochSteps=report.meta?.sampling?.estimated_steps_per_epoch||337;
  return step%50===0||(step>1&&(step-1)%epochSteps===0)||step===1000;
}
function renderExperimentLoss(prefix) {
  const report=currentExperimentReport(prefix),stride=Number(document.getElementById(`${prefix}-loss-stride`).value);
  const train=thin(report.onlineLoss,stride),validation=report.validation;
  const descriptor=prefix==="stage3r"?stage3rStrategyLabel(report):`beta2=${report.beta2.toFixed(3)} · LR x${formatScale(report.lrScale)}`;
  Plotly.react(`${prefix}-loss-chart`,[
    {x:train.map(row=>row.step),y:train.map(row=>row.train_writer_loss),mode:"lines",name:"writer train loss",line:{color:"#2962ff",width:1.4},hovertemplate:"step=%{x}<br>writer train loss=%{y:.5f}<extra></extra>"},
    {x:validation.map(row=>row.step),y:validation.map(row=>row.val_loss),mode:"lines+markers",name:"validation loss",line:{color:"#d84315",width:2},hovertemplate:"step=%{x}<br>validation loss=%{y:.5f}<extra></extra>"}
  ],{title:`Online loss · ${descriptor}`,xaxis:{title:"step"},yaxis:{title:"cross-entropy loss"},shapes:stageEpochShapes(report),margin:{l:65,r:24,t:50,b:54},legend:{x:.02,y:.98}},plotConfig);
}
function renderStage3rGlobalGap() {
  const report=currentExperimentReport("stage3r"),dense=stage3rDenseSamplingEnabled();
  const rows=dense?report.onlineGap:report.onlineGap.filter(row=>stage3rKeepSparseSample(report,row.step));
  Plotly.react("stage3r-global-gap-chart",[{
    x:rows.map(row=>row.step),y:rows.map(row=>row.gap),customdata:rows.map(row=>[row.epoch,row.logicalBatchId,row.trainLoss,row.validationLoss,row.reason]),mode:"lines+markers",showlegend:false,line:{color:stage3rColor(report),width:2},marker:{size:dense?4:7},hovertemplate:"step=%{x} (epoch %{customdata[0]})<br>global gap=%{y:+.6f}<br>logical batch=%{customdata[1]}<br>writer train=%{customdata[2]:.6f}<br>moving validation=%{customdata[3]:.6f}<br>reason=%{customdata[4]}<extra></extra>"
  }],{title:`Global online gap · ${stage3rStrategyLabel(report)} · ${dense?"dense replay-edge sampling":"50-step intervals + epoch-start points"}`,xaxis:{title:"optimizer step"},yaxis:{title:"online validation loss - writer train loss",zeroline:true},shapes:stageEpochShapes(report),margin:{l:82,r:30,t:55,b:55}},plotConfig);
}
function renderExperimentFrequency(prefix) {
  const report=currentExperimentReport(prefix),source=document.getElementById(`${prefix}-source`).value,branch=document.getElementById(`${prefix}-branch`).value;
  const metric=experimentMetrics[prefix],data=report.frequency[source]?.[branch],status=document.getElementById(`${prefix}-status`);
  if (!data) { Plotly.purge(`${prefix}-frequency-chart`); status.textContent="当前运行尚未写入该统计。"; return; }
  if (source==="fixed") status.textContent=`bucket 使用 train corpus hit count；浅蓝竖带是固定 train probe 在 replay epoch 中的位置，三角标记是评估 checkpoint。`;
  else if (source==="fixed_gram") status.textContent="每个 source、branch、bucket 的 occurrence 在训练前固定；gap = validation mean token loss - train mean token loss，不乘 token fraction。";
  else status.textContent="曲线来自实际 writer batch 与独立 moving validation batch，密集点围绕 replay edge 取得。";
  const dense=prefix!=="stage3r"||stage3rDenseSamplingEnabled();
  if (prefix==="stage3r"&&!dense) status.textContent += source==="online"?" 当前隐藏边界窗口的逐步采样，只保留每 50 step 与 epoch-start 点。":" 当前 fixed-gram 只保留 epoch-start 点与末态；它没有常规 50-step checkpoint。";
  const traces=[];
  buckets.forEach((bucket,index)=>{
    const row=data[bucket];
    const gap=metric==="loss"?row.val_loss.map((value,j)=>value==null||row.train_loss[j]==null?null:value-row.train_loss[j]):row.mean_loss_gap;
    const indices=row.steps.map((_,j)=>j).filter(j=>dense||stage3rKeepSparseSample(report,row.steps[j]));
    const take=values=>indices.map(j=>values[j]);
    const common={x:take(row.steps),customdata:indices.map(j=>[row.epochs[j],gap[j],row.reasons[j],row.train_loss[j],row.val_loss[j],row.train_sample_count?.[j]??null,row.val_sample_count?.[j]??null]),line:{color:bucketColors[index],width:1.8}};
    if (metric==="gap") traces.push({...common,y:take(gap),mode:"lines+markers",name:bucket,hovertemplate:`bucket=${bucket}<br>step=%{x} (epoch %{customdata[0]})<br>mean gap=%{y:.6f}<br>train loss=%{customdata[3]:.6f}<br>validation loss=%{customdata[4]:.6f}<br>reason=%{customdata[2]}<extra></extra>`});
    else {
      traces.push({...common,y:take(row.val_loss),mode:"lines+markers",name:bucket+" (validation)"});
      traces.push({...common,y:take(row.train_loss),mode:"lines",name:bucket+" (train)",showlegend:false,line:{color:bucketColors[index],width:1,dash:"dash"}});
    }
  });
  if (source==="fixed") traces.push({x:report.fixedReads.map(row=>row.step),y:report.fixedReads.map(()=>0),yaxis:"y2",mode:"markers",name:"fixed probe evaluation",marker:{symbol:"triangle-up",size:8,color:"#222"}});
  const sourceTitle=source==="online"?"online moving batches":source==="fixed"?"fixed batch probe":"fixed gram sample";
  const shapes=source==="fixed"?[...stageEpochShapes(report),...stageFixedProbeShapes(report)]:stageEpochShapes(report);
  const stage3rLegend=prefix==="stage3r";
  Plotly.react(`${prefix}-frequency-chart`,traces,{title:`${sourceTitle} · ${branch} · ${metric==="gap"?"mean per-token gap":"train / validation per-token loss"}`,xaxis:{title:"step"},yaxis:{title:metric==="gap"?"mean per-token loss gap":"per-token loss",zeroline:metric==="gap"},yaxis2:{overlaying:"y",side:"right",range:[-1,1],visible:source==="fixed",showticklabels:false},shapes,margin:{l:72,r:stage3rLegend?170:80,t:50,b:55},legend:stage3rLegend?{x:1.02,y:1,xanchor:"left",font:{size:9}}:{x:.01,y:.99,font:{size:9}}},plotConfig);
}
function selectExperimentRun(prefix,runId,scroll=false) {
  const selector=document.getElementById(`${prefix}-run-select`);
  selector.value=runId;
  const report=currentExperimentReport(prefix);
  const orderText=prefix==="stage3r"?` · ${stage3rStrategyLabel(report)} · order seed=${report.orderSeed}`:"";
  document.getElementById(`${prefix}-run-meta`).textContent=`${report.runId}${orderText} · table RMSProp beta2=${report.beta2.toFixed(3)} · LR scale=${formatScale(report.lrScale)} · table base LR=${report.tableBaseLr.toFixed(6)} · weight decay=${report.optimizer.weight_decay} · eps=${report.optimizer.eps}`;
  document.querySelectorAll(`#${prefix}-convergence-table tbody tr[data-run-id],#${prefix}-edge-table tbody tr[data-run-id]`).forEach(row=>row.classList.toggle("selected",row.dataset.runId===runId));
  renderExperimentLoss(prefix); renderExperimentFrequency(prefix);
  if (prefix==="stage3r") renderStage3rGlobalGap();
  if (scroll) document.getElementById(`${prefix}-run-explorer`).scrollIntoView({behavior:"smooth",block:"start"});
}

document.getElementById("historical-frequency-branch").addEventListener("change",renderHistoricalFrequency);
document.getElementById("historical-frequency-view").addEventListener("change",renderHistoricalFrequency);
document.getElementById("historical-frequency-metric").addEventListener("change",renderHistoricalFrequency);
document.getElementById("historical-distribution-branch").addEventListener("change",renderHistoricalDistribution);
document.getElementById("historical-log-branch").addEventListener("change",renderHistoricalLog);
document.getElementById("stage2a-analysis-beta").addEventListener("change",renderLrAnalysis);
document.getElementById("stage2a-analysis-branch").addEventListener("change",renderLrAnalysis);
document.querySelectorAll(".stage3r-loss-toggle").forEach(input=>input.addEventListener("change",renderStage3rFullOnlineLoss));
document.getElementById("stage3r-dense-sampling").addEventListener("change",()=>{renderStage3rGlobalGap();renderExperimentFrequency("stage3r");});
document.getElementById("frequency-mask-x-scale").addEventListener("change",renderFrequencyMaskSweep);
document.getElementById("frequency-mask-y-view").addEventListener("change",renderFrequencyMaskSweep);
document.getElementById("frequency-mask-x-max").addEventListener("input",renderFrequencyMaskSweep);
document.getElementById("frequency-mask-x-max").addEventListener("change",event=>{
  const value=Number(event.target.value);
  event.target.value=Number.isFinite(value)&&value>=0?Math.floor(value):210;
  renderFrequencyMaskSweep();
});
["stage","stage2a","stage3r"].forEach(prefix=>{
  document.getElementById(`${prefix}-run-select`).addEventListener("change",event=>selectExperimentRun(prefix,event.target.value));
  document.getElementById(`${prefix}-loss-stride`).addEventListener("change",()=>renderExperimentLoss(prefix));
  document.getElementById(`${prefix}-source`).addEventListener("change",()=>renderExperimentFrequency(prefix));
  document.getElementById(`${prefix}-branch`).addEventListener("change",()=>renderExperimentFrequency(prefix));
  document.querySelectorAll(`#${prefix}-convergence-table tbody tr[data-run-id],#${prefix}-edge-table tbody tr[data-run-id]`).forEach(row=>{
    row.addEventListener("click",()=>selectExperimentRun(prefix,row.dataset.runId,true));
    row.addEventListener("keydown",event=>{if(event.key==="Enter"||event.key===" "){event.preventDefault();selectExperimentRun(prefix,row.dataset.runId,true);}});
  });
});
document.querySelectorAll(".experiment-metric").forEach(button=>button.addEventListener("click",()=>{
  const prefix=button.dataset.prefix;
  experimentMetrics[prefix]=button.dataset.metric;
  document.querySelectorAll(`.experiment-metric[data-prefix="${prefix}"]`).forEach(item=>item.classList.toggle("active",item===button));
  renderExperimentFrequency(prefix);
}));

renderHistoricalInjection();
renderHistoricalFrequency();
renderHistoricalDistribution();
renderHistoricalLog();
renderLrAnalysis();
renderStage3rFullOnlineLoss();
renderStage3rAudit();
renderFrequencyMaskSweep();
selectExperimentRun("stage","__STAGE1_DEFAULT_RUN_ID__");
selectExperimentRun("stage2a","__STAGE2A_DEFAULT_RUN_ID__");
selectExperimentRun("stage3r","__STAGE3R_DEFAULT_RUN_ID__");
</script>
</body>
</html>
'''


def build_document(
    historical: dict,
    stage1_runs: list[dict],
    stage2a_runs: list[dict],
    lr_analysis: dict,
    stage3r_runs: list[dict],
    stage3r_analysis: dict,
    frequency_mask_analysis: dict,
) -> str:
    convergence_rows, edge_rows = render_tables(stage1_runs)
    stage2a_convergence_rows, stage2a_edge_rows = render_tables(stage2a_runs)
    stage3r_convergence_rows, stage3r_edge_rows = render_stage3r_tables(
        stage3r_runs, stage3r_analysis
    )
    stage1_payload = {run["run_id"]: run["payload"] for run in stage1_runs}
    stage2a_payload = {run["run_id"]: run["payload"] for run in stage2a_runs}
    stage3r_payload = {run["run_id"]: run["payload"] for run in stage3r_runs}
    replacements = {
        "__CONVERGENCE_ROWS__": convergence_rows,
        "__EDGE_ROWS__": edge_rows,
        "__STAGE1_RUN_OPTIONS__": render_run_options(
            stage1_runs, STAGE1_DEFAULT_RUN_ID
        ),
        "__STAGE2A_CONVERGENCE_ROWS__": stage2a_convergence_rows,
        "__STAGE2A_EDGE_ROWS__": stage2a_edge_rows,
        "__STAGE2A_RUN_OPTIONS__": render_run_options(
            stage2a_runs, STAGE2A_DEFAULT_RUN_ID
        ),
        "__STAGE3R_CONVERGENCE_ROWS__": stage3r_convergence_rows,
        "__STAGE3R_EDGE_ROWS__": stage3r_edge_rows,
        "__STAGE3R_RUN_OPTIONS__": render_stage3r_options(stage3r_runs),
        "__STAGE3R_LOSS_TOGGLES__": render_stage3r_loss_toggles(stage3r_runs),
        "__HISTORICAL_DATA__": json.dumps(
            historical, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ),
        "__STAGE1_DATA__": json.dumps(
            stage1_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ),
        "__STAGE2A_DATA__": json.dumps(
            stage2a_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ),
        "__STAGE3R_DATA__": json.dumps(
            stage3r_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ),
        "__LR_ANALYSIS_DATA__": json.dumps(
            lr_analysis, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ),
        "__STAGE3R_ANALYSIS_DATA__": json.dumps(
            stage3r_analysis, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ),
        "__FREQUENCY_MASK_ANALYSIS_DATA__": json.dumps(
            frequency_mask_analysis,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "__BUCKETS__": json.dumps(BUCKETS),
        "__COLORS__": json.dumps(COLORS),
        "__STAGE1_DEFAULT_RUN_ID__": STAGE1_DEFAULT_RUN_ID,
        "__STAGE2A_DEFAULT_RUN_ID__": STAGE2A_DEFAULT_RUN_ID,
        "__STAGE3R_DEFAULT_RUN_ID__": STAGE3R_DEFAULT_RUN_ID,
    }
    document = HTML_TEMPLATE
    for marker, value in replacements.items():
        require(marker in document, f"template marker {marker} is missing")
        document = document.replace(marker, value)
    require("__" not in document, "unresolved report template marker")
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("data/runs"))
    parser.add_argument(
        "--historical-data",
        type=Path,
        default=Path("docs/data/historical-figures.json"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/frequency-gap-by-hit-count.html")
    )
    args = parser.parse_args()

    historical = load_historical_snapshot(args.historical_data)
    stage1_runs = [
        load_stage_run(args.runs_root, *condition) for condition in STAGE1_CONDITIONS
    ]
    stage2a_runs = [
        load_stage_run(args.runs_root, *condition) for condition in STAGE2A_CONDITIONS
    ]
    stage3r_runs = [
        load_stage3r_run(args.runs_root, *condition) for condition in STAGE3R_CONDITIONS
    ]
    frequency_mask_runs = [
        load_frequency_mask_run(args.runs_root, *condition)
        for condition in FREQUENCY_MASK_CONDITIONS
    ]
    frequency_mask_bridge = load_frequency_mask_run(
        args.runs_root, "none bridge", None, FREQUENCY_MASK_BRIDGE_RUN_ID
    )
    require(len(stage1_runs) == 9, "Stage 1 must contain exactly nine conditions")
    require(len(stage2a_runs) == 13, "Stage 2A must contain exactly thirteen conditions")
    require(len(stage3r_runs) == 4, "Stage 3R must contain exactly four order-control conditions")
    require(len(frequency_mask_runs) == 49, "frequency-mask sweep must contain exactly 49 conditions")
    all_runs = stage1_runs + stage2a_runs + stage3r_runs
    require(len({run["run_id"] for run in all_runs}) == 26, "experiment run IDs must be unique")
    require(
        len({run["manifest_sha256"] for run in all_runs}) == 1,
        "all optimizer experiments must share one fixed-gram manifest",
    )
    lr_analysis = build_lr_analysis(stage1_runs + stage2a_runs)
    stage3r_analysis = build_stage3r_analysis(stage3r_runs)
    frequency_mask_analysis = build_frequency_mask_analysis(
        frequency_mask_runs, frequency_mask_bridge
    )
    document = build_document(
        historical, stage1_runs, stage2a_runs, lr_analysis,
        stage3r_runs, stage3r_analysis, frequency_mask_analysis,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(args.out)
    print(
        f"wrote {args.out} with {len(historical['charts'])} historical charts, "
        f"{len(stage1_runs)} Stage 1 conditions, {len(stage2a_runs)} Stage 2A conditions, "
        f"{len(stage3r_runs)} Stage 3R conditions, "
        f"{len(frequency_mask_runs)} frequency-mask conditions, and "
        f"{sum(len(run.get('edges', [])) for run in all_runs)} optimizer edge rows"
    )


if __name__ == "__main__":
    main()
