#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS_FIXED = ROOT / "data" / "runs_fixed"
RUNS_SCALING = ROOT / "data" / "runs_scaling"
OUT = ROOT / "docs" / "figs" / "main"


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_run(run_id, runs_dir=RUNS_FIXED):
    run_dir = runs_dir / f"{run_id}_fixed"
    return read_json(run_dir / "summary.json"), read_jsonl(run_dir / "train_log.jsonl")


def smooth(values):
    if len(values) < 3:
        return np.asarray(values), np.arange(len(values))
    return np.convolve(values, np.ones(3) / 3, mode="valid"), np.arange(1, len(values) - 1)


def add_raw_and_smoothed(axis, rows, metric, color, label):
    steps = np.asarray([row["step"] for row in rows])
    values = np.asarray([row[metric] for row in rows])
    axis.scatter(steps, values, color=color, s=9, alpha=0.5, zorder=2)
    averaged, offsets = smooth(values)
    axis.plot(steps[offsets], averaged, color=color, linewidth=1.0, label=label, zorder=3)


def plot_injection():
    arms = [
        ("input", "#2d6f9f"),
        ("y", "#c4493d"),
        ("v", "#b67524"),
        ("nogram", "#686d73"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharex=True)
    for arm, color in arms:
        _, rows = read_run(f"nglab1x_{arm}_v5")
        add_raw_and_smoothed(axes[0], rows, "gap", color, arm)
        add_raw_and_smoothed(axes[1], rows, "train_loss", color, f"{arm} train")
        values = [row["val_loss"] for row in rows]
        averaged, offsets = smooth(values)
        steps = np.asarray([row["step"] for row in rows])
        axes[1].plot(steps[offsets], averaged, color=color, linewidth=1.0, linestyle="--",
                     label=f"{arm} val")
    axes[0].axhline(0, color="#686d73", linewidth=0.7, linestyle=":")
    axes[0].set_title("V5 injection arms: online gap")
    axes[0].set_ylabel("fixed validation loss − online train loss")
    axes[1].set_title("V5 injection arms: loss")
    axes[1].set_ylabel("loss")
    for axis in axes:
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5, frameon=False, ncol=2)
    figure.suptitle(
        "V5 · seed 42 · clean bigram+trigram R=2²⁰ · warmup_constant(100) · bf16 · no compile",
        fontsize=10,
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(OUT / "fig_v5_injection.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_fixed_step_dose():
    doses = [
        ("0.25×", "nglab0_25x_input_v5", 0.25),
        ("0.5×", "nglab0_5x_input_v5", 0.5),
        ("0.75×", "nglab0_75x_input_v5", 0.75),
        ("1.5×", "nglab1_5x_input_v5", 1.5),
        ("2×", "nglab2x_input_v5", 2.0),
        ("2.5×", "nglab2_5x_input_v5", 2.5),
        ("3×", "nglab3x_input_v5", 3.0),
        ("4×", "nglab4x_input_v5", 4.0),
        ("5×", "nglab5x_input_v5", 5.0),
        ("6×", "nglab6x_input_v5", 6.0),
        ("8×", "nglab8x_input_v5", 8.0),
    ]
    xs, gaps = [], []
    for _, run_id, dose in doses:
        summary, _ = read_run(run_id)
        xs.append(dose)
        gaps.append(summary["final_gap"])
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    axis.plot(xs, gaps, color="#2d6f9f", linewidth=1.1, marker="o", markersize=4)
    axis.axhline(0, color="#686d73", linewidth=0.7, linestyle=":")
    axis.set_xscale("log")
    axis.set_xlabel("train-shard dose relative to 1× (log scale)")
    axis.set_ylabel("final online gap at step 2000")
    axis.set_title("V5 fixed-step dose scan")
    axis.grid(alpha=0.25, which="both")
    for label, dose, gap in zip((item[0] for item in doses), xs, gaps):
        axis.annotate(label, (dose, gap), xytext=(0, 6), textcoords="offset points",
                      ha="center", fontsize=7)
    figure.text(
        0.5,
        0.01,
        "input · seed 42 · clean R=2²⁰ · warmup_constant(100) · bf16 · no compile",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(OUT / "fig_v5_dose_fixedstep.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_s1_epoch():
    lengths = ["L1", "L2", "L3", "L4"]
    both, nogram = [], []
    for length in lengths:
        both_summary, _ = read_run(f"s1v5_{length}_both_fs", RUNS_SCALING)
        nogram_summary, _ = read_run(f"s1v5_{length}_nogram_fs", RUNS_SCALING)
        both.append(both_summary["final_gap"])
        nogram.append(nogram_summary["final_gap"])
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(lengths))
    axis.plot(x, both, color="#353d79", marker="o", linewidth=1.2, label="bigram + trigram")
    axis.plot(x, nogram, color="#8a8f8a", marker="o", linewidth=1.2, label="no-gram")
    axis.plot(x, np.asarray(both) - np.asarray(nogram), color="#c4493d", marker="o",
              linewidth=1.2, label="table-induced Δgap")
    axis.set_xticks(x, ["L1\n42", "L2\n84", "L3\n168", "L4\n337"])
    axis.set_xlabel("epoch-prefix length (device batches per epoch)")
    axis.set_ylabel("final online gap at step 1000")
    axis.set_title("V5 S1 epoch-prefix axis")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False)
    figure.text(
        0.5,
        0.01,
        "seed 42 · fixed-step · clean R=2²⁰ · warmup_constant(100) · bf16 · no compile",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(OUT / "fig_v5_s1_epoch_prefix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_table_size():
    points = []
    for run_dir in sorted(RUNS_SCALING.glob("ctbl_v5_both_*_fixed")):
        summary = read_json(run_dir / "summary.json")
        rows = int(run_dir.name.removeprefix("ctbl_v5_both_").removesuffix("_fixed"))
        points.append((rows, summary["final_gap"]))
    points.sort()
    rows, gaps = zip(*points)
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    axis.plot(rows, gaps, color="#353d79", marker="o", markersize=3.5, linewidth=1.1)
    axis.set_xscale("log")
    axis.set_xlabel("physical rows R per clean table (log scale)")
    axis.set_ylabel("final online gap at step 1000")
    axis.set_title("V5 clean double-table size scan")
    axis.grid(alpha=0.25, which="both")
    figure.text(
        0.5,
        0.01,
        "bigram and trigram enlarged together · seed 42 · warmup_constant(100) · bf16 · no compile",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(OUT / "fig_v5_s1_table_size.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_injection()
    plot_fixed_step_dose()
    plot_s1_epoch()
    plot_table_size()
    print(OUT / "fig_v5_injection.png")
    print(OUT / "fig_v5_dose_fixedstep.png")
    print(OUT / "fig_v5_s1_epoch_prefix.png")
    print(OUT / "fig_v5_s1_table_size.png")


if __name__ == "__main__":
    main()