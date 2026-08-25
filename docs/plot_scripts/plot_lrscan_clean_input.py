#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "data" / "runs_fixed"
OUT = ROOT / "docs" / "figs" / "main" / "fig_lrscan_clean_input.png"
ARMS = [
    ("lrscan_input_lr0p004_wc_fixed", "backbone LR 4e-3 · table 8e-3", "#c4493d"),
    ("lrscan_input_lr0p0006_wc_fixed", "backbone LR 6e-4 · table 1.2e-3", "#2d6f9f"),
    ("lrscan_input_lr0p0004_wc_fixed", "backbone LR 4e-4 · table 8e-4", "#3a8f5d"),
]


def load_rows(run_id):
    path = RUNS_DIR / run_id / "train_log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def moving_average(values):
    if len(values) < 3:
        return np.asarray(values)
    return np.convolve(values, np.ones(3) / 3, mode="valid")


def plot_metric(axis, rows, metric, label, color):
    steps = np.asarray([row["step"] for row in rows])
    values = np.asarray([row[metric] for row in rows])
    axis.scatter(steps, values, s=14, color=color, alpha=0.85, zorder=2)
    if len(values) >= 3:
        axis.plot(
            steps[1:-1],
            moving_average(values),
            color=color,
            linewidth=0.9,
            label=label,
            zorder=3,
        )
    else:
        axis.plot(steps, values, color=color, linewidth=0.9, label=label, zorder=3)


def main():
    data = [(label, color, load_rows(run_id)) for run_id, label, color in ARMS]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)

    for label, color, rows in data:
        steps = np.asarray([row["step"] for row in rows])
        train = np.asarray([row["train_loss"] for row in rows])
        val = np.asarray([row["val_loss"] for row in rows])
        axes[0].scatter(steps, train, s=14, color=color, alpha=0.85, zorder=2)
        axes[0].scatter(steps, val, s=14, color=color, alpha=0.4, marker="x", zorder=2)
        if len(train) >= 3:
            smoothed_steps = steps[1:-1]
            axes[0].plot(smoothed_steps, moving_average(train), color=color, linewidth=0.9,
                         label=label, zorder=3)
            axes[0].plot(smoothed_steps, moving_average(val), color=color, linewidth=0.9,
                         linestyle="--", zorder=3)
        else:
            axes[0].plot(steps, train, color=color, linewidth=0.9, label=label, zorder=3)
            axes[0].plot(steps, val, color=color, linewidth=0.9, linestyle="--", zorder=3)
        plot_metric(axes[1], rows, "gap", label, color)

    axes[0].set_title("Clean-table input: loss trajectory")
    axes[0].set_ylabel("loss")
    axes[0].set_xlabel("optimizer step")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7.2, frameon=False)
    axes[0].text(
        0.015,
        0.02,
        "● solid: online train loss\n× dashed: fixed validation loss",
        transform=axes[0].transAxes,
        fontsize=7.5,
        va="bottom",
    )

    axes[1].axhline(0, color="#777777", linestyle=":", linewidth=0.8)
    axes[1].set_title("Online gap trajectory")
    axes[1].set_ylabel("fixed val loss − online train loss")
    axes[1].set_xlabel("optimizer step")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7.5, frameon=False)

    figure.suptitle(
        "Single-seed LR screening: clean R=2²⁰, warmup_constant (100 steps), bf16, no compile",
        fontsize=10,
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(OUT, dpi=180, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()