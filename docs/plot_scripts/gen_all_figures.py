#!/usr/bin/env python3
"""Regenerate the publication SVG assets from the frozen historical snapshot.

Interactive historical plots live only in the consolidated report. This script
deliberately emits SVG files and never creates standalone HTML pages.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BUCKETS = (
    "novel", "1", "2", "3", "4", "5", "6-10", "11-20", "21-50",
    "51-100", "101-200", "201-500", "501-1k", "1k-5k", "5k+",
)
BUCKET_COLORS = (
    "#E91E63", "#F44336", "#FF5722", "#FF9800", "#FFC107", "#FFEB3B",
    "#CDDC39", "#8BC34A", "#4CAF50", "#009688", "#00BCD4", "#03A9F4",
    "#2196F3", "#3F51B5", "#673AB7",
)
INK = "#232426"
MUTED = "#6b6f75"
LINE = "#d7d9dc"
PAPER = "#fbfbfa"
ANCHOR = "#5b3f91"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def style_axis(axis) -> None:
    axis.set_facecolor(PAPER)
    axis.grid(axis="y", color=LINE, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(LINE)
    axis.spines["bottom"].set_color(LINE)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)


def add_epoch_lines(axis) -> None:
    for step in (337, 686):
        axis.axvline(step, color="#aeb2b7", linewidth=1, linestyle=":")


def save_svg(figure, name: str, output_dir: Path, mirror_dir: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / name, format="svg", facecolor=PAPER, bbox_inches="tight")
    if mirror_dir is not None:
        mirror_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(mirror_dir / name, format="svg", facecolor=PAPER, bbox_inches="tight")
    plt.close(figure)


def plot_injection(charts: dict, output_dir: Path, mirror_dir: Path | None) -> None:
    gap_traces = charts["injection_gap"]["traces"]
    loss_traces = charts["injection_loss"]["traces"]
    colors = ("#2196F3", "#F44336", "#4CAF50")

    figure, axis = plt.subplots(figsize=(10.8, 4.8), facecolor=PAPER)
    style_axis(axis)
    for trace, color in zip(gap_traces, colors):
        axis.plot(trace["x"], trace["y"], color=color, linewidth=2.2,
                  marker="o", markersize=2.8, label=trace["name"])
    add_epoch_lines(axis)
    axis.set_title("Train / validation gap", loc="left", fontsize=15, fontweight="bold")
    axis.set_xlabel("step")
    axis.set_ylabel("val loss - train loss")
    axis.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    figure.tight_layout()
    save_svg(figure, "fig_gap.svg", output_dir, mirror_dir)

    figure, axis = plt.subplots(figsize=(10.8, 5.3), facecolor=PAPER)
    style_axis(axis)
    for index, trace in enumerate(loss_traces):
        color = colors[index // 2]
        is_train = "(train)" in trace["name"]
        axis.plot(trace["x"], trace["y"], color=color,
                  linewidth=1.5 if is_train else 2.1,
                  linestyle="--" if is_train else "-", alpha=0.76 if is_train else 1,
                  label=trace["name"])
    add_epoch_lines(axis)
    axis.set_title("Train / validation loss", loc="left", fontsize=15, fontweight="bold")
    axis.set_xlabel("step")
    axis.set_ylabel("cross-entropy loss")
    axis.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.2)
    figure.tight_layout()
    save_svg(figure, "fig_loss.svg", output_dir, mirror_dir)


def plot_norm_alignment(charts: dict, output_dir: Path, mirror_dir: Path | None) -> None:
    norm_traces = charts["table_norm"]["traces"]
    figure, axis = plt.subplots(figsize=(10.8, 4.8), facecolor=PAPER)
    style_axis(axis)
    for trace, color in zip(norm_traces, ("#2d6f9f", "#c4493d")):
        axis.plot(trace["x"], trace["y"], color=color, linewidth=2.2, label=trace["name"])
    add_epoch_lines(axis)
    axis.set_title("N-gram table norm", loc="left", fontsize=15, fontweight="bold")
    axis.set_xlabel("step")
    axis.set_ylabel("RMS")
    axis.legend(frameon=False, loc="upper left", fontsize=9)
    figure.tight_layout()
    save_svg(figure, "fig_table_norm.svg", output_dir, mirror_dir)

    traces = charts["input_alignment"]["traces"]
    figure, axis = plt.subplots(figsize=(10.8, 4.8), facecolor=PAPER)
    style_axis(axis)
    axis.plot(traces[0]["x"], traces[0]["y"], color="#2d6f9f", linewidth=1.6,
              linestyle="--", label=traces[0]["name"])
    axis.plot(traces[1]["x"], traces[1]["y"], color="#c4493d", linewidth=2.1,
              label=traces[1]["name"])
    axis.set_ylabel("loss")
    second = axis.twinx()
    second.plot(traces[2]["x"], traces[2]["y"], color=ANCHOR, linewidth=2,
                label=traces[2]["name"])
    second.set_ylabel("gap", color=ANCHOR)
    second.tick_params(colors=ANCHOR, labelsize=9)
    second.spines["right"].set_color(ANCHOR)
    add_epoch_lines(axis)
    axis.set_title("Input run: loss and gap alignment", loc="left", fontsize=15,
                   fontweight="bold")
    axis.set_xlabel("step")
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, frameon=False,
                loc="upper left", fontsize=9)
    figure.tight_layout()
    save_svg(figure, "fig_input_alignment.svg", output_dir, mirror_dir)


def final_frequency_values(series: dict, branch: str) -> dict:
    values = {}
    for bucket in BUCKETS:
        row = series[branch][bucket]
        values[bucket] = {
            "train_loss": row["train_loss"][-1],
            "val_loss": row["val_loss"][-1],
            "gap": row["val_loss"][-1] - row["train_loss"][-1],
            "train_frac": row["train_frac"][-1],
            "val_frac": row["val_frac"][-1],
        }
    return values


def plot_frequency(charts: dict, output_dir: Path, mirror_dir: Path | None) -> None:
    series = charts["frequency_bins"]["series"]
    positions = np.arange(len(BUCKETS))
    width = 0.38
    for branch in ("bigram", "trigram"):
        values = final_frequency_values(series, branch)
        train_loss = [values[bucket]["train_loss"] for bucket in BUCKETS]
        val_loss = [values[bucket]["val_loss"] for bucket in BUCKETS]
        gap = [values[bucket]["gap"] for bucket in BUCKETS]
        train_contribution = [
            values[bucket]["train_frac"] * values[bucket]["train_loss"] for bucket in BUCKETS
        ]
        val_contribution = [
            values[bucket]["val_frac"] * values[bucket]["val_loss"] for bucket in BUCKETS
        ]
        figure, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True,
                                    facecolor=PAPER,
                                    gridspec_kw={"height_ratios": [1.3, 1, 1]})
        for axis in axes:
            style_axis(axis)
        axes[0].bar(positions-width/2, train_loss, width, color="#2d6f9f",
                    alpha=.82, label="train mean loss")
        axes[0].bar(positions+width/2, val_loss, width, color="#c4493d",
                    alpha=.82, label="validation mean loss")
        axes[0].set_ylabel("mean loss")
        axes[0].set_title(f"{branch.capitalize()} frequency decomposition · final step",
                          loc="left", fontsize=15, fontweight="bold")
        axes[0].legend(frameon=False, loc="upper right", fontsize=9)
        axes[1].bar(positions, gap, color=BUCKET_COLORS, alpha=.88)
        axes[1].axhline(0, color=LINE, linewidth=1)
        axes[1].set_ylabel("val - train")
        axes[2].bar(positions-width/2, train_contribution, width, color="#2d6f9f",
                    alpha=.82, label="train fraction x loss")
        axes[2].bar(positions+width/2, val_contribution, width, color="#c4493d",
                    alpha=.82, label="validation fraction x loss")
        axes[2].set_ylabel("total contribution")
        axes[2].legend(frameon=False, loc="upper right", fontsize=9)
        axes[2].set_xticks(positions)
        axes[2].set_xticklabels(BUCKETS, rotation=42, ha="right")
        axes[2].set_xlabel("training hit-count bucket")
        figure.tight_layout(h_pad=1.1)
        save_svg(figure, f"fig_freq_{branch}.svg", output_dir, mirror_dir)


def plot_distribution(charts: dict, output_dir: Path, mirror_dir: Path | None) -> None:
    series = charts["hitcount_distribution"]["series"]
    positions = np.arange(len(BUCKETS))
    width = .38
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.8), sharex=True, facecolor=PAPER)
    for axis, branch in zip(axes, ("bigram", "trigram")):
        style_axis(axis)
        rows = series[branch]
        train = [row["train_frac"] for row in rows]
        validation = [row["val_frac"] for row in rows]
        axis.bar(positions-width/2, train, width, color="#2d6f9f", alpha=.82,
                 label="train fraction")
        axis.bar(positions+width/2, validation, width, color="#c4493d", alpha=.82,
                 label="validation fraction")
        cumulative = axis.twinx()
        cumulative.plot(positions, np.cumsum(train), color="#2d6f9f", linestyle=":",
                        marker="o", markersize=2.5, linewidth=1.6)
        cumulative.plot(positions, np.cumsum(validation), color="#c4493d", linestyle=":",
                        marker="o", markersize=2.5, linewidth=1.6)
        cumulative.set_ylim(0, 1.04)
        cumulative.set_ylabel("cumulative", color=MUTED)
        cumulative.tick_params(colors=MUTED, labelsize=8)
        axis.set_ylabel("token fraction")
        axis.set_title(f"{branch.capitalize()} context frequency", loc="left",
                       fontsize=14, fontweight="bold")
        axis.legend(frameon=False, loc="upper left", fontsize=8.5)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(BUCKETS, rotation=42, ha="right")
    axes[-1].set_xlabel("training hit-count bucket")
    figure.suptitle("Context frequency distribution · all buckets", x=.06, ha="left",
                     y=.995, fontsize=16, fontweight="bold", color=INK)
    figure.tight_layout(rect=[0, 0, 1, .97])
    save_svg(figure, "fig_hitcount_dist.svg", output_dir, mirror_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path,
                        default=Path("docs/data/historical-figures.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figs"))
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    require(snapshot.get("schema_version") == 1, "unsupported historical snapshot")
    charts = snapshot.get("charts", {})
    require(len(charts) == 7, "historical snapshot must contain seven charts")
    mirror_value = os.environ.get("NGRAM_GAP_BLOG_FIGS_DIR")
    mirror_dir = Path(mirror_value) if mirror_value else None
    plot_injection(charts, args.out_dir, mirror_dir)
    plot_norm_alignment(charts, args.out_dir, mirror_dir)
    plot_frequency(charts, args.out_dir, mirror_dir)
    plot_distribution(charts, args.out_dir, mirror_dir)
    print(f"wrote seven SVG assets to {args.out_dir}")


if __name__ == "__main__":
    main()
