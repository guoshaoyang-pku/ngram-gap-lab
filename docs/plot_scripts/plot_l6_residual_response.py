#!/usr/bin/env python3
"""Plot the registered L6 exact-enumeration results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "tasks" / "l6_residual_response" / "results"
OUT = ROOT / "docs" / "figs" / "theory"

PAPER = "#f7f5ef"
PANEL = "#fffdf8"
INK = "#232426"
MUTED = "#686d73"
BORDER = "#c8c1b6"
COLORS = ("#2d6f9f", "#c4493d", "#3c8d5a")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_summary(run_id: str) -> dict:
    return json.loads((RESULTS / run_id / "summary.json").read_text(encoding="utf-8"))


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.grid(True, which="major", color=BORDER, alpha=0.40, linewidth=0.7)
    axis.grid(True, which="minor", color=BORDER, alpha=0.18, linewidth=0.5)
    axis.tick_params(colors=INK, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(BORDER)


def main() -> None:
    count_id = "l6_counttable_freq_exact_v1"
    response_id = "l6_response_moments_exact_v1"
    count_rows = read_csv(RESULTS / count_id / "metrics.csv")
    response_rows = read_csv(RESULTS / response_id / "metrics.csv")
    count_slopes = read_summary(count_id)["fitted_slopes"]
    response_slopes = read_summary(response_id)["fitted_slopes"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.35), facecolor=PAPER)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.82, wspace=0.27)

    for color, p in zip(COLORS, ("0.5", "0.2", "0.05")):
        selected = [row for row in count_rows if row["p"] == p]
        f = np.array([float(row["f"]) for row in selected])
        gap = np.array([float(row["exact_gap"]) for row in selected])
        axes[0].loglog(
            f,
            gap,
            marker="o",
            markersize=3.7,
            linewidth=1.7,
            color=color,
            label=f"p={p}; large-f slope {count_slopes[p]:.2f}",
        )
    reference_f = np.array([512.0, 4096.0])
    axes[0].loglog(reference_f, 1.0 / reference_f, "--", color=MUTED, linewidth=1.1, label="f⁻¹ guide")
    axes[0].set_title("A  Count table: −1 is asymptotic", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    axes[0].set_xlabel("samples per context  f", color=INK, fontsize=9)
    axes[0].set_ylabel("expected loss gap", color=INK, fontsize=9)
    style_axis(axes[0])
    axes[0].legend(frameon=False, fontsize=7.6, loc="lower left")

    response_order = ("linear", "sign", "cubic")
    response_labels = {"linear": "u(δ)=δ", "sign": "u(δ)=sign(δ)", "cubic": "u(δ)=δ³"}
    for color, response in zip(COLORS, response_order):
        selected = [row for row in response_rows if row["response"] == response]
        f = np.array([float(row["f"]) for row in selected])
        gap = np.array([float(row["exact_gap"]) for row in selected])
        axes[1].loglog(
            f,
            gap,
            marker="o",
            markersize=3.7,
            linewidth=1.7,
            color=color,
            label=f"{response_labels[response]}; slope {response_slopes[response]:.2f}",
        )
    axes[1].set_title("B  Same residual, different response", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    axes[1].set_xlabel("number of samples  f", color=INK, fontsize=9)
    axes[1].set_ylabel("E[δ · u(δ)]", color=INK, fontsize=9)
    style_axis(axes[1])
    axes[1].legend(frameon=False, fontsize=7.6, loc="lower left")

    fig.suptitle("Loss gap is residual–response covariance, not a universal variance law", x=0.075, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(
        0.075,
        0.070,
        f"Runs: {count_id} + {response_id} · exact enumeration · seed N/A.",
        fontsize=7.4,
        color=MUTED,
    )
    fig.text(
        0.075,
        0.040,
        "Fits: f≥512 (A), f≥128 (B). Metrics: expected loss gap (A), E[δ·u(δ)] (B).",
        fontsize=7.4,
        color=MUTED,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"fig_l6_residual_response.{suffix}", dpi=180, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
