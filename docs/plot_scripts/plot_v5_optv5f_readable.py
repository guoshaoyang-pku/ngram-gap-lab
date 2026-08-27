#!/usr/bin/env python3
"""Readable v5 table-LR and beta2 comparison figures.

The figure design intentionally avoids using color for table-LR scale:
scale is encoded by facet title, while beta2 always has the same two styles.
This makes the beta2 comparison readable even when trajectories overlap.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data" / "runs_fixed"
FIGDIR = REPO / "docs" / "figs" / "main"
FIGDIR.mkdir(parents=True, exist_ok=True)

FAMILY = {
    "optv5c_rms_b099_s0p5": (0.5, 0.99, 1000),
    "optv5c_rms_b099_s1p0": (1.0, 0.99, 1000),
    "optv5c_rms_b099_s2p0_r1": (2.0, 0.99, 1000),
    "optv5c_rms_b099_s3p0": (3.0, 0.99, 1000),
    "optv5c_rms_b099_s4p0": (4.0, 0.99, 1000),
    "optv5f_rms_b099_s8p0_2k": (8.0, 0.99, 2000),
    "optv5f_rms_b0999_s8p0_2k": (8.0, 0.999, 2000),
    "optv5f_rms_b099_s16p0_2k": (16.0, 0.99, 2000),
    "optv5f_rms_b0999_s16p0_2k": (16.0, 0.999, 2000),
    "optv5f_rms_b099_s32p0": (32.0, 0.99, 1000),
    "optv5f_rms_b099_s32p0_2k": (32.0, 0.99, 2000),
    "optv5f_rms_b0999_s32p0_2k_r1": (32.0, 0.999, 2000),
    "optv5f_rms_b099_s64p0": (64.0, 0.99, 1000),
    "optv5f_rms_b0999_s64p0": (64.0, 0.999, 1000),
    "optv5f_rms_b099_s64p0_2k": (64.0, 0.99, 2000),
    "optv5f_rms_b0999_s64p0_2k": (64.0, 0.999, 2000),
    "optv5f_rms_b099_s128p0": (128.0, 0.99, 1000),
    "optv5f_rms_b0999_s128p0": (128.0, 0.999, 1000),
    "optv5f_rms_b099_s128p0_2k": (128.0, 0.99, 2000),
    "optv5f_rms_b0999_s128p0_2k": (128.0, 0.999, 2000),
}

SCALES = (0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
HIGH_SCALES = (8.0, 16.0, 32.0, 64.0, 128.0)
BETA_STYLES = {
    0.99: {"color": "#123b5d", "marker": "o", "line": "-", "label": "β₂=.99"},
    0.999: {"color": "#c4511c", "marker": "s", "line": "--", "label": "β₂=.999"},
}


def load_rows(run_id):
    path = RUNS / f"{run_id}_fixed" / "train_log.jsonl"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows or None


def load_summary(run_id):
    path = RUNS / f"{run_id}_fixed" / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def run_id_for(scale, beta2, steps):
    for run_id, values in FAMILY.items():
        run_scale, run_beta, run_steps = values
        if run_scale == scale and run_beta == beta2 and run_steps == steps:
            return run_id
    return None


def latest_value(run_id, metric="gap"):
    summary = load_summary(run_id)
    if summary and metric == "gap":
        value = summary.get("final_gap")
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    rows = load_rows(run_id)
    if not rows:
        return None
    value = rows[-1].get(metric)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def complete(run_id):
    summary = load_summary(run_id)
    return bool(summary and str(summary.get("status", "done")).lower() == "done")


def rows_for(scale, beta2, steps):
    run_id = run_id_for(scale, beta2, steps)
    return load_rows(run_id) if run_id else None


def plot_overview():
    """Endpoint comparison across low and high table-LR scales."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, steps, scales, title in (
        (axes[0], 1000, SCALES, "1000-step records"),
        (axes[1], 2000, HIGH_SCALES, "2000-step records to date"),
    ):
        for beta2, style in BETA_STYLES.items():
            xs, ys = [], []
            for scale in scales:
                run_id = run_id_for(scale, beta2, steps)
                if not run_id:
                    continue
                value = latest_value(run_id)
                if value is None:
                    continue
                xs.append(scale)
                ys.append(value)
                axis.plot(
                    scale,
                    value,
                    marker=style["marker"],
                    markersize=8,
                    color=style["color"],
                    markerfacecolor=style["color"] if complete(run_id) else "white",
                    markeredgewidth=1.8,
                    linestyle="none",
                    zorder=3,
                )
                axis.annotate(
                    f"{value:.2f}",
                    (scale, value),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=style["color"],
                )
            if len(xs) > 1:
                axis.plot(
                    xs,
                    ys,
                    color=style["color"],
                    linestyle=style["line"],
                    linewidth=1.1,
                    alpha=0.7,
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks(scales, [f"{scale:g}×" for scale in scales])
        axis.set_xlabel("table LR scale")
        axis.set_title(title)
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("latest recorded gap = fixed val − online train")
    handles = [
        Line2D(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["line"],
            markerfacecolor=style["color"],
            label=style["label"],
        )
        for style in BETA_STYLES.values()
    ]
    handles += [
        Line2D([], [], color="#555", marker="o", linestyle="none", markerfacecolor="#555", label="completed"),
        Line2D([], [], color="#555", marker="o", linestyle="none", markerfacecolor="white", label="running / partial"),
    ]
    axes[0].legend(handles=handles, frameon=False, fontsize=9, loc="upper left")
    figure.suptitle(
        "V5 table-LR scale comparison: low-scale optv5c + high-scale optv5f\n"
        "blue/orange = β₂; circle/square = β₂; filled/open = completed/partial; "
        "labels are raw latest gaps",
        fontsize=11,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.92])
    out = FIGDIR / "fig_v5_optv5f_readable_overview.png"
    figure.savefig(out, dpi=180)
    plt.close(figure)
    print(f"wrote {out}")


def epoch_boundaries(rows):
    boundaries = []
    previous = None
    for row in rows:
        epoch = row.get("epoch")
        if previous is not None and epoch != previous:
            boundaries.append(row["step"])
        previous = epoch
    return boundaries


def plot_high_scale_2000_facets():
    """One row per scale; raw metrics plus direct beta2 gap difference."""
    figure, axes = plt.subplots(
        len(HIGH_SCALES),
        4,
        figsize=(17, 15),
        sharex=False,
        squeeze=False,
    )
    metrics = (
        ("train_loss", "online train loss"),
        ("val_loss", "fixed validation loss"),
        ("gap", "online gap"),
    )
    for row_index, scale in enumerate(HIGH_SCALES):
        loaded = {}
        for beta2 in BETA_STYLES:
            rows = rows_for(scale, beta2, 2000)
            if rows:
                loaded[beta2] = rows
        for col_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index, col_index]
            for beta2, rows in loaded.items():
                style = BETA_STYLES[beta2]
                x_values = np.asarray([row["step"] for row in rows])
                values = np.asarray([row[metric] for row in rows], dtype=float)
                axis.scatter(
                    x_values,
                    values,
                    color=style["color"],
                    s=8,
                    alpha=0.42,
                    linewidths=0,
                )
                if len(values) >= 3:
                    connector = np.convolve(
                        np.pad(values, (1, 1), mode="edge"),
                        np.ones(3) / 3,
                        mode="valid",
                    )
                else:
                    connector = values
                axis.plot(
                    x_values,
                    connector,
                    color=style["color"],
                    linestyle=style["line"],
                    linewidth=1.45,
                    marker=style["marker"],
                    markevery=max(1, len(x_values) // 12),
                    markersize=3.5,
                    label=style["label"],
                )
                for boundary in epoch_boundaries(rows):
                    axis.axvline(boundary, color="#777", linewidth=0.65, alpha=0.35)
            axis.grid(alpha=0.22)
            axis.set_xlabel("step", fontsize=8)
            if col_index == 0:
                axis.set_ylabel(f"×{scale:g}\n{ylabel}", fontsize=8)
            else:
                axis.set_ylabel(ylabel, fontsize=8)
            if row_index == 0:
                axis.set_title(ylabel)

        difference_axis = axes[row_index, 3]
        beta99_rows = loaded.get(0.99, [])
        beta999_rows = loaded.get(0.999, [])
        if beta99_rows and beta999_rows:
            base = {row["step"]: row["gap"] for row in beta99_rows}
            other = {row["step"]: row["gap"] for row in beta999_rows}
            common_steps = sorted(set(base) & set(other))
            delta = np.asarray([other[step] - base[step] for step in common_steps])
            difference_axis.scatter(
                common_steps,
                delta,
                color="#3b3b3b",
                s=9,
                alpha=0.5,
                linewidths=0,
            )
            connector = (
                np.convolve(np.pad(delta, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid")
                if len(delta) >= 3
                else delta
            )
            difference_axis.plot(
                common_steps,
                connector,
                color="#3b3b3b",
                linewidth=1.5,
                marker="o",
                markevery=max(1, len(common_steps) // 12),
                markersize=3.5,
            )
            difference_axis.axhline(0, color="#c2410c", linewidth=0.9, linestyle=":")
            difference_axis.set_ylabel("Δgap\n(.999 − .99)", fontsize=8)
        else:
            difference_axis.text(
                0.5,
                0.5,
                "missing one β₂ arm",
                ha="center",
                va="center",
                transform=difference_axis.transAxes,
            )
        difference_axis.set_xlabel("step", fontsize=8)
        difference_axis.grid(alpha=0.22)
        if row_index == 0:
            difference_axis.set_title("direct difference: Δgap")

    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=style["color"],
                linestyle=style["line"],
                marker=style["marker"],
                markersize=5,
                label=style["label"],
            )
            for style in BETA_STYLES.values()
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.suptitle(
        "V5 high-scale 2000-step comparison, one table-LR scale per row\n"
        "β₂=.99 = navy solid circles; β₂=.999 = orange dashed squares; "
        "right column is direct Δgap=(.999−.99); vertical gray lines = epoch boundaries",
        fontsize=11,
        y=0.975,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.955], h_pad=1.0, w_pad=1.1)
    out = FIGDIR / "fig_v5_optv5f_readable_2000_facets.png"
    figure.savefig(out, dpi=180)
    plt.close(figure)
    print(f"wrote {out}")


def plot_all_scale_gap_facets():
    figure, axes = plt.subplots(2, 5, figsize=(18, 7.8), sharey=True)
    axes = axes.ravel()
    for axis, scale in zip(axes, SCALES):
        plotted = False
        for beta2, style in BETA_STYLES.items():
            candidates = [
                (steps, run_id_for(scale, beta2, steps))
                for steps in (1000, 2000)
                if run_id_for(scale, beta2, steps)
            ]
            for steps, run_id in candidates:
                rows = load_rows(run_id)
                if not rows:
                    continue
                x_values = np.asarray([row["step"] for row in rows])
                values = np.asarray([row["gap"] for row in rows], dtype=float)
                connector = (
                    np.convolve(np.pad(values, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid")
                    if len(values) >= 3
                    else values
                )
                axis.scatter(
                    x_values,
                    values,
                    color=style["color"],
                    s=8,
                    alpha=0.35,
                    linewidths=0,
                )
                axis.plot(
                    x_values,
                    connector,
                    color=style["color"],
                    linestyle=style["line"],
                    linewidth=1.55,
                    marker=style["marker"],
                    markevery=max(1, len(x_values) // 10),
                    markersize=4,
                )
                for boundary in epoch_boundaries(rows):
                    axis.axvline(boundary, color="#999", linewidth=0.6, alpha=0.3)
                plotted = True
        axis.set_title(f"table LR × {scale:g}", fontsize=10)
        axis.set_xlabel("step", fontsize=8)
        axis.grid(alpha=0.22)
        if not plotted:
            axis.text(0.5, 0.5, "no local record", ha="center", va="center", transform=axis.transAxes)
    for axis in axes[::5]:
        axis.set_ylabel("online gap = fixed val − online train")
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=style["color"],
                linestyle=style["line"],
                marker=style["marker"],
                markersize=5,
                label=f"β₂={beta2:g}",
            )
            for beta2, style in BETA_STYLES.items()
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.suptitle(
        "V5 table-LR scale comparison: low scale and high scale in one figure\n"
        "navy solid circles = β₂=.99; orange dashed squares = β₂=.999; "
        "points = raw online records; lines = 3-point visual connectors; gray lines = epoch boundaries",
        fontsize=11,
        y=0.965,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.25, w_pad=1.0)
    out = FIGDIR / "fig_v5_optv5f_all_scale_gap_facets.png"
    figure.savefig(out, dpi=180)
    plt.close(figure)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_overview()
    plot_high_scale_2000_facets()