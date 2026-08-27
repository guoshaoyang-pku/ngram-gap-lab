#!/usr/bin/env python3
"""Plot readable small-multiple figures for the v5 table-LR evidence."""

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
SERIES = {
    (1000, 0.99): {"color": "#164e63", "line": "-", "marker": "o", "label": "1000 · β₂=.99"},
    (1000, 0.999): {"color": "#c2410c", "line": "--", "marker": "D", "label": "1000 · β₂=.999"},
    (2000, 0.99): {"color": "#047857", "line": "-", "marker": "s", "label": "2000 · β₂=.99"},
    (2000, 0.999): {"color": "#7e22ce", "line": "--", "marker": "^", "label": "2000 · β₂=.999"},
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


def is_complete(run_id):
    summary = load_summary(run_id)
    return bool(summary and str(summary.get("status", "done")).lower() == "done")


def final_gap(run_id):
    summary = load_summary(run_id)
    if summary:
        value = summary.get("final_gap")
        if isinstance(value, (int, float)) and math.isfinite(value):
            return value
    rows = load_rows(run_id)
    if rows:
        value = rows[-1].get("gap")
        if isinstance(value, (int, float)) and math.isfinite(value):
            return value
    return None


def smooth(values):
    if len(values) < 3:
        return np.asarray(values, dtype=float)
    kernel = np.ones(3) / 3
    padded = np.pad(np.asarray(values, dtype=float), (1, 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def style_for(steps, beta2):
    return SERIES[(steps, beta2)]


def legend_handles(include_status=False):
    handles = [
        Line2D(
            [],
            [],
            color=style["color"],
            linestyle=style["line"],
            marker=style["marker"],
            markersize=5,
            linewidth=1.8,
            label=style["label"],
        )
        for style in SERIES.values()
    ]
    if include_status:
        handles.extend(
            [
                Line2D([], [], color="#555", marker="o", linestyle="none", markerfacecolor="#555", label="已完成"),
                Line2D([], [], color="#555", marker="o", linestyle="none", markerfacecolor="white", label="运行中/中途"),
            ]
        )
    return handles


def plot_final_gap_vs_scale():
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for axis, budget in zip(axes, (1000, 2000)):
        for beta2 in (0.99, 0.999):
            for scale in SCALES:
                run_id = next(
                    (
                        rid
                        for rid, (run_scale, run_beta, run_steps) in FAMILY.items()
                        if run_scale == scale and run_beta == beta2 and run_steps == budget
                    ),
                    None,
                )
                if run_id is None:
                    continue
                gap = final_gap(run_id)
                if gap is None:
                    continue
                style = style_for(budget, beta2)
                complete = is_complete(run_id)
                axis.plot(
                    scale,
                    gap,
                    marker=style["marker"],
                    markersize=8,
                    markerfacecolor=style["color"] if complete else "white",
                    markeredgecolor=style["color"],
                    markeredgewidth=1.8,
                    color=style["color"],
                    linestyle="none",
                    zorder=3,
                )
                axis.annotate(
                    f"{gap:.2f}",
                    (scale, gap),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=style["color"],
                )
        axis.set_xscale("log", base=2)
        axis.set_xticks(SCALES, [f"{scale:g}×" for scale in SCALES])
        axis.set_xlabel("table LR scale")
        axis.grid(alpha=0.25, which="both")
        axis.set_title(f"{budget} steps")
    axes[0].set_ylabel("latest recorded gap = fixed val − online train")
    axes[0].legend(handles=legend_handles(True), fontsize=8, frameon=False, loc="upper left")
    figure.suptitle(
        "V5 table-LR sweep: low scale and high scale on one readable axis\n"
        "color/shape = budget × β₂; filled = completed, hollow = running/partial; "
        "labels are raw latest recorded gaps",
        fontsize=11,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.91])
    out = FIGDIR / "fig_v5_optv5f_final_gap.png"
    figure.savefig(out, dpi=170)
    plt.close(figure)
    print(f"wrote {out}")


def plot_metric_facets(metric, ylabel, filename, title):
    figure, axes = plt.subplots(2, 5, figsize=(18, 7.5), sharey=True)
    axes = axes.ravel()
    for axis, scale in zip(axes, SCALES):
        plotted = False
        for run_id, (run_scale, beta2, steps) in FAMILY.items():
            if run_scale != scale:
                continue
            rows = load_rows(run_id)
            if not rows:
                continue
            style = style_for(steps, beta2)
            x_values = np.asarray([row["step"] for row in rows])
            values = np.asarray(
                [
                    row.get(metric, row["val_loss"] - row["train_loss"])
                    for row in rows
                ],
                dtype=float,
            )
            axis.scatter(
                x_values,
                values,
                color=style["color"],
                s=7,
                alpha=0.42,
                linewidths=0,
            )
            axis.plot(
                x_values,
                smooth(values),
                color=style["color"],
                linestyle=style["line"],
                linewidth=1.35,
            )
            plotted = True
        axis.set_title(f"table LR × {scale:g}", fontsize=10)
        axis.grid(alpha=0.22)
        axis.set_xlabel("step", fontsize=8)
        if not plotted:
            axis.text(0.5, 0.5, "无本地记录", ha="center", va="center", transform=axis.transAxes)
    for axis in axes[::5]:
        axis.set_ylabel(ylabel)
    figure.legend(
        handles=legend_handles(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.suptitle(
        f"{title}\n每个小面板只比较同一个 table-LR scale；点=原始记录，线=3点视觉连接",
        fontsize=11,
        y=0.955,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.91])
    out = FIGDIR / filename
    figure.savefig(out, dpi=170)
    plt.close(figure)
    print(f"wrote {out}")


def plot_beta2_comparison():
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, budget in zip(axes, (1000, 2000)):
        for beta2 in (0.99, 0.999):
            xs, ys = [], []
            for scale in (8.0, 16.0, 32.0, 64.0, 128.0):
                run_id = next(
                    (
                        rid
                        for rid, (run_scale, run_beta, run_steps) in FAMILY.items()
                        if run_scale == scale and run_beta == beta2 and run_steps == budget
                    ),
                    None,
                )
                if run_id is None:
                    continue
                gap = final_gap(run_id)
                if gap is not None:
                    xs.append(scale)
                    ys.append(gap)
            style = style_for(budget, beta2)
            axis.plot(
                xs,
                ys,
                color=style["color"],
                linestyle=style["line"],
                marker=style["marker"],
                label=style["label"],
                linewidth=1.5,
                markersize=5,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([8, 16, 32, 64, 128], ["8×", "16×", "32×", "64×", "128×"])
        axis.set_xlabel("table LR scale")
        axis.set_ylabel("latest recorded gap")
        axis.set_title(f"{budget} steps")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle("V5 high-scale β₂ comparison · direct style encoding", fontsize=11)
    figure.tight_layout()
    out = FIGDIR / "fig_v5_optv5f_beta2_compare.png"
    figure.savefig(out, dpi=170)
    plt.close(figure)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_final_gap_vs_scale()
    plot_metric_facets(
        "gap",
        "online gap",
        "fig_v5_optv5f_conv_curves.png",
        "V5 gap trajectories by table-LR scale",
    )
    plot_metric_facets(
        "train_loss",
        "online train loss",
        "fig_v5_optv5f_train_facets.png",
        "V5 online train-loss trajectories by table-LR scale",
    )
    plot_metric_facets(
        "val_loss",
        "fixed validation loss",
        "fig_v5_optv5f_val_facets.png",
        "V5 fixed-validation trajectories by table-LR scale",
    )
    plot_beta2_comparison()
    print("done")