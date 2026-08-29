#!/usr/bin/env python3
"""Plot beta2=.99 gap at optimizer step 1000 across table LR scales."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data" / "runs_fixed"
FIGDIR = REPO / "docs" / "figs" / "main"
FIGDIR.mkdir(parents=True, exist_ok=True)

RUNS_BY_SCALE = {
    0.5: "optv5c_rms_b099_s0p5",
    1.0: "optv5c_rms_b099_s1p0",
    2.0: "optv5c_rms_b099_s2p0_r1",
    3.0: "optv5c_rms_b099_s3p0",
    4.0: "optv5c_rms_b099_s4p0",
    8.0: "optv5f_rms_b099_s8p0_2k",
    16.0: "optv5f_rms_b099_s16p0_2k",
    32.0: "optv5f_rms_b099_s32p0",
    64.0: "optv5f_rms_b099_s64p0",
    128.0: "optv5f_rms_b099_s128p0",
    256.0: "optv5f_rms_b099_s256p0_2k",
    512.0: "optv5f_rms_b099_s512p0_2k",
    1024.0: "optv5f_rms_b099_s1024p0_2k",
}


def step_row(run_id, step=1000):
    path = RUNS / f"{run_id}_fixed" / "train_log.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("step") == step:
            rows.append(row)
    if len(rows) != 1:
        raise RuntimeError(f"{run_id}: expected one step-{step} row, found {len(rows)}")
    return rows[0]


def main():
    scales = []
    gaps = []
    source_labels = []
    for scale, run_id in RUNS_BY_SCALE.items():
        row = step_row(run_id)
        gap = row.get("gap")
        if not isinstance(gap, (int, float)) or not math.isfinite(gap):
            raise RuntimeError(f"{run_id}: invalid gap at step 1000")
        scales.append(scale)
        gaps.append(gap)
        source_labels.append(run_id)

    figure, axis = plt.subplots(figsize=(11, 6.2))
    axis.plot(
        scales,
        gaps,
        color="#123b5d",
        linewidth=1.45,
        marker="o",
        markersize=7,
        markerfacecolor="#123b5d",
        markeredgecolor="white",
        markeredgewidth=1.1,
        zorder=3,
    )
    for scale, gap in zip(scales, gaps):
        axis.annotate(
            f"{gap:.2f}",
            (scale, gap),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color="#123b5d",
        )
    axis.axvline(128.0, color="#c2410c", linestyle=":", linewidth=1.1, alpha=0.85)
    axis.text(
        128.0,
        0.02,
        "SSOT table-LR scale = 128×",
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#c2410c",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(scales, [f"{scale:g}×" for scale in scales])
    axis.set_xlabel("table LR scale")
    axis.set_ylabel("gap at optimizer step 1000 = fixed val − online train")
    axis.set_title("V5 β₂=.99: gap at step 1000 versus table LR scale")
    axis.grid(alpha=0.25, which="both")
    axis.set_ylim(bottom=min(-0.1, min(gaps) - 0.15))
    axis.text(
        0.01,
        0.02,
        "All points: β₂=.99, seed 42, raw step-1000 records.\n"
        "8×/16×/256×/512×/1024× are read from the corresponding 2000-step runs at step 1000.",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    figure.suptitle(
        "Only β₂=.99 · input injection · clean bigram+trigram tables · v5 fixed measurement contract",
        fontsize=11,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    output = FIGDIR / "fig_v5_beta099_gap_step1000_vs_table_lr.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"wrote {output}")
    for scale, gap, run_id in zip(scales, gaps, source_labels):
        print(f"{scale:g}x\t{gap:.9f}\t{run_id}")


if __name__ == "__main__":
    main()