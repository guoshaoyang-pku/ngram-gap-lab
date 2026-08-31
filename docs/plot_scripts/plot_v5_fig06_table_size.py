#!/usr/bin/env python3
"""Fig 6: S1 clean table-size scaling, log-log, unified style (PNG+SVG).

Data: docs/appendices/s1_scaling_three_axis/s1_table_size_points.csv
(single-table scans: only the scanned table is enabled, the other disabled).
All explanatory text lives in the HTML caption -- nothing but data, fits and
a minimal legend inside the figure.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_table_size_points.csv"
OUT = ROOT / "docs" / "figs" / "main"
NAME = "fig_v5_s1_table_size_loglog_clean"

NO_GRAM_FLOOR = 0.02
SMALL_R_CUTOFF = 10000
WINDOWS = {"bigram": (2e3, 2e5), "trigram": (1.0e5, 9.3e5)}


def smooth3(y):
    y = np.asarray(y, dtype=float)
    out = np.copy(y)
    for i in range(1, len(y) - 1):
        out[i] = (y[i - 1] + y[i] + y[i + 1]) / 3.0
    return out


def main():
    S.apply_style()
    rows = list(csv.DictReader(CSV.open()))
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for branch in ("bigram", "trigram"):
        color = S.BRANCH_COLORS[branch]
        br = sorted([r for r in rows if r["branch"] == branch], key=lambda r: float(r["R"]))
        x = np.array([float(r["R"]) for r in br])
        y = np.array([float(r["final_gap"]) for r in br])
        large = x > SMALL_R_CUTOFF
        ax.scatter(x[large], y[large], color=color, s=22, alpha=0.85, zorder=3,
                   label=f"{branch} runs")
        ax.scatter(x[~large], y[~large], facecolors="none", edgecolors=color,
                   s=26, linewidths=1.0, alpha=0.85, zorder=3)
        order = np.argsort(x)
        ax.plot(x[order], smooth3(y[order]), color=color, lw=0.8, alpha=0.7, zorder=2)
        lo, hi = WINDOWS[branch]
        m = (x >= lo) & (x <= hi) & (y - NO_GRAM_FLOOR > 0)
        slope, intercept = np.polyfit(np.log(x[m]), np.log(y[m] - NO_GRAM_FLOOR), 1)
        fx = np.geomspace(x[m].min(), x[m].max(), 120)
        ax.plot(fx, NO_GRAM_FLOOR + np.exp(intercept) * fx ** slope,
                color=color, ls="--", lw=1.3, zorder=4,
                label=f"{branch}: (gap$-${NO_GRAM_FLOOR}) $\\propto R^{{{slope:.3f}}}$")
    ax.axvspan(1, SMALL_R_CUTOFF, color="#888888", alpha=0.07)
    ax.text(1.6, 0.085, "small-R collapse\n(gap -> no-gram floor)", fontsize=8,
            color="#666", ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical rows R of the varied table")
    ax.set_ylabel("online gap @ step 1000")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    png, svg = S.save(fig, OUT, NAME)
    print(png.relative_to(ROOT))
    print(svg.relative_to(ROOT))


if __name__ == "__main__":
    main()
