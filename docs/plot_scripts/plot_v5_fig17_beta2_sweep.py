#!/usr/bin/env python3
"""Fig 17 (redo): table beta2 sweep at step 1000, unified style (PNG+SVG).

Panel A: step-1000 final gap vs table beta2 (RMSProp, scale=2, seed 42)
from the committed v5_optimizer_points.csv (optv5c batch, 1k steps).
The doc text adds the high-scale (.99 vs .999) comparison separately; only
1k-step records are used for endpoint comparisons.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "v5_optimizer_points.csv"
OUT = ROOT / "docs" / "figs" / "main"


def main():
    S.apply_style()
    rows = [r for r in csv.DictReader(CSV.open()) if r["family"] == "beta2"]
    rows.sort(key=lambda r: float(r["table_betas"].split(",")[1]))
    beta = np.array([float(r["table_betas"].split(",")[1]) for r in rows])
    gap = np.array([float(r["final_gap"]) for r in rows])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(beta))
    ax.plot(xs, gap, color="#2d6f9f", lw=1.0, zorder=2)
    ax.scatter(xs, gap, color="#2d6f9f", s=42, zorder=3)
    for x, g in zip(xs, gap):
        ax.annotate(f"{g:.2f}", (x, g), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, color="#1c2733")
    ax.axvline(2, color="#c4493d", lw=0.9, ls="--")
    ax.text(2.05, ax.get_ylim()[0] + 0.02, " chosen: 0.99", color="#c4493d",
            fontsize=8.5, va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{b:g}" for b in beta])
    ax.set_xlabel(r"table RMSProp $\beta_2$ (scale = 2, 1000 steps, seed 42)")
    ax.set_ylabel("online gap @ step 1000")
    fig.tight_layout()
    png, svg = S.save(fig, OUT, "fig_v5_beta2_sweep_1k")
    print(png.relative_to(ROOT))
    print(svg.relative_to(ROOT))


if __name__ == "__main__":
    main()
