#!/usr/bin/env python3
"""Fig 4 (restyle): gap vs exact train hit-count f, log-log, unified style.

Both branches in one figure (two panels sharing the y axis), raw exact-f
points behind 7 wide geometric bins with the registered power-law fit.
Data: docs/appendices/s1_scaling_three_axis/s1_frequency_exact_points.csv
      docs/appendices/s1_scaling_three_axis/s1_scaling_fits.csv
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
PTS = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_frequency_exact_points.csv"
FITS = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_scaling_fits.csv"
OUT = ROOT / "docs" / "figs" / "main"

N_BINS = 7
MIN_CONTEXTS = 32


def geometric_bins(rows):
    """Pool exact-f rows into N_BINS wide geometric bins by token mass."""
    fs = np.array([float(r["f"]) for r in rows])
    lo, hi = fs.min(), fs.max()
    edges = np.geomspace(lo, hi * 1.0001, N_BINS + 1)
    bins = []
    for i in range(N_BINS):
        sel = [(float(r["f"]), float(r["gap"]), float(r["shared_token_mass"]))
               for r in rows if edges[i] <= float(r["f"]) < edges[i + 1]]
        if not sel:
            continue
        f = np.array([s[0] for s in sel])
        g = np.array([s[1] for s in sel])
        w = np.array([s[2] for s in sel])
        fbar = np.exp(np.average(np.log(f), weights=w))
        gbar = np.average(g, weights=w)
        bins.append((fbar, gbar, f.min(), f.max()))
    return bins


def main():
    S.apply_style()
    fits = {r["branch"]: float(r["slope"]) for r in csv.DictReader(FITS.open())
            if r["family"] == "frequency_exact"}
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    for ax, branch in zip(axes, ("bigram", "trigram")):
        color = S.BRANCH_COLORS[branch]
        rows = [r for r in csv.DictReader(PTS.open())
                if r["branch"] == branch and int(r["step"]) == 1000
                and float(r["gap"]) > 0 and int(r["shared_contexts"]) >= MIN_CONTEXTS]
        f = np.array([float(r["f"]) for r in rows])
        g = np.array([float(r["gap"]) for r in rows])
        ax.scatter(f, g, s=7, color="#9aa3ad", alpha=0.45, zorder=2,
                   label="exact-f points (positive gap)")
        bins = geometric_bins(rows)
        bx = np.array([b[0] for b in bins])
        bg = np.array([b[1] for b in bins])
        xerr = np.array([[b[0] - b[2] for b in bins], [b[3] - b[0] for b in bins]])
        ax.errorbar(bx, bg, xerr=xerr, fmt="o-", color=color, ms=5, lw=1.2,
                    capsize=2, zorder=3, label="geometric bins (mass-weighted)")
        slope = fits[branch]
        fx = np.geomspace(bx.min(), bx.max(), 80)
        amp = np.exp(np.mean(np.log(bg) - slope * np.log(bx)))
        ax.plot(fx, np.exp(np.log(amp) + slope * np.log(fx)), ls="--", lw=1.2,
                color=S.HOLDOUT_COLOR, zorder=4,
                label=f"fit: $G \\propto f^{{{slope:.3f}}}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("exact train hit-count per context f")
        ax.set_title(branch)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("gap (val - train probe) @ step 1000")
    fig.tight_layout()
    png, svg = S.save(fig, OUT, "fig_v5_s1_frequency_exact_f")
    print(png.relative_to(ROOT))
    print(svg.relative_to(ROOT))


if __name__ == "__main__":
    main()
