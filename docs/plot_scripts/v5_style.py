"""Shared v5 figure style for ngram-gap-lab doc plots.

Import as `from v5_style import *` (or `import v5_style`) at the top of a
plotting script after matplotlib.use("Agg").  Provides:

  apply_style()      -- set the unified rcParams (call once)
  ARM_COLORS         -- input/y/v/nogram
  BRANCH_COLORS      -- bigram/trigram
  ERA_COLORS         -- 2x historical / 128x current
  add_epoch_lines()  -- dotted epoch-boundary vlines
  save()             -- save PNG + SVG with the standard dpi
"""

from pathlib import Path

import matplotlib.pyplot as plt

ARM_COLORS = {
    "input": "#2d6f9f",
    "y": "#c4493d",
    "v": "#c58a0b",
    "nogram": "#686d73",
}
BRANCH_COLORS = {"bigram": "#14736f", "trigram": "#7b4fa6"}
ERA_COLORS = {"2x": "#9aa3ad", "128x": "#2d6f9f"}
FIT_COLOR = "#1a7f37"
HOLDOUT_COLOR = "#c4493d"


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "Hiragino Sans GB",
                            "PingFang SC", "Arial Unicode MS", "DejaVu Sans"],
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })


def add_epoch_lines(axis, epoch_len=337, xmax=None, ymax=None, color="#b0b7bf"):
    """Dotted vertical epoch boundaries (1 epoch = 337 steps by default)."""
    if xmax is None:
        xmax = axis.get_xlim()[1]
    e = epoch_len
    while e <= xmax:
        axis.axvline(e, color=color, lw=0.7, ls=":", zorder=1)
        e += epoch_len


def save(fig, out_dir, name, dpi=190):
    """Save <name>.png and <name>.svg into out_dir; return both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg
