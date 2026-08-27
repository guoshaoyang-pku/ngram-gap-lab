#!/usr/bin/env python3
"""X1/X2 verdict figures (2026-08-26 wave).

Reads only data/runs_fixed/{optv5c_*,ctbl_dim*}_fixed/summary.json +
train_log.jsonl.  Zero GPU, read-only.  Outputs docs/figs/theory/.

X1: table optimizer x seed 43/44 -> final gap (rmsprop vs adamw vs sgd, scale 2.0)
X2: clean-table row width d in {768(baseline),192,48,12} -> final gap
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "runs_fixed"
OUT = ROOT / "docs" / "figs" / "theory"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#26364d"
MUTED = "#6b7280"
BLUE = "#2f63a6"
RED = "#bb4b4b"
GREEN = "#3d8b68"
GOLD = "#b07d24"

# X1 arms: label, seed43 run, seed44 run, color
X1_ARMS = [
    ("RMSProp (0, 0.99)", "optv5c_rms_b099_s2p0_r1_s43_fixed",
     "optv5c_rms_b099_s2p0_r1_s44_fixed", BLUE),
    ("AdamW (0, 0.99)", "optv5c_adamw_b099_s2p0_s43_fixed",
     "optv5c_adamw_b099_s2p0_s44_fixed", GOLD),
    ("SGD m=0", "optv5c_sgd_m0_s2p0_s43_fixed",
     "optv5c_sgd_m0_s2p0_s44_fixed", GREEN),
]

# X2 rows: label, run, color  (d=768 is the v5 baseline ng-lab 1x input)
X2_ROWS = [
    ("d=768 (v5 base)", None, MUTED, 5.741),  # ng-lab1x input v5, experiment-lines
    ("d=192", "ctbl_dim192_input_v5_fixed", BLUE, None),
    ("d=48", "ctbl_dim48_input_v5_fixed", GREEN, None),
    ("d=12", "ctbl_dim12_input_v5_fixed", RED, None),
]


def gap_of(run: str) -> float | None:
    p = RUNS / run / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("final_gap")


def load_curve(run: str) -> list[tuple[int, float, float]]:
    """Return [(step, train, val)] from train_log.jsonl."""
    p = RUNS / run / "train_log.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        s, tr, va = d.get("step"), d.get("train_loss"), d.get("val_loss")
        if s is not None and tr is not None and va is not None:
            out.append((int(s), float(tr), float(va)))
    return out


def plot_x1() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    # left: final gap by arm, two seeds + mean
    ax = axes[0]
    means = []
    for i, (label, r43, r44, color) in enumerate(X1_ARMS):
        g43, g44 = gap_of(r43), gap_of(r44)
        xs, gs = [], []
        if g43 is not None:
            xs.append(-0.18); gs.append(g43)
        if g44 is not None:
            xs.append(0.18); gs.append(g44)
        ax.scatter([i + x for x in xs], gs, s=55, color=color, zorder=3,
                   label=f"{label}" if i == 0 else None)
        both = [g for g in (g43, g44) if g is not None]
        if both:
            m = float(np.mean(both))
            means.append((i, m, color))
            ax.errorbar([i], [m], yerr=[(max(both) - min(both)) / 2], fmt="none",
                        ecolor=color, elinewidth=1.4, capsize=4, zorder=2)
    for i, m, color in means:
        ax.scatter([i], [m], marker="_", s=160, color=color, zorder=4)
        ax.annotate(f"{m:.3f}", (i, m + 0.05), ha="center", fontsize=9, color=color)
    ax.axhline(1.534, color=MUTED, lw=1.2, ls="--")
    ax.text(2.42, 1.545, "seed-42 v5 baseline 1.534", fontsize=8, color=MUTED, ha="right")
    ax.set_xticks(range(len(X1_ARMS)))
    ax.set_xticklabels([l for l, *_ in X1_ARMS], fontsize=10)
    ax.set_ylabel("final gap (val − train, step 1000)")
    ax.set_title("X1 · table optimizer × seed 43/44 (scale 2.0)")
    ax.set_ylim(-0.1, 2.0)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # right: train/val/gap curves for the three arms (seed 43)
    ax = axes[1]
    for i, (label, r43, r44, color) in enumerate(X1_ARMS):
        curve = load_curve(r43)
        if not curve:
            continue
        steps = [c[0] for c in curve]
        gap = [c[2] - c[1] for c in curve]
        ax.plot(steps, gap, color=color, lw=1.6, label=label)
    ax.axhline(0, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("gap (val − train)")
    ax.set_title("X1 · gap trajectory, seed 43")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "fig_theory_x1_optimizer.svg")
    fig.savefig(OUT / "fig_theory_x1_optimizer.png", dpi=180)
    plt.close(fig)


def plot_x2() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    labels, gaps, colors = [], [], []
    for label, run, color, fixed in X2_ROWS:
        g = fixed if run is None else gap_of(run)
        labels.append(label)
        gaps.append(g)
        colors.append(color)
    xs = np.arange(len(labels))
    ax.bar(xs, gaps, color=colors, alpha=0.9, width=0.62)
    for x, g in zip(xs, gaps):
        ax.annotate(f"{g:.3f}", (x, g + 0.05), ha="center", fontsize=10, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("final gap (val − train, step 1000)")
    ax.set_title("X2 · clean-table row width d (seed 42, scale 2.0)")
    ax.set_ylim(0, 6.4)
    ax.grid(alpha=0.25, axis="y")
    ax.axhline(0, color=MUTED, lw=1)
    # annotate monotone drop
    ax.text(1.5, 5.95, "monotone: narrower rows → smaller gap",
            ha="center", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_theory_x2_tabledim.svg")
    fig.savefig(OUT / "fig_theory_x2_tabledim.png", dpi=180)
    plt.close(fig)


def main() -> None:
    plot_x1()
    plot_x2()
    for label, r43, r44, _ in X1_ARMS:
        g43, g44 = gap_of(r43), gap_of(r44)
        s43 = f"{g43:.4f}" if g43 is not None else "None"
        s44 = f"{g44:.4f}" if g44 is not None else "None"
        print(f"{label}: s43={s43} s44={s44}")
    for label, run, _, fixed in X2_ROWS:
        if run:
            print(f"{label}: {gap_of(run):.4f}")
        else:
            print(f"{label}: {fixed} (baseline)")


if __name__ == "__main__":
    main()
