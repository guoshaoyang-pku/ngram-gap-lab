#!/usr/bin/env python3
"""Generate figures for the natural-language 5gram (order=5) line.

Reads ``data/runs_fixed/ngram5_order5_*_fixed/`` (trainer.py outputs) and
produces ``docs/figs/fig_ngram5_order5_*.{svg,png}``:

  1. fig_ngram5_order5_gap.svg      -- global gap vs step, all arms
  2. fig_ngram5_order5_gap_freq.svg -- per-bucket gap vs frequency @step 2000

Runs (table LR):
  ngram5_order5_trigram_fixed          (+trigram, LR x2)   [anchor]
  ngram5_order5_trigram_s43_fixed      (+trigram, LR x2)   [seed 43]
  ngram5_order5_trigram_lr1x_fixed     (+trigram, LR x1)
  ngram5_order5_trigram_lr4x_fixed     (+trigram, LR x4)
  ngram5_order5_puretransformer_fixed  (no table)

Frequency bins (exact train-context count r):
  0,1,2,3,4,5,6,11,21,51,101,201,501,1001,5001
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data" / "runs_fixed"
FIGS = REPO / "docs" / "figs"

PAPER = "#f7f5ef"
BORDER = "#c8c1b6"
MUTED = "#686d73"
ANCHOR = "#353d79"
TRIGRAM = "#c4493d"
TRAIN = "#2d6f9f"
VAL = "#c4493d"
GREEN = "#3c8d5a"
ORANGE = "#d97932"

RUNS_DEF = [
    ("ngram5_order5_trigram_fixed", "+trigram (table LR x2)", "#2d6f9f", "-o"),
    ("ngram5_order5_trigram_s43_fixed", "+trigram (table LR x2, seed 43)", "#6a86b8", "--o"),
    ("ngram5_order5_trigram_lr1x_fixed", "+trigram (table LR x1)", "#b67524", "-s"),
    ("ngram5_order5_trigram_lr4x_fixed", "+trigram (table LR x4)", "#c4493d", "-^"),
    ("ngram5_order5_puretransformer_fixed", "pure transformer", "#3c8d5a", "-D"),
]

EDGES = [0, 1, 2, 3, 4, 5, 6, 11, 21, 51, 101, 201, 501, 1001, 5001]
BUCKET_LABELS = [f"[{EDGES[i]},{EDGES[i+1]})" for i in range(len(EDGES) - 1)]
BUCKET_LABELS.append("[5001,inf)")


def bucket_index(freq: int) -> int:
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= freq < EDGES[i + 1]:
            return i
    return len(EDGES) - 1


def load_train_val(run_dir: Path):
    """Return (steps, train_loss, val_loss, gap) from training/validation jsonl."""
    tr = {}
    for line in (run_dir / "training_loss.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tr[int(r["step"])] = r["train_loss"]
    val = {}
    for line in (run_dir / "validation_loss.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        val[int(r["step"])] = r["val_loss"]
    steps = sorted(set(tr) | set(val))
    return steps, tr, val


def load_bucket_gap(run_dir: Path, step: int):
    """Return per-bucket weighted gap at a given probe step (train probe_set)."""
    rows = [
        json.loads(l)
        for l in (run_dir / "allgram_frequency_decomposition.jsonl").read_text().splitlines()
        if l.strip()
    ]
    sel = [r for r in rows if r["step"] == step and r["probe_set"] == "train"
           and r["val_loss"] is not None]
    agg = defaultdict(lambda: [0.0, 0.0])  # bucket -> [sum(gap*w), sum(w)]
    for r in sel:
        b = bucket_index(r["frequency"])
        w = r["train_fraction"] or 0.0
        agg[b][0] += (r["val_loss"] - r["train_loss"]) * w
        agg[b][1] += w
    return [agg[b][0] / agg[b][1] if agg[b][1] > 0 else np.nan
            for b in range(len(BUCKET_LABELS))]


def make_style():
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": MUTED,
        "text.color": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e5e0d6",
        "grid.linewidth": 0.7,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    })


def fig_gap_curves():
    make_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for run_id, label, color, marker in RUNS_DEF:
        run_dir = RUNS / run_id
        if not run_dir.exists():
            print(f"[skip] {run_id} missing")
            continue
        steps, tr, val = load_train_val(run_dir)
        xs = sorted(val)
        gaps = [val[s] - tr[s] for s in xs]
        ax.plot(xs, gaps, marker, color=color, label=label, lw=2.2, ms=2.8)
    ax.axhline(0, color=BORDER, lw=1.0, ls="--")
    ax.set_xlabel("step")
    ax.set_ylabel("val loss − train loss (global gap)")
    ax.set_title("Natural-language 5gram (order=5): global gap, seeds 42/43")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    out = FIGS / "fig_ngram5_order5_gap"
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"wrote {out}.svg and {out}.png")


def fig_gap_vs_freq():
    make_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(BUCKET_LABELS))
    width = 0.2
    for i, (run_id, label, color, marker) in enumerate(RUNS_DEF):
        run_dir = RUNS / run_id
        if not run_dir.exists():
            print(f"[skip] {run_id} missing")
            continue
        gaps = load_bucket_gap(run_dir, 2000)
        ax.plot(x + (i - 1.5) * width, gaps, marker, color=color, label=label,
                lw=2.0, ms=4.5)
    ax.axhline(0, color=BORDER, lw=1.0, ls="--")
    ax.axvspan(7.5, 12.5, color="#2d6f9f", alpha=0.06, zorder=0)
    ax.text(10, 0.92, "exploratory mid-frequency region",
            transform=ax.get_yaxis_transform(), color="#2d6f9f",
            fontsize=9, ha="center", va="top")
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("train-context frequency r (exact count per epoch)")
    ax.set_ylabel("within-bucket gap (val − train), weighted")
    ax.set_title("Natural-language 5gram: per-bucket gap @2000, seeds 42/43")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.25, top=0.90)
    out = FIGS / "fig_ngram5_order5_gap_freq"
    fig.savefig(out.with_suffix(".svg"))
    fig.savefig(out.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"wrote {out}.svg and {out}.png")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_gap_curves()
    fig_gap_vs_freq()


if __name__ == "__main__":
    main()
