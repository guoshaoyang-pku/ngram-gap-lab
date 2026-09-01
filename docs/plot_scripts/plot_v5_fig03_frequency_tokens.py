#!/usr/bin/env python3
"""Fig 3 (redesign): token mass by log-spaced frequency bucket + per-bucket gap.

Absorbs the old token-mass figure (fig 5).  For each branch (bigram, trigram):
  * bars  = token counts per bucket: train = exact full-epoch mass from the
            frequency index (sum of f * n(f)); val = fixed-val counts from the
            main run's freq_bin_loss.jsonl (step-1000 record);
  * a separate leftmost bin holds contexts unseen in train (f = 0; train mass
    is identically zero, val carries the mass);
  * line  = per-bucket online gap (val mean loss - current-batch train mean
    loss) on a secondary axis; undefined at f = 0 (no train loss).

Two output versions:
  raw    -- counts as measured (train epoch ~ 49.7M tokens vs val ~ 0.59M);
  scaled -- val counts multiplied by train_total/val_total (~84.29) so the
            totals match and the shapes are directly comparable.

Set NGLAB_RUNS_FIXED to mirror location of nglab1x_input_v5_128x_freq10_fixed.
"""
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
RUNS_FIXED = Path(os.environ.get("NGLAB_RUNS_FIXED", ROOT / "data" / "runs_fixed"))
FREQ_JSONL = RUNS_FIXED / "nglab1x_input_v5_128x_freq10_fixed" / "freq_bin_loss.jsonl"
TRAIN_CSV = ROOT / "docs" / "figs" / "theory" / "freq_train_mass_hist.csv"
OUT = ROOT / "docs" / "figs" / "main"
STEP = 1000

BUCKETS = ["1", "2", "3", "4", "5", "6-8", "9-12", "13-20", "21-30", "31-50",
           "51-75", "76-100", "101-150", "151-200", "201-300", "301-500",
           "501-750", "751-1k", "1k-2k", "2k-5k", "5k-10k", "10k+"]


def load_val_side():
    for line in FREQ_JSONL.open():
        rec = json.loads(line)
        if rec["step"] == STEP:
            return rec
    raise SystemExit(f"step {STEP} not found in {FREQ_JSONL}")


def load_train_mass():
    mass = {"bigram": {}, "trigram": {}}
    for row in csv.DictReader(TRAIN_CSV.open()):
        mass[row["branch"]][row["bucket"]] = int(row["token_mass"])
    return mass


def make(version):
    S.apply_style()
    rec = load_val_side()
    train_mass = load_train_mass()
    scale = 1.0
    if version == "scaled":
        # ratio of full-epoch tokens to evaluated fixed-val tokens
        val_total = sum(rec["val"]["bigram"][b]["token_count"] for b in BUCKETS + ["novel"])
        train_total = sum(train_mass["bigram"].values())
        scale = train_total / val_total

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=False)
    for ax, branch in zip(axes, ("bigram", "trigram")):
        color = S.BRANCH_COLORS[branch]
        labels = ["f=0\n(unseen)"] + BUCKETS
        xs = np.arange(len(labels))
        train_counts = [0.0] + [train_mass[branch][b] for b in BUCKETS]
        val_counts = [rec["val"][branch]["novel"]["token_count"]] + [
            rec["val"][branch][b]["token_count"] for b in BUCKETS]
        train_counts = np.array(train_counts, dtype=float)
        val_counts = np.array(val_counts, dtype=float) * scale

        w = 0.42
        ax.bar(xs - w / 2, np.maximum(train_counts, 0.5), width=w, color="#2d6f9f",
               alpha=0.72, label="train tokens (full epoch)" if branch == "bigram" else None)
        ax.bar(xs + w / 2, np.maximum(val_counts, 0.5), width=w, color="#c4493d",
               alpha=0.72,
               label=("val tokens (fixed set)" if version == "raw"
                      else "val tokens x84.3 (totals matched)") if branch == "bigram" else None)
        ax.set_yscale("log")
        ax.set_ylim(1e4 if version == "raw" else 1e5, 10**7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_xlabel(f"{branch} context train hit-count f (log-spaced buckets)")
        ax.set_ylabel("token count (log)")
        ax.set_title(f"{branch}")

        ax2 = ax.twinx()
        ax2.grid(False)
        gaps, gx = [], []
        for i, b in enumerate(BUCKETS, start=1):
            tr = rec["train"][branch][b]
            va = rec["val"][branch][b]
            if tr["token_count"] > 0 and va["token_count"] > 0:
                gaps.append(va["mean_loss"] - tr["mean_loss"])
                gx.append(i)
        ax2.plot(gx, gaps, color=color, marker="o", ms=3.5, lw=1.2,
                 label="gap (val - train)" if branch == "bigram" else None)
        ax2.axhline(0, color="#999", lw=0.7, ls=":")
        ax2.set_ylabel("online gap @ step 1000", color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim(-1, 8)
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(color)

    handles1, labels1 = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[0].get_legend().get_children() if False else ([], [])
    extra = axes[0].twinned if hasattr(axes[0], "twinned") else []
    fig.legend(handles1 + [plt.Line2D([], [], color="#333", marker="o", ms=3.5, lw=1.2)],
               labels1 + ["gap (val - train), right axis"],
               loc="upper center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"fig_v5_128x_injection_frequency_tokens_{version}"
    png, svg = S.save(fig, OUT, name)
    print(png.relative_to(ROOT))
    print(svg.relative_to(ROOT))


if __name__ == "__main__":
    make("raw")
    make("scaled")
