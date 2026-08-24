#!/usr/bin/env python3
"""ngram-gap-lab · row-level pilot figure generator.

Reads the CSV produced by row_level_gap.py for a set of runs and draws:

  panel A: per-row scatter of row-level gap vs distinct contexts hashed to
           the row (train probe), one color per run, log-x, with per-bin
           weighted mean curves (token-weighted) overlaid.
  panel B: token-weighted mean gap in distinct-context bins, one line per run,
           to directly compare "gap vs collision load" across table sizes.

Usage:
  python3 plot_row_level_pilot.py <out.png> [--csv name=path ...]
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        out.append({
            "ctx": int(r["distinct_contexts_train"]),
            "gap": float(r["gap"]),
            "n": int(r["train_tokens"]),
        })
    return out


def bin_stats(rows, edges):
    """Token-weighted mean gap + token count per context bin."""
    xs, ys, ws = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r["ctx"] < hi]
        if not sel:
            continue
        tot = sum(r["n"] for r in sel)
        wg = sum(r["gap"] * r["n"] for r in sel) / tot
        xs.append(np.sqrt(lo * hi))  # geometric midpoint
        ys.append(wg)
        ws.append(tot)
    return np.array(xs), np.array(ys), np.array(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--csv", action="append", default=[],
                    help="name=path")
    ap.add_argument("--edges", default="1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,4096")
    args = ap.parse_args()

    runs = []
    for item in args.csv:
        name, path = item.split("=", 1)
        runs.append((name, load_csv(path)))
    edges = [int(x) for x in args.edges.split(",")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colors = {"tbl_64": "#2d6f9f", "tbl_1": "#c4493d", "bigram": "#2d6f9f",
              "trigram": "#c4493d"}

    # Panel A: scatter + weighted binned curves
    ax = axes[0]
    for name, rows in runs:
        c = colors.get(name, "#353d79")
        xs = [r["ctx"] for r in rows]
        ys = [r["gap"] for r in rows]
        ax.scatter(xs, ys, s=6, alpha=0.25, color=c, label=f"{name} (n={len(rows)})")
        bx, by, bw = bin_stats(rows, edges)
        ax.plot(bx, by, "-o", color=c, lw=2, markersize=4, alpha=1.0)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("distinct contexts hashed to row (train probe)")
    ax.set_ylabel("row-level gap (val_mean − train_mean)")
    ax.set_title("Row-level gap vs collision load (scatter + token-weighted bins)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    # Panel B: weighted mean gap per bin, both runs on same axes
    ax = axes[1]
    for name, rows in runs:
        c = colors.get(name, "#353d79")
        bx, by, bw = bin_stats(rows, edges)
        ax.plot(bx, by, "-o", color=c, lw=2.2, markersize=5, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("distinct contexts hashed to row (train probe)")
    ax.set_ylabel("token-weighted mean row gap")
    ax.set_title("Weighted gap vs distinct-context load (binned)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Pilot · row-level gap recovered from final model "
                 "(fixed train/val probe, layer 0, hash 0/1)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
