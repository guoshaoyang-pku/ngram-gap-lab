#!/usr/bin/env python3
"""ngram-gap-lab · row-level multi-table-size figure generator.

Reads the CSVs produced by row_level_gap.py for a set of table sizes and draws:

  panel A: per-row scatter of row-level gap vs distinct contexts hashed to
           the row (train probe), one color per table size, log-x, with
           token-weighted binned-mean curves overlaid.
  panel B: token-weighted mean gap in distinct-context bins, one line per
           table size (collision-load view).
  panel C: token-weighted mean row gap vs table size (2R logical addresses) --
           the macroscopic "does table size drive gap" curve.

Usage:
  python3 plot_row_level_multi.py <out.png> \
      --csv tbl_64=path --csv tbl_32=path ... [--table_sizes 64,32,...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# table size -> 2R logical addresses (vocab 8192 * table_mult * 2 hashes)
LOGICAL = {1: 16384, 2: 32768, 4: 65536, 8: 131072, 16: 262144,
           32: 524288, 64: 1048576}

CMAP = {64: "#1a3a5c", 32: "#2d6f9f", 16: "#4d9fc7", 8: "#7fc4d9",
        4: "#b9d98c", 2: "#e0b04c", 1: "#c4493d"}


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    out = []
    for r in rows:
        out.append({
            "ctx": int(r["distinct_contexts_train"]),
            "hits": int(r["train_tokens"]),
            "gap": float(r["gap"]),
            "n": int(r["train_tokens"]),
        })
    return out


def bin_stats(rows, edges, xkey="ctx"):
    """Token-weighted mean gap + token count per context bin."""
    xs, ys, ws = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [r for r in rows if lo <= r[xkey] < hi]
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
                    help="name=path  (name should be tbl_<mult>)")
    ap.add_argument("--summary", action="append", default=[],
                    help="name=path to formal summary.json (optional)")
    ap.add_argument("--edges",
                    default="1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,4096")
    ap.add_argument("--table_sizes", default="64,32,16,8,4,2,1",
                    help="comma-separated table_mult values (plot order)")
    ap.add_argument("--xfield", choices=("contexts", "hits"),
                    default="contexts",
                    help="row-load variable for panels A/B")
    args = ap.parse_args()

    runs = {}
    for item in args.csv:
        name, path = item.split("=", 1)
        tm = int(name.replace("tbl_", "").replace("_bigram", ""))
        runs[tm] = load_csv(path)
    fixed_gaps = {}
    for item in args.summary:
        name, path = item.split("=", 1)
        tm = int(name.replace("tbl_", "").replace("_bigram", ""))
        with open(path) as f:
            fixed_gaps[tm] = float(json.load(f)["final_fixed_gap"])
    sizes = [int(x) for x in args.table_sizes.split(",") if int(x) in runs]
    edges = [int(x) for x in args.edges.split(",")]
    xkey = "ctx" if args.xfield == "contexts" else "hits"
    x_label = (
        "distinct contexts hashed to row (train probe)"
        if xkey == "ctx" else
        "train-probe token hits for row"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    # Panel A: scatter + weighted binned curves (one color per table size)
    ax = axes[0]
    for tm in sizes:
        rows = runs[tm]
        c = CMAP[tm]
        xs = [r[xkey] for r in rows]
        ys = [r["gap"] for r in rows]
        # The per-row scatter is useful for distribution shape, but plotting
        # all rows for seven sizes obscures the binned curves.  Keep the
        # largest and smallest tables as representative point clouds.
        if tm in (64, 1):
            ax.scatter(xs, ys, s=5, alpha=0.18, color=c,
                       label=f"tbl_{tm} (n={len(rows):,})")
        bx, by, bw = bin_stats(rows, edges, xkey)
        ax.plot(bx, by, "-o", color=c, lw=2, markersize=4,
                label=f"tbl_{tm} bins")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(0.7, 5000)
    ax.set_xlabel(x_label)
    ax.set_ylabel("row-level gap (val_mean − train_mean)")
    ax.set_title("Row-level gap vs collision load\n(scatter + token-weighted bins)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3, which="both")

    # Panel B: weighted mean gap per bin (collision-load view)
    ax = axes[1]
    for tm in sizes:
        bx, by, bw = bin_stats(runs[tm], edges, xkey)
        ax.plot(bx, by, "-o", color=CMAP[tm], lw=2.2, markersize=5, label=f"tbl_{tm}")
    ax.set_xscale("log")
    ax.set_xlim(0.7, 5000)
    ax.set_xlabel(x_label)
    ax.set_ylabel("token-weighted mean row gap")
    ax.set_title(
        "Weighted gap vs distinct-context load (binned)"
        if xkey == "ctx" else
        "Weighted gap vs row token hits (binned)"
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    # Panel C: macroscopic gap vs table size (2R logical addresses)
    ax = axes[2]
    xs = [LOGICAL[tm] for tm in sizes]
    ys = []
    for tm in sizes:
        rows = runs[tm]
        tot = sum(r["n"] for r in rows)
        wg = sum(r["gap"] * r["n"] for r in rows) / tot
        ys.append(wg)
    ax.plot(xs, ys, "-o", color="#2d6f9f", lw=2.4, markersize=6,
            label="row-level overlap mean")
    for x, y, tm in zip(xs, ys, sizes):
        ax.annotate(f"tbl_{tm}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    fixed_sizes = sorted(fixed_gaps, reverse=True)
    if fixed_sizes:
        fixed_xs = [LOGICAL[tm] for tm in fixed_sizes]
        fixed_ys = [fixed_gaps[tm] for tm in fixed_sizes]
        ax.plot(fixed_xs, fixed_ys, "--s", color="#555555", lw=1.8,
                markersize=4, label="fixed-probe gap (diagnostic)")
    ax.set_xscale("log")
    ax.set_xlabel("table size (logical addresses 2R)")
    ax.set_ylabel("gap")
    ax.set_title("Gap vs table size")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Row-level gap recovered from final model · bigram, layer 0, "
        f"hash 0/1 · fixed train/val probe · seed 42 · x={args.xfield}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
