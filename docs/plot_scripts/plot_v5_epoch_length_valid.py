#!/usr/bin/env python3
"""Plot the provenance-safe epoch-length scan, 0.125xL4 .. 3.95xL4.

<=1xL4 points: nested prefixes of the shard-1 pool (old val set).
>1xL4 points: epfx fix batch with real multi-shard pools (train shards 1,2 /
1,2,3 / 1,2,3,4; val shards 5,6,7,8,9,10,6542), so each epoch is exactly one
full pass over a larger pool -- no wrap-around replay.  The two segments use
different val-set compositions; the join at 1xL4 is marked.

Set NGLAB_RUNS_DIR to read summaries from a mirror instead of ROOT/data.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("NGLAB_RUNS_DIR", ROOT / "data" / "runs_scaling"))
OUTPUT = ROOT / "docs" / "figs" / "main" / "fig_v5_s1_epoch_length_valid.png"

# (epoch length in units of L4=337 batches, tick label, run_id, segment)
RUNS_BY_LENGTH = [
    (0.125, "1/8x", "s1v5_128_ep_tri_0p125xL4_3ep", "prefix"),
    (0.1667, "1/6x", "s1v5_128_ep_tri_0p1667xL4_3ep", "prefix"),
    (0.25, "1/4x", "s1v5_128_ep_tri_0p25xL4_3ep", "prefix"),
    (0.3333, "1/3x", "s1v5_128_ep_tri_0p3333xL4_3ep", "prefix"),
    (0.5, "1/2x", "s1v5_128_ep_tri_0p5xL4_3ep", "prefix"),
    (0.6667, "2/3x", "s1v5_128_ep_tri_0p6667xL4_3ep", "prefix"),
    (0.75, "3/4x", "s1v5_128_ep_tri_0p75xL4_3ep", "prefix"),
    (1.0, "1x", "s1v5_128_ep_tri_1p0xL4_3ep", "prefix"),
    (670 / 337, "1.99x", "s1v5_128_epfx_tri_2p0xL4_3ep", "multishard"),
    (1000 / 337, "2.97x", "s1v5_128_epfx_tri_3p0xL4_3ep", "multishard"),
    (1330 / 337, "3.95x", "s1v5_128_epfx_tri_4p0xL4_3ep", "multishard"),
]


def main():
    points = []
    for length, label, run_id, segment in RUNS_BY_LENGTH:
        path = RUNS / f"{run_id}_fixed" / "summary.json"
        with path.open() as handle:
            summary = json.load(handle)
        points.append((length, label, run_id, float(summary["final_gap"]), segment))

    x = [p[0] for p in points]
    gap = [p[3] for p in points]
    fig, axis = plt.subplots(figsize=(9.2, 5.0))
    axis.plot(x, gap, color="#0f766e", marker="o", linewidth=1.3, zorder=3)
    for xi, gi, p in zip(x, gap, points):
        axis.annotate(f"{gi:.2f}", (xi, gi), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=8, color="#0f766e")
    axis.axvline(1.0, color="#c4493d", lw=1.0, ls="--", alpha=0.7)
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, [p[1] for p in points], rotation=35, ha="right")
    axis.set_xlabel("unique training-pool length per epoch, relative to L4 (L4 = 337 device batches)")
    axis.set_ylabel("online gap after 3 complete passes")
    axis.set_title("V5 S1 epoch-length scan, fixed to 3.95xL4 with real multi-shard pools")
    axis.grid(alpha=0.24)
    axis.text(
        0.02,
        0.04,
        "Each point: trigram-only - clean R=2^20 - table LR=128x - seed 42 - 3 passes.\n"
        "Left of dashed line: shard-1 nested prefixes (old val set); right: epfx multi-shard\n"
        "pools (val 5,6,7,8,9,10,6542). Old >1xL4 wrap-around 'U-shape' retired as artifact.",
        transform=axis.transAxes,
        fontsize=8.5,
        color="#475569",
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
