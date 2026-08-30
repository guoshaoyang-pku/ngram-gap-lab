#!/usr/bin/env python3
"""Plot the provenance-safe <=1xL4 epoch-length prefix scan.

The >1xL4 runs reuse shard 1 through wrap-around and therefore measure replay
passes rather than a longer dataset.  They are intentionally excluded here.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "runs_scaling"
OUTPUT = ROOT / "docs" / "figs" / "main" / "fig_v5_s1_epoch_length_valid.png"

RUNS_BY_LENGTH = [
    (0.125, "⅛×", "s1v5_128_ep_tri_0p125xL4_3ep"),
    (0.1667, "⅙×", "s1v5_128_ep_tri_0p1667xL4_3ep"),
    (0.25, "¼×", "s1v5_128_ep_tri_0p25xL4_3ep"),
    (0.3333, "⅓×", "s1v5_128_ep_tri_0p3333xL4_3ep"),
    (0.5, "½×", "s1v5_128_ep_tri_0p5xL4_3ep"),
    (0.6667, "⅔×", "s1v5_128_ep_tri_0p6667xL4_3ep"),
    (0.75, "¾×", "s1v5_128_ep_tri_0p75xL4_3ep"),
    (1.0, "1×", "s1v5_128_ep_tri_1p0xL4_3ep"),
]


def main():
    points = []
    for length, label, run_id in RUNS_BY_LENGTH:
        path = RUNS / f"{run_id}_fixed" / "summary.json"
        with path.open() as handle:
            summary = json.load(handle)
        points.append((length, label, run_id, float(summary["final_gap"])))

    x = [point[0] for point in points]
    gap = [point[3] for point in points]
    fig, axis = plt.subplots(figsize=(8.6, 5.0))
    axis.plot(x, gap, color="#0f766e", marker="o", linewidth=1.3)
    axis.set_xticks(x, [point[1] for point in points])
    axis.set_xlabel("unique training-prefix length relative to L4 (L4 = 337 device batches)")
    axis.set_ylabel("online gap after 3 complete passes")
    axis.set_title("V5 S1 valid epoch-length scan · nested prefixes only (≤1×L4)")
    axis.grid(alpha=0.24)
    axis.text(
        0.02,
        0.04,
        "Each point: trigram-only · clean R=2²⁰ · table LR=128× · seed 42\n"
        ">1×L4 wrap-around runs excluded: they vary replay passes, not dataset length.",
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
