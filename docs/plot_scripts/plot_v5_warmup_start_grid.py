#!/usr/bin/env python3
"""Plot the completed v5 warmup-start-multiplier schedule ablation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "runs_fixed"
OUT = ROOT / "docs" / "figs" / "main" / "fig_v5_warmup_start_grid.png"
ARMS = [
    ("0.1×", "optv5h_rms_b099_s2p0_warmstart0p1", "#2563eb", "warmup_constant"),
    ("0.25×", "optv5c_rms_b099_s2p0_r1", "#d97706", "warmup_constant"),
    ("0.5×", "optv5h_rms_b099_s2p0_warmstart0p5_r2", "#7c3aed", "warmup_constant"),
    ("1.0×", "optv5g_rms_b099_s2p0_constant", "#0f766e", "constant"),
]


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    runs = []
    for label, run_id, color, schedule in ARMS:
        run_dir = RUNS / f"{run_id}_fixed"
        summary = read_json(run_dir / "summary.json")
        rows = read_jsonl(run_dir / "train_log.jsonl")
        if len(rows) != 100 or rows[-1]["step"] != 1000:
            raise SystemExit(f"incomplete curve: {run_id}")
        runs.append((label, run_id, color, schedule, summary, rows))

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.4), sharex=True)
    panels = [
        ("train_loss", "online train loss", "loss"),
        ("val_loss", "fixed validation loss", "loss"),
        ("gap", "online gap = fixed val − online train", "gap"),
        ("lr_mult", "applied LR multiplier", "multiplier"),
    ]
    for axis, (key, title, ylabel) in zip(axes.flat, panels):
        for label, _, color, _, _, rows in runs:
            x = np.asarray([row["step"] for row in rows])
            y = np.asarray([row[key] for row in rows])
            axis.plot(x, y, color=color, linewidth=1.15, label=label)
        if key == "gap":
            axis.axhline(0, color="#686d73", linewidth=0.8, linestyle=":")
        if key != "lr_mult":
            for boundary in (337, 674):
                axis.axvline(boundary, color="#9ca3af", linewidth=0.75, linestyle=":", zorder=0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.24)
        axis.legend(title="warmup start", fontsize=8, frameon=False)
        axis.set_xlim(0, 1000)
        axis.set_xlabel("optimizer step")

    finals = " · ".join(
        f"{label}: {summary['final_gap']:.3f}"
        for label, _, _, _, summary, _ in runs
    )
    figure.suptitle(
        "V5 warmup-start-multiplier ablation · seed 42 · 1000 steps\n"
        "Only schedule start differs; all arms share clean R=2²⁰, RMSProp (0,.99), "
        "table LR scale 2, backbone LR 6e-4, bf16, no compile\n"
        f"final online gap (fixed val − current train): {finals}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=190, bbox_inches="tight")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
