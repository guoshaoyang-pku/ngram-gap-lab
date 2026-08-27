#!/usr/bin/env python3
"""Live paired schedule diagnostic for the v5 table-LR centre point.

The constant run is intentionally allowed to be partial.  The plot reads only
logged JSONL records and clips the completed warmup control to the largest
step currently present in the constant run.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "runs_fixed"
WARMUP_ID = "optv5c_rms_b099_s2p0_r1"
CONSTANT_ID = "optv5g_rms_b099_s2p0_constant"
OUT = ROOT / "docs" / "figs" / "main" / "fig_v5_warmup_vs_constant_live.png"


def read_jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def matched_rows(rows, last_step):
    return [row for row in rows if row["step"] <= last_step]


def series(rows, key):
    return np.asarray([row[key] for row in rows], dtype=float)


def main():
    warmup = read_jsonl(RUNS / f"{WARMUP_ID}_fixed" / "train_log.jsonl")
    constant = read_jsonl(RUNS / f"{CONSTANT_ID}_fixed" / "train_log.jsonl")
    if not constant:
        raise SystemExit("constant run has no logged points yet")
    last_step = constant[-1]["step"]
    warmup = matched_rows(warmup, last_step)
    if [row["step"] for row in warmup] != [row["step"] for row in constant]:
        raise SystemExit("logged step grids do not match")

    steps = series(constant, "step")
    colors = {"warmup": "#b35c1e", "constant": "#0f766e"}
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True)
    panels = [
        ("train_loss", "online train loss"),
        ("val_loss", "fixed validation loss"),
        ("gap", "online gap = fixed val − online train"),
    ]
    for axis, (key, ylabel) in zip(axes.flat[:3], panels):
        axis.plot(steps, series(warmup, key), "o--", color=colors["warmup"],
                  markersize=3.4, linewidth=1.2, label="warmup_constant(100)")
        axis.plot(steps, series(constant, key), "o-", color=colors["constant"],
                  markersize=3.4, linewidth=1.3, label="constant (zero warmup)")
        if key == "gap":
            axis.axhline(0, color="#686d73", linewidth=0.8, linestyle=":")
        axis.set_title(ylabel)
        axis.set_ylabel("loss" if key != "gap" else "gap")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=False)

    axis = axes.flat[3]
    axis.plot(steps, series(warmup, "lr_mult"), "o--", color=colors["warmup"],
              markersize=3.4, linewidth=1.2, label="warmup_constant(100)")
    axis.plot(steps, series(constant, "lr_mult"), "o-", color=colors["constant"],
              markersize=3.4, linewidth=1.3, label="constant (zero warmup)")
    axis.set_title("applied LR multiplier")
    axis.set_ylabel("multiplier")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False)

    for axis in axes.flat:
        axis.set_xlabel("optimizer step")
        axis.set_xlim(0, last_step + 5)

    final_warmup, final_constant = warmup[-1], constant[-1]
    figure.suptitle(
        "V5 schedule-only live comparison · seed 42 · clean bigram+trigram R=2²⁰ · "
        "RMSProp (0,.99), table LR scale 2 · bf16, no compile\n"
        f"partial snapshot through step {last_step}: "
        f"Δ(constant − warmup) train={final_constant['train_loss'] - final_warmup['train_loss']:+.3f}, "
        f"val={final_constant['val_loss'] - final_warmup['val_loss']:+.3f}, "
        f"gap={final_constant['gap'] - final_warmup['gap']:+.3f}",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT, dpi=190, bbox_inches="tight")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
