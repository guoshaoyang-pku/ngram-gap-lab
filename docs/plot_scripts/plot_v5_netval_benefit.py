#!/usr/bin/env python3
"""Net-benefit verdict figure: val(arm) - val(nogram) over training (§43).

Three panels (injection position input / y / v).  Each panel shows the
matched-step val-loss difference against the nogram control for:
  * control (unconstrained table, solid gray)
  * mask_low f<=8 from step 0 (dashed, arm color)
  * hash reseed every epoch (solid, arm color)
Negative = the arm beats nogram on val (net benefit).  Raw 10-step records
are points; thin lines are 3-point moving averages.

Data: /tmp mirror of runs_fixed (NGLAB_RUNS_FIXED env or data/runs_fixed).
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("NGLAB_RUNS_FIXED", ROOT / "data" / "runs_fixed"))
OUT = ROOT / "docs" / "figs" / "main"

RUNS = {
    "nogram": "nglab1x_nogram_v5_128x_freq10_fixed",
    "input-control": "nglab1x_input_v5_128x_freq10_fixed",
    "y-control": "nglab1x_y_v5_128x_freq10_fixed",
    "v-control": "nglab1x_v_v5_128x_freq10_fixed",
    "input-masklow": "netv5_input_masklowf8_e0_128x_fixed",
    "y-masklow": "netv5_y_masklowf8_e0_128x_fixed",
    "v-masklow": "netv5_v_masklowf8_e0_128x_fixed",
    "input-reseed": "netv5_input_reseed_eall_128x_fixed",
    "y-reseed": "netv5_y_reseed_eall_128x_fixed",
    "v-reseed": "netv5_v_reseed_eall_128x_fixed",
}
POS_COLORS = {"input": "#2d6f9f", "y": "#c4493d", "v": "#c58a0b"}


def load(run_dir):
    out = {}
    with open(MIRROR / run_dir / "train_log.jsonl") as fh:
        for ln in fh:
            e = json.loads(ln)
            out[e["step"]] = (e["train_loss"], e["val_loss"], e["gap"])
    return out


def movavg(y, w=3):
    return [sum(y[max(0, i - w // 2):i + w // 2 + 1]) /
            len(y[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(y))]


def main():
    S.apply_style()
    data = {k: load(v) for k, v in RUNS.items()}
    nog = data["nogram"]
    steps = sorted(nog)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), sharey=True)
    for ax, pos in zip(axes, ("input", "y", "v")):
        c = POS_COLORS[pos]
        for suffix, label, style, color, alpha in [
            ("control", "unconstrained table", "-", "#9aa3ad", 0.9),
            ("masklow", "mask f≤8 from start", "--", c, 0.9),
            ("reseed", "hash reseed every epoch", "-", c, 1.0),
        ]:
            rec = data[f"{pos}-{suffix}"]
            xs = [s for s in steps if s in rec]
            dv = [rec[s][1] - nog[s][1] for s in xs]
            ax.scatter(xs, dv, s=5, color=color, alpha=0.25, zorder=2)
            ax.plot(xs, movavg(dv), style, color=color, lw=1.6, alpha=alpha,
                    zorder=3, label=label)
        for e in range(2, 7):
            ax.axvline(337 * (e - 1), color="#cccccc", lw=0.6, ls=":", zorder=1)
        ax.axhline(0, color="#333", lw=1.0, zorder=2)
        ax.fill_between([0, 2022], [-1.2, -1.2], [0, 0], color="#2a8c62",
                        alpha=0.06, zorder=0)
        ax.set_title(f"{pos} injection")
        ax.set_xlabel("step")
        ax.set_xlim(0, 2022)
        ax.set_ylim(-1.2, 5.2)
    axes[0].set_ylabel("val(arm) − val(nogram), matched step")
    axes[0].text(30, -1.05, "net benefit (beats nogram)", fontsize=8,
                 color="#2a8c62")
    axes[2].legend(fontsize=8, loc="upper left", frameon=False)
    fig.suptitle(
        "Net val benefit of constrained n-gram tables · 128× · 2000 steps (6 epochs) · seed 42\n"
        "points = raw 10-step records; lines = 3-point mean; vertical dotted = epoch boundaries",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png, svg = S.save(fig, OUT, "fig_v5_netval_benefit")
    print(png.relative_to(ROOT))
    print(svg.relative_to(ROOT))


if __name__ == "__main__":
    main()
