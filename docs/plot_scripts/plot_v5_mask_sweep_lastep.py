#!/usr/bin/env python3
"""§44 verdict figure: branch-separated mask sweep + last-epoch-only readout.

Left panel: trigram-only table, mask_low f<=t from epoch 2 (inclusive).
Step-1000 gap vs threshold t; annotations give the dynamic removal rate
against the unmasked trigram-only control.  The gray marker at t=8 is the
static-attribution expectation (79.3% of trigram gap attributed to f<=8
from the dual-table exact-f decomposition), so the gap between the gray
marker and the measured point visualises dynamic rebalancing.

Right panel: readout_last_epoch (readout fully masked for epochs 1-5,
released at the epoch-6 boundary step 1680).  Matched-step
val(arm) - val(nogram) for the three injection positions; reseed-every-epoch
endpoints (§43) are shown as right-edge ticks for comparison.
Negative = beats nogram on val.  Raw 10-step records are points; thin
lines are 3-point moving averages.

Data: /tmp mirror of runs_fixed (NGLAB_RUNS_FIXED env or data/runs_fixed).
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("NGLAB_RUNS_FIXED", ROOT / "data" / "runs_fixed"))
OUT = ROOT / "docs" / "figs" / "main"

TRI_RUNS = [("ctl", "s1v5_128_tri_masklow_ctl_fixed", "control"),
            ("1", "s1v5_128_tri_masklowf1_e1_fixed", "t=1"),
            ("2", "s1v5_128_tri_masklowf2_e1_fixed", "t=2"),
            ("4", "s1v5_128_tri_masklowf4_e1_fixed", "t=4"),
            ("8", "s1v5_128_tri_masklowf8_e1_fixed", "t=8")]
LASTEP = {"input": "netv5_input_lastep_128x_fixed",
          "y": "netv5_y_lastep_128x_fixed",
          "v": "netv5_v_lastep_128x_fixed"}
NOGRAM = "nglab1x_nogram_v5_128x_freq10_fixed"
RESEED_END = {"input": 0.807, "y": -0.108, "v": -0.131}  # §43.4 step-2000 Δval
STATIC_TRI_F8 = 0.793  # static attribution: share of trigram gap from f<=8
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    # ---- left: trigram-only dynamic mask sweep ---------------------------
    gaps, labels = [], []
    ctl_gap = None
    for key, run, lab in TRI_RUNS:
        rec = load(run)
        g = rec[max(rec)][2]
        if key == "ctl":
            ctl_gap = g
        gaps.append(g)
        labels.append(lab)
    xs = list(range(len(gaps)))
    ax1.scatter(xs, gaps, s=42, color="#c4493d", zorder=3)
    ax1.plot(xs, gaps, "-", color="#c4493d", lw=1.4, alpha=0.7, zorder=2)
    for x, g in zip(xs, gaps):
        if x == 0:
            ax1.annotate(f"{g:.3f}", (x, g), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
        else:
            ax1.annotate(f"{g:.3f}\n(−{1 - g / ctl_gap:.0%})", (x, g),
                         textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=8)
    static_exp = ctl_gap * (1 - STATIC_TRI_F8)
    ax1.scatter([xs[-1]], [static_exp], marker="s", s=46, color="#9aa3ad",
                zorder=3, label=f"static-attribution expectation @f<=8 "
                                f"(-{STATIC_TRI_F8:.0%}, i.e. {static_exp:.2f})")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("mask threshold t (mask f ≤ t, from epoch 2)")
    ax1.set_ylabel("step-1000 gap")
    ax1.set_ylim(0, 3.0)
    ax1.set_title("trigram-only table · dynamic mask sweep")
    ax1.legend(fontsize=8, loc="upper right", frameon=False)

    # ---- right: last-epoch-only readout ----------------------------------
    nog = load(NOGRAM)
    for pos, run in LASTEP.items():
        rec = load(run)
        xs2 = sorted(rec)
        dv = [rec[s][1] - nog[s][1] for s in xs2]
        c = POS_COLORS[pos]
        ax2.scatter(xs2, dv, s=5, color=c, alpha=0.25, zorder=2)
        ax2.plot(xs2, movavg(dv), "-", color=c, lw=1.6, zorder=3,
                 label=f"{pos} (end {dv[-1]:+.2f})")
        ax2.plot([2030, 2050], [RESEED_END[pos]] * 2, color=c, lw=2.4,
                 solid_capstyle="butt", zorder=3)
    ax2.text(2055, RESEED_END["input"], "reseed endpoints (§43)",
             fontsize=8, color="#666", va="center")
    for e in range(2, 7):
        ax2.axvline(337 * (e - 1), color="#cccccc", lw=0.6, ls=":", zorder=1)
    ax2.axvline(1680, color="#333", lw=1.0, ls="--", zorder=1)
    ax2.text(1665, 2.85, "readout released (epoch 6) ", fontsize=8, ha="right")
    ax2.axhline(0, color="#333", lw=1.0, zorder=2)
    ax2.set_xlim(0, 2120)
    ax2.set_ylim(-0.6, 3.3)
    ax2.set_xlabel("step")
    ax2.set_ylabel("val(arm) − val(nogram), matched step")
    ax2.set_title("readout only in the last epoch (epochs 1–5 masked)")
    ax2.legend(fontsize=8, loc="upper left", frameon=False)

    fig.suptitle(
        "§44 · branch-separated mask sweep and last-epoch-only readout · "
        "128× · seed 42\npoints = raw 10-step records; lines = 3-point mean",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    png, svg = S.save(fig, OUT, "fig_v5_mask_sweep_lastep")
    print(png.relative_to(ROOT))
    print(svg.relative_to(ROOT))


if __name__ == "__main__":
    main()
