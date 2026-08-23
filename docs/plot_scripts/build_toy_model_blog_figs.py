#!/usr/bin/env python3
"""Blog figures for the 'toy model (小世界)' chapter 8.2 (Chinese labels).

Reads docs/figs/synth_{A,B}_summary.json (synthetic transition pilot, order=5,
clean NanoGPTOriginal line).  The old v5 2x2 (current-shell) figure was removed
with the deprecated current-shell toy experiments.

Writes to docs/figs/:
  - fig_toy_synth_gap_vs_freq.svg
  - fig_toy_synth_gap_vs_step.svg
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[1]
FIGS = REPO / "docs" / "figs"

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "PingFang HK",
                                   "Arial Unicode MS", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False

STYLE = {
    "synth_A_sparse_restart_s42":        ("A 私密+全局 · 带表", "#1f77b4", "-o"),
    "synth_A_sparse_restart_nogram_s42": ("A 私密+全局 · 不带表", "#1f77b4", "--s"),
    "synth_B_lowrank_sparse_s42":        ("B 共享主题 · 带表", "#d62728", "-o"),
    "synth_B_lowrank_sparse_nogram_s42": ("B 共享主题 · 不带表", "#d62728", "--s"),
}


def synth_pair(fig_name: str, per_frequency: bool) -> None:
    data = [
        json.loads((FIGS / "synth_A_summary.json").read_text()),
        json.loads((FIGS / "synth_B_summary.json").read_text()),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for ds in data:
        for res in ds["runs"]:
            label, color, m = STYLE[res["run"]]
            if per_frequency:
                xs = [pf["frequency"] for pf in res["per_frequency"]]
                ys = [pf["excess_ce"] for pf in res["per_frequency"]]
            else:
                xs = [s["step"] for s in res["steps"]]
                ys = [s["excess_ce"] for s in res["steps"]]
            ax.plot(xs, ys, m, color=color, label=label, ms=5, lw=1.6)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    if per_frequency:
        ax.set_xscale("log")
        ax.set_xticks([8, 16, 32, 64, 128, 512, 2048, 8192])
        ax.set_xticklabels(["8", "16", "32", "64", "128", "512", "2k", "8k"])
        ax.set_xlabel("上下文在训练里出现的次数（对数）")
        ax.set_title("第二步：越少见的上下文，不带表时差得越多（step 2000）")
    else:
        ax.set_xlabel("训练步数")
        ax.set_title("训练过程中，总体差距的变化（离理论最好水平多远）")
    ax.set_ylabel("比理论最好水平多出的损失")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / fig_name)
    plt.close(fig)


if __name__ == "__main__":
    synth_pair("fig_toy_synth_gap_vs_freq.svg", per_frequency=True)
    synth_pair("fig_toy_synth_gap_vs_step.svg", per_frequency=False)
    print("wrote", sorted(p.name for p in FIGS.glob("fig_toy_*.svg")))
