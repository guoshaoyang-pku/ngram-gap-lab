#!/usr/bin/env python3
"""No-gram control vs n-gram multi-epoch (epoch-aligned) training comparison.

Adds the no-gram control curve (nglab1x_v10_nogram_fixed) next to:
  - nglab1x_v10_input_fixed  : same config with n-gram tables on (1x shard, 2000 steps)
  - nglab{0.25..3}x_e6 : multi-epoch n-gram family (epoch-anchored LR, ~6 passes)

Figures:
  1. nogram_vs_ngram_epochaligned.{png,svg} — train / val / gap vs step
  2. gap_vs_passes_nogram.{png,svg}         — per-epoch mean gap vs pass number
Also prints a wall-clock speed table (s/step from train_log.jsonl elapsed_s).

Usage: python3 docs/plot_scripts/gen_nogram_vs_epochaligned_figs.py
"""
import json
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_fixed")
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs", "epoch_scale")
os.makedirs(FIGS_DIR, exist_ok=True)

NOGRAM = {"dir": "nglab1x_v10_nogram_fixed", "color": "#111111", "label": "no n-gram (1x, 2000 steps)"}
INPUT = {"dir": "nglab1x_v10_input_fixed", "color": "#D32F2F", "label": "input + n-gram (1x, 2000 steps)"}
E6_SIZES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
E6_NAMES = {0.25: "nglab0_25x_e6_fixed", 0.5: "nglab0_5x_e6_fixed", 0.75: "nglab0_75x_e6_fixed",
            1.0: "nglab1x_e6_fixed", 1.5: "nglab1_5x_e6_fixed", 2.0: "nglab2x_e6_fixed",
            2.5: "nglab2_5x_e6_fixed", 3.0: "nglab3x_e6_fixed"}
CMAP = plt.get_cmap("viridis")


def load_jsonl(path):
    pts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    pts.append(json.loads(line))
    return pts


def e6_color(s):
    lo, hi = np.log2(0.25), np.log2(3.0)
    return CMAP((np.log2(s) - lo) / (hi - lo))


def per_epoch_mean_gap(log):
    by_ep = defaultdict(list)
    for p in log:
        by_ep[p["epoch"]].append(p["gap"])
    return {e: float(np.mean(v)) for e, v in sorted(by_ep.items())}


def main():
    nogram = load_jsonl(os.path.join(RUNS_DIR, NOGRAM["dir"], "train_log.jsonl"))
    input_ = load_jsonl(os.path.join(RUNS_DIR, INPUT["dir"], "train_log.jsonl"))
    e6 = {}
    for s in E6_SIZES:
        log = load_jsonl(os.path.join(RUNS_DIR, E6_NAMES[s], "train_log.jsonl"))
        if log:
            e6[s] = log
        else:
            print(f"[nogram-vs-e6] WARNING: no train_log for {E6_NAMES[s]} — skipped")

    # ---- wall-clock speed table (same machine: ophis-gpu, parallel v10 batch) ----
    print("=== wall-clock speed (v10, ophis-gpu, parallel batch) ===")
    print(f"{'run':24s} {'steps':>5s} {'elapsed_s':>9s} {'s/step':>7s} {'train@2000':>10s} {'val@2000':>9s} {'gap@2000':>8s}")
    for meta in (NOGRAM, INPUT):
        log = nogram if meta is NOGRAM else input_
        last = log[-1]
        print(f"{meta['dir']:24s} {last['step']:5d} {last['elapsed_s']:9.1f} "
              f"{last['elapsed_s']/last['step']:7.3f} {last['train_loss']:10.4f} "
              f"{last['val_loss']:9.4f} {last['gap']:8.4f}")
    r_nogram = nogram[-1]["elapsed_s"] / nogram[-1]["step"]
    r_input = input_[-1]["elapsed_s"] / input_[-1]["step"]
    print(f"=> n-gram table adds {(r_input/r_nogram - 1)*100:.1f}% wall time per step "
          f"({r_nogram:.3f} -> {r_input:.3f} s/step)")

    # ---- Figure 1: train / val / gap vs step ----
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    titles = ["train loss", "val loss (fixed batches)", "gap = val − train"]
    for ax, key, title in zip(axes, ["train_loss", "val_loss", "gap"], titles):
        # e6 multi-epoch n-gram family (context, thin)
        for s, log in e6.items():
            st = [p["step"] for p in log]
            ax.plot(st, [p[key] for p in log], color=e6_color(s), lw=0.9, alpha=0.75,
                    label=f"n-gram multi-epoch · {s:g}x")
        # n-gram input 1x (same length as nogram)
        st = [p["step"] for p in input_]
        ax.plot(st, [p[key] for p in input_], color=INPUT["color"], lw=1.6, zorder=5,
                label=INPUT["label"])
        # no-gram control
        st = [p["step"] for p in nogram]
        ax.plot(st, [p[key] for p in nogram], color=NOGRAM["color"], lw=2.0,
                ls="--", zorder=6, label=NOGRAM["label"])
        ax.set_ylabel(key)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    axes[2].set_xlabel("step")
    fig.suptitle("No-gram control vs n-gram training (v10, fixed-val, seed 42, input injection)",
                 fontsize=12)
    fig.tight_layout()
    out1 = os.path.join(FIGS_DIR, "nogram_vs_ngram_epochaligned.png")
    out1s = os.path.join(FIGS_DIR, "nogram_vs_ngram_epochaligned.svg")
    fig.savefig(out1, dpi=150)
    fig.savefig(out1s)
    print(f"\n[nogram-vs-e6] wrote {out1}")
    print(f"[nogram-vs-e6] wrote {out1s}")

    # ---- Figure 2: per-epoch mean gap vs pass number ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for s, log in e6.items():
        g = per_epoch_mean_gap(log)
        ax.plot(list(g), list(g.values()), "o-", color=e6_color(s), ms=4, lw=1.1,
                label=f"n-gram multi-epoch · {s:g}x")
    for meta, log, marker in ((INPUT, input_, "o"), (NOGRAM, nogram, "s")):
        g = per_epoch_mean_gap(log)
        ax.plot(list(g), list(g.values()), marker, ls="-", color=meta["color"], ms=5,
                lw=1.6, label=meta["label"])
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("pass number (epoch over the train shard)")
    ax.set_ylabel("mean gap inside epoch (val − train)")
    ax.set_title("Gap vs number of passes: no-gram stays flat, n-gram grows with replay")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out2 = os.path.join(FIGS_DIR, "gap_vs_passes_nogram.png")
    out2s = os.path.join(FIGS_DIR, "gap_vs_passes_nogram.svg")
    fig.savefig(out2, dpi=150)
    fig.savefig(out2s)
    print(f"[nogram-vs-e6] wrote {out2}")
    print(f"[nogram-vs-e6] wrote {out2s}")


if __name__ == "__main__":
    main()
