#!/usr/bin/env python3
"""ngram-gap-lab · epoch-length scaling comparison figures (v10 · fixed-val).

Reads data/runs/{nglab0_5x_input_fv_fixed, nglab1x_v10_input_fixed, nglab2x_input_v10_fv_fixed}/
train_log.jsonl and plots train loss / val loss / gap (val - train) over 2000
steps with observed epoch boundaries (from the log's epoch field) marked.

Usage: python3 docs/plot_scripts/gen_epoch_scale_figs.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_fixed")
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs_epoch_scale")
os.makedirs(FIGS_DIR, exist_ok=True)

ARMS = [
    {"key": "0.5x",  "label": "0.5x epoch (shard 60)",   "color": "#9C27B0", "dir": "nglab0_5x_input_fv_fixed"},
    {"key": "1x",    "label": "1x epoch (shard 1)",       "color": "#4CAF50", "dir": "nglab1x_v10_input_fixed"},
    {"key": "2x",    "label": "2x epoch (shards 1+2)",    "color": "#2196F3", "dir": "nglab2x_input_v10_fv_fixed"},
]


def load_jsonl(path):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pts.append(json.loads(line))
    return pts


def epoch_boundaries(train_log):
    """Steps where the epoch field increments (observed epoch boundaries)."""
    bnds = []
    prev = None
    for p in train_log:
        ep = p.get("epoch")
        if ep is not None and prev is not None and ep != prev:
            bnds.append(p["step"])
        prev = ep if ep is not None else prev
    return bnds


def main():
    series = {}
    for arm in ARMS:
        run_dir = os.path.join(RUNS_DIR, arm["dir"])
        train_log = load_jsonl(os.path.join(run_dir, "train_log.jsonl"))
        if not train_log:
            print(f"[epoch-scale] WARNING: no train_log for {arm['dir']} — skipped")
            continue
        steps = [p["step"] for p in train_log]
        series[arm["key"]] = {
            "steps": steps,
            "train": [p["train_loss"] for p in train_log],
            "val": [p["val_loss"] for p in train_log],
            "gap": [p["gap"] for p in train_log],
            "boundaries": epoch_boundaries(train_log),
            "color": arm["color"],
            "label": arm["label"],
        }
        print(f"[epoch-scale] {arm['dir']}: {len(steps)} pts, "
              f"final gap={train_log[-1]['gap']:+.4f}, "
              f"epoch boundaries={series[arm['key']]['boundaries']}")

    if not series:
        raise SystemExit("[epoch-scale] no data yet")

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    titles = ["train loss", "val loss (fixed batches)", "gap = val − train"]
    for ax, key, title in zip(axes, ["train", "val", "gap"], titles):
        for s in series.values():
            ax.plot(s["steps"], s[key], color=s["color"], lw=1.2,
                    label=s["label"])
            for b in s["boundaries"]:
                ax.axvline(b, color=s["color"], ls=":", lw=0.8, alpha=0.5)
        ax.set_ylabel(key)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[2].set_xlabel("step")
    fig.suptitle("Epoch-length scaling (v10, fixed-val, seed 42, input injection, 2000 steps)",
                 fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(FIGS_DIR, "epoch_scale_train_val_gap.png")
    out_svg = os.path.join(FIGS_DIR, "epoch_scale_train_val_gap.svg")
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_svg)
    print(f"[epoch-scale] wrote {out_png}")
    print(f"[epoch-scale] wrote {out_svg}")


if __name__ == "__main__":
    main()
