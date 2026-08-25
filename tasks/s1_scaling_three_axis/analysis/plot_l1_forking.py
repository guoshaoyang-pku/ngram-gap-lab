#!/usr/bin/env python3
"""Forking 对比：collision-free (perfect map) vs 单层碰撞对照。

左面板: train/val loss 曲线（freq=50）；右面板: gap(step)。
epoch 边界（337 / 674 batches，L4=337）用竖虚线标出；注释边界两侧
（step 300->350, 650->700）的 train-loss 下跳幅度，作为 forking 强度度量。

输出: docs/appendices/s1_scaling_three_axis/figs/fig_l1_forking.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "docs/appendices/s1_scaling_three_axis/figs/fig_l1_forking.png")
EPOCH = 337
RUNS = [
    ("tbl_perfect_bigram_l1_fixed", "collision-free (perfect map)", "#C44E52"),
    ("tbl_64_bigram_l1_fixed", "single-layer control (mult=64, collision)", "#55A868"),
]


def load(run_id):
    rows = [json.loads(l) for l in open(
        os.path.join(ROOT, "data/runs_scaling", run_id, "train_log.jsonl"))]
    return ([r["step"] for r in rows], [r["train_loss"] for r in rows],
            [r["val_loss"] for r in rows], [r["gap"] for r in rows])


def boundary_jumps(steps, losses):
    """train-loss drop across each epoch boundary (nearest logged steps)."""
    jumps = []
    for b in (EPOCH, 2 * EPOCH):
        pre = max((s, l) for s, l in zip(steps, losses) if s < b)
        post = min((s, l) for s, l in zip(steps, losses) if s >= b)
        jumps.append((b, pre, post, post[1] - pre[1]))
    return jumps


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    for run_id, label, color in RUNS:
        steps, tr, va, gap = load(run_id)
        ax1.plot(steps, tr, color=color, lw=1.4, label=f"{label} (train)")
        ax1.plot(steps, va, color=color, lw=1.4, ls="--", label=f"{label} (val)")
        ax2.plot(steps, gap, color=color, lw=1.6, label=label)
        for b, pre, post, d in boundary_jumps(steps, tr):
            print(f"{run_id}: boundary@{b} train {pre[1]:.4f}->{post[1]:.4f} (d={d:+.4f})")
    for ax in (ax1, ax2):
        for b in (EPOCH, 2 * EPOCH):
            ax.axvline(b, color="gray", ls=":", lw=1)
        ax.grid(alpha=0.25)
        ax.set_xlabel("step (epoch boundary = 337 / 674)")
    ax1.set_ylabel("loss")
    ax1.set_title("train / val loss")
    ax1.legend(fontsize=7.5, loc="upper right")
    ax2.set_ylabel("online gap (val - train)")
    ax2.set_title("gap trajectory")
    ax2.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("forking under zero-collision vs collision (single-layer bigram, L4 epoch)")
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
