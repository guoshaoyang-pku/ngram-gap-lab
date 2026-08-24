#!/usr/bin/env python3
"""fig_val_loss_debug.py — val loss 曲线簇：检查表学习率是否过大

三张图：
  1. β₂=0.99 · 1x shard · LR×2/×4（对比默认 0.999 · LR×1）
  2. β₂=0.99 · 2-epoch shard · LR×2/×4（对比默认 0.999 · LR×1）
  3. 短 epoch 家族（0.25x / 0.5x · β₂=0.99 · LR×1）

输出：
  docs/figs/table_opt/fig_val_loss_beta2_lr.{svg,png}
  docs/figs/table_opt/fig_val_loss_short_epoch.{svg,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data" / "runs_fixed"
OUT = REPO / "docs" / "figs" / "table_opt"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str, field: str = "val_loss"):
    p = RUNS / name / "train_log.jsonl"
    if not p.exists():
        return None, None, []
    steps, vals, bounds = [], [], []
    prev_ep = 0
    for line in p.read_text().splitlines():
        r = json.loads(line)
        steps.append(r["step"])
        vals.append(r.get(field, float("nan")))
        ep = r.get("epoch", 0)
        if ep != prev_ep:
            if prev_ep > 0:
                bounds.append(r["step"])
            prev_ep = ep
    return steps, vals, bounds


def panel(ax, series: dict, colors: dict, title: str, ylabel: str = "val loss",
          field: str = "val_loss"):
    for label, names in series.items():
        for i, n in enumerate(names):
            steps, vals, bounds = load(n, field)
            if steps is None:
                print(f"  missing: {n}")
                continue
            ax.plot(steps, vals, color=colors[label], lw=1.5,
                    label=label if i == 0 else None)
            ax.plot(steps[-1], vals[-1], "o", color=colors[label], ms=4)
            for b in bounds:
                ax.axvline(b, color="#bbb", ls=":", lw=0.8, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", framealpha=0.9, fontsize=8)


# ---------------------------------------------------------------------------
# 图 1：β₂=0.99 · 1x shard · LR×2/×4 + 默认 0.999 · LR×1 对照
# ---------------------------------------------------------------------------
def fig_beta2_lr_val() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150)

    c1 = {"LR×1 · β₂=0.999 (default)": "#4878CF",
          "LR×2 · β₂=0.99": "#E8A838",
          "LR×4 · β₂=0.99": "#D65F5F"}
    s1 = {"LR×1 · β₂=0.999 (default)": ["nglab1x_v10_input_fixed"],
          "LR×2 · β₂=0.99": ["nglab1x_opt_rmsprop_2x_b2_099_fixed"],
          "LR×4 · β₂=0.99": ["nglab1x_opt_rmsprop_4x_b2_099_fixed"]}

    c2 = {"LR×1 · β₂=0.999 (default)": "#4878CF",
          "LR×2 · β₂=0.99": "#E8A838",
          "LR×4 · β₂=0.99": "#D65F5F"}
    s2 = {"LR×1 · β₂=0.999 (default)": ["nglab2x_input_v10_fv_fixed"],
          "LR×2 · β₂=0.99": ["nglab2x_opt_rmsprop_2x_b2_099_fixed"],
          "LR×4 · β₂=0.99": ["nglab2x_opt_rmsprop_4x_b2_099_fixed"]}

    panel(axes[0], s1, c1, "1x shard · val loss\n(blue = LR×1 β₂=0.999)")
    panel(axes[1], s2, c2, "2-epoch shard · val loss\n(blue = LR×1 β₂=0.999)")

    fig.suptitle(
        "Val loss under β₂=0.99 · high table LR — is the LR too aggressive?",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_val_loss_beta2_lr.svg")
    fig.savefig(OUT / "fig_val_loss_beta2_lr.png")
    plt.close(fig)
    print("saved fig_val_loss_beta2_lr")


# ---------------------------------------------------------------------------
# 图 2：短 epoch 家族（0.25x / 0.5x · β₂=0.99 · LR×1）
# ---------------------------------------------------------------------------
def fig_short_epoch_val() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)

    c = {"0.25x shard · β₂=0.99 · LR×1": "#D65F5F",
         "0.5x shard · β₂=0.99 · LR×1": "#E8A838",
         "0.25x shard · β₂=0.999 · LR×1": "#B4A7D6",
         "0.5x shard · β₂=0.999 · LR×1": "#86C5AC"}
    s = {"0.25x shard · β₂=0.99 · LR×1": ["nglab025x_b2_099_fixed"],
         "0.5x shard · β₂=0.99 · LR×1": ["nglab05x_b2_099_fixed"],
         "0.25x shard · β₂=0.999 · LR×1": ["nglab0_25x_input_fv_fixed"],
         "0.5x shard · β₂=0.999 · LR×1": ["nglab0_5x_input_fv_fixed"]}

    panel(ax, s, c, "Short-epoch family · val loss\n(0.25x & 0.5x shard, table LR×1)")

    fig.suptitle("Short-epoch shard · val loss (post-fix _fixed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_val_loss_short_epoch.svg")
    fig.savefig(OUT / "fig_val_loss_short_epoch.png")
    plt.close(fig)
    print("saved fig_val_loss_short_epoch")


if __name__ == "__main__":
    fig_beta2_lr_val()
    fig_short_epoch_val()
