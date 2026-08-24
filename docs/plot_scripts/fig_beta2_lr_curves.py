#!/usr/bin/env python3
"""fig_beta2_lr_curves.py — β₂ / 学习率消融的完整曲线簇版

每个 run 画完整的 gap-vs-step 轨迹（每 10 步一点），竖虚线标 epoch 边界。
注意：学习率消融中 **只有 n-gram 表的 LR 被放大**（table_lr_scale），
backbone 的 LR 固定为 0.004。

数据源：data/runs_fixed/*_fixed/train_log.jsonl（post-fix 权威数据）。

输出：
  docs/figs/table_opt/fig_beta2_curves.{svg,png}
  docs/figs/table_opt/fig_lr_curves.{svg,png}
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


def load_log(name: str):
    p = RUNS / name / "train_log.jsonl"
    if not p.exists():
        return None, None, []
    steps, gaps, boundaries = [], [], []
    prev_ep = 0
    for line in p.read_text().splitlines():
        r = json.loads(line)
        steps.append(r["step"])
        gaps.append(r["gap"])
        ep = r.get("epoch", 0)
        if ep != prev_ep:
            if prev_ep > 0:
                boundaries.append(r["step"])
            prev_ep = ep
    return steps, gaps, boundaries


def draw_panel(ax, series: dict, colors: dict, title: str):
    """series: label -> list of run names sharing the same color."""
    drawn_bounds = set()
    final_gaps = {}
    for label, names in series.items():
        for i, n in enumerate(names):
            steps, gaps, bounds = load_log(n)
            if steps is None:
                print(f"missing: {n}")
                continue
            ax.plot(steps, gaps, color=colors[label], lw=1.7,
                    label=label if i == 0 else None)
            ax.plot(steps[-1], gaps[-1], "o", color=colors[label], ms=5)
            final_gaps[label] = gaps[-1]
            for b in bounds:
                if b not in drawn_bounds:
                    drawn_bounds.add(b)
                    ax.axvline(b, color="#999", ls="--", lw=1, alpha=0.7,
                               zorder=0)
    for k, b in enumerate(sorted(drawn_bounds), 1):
        ax.text(b, 0.97, f"epoch {k + 1}", color="#777", fontsize=7.5,
                ha="center", va="top", transform=ax.get_xaxis_transform())
    # β₂ 压差标注：首末两条曲线在终点的差
    labels = list(series.keys())
    if len(labels) >= 2 and labels[0] in final_gaps and labels[-1] in final_gaps:
        spread = final_gaps[labels[0]] - final_gaps[labels[-1]]
        pct = spread / final_gaps[labels[-1]] * 100 if final_gaps[labels[-1]] else 0
        ax.text(0.97, 0.55, f"spread @2000: {spread:+.2f} ({pct:+.0f}%)",
                transform=ax.transAxes, fontsize=9, color="#333",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.35", fc="#fffbe8", ec="#e0d5a0"))
    ax.set_xlabel("training step")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)


# ---------------------------------------------------------------------------
# 图 1：β₂ 曲线簇
# ---------------------------------------------------------------------------
def fig_beta2() -> None:
    c = {"β₂=0.98": "#D65F5F", "β₂=0.99": "#E8A838",
         "β₂=0.999 (default)": "#4878CF", "β₂=0.9999": "#6ACC65",
         "β₂=0.99999": "#9772B5"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150, sharey=True)

    s1 = {"β₂=0.98": ["nglab1x_opt_rmsprop_2x_b2_098_fixed"],
          "β₂=0.99": ["nglab1x_opt_rmsprop_2x_b2_099_fixed"],
          "β₂=0.999 (default)": ["nglab1x_opt_rmsprop_2x_fixed"]}
    s2 = {"β₂=0.98": ["nglab2x_opt_rmsprop_2x_b2_098_fixed"],
          "β₂=0.99": ["nglab2x_opt_rmsprop_2x_b2_099_fixed"],
          "β₂=0.999 (default)": ["nglab2x_opt_rmsprop_2x_fixed"],
          "β₂=0.9999": ["nglab2x_opt_rmsprop_2x_b2_09999_fixed"],
          "β₂=0.99999": ["nglab2x_opt_rmsprop_2x_b2_099999_fixed"]}
    s3 = {"β₂=0.98": ["nglab1x_opt_rmsprop_4x_b2_098_fixed"],
          "β₂=0.99": ["nglab1x_opt_rmsprop_4x_b2_099_fixed"],
          "β₂=0.999 (default)": ["nglab1x_opt_rmsprop_4x_fixed"]}

    draw_panel(axes[0], s1, c, "1x shard · table LR×2")
    draw_panel(axes[1], s2, c, "2-epoch shard · table LR×2")
    draw_panel(axes[2], s3, c, "1x shard · table LR×4")

    axes[0].set_ylabel("gap = val − train")
    for ax in axes:
        ax.set_ylim(bottom=-0.2)

    fig.suptitle(
        "β₂ ablation — full gap trajectories · dashed = epoch boundaries "
        "(backbone LR fixed at 0.004; only the TABLE lr-scale varies between panels)",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_beta2_curves.svg")
    fig.savefig(OUT / "fig_beta2_curves.png")
    plt.close(fig)
    print("saved fig_beta2_curves")


# ---------------------------------------------------------------------------
# 图 2：学习率曲线簇（只有表的 LR 变，backbone 固定）
# ---------------------------------------------------------------------------
def fig_lr() -> None:
    c = {"LR×1": "#4878CF", "LR×2": "#E8A838", "LR×4": "#D65F5F"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, sharey=True)

    s1 = {"LR×1": ["nglab1x_v10_input_fixed"],
          "LR×2": ["nglab1x_opt_rmsprop_2x_fixed"],
          "LR×4": ["nglab1x_opt_rmsprop_4x_fixed"]}
    s2 = {"LR×1": ["nglab2x_input_v10_fv_fixed"],
          "LR×2": ["nglab2x_opt_rmsprop_2x_fixed"],
          "LR×4": ["nglab2x_opt_rmsprop_4x_fixed"]}

    draw_panel(axes[0], s1, c, "1x shard · β₂=0.999")
    draw_panel(axes[1], s2, c, "2-epoch shard · β₂=0.999")

    axes[0].set_ylabel("gap = val − train")
    for ax in axes:
        ax.set_ylim(bottom=-0.2)

    fig.suptitle(
        "Table LR ablation — full gap trajectories · dashed = epoch boundaries\n"
        "ONLY the n-gram table LR is scaled; backbone LR stays at 0.004",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lr_curves.svg")
    fig.savefig(OUT / "fig_lr_curves.png")
    plt.close(fig)
    print("saved fig_lr_curves")


if __name__ == "__main__":
    fig_beta2()
    fig_lr()
