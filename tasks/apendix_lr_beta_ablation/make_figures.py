#!/usr/bin/env python3
"""make_figures.py — 表学习率 × β₂ 消融附录的全部图表（v2：一图一变量）

设计原则（用户要求）：
1. 每张图只变一个变量（其余全部固定），变量写在标题里。
2. 每张图三个面板：train loss / val loss / gap —— 用户要看到训练崩坏的全貌。
3. 数据来自 results/appendix_data.json（extract_data.py 生成）；缺失的补点自动跳过。

图片清单（figs/）：
  A. 表学习率扫描（β₂ 固定，变表学习率）：
     fig_lr_sweep_b2_099_1x.svg      1x shard · β₂=0.99 · LR×1/×2/×4
     fig_lr_sweep_b2_099_2ep.svg     2-epoch shard · β₂=0.99 · LR×1/×2/×4
     fig_lr_sweep_b2_0999_1x.svg     1x shard · β₂=0.999 · LR×1/×2/×4
     fig_lr_sweep_b2_0999_2ep.svg    2-epoch shard · β₂=0.999 · LR×1/×2/×4
  B. β₂ 扫描（表学习率固定，变 β₂）：
     fig_b2_sweep_1x_lr2.svg         1x shard · LR×2 · β₂=0.98/0.99/0.999
     fig_b2_sweep_2ep_lr2.svg        2-epoch shard · LR×2 · β₂=0.98…0.99999
     fig_b2_sweep_1x_lr4.svg         1x shard · LR×4 · β₂=0.98/0.99/0.999
     fig_b2_sweep_2ep_lr4.svg        2-epoch shard · LR×4 · β₂=0.98…0.9999
  C. 交互与汇总：
     fig_beta2_spread_vs_lr.svg      β₂ 压差（0.98−0.999）随表学习率衰减
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "results" / "appendix_data.json").read_text())
OUT = HERE / "figs"
OUT.mkdir(parents=True, exist_ok=True)

B2_COLORS = {"0.98": "#D65F5F", "0.99": "#E8A838", "0.999": "#4878CF",
             "0.9999": "#6ACC65", "0.99999": "#9772B5"}
LR_COLORS = {"×1 (0.004)": "#2C9C5A", "×2 (0.008)": "#E8A838", "×4 (0.016)": "#D65F5F"}


def traj(key):
    rec = DATA.get(key)
    return rec["traj"] if rec else None


def sweep_figure(filename, title, fixed_desc, keys_colors, xticklabels=None):
    """一图一变量：三个面板 = train loss / val loss / gap。

    keys_colors: [(key, label, color), ...] —— 每条曲线只变一个维度。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), dpi=150)
    panels = [("train_loss", "train loss"), ("val_loss", "val loss"),
              ("gap", "gap = val − train")]
    for ax, (field, ylabel) in zip(axes, panels):
        bset = set()
        for key, label, color in keys_colors:
            t = traj(key)
            if t is None:
                print(f"  skip (missing): {key}")
                continue
            ax.plot(t["step"], t[field], color=color, lw=1.7, label=label)
            ax.plot(t["step"][-1], t[field][-1], "o", color=color, ms=5)
            bset.update(t["epoch_bounds"])
        for b in sorted(bset):
            ax.axvline(b, color="#999", ls="--", lw=1, alpha=0.55)
        for k, b in enumerate(sorted(bset), 2):
            ax.text(b, 0.96, f"epoch {k}", color="#777", fontsize=7.5,
                    ha="center", va="top",
                    transform=ax.get_xaxis_transform())
        ax.set_title(ylabel, fontsize=11)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.9)
        if field == "gap":
            ax.axhline(0, color="#aaa", lw=0.8)
    fig.suptitle(f"{title}\n(fixed: {fixed_desc})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)
    print(filename)


# ------------------------------------------------------------------ A. LR 扫描
def lr_sweeps():
    sweep_figure(
        "fig_lr_sweep_b2_099_1x.svg",
        "Table LR sweep · 1x shard · β₂=0.99 (momentum-free)",
        "seed 42 · 2000 steps · injection=input · β₂=0.99 · backbone LR 0.004",
        [("b2_099_1x_lr1", "table LR×1 (0.004) [new]", LR_COLORS["×1 (0.004)"]),
         ("b2_099_1x_lr2", "table LR×2 (0.008)", LR_COLORS["×2 (0.008)"]),
         ("b2_099_1x_lr4", "table LR×4 (0.016)", LR_COLORS["×4 (0.016)"])])
    sweep_figure(
        "fig_lr_sweep_b2_099_2ep.svg",
        "Table LR sweep · 2-epoch shard · β₂=0.99 (momentum-free)",
        "seed 42 · 2000 steps · injection=input · β₂=0.99 · backbone LR 0.004",
        [("b2_099_2ep_lr1", "table LR×1 (0.004) [new]", LR_COLORS["×1 (0.004)"]),
         ("b2_099_2ep_lr2", "table LR×2 (0.008)", LR_COLORS["×2 (0.008)"]),
         ("b2_099_2ep_lr4", "table LR×4 (0.016)", LR_COLORS["×4 (0.016)"])])
    sweep_figure(
        "fig_lr_sweep_b2_0999_1x.svg",
        "Table LR sweep · 1x shard · β₂=0.999 (default)",
        "seed 42 · 2000 steps · injection=input · β₂=0.999 · backbone LR 0.004",
        [("lr1_1x", "table LR×1 (0.004)", LR_COLORS["×1 (0.004)"]),
         ("b2_0999_1x_lr2", "table LR×2 (0.008)", LR_COLORS["×2 (0.008)"]),
         ("b2_0999_1x_lr4", "table LR×4 (0.016)", LR_COLORS["×4 (0.016)"])])
    sweep_figure(
        "fig_lr_sweep_b2_0999_2ep.svg",
        "Table LR sweep · 2-epoch shard · β₂=0.999 (default)",
        "seed 42 · 2000 steps · injection=input · β₂=0.999 · backbone LR 0.004",
        [("lr1_2ep", "table LR×1 (0.004)", LR_COLORS["×1 (0.004)"]),
         ("b2_0999_2ep_lr2", "table LR×2 (0.008)", LR_COLORS["×2 (0.008)"]),
         ("b2_0999_2ep_lr4", "table LR×4 (0.016)", LR_COLORS["×4 (0.016)"])])


# ------------------------------------------------------------------ B. β₂ 扫描
def b2_sweeps():
    sweep_figure(
        "fig_b2_sweep_1x_lr2.svg",
        "β₂ sweep · 1x shard · table LR×2",
        "seed 42 · 2000 steps · injection=input · table LR×2 · backbone LR 0.004",
        [("b2_098_1x_lr2", "β₂=0.98", B2_COLORS["0.98"]),
         ("b2_099_1x_lr2", "β₂=0.99", B2_COLORS["0.99"]),
         ("b2_0999_1x_lr2", "β₂=0.999 (default)", B2_COLORS["0.999"])])
    sweep_figure(
        "fig_b2_sweep_2ep_lr2.svg",
        "β₂ sweep · 2-epoch shard · table LR×2",
        "seed 42 · 2000 steps · injection=input · table LR×2 · backbone LR 0.004",
        [("b2_098_2ep_lr2", "β₂=0.98", B2_COLORS["0.98"]),
         ("b2_099_2ep_lr2", "β₂=0.99", B2_COLORS["0.99"]),
         ("b2_0999_2ep_lr2", "β₂=0.999 (default)", B2_COLORS["0.999"]),
         ("b2_09999_2ep_lr2", "β₂=0.9999", B2_COLORS["0.9999"]),
         ("b2_099999_2ep_lr2", "β₂=0.99999", B2_COLORS["0.99999"])])
    sweep_figure(
        "fig_b2_sweep_1x_lr4.svg",
        "β₂ sweep · 1x shard · table LR×4",
        "seed 42 · 2000 steps · injection=input · table LR×4 · backbone LR 0.004",
        [("b2_098_1x_lr4", "β₂=0.98", B2_COLORS["0.98"]),
         ("b2_099_1x_lr4", "β₂=0.99", B2_COLORS["0.99"]),
         ("b2_0999_1x_lr4", "β₂=0.999 (default)", B2_COLORS["0.999"])])
    sweep_figure(
        "fig_b2_sweep_2ep_lr4.svg",
        "β₂ sweep · 2-epoch shard · table LR×4",
        "seed 42 · 2000 steps · injection=input · table LR×4 · backbone LR 0.004",
        [("b2_098_2ep_lr4", "β₂=0.98", B2_COLORS["0.98"]),
         ("b2_099_2ep_lr4", "β₂=0.99", B2_COLORS["0.99"]),
         ("b2_0999_2ep_lr4", "β₂=0.999 (default)", B2_COLORS["0.999"]),
         ("b2_09999_2ep_lr4", "β₂=0.9999", B2_COLORS["0.9999"])])


# ------------------------------------------------------------------ C. 交互图
def fig_beta2_spread_vs_lr():
    pts = []
    for lr, shard, hi, lo in [
        (2, "1x shard", "b2_098_1x_lr2", "b2_0999_1x_lr2"),
        (4, "1x shard", "b2_098_1x_lr4", "b2_0999_1x_lr4"),
        (2, "2-epoch shard", "b2_098_2ep_lr2", "b2_0999_2ep_lr2"),
        (4, "2-epoch shard", "b2_098_2ep_lr4", "b2_0999_2ep_lr4"),
    ]:
        a, b = DATA.get(hi), DATA.get(lo)
        if a is None or b is None:
            print(f"  skip (missing): {hi} / {lo}")
            continue
        spread = a["final_gap"] - b["final_gap"]
        pct = spread / b["final_gap"] * 100 if b["final_gap"] else 0
        pts.append((lr, shard, spread, pct))
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=150)
    markers = {"1x shard": "o", "2-epoch shard": "s"}
    colors = {"1x shard": "#E8A838", "2-epoch shard": "#D65F5F"}
    for shard in ["1x shard", "2-epoch shard"]:
        sub = sorted(p for p in pts if p[1] == shard)
        if not sub:
            continue
        ax.plot([p[0] for p in sub], [p[2] for p in sub], marker=markers[shard],
                ms=10, color=colors[shard], lw=2, label=shard)
        for p in sub:
            ax.annotate(f"{p[2]:+.2f} ({p[3]:+.0f}%)", (p[0], p[2]),
                        textcoords="offset points", xytext=(16, 2),
                        fontsize=9, color=colors[shard])
    ax.set_xticks([2, 4])
    ax.set_xticklabels(["table LR×2 (0.008)", "table LR×4 (0.016)"])
    ax.set_xlabel("table learning-rate multiplier")
    ax.set_ylabel("gap spread @2000:  β₂=0.98 minus β₂=0.999")
    ax.set_title("The β₂ effect shrinks as table LR grows —\n"
                 "but that is because high table LR breaks training", fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9.5, framealpha=0.9)
    ax.text(0.02, 0.97,
            "At table LR×4, val loss explodes (peak ≈ 10.8 = 2.2× the LR×1\n"
            "minimum) and train loss collapses, so all β₂ values fail\n"
            "together. The small spread there reflects a broken run,\n"
            "not that β₂ is unimportant.",
            transform=ax.transAxes, fontsize=9, color="#444", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fdeeee", ec="#d9a0a0"))
    fig.tight_layout()
    fig.savefig(OUT / "fig_beta2_spread_vs_lr.svg")
    plt.close(fig)
    print("fig_beta2_spread_vs_lr.svg")


def main():
    lr_sweeps()
    b2_sweeps()
    fig_beta2_spread_vs_lr()
    print(f"all figures -> {OUT}/")


if __name__ == "__main__":
    main()
