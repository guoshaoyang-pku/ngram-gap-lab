#!/usr/bin/env python3
"""plot_probe_compare.py — 消融现象的两种 train-loss 口径可视化

A 组（在线 loss，旧 run 的 train_log.jsonl）：
  - fig_online_lr_sweep_b2_099_1x.png   1x shard · β₂=0.99 · 表LR ×1/×2/×4
  - fig_online_b2_sweep_1x_lr2.png      1x shard · LR×2 · β₂ 0.98/0.99/0.999
  - fig_online_b2_sweep_2ep_lr2.png     2-epoch shard · LR×2 · β₂ 0.98..0.99999

B 组（uniform probe，新 run 的 fixed_train_loss.jsonl fgap）：
  - fig_uniform_compare.png            input vs nogram：fgap 与在线 gap 对照
  - fig_uniform_b2_compare.png         β₂=0.98 vs 0.99 @ LR×2 的 fgap 对照

在线 loss 与 uniform probe 的 fgap 一并画出，确认两个口径现象一致。
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "data" / "runs_fixed"
OUT = HERE.parent / "docs" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

C_B2 = {"0.98": "#D65F5F", "0.99": "#E8A838", "0.999": "#4878CF",
        "0.9999": "#6ACC65", "0.99999": "#9772B5"}
C_LR = {"x1": "#2C9C5A", "x2": "#E8A838", "x4": "#D65F5F"}


def load_online(rid):
    """train_log.jsonl: 在线 loss 曲线."""
    p = RUNS / rid / "train_log.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return {"step": [r["step"] for r in rows],
            "train": [r["train_loss"] for r in rows],
            "val": [r["val_loss"] for r in rows],
            "gap": [r["gap"] for r in rows],
            "epoch": [r["epoch"] for r in rows]}


def load_probe(rid):
    """fixed_train_loss.jsonl: uniform probe 的 fgap 曲线."""
    p = RUNS / rid / "fixed_train_loss.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    return {"step": [r["step"] for r in rows],
            "ftrain": [r["fixed_train_loss"] for r in rows],
            "fval": [r["fixed_val_loss"] for r in rows],
            "fgap": [r["fixed_gap"] for r in rows]}


def epoch_bounds(traj):
    """从 epoch 列推断 epoch 边界 step."""
    bounds = []
    prev = traj["epoch"][0]
    for st, ep in zip(traj["step"], traj["epoch"]):
        if ep != prev:
            if prev > 0:
                bounds.append(st)
            prev = ep
    return bounds


def draw_ax(ax, x, y, color, label, marker_end=True):
    ax.plot(x, y, color=color, lw=1.7, label=label)
    if marker_end and x:
        ax.plot(x[-1], y[-1], "o", color=color, ms=5)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8.5, framealpha=0.9)


def three_panel_figure(title, series, filename, bset=None):
    """series: [(label, color, traj_dict)] 画 train/val/gap 三面板."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), dpi=150)
    panels = [("train", "train loss (online)"), ("val", "val loss (fixed)"),
              ("gap", "gap = val − train (online)")]
    for ax, (field, ylabel) in zip(axes, panels):
        for label, color, t in series:
            draw_ax(ax, t["step"], t[field], color, label)
        for b in sorted(bset or set()):
            ax.axvline(b, color="#999", ls="--", lw=1, alpha=0.55)
        ax.set_title(ylabel, fontsize=11)
        ax.set_xlabel("training step")
        if field == "gap":
            ax.axhline(0, color="#aaa", lw=0.8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / filename)
    plt.close(fig)
    print("wrote", filename)


# ------------------------------------------------------------------ A 组：在线 loss 消融
def online_lr_sweep_b2_099_1x():
    series = [
        ("table LR×1 (0.004)", C_LR["x1"], load_online("nglab1x_opt_rmsprop_b2_099_lr1_fixed")),
        ("table LR×2 (0.008)", C_LR["x2"], load_online("nglab1x_opt_rmsprop_2x_b2_099_fixed")),
        ("table LR×4 (0.016)", C_LR["x4"], load_online("nglab1x_opt_rmsprop_4x_b2_099_fixed")),
    ]
    bounds = set()
    for _, _, t in series:
        bounds.update(epoch_bounds(t))
    three_panel_figure("Table LR sweep · 1x shard · β₂=0.99 (online loss)",
                       series, "fig_online_lr_sweep_b2_099_1x.png", bounds)


def online_b2_sweep_1x_lr2():
    series = [
        ("β₂=0.98", C_B2["0.98"], load_online("nglab1x_opt_rmsprop_2x_b2_098_fixed")),
        ("β₂=0.99", C_B2["0.99"], load_online("nglab1x_opt_rmsprop_2x_b2_099_fixed")),
        ("β₂=0.999", C_B2["0.999"], load_online("nglab1x_opt_rmsprop_2x_fixed")),
    ]
    bounds = set()
    for _, _, t in series:
        bounds.update(epoch_bounds(t))
    three_panel_figure("β₂ sweep · 1x shard · table LR×2 (online loss)",
                       series, "fig_online_b2_sweep_1x_lr2.png", bounds)


def online_b2_sweep_2ep_lr2():
    series = [
        ("β₂=0.98", C_B2["0.98"], load_online("nglab2x_opt_rmsprop_2x_b2_098_fixed")),
        ("β₂=0.99", C_B2["0.99"], load_online("nglab2x_opt_rmsprop_2x_b2_099_fixed")),
        ("β₂=0.999", C_B2["0.999"], load_online("nglab2x_opt_rmsprop_2x_fixed")),
        ("β₂=0.9999", C_B2["0.9999"], load_online("nglab2x_opt_rmsprop_2x_b2_09999_fixed")),
        ("β₂=0.99999", C_B2["0.99999"], load_online("nglab2x_opt_rmsprop_2x_b2_099999_fixed")),
    ]
    bounds = set()
    for _, _, t in series:
        bounds.update(epoch_bounds(t))
    three_panel_figure("β₂ sweep · 2-epoch shard · table LR×2 (online loss)",
                       series, "fig_online_b2_sweep_2ep_lr2.png", bounds)


# ------------------------------------------------------------------ B 组：uniform probe 对照
def uniform_compare():
    """input vs nogram：fgap（uniform probe）与在线 gap 对照."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for ax, rid, title in [
        (axes[0], "vanilla_probe_input_fixed", "input injection (bigram+trigram)"),
        (axes[1], "vanilla_probe_nogram_fixed", "nogram control"),
    ]:
        onl = load_online(rid)
        prb = load_probe(rid)
        ax.plot(onl["step"], onl["gap"], color="#4878CF", lw=1.7,
                label="gap = val − train (online)")
        ax.plot(onl["step"], onl["train"], color="#2C9C5A", lw=1.3, alpha=0.7,
                label="train (online)")
        if prb:
            ax.plot(prb["step"], prb["fgap"], color="#D65F5F", lw=1.9,
                    ls="--", label="fgap = fval − ftrain (uniform probe)")
            ax.plot(prb["step"], prb["ftrain"], color="#9772B5", lw=1.3,
                    ls=":", alpha=0.8, label="ftrain (uniform probe)")
        for b in epoch_bounds(onl):
            ax.axvline(b, color="#999", ls="--", lw=1, alpha=0.4)
        ax.axhline(0, color="#aaa", lw=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle("online loss vs uniform probe · 1000 steps · bf16 · seed42", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_uniform_compare.png")
    plt.close(fig)
    print("wrote fig_uniform_compare.png")


def uniform_b2_compare():
    """β₂=0.98 vs 0.99 @ LR×2：fgap 对照（uniform probe）."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    # 左：在线 gap；右：uniform probe fgap
    pairs = [
        ("β₂=0.98", C_B2["0.98"], "probe_b2_098_lr2_fixed"),
        ("β₂=0.99 (std)", C_B2["0.99"], "vanilla_probe_input_fixed"),
        ("β₂=0.99 · LR×1", C_LR["x1"], "probe_b2_099_lr1_fixed"),
    ]
    for ax, field, ylabel in [
        (axes[0], "gap", "gap (online)"),
        (axes[1], "fgap", "fgap (uniform probe)"),
    ]:
        for label, color, rid in pairs:
            onl = load_online(rid)
            prb = load_probe(rid)
            if field == "gap":
                ax.plot(onl["step"], onl["gap"], color=color, lw=1.7, label=label)
                ax.plot(onl["step"][-1], onl["gap"][-1], "o", color=color, ms=5)
            elif prb:
                ax.plot(prb["step"], prb["fgap"], color=color, lw=1.7, label=label)
                ax.plot(prb["step"][-1], prb["fgap"][-1], "o", color=color, ms=5)
        ax.axhline(0, color="#aaa", lw=0.8)
        ax.set_title(ylabel, fontsize=12)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.9)
    fig.suptitle("beta2 0.98 vs 0.99 vs LR×1 · 1x shard · LR×2 · 1000 steps (uniform probe)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_uniform_b2_compare.png")
    plt.close(fig)
    print("wrote fig_uniform_b2_compare.png")


def main():
    online_lr_sweep_b2_099_1x()
    online_b2_sweep_1x_lr2()
    online_b2_sweep_2ep_lr2()
    uniform_compare()
    uniform_b2_compare()
    print("all done ->", OUT)


if __name__ == "__main__":
    main()
