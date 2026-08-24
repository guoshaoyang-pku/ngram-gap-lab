#!/usr/bin/env python3
"""plot_online_vs_uniform.py — online loss 与 uniform train probe 的大批量对比

把每个 run 的两种 train 侧口径画在同一张图：
  - online:  train_log.jsonl 的 train_loss / gap
  - uniform: fixed_train_loss.jsonl 的 fixed_train_loss / fixed_gap
两种口径共用同一固定 val（已验证 val_loss == fixed_val_loss）。

产出（docs/figs/）：
  fig_cmp_loss_<run>.png      每个 run：train vs ftrain (+共用 val) 双线
  fig_cmp_gap_<run>.png       每个 run：online gap vs uniform fgap
  fig_cmp_gap_all.png         4 run 的 gap 全家福（虚线=uniform，实线=online）
  fig_cmp_train_all.png       4 run 的 train-loss 全家福
  fig_cmp_grid_loss.png       2x2 loss 面板
  fig_cmp_grid_gap.png        2x2 gap 面板
  fig_cmp_scatter.png         online-vs-uniform 逐点散点 + 对角线
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "data" / "runs_fixed"
OUT = HERE.parent / "docs" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

RUNS_DEF = [
    # (id, display label, color)
    ("vanilla_probe_input_fixed", "input bigram+trigram", "#D65F5F"),
    ("vanilla_probe_nogram_fixed", "nogram control", "#6ACC65"),
    ("probe_b2_098_lr2_fixed", "b2=0.98 lr2", "#4878CF"),
    ("probe_b2_099_lr1_fixed", "b2=0.99 lr1", "#9772B5"),
]


def load(rid):
    d = RUNS / rid
    onl = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines()]
    prb = [json.loads(l) for l in (d / "fixed_train_loss.jsonl").read_text().splitlines()]
    so = {r["step"]: r for r in onl}
    sp = {r["step"]: r for r in prb}
    return so, sp


def aligned(so, sp):
    """按 step 对齐两种口径（取交集，probe 的 338/675 额外点忽略）。"""
    steps = sorted(set(so) & set(sp))
    return steps, so, sp


def epoch_bounds(so):
    bounds, prev = [], None
    for st in sorted(so):
        ep = so[st]["epoch"]
        if prev is not None and ep != prev:
            bounds.append(st)
        prev = ep
    return bounds


def panel_loss(ax, so, sp, label, color, with_epoch=True):
    steps = sorted(so)
    ax.plot(steps, [so[s]["train_loss"] for s in steps], color=color, lw=1.8,
            label="online train")
    ax.plot(steps, [so[s]["val_loss"] for s in steps], color="#333", lw=1.2,
            ls="--", alpha=0.6, label="fixed val (shared)")
    ax.plot(sorted(sp), [sp[s]["fixed_train_loss"] for s in sorted(sp)],
            color=color, lw=1.8, ls=":", label="uniform probe train")
    if with_epoch:
        for b in epoch_bounds(so):
            ax.axvline(b, color="#bbb", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("step")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def panel_gap(ax, so, sp, label, color, with_epoch=True):
    steps = sorted(so)
    ax.plot(steps, [so[s]["gap"] for s in steps], color=color, lw=1.8,
            label="gap = val − train (online)")
    ax.plot(sorted(sp), [sp[s]["fixed_gap"] for s in sorted(sp)],
            color=color, lw=1.8, ls="--", label="fgap (uniform probe)")
    ax.axhline(0, color="#aaa", lw=0.8)
    if with_epoch:
        for b in epoch_bounds(so):
            ax.axvline(b, color="#bbb", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("step")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def fig_single(rid, label, color, kind):
    so, sp = load(rid)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    if kind == "loss":
        panel_loss(ax, so, sp, label, color)
        ax.set_title(f"{label} · train loss: online vs uniform probe (shared val)", fontsize=11)
    else:
        panel_gap(ax, so, sp, label, color)
        ax.set_title(f"{label} · gap: online vs uniform probe", fontsize=11)
    fig.tight_layout()
    fn = OUT / f"fig_cmp_{kind}_{rid.replace('_fixed','')}.png"
    fig.savefig(fn)
    plt.close(fig)
    print("wrote", fn.name)


def fig_all(kind):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for rid, label, color in RUNS_DEF:
        so, sp = load(rid)
        steps = sorted(so)
        if kind == "gap":
            ax.plot(steps, [so[s]["gap"] for s in steps], color=color, lw=1.8,
                    label=f"{label} (online)")
            ax.plot(sorted(sp), [sp[s]["fixed_gap"] for s in sorted(sp)],
                    color=color, lw=1.6, ls="--", alpha=0.85,
                    label=f"{label} (uniform)")
        else:
            ax.plot(steps, [so[s]["train_loss"] for s in steps], color=color, lw=1.8,
                    label=f"{label} (online)")
            ax.plot(sorted(sp), [sp[s]["fixed_train_loss"] for s in sorted(sp)],
                    color=color, lw=1.6, ls="--", alpha=0.85,
                    label=f"{label} (uniform)")
    ax.axhline(0, color="#aaa", lw=0.8) if kind == "gap" else None
    ax.set_xlabel("step")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, framealpha=0.9, ncol=2)
    ax.set_title(f"all runs · {kind}: online (solid) vs uniform probe (dashed)", fontsize=12)
    fig.tight_layout()
    fn = OUT / f"fig_cmp_{kind}_all.png"
    fig.savefig(fn)
    plt.close(fig)
    print("wrote", fn.name)


def fig_grid(kind):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    for ax, (rid, label, color) in zip(axes.ravel(), RUNS_DEF):
        so, sp = load(rid)
        if kind == "gap":
            panel_gap(ax, so, sp, label, color)
        else:
            panel_loss(ax, so, sp, label, color)
        ax.set_title(label, fontsize=11)
    fig.suptitle(f"{kind}: online vs uniform probe (dashed) — 4 runs", fontsize=13)
    fig.tight_layout()
    fn = OUT / f"fig_cmp_grid_{kind}.png"
    fig.savefig(fn)
    plt.close(fig)
    print("wrote", fn.name)


def fig_scatter():
    """online train loss vs uniform probe train loss 逐点散点（同 step）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    for ax, field, ylabel in [
        (axes[0], "train", "uniform probe train loss"),
        (axes[1], "gap", "uniform probe gap"),
    ]:
        for rid, label, color in RUNS_DEF:
            so, sp = load(rid)
            steps, so, sp = aligned(so, sp)
            x = [so[s]["gap"] if field == "gap" else so[s]["train_loss"] for s in steps]
            y = [sp[s]["fixed_gap"] if field == "gap" else sp[s]["fixed_train_loss"] for s in steps]
            ax.scatter(x, y, s=8, color=color, alpha=0.6, label=label)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, color="#888", ls="--", lw=1, label="diagonal y=x")
        ax.set_xlabel("online")
        ax.set_ylabel(ylabel)
        ax.set_title(f"online vs uniform ({field})", fontsize=12)
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=7.5, framealpha=0.9)
    fig.suptitle("online vs uniform probe: point-wise agreement (diagonal = identical)", fontsize=12)
    fig.tight_layout()
    fn = OUT / "fig_cmp_scatter.png"
    fig.savefig(fn)
    plt.close(fig)
    print("wrote", fn.name)


def main():
    for rid, label, color in RUNS_DEF:
        fig_single(rid, label, color, "loss")
        fig_single(rid, label, color, "gap")
    fig_all("gap")
    fig_all("train")
    fig_grid("loss")
    fig_grid("gap")
    fig_scatter()
    print("all done ->", OUT)


if __name__ == "__main__":
    main()
