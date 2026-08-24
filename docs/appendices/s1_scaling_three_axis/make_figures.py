#!/usr/bin/env python3
"""Appendix S1 · scaling 三视图绘图（一图一变量：train / val / gap）。

数据源：data/runs_scaling/（pilot run 或 full grid run）。

每张图只变化一个变量，同时画 train / val / gap 三行：
  - epoch 组：固定模块（both），只变 epoch 长度 L（L1/L4 pilot；full 后 L1-L4）
  - table 组：固定模块（bigram-only）+ 固定 L4，只变 table size（1M/128K/16K）
"""
import json
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_scaling")
FIGS_DIR = os.path.join(HERE, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)


def load_train_log(run_dir):
    path = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            out.append(e)
    return out


def load_fixed(run_dir):
    path = os.path.join(run_dir, "fixed_train_loss.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            out.append(e)
    return out


def three_panel(axs, xs, train, val, xlabel, label, color):
    axs[0].plot(xs, train, color=color, marker=".", markersize=4)
    axs[0].set_ylabel("train loss")
    axs[0].grid(alpha=0.3)
    axs[1].plot(xs, val, color=color, marker=".", markersize=4)
    axs[1].set_ylabel("val loss")
    axs[1].grid(alpha=0.3)
    axs[2].plot(xs, np.array(val) - np.array(train), color=color, marker=".",
                markersize=4)
    axs[2].set_ylabel("gap (val−train)")
    axs[2].set_xlabel(xlabel)
    axs[2].grid(alpha=0.3)
    axs[2].axhline(0, color="gray", lw=0.6)
    for ax, ys in ((axs[0], train), (axs[1], val),
                   (axs[2], np.array(val) - np.array(train))):
        if len(xs) and len(ys):
            ax.annotate(
                label,
                xy=(xs[-1], ys[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                va="center",
            )


def first_run(runs, *run_ids):
    """Return the first available run, preferring full-grid ids."""
    for run_id in run_ids:
        if run_id in runs:
            return run_id
    return None


def epoch_run_id(runs, L, module, align):
    return first_run(
        runs,
        f"ep_{L}_{module}_{align}",
        f"pilot_ep_{L}_{module}_{align}",
    )


def table_run_id(runs, mult, module):
    pilot_names = {
        64: "1M",
        8: "128K",
        1: "16K",
    }
    pilot = pilot_names.get(mult)
    candidates = [f"tbl_{mult}_{module}"]
    if pilot is not None:
        candidates.append(f"tbl_pilot_{pilot}_{module}")
    return first_run(runs, *candidates)


def plot_epoch_fixedstep(runs):
    """固定模块=both，只改变 epoch length L；三视图 vs step."""
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    colors = {"L1": "#2196F3", "L2": "#4CAF50", "L3": "#FF9800", "L4": "#E91E63"}
    for L, color in colors.items():
        run_id = epoch_run_id(runs, L, "both", "fs")
        if run_id is None:
            continue
        log = runs[run_id]["train_log"]
        if not log:
            continue
        xs = [e["step"] for e in log]
        tr = [e["train_loss"] for e in log]
        va = [e["val_loss"] for e in log]
        three_panel(axs, xs, tr, va, "step", L, color)
    fig.suptitle("Epoch length scaling (fixed-step; module=both)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_fixedstep_both_train_val_gap.png"), dpi=150)
    plt.close(fig)
    print("saved epoch_fixedstep_both_train_val_gap.png")


def plot_epoch_fixedstep_nogram(runs):
    """backbone safety 对照：固定 no-ngram，只改变 epoch length L."""
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    colors = {"L1": "#2196F3", "L2": "#4CAF50", "L3": "#FF9800", "L4": "#E91E63"}
    for L, color in colors.items():
        run_id = epoch_run_id(runs, L, "nogram", "fs")
        if run_id is None:
            continue
        log = runs[run_id]["train_log"]
        if not log:
            continue
        xs = [e["step"] for e in log]
        tr = [e["train_loss"] for e in log]
        va = [e["val_loss"] for e in log]
        three_panel(axs, xs, tr, va, "step", L, color)
    fig.suptitle("Backbone safety (fixed-step; module=no-ngram)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_fixedstep_nogram_train_val_gap.png"), dpi=150)
    plt.close(fig)
    print("saved epoch_fixedstep_nogram_train_val_gap.png")


def plot_epoch_fixed_probe(runs):
    """固定 probe 的 train / val / gap 三视图，只改变 epoch length L."""
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    colors = {"L1": "#2196F3", "L2": "#4CAF50", "L3": "#FF9800", "L4": "#E91E63"}
    for L, color in colors.items():
        run_id = epoch_run_id(runs, L, "both", "fs")
        if run_id is None:
            continue
        fixed = runs[run_id]["fixed"]
        if not fixed:
            continue
        xs = [e["step"] for e in fixed]
        tr = [e["fixed_train_loss"] for e in fixed]
        va = [e["fixed_val_loss"] for e in fixed]
        three_panel(axs, xs, tr, va, "step", L, color)
    fig.suptitle("Fixed-probe scaling (fixed-step; module=both)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_fixedprobe_gap_vs_step.png"), dpi=150)
    plt.close(fig)
    print("saved epoch_fixedprobe_gap_vs_step.png")


def plot_table_fixedstep(runs):
    """固定 L4 + bigram-only，只改变 table size；三视图."""
    fig, axs = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    sizes = (
        (64, "1M", "#2196F3"),
        (32, "512K", "#42A5F5"),
        (16, "256K", "#66BB6A"),
        (8, "128K", "#FF9800"),
        (4, "64K", "#FFB74D"),
        (2, "32K", "#AB47BC"),
        (1, "16K", "#E91E63"),
    )
    for mult, label, color in sizes:
        run_id = table_run_id(runs, mult, "bigram")
        if run_id is None:
            continue
        log = runs[run_id]["train_log"]
        if not log:
            continue
        xs = [e["step"] for e in log]
        tr = [e["train_loss"] for e in log]
        va = [e["val_loss"] for e in log]
        three_panel(axs, xs, tr, va, "step", label, color)
    fig.suptitle("Table-size scaling (fixed-step; L4, module=bigram-only)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "table_bigram_train_val_gap.png"), dpi=150)
    plt.close(fig)
    print("saved table_bigram_train_val_gap.png")


def plot_table_final_vs_size(runs):
    """table size 扫描：final train / val / gap vs logical 2R."""
    fig, axs = plt.subplots(3, 1, figsize=(8, 11), sharex=True)
    pts = []
    for mult, label in ((64, "1M"), (32, "512K"), (16, "256K"),
                        (8, "128K"), (4, "64K"), (2, "32K"), (1, "16K")):
        run_id = table_run_id(runs, mult, "bigram")
        if run_id is None:
            continue
        fixed = runs[run_id]["fixed"]
        if not fixed:
            continue
        logical = 2 * 8192 * mult
        last = fixed[-1]
        pts.append((logical, label, last["fixed_train_loss"],
                    last["fixed_val_loss"], last["fixed_gap"]))
    pts.sort()
    xs = [p[0] for p in pts]
    for ax, pos in zip(axs, (2, 3, 4)):
        ys = [p[pos] for p in pts]
        ax.plot(xs, ys, marker="o", color="#9C27B0")
        for x, label, *_ in pts:
            ax.annotate(label, (x, ys[xs.index(x)]), xytext=(4, 3),
                        textcoords="offset points", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    axs[0].set_ylabel("final train loss")
    axs[1].set_ylabel("final val loss")
    axs[2].set_ylabel("final gap")
    axs[2].set_xlabel("logical addresses 2R (per n-gram, per layer)")
    for ax in axs:
        ax.set_xscale("log")
    fig.suptitle("Table-size scaling (final fixed probe; L4, module=bigram-only)", y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "table_final_gap_vs_2R.png"), dpi=150)
    plt.close(fig)
    print("saved table_final_gap_vs_2R.png")


def main():
    runs = {}
    for run_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
        run_id = os.path.basename(run_dir)
        if not run_id.startswith(("ep_", "pilot_ep_", "tbl_")):
            continue
        runs[run_id] = {
            "train_log": load_train_log(run_dir),
            "fixed": load_fixed(run_dir),
        }
    print(f"found {len(runs)} runs")
    plot_epoch_fixedstep(runs)
    plot_epoch_fixedstep_nogram(runs)
    plot_epoch_fixed_probe(runs)
    plot_table_fixedstep(runs)
    plot_table_final_vs_size(runs)


if __name__ == "__main__":
    main()
