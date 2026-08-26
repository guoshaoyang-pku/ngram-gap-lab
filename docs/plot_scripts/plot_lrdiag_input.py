#!/usr/bin/env python3
"""带 n-gram 表 input 注入在 constant 6e-4 下的 gap 现象图（对照 v10 input warmdown）。

对比：
  A  ng lab1x_v10_input_fixed        旧框架 ve 表 + lr 4e-3 warmdown → gap ~1.87 @2000
  B  lrdiag_input_6e4_2k_fixed       clean 单表 R=2^20 + lr 6e-4 constant → 观察 gap 拉开
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab"
OUT = os.path.join(ROOT, "docs/figs/main")
os.makedirs(OUT, exist_ok=True)

RUNS = [
    ("nglab1x_v10_input_fixed", "v10 input (legacy ve, lr4e-3 warmdown)", "#1f77b4", "-"),
    ("lrdiag_input_6e4_2k_fixed", "clean table R=2^20, lr6e-4 constant", "#d62728", "-"),
]


def load(run):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, f"data/runs_fixed/{run}/train_log.jsonl"))]
    return rows


def main():
    data = {r: load(r) for r, *_ in RUNS}

    for metric, ylabel, fname in [
        ("train_loss", "online train loss", "fig_lrdiag_input_train.png"),
        ("val_loss", "fixed val loss", "fig_lrdiag_input_val.png"),
        ("gap", "gap = val − train", "fig_lrdiag_input_gap.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        for run_id, label, color, ls in RUNS:
            rows = data[run_id]
            ax.plot([r["step"] for r in rows], [r[metric] for r in rows],
                    label=label, color=color, ls=ls, lw=2, marker="o", ms=3)
        if metric == "gap":
            ax.axhline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"input injection (bigram+trigram): {ylabel}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(OUT, fname)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("saved:", p)


if __name__ == "__main__":
    main()
