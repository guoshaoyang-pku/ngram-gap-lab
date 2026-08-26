#!/usr/bin/env python3
"""nogram LR-schedule 诊断图：回答"是 warmup 少还是不收敛，还是缺 cooldown"。

六臂（全部 nogram 纯 backbone，排除表干扰）：
  A  ng lab1x_v10_nogram_fixed        warmdown（Karpathy warmup+decay）    2000 步 ✅
  B  ng lab1x_nogram_v2_fixed         warmdown（另一版）                   2000 步 ✅
  C  lrdiag_nogram_warmup_const       warmup 100 步 + 恒定 4e-3            1000 步 ❌
  D  lrdiag_nogram_warmup_cosine      warmup 100 步 + cosine decay→5%      1000 步 ⚠️
  E  lrdiag_nogram_const_4e4          constant 4e-4（1/10 lr）             1000 步 ⚠️
  F  lrdiag_nogram_const_6e4_2k       constant 6e-4（官方 nanoGPT max_lr） 2000 步 ✅

图1：train loss 轨迹（线性 x，log 不需要）
图2：train loss 轨迹（log-log，看幂律区）
图3：lr_mult 轨迹（确认各臂实际 lr 行为）
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab"
OUT = os.path.join(ROOT, "docs/figs/main")
os.makedirs(OUT, exist_ok=True)

ARMS = [
    # (run_id(含_fixed), label, color, ls)
    ("nglab1x_v10_nogram_fixed", "A warmdown (v10)", "#1f77b4", "-"),
    ("nglab1x_nogram_v2_fixed", "B warmdown (v2)", "#aec7e8", "--"),
    ("lrdiag_nogram_warmup_const_fixed", "C warmup_constant 4e-3", "#d62728", "-"),
    ("lrdiag_nogram_warmup_cosine_fixed", "D warmup_cosine 4e-3", "#ff7f0e", "-"),
    ("lrdiag_nogram_const_4e4_fixed", "E constant 4e-4", "#2ca02c", "-"),
    ("lrdiag_nogram_const_6e4_2k_fixed", "F constant 6e-4 (2k)", "#9467bd", "-"),
]


def load(run_id):
    # run_id 已含 _fixed 后缀
    path = os.path.join(ROOT, f"data/runs_fixed/{run_id}/train_log.jsonl")
    rows = [json.loads(l) for l in open(path)]
    return rows


def main():
    data = {r: load(r) for r, *_ in ARMS}

    # --- 图1: train loss 轨迹（线性） ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for run_id, label, color, ls in ARMS:
        rows = data[run_id]
        ax.plot([r["step"] for r in rows], [r["train_loss"] for r in rows],
                label=label, color=color, ls=ls, lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("online train loss (nogram, pure backbone)")
    ax.set_title("nogram LR-schedule diagnosis: train loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(OUT, "fig_lrdiag_nogram_train.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # --- 图2: train loss 轨迹（log-log） ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for run_id, label, color, ls in ARMS:
        rows = data[run_id]
        ax.plot([r["step"] for r in rows], [r["train_loss"] for r in rows],
                label=label, color=color, ls=ls, lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step (log)")
    ax.set_ylabel("train loss (log)")
    ax.set_title("nogram LR-schedule diagnosis: train loss (log-log)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    f2 = os.path.join(OUT, "fig_lrdiag_nogram_train_loglog.png")
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    # --- 图3: lr_mult 轨迹 ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for run_id, label, color, ls in ARMS:
        rows = data[run_id]
        ax.plot([r["step"] for r in rows], [r.get("lr_mult", 1.0) for r in rows],
                label=label, color=color, ls=ls, lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("lr multiplier (lr = mult × base)")
    ax.set_title("nogram LR-schedule diagnosis: lr multiplier trajectory")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = os.path.join(OUT, "fig_lrdiag_nogram_lrmult.png")
    fig.savefig(f3, dpi=150)
    plt.close(fig)

    # --- 图4: val loss 轨迹（线性） ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for run_id, label, color, ls in ARMS:
        rows = data[run_id]
        ax.plot([r["step"] for r in rows], [r["val_loss"] for r in rows],
                label=label, color=color, ls=ls, lw=2)
    ax.set_xlabel("step")
    ax.set_ylabel("fixed val loss (nogram, pure backbone)")
    ax.set_title("nogram LR-schedule diagnosis: val loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f4 = os.path.join(OUT, "fig_lrdiag_nogram_val.png")
    fig.savefig(f4, dpi=150)
    plt.close(fig)

    # --- 图5: gap 轨迹 ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for run_id, label, color, ls in ARMS:
        rows = data[run_id]
        ax.plot([r["step"] for r in rows], [r["gap"] for r in rows],
                label=label, color=color, ls=ls, lw=2)
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("gap = val − train")
    ax.set_title("nogram LR-schedule diagnosis: gap")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f5 = os.path.join(OUT, "fig_lrdiag_nogram_gap.png")
    fig.savefig(f5, dpi=150)
    plt.close(fig)

    print("saved:")
    for f in (f1, f2, f3, f4, f5):
        print(" ", f)


if __name__ == "__main__":
    main()
