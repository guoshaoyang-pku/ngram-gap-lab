#!/usr/bin/env python3
"""plot_probe_artifact.py — 展示 uniform probe 的 epoch-1 artifact

核心论点：
  uniform probe 的 fgap 在 epoch 1 内就出现，是因为 probe 的 4 个固定 batch
  分布在 train 流的不同位置，随训练推进被"逐步消费"：已见 chunk 的 loss 已降、
  未见 chunk 的 loss 未降。fgap = fval − ftrain 因此混入了"训练进度"信息，
  把"模型还没看到 chunk X"误当成"泛化 gap"。online 记的是当前 batch（已见），
  在 epoch 1 内保持 gap≈0，才是真实信号。

图：
  fig_probe_artifact_1.png — 上面：4 个 probe batch 的 chunk 覆盖 vs 训练进度
                             下面：epoch1 内 online gap vs uniform fgap
  fig_probe_artifact_2.png — probe batch 的"已训比例"时间曲线（4 条阶梯线）
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

N_CHUNKS = 24264
BSZ = 72
NEED = 4 * 72
IDX = np.array([int(round(i * (N_CHUNKS - 1) / (NEED - 1))) for i in range(NEED)])
BATCHES = [IDX[b * 72:(b + 1) * 72] for b in range(4)]


def consumed_frac(step):
    return min(step * BSZ / N_CHUNKS, 1.0)


def fig_artifact_1():
    rid = "vanilla_probe_input_fixed"
    d = RUNS / rid
    onl = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines()]
    prb = [json.loads(l) for l in (d / "fixed_train_loss.jsonl").read_text().splitlines()]
    so = {r["step"]: r for r in onl}
    sp = {r["step"]: r for r in prb}
    steps = sorted(set(so) & set(sp))
    ep1 = [s for s in steps if so[s]["epoch"] == 1]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=150,
                             gridspec_kw={"height_ratios": [1.1, 1]})
    ax = axes[0]
    colors = ["#D65F5F", "#E8A838", "#6ACC65", "#4878CF"]
    for b, arr in enumerate(BATCHES):
        # 已训比例随 step 变化
        xs = np.arange(0, 340, 5)
        ys = [np.clip((arr < x * BSZ).mean(), 0, 1) * 100 for x in xs]
        ax.plot(xs, ys, color=colors[b], lw=2,
                label=f"probe batch {b} (chunk {arr.min()//1000:.0f}k-{arr.max()//1000:.0f}k)")
    ax.axvline(337, color="#333", ls="--", lw=1.5, label="epoch 1 end (step 337)")
    ax.set_ylabel("probe chunk trained (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("uniform probe: fraction of each fixed batch already trained during epoch 1",
                 fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    ax = axes[1]
    ax.plot([so[s]["step"] for s in ep1], [so[s]["gap"] for s in ep1],
            color="#4878CF", lw=1.8, label="gap = val − train (online)")
    ax.plot([sp[s]["step"] for s in ep1], [sp[s]["fixed_gap"] for s in ep1],
            color="#D65F5F", lw=1.8, ls="--",
            label="fgap = fval − ftrain (uniform probe)")
    ax.axhline(0, color="#aaa", lw=0.8)
    ax.axvline(337, color="#333", ls="--", lw=1.5)
    ax.set_xlabel("training step (epoch 1 only)")
    ax.set_ylabel("gap")
    ax.set_title("epoch-1 gap: online ≈ 0 (real) vs uniform probe +0.2 (artifact)",
                 fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fn = OUT / "fig_probe_artifact_1.png"
    fig.savefig(fn)
    plt.close(fig)
    print("wrote", fn.name)


def fig_artifact_2():
    rid = "vanilla_probe_input_fixed"
    d = RUNS / rid
    prb = [json.loads(l) for l in (d / "fixed_train_loss.jsonl").read_text().splitlines()]
    sp = {r["step"]: r for r in prb}
    steps = sorted(sp)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    # 左：全 1000 步；右：epoch1 放大
    for ax, xmax in [(axes[0], 1000), (axes[1], 337)]:
        st = [s for s in steps if s <= xmax]
        ax.plot(st, [sp[s]["fixed_gap"] for s in st], color="#D65F5F", lw=2,
                label="fgap (uniform probe)")
        ax.plot(st, [sp[s]["fixed_train_loss"] - sp[s]["fixed_train_loss"] + sp[s]["fixed_gap"]
                     for s in st], color="#D65F5F", lw=2)
        ax.plot(st, [sp[s]["fixed_train_loss"] for s in st], color="#2C9C5A", lw=1.5,
                alpha=0.7, label="ftrain (uniform)")
        ax.plot(st, [sp[s]["fixed_val_loss"] for s in st], color="#333", lw=1.5,
                ls="--", alpha=0.6, label="fval (fixed)")
        # epoch 边界
        for s in st:
            pass
        ebs = []
        prev = None
        for s in st:
            ep = sp[s]["epoch"]
            if prev is not None and ep != prev:
                ebs.append(s)
            prev = ep
        for b in ebs:
            ax.axvline(b, color="#bbb", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(0, color="#aaa", lw=0.8)
        ax.set_xlabel("step")
        ax.set_ylabel("loss / gap")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.set_title(f"uniform probe (steps ≤ {xmax})", fontsize=11)
    fig.suptitle("uniform probe fgap during epoch 1: driven by progressive chunk consumption",
                 fontsize=12)
    fig.tight_layout()
    fn = OUT / "fig_probe_artifact_2.png"
    fig.savefig(fn)
    plt.close(fig)
    print("wrote", fn.name)


def main():
    fig_artifact_1()
    fig_artifact_2()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
