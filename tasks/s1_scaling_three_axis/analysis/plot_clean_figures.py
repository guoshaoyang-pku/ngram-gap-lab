#!/usr/bin/env python3
"""clean 单表相图 + forking 对比（SSOT: clean-table-rework.md）。

图 1 fig_clean_gap_vs_KN.png:
  - clean 单表 13 点（11 hash sparse + R=1M freq=50 + perfect 零碰撞锚点）
  - 旧 [HISTORICAL 4-LAYER FRAMEWORK] 34 点折线淡色对照
  - min(N,K) 参考线
图 2 fig_clean_forking.png:
  - clean perfect vs clean R=1M（同架构，唯一差异=碰撞）的 train/val/gap 曲线
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
N_BI = 3_538_293
FIGS = os.path.join(ROOT, "docs/appendices/s1_scaling_three_axis/figs")

CLEAN_RUNS = [  # (run_id, R)  perfect 用 R=N+1
    ("ctbl_65536_bigram_fixed", 65536),
    ("ctbl_131072_bigram_fixed", 131072),
    ("ctbl_262144_bigram_fixed", 262144),
    ("ctbl_393216_bigram_fixed", 393216),
    ("ctbl_524288_bigram_fixed", 524288),
    ("ctbl_786432_bigram_fixed", 786432),
    ("ctbl_1048576_bigram_fixed", 1048576),
    ("ctbl_1572864_bigram_fixed", 1572864),
    ("ctbl_2097152_bigram_fixed", 2097152),
    ("ctbl_2621440_bigram_fixed", 2621440),
    ("ctbl_3145728_bigram_fixed", 3145728),
    ("ctbl_4194304_bigram_fixed", 4194304),
]
PERFECT = ("ctbl_perfect_bigram_fixed", 3538294)


def gap_of(run_id):
    with open(os.path.join(ROOT, "data/runs_scaling", run_id, "summary.json")) as f:
        return json.load(f)["final_gap"]


def curve_of(run_id):
    rows = [json.loads(l) for l in open(
        os.path.join(ROOT, "data/runs_scaling", run_id, "train_log.jsonl"))]
    return ([r["step"] for r in rows], [r["train_loss"] for r in rows],
            [r["val_loss"] for r in rows], [r["gap"] for r in rows])


def fig_phase():
    pts = sorted((R / N_BI, gap_of(rid), R) for rid, R in CLEAN_RUNS)
    gp = gap_of(PERFECT[0])
    INF_X = 4.0
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    # historical 4-layer framework, faded
    import csv
    old = {}
    with open(os.path.join(FIGS, "table_summary.csv")) as f:
        for r in csv.DictReader(f):
            if r["module"] == "bigram" and r["seed"] == "42":
                old[int(r["mult"])] = float(r["final_gap"])
    for m in (44, 52, 60, 80, 96, 112, 128, 160, 192, 224, 256):
        old[m] = gap_of(f"tbl_{m}_bigram_fixed")
    ox = sorted(old)
    ax.plot([16384 * m / N_BI for m in ox], [old[m] for m in ox], "-",
            color="#999999", lw=1, alpha=0.65, zorder=1,
            label="historical 4-layer framework (34 pts, faded)")
    # clean series
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#4C72B0",
            ms=5, lw=1.6, zorder=3, label="clean single table (12 hash pts)")
    ax.plot([INF_X], [gp], "*", color="#C44E52", ms=17, zorder=4,
            label=f"clean collision-free (R=N+1, gap={gp:+.3f})")
    # annotate the R>N hash point vs perfect
    r4m = [p for p in pts if p[2] == 4194304][0]
    ax.annotate(f"R=4M (K/N=1.19)\nhash still collides:\n{r4m[1]:+.3f}",
                (r4m[0], r4m[1]), textcoords="offset points", xytext=(-8, -42),
                fontsize=8, color="#4C72B0")
    xr = np.logspace(np.log10(pts[0][0]), np.log10(INF_X), 200)
    ax.plot(xr, np.minimum(1.0, xr) * gp, "--", color="gray", lw=1, alpha=0.7,
            label="min(N,K) model (norm. to perfect)")
    ax.set_xscale("log")
    ax.set_xlabel("K / N  (table rows R / distinct bigram contexts = 3.54M)")
    ax.set_ylabel("final online gap (val - train) @ step 1000")
    ax.set_title("clean single-table bigram: smooth monotone gap vs R, "
                 "collision-free anchor above all hash points")
    xticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.set_xticks(xticks + [INF_X])
    ax.set_xticklabels([str(t) for t in xticks] + ["inf\n(collision-free)"])
    ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_clean_gap_vs_KN.png")
    fig.savefig(out)
    print(f"wrote {out}")
    for x, y, R in pts:
        print(f"  R={R:>8d} K/N={x:.4f} gap={y:+.4f}")
    print(f"  perfect gap={gp:+.4f}")


def fig_forking():
    EPOCH = 337
    runs = [
        ("ctbl_perfect_bigram_fixed", "clean collision-free (R=N+1)", "#C44E52"),
        ("ctbl_1048576_bigram_fixed", "clean hash R=1M (collision)", "#4C72B0"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    for run_id, label, color in runs:
        steps, tr, va, gap = curve_of(run_id)
        ax1.plot(steps, tr, color=color, lw=1.4, label=f"{label} (train)")
        ax1.plot(steps, va, color=color, lw=1.4, ls="--", label=f"{label} (val)")
        ax2.plot(steps, gap, color=color, lw=1.6, label=label)
        for b in (EPOCH, 2 * EPOCH):
            pre = max((s, l) for s, l in zip(steps, tr) if s < b)
            post = min((s, l) for s, l in zip(steps, tr) if s >= b)
            print(f"{run_id} @{b}: train {pre[1]:.4f}->{post[1]:.4f} "
                  f"(d={post[1]-pre[1]:+.4f})")
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
    fig.suptitle("forking under clean single table: zero-collision vs collision "
                 "(same architecture, only collision differs)")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_clean_forking.png")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_phase()
    fig_forking()
