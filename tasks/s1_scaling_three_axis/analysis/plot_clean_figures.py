#!/usr/bin/env python3
"""clean 单表相图 + forking 对比（SSOT: clean-table-rework.md）。

图 1 fig_clean_gap_vs_KN.png（semilog-x）:
  - clean 单表 bigram 30 点（wave-1 13 + wave-2 加密 17）+ perfect 零碰撞锚点
  - clean 单表 trigram 6 点（N_tri = 19.0M，K/N 只到 0.42）
  - 旧 [HISTORICAL 4-LAYER FRAMEWORK] 34 点折线淡色对照
  - min(N,K) 参考线
图 2 fig_clean_gap_vs_KN_loglog.png（双对数）:
  - 同数据，x=K/N log、y=gap log，检验低 gap 区幂律形态
图 3 fig_clean_forking.png:
  - 用户指定只画 3 条曲线：小表 (64K)、大表 (4M)、零碰撞 (perfect)，freq=50
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
N_BI = 3_538_293
N_TRI = 18_989_467
FIGS = os.path.join(ROOT, "docs/appendices/s1_scaling_three_axis/figs")

CLEAN_BI = [65536, 131072, 262144, 393216, 524288, 786432, 1048576, 1572864,
            2097152, 2621440, 3145728, 4194304,           # wave-1
            16384, 32768, 49152, 98304, 163840, 196608, 327680, 458752,
            655360, 917504, 1310720, 1835008, 2359296, 2883584, 3670016,
            5242880, 6291456]                              # wave-2 dense fill
CLEAN_TRI = [65536, 262144, 1048576, 2097152, 4194304, 8388608]
PERFECT = ("ctbl_perfect_bigram_fixed", 3538294)
CURVE_RUNS = [  # 用户指定：只画小 / 大 / 零碰撞三条曲线
    ("ctbl_65536_bigram_curve_fixed", "clean R=64K (K/N=0.02)", "#55A868"),
    ("ctbl_4194304_bigram_curve_fixed", "clean R=4M (K/N=1.19, collision)", "#4C72B0"),
    ("ctbl_perfect_bigram_fixed", "clean collision-free (R=N+1)", "#C44E52"),
]


def gap_of(run_id):
    with open(os.path.join(ROOT, "data/runs_scaling", run_id, "summary.json")) as f:
        return json.load(f)["final_gap"]


def curve_of(run_id):
    rows = [json.loads(l) for l in open(
        os.path.join(ROOT, "data/runs_scaling", run_id, "train_log.jsonl"))]
    return ([r["step"] for r in rows], [r["train_loss"] for r in rows],
            [r["val_loss"] for r in rows], [r["gap"] for r in rows])


def collect():
    bi = sorted((R / N_BI, gap_of(f"ctbl_{R}_bigram_fixed"), R) for R in CLEAN_BI)
    tri = []
    for R in CLEAN_TRI:
        d = os.path.join(ROOT, "data/runs_scaling", f"ctbl_{R}_trigram_fixed")
        if os.path.exists(os.path.join(d, "summary.json")):
            tri.append((R / N_TRI, gap_of(f"ctbl_{R}_trigram_fixed"), R))
    tri.sort()
    gp = gap_of(PERFECT[0])
    return bi, tri, gp


def old_framework_series():
    old = {}
    with open(os.path.join(FIGS, "table_summary.csv")) as f:
        for r in csv.DictReader(f):
            if r["module"] == "bigram" and r["seed"] == "42":
                old[int(r["mult"])] = float(r["final_gap"])
    for m in (44, 52, 60, 80, 96, 112, 128, 160, 192, 224, 256):
        old[m] = gap_of(f"tbl_{m}_bigram_fixed")
    ox = sorted(old)
    return [16384 * m / N_BI for m in ox], [old[m] for m in ox]


def fig_phase(bi, tri, gp):
    INF_X = 4.0
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=150)
    ox, oy = old_framework_series()
    ax.plot(ox, oy, "-", color="#999999", lw=1, alpha=0.6, zorder=1,
            label="historical 4-layer framework bigram (34 pts, faded)")
    ax.plot([p[0] for p in bi], [p[1] for p in bi], "o-", color="#4C72B0",
            ms=4.5, lw=1.6, zorder=3, label=f"clean single-table bigram ({len(bi)} pts)")
    if tri:
        ax.plot([p[0] for p in tri], [p[1] for p in tri], "^-", color="#DD8452",
                ms=6, lw=1.4, zorder=3, label=f"clean single-table trigram ({len(tri)} pts)")
    ax.plot([INF_X], [gp], "*", color="#C44E52", ms=17, zorder=4,
            label=f"clean collision-free bigram (R=N+1, gap={gp:+.3f})")
    xr = np.logspace(np.log10(bi[0][0]), np.log10(INF_X), 200)
    ax.plot(xr, np.minimum(1.0, xr) * gp, "--", color="gray", lw=1, alpha=0.7,
            label="min(N,K) model (norm. to perfect)")
    ax.set_xscale("log")
    ax.set_xlabel("K / N  (table rows R / distinct contexts; N_bi=3.54M, N_tri=19.0M)")
    ax.set_ylabel("final online gap (val - train) @ step 1000")
    ax.set_title("clean single-table gap vs R: smooth monotone bigram, "
                 "collision-free anchor above all hash points")
    xticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.set_xticks(xticks + [INF_X])
    ax.set_xticklabels([str(t) for t in xticks] + ["inf\n(collision-free)"])
    ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_clean_gap_vs_KN.png")
    fig.savefig(out)
    print(f"wrote {out}")


def fig_loglog(bi, tri, gp):
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    ax.plot([p[0] for p in bi], [p[1] for p in bi], "o-", color="#4C72B0",
            ms=4.5, lw=1.4, label=f"clean bigram ({len(bi)} pts)")
    if tri:
        ax.plot([p[0] for p in tri], [p[1] for p in tri], "^-", color="#DD8452",
                ms=6, lw=1.4, label=f"clean trigram ({len(tri)} pts)")
    # log-log slopes over the smooth low-K region (K/N <= 0.3)
    for pts, name, color in ((bi, "bigram", "#4C72B0"), (tri, "trigram", "#DD8452")):
        lo = [(x, y) for x, y, _ in pts if x <= 0.3]
        if len(lo) >= 3:
            lx = np.log([p[0] for p in lo])
            ly = np.log([p[1] for p in lo])
            slope = np.polyfit(lx, ly, 1)[0]
            xr = np.array([lo[0][0], lo[-1][0]])
            y0 = lo[0][1] * (xr / lo[0][0]) ** slope
            ax.plot(xr, y0, ":", color=color, lw=1.2,
                    label=f"{name} low-K slope={slope:.2f}")
    ax.plot([4.0], [gp], "*", color="#C44E52", ms=15, label="collision-free bigram")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K / N")
    ax.set_ylabel("final online gap (log)")
    ax.set_title("log-log view: low-gap region tests the power-law form")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_clean_gap_vs_KN_loglog.png")
    fig.savefig(out)
    print(f"wrote {out}")


def fig_forking():
    EPOCH = 337
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=150)
    for run_id, label, color in CURVE_RUNS:
        steps, tr, va, gap = curve_of(run_id)
        ax1.plot(steps, tr, color=color, lw=1.3, label=f"{label} (train)")
        ax1.plot(steps, va, color=color, lw=1.3, ls="--", label=f"{label} (val)")
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
    ax1.legend(fontsize=7, loc="upper right")
    ax2.set_ylabel("online gap (val - train)")
    ax2.set_title("gap trajectory")
    ax2.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("clean single-table curves: small table / large table / collision-free")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig_clean_forking.png")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    bi, tri, gp = collect()
    print(f"bigram {len(bi)} pts | trigram {len(tri)} pts | perfect {gp:+.4f}")
    for x, y, R in bi:
        print(f"  bi R={R:>8d} K/N={x:.4f} gap={y:+.4f}")
    for x, y, R in tri:
        print(f"  tri R={R:>8d} K/N={x:.4f} gap={y:+.4f}")
    fig_phase(bi, tri, gp)
    fig_loglog(bi, tri, gp)
    fig_forking()
