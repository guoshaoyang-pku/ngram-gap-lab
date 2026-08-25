#!/usr/bin/env python3
"""S1 相图：bigram final gap vs K/N（table slots / distinct contexts）。

系列：
  - 4-layer grid: 23 个正式网格点（table_summary.csv, seed 42）+ 11 个加密臂
    （mult 44-256，runs_scaling summary.json），共 34 点一条折线。
  - single-layer pair: mult=64 --bigram_single_layer 对照（碰撞）与
    perfect-map 零碰撞锚点。perfect 无有限 K，画在轴右端 "inf" 刻度处。
  - min(N,K) 参考线：粗糙模型 Delta ~ min(N,K) 的归一化形状。

输出: docs/appendices/s1_scaling_three_axis/figs/fig_gap_vs_KN.png
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
N_BI = 3_538_293  # distinct bigram contexts, shard 1 (table_occupancy / perfect map)
OUT = os.path.join(ROOT, "docs/appendices/s1_scaling_three_axis/figs/fig_gap_vs_KN.png")


def load_grid():
    pts = {}
    with open(os.path.join(
            ROOT, "docs/appendices/s1_scaling_three_axis/figs/table_summary.csv")) as f:
        for r in csv.DictReader(f):
            if r["module"] == "bigram" and r["seed"] == "42":
                pts[int(r["mult"])] = float(r["final_gap"])
    # merge dense-fill arms (mult 44-256) straight from run summaries
    for m in (44, 52, 60, 80, 96, 112, 128, 160, 192, 224, 256):
        g, _ = load_summary_gap(f"tbl_{m}_bigram_fixed")
        pts[m] = g
    return sorted((16384 * m / N_BI, g, m) for m, g in pts.items())


def load_summary_gap(run_id):
    with open(os.path.join(ROOT, "data/runs_scaling", run_id, "summary.json")) as f:
        s = json.load(f)
    return s["final_gap"], s["config"]


def main():
    grid = load_grid()  # 34 points, one polyline
    grid_mults = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
                  24, 28, 32, 36, 40, 48, 56, 64}
    l1_ctrl, _ = load_summary_gap("tbl_64_bigram_l1_fixed")
    l1_perf, _ = load_summary_gap("tbl_perfect_bigram_l1_fixed")

    INF_X = 4.0  # 人为放置零碰撞点的 x 位置（>1 即无碰撞区）

    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    xs = [p[0] for p in grid]
    ys = [p[1] for p in grid]
    ax.plot(xs, ys, "-", color="#4C72B0", lw=1.2, zorder=1)
    gxs = [p[0] for p in grid if p[2] in grid_mults]
    gys = [p[1] for p in grid if p[2] in grid_mults]
    fxs = [p[0] for p in grid if p[2] not in grid_mults]
    fys = [p[1] for p in grid if p[2] not in grid_mults]
    ax.plot(gxs, gys, "o", color="#4C72B0", ms=4, zorder=2,
            label="4-layer grid (mult 1-64)")
    ax.plot(fxs, fys, "s", color="#DD8452", ms=6, mfc="none", mew=1.8, zorder=3,
            label="dense-fill arms (mult 44-256)")
    # single-layer pair
    ax.plot([16384 * 64 / N_BI], [l1_ctrl], "D", color="#55A868", ms=7, zorder=4,
            label=f"single-layer control (collision, gap={l1_ctrl:+.3f})")
    ax.plot([INF_X], [l1_perf], "*", color="#C44E52", ms=16, zorder=5,
            label=f"single-layer collision-free (perfect map, gap={l1_perf:+.3f})")
    # min(N,K) reference (normalized to peak of grid)
    import numpy as np
    xr = np.logspace(np.log10(min(xs)), np.log10(INF_X), 200)
    ref = np.minimum(1.0, xr) * max(ys)
    ax.plot(xr, ref, "--", color="gray", lw=1, alpha=0.7,
            label="min(N,K) model (normalized)")

    ax.set_xscale("log")
    ax.set_xlabel("K / N  (table logical addresses / distinct bigram contexts = 3.54M)")
    ax.set_ylabel("final online gap (val - train) @ step 1000")
    ax.set_title("bigram gap vs table capacity: dense sampling reveals "
                 "sawtooth across jamming region")
    xticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.set_xticks(xticks + [INF_X])
    ax.set_xticklabels([str(t) for t in xticks] + ["inf\n(collision-free)"])
    ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
    ax.annotate("K = N", (1.0, ax.get_ylim()[0]), textcoords="offset points",
                xytext=(4, 2), fontsize=8, color="gray")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    print(f"n={len(grid)} | peak: mult={max(grid, key=lambda p: p[1])[2]} "
          f"gap={max(ys):.4f} | l1 pair: {l1_ctrl:.4f} {l1_perf:.4f}")


if __name__ == "__main__":
    main()
