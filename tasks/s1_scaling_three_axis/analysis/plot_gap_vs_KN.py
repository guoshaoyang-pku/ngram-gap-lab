#!/usr/bin/env python3
"""S1 相图：bigram final gap vs K/N（table slots / distinct contexts）。

系列：
  - 4-layer grid: 23 个正式网格点（table_summary.csv, seed 42）+ mult 128/256
    两个新臂（runs_scaling summary.json）。
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
    pts = []
    with open(os.path.join(
            ROOT, "docs/appendices/s1_scaling_three_axis/figs/table_summary.csv")) as f:
        for r in csv.DictReader(f):
            if r["module"] == "bigram" and r["seed"] == "42":
                pts.append((float(r["logical_2R"]) / N_BI, float(r["final_gap"]),
                            int(r["mult"])))
    pts.sort()
    return pts


def load_summary_gap(run_id):
    with open(os.path.join(ROOT, "data/runs_scaling", run_id, "summary.json")) as f:
        s = json.load(f)
    return s["final_gap"], s["config"]


def main():
    grid = load_grid()
    g128, _ = load_summary_gap("tbl_128_bigram_fixed")
    g256, _ = load_summary_gap("tbl_256_bigram_fixed")
    new = [(16384 * 128 / N_BI, g128, 128), (16384 * 256 / N_BI, g256, 256)]
    l1_ctrl, _ = load_summary_gap("tbl_64_bigram_l1_fixed")
    l1_perf, _ = load_summary_gap("tbl_perfect_bigram_l1_fixed")

    INF_X = 4.0  # 人为放置零碰撞点的 x 位置（>1 即无碰撞区）

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    xs = [p[0] for p in grid]
    ys = [p[1] for p in grid]
    ax.plot(xs, ys, "o-", color="#4C72B0", ms=4, lw=1.2,
            label="4-layer grid (seed 42, mult 1-64)")
    ax.plot([p[0] for p in new], [p[1] for p in new], "s", color="#DD8452",
            ms=8, mfc="none", mew=2, label="new arms mult=128/256")
    for x, y, m in new:
        ax.annotate(f"mult={m}\ngap={y:+.3f}", (x, y), textcoords="offset points",
                    xytext=(10, -26), fontsize=8, color="#DD8452")
    # single-layer pair
    ax.plot([16384 * 64 / N_BI], [l1_ctrl], "D", color="#55A868", ms=7,
            label=f"single-layer control (collision, gap={l1_ctrl:+.3f})")
    ax.plot([INF_X], [l1_perf], "*", color="#C44E52", ms=16,
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
    ax.set_title("bigram gap vs table capacity: jamming region + collision-free anchor")
    xt = list(ax.get_xticks())
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
    print("grid peak:", max(zip(ys, xs)), "| new arms:", [(m, round(y, 4)) for _, y, m in new],
          "| l1 pair:", round(l1_ctrl, 4), round(l1_perf, 4))


if __name__ == "__main__":
    main()
