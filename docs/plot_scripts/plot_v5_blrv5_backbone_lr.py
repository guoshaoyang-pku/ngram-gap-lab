#!/usr/bin/env python3
"""fig_v5_blrv5_backbone_lr — backbone LR 扫描（blrv5 批，§39）的 net-gap 判决图。

数据源：docs/figs/main/blrv5_backbone_lr_summary.csv（由 12 个 run 的 summary.json
回传汇总，字段含 run_id/host/steps/seed，可逐行溯源）。
口径：gap = fixed val − current-batch online train，step 1000，seed 42；
table LR 全部锁 128×（run_v5_clean.sh 标准），唯一变量 = backbone --lr。
图：点 = 各 run 的原始 final gap；细线 = 视觉连接（log-x）。
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "Hiragino Sans GB",
    "PingFang SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "docs/figs/main/blrv5_backbone_lr_summary.csv"
OUT = ROOT / "docs/figs/main/fig_v5_blrv5_backbone_lr.png"

rows = list(csv.DictReader(open(CSV)))
series = {
    "input": ("#2d6f9f", "input 臂 gap"),
    "nogram": ("#236b70", "nogram 对照 gap"),
}
NET_COLOR = "#c4493d"

fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
lrs = sorted({float(r["lr"]) for r in rows})

for arm, (color, label) in series.items():
    pts = sorted((float(r["lr"]), float(r["final_gap"])) for r in rows if r["arm"] == arm)
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "-", color=color, lw=1.2, alpha=0.85, zorder=2)
    ax.scatter(xs, ys, s=42, color=color, label=label, zorder=3)

net = sorted(
    (lr, by_arm[lr])
    for lr in lrs
    for by_arm in [{float(r["lr"]): float(r["final_gap"]) for r in rows if r["arm"] == "input"}]
)
nog = {float(r["lr"]): float(r["final_gap"]) for r in rows if r["arm"] == "nogram"}
inp = {float(r["lr"]): float(r["final_gap"]) for r in rows if r["arm"] == "input"}
xs = sorted(inp)
ys = [inp[x] - nog[x] for x in xs]
ax.plot(xs, ys, "-", color=NET_COLOR, lw=1.2, alpha=0.85, zorder=2)
ax.scatter(xs, ys, s=42, color=NET_COLOR, label="net gap = input - nogram", zorder=3)

peak_x = xs[ys.index(max(ys))]
ax.annotate(
    f"peak net {max(ys):.2f} @ lr={peak_x:g}",
    (peak_x, max(ys)),
    textcoords="offset points",
    xytext=(8, -14),
    fontsize=10,
    color=NET_COLOR,
)

ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(["1e-4", "3e-4", "6e-4", "1e-3", "2e-3", "4e-3"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.axhline(0, color="#999", lw=0.8, ls=":")
ax.set_xlabel("backbone learning rate（table LR 固定 128×）")
ax.set_ylabel("gap @ step 1000（fixed val - online train）")
ax.set_title(
    "backbone LR 扫描：net gap 强响应 → A 因子（backbone 读出放大）成立\n"
    "blrv5_{input,nogram}_lr{0p0001..0p0040} · seed 42 · 1000 steps · 360-1/360-2",
    fontsize=11,
)
ax.legend(frameon=False, fontsize=10)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
