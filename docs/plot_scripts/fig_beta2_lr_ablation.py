#!/usr/bin/env python3
"""fig_beta2_lr_ablation.py — β₂ 消融图 + 学习率消融图（两张）

数据源：data/runs_fixed/*_fixed/summary.json（唯一权威数据，post-fix）。
只取 seed 42 的 run（s43/s44 是 1000 步变体，不进图）。

输出：
  docs/figs/table_opt/fig_beta2_ablation.{svg,png}
  docs/figs/table_opt/fig_table_lr_ablation.{svg,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "data" / "runs_fixed"
OUT = REPO / "docs" / "figs" / "table_opt"
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> dict | None:
    p = RUNS / name / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return s


def rmsprop_runs() -> dict[str, dict]:
    out = {}
    for d in sorted(RUNS.iterdir()):
        if not d.name.endswith("_fixed"):
            continue
        s = load(d.name)
        if s is None:
            continue
        c = s.get("config", {})
        if c.get("table_optimizer") == "rmsprop":
            out[d.name] = s
    return out


# ---------------------------------------------------------------------------
# 图 1：β₂ 消融（table optimizer = RMSProp，beta1=0，变 beta2）
# 三条线：1x shard LR×2 / 1x shard LR×4 / 2-epoch shard LR×2
# ---------------------------------------------------------------------------
def fig_beta2() -> None:
    runs = rmsprop_runs()
    series = {
        "1x shard · LR×2": ["nglab1x_opt_rmsprop_2x_b2_098_fixed",
                            "nglab1x_opt_rmsprop_2x_b2_099_fixed",
                            "nglab1x_opt_rmsprop_2x_fixed"],
        "1x shard · LR×4": ["nglab1x_opt_rmsprop_4x_b2_098_fixed",
                            "nglab1x_opt_rmsprop_4x_b2_099_fixed",
                            "nglab1x_opt_rmsprop_4x_fixed"],
        "2-epoch shard · LR×2": ["nglab2x_opt_rmsprop_2x_b2_098_fixed",
                                 "nglab2x_opt_rmsprop_2x_b2_099_fixed",
                                 "nglab2x_opt_rmsprop_2x_fixed",
                                 "nglab2x_opt_rmsprop_2x_b2_09999_fixed",
                                 "nglab2x_opt_rmsprop_2x_b2_099999_fixed"],
    }
    colors = {"1x shard · LR×2": "#4878CF",
              "1x shard · LR×4": "#6ACC65",
              "2-epoch shard · LR×2": "#D65F5F"}

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    for label, names in series.items():
        xs, ys = [], []
        for n in names:
            s = runs.get(n)
            if s is None:
                continue
            b2 = s["config"]["table_betas"][1]
            xs.append(-np.log10(1 - b2))
            ys.append(s["final_gap"])
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        ax.plot(xs, ys, "o-", color=colors[label], lw=2, ms=7, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8,
                        color=colors[label])

    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xticklabels(["0.98", "0.99", "0.999", "0.9999", "0.99999"])
    ax.set_xlim(1.7, 6.3)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("table optimizer beta2  (RMSProp, no momentum)")
    ax.set_ylabel("final gap = val − train  (step 2000, seed 42)")
    ax.set_title("β₂ ablation of the n-gram table optimizer (post-fix _fixed data)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.text(0.02, 0.02,
            "Missing points: no 0.99/0.98 at mainline LR×1 (only 0.999 & 0.9999).\n"
            "1x shard LR×2/4 have no 0.9999 point (only the 2-epoch family does).",
            transform=ax.transAxes, fontsize=8, color="#444",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f2e7", ec="#d9c9a0"))

    fig.tight_layout()
    fig.savefig(OUT / "fig_beta2_ablation.svg")
    fig.savefig(OUT / "fig_beta2_ablation.png")
    plt.close(fig)
    print("saved fig_beta2_ablation")


# ---------------------------------------------------------------------------
# 图 2：学习率消融（table optimizer = RMSProp，β₂=0.999 固定，变 table_lr_scale）
# 两条线：1x shard / 2-epoch shard；参考线 = no-ngram 对照
# ---------------------------------------------------------------------------
def fig_lr() -> None:
    runs = rmsprop_runs()
    series = {
        "1x shard (2000 steps)": ["nglab1x_v10_input_fixed",
                                  "nglab1x_opt_rmsprop_2x_fixed",
                                  "nglab1x_opt_rmsprop_4x_fixed"],
        "2-epoch shard (2000 steps)": ["nglab2x_input_v10_fv_fixed",
                                       "nglab2x_opt_rmsprop_2x_fixed",
                                       "nglab2x_opt_rmsprop_4x_fixed"],
    }
    colors = {"1x shard (2000 steps)": "#4878CF",
              "2-epoch shard (2000 steps)": "#D65F5F"}
    xlabels = ["1×", "2×", "4×"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    for label, names in series.items():
        xs, ys = [], []
        for i, n in enumerate(names):
            s = runs.get(n)
            if s is None:
                continue
            xs.append(i)
            ys.append(s["final_gap"])
        ax.plot(xs, ys, "o-", color=colors[label], lw=2, ms=8, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5,
                        color=colors[label])

    # 参考：无 n-gram 对照（LR×1）
    nog = runs.get("nglab1x_v10_nogram_fixed")
    if nog:
        ax.axhline(nog["final_gap"], color="#888", ls="--", lw=1.5)
        ax.annotate(f"no n-gram table (LR×1): {nog['final_gap']:.2f}",
                    (2.02, nog["final_gap"]), fontsize=8.5, color="#555",
                    va="bottom", ha="right")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("table learning-rate multiplier  (backbone LR = 0.004 fixed)")
    ax.set_ylabel("final gap = val − train  (step 2000, seed 42)")
    ax.set_title("Table LR ablation (RMSProp, β₂ = 0.999, post-fix _fixed data)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.text(0.02, 0.02,
            "beta2 fixed at 0.999. LR×2 (1x shard) also has seed 43/44 @1000-step\n"
            "variants (2.17 / 2.34; different step count, not plotted).",
            transform=ax.transAxes, fontsize=8, color="#444",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f2e7", ec="#d9c9a0"))

    fig.tight_layout()
    fig.savefig(OUT / "fig_table_lr_ablation.svg")
    fig.savefig(OUT / "fig_table_lr_ablation.png")
    plt.close(fig)
    print("saved fig_table_lr_ablation")


if __name__ == "__main__":
    fig_beta2()
    fig_lr()
