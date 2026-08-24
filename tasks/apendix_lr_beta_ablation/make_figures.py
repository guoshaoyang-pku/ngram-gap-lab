#!/usr/bin/env python3
"""make_figures.py — 本附录的全部图表

自包含：数据来自 results/appendix_data.json（由 extract_data.py 生成）。
缺数据的 run 自动跳过并打印提示（补点跑完重跑即可）。

输出到 figs/（全部 SVG）：
  fig_beta2_curves.svg          β₂ 消融 · gap 全程曲线簇（3 面板）
  fig_lr_curves.svg             表 LR 消融 · gap 全程曲线簇（2 面板）
  fig_val_loss_beta2_lr.svg     β₂=0.99 高表 LR 的 val loss 体检
  fig_val_loss_short_epoch.svg  短 epoch 家族 val loss
  fig_beta2_finalgap.svg        β₂ 消融终点汇总
  fig_beta2_spread_vs_lr.svg    关键分析：β₂ 压差随表 LR 衰减
  fig_lr_sweep_b2_099.svg       β₂=0.99 下的表 LR 扫描（补点完成后完整）
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "results" / "appendix_data.json").read_text())
OUT = HERE / "figs"
OUT.mkdir(parents=True, exist_ok=True)

C = {"0.98": "#D65F5F", "0.99": "#E8A838", "0.999": "#4878CF",
     "0.9999": "#6ACC65", "0.99999": "#9772B5",
     "lr1": "#4878CF", "lr2": "#E8A838", "lr4": "#D65F5F",
     "ok": "#2C9C5A", "ref": "#888888"}


def traj(key):
    rec = DATA.get(key)
    return rec["traj"] if rec else None


def curves(ax, keys_labels, title):
    bset, finals = set(), {}
    for key, label, color in keys_labels:
        t = traj(key)
        if t is None:
            print(f"  skip (missing): {key}")
            continue
        ax.plot(t["step"], t["gap"], color=color, lw=1.7, label=label)
        ax.plot(t["step"][-1], t["gap"][-1], "o", color=color, ms=5)
        finals[label] = t["gap"][-1]
        bset.update(t["epoch_bounds"])
    for b in sorted(bset):
        ax.axvline(b, color="#999", ls="--", lw=1, alpha=0.6)
    for k, b in enumerate(sorted(bset), 2):
        ax.text(b, 0.96, f"epoch {k}", color="#777", fontsize=7.5,
                ha="center", va="top", transform=ax.get_xaxis_transform())
    if len(finals) >= 2:
        vals = list(finals.values())
        spread = max(vals) - min(vals)
        base = max(abs(v) for v in vals) or 1
        ax.text(0.97, 0.55, f"spread @end: {spread:+.2f} ({spread / base * 100:+.0f}%)",
                transform=ax.transAxes, fontsize=9, color="#333", ha="right",
                va="top", bbox=dict(boxstyle="round,pad=0.35",
                                    fc="#fffbe8", ec="#e0d5a0"))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def valpanel(ax, keys, title):
    for key, label, color in keys:
        t = traj(key)
        if t is None:
            print(f"  skip (missing): {key}")
            continue
        ax.plot(t["step"], t["val_loss"], color=color, lw=1.5, label=label)
        ax.plot(t["step"][-1], t["val_loss"][-1], "o", color=color, ms=4)
        for b in t["epoch_bounds"]:
            ax.axvline(b, color="#bbb", ls=":", lw=0.8, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("training step")
    ax.set_ylabel("val loss")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)


def fig_beta2_curves():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150, sharey=True)
    curves(axes[0], [("b2_098_1x_lr2", "β₂=0.98", C["0.98"]),
                     ("b2_099_1x_lr2", "β₂=0.99", C["0.99"]),
                     ("b2_0999_1x_lr2", "β₂=0.999 (default)", C["0.999"])],
           "1x shard · table LR×2")
    curves(axes[1], [("b2_098_2ep_lr2", "β₂=0.98", C["0.98"]),
                     ("b2_099_2ep_lr2", "β₂=0.99", C["0.99"]),
                     ("b2_0999_2ep_lr2", "β₂=0.999 (default)", C["0.999"]),
                     ("b2_09999_2ep_lr2", "β₂=0.9999", C["0.9999"]),
                     ("b2_099999_2ep_lr2", "β₂=0.99999", C["0.99999"])],
           "2-epoch shard · table LR×2")
    curves(axes[2], [("b2_098_1x_lr4", "β₂=0.98", C["0.98"]),
                     ("b2_099_1x_lr4", "β₂=0.99", C["0.99"]),
                     ("b2_0999_1x_lr4", "β₂=0.999 (default)", C["0.999"])],
           "1x shard · table LR×4")
    axes[0].set_ylabel("gap = val − train")
    for ax in axes:
        ax.set_ylim(bottom=-0.3)
    fig.suptitle("β₂ ablation — gap trajectories (backbone LR fixed 0.004, "
                 "only the TABLE lr-scale differs between panels)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_beta2_curves.svg")
    plt.close(fig)
    print("fig_beta2_curves")


def fig_lr_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, sharey=True)
    curves(axes[0], [("lr1_1x", "table LR×1 (0.004)", C["lr1"]),
                     ("b2_0999_1x_lr2", "table LR×2 (0.008)", C["lr2"]),
                     ("b2_0999_1x_lr4", "table LR×4 (0.016)", C["lr4"])],
           "1x shard · β₂=0.999")
    curves(axes[1], [("lr1_2ep", "table LR×1 (0.004)", C["lr1"]),
                     ("b2_0999_2ep_lr2", "table LR×2 (0.008)", C["lr2"]),
                     ("b2_0999_2ep_lr4", "table LR×4 (0.016)", C["lr4"])],
           "2-epoch shard · β₂=0.999")
    axes[0].set_ylabel("gap = val − train")
    for ax in axes:
        ax.set_ylim(bottom=-0.3)
    fig.suptitle("Table LR ablation — gap trajectories (β₂=0.999)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lr_curves.svg")
    plt.close(fig)
    print("fig_lr_curves")


def fig_val_loss_beta2_lr():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150)
    valpanel(axes[0], [("lr1_1x", "LR×1 · β₂=0.999 (default)", C["lr1"]),
                       ("b2_099_1x_lr2", "LR×2 · β₂=0.99", C["lr2"]),
                       ("b2_099_1x_lr4", "LR×4 · β₂=0.99", C["lr4"]),
                       ("b2_099_1x_lr1", "LR×1 · β₂=0.99 (new)", C["ok"])],
             "1x shard · val loss")
    valpanel(axes[1], [("lr1_2ep", "LR×1 · β₂=0.999 (default)", C["lr1"]),
                       ("b2_099_2ep_lr2", "LR×2 · β₂=0.99", C["lr2"]),
                       ("b2_099_2ep_lr4", "LR×4 · β₂=0.99", C["lr4"]),
                       ("b2_099_2ep_lr1", "LR×1 · β₂=0.99 (new)", C["ok"])],
             "2-epoch shard · val loss")
    fig.suptitle("Val loss health check — β₂=0.99 at high table LR "
                 "(dotted = epoch boundaries)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_val_loss_beta2_lr.svg")
    plt.close(fig)
    print("fig_val_loss_beta2_lr")


def fig_val_loss_short_epoch():
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    valpanel(ax, [("short_025x_b2_099", "0.25x shard · β₂=0.99 · LR×1", C["lr4"]),
                  ("short_05x_b2_099", "0.5x shard · β₂=0.99 · LR×1", "#F2B366"),
                  ("short_025x_b2_0999", "0.25x shard · β₂=0.999 · LR×1", "#B4A7D6"),
                  ("short_05x_b2_0999", "0.5x shard · β₂=0.999 · LR×1", "#86C5AC")],
             "Short-epoch family · val loss (table LR×1)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_val_loss_short_epoch.svg")
    plt.close(fig)
    print("fig_val_loss_short_epoch")


def fig_beta2_finalgap():
    order = ["0.98", "0.99", "0.999", "0.9999", "0.99999"]
    panels = [
        ("1x shard · LR×2", [("0.98", "b2_098_1x_lr2"), ("0.99", "b2_099_1x_lr2"),
                             ("0.999", "b2_0999_1x_lr2")]),
        ("2-epoch shard · LR×2", [("0.98", "b2_098_2ep_lr2"), ("0.99", "b2_099_2ep_lr2"),
                                  ("0.999", "b2_0999_2ep_lr2"), ("0.9999", "b2_09999_2ep_lr2"),
                                  ("0.99999", "b2_099999_2ep_lr2")]),
        ("2-epoch shard · LR×4", [("0.98", "b2_098_2ep_lr4"), ("0.99", "b2_099_2ep_lr4"),
                                  ("0.999", "b2_0999_2ep_lr4"), ("0.9999", "b2_09999_2ep_lr4")]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    for ax, (title, items) in zip(axes, panels):
        xs, ys = [], []
        for b2, key in items:
            rec = DATA.get(key)
            if rec is None:
                print(f"  skip (missing): {key}")
                continue
            xs.append(order.index(b2))
            ys.append(rec["final_gap"])
        ax.bar(range(len(xs)), ys, color=[C[order[x]] for x in xs], alpha=0.85)
        for i, v in zip(range(len(xs)), ys):
            ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=8)
        ax.set_title(title, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("final gap @2000")
        ax.grid(alpha=0.2, axis="y")
        ax.set_ylim(bottom=0)
    fig.suptitle("β₂ ablation — final gap summary (post-fix _fixed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_beta2_finalgap.svg")
    plt.close(fig)
    print("fig_beta2_finalgap")


def fig_beta2_spread_vs_lr():
    """β₂ 压差（0.98 vs 0.999 终点差）随表 LR 衰减 —— 关键交互效应。"""
    pts = []
    for lr, shard, hi, lo in [
        (2, "1x shard", "b2_098_1x_lr2", "b2_0999_1x_lr2"),
        (4, "1x shard", "b2_098_1x_lr4", "b2_0999_1x_lr4"),
        (2, "2-epoch shard", "b2_098_2ep_lr2", "b2_0999_2ep_lr2"),
        (4, "2-epoch shard", "b2_098_2ep_lr4", "b2_0999_2ep_lr4"),
    ]:
        a, b = DATA.get(hi), DATA.get(lo)
        if a is None or b is None:
            print(f"  skip (missing): {hi} / {lo}")
            continue
        spread = a["final_gap"] - b["final_gap"]
        pct = spread / b["final_gap"] * 100 if b["final_gap"] else 0
        pts.append((lr, shard, spread, pct))
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    markers = {"1x shard": "o", "2-epoch shard": "s"}
    colors = {"1x shard": C["lr2"], "2-epoch shard": C["lr4"]}
    for shard in ["1x shard", "2-epoch shard"]:
        sub = sorted(p for p in pts if p[1] == shard)
        if not sub:
            continue
        ax.plot([p[0] for p in sub], [p[2] for p in sub], marker=markers[shard],
                ms=9, color=colors[shard], lw=2, label=shard)
        for p in sub:
            ax.annotate(f"{p[2]:+.2f} ({p[3]:+.0f}%)", (p[0], p[2]),
                        textcoords="offset points", xytext=(14, 2),
                        fontsize=8.5, color=colors[shard])
    ax.set_xticks([2, 4])
    ax.set_xticklabels(["table LR×2 (0.008)", "table LR×4 (0.016)"])
    ax.set_ylabel("gap spread at step 2000:  β₂=0.98 minus β₂=0.999")
    ax.set_title("β₂ effect collapses as table LR grows — the interaction\n"
                 "that earlier 'β₂ barely matters at high LR' readings missed",
                 fontsize=10)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.text(0.02, 0.98,
            "LR×4 also inflates val loss to ~10.7 (2.2× the LR×1 minimum),\n"
            "so its small β₂ spread reflects a broken training run,\n"
            "not that β₂ is unimportant.",
            transform=ax.transAxes, fontsize=8.5, color="#444", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdeeee", ec="#d9a0a0"))
    fig.tight_layout()
    fig.savefig(OUT / "fig_beta2_spread_vs_lr.svg")
    plt.close(fig)
    print("fig_beta2_spread_vs_lr")


def fig_lr_sweep_b2_099():
    """β₂=0.99（全部无动量）下的表学习率扫描：LR×1（补点）vs ×2 vs ×4。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    keys_1x = [("b2_099_1x_lr1", "LR×1 (0.004)  [new]", C["ok"]),
               ("b2_099_1x_lr2", "LR×2 (0.008)", C["lr2"]),
               ("b2_099_1x_lr4", "LR×4 (0.016)", C["lr4"])]
    keys_2ep = [("b2_099_2ep_lr1", "LR×1 (0.004)  [new]", C["ok"]),
                ("b2_099_2ep_lr2", "LR×2 (0.008)", C["lr2"]),
                ("b2_099_2ep_lr4", "LR×4 (0.016)", C["lr4"])]
    curves(axes[0, 0], keys_1x, "1x shard · gap")
    curves(axes[0, 1], keys_2ep, "2-epoch shard · gap")

    def valpanel2(ax, keys, title):
        for key, label, color in keys:
            t = traj(key)
            if t is None:
                print(f"  skip (missing): {key}")
                continue
            ax.plot(t["step"], t["val_loss"], color=color, lw=1.5, label=label)
            ax.plot(t["step"][-1], t["val_loss"][-1], "o", color=color, ms=4)
            for b in t["epoch_bounds"]:
                ax.axvline(b, color="#bbb", ls=":", lw=0.8, alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("val loss")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    valpanel2(axes[1, 0], keys_1x, "1x shard · val loss")
    valpanel2(axes[1, 1], keys_2ep, "2-epoch shard · val loss")
    fig.suptitle("Table LR sweep at β₂=0.99 (momentum-free) — gap & val loss\n"
                 "(LR×1 runs are new in this appendix)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_lr_sweep_b2_099.svg")
    plt.close(fig)
    print("fig_lr_sweep_b2_099")


def main():
    fig_beta2_curves()
    fig_lr_curves()
    fig_val_loss_beta2_lr()
    fig_val_loss_short_epoch()
    fig_beta2_finalgap()
    fig_beta2_spread_vs_lr()
    fig_lr_sweep_b2_099()
    print(f"all figures -> {OUT}/")


if __name__ == "__main__":
    main()