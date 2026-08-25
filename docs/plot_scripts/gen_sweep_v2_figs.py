#!/usr/bin/env python3
"""ngram-gap-lab · v2 new-standard sweep figures, log-log (dual-log) axes.

Uses ONLY the new-standard wave (β₂=0.99 · table_lr_scale=2.0 · *_v2_fixed).
Runs that have not finished are plotted with the data they have (8x = 1870
steps at snapshot time; this is stated in the figure notes).

Outputs (warm paper language, ngram-gap-plotting skill):
  fig_sweep_v2_family.svg    — log-log train/val/gap curves, one per dose,
                               dashed epoch boundaries
  fig_sweep_v2_meta.svg      — log-log: train/val/gap vs dose at fixed steps,
                               power-law exponent α in the gap legend
  fig_sweep_v2_dose_resp.svg — gap @ step 2000 vs dose: log-x + log-log panels
  fig_sweep_v2_injpos.svg    — log-log gap curves by injection point (v2)

Log-log gap panels exclude non-positive gap values (log undefined) — noted.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.environ.get("NGLAB_RUNS_DIR", os.path.join(REPO_ROOT, "data", "runs_fixed"))
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs", "epoch_scale")
os.makedirs(FIGS_DIR, exist_ok=True)

BATCH_TOKENS = 147456
GAP_MIN = 0.02  # log-log cut for positive gaps

PAPER = "#f7f5ef"
PANEL = "#fffdf8"
INK = "#232426"
MUTED = "#686d73"
LINE = "#c8c1b6"
ANCHOR = "#353d79"

# v2 new-standard injection sweep (input), dose 0.25x .. 8x.
ARMS = [
    {"key": "0.25x", "dir": "nglab0_25x_input_v2_fixed", "mult": 0.25},
    {"key": "0.5x",  "dir": "nglab0_5x_input_v2_fixed",  "mult": 0.5},
    {"key": "0.75x", "dir": "nglab0_75x_input_v2_fixed", "mult": 0.75},
    {"key": "1x",    "dir": "nglab1x_input_v2_fixed",    "mult": 1.0},
    {"key": "1.5x",  "dir": "nglab1_5x_input_v2_fixed",  "mult": 1.5},
    {"key": "2x",    "dir": "nglab2x_input_v2_fixed",    "mult": 2.0},
    {"key": "2.5x",  "dir": "nglab2_5x_input_v2_fixed",  "mult": 2.5},
    {"key": "3x",    "dir": "nglab3x_input_v2_fixed",    "mult": 3.0},
    {"key": "4x",    "dir": "nglab4x_input_v2_fixed",    "mult": 4.0},
    {"key": "5x",    "dir": "nglab5x_input_v2_fixed",    "mult": 5.0},
    {"key": "6x",    "dir": "nglab6x_input_v2_fixed",    "mult": 6.0},
    {"key": "8x",    "dir": "nglab8x_input_v2_fixed",    "mult": 8.0},
]

INJ = [
    {"key": "input",  "dir": "nglab1x_input_v2_fixed", "color": "#2d6f9f"},
    {"key": "y",      "dir": "nglab1x_y_v2_fixed",     "color": "#c4493d"},
    {"key": "v",      "dir": "nglab1x_v_v2_fixed",     "color": "#b67524"},
    {"key": "nogram", "dir": "nglab1x_nogram_v2_fixed", "color": "#236b70"},
]

STD_FOOT = ("v2 wave (table β=(0.0,0.99) · table_lr ×2 · bf16) · seed 42 · "
            "train loss = ONLINE (current training batch, same step as fixed val)")


def load_jsonl(path):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pts.append(json.loads(line))
    return pts


def epoch_boundaries(log):
    bnds = []
    prev = None
    for p in log:
        ep = p.get("epoch")
        if ep is not None and prev is not None and ep != prev:
            bnds.append((p["step"], ep))
        prev = ep if ep is not None else prev
    return bnds


def style_axis(ax):
    ax.set_facecolor(PANEL)
    ax.grid(which="both", color=LINE, linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def smooth(pts, window=7):
    n = len(pts)
    half = window // 2
    return [sum(pts[max(0, i - half):min(n, i + half + 1)]) /
            (min(n, i + half + 1) - max(0, i - half)) for i in range(n)]


def main():
    series = {}
    for arm in ARMS:
        log = load_jsonl(os.path.join(RUNS_DIR, arm["dir"], "train_log.jsonl"))
        if not log:
            print(f"[v2] WARNING: no log for {arm['dir']} — skipped")
            continue
        series[arm["key"]] = {
            "log": log,
            "steps": np.array([p["step"] for p in log]),
            "train": np.array([p["train_loss"] for p in log]),
            "val": np.array([p["val_loss"] for p in log]),
            "gap": np.array([p["gap"] for p in log]),
            "boundaries": epoch_boundaries(log),
            "mult": arm["mult"],
            "final_step": log[-1]["step"],
        }
        print(f"[v2] {arm['dir']}: {len(log)} pts, final step={log[-1]['step']}, "
              f"gap={log[-1]['gap']:+.3f}")

    keys = list(series)
    base_m = 50.1  # 1x standard dataset ≈ 50M unique tokens (observed)
    for k in keys:
        series[k]["dataset_m"] = series[k]["mult"] * base_m
    PARTIAL = "".join(
        f" · {k} partial ({series[k]['final_step']} steps)"
        for k in keys if series[k]["final_step"] < 2000)
    cmap = plt.cm.RdYlBu_r
    lv = np.log2([series[k]["mult"] for k in keys])
    lo, hi = lv.min(), lv.max()
    colors = {k: cmap((np.log2(series[k]["mult"]) - lo) / (hi - lo)) for k in keys}
    labels = {k: f"{k} · {series[k]['dataset_m']:.0f}M" for k in keys}

    # ---- Fig 1: log-log curve family ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), facecolor=PAPER)
    for ax, key, ylabel, title in zip(
            axes, ["train", "val", "gap"],
            ["train loss (online)", "val loss (fixed)", "gap = val − train"],
            ["Train loss, online (log–log)", "Val loss, fixed (log–log)",
             "Gap (log–log)"]):
        style_axis(ax)
        for k in keys:
            s = series[k]
            col = colors[k]
            if key == "gap":
                m = s["gap"] > GAP_MIN
                ax.plot(s["steps"][m], s["gap"][m], "o-", color=col,
                        linewidth=2.0, markersize=2.4, label=labels[k])
            else:
                ax.plot(s["steps"], smooth(s[key].tolist()), "o-", color=col,
                        linewidth=2.0, markersize=2.4, label=labels[k])
            for step, _ in s["boundaries"]:
                ax.axvline(step, color=col, linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="best")
    fig.suptitle("v2 sweep · log–log curve family (dashed = epoch boundary)",
                 fontsize=13, color=INK)
    fig.text(0.5, 0.012, STD_FOOT + PARTIAL + " · gap panel keeps only gap > 0.02 (log scale).",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIGS_DIR, f"fig_sweep_v2_family.{ext}"),
                    facecolor=PAPER, bbox_inches="tight",
                    **({"dpi": 150} if ext == "png" else {}))
    plt.close(fig)
    print("[v2] wrote fig_sweep_v2_family")

    # ---- Fig 2: log-log meta — loss/gap vs dose at fixed steps ----
    sample_steps = [500, 1000, 2000]
    markers = {500: "o", 1000: "s", 2000: "D"}
    meta = {m: {st: [] for st in sample_steps} for m in ["train", "val", "gap"]}
    for k in keys:
        s = series[k]
        for st in sample_steps:
            idx = np.argmin(np.abs(s["steps"] - st))
            ok = abs(s["steps"][idx] - st) <= 20
            for m in ["train", "val", "gap"]:
                meta[m][st].append(s[m][idx] if ok else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), facecolor=PAPER, sharex=True)
    dose_vals = [series[k]["mult"] for k in keys]
    specs = [("train", "train loss (online)", "#2d6f9f", "Train loss vs dose (log–log, online)"),
             ("val", "val loss (fixed)", "#c4493d", "Val loss vs dose (log–log, fixed)"),
             ("gap", "gap = val − train", ANCHOR, "Gap vs dose (log–log)")]
    for ax, (m, ylabel, color, title) in zip(axes, specs):
        style_axis(ax)
        for st in sample_steps:
            ys = np.array(meta[m][st], dtype=float)
            if m == "gap":
                keep = ~np.isnan(ys) & (ys > GAP_MIN)
                ax.plot(np.array(dose_vals)[keep], ys[keep], color=color,
                        marker=markers[st], markersize=6, linewidth=2.0,
                        linestyle={500: "-", 1000: "--", 2000: "-."}[st],
                        label=f"step {st}")
            else:
                ax.plot(dose_vals, ys, color=color, marker=markers[st],
                        markersize=6, linewidth=2.0,
                        linestyle={500: "-", 1000: "--", 2000: "-."}[st],
                        label=f"step {st}")
        if m == "gap":  # power-law fits on positive gaps
            for st, ls in [(500, "-"), (1000, "--"), (2000, "-.")]:
                ys = np.array(meta[m][st], dtype=float)
                keep = ~np.isnan(ys) & (ys > GAP_MIN)
                d = np.array(dose_vals)[keep]
                if keep.sum() >= 4:
                    alpha, c = np.polyfit(np.log10(d), np.log10(ys[keep]), 1)
                    xx = np.logspace(np.log10(d.min()), np.log10(d.max()), 50)
                    ax.plot(xx, 10 ** (alpha * np.log10(xx) + c), ":",
                            color=MUTED, linewidth=1.0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
        ax.set_xlabel("shard dose")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    # annotate exponents on gap panel
    ex_txt = []
    for st in sample_steps:
        ys = np.array(meta["gap"][st], dtype=float)
        keep = ~np.isnan(ys) & (ys > GAP_MIN)
        d = np.array(dose_vals)[keep]
        if keep.sum() >= 4:
            alpha, _ = np.polyfit(np.log10(d), np.log10(ys[keep]), 1)
            ex_txt.append(f"step {st}: α={alpha:.2f}")
    axes[2].text(0.98, 0.98, "\n".join(ex_txt), transform=axes[2].transAxes,
                 va="top", ha="right", fontsize=9, color=MUTED)
    fig.suptitle("v2 meta-relation · log–log (gap = C·dose^α, fits dotted)",
                 fontsize=13, color=INK)
    fig.text(0.5, 0.012, STD_FOOT + " · non-positive gaps excluded from log axes.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIGS_DIR, f"fig_sweep_v2_meta.{ext}"),
                    facecolor=PAPER, bbox_inches="tight",
                    **({"dpi": 150} if ext == "png" else {}))
    plt.close(fig)
    print("[v2] wrote fig_sweep_v2_meta:", "; ".join(ex_txt))

    # ---- Fig 3: dose response @ step 2000: log-x + log-log side by side ----
    resp = []
    for k in keys:
        s = series[k]
        idx = np.argmin(np.abs(s["steps"] - 2000))
        partial = abs(s["steps"][idx] - 2000) > 20
        if partial:
            idx = len(s["steps"]) - 1
        resp.append((series[k]["mult"], s["gap"][idx], partial, k))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), facecolor=PAPER)
    for ax, loglog in [(axes[0], False), (axes[1], True)]:
        style_axis(ax)
        for d, g, partial, k in resp:
            if loglog and g <= GAP_MIN:
                continue
            ax.plot(d, g, "o" if not partial else "D",
                    mfc="none" if partial else colors[k],
                    mec=colors[k] if partial else colors[k],
                    ms=7, color=ANCHOR, zorder=3)
        xs = [r[0] for r in resp]
        ax.plot(xs, [r[1] for r in resp], "-", color=LINE, linewidth=1.2, zorder=1)
        ax.axhline(0, color=LINE, linewidth=1.0, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0.25×", "0.5×", "1×", "2×", "4×", "8×"])
        ax.set_xlabel("shard dose (1× = standard training set)")
        ax.set_ylabel("gap @ 2000 steps")
        if loglog:
            ax.set_yscale("symlog", linthresh=0.1)
            pos = [(d, g) for d, g, p, k in resp if g > GAP_MIN]
            if len(pos) >= 4:
                alpha, c = np.polyfit(np.log10([p[0] for p in pos]),
                                      np.log10([p[1] for p in pos]), 1)
                xx = np.logspace(np.log10(0.25), np.log10(5), 50)
                ax.plot(xx, 10 ** (alpha * np.log10(xx) + c), ":",
                        color=MUTED, linewidth=1.2,
                        label=f"power law α={alpha:.2f}")
                ax.legend(frameon=False, fontsize=9)
            ax.set_title("log–log (positive gaps; symlog near 0)",
                         loc="left", fontsize=12, fontweight="bold")
        else:
            ax.set_title("log-x dose–response", loc="left",
                         fontsize=12, fontweight="bold")
    fig.suptitle("v2 gap @ step 2000 vs shard dose", fontsize=13, color=INK)
    fig.text(0.5, 0.012, STD_FOOT + PARTIAL + " · ◇ = partial run.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIGS_DIR, f"fig_sweep_v2_dose_resp.{ext}"),
                    facecolor=PAPER, bbox_inches="tight",
                    **({"dpi": 150} if ext == "png" else {}))
    plt.close(fig)
    print("[v2] wrote fig_sweep_v2_dose_resp")

    # ---- Fig 4: injection points, linear axes ----
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=PAPER)
    style_axis(ax)
    for arm in INJ:
        log = load_jsonl(os.path.join(RUNS_DIR, arm["dir"], "train_log.jsonl"))
        if not log:
            continue
        st = np.array([p["step"] for p in log])
        gp = np.array([p["gap"] for p in log])
        ax.plot(st, smooth(gp.tolist()), "o-", color=arm["color"], linewidth=2.1,
                markersize=2.6, label=f"{arm['key']} (final {gp[-1]:+.2f})")
    ax.axhline(0, color=LINE, linewidth=1.0, linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("gap = val − train")
    ax.set_title("v2 injection-point ablation · gap (linear axes)", loc="left",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="best")
    fig.text(0.5, 0.012, STD_FOOT,
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(FIGS_DIR, f"fig_sweep_v2_injpos.{ext}"),
                    facecolor=PAPER, bbox_inches="tight",
                    **({"dpi": 150} if ext == "png" else {}))
    plt.close(fig)
    print("[v2] wrote fig_sweep_v2_injpos")


if __name__ == "__main__":
    main()
