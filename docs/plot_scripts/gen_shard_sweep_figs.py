#!/usr/bin/env python3
"""ngram-gap-lab · training-set-size sweep canonical figures.

Implements the ngram-gap-plotting skill for the M5 sweep: how train / val
loss and the val−train gap evolve as the training-set size grows (0.25x..8x
of the 1x standard dataset) over a fixed 2000-step budget.

Reads data/runs_fixed/<run_id>/train_log.jsonl for every arm; the run-name
"x" multiplier is converted to an actual training-set size in unique tokens
(anchored to the observed 1x steps-per-epoch) and printed as the legend.

Outputs (warm paper visual language, see skill):
  fig_sweep_family.svg   — train/val/gap panels vs step, epoch boundaries
  fig_sweep_gap_step.svg — gap vs step, one curve per dataset size
  fig_sweep_gap_size.svg — gap @ step 2000 vs actual training-set size (log-x)
  fig_sweep_meta.svg     — train/val/gap at fixed steps vs shard dose
  fig_sweep_epoch_map.svg — observed epoch boundaries by shard dose
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

BATCH_TOKENS = 147456  # device_batch 72 x seq_len 2048

PAPER = "#f7f5ef"
PANEL = "#fffdf8"
INK = "#232426"
MUTED = "#686d73"
LINE = "#c8c1b6"
ANCHOR = "#353d79"

# One arm per historical dataset-size multiplier. The multiplier is a legacy
# label; the actual dataset size is measured from each run's log.
ARMS = [
    {"key": "0.25x", "dir": "nglab0_25x_input_fv_fixed"},
    {"key": "0.5x",  "dir": "nglab0_5x_input_fv_fixed"},
    {"key": "1x",    "dir": "nglab1x_v10_input_fixed"},
    {"key": "1.5x",  "dir": "nglab1_5x_input_fv_fixed"},
    {"key": "2x",    "dir": "nglab2x_input_v10_fv_fixed"},
    {"key": "2.5x",  "dir": "nglab2_5x_input_fv_v3_fixed"},
    {"key": "3x",    "dir": "nglab3x_input_fv_v3_fixed"},
    {"key": "4x",    "dir": "nglab4x_input_fv_v3_fixed"},
    {"key": "5x",    "dir": "nglab5x_input_fv_fixed"},
    {"key": "6x",    "dir": "nglab6x_input_fv_fixed"},
    {"key": "8x",    "dir": "nglab8x_input_fv_fixed"},
]


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


def epoch_boundaries(train_log):
    """(step, epoch) pairs where the epoch field increments (dataset passes)."""
    bnds = []
    prev = None
    for p in train_log:
        ep = p.get("epoch")
        if ep is not None and prev is not None and ep != prev:
            bnds.append((p["step"], ep))
        prev = ep if ep is not None else prev
    return bnds


def steps_per_epoch(train_log):
    bnds = epoch_boundaries(train_log)
    if len(bnds) < 2:
        last = train_log[-1]
        return float(last["step"]) / max(1, last.get("epoch", 1))
    return float(np.median(np.diff([step for step, _ in bnds])))


def smooth(pts, window=7):
    """Centered moving average to soften single-step noise / early spikes."""
    n = len(pts)
    if n == 0:
        return pts
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(pts[lo:hi]) / (hi - lo))
    return out


def style_axis(ax):
    ax.set_facecolor(PANEL)
    ax.grid(axis="y", color=LINE, linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def main():
    series = {}
    for arm in ARMS:
        run_dir = os.path.join(RUNS_DIR, arm["dir"])
        log = load_jsonl(os.path.join(run_dir, "train_log.jsonl"))
        if not log:
            print(f"[sweep] WARNING: no train_log for {arm['dir']} — skipped")
            continue
        spe = steps_per_epoch(log)
        # gap at step 2000 (fixed comparison step; 3x/4x are long runs beyond 2000)
        gap2000, gap2000_step = None, None
        for p in log:
            if p["step"] >= 2000:
                gap2000, gap2000_step = p["gap"], p["step"]
                break
        # Cut all curves to the fixed 2000-step comparison window so long runs
        # (3x/4x extended to 3800/5000) do not stretch the shared axis.
        cut = [p for p in log if p["step"] <= 2000]
        series[arm["key"]] = {
            "log": log,
            "steps": np.array([p["step"] for p in cut]),
            "train": np.array([p["train_loss"] for p in cut]),
            "val": np.array([p["val_loss"] for p in cut]),
            "gap": np.array([p["gap"] for p in cut]),
            "boundaries": [b for b in epoch_boundaries(log) if b[0] <= 2000],
            "spe": spe,
            "gap2000": gap2000,
            "gap2000_step": gap2000_step,
            "mult": float(arm["key"].rstrip("x")),
        }
        gtxt = f"{gap2000:+.3f}" if gap2000 is not None else "n/a"
        print(f"[sweep] {arm['dir']}: {len(log)} pts, spe={spe:.0f}, "
              f"gap@2000={gtxt} (at step {gap2000_step}), final gap={log[-1]['gap']:+.4f}")

    if not series:
        raise SystemExit("[sweep] no data yet")

    # Anchor the actual dataset size to the observed 1x run (spe = 340 steps =
    # 50M tokens). 5x/6x/8x run only ~2000 steps (2 passes) so their spe is not
    # directly measurable; size = multiplier x base is the correct quantity.
    base_m = series["1x"]["spe"] * BATCH_TOKENS / 1e6
    for k in series:
        series[k]["dataset_m"] = series[k]["mult"] * base_m

    keys = list(series)
    cmap = plt.cm.RdYlBu_r
    lv = np.log2(np.array([series[k]["dataset_m"] for k in keys]))
    lo, hi = lv.min(), lv.max()
    colors = {k: cmap((np.log2(series[k]["dataset_m"]) - lo) / (hi - lo))
              for k in keys}
    labels = {k: f"{k} · {series[k]['dataset_m']:.0f}M tokens" for k in keys}

    # ---- Fig 1: train / val / gap family vs step, epoch boundaries ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), facecolor=PAPER)
    for ax, key, ylabel, title in zip(
            axes, ["train", "val", "gap"],
            ["train loss", "val loss", "gap = val − train"],
            ["Train loss by dataset size",
             "Val loss by dataset size",
             "Gap by dataset size"]):
        style_axis(ax)
        for k in keys:
            s = series[k]
            col = colors[k]
            ax.plot(s["steps"], smooth(s[key].tolist()), color=col, linewidth=2.2,
                    marker="o", markersize=2.8, label=labels[k])
            for step, _ in s["boundaries"]:
                ax.axvline(step, color=col, linestyle="--", linewidth=0.9, alpha=0.5)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        if key == "gap":
            ax.axhline(0, color=LINE, linewidth=1.0, linestyle="--")
        ax.legend(frameon=False, ncol=2, fontsize=8, loc="best")
    fig.suptitle("Training-set-size sweep (v10 · fixed-val · seed 42 · input injection · 2000 steps)",
                 fontsize=13, color=INK)
    fig.text(0.5, 0.012,
             "OLD standard: table β=(0.0, 0.999) · table_lr_scale=1.0 · fp32 · *_fixed data",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_family.svg"), format="svg",
                facecolor=PAPER, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_family.png"), dpi=150,
                facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    print("[sweep] wrote fig_sweep_family.svg / .png")

    # ---- Fig 2: gap vs step, one curve per dataset size ----
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=PAPER)
    style_axis(ax)
    for k in keys:
        s = series[k]
        ax.plot(s["steps"], smooth(s["gap"].tolist()), color=colors[k], linewidth=2.0,
                marker="o", markersize=2.6, label=labels[k])
        for step, _ in s["boundaries"]:
            ax.plot(step, s["gap"][np.argmin(np.abs(s["steps"] - step))],
                    "o", ms=4, mfc="white", mec=colors[k], mew=1.2, zorder=4)
    ax.axhline(0, color=LINE, linewidth=1.0, linestyle="--")
    ax.set_xlabel("step")
    ax.set_ylabel("gap = val − train")
    ax.set_title("Gap evolution by dataset size (○ = start of a new pass)",
                 loc="left", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="best")
    fig.text(0.5, 0.012,
             "OLD standard: table β=(0.0, 0.999) · table_lr_scale=1.0 · fp32 · *_fixed data",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_gap_step.svg"), format="svg",
                facecolor=PAPER, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_gap_step.png"), dpi=150,
                facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    print("[sweep] wrote fig_sweep_gap_step.svg / .png")

    # ---- Fig 3: gap @ 2000 vs actual dataset size (log-x) ----
    # Uses gap at the fixed comparison step 2000 (not the run-final gap), so
    # the 3x/4x long runs (3800/5000 steps) are compared on the same axis.
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=PAPER)
    style_axis(ax)
    points = [(series[k]["dataset_m"], series[k]["gap2000"], k)
              for k in keys if series[k]["gap2000"] is not None]
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    ax.plot(xs, ys, "o-", color=ANCHOR, linewidth=2.0, markersize=7, zorder=3)
    for x, y, k in points:
        ax.annotate(labels[k], (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks([12, 25, 50, 100, 200, 400])
    ax.set_xticklabels(["12M", "25M", "50M", "100M", "200M", "400M"])
    ax.set_xlabel("training-set size (unique tokens)")
    ax.set_ylabel("gap @ 2000 steps (val − train)")
    ax.set_title("Gap at fixed step 2000 vs training-set size",
                 loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="both", color=LINE, linewidth=0.7, alpha=0.55)
    fig.text(0.5, 0.012,
             "The historical x× label means shard dose: 1× is the standard training set; 0.25× is one quarter of it.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_gap_size.svg"), format="svg",
                facecolor=PAPER, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_gap_size.png"), dpi=150,
                facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    print("[sweep] wrote fig_sweep_gap_size.svg / .png")

    # ---- Fig 4: meta relation — loss/gap vs shard dose at fixed steps ----
    sample_steps = [500, 1000, 1500, 2000]
    step_markers = {500: "o", 1000: "s", 1500: "^", 2000: "D"}
    step_styles = {500: "-", 1000: "--", 1500: ":", 2000: "-."}
    dose_values = [series[k]["mult"] for k in keys]
    meta = {metric: {step: [] for step in sample_steps}
            for metric in ["train", "val", "gap"]}
    for k in keys:
        s = series[k]
        for step in sample_steps:
            idx = np.argmin(np.abs(s["steps"] - step))
            if abs(s["steps"][idx] - step) <= 20:
                for metric in ["train", "val", "gap"]:
                    meta[metric][step].append(s[metric][idx])
            else:
                for metric in ["train", "val", "gap"]:
                    meta[metric][step].append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), facecolor=PAPER, sharex=True)
    metric_specs = [
        ("train", "train loss", TRAIN if "TRAIN" in globals() else "#2d6f9f",
         "Train loss vs shard dose"),
        ("val", "val loss", VAL if "VAL" in globals() else "#c4493d",
         "Val loss vs shard dose"),
        ("gap", "gap = val − train", ANCHOR,
         "Gap vs shard dose"),
    ]
    for ax, (metric, ylabel, color, title) in zip(axes, metric_specs):
        style_axis(ax)
        for step in sample_steps:
            ax.plot(dose_values, meta[metric][step], color=color,
                    linewidth=2.0, marker=step_markers[step],
                    markersize=5.5, linestyle=step_styles[step],
                    label=f"step {step}")
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8])
        ax.set_xticklabels(["0.25×", "0.5×", "1×", "1.5×", "2×", "2.5×",
                            "3×", "4×", "5×", "6×", "8×"], rotation=45, ha="right")
        ax.set_xlabel("shard dose (relative training-set size; 1× = standard)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, loc="best")
        if metric == "gap":
            ax.axhline(0, color=LINE, linewidth=1.0, linestyle="--")
    fig.suptitle("Meta-relation: fixed-step loss as a function of shard dose",
                 fontsize=13, color=INK)
    fig.text(0.5, 0.012,
             "OLD standard: table β=(0.0, 0.999) · table_lr_scale=1.0 · fp32 · *_fixed data",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_meta.svg"), format="svg",
                facecolor=PAPER, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_meta.png"), dpi=150,
                facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    print("[sweep] wrote fig_sweep_meta.svg / .png")

    # ---- Fig 5: epoch-boundary map in the same step/dose coordinates ----
    fig, ax = plt.subplots(figsize=(10, 6.8), facecolor=PAPER)
    style_axis(ax)
    y_positions = np.arange(len(keys))
    for y, k in zip(y_positions, keys):
        s = series[k]
        col = colors[k]
        ax.plot([0, 2000], [y, y], color=col, linewidth=2.0, alpha=0.75)
        for step, ep in s["boundaries"]:
            if step <= 2000:
                ax.plot([step, step], [y - 0.34, y + 0.34],
                        color=col, linewidth=1.8)
                ax.text(step, y + 0.38, f"E{ep}", color=col, fontsize=7,
                        ha="center", va="bottom", clip_on=True)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{k}  ({labels[k]})" for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 2000)
    ax.set_xlabel("step")
    ax.set_ylabel("shard dose (actual training-set size in parentheses)")
    ax.set_title("Observed epoch boundaries by shard dose",
                 loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color=LINE, linewidth=0.7, alpha=0.55)
    fig.text(0.5, 0.012,
             "Each | marks the start of a new pass over that dose's training set. "
             "OLD standard: β₂=0.999 · table_lr_scale=1.0 · fp32.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_epoch_map.svg"), format="svg",
                facecolor=PAPER, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS_DIR, "fig_sweep_epoch_map.png"), dpi=150,
                facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)
    print("[sweep] wrote fig_sweep_epoch_map.svg / .png")


if __name__ == "__main__":
    main()
