#!/usr/bin/env python3
"""Historical within-epoch structure figures (v12, 2026-08-07).

The canonical view is the structure INSIDE a single epoch (step within epoch),
not the staircase across epochs. Longer epochs = more val points per epoch =
the within-epoch rise is clearly resolved.

Figures written to docs/figs/theory/:
  fig_toy_within_epoch_train_val_gap.png  toy epochs, global-step window
  fig_toy_within_epoch_aligned.png        toy gap vs step-within-epoch (overlay)
  fig_main_within_epoch_aligned.png       main gap vs step-within-epoch per arm
  fig_main_within_epoch_train_val_gap.png main train/val/gap, 2-epoch window
  fig_epochlen_clarity.png                within-epoch rise vs epoch length

The toy T1 metadata is not shipped in this repository. Supply
NGLAB_TOY_RESULTS explicitly from a reviewed result bundle before running.

Usage: NGLAB_TOY_RESULTS=/reviewed/t1/results \
       python3 docs/plot_scripts/gen_within_epoch_figs.py
"""
import bisect
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(os.environ.get("NGLAB_RUNS_DIR", REPO_ROOT / "data" / "runs_fixed"))
FIGS_DIR = Path(os.environ.get("NGLAB_FIG_DIR", REPO_ROOT / "docs" / "figs" / "theory"))
TOY_RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get(
    "NGLAB_TOY_RESULTS",
    REPO_ROOT / "tasks" / "l1_lookup_replay" / "results" / "inputs",
))
if "NGLAB_TOY_RESULTS" not in os.environ and len(sys.argv) <= 1:
    raise SystemExit(
        "historical T1 inputs are not bundled; provide a positional toy result "
        "directory or set NGLAB_TOY_RESULTS"
    )
FIGS_DIR.mkdir(parents=True, exist_ok=True)
TOY_BASELINE = "t5b_beta_000_999_low"

# palette (warm paper, from ngram-gap-plotting skill)
BG = "#f7f5ef"; BORDER = "#c8c1b6"; TEXT = "#686d73"; ANCHOR = "#353d79"
TRAIN = "#3c8d5a"; VAL = "#d97932"

ARMS = [
    {"key": "0.25x", "label": "0.25x epoch", "color": "#9C27B0", "dir": "nglab0_25x_input_fv_fixed"},
    {"key": "0.5x",  "label": "0.5x epoch",  "color": "#4CAF50", "dir": "nglab0_5x_input_fv_fixed"},
    {"key": "1x",    "label": "1x epoch",    "color": "#2196F3", "dir": "nglab1x_v10_input_fixed"},
    {"key": "2x",    "label": "2x epoch",    "color": "#FF9800", "dir": "nglab2x_input_v10_fv_fixed"},
    {"key": "4x",    "label": "4x epoch",    "color": "#C44E52", "dir": "nglab4x_input_fv_v3_fixed"},
    {"key": "8x",    "label": "8x epoch",    "color": ANCHOR,    "dir": "nglab8x_input_fv_fixed"},
]


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(alpha=0.25, color=BORDER)


def load_jsonl(path):
    pts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pts.append(json.loads(line))
    return pts


def parse_toy_run(run_dir):
    meta_path = os.path.join(run_dir, "run_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    boundaries = meta.get("epoch_boundaries", [])
    text = open(os.path.join(run_dir, "train.log"), errors="replace").read()
    train, val = {}, {}
    for ln in text.split("\n"):
        m = re.match(r"\s*step\s+(\d+)", ln)
        if m:
            s = int(m.group(1))
            ml = re.search(r"loss:\s*([\d.]+)", ln)
            if ml:
                train[s] = float(ml.group(1))
    for ln in text.split("\n"):
        m = re.match(r"\[val_loss\]\s+step\s+(\d+)", ln)
        if m:
            s = int(m.group(1))
            ml = re.search(r"loss:\s*([\d.]+)", ln)
            if ml:
                val[s] = float(ml.group(1))
    per_epoch = meta.get("per_epoch", [])
    return {
        "boundaries": boundaries,
        "train": train,
        "val": val,
        "exact": {p["step"]: p.get("headline_gap") for p in per_epoch},
        "final_gap": meta.get("headline_gap"),
    }


def main():
    # ---------------- toy ----------------
    toy = parse_toy_run(os.path.join(TOY_RUNS, TOY_BASELINE))
    bnd = toy["boundaries"]
    # log-epoch e (e>=5) spans [bnd[e-5], bnd[e-4]); bnd[i] = start of log-epoch i+5

    # fig 1: toy train/val/gap, global-step window epochs 5-9
    lo, hi = 60, 470
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    xs = [s for s in range(lo, hi + 1) if s in toy["train"]]
    ys = [toy["train"][s] for s in xs]
    ax.plot(xs, ys, color=TRAIN, lw=1.0, alpha=0.85, ls="--", label="train loss (per step)")
    vxs = [s for s in range(lo, hi + 1) if s in toy["val"]]
    ax.plot(vxs, [toy["val"][s] for s in vxs], color=VAL, lw=1.8, marker="o", ms=3.5,
            label="val loss (every 10 steps)")
    for i, b in enumerate(bnd):
        if lo <= b <= hi:
            ax.axvline(b, color=BORDER, ls=":", lw=1.2)
    ax.set_ylabel("loss (nats)")
    ax.set_title("toy (β=(0, 0.999), epoch ≈ 80 steps): inside-epoch structure — "
                 "train drops faster than val, gap opens inside each epoch", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    style(ax)

    ax = axes[1]
    gxs = [s for s in vxs if s in toy["train"]]
    gys = [toy["val"][s] - toy["train"][s] for s in gxs]
    ax.plot(gxs, gys, color=ANCHOR, lw=1.8, marker="o", ms=3.5, label="gap = val − train")
    for i, b in enumerate(bnd):
        if lo <= b <= hi:
            ax.axvline(b, color=BORDER, ls=":", lw=1.2)
            ax.text(b, ax.get_ylim()[1] - 0.15, f"ep{i + 5}", ha="center",
                    fontsize=7, color=TEXT, va="top")
    for s, g in sorted(toy["exact"].items()):
        if lo <= s <= hi:
            ax.plot([s], [g], marker="D", ms=7, color="#C44E52", zorder=5,
                    label="exact eval (sparse)" if s == 200 else None)
    ax.axhline(0, color=TEXT, lw=0.8)
    ax.set_ylabel("gap (nats)")
    ax.set_xlabel("global step")
    ax.legend(fontsize=8, loc="upper left")
    style(ax)
    fig.tight_layout()
    for name in ("fig_toy_within_epoch_train_val_gap",):
        fig.savefig(os.path.join(FIGS_DIR, name + ".png"), dpi=150)
        fig.savefig(os.path.join(FIGS_DIR, name + ".svg"))
    plt.close(fig)

    # fig 2: toy gap vs step-within-epoch (overlay several epochs)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ep_ids = [6, 7, 8, 9]
    for ep in ep_ids:
        s0, s1 = bnd[ep - 5], bnd[ep - 4]
        gxs = [s - s0 for s in range(s0, s1 + 1, 10) if s in toy["train"] and s in toy["val"]]
        gys = [toy["val"][s] - toy["train"][s] for s in range(s0, s1 + 1, 10)
               if s in toy["train"] and s in toy["val"]]
        ax.plot(gxs, gys, marker="o", ms=4, lw=1.6, label=f"epoch {ep}")
    ax.set_xlabel("step within epoch")
    ax.set_ylabel("gap = val − train (nats)")
    ax.set_title("toy: gap rises inside each epoch (epoch ≈ 80 steps, val every 10)",
                 fontsize=11)
    ax.legend(fontsize=9)
    style(ax)
    fig.tight_layout()
    for name in ("fig_toy_within_epoch_aligned",):
        fig.savefig(os.path.join(FIGS_DIR, name + ".png"), dpi=150)
        fig.savefig(os.path.join(FIGS_DIR, name + ".svg"))
    plt.close(fig)

    # ---------------- main runs: within-epoch aligned ----------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()
    summary = []
    for ax, arm in zip(axes, ARMS):
        pts = load_jsonl(os.path.join(RUNS_DIR, arm["dir"], "train_log.jsonl"))
        if not pts:
            ax.set_visible(False)
            continue
        steps = np.array([p["step"] for p in pts])
        eps = np.array([p["epoch"] for p in pts])
        gaps = np.array([p["gap"] for p in pts])
        bnds = []
        prev = None
        for s, e in zip(steps, eps):
            if prev is not None and e != prev:
                bnds.append(int(s))
            prev = e
        ep_starts = [int(steps[0])] + bnds
        first_ep = int(eps[0])
        byep = {}
        for s, e, g in zip(steps, eps, gaps):
            byep.setdefault(int(e), []).append((int(s), float(g)))
        for ep, arr in byep.items():
            arr.sort()
        med_x, med_y, all_rise = [], [], []
        for ep, arr in sorted(byep.items()):
            eidx = ep - first_ep
            if eidx < 0 or eidx >= len(ep_starts):
                continue
            x = np.array([s - ep_starts[eidx] for s, g in arr], float)
            y = np.array([g for s, g in arr], float)
            ax.plot(x, y, lw=0.7, alpha=0.35, color=arm["color"])
            if len(x) >= 4:
                grid = np.linspace(0, x.max(), 40)
                med_x.append(grid)
                med_y.append(np.interp(grid, x, y))
                all_rise.append(y[-1] - y[0])
        if med_x:
            allx = np.linspace(0, max(m[-1] for m in med_x), 80)
            interp = np.array([np.interp(allx, mx, my) for mx, my in zip(med_x, med_y)])
            ax.plot(allx, interp.mean(0), color=arm["color"], lw=2.2,
                    label=f"mean across {len(med_y)} epochs")
        npts_epoch = int(np.mean([len(v) for v in byep.values()]))
        elen = int(np.mean([v[-1][0] - v[0][0] for v in byep.values()]))
        ax.set_title(f"{arm['label']} · epoch≈{elen} steps · {npts_epoch} pts/epoch · "
                     f"final gap {pts[-1]['gap']:+.2f}", fontsize=9)
        ax.set_xlabel("step within epoch")
        ax.set_ylabel("gap (nats)")
        ax.legend(fontsize=7)
        style(ax)
        summary.append({"key": arm["key"], "elen": elen, "npts": npts_epoch,
                        "rise": float(np.mean(all_rise)) if all_rise else float("nan"),
                        "n_epochs": len(byep)})
    fig.suptitle("main model: gap vs step-within-epoch (all epochs overlaid) — "
                 "longer epoch ⇒ more points ⇒ within-epoch rise clearly resolved",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for name in ("fig_main_within_epoch_aligned",):
        fig.savefig(os.path.join(FIGS_DIR, name + ".png"), dpi=150)
        fig.savefig(os.path.join(FIGS_DIR, name + ".svg"))
    plt.close(fig)

    # ---------------- main train/val/gap in a 2-epoch window ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, arm in zip(axes, [ARMS[2], ARMS[4]]):
        pts = load_jsonl(os.path.join(RUNS_DIR, arm["dir"], "train_log.jsonl"))
        steps = np.array([p["step"] for p in pts])
        eps = np.array([p["epoch"] for p in pts])
        bnds = []
        prev = None
        for s, e in zip(steps, eps):
            if prev is not None and e != prev:
                bnds.append(int(s))
            prev = e
        if len(bnds) >= 2:
            w0, w1 = bnds[-2], bnds[-1] + int(0.4 * (bnds[-1] - bnds[-2]))
        else:
            w0, w1 = int(steps[0]), int(steps[-1])
        sel = [p for p in pts if w0 <= p["step"] <= w1]
        ax.plot([p["step"] for p in sel], [p["train_loss"] for p in sel],
                color=TRAIN, lw=1.3, ls="--", label="train loss")
        ax.plot([p["step"] for p in sel], [p["val_loss"] for p in sel],
                color=VAL, lw=1.8, label="val loss")
        axb = ax.twinx()
        axb.plot([p["step"] for p in sel], [p["gap"] for p in sel],
                 color=ANCHOR, lw=1.6, marker="o", ms=2.5, label="gap")
        axb.axhline(0, color=TEXT, lw=0.7)
        axb.set_ylabel("gap (nats)")
        for b in bnds:
            if w0 <= b <= w1:
                ax.axvline(b, color=BORDER, ls=":", lw=1.2)
        ax.set_title(f"{arm['label']} (window steps {w0}–{w1}, epoch boundary dashed)",
                     fontsize=10)
        ax.set_xlabel("global step")
        ax.set_ylabel("loss (nats)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = axb.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower left")
        style(ax)
        for s in axb.spines.values():
            s.set_color(BORDER)
        axb.tick_params(colors=TEXT)
        axb.yaxis.label.set_color(TEXT)
    fig.suptitle("main model: inside one epoch — train falls faster than val, gap opens; "
                 "new epoch data resets it", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for name in ("fig_main_within_epoch_train_val_gap",):
        fig.savefig(os.path.join(FIGS_DIR, name + ".png"), dpi=150)
        fig.savefig(os.path.join(FIGS_DIR, name + ".svg"))
    plt.close(fig)

    # ---------------- normalized within-epoch shape (all arms + toy) ----------------
    grid = np.linspace(0, 1, 21)
    norm_curves = {}
    for arm in ARMS[:5]:  # 0.25x..4x (8x has only 2 epochs)
        pts = load_jsonl(os.path.join(RUNS_DIR, arm["dir"], "train_log.jsonl"))
        if not pts:
            continue
        steps = np.array([p["step"] for p in pts])
        eps = np.array([p["epoch"] for p in pts])
        gaps = np.array([p["gap"] for p in pts])
        bnds = []
        prev = None
        for s, e in zip(steps, eps):
            if prev is not None and e != prev:
                bnds.append(int(s))
            prev = e
        ep_starts = [int(steps[0])] + bnds
        first = int(eps[0])
        curves = []
        for ep in sorted(set(int(e) for e in eps)):
            eidx = ep - first
            if eidx < 0 or eidx >= len(ep_starts):
                continue
            m = eps == ep
            x = steps[m] - ep_starts[eidx]
            y = gaps[m]
            if len(x) < 4:
                continue
            frac = (x - x.min()) / (x.max() - x.min() + 1e-9)
            g = np.interp(grid, frac, y)
            curves.append(g - g[0])
        if curves:
            norm_curves[arm["key"]] = (np.mean(curves, axis=0), arm, len(curves))
    # toy
    tcurves = []
    for ep in range(6, 16):
        if ep - 5 >= len(bnd) - 1:
            break
        s0, s1 = bnd[ep - 5], bnd[ep - 4]
        xs = np.array([s - s0 for s in range(s0, s1 + 1, 10)
                       if s in toy["train"] and s in toy["val"]], float)
        ys = np.array([toy["val"][s] - toy["train"][s] for s in range(s0, s1 + 1, 10)
                       if s in toy["train"] and s in toy["val"]], float)
        if len(xs) < 4:
            continue
        frac = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
        g = np.interp(grid, frac, ys)
        tcurves.append(g - g[0])
    if tcurves:
        norm_curves["toy"] = (np.mean(tcurves, axis=0), None, len(tcurves))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, (mean, arm, n) in norm_curves.items():
        if key == "toy":
            ax.plot(grid, mean, color="#C44E52", lw=2.4, marker="*", ms=6,
                    label=f"toy (epoch≈80 steps, ~8 pts/epoch, {n} epochs)")
        else:
            ax.plot(grid, mean, color=arm["color"], lw=2.0,
                    label=f"{key} ({arm['dir']}) — {n} epochs")
    ax.axhline(0, color=TEXT, lw=0.8)
    ax.set_xlabel("fraction of epoch (0 = epoch start, 1 = epoch end)")
    ax.set_ylabel("Δgap within epoch (gap − gap@epoch start, nats)")
    ax.set_title("within-epoch Δgap: same mechanism at every epoch length "
                 "(val pts/epoch 5 → 83)", fontsize=12)
    ax.legend(fontsize=8)
    style(ax)
    fig.tight_layout()
    for name in ("fig_within_epoch_normalized",):
        fig.savefig(os.path.join(FIGS_DIR, name + ".png"), dpi=150)
        fig.savefig(os.path.join(FIGS_DIR, name + ".svg"))
    plt.close(fig)

    print("[v12] wrote figures to", FIGS_DIR)
    for s in summary:
        print(f"  {s['key']:5s} epoch~{s['elen']:4d} steps  {s['npts']:3d} pts/epoch  "
              f"within-rise {s['rise']:+.3f}  epochs {s['n_epochs']}")


if __name__ == "__main__":
    main()
