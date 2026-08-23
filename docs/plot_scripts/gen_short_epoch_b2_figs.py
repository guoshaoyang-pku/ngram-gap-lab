#!/usr/bin/env python3
"""Worker B: short-epoch x beta2 comparison figures (v11, 2026-08-07).

Compares gap-vs-step curves for 0.25x / 0.5x epoch at beta2 = 0.999 (§10 refs)
vs beta2 = 0.99 (nglab{025x,05x}_b2_099), plus staircase-clarity metrics.

Usage: python3 docs/plot_scripts/gen_short_epoch_b2_figs.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_fixed")
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs_v11")
os.makedirs(FIGS_DIR, exist_ok=True)

# key | label | color | run dir | beta2
ARMS = [
    {"key": "0.25x", "b2": "0.999", "label": "0.25x · b2=0.999", "color": "#9C27B0", "dir": "nglab0_25x_input_fv_fixed"},
    {"key": "0.25x", "b2": "0.99",  "label": "0.25x · b2=0.99",  "color": "#C44E52", "dir": "nglab025x_b2_099_fixed"},
    {"key": "0.5x",  "b2": "0.999", "label": "0.5x · b2=0.999",  "color": "#4CAF50", "dir": "nglab0_5x_input_fv_fixed"},
    {"key": "0.5x",  "b2": "0.99",  "label": "0.5x · b2=0.99",   "color": "#DD8452", "dir": "nglab05x_b2_099_fixed"},
]


def load_jsonl(path):
    pts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pts.append(json.loads(line))
    return pts


def epoch_boundaries(train_log):
    bnds, prev = [], None
    for p in train_log:
        ep = p.get("epoch")
        if ep is not None and prev is not None and ep != prev:
            bnds.append(p["step"])
        prev = ep if ep is not None else prev
    return bnds


def staircase_metrics(train_log, boundaries):
    """Quantify staircase clarity from gap-vs-step data.

    For each epoch segment: within-epoch drift (gap change inside the epoch,
    per step) vs boundary jump (gap change at the epoch transition). A clear
    staircase has most of the gap increase concentrated at boundaries; a blurry
    monotone curve has it spread inside epochs.
    """
    if not train_log:
        return {}
    steps = np.array([p["step"] for p in train_log])
    gaps = np.array([p["gap"] for p in train_log])
    eps = np.array([p["epoch"] for p in train_log])
    within = []   # |dgap| per step inside epoch
    atbound = []  # |dgap| across boundary
    prev_ep = eps[0]
    prev_gap = gaps[0]
    prev_step = steps[0]
    for s, g, e in zip(steps, gaps, eps):
        if e == prev_ep and s > prev_step:
            within.append(abs(g - prev_gap) / max(1, s - prev_step))
        elif e != prev_ep:
            atbound.append(abs(g - prev_gap))
        prev_ep, prev_gap, prev_step = e, g, s
    w = float(np.mean(within)) if within else float("nan")
    b = float(np.mean(atbound)) if atbound else float("nan")
    return {
        "n_epochs": int(eps[-1]),
        "within_per_step": w,
        "boundary_jump": b,
        "clarity_ratio": (b / w) if w and w > 1e-9 else float("nan"),
    }


def main():
    series = {}
    for arm in ARMS:
        tl = load_jsonl(os.path.join(RUNS_DIR, arm["dir"], "train_log.jsonl"))
        if not tl:
            print(f"[v11] WARNING: no train_log for {arm['dir']} — skipped")
            continue
        bnds = epoch_boundaries(tl)
        m = staircase_metrics(tl, bnds)
        series[arm["dir"]] = {
            "steps": [p["step"] for p in tl],
            "gap": [p["gap"] for p in tl],
            "train": [p["train_loss"] for p in tl],
            "val": [p["val_loss"] for p in tl],
            "boundaries": bnds,
            "metrics": m,
            "arm": arm,
        }
        print(f"[v11] {arm['dir']}: final gap={tl[-1]['gap']:+.3f} "
              f"epochs={m.get('n_epochs')} within={m.get('within_per_step')} "
              f"boundary={m.get('boundary_jump')} ratio={m.get('clarity_ratio')}")

    if not series:
        raise SystemExit("[v11] no data yet")

    # --- fig 1: gap vs step, 4 curves, epoch boundaries marked ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for run, d in series.items():
        arm = d["arm"]
        ax.plot(d["steps"], d["gap"], lw=1.6, color=arm["color"], label=arm["label"])
        for b in d["boundaries"]:
            ax.axvline(b, color=arm["color"], alpha=0.12, lw=0.5)
    ax.set_xlabel("step"); ax.set_ylabel("gap (val − train, nats)")
    ax.set_title("short-epoch × beta2: gap curves (val every 10 steps, 2000 steps)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "short_epoch_b2_gap.png"), dpi=150)
    fig.savefig(os.path.join(FIGS_DIR, "short_epoch_b2_gap.svg"))
    plt.close(fig)

    # --- fig 2: per-epoch mean gap (epoch-aligned) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for run, d in series.items():
        arm = d["arm"]
        xs = [p["step"] for p in load_jsonl(os.path.join(RUNS_DIR, arm["dir"], "train_log.jsonl"))]
        ys = [p["gap"] for p in load_jsonl(os.path.join(RUNS_DIR, arm["dir"], "train_log.jsonl"))]
        ax.plot(xs, ys, lw=1.4, color=arm["color"], label=arm["label"])
    ax.set_xlabel("step"); ax.set_ylabel("gap (nats)")
    ax.set_title("short-epoch × beta2 (raw 10-step gap)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "short_epoch_b2_gap_raw.png"), dpi=150)
    plt.close(fig)

    # --- metrics table ---
    print("\n" + "=" * 90)
    print(f"{'run':<28} {'b2':>6} {'epochs':>7} {'within/step':>12} {'boundary':>10} {'clarity':>8}")
    for run, d in series.items():
        m = d["metrics"]
        print(f"{run:<28} {d['arm']['b2']:>6} {m.get('n_epochs', float('nan')):>7.0f} "
              f"{m.get('within_per_step', float('nan')):>12.4f} {m.get('boundary_jump', float('nan')):>10.3f} "
              f"{m.get('clarity_ratio', float('nan')):>8.2f}")


if __name__ == "__main__":
    main()
