#!/usr/bin/env python3
"""Shard-size sweep analysis: does gap shrink as epoch/shard size grows?

Reads the 12 sweep runs (0.25x..8x, v10 fixed-val, input injection) and writes:
  1. dose_response_gap2000.png   — gap@2000 vs shard size (log-x + power-law fit)
  2. gap_vs_epochs.png           — gap vs epoch number (scaled x-axis: epochs elapsed)
  3. sweep_train_val_gap.png     — full train/val/gap curves, step axis

Usage: python3 docs/plot_scripts/gen_shard_sweep_figs.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_fixed")
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs_epoch_scale")
os.makedirs(FIGS_DIR, exist_ok=True)

# size multiplier -> run_id
SWEEP = {
    0.25: "nglab0_25x_input_fv_fixed",
    0.5:  "nglab0_5x_input_fv_fixed",
    0.75: "nglab0_75x_input_fv_fixed",
    1.0:  "nglab1x_v10_input_fixed",
    1.5:  "nglab1_5x_input_fv_fixed",
    2.0:  "nglab2x_input_v10_fv_fixed",
    2.5:  "nglab2_5x_input_fv_v3_fixed",
    3.0:  "nglab3x_input_fv_v3_fixed",
    4.0:  "nglab4x_input_fv_v3_fixed",
    5.0:  "nglab5x_input_fv_fixed",
    6.0:  "nglab6x_input_fv_fixed",
    8.0:  "nglab8x_input_fv_fixed",
}
CMAP = plt.get_cmap("viridis")

# Fair-LR verification reruns (max_steps=2000, same LR schedule as the sweep):
# 2.5x/3x/4x were originally extended to 3200/3800/5000 steps, stretching the
# LR schedule; the _v3 reruns fix that confound for the gap@2000 comparison.
V3 = {
    2.5: "nglab2_5x_input_fv_v3_fixed",
    3.0: "nglab3x_input_fv_v3_fixed",
    4.0: "nglab4x_input_fv_v3_fixed",
}


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


def gap_at_step(log, step):
    for p in log:
        if p["step"] >= step:
            return p["gap"], p["step"]
    return None, None


def epoch_series(log):
    """(epoch_number, gap) at first eval of each epoch (scaled x-axis)."""
    out = {}
    for p in log:
        ep = p["epoch"]
        if ep not in out:
            out[ep] = p["gap"]
    return sorted(out.items())


def main():
    data = {}
    for size, run_id in sorted(SWEEP.items()):
        run_dir = os.path.join(RUNS_DIR, run_id)
        log = load_jsonl(os.path.join(run_dir, "train_log.jsonl"))
        if not log:
            print(f"[sweep] WARNING: no data for {run_id} (size {size}) — skipped")
            continue
        g, st = gap_at_step(log, 2000)
        data[size] = {
            "run_id": run_id,
            "log": log,
            "gap2000": g,
            "gap2000_step": st,
            "epochs": epoch_series(log),
            "final_gap": log[-1]["gap"],
            "final_step": log[-1]["step"],
        }
        gtxt = f"{g:+.3f}" if g is not None else "n/a"
        print(f"[sweep] {size:>4}x {run_id:24s} gap@{st}={gtxt} final_gap={log[-1]['gap']:+.3f} "
              f"epoch={log[-1]['epoch']}")
    # fair-LR verification overlay
    for size, rid in V3.items():
        run_dir = os.path.join(RUNS_DIR, rid)
        log = load_jsonl(os.path.join(run_dir, "train_log.jsonl"))
        if not log:
            print(f"[sweep] WARNING: no v3 data for {rid} (size {size})")
            continue
        g, st = gap_at_step(log, 2000)
        data[size]["v3_gap2000"] = g
        data[size]["v3_run_id"] = rid
        print(f"[sweep] v3 {size:>4}x {rid:24s} gap@{st}={g:+.3f} (fair-LR)")

    if not data:
        raise SystemExit("no sweep data")

    sizes = sorted(data)
    colors = {s: CMAP((np.log2(s) - np.log2(0.25)) / (np.log2(8) - np.log2(0.25))) for s in sizes}

    # ---- Plot 1: dose-response gap@2000 vs shard size (log-x) ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs, ys = [], []
    for s in sizes:
        g = data[s]["gap2000"]
        if g is None:
            continue
        xs.append(s)
        ys.append(g)
        ax.plot(s, g, "o", ms=8, color=colors[s], zorder=3)
        ax.annotate(f"{s:g}x", (s, g), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
        if "v3_gap2000" in data[s] and data[s]["v3_gap2000"] is not None:
            g3 = data[s]["v3_gap2000"]
            ax.plot(s, g3, "D", ms=7, mfc="none", mec=colors[s], zorder=4)
    xs, ys = np.array(xs), np.array(ys)
    mask = ys > 0.005
    if mask.sum() >= 3:
        lx, ly = np.log(xs[mask]), np.log(ys[mask])
        k, b = np.polyfit(lx, ly, 1)
        xline = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 100)
        ax.plot(xline, np.exp(b) * xline ** k, "--", color="gray", lw=1,
                label=f"power law: gap ∝ size$^{{{k:.2f}}}$")
    ax.set_xscale("log")
    ax.set_xlabel("epoch shard size (× shard_00001)")
    ax.set_ylabel("gap @ 2000 steps (val − train)")
    if any("v3_gap2000" in data[s] for s in data):
        ax.plot([], [], "D", ms=7, mfc="none", mec="gray", label="_nolegend_")
        ax.text(0.99, 0.02, "◇ = fair-LR rerun (max_steps=2000)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="gray")
    ax.set_title("Dose–response: larger epoch shard → smaller gap (v10, fixed-val, seed 42)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out1 = os.path.join(FIGS_DIR, "dose_response_gap2000.png")
    fig.savefig(out1, dpi=150)
    print(f"[sweep] wrote {out1}")

    # ---- Plot 2: gap vs epochs elapsed (scaled x-axis) ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for s in sizes:
        eps = data[s]["epochs"]
        if not eps:
            continue
        ep_x = [e for e, _ in eps]
        ep_y = [g for _, g in eps]
        ax.plot(ep_x, ep_y, "-o", ms=3, lw=1.2, color=colors[s],
                label=f"{s:g}x (obs. ~{data[s]['final_step']//max(1,ep_x[-1])} st/epoch)")
    ax.set_xlabel("epochs elapsed (observed, from log)")
    ax.set_ylabel("gap")
    ax.set_title("Gap vs epochs elapsed — does epoch length matter beyond epoch count?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out2 = os.path.join(FIGS_DIR, "gap_vs_epochs.png")
    fig.savefig(out2, dpi=150)
    print(f"[sweep] wrote {out2}")

    # ---- Plot 3: train/val/gap full curves ----
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    for s in sizes:
        log = data[s]["log"]
        st = [p["step"] for p in log]
        axes[0].plot(st, [p["train_loss"] for p in log], color=colors[s], lw=1.0, label=f"{s:g}x")
        axes[1].plot(st, [p["val_loss"] for p in log], color=colors[s], lw=1.0)
        axes[2].plot(st, [p["gap"] for p in log], color=colors[s], lw=1.0)
    for ax, yl in zip(axes, ["train loss", "val loss", "gap"]):
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=4)
    axes[2].set_xlabel("step")
    axes[0].set_title("Shard-size sweep: train / val / gap (fixed-val, v10, seed 42)")
    fig.tight_layout()
    out3 = os.path.join(FIGS_DIR, "sweep_train_val_gap.png")
    fig.savefig(out3, dpi=150)
    print(f"[sweep] wrote {out3}")

    # ---- summary table ----
    print("\n=== summary (gap@2000) ===")
    for s in sizes:
        d = data[s]
        g = d["gap2000"]
        print(f"{s:>4g}x  {d['run_id']:24s} gap@2000 = {g:+.3f}" if g is not None
              else f"{s:>4g}x  {d['run_id']:24s} gap@2000 = n/a (running)")


if __name__ == "__main__":
    main()
