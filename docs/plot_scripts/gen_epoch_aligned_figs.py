#!/usr/bin/env python3
"""Epoch-aligned sweep analysis: all runs trained to ~6-7 epochs with
epoch-anchored LR (--lr_schedule_epochs 6), so every run shares the SAME
LR-vs-epoch trajectory.  This isolates "same number of replays" from the
step-aligned sweep (gap@2000, where big shards simply saw fewer epochs).

Metrics per run (robust to the intra-epoch sawtooth):
  - gap_bnd[k] : gap at first eval of epoch k+1 (right after k full passes)
  - gap_mean[k]: mean gap over all evals inside epoch k
  - gap_peak[k]: max gap inside epoch k
Figures:
  1. gap_vs_shard_size_epoch_aligned.png  — gap after 6 passes vs shard size
  2. gap_vs_epoch_curves.png              — gap vs epoch number (aligned x-axis)
  3. epoch_aligned_train_val_gap.png      — train/val/gap curves (step axis)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_fixed")
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs", "epoch_scale")
os.makedirs(FIGS_DIR, exist_ok=True)

SWEEP = {
    0.25: "nglab0_25x_e6_fixed", 0.5: "nglab0_5x_e6_fixed", 0.75: "nglab0_75x_e6_fixed",
    1.0: "nglab1x_e6_fixed", 1.5: "nglab1_5x_e6_fixed", 2.0: "nglab2x_e6_fixed",
    2.5: "nglab2_5x_e6_fixed", 3.0: "nglab3x_e6_fixed",
}
GAP2000 = {
    0.25: 9.250, 0.5: 3.994, 0.75: 2.436, 1.0: 1.868, 1.5: 0.966, 2.0: 0.581,
    2.5: 0.813, 3.0: 0.500,
}
N_PASSES = 4
CMAP = plt.get_cmap("viridis")


def load_jsonl(path):
    pts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    pts.append(json.loads(line))
    return pts


def per_epoch_stats(log):
    by_ep = {}
    for p in log:
        by_ep.setdefault(p["epoch"], []).append(p)
    stats = {}
    for ep, pts in by_ep.items():
        stats[ep] = {
            "bnd": pts[0]["gap"],          # first eval of this epoch
            "mean": float(np.mean([p["gap"] for p in pts])),
            "peak": float(np.max([p["gap"] for p in pts])),
            "step_start": pts[0]["step"],
            "step_end": pts[-1]["step"],
            "lr": pts[0]["lr_mult"],
        }
    return stats


def main():
    data = {}
    for size, run_id in sorted(SWEEP.items()):
        log = load_jsonl(os.path.join(RUNS_DIR, run_id, "train_log.jsonl"))
        if not log:
            print(f"[ealigned] WARNING: no data for {run_id} — skipped")
            continue
        stats = per_epoch_stats(log)
        data[size] = {"run_id": run_id, "log": log, "stats": stats,
                      "final": log[-1]}
        if N_PASSES + 1 in stats:
            print(f"[ealigned] {size:>4}x {run_id:22s} "
                  f"gap@6passes bnd={stats[N_PASSES+1]['bnd']:+.3f} "
                  f"mean={stats[N_PASSES]['mean']:+.3f} peak={stats[N_PASSES]['peak']:+.3f}")

    if not data:
        raise SystemExit("no epoch-aligned data yet")

    sizes = sorted(data)
    cmap_v = {s: CMAP((np.log2(s) - np.log2(0.25)) / (np.log2(8) - np.log2(0.25))) for s in sizes}
    colors = {s: cmap_v[s] for s in sizes}

    # ---- Plot 1: gap after 6 passes vs shard size ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = np.array(sizes)
    yb = np.array([data[s]["stats"][N_PASSES + 1]["bnd"] for s in sizes])
    ym = np.array([data[s]["stats"][N_PASSES]["mean"] for s in sizes])
    ax.plot(xs, yb, "o-", ms=7, color="#C2185B", lw=1.5, label="gap @ first eval of pass 7 (bnd)")
    ax.plot(xs, ym, "^-", ms=7, color="#1565C0", lw=1.2, label="mean gap inside pass 6")
    for s in sizes:
        ax.annotate(f"{s:g}x", (s, data[s]["stats"][N_PASSES + 1]["bnd"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    y2 = np.array([GAP2000[s] for s in sizes])
    ax.plot(xs, y2, "s--", ms=6, color="gray", lw=1.2, label="gap @ 2000 steps (step-aligned)")
    ax.set_xscale("log")
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([f"{s:g}" for s in sizes])
    ax.set_xlabel("epoch shard size (× shard_00001)")
    ax.set_ylabel("gap (val − train)")
    ax.set_title("Epoch-aligned comparison: gap after 6 full epochs (LR anchored to 6 epochs)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out1 = os.path.join(FIGS_DIR, "gap_vs_shard_size_epoch_aligned.png")
    fig.savefig(out1, dpi=150)
    print(f"[ealigned] wrote {out1}")

    # ---- Plot 2: gap vs epoch number (aligned x-axis = passes) ----
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for s in sizes:
        st = data[s]["stats"]
        eps = sorted(st)
        ax.plot(eps, [st[e]["mean"] for e in eps], "o-", color=colors[s], ms=4,
                lw=1.2, label=f"{s:g}x")
        ax.plot(eps, [st[e]["peak"] for e in eps], "--", color=colors[s], lw=0.8, alpha=0.5)
    ax.axvline(N_PASSES, color="k", ls=":", lw=1)
    ax.set_xlabel("epoch (number of full passes of the train shard)")
    ax.set_ylabel("gap (val − train)")
    ax.set_title("Epoch-aligned gap trajectories (solid = mean gap in epoch, dashed = peak)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    out2 = os.path.join(FIGS_DIR, "gap_vs_epoch_curves.png")
    fig.savefig(out2, dpi=150)
    print(f"[ealigned] wrote {out2}")

    # ---- Plot 3: train/val/gap curves (step axis) ----
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    for s in sizes:
        log = data[s]["log"]
        st = [p["step"] for p in log]
        axes[0].plot(st, [p["train_loss"] for p in log], color=colors[s], lw=1.0, label=f"{s:g}x")
        axes[1].plot(st, [p["val_loss"] for p in log], color=colors[s], lw=1.0)
        axes[2].plot(st, [p["gap"] for p in log], color=colors[s], lw=1.0)
    for ax_i, yl in zip(axes, ["train loss", "val loss", "gap"]):
        ax_i.set_ylabel(yl)
        ax_i.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=4)
    axes[2].set_xlabel("step")
    axes[0].set_title("Epoch-aligned sweep (6 epochs, epoch-anchored LR): train / val / gap")
    fig.tight_layout()
    out3 = os.path.join(FIGS_DIR, "epoch_aligned_train_val_gap.png")
    fig.savefig(out3, dpi=150)
    print(f"[ealigned] wrote {out3}")

    # ---- summary table ----
    print("\n=== summary: gap after 6 full passes (bnd = first eval of pass 7) ===")
    print(f"{'size':>5} | {'gap6_bnd':>8} | {'gap6_mean':>9} | {'gap6_peak':>9} | {'step':>6} | {'gap@2000':>9}")
    for s in sizes:
        d = data[s]
        print(f"{s:>5g} | {d['stats'][N_PASSES+1]['bnd']:>+8.3f} | "
              f"{d['stats'][N_PASSES]['mean']:>+9.3f} | {d['stats'][N_PASSES]['peak']:>+9.3f} | "
              f"{d['stats'][N_PASSES+1]['step_start']:>6} | {GAP2000[s]:>+9.3f}")


if __name__ == "__main__":
    main()
