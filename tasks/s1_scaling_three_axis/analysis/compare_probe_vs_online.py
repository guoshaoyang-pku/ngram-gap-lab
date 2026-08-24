#!/usr/bin/env python3
"""ngram-gap-lab · fixed-probe vs online-loss gap comparison (no reruns).

Reads existing run logs under data/runs_scaling/<run>/:
  - fixed_train_loss.jsonl : fixed-train-probe gap (diagnostic only)
  - train_log.jsonl        : online per-step gap (val_loss - train_loss)

Produces:
  - figs/epoch_gap_probe_vs_online.png : probe vs online overlaid, fixed-step
                                         alignment, both module, all L + L1/L4 zoom
  - figs/epoch_gap_metric_comparison.csv : per-run final gap under both metrics

Purpose: show that (a) endpoints agree, (b) the sawtooth at epoch boundaries is
a fixed-probe artifact (probe = first 4 train batches, re-read every replay).
"""
import json
import os
import sys
import glob
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RUNS_DIR = os.environ.get(
    "NGLAB_SCALING_RUNS_DIR", os.path.join(REPO_ROOT, "data", "runs_scaling"))
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "appendices",
                        "s1_scaling_three_axis", "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

EPB = {"L1": 42, "L2": 84, "L3": 168, "L4": 337}
MOD_COLOR = {
    "bigram": "#9C27B0", "trigram": "#FF9800", "both": "#2196F3", "nogram": "#4CAF50",
}
MOD_LABEL = {"bigram": "bigram", "trigram": "trigram", "both": "both", "nogram": "no-ngram"}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def collect_series():
    """Return {run_id: {'probe': [(step,gap)], 'online': [(step,gap)], 'L','mod'}}."""
    out = {}
    for run_dir in sorted(glob.glob(os.path.join(RUNS_DIR, "ep_*_fixed"))):
        if not os.path.isdir(run_dir):
            continue
        run_id = os.path.basename(run_dir)[:-len("_fixed")]
        L = mod = align = None
        for k in EPB:
            if f"_{k}_" in run_id:
                L = k
                break
        for m in MOD_COLOR:
            if f"_{m}_" in run_id:
                mod = m
                break
        if run_id.endswith("_fs"):
            align = "fs"
        elif run_id.endswith("_fe"):
            align = "fe"
        if not (L and mod and align):
            continue
        fp = load_jsonl(os.path.join(run_dir, "fixed_train_loss.jsonl"))
        tl = load_jsonl(os.path.join(run_dir, "train_log.jsonl"))
        if not fp or not tl:
            continue
        probe = [(int(r["step"]), float(r["fixed_val_loss"]) - float(r["fixed_train_loss"])) for r in fp]
        online = [(int(r["step"]), float(r["val_loss"]) - float(r["train_loss"])) for r in tl]
        out[run_id] = {"probe": probe, "online": online, "L": L, "mod": mod, "align": align}
    return out


def main():
    runs = collect_series()
    if not runs:
        print("no runs found")
        return
    print(f"found {len(runs)} runs")

    # ---- Fig 1: fixed-step alignment, both module, all L ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, metric, title in ((axes[0], "probe", "fixed train-probe (diagnostic)"),
                              (axes[1], "online", "online train loss (train_log.jsonl)")):
        for L in ["L1", "L2", "L3", "L4"]:
            run_id = f"ep_{L}_both_fs"
            if run_id not in runs:
                continue
            pts = runs[run_id][metric]
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    label=f"{L}", lw=1.6, marker=".", markersize=3, alpha=0.9)
            # mark epoch boundaries
            for b in range(EPB[L], 1001, EPB[L]):
                ax.axvline(b, color="gray", lw=0.5, alpha=0.4, zorder=0)
        ax.set_xlabel("step")
        ax.set_ylabel("gap (val − train)")
        ax.set_title(f"{title}\nfixed-step · both")
        ax.legend(title="epoch len", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Epoch-length scaling: probe gap vs online gap (same runs, no rerun)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_gap_probe_vs_online.png"), dpi=150)
    plt.close(fig)
    print(f"saved {FIGS_DIR}/epoch_gap_probe_vs_online.png")

    # ---- Fig 2: L1 vs L4 zoom, both, both metrics overlaid ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, L in ((axes[0], "L1"), (axes[1], "L4")):
        run_id = f"ep_{L}_both_fs"
        if run_id not in runs:
            continue
        for metric, style, label in (("probe", "-", "probe"),
                                     ("online", "--", "online")):
            pts = runs[run_id][metric]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                    label=label, lw=1.4, alpha=0.95)
        for b in range(EPB[L], 1001, EPB[L]):
            ax.axvline(b, color="gray", lw=0.6, alpha=0.5)
        ax.set_title(f"{L} both fs (epoch={EPB[L]})")
        ax.set_xlabel("step")
        ax.set_ylabel("gap")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Sawtooth is a fixed-probe artifact (probe = first train batches, re-read each replay)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_gap_probe_vs_online_L1L4_zoom.png"), dpi=150)
    plt.close(fig)
    print(f"saved {FIGS_DIR}/epoch_gap_probe_vs_online_L1L4_zoom.png")

    # ---- CSV: final gap under both metrics, per run ----
    rows = []
    for run_id, r in sorted(runs.items()):
        pf = r["probe"][-1][1]
        of = r["online"][-1][1]
        # shared-step correlation
        pmap = {s: g for s, g in r["probe"]}
        omap = {s: g for s, g in r["online"]}
        ss = sorted(set(pmap) & set(omap))
        xs = [omap[s] for s in ss]
        ys = [pmap[s] for s in ss]
        n = len(xs)
        corr = float("nan")
        if n > 2:
            mx = sum(xs) / n
            my = sum(ys) / n
            cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / n
            vx = sum((a - mx) ** 2 for a in xs) / n
            vy = sum((b - my) ** 2 for b in ys) / n
            corr = cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
        rows.append((run_id, r["L"], r["mod"], r["align"], pf, of, of - pf, corr, n))
    header = ["run", "L", "module", "align", "probe_gap_final", "online_gap_final",
              "online-probe", "corr_shared", "n_shared"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row))
    csv_path = os.path.join(FIGS_DIR, "epoch_gap_metric_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {csv_path}")

    print("\nper-run final gap (probe vs online):")
    for row in rows:
        print(f"  {row[0]:<26} probe {row[4]:+.4f}  online {row[5]:+.4f}  diff {row[6]:+.4f}  corr {row[7]:.3f}")


if __name__ == "__main__":
    main()
