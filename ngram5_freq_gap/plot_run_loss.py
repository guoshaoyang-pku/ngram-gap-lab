#!/usr/bin/env python3
"""Pull training/validation loss JSONLs from a remote run dir and plot them.

Usage:
  python ngram5_freq_gap/plot_run_loss.py <run_dir> [--out plot.png] [--ssh-host ophis-gpu]
  python ngram5_freq_gap/plot_run_loss.py <local_run_dir>   # local dirs work too

Reads training_loss.jsonl + validation_loss.jsonl from the run dir (via
scp when the dir is remote) and writes a two-panel PNG: train loss (per
step, smoothed) and val loss (per interval).  Prints a small summary.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pull_file(host, run_dir, fname, local_dir):
    src = f"{host}:{run_dir}/{fname}"
    dst = Path(local_dir) / fname
    subprocess.run(["scp", "-o", "ConnectTimeout=25", "-o", "ServerAliveInterval=20",
                    src, str(dst)], check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run dir (local path or ssh host:path via --ssh-host)")
    ap.add_argument("--out", default="", help="output png path")
    ap.add_argument("--ssh-host", default="ophis-gpu")
    args = ap.parse_args()

    run_dir = args.run_dir
    remote = not os.path.isdir(run_dir)
    host = args.ssh_host if remote else None

    tmp = Path(tempfile.mkdtemp(prefix="lossplot_"))
    train_p = Path(run_dir) / "training_loss.jsonl"
    val_p = Path(run_dir) / "validation_loss.jsonl"
    if remote:
        train_p = pull_file(host, run_dir, "training_loss.jsonl", tmp)
        val_p = pull_file(host, run_dir, "validation_loss.jsonl", tmp)

    steps, losses = [], []
    with open(train_p) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            steps.append(rec.get("updates_completed", rec["step"]))
            losses.append(rec["train_loss"])
    vsteps, vloss = [], []
    with open(val_p) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            vsteps.append(rec["step"])
            vloss.append(rec["val_loss"])

    if not steps:
        print("no training_loss.jsonl records found")
        sys.exit(1)

    out = args.out or (Path(run_dir).name + "_loss.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    ax1.plot(steps, losses, lw=0.8, alpha=0.7, label="train (per step)")
    if vsteps:
        ax1.plot(vsteps, vloss, "o-", ms=4, label="val")
    ax1.set_xlabel("step"); ax1.set_ylabel("loss"); ax1.legend()
    ax1.set_title(f"{Path(run_dir).name}  (n_steps={max(steps):,})")
    ax1.grid(alpha=0.3)

    ax2.plot(vsteps, vloss, "o-", ms=4)
    ax2.set_xlabel("step"); ax2.set_ylabel("val loss")
    ax2.set_title("val loss (unseen shard 06542 stream)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"steps={len(steps):,}  last_train={losses[-1]:.4f}  "
          f"val_points={len(vsteps)}  last_val={vloss[-1]:.4f}  -> {out}")


if __name__ == "__main__":
    main()
