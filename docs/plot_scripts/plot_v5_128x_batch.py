#!/usr/bin/env python3
"""Plot the 128x standard-rerun batch (2026-08-29).

Reads data/runs_fixed/*_128x* and writes v5-128x figures to docs/figs/main/.
Figures:
  fig_v5_128x_injection_curves.png      M2 injection train/val/gap curves
  fig_v5_128x_injection_bars.png        M2 final gap bars (vs 2x historical)
  fig_v5_128x_dose_gap.png              M5 dose final gap vs dose
  fig_v5_128x_causal_gap.png            causal 6-arm final gap bars
  fig_v5_128x_rowwidth_gap.png          X2 row width final gap
  fig_v5_128x_optimizer_gap.png         X1 optimizer final gap bars
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/Users/guoshaoyang/Desktop/workdir/ngram-gap-lab"
DATA = os.path.join(ROOT, "data/runs_fixed")
OUT = os.path.join(ROOT, "docs/figs/main")
os.makedirs(OUT, exist_ok=True)

def load(run_id):
    d = os.path.join(DATA, run_id + "_fixed")
    sp = os.path.join(d, "summary.json")
    s = json.load(open(sp))
    rows = []
    tp = os.path.join(d, "train_log.jsonl")
    if os.path.exists(tp):
        for line in open(tp):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return s, rows

def series(rows, key, default=None):
    return [(r.get("step"), r.get(key, default)) for r in rows]

def plot_curves(ax, rows, key, color, label, lw=1.2):
    steps = [r["step"] for r in rows]
    vals = [r.get(key) for r in rows]
    ax.plot(steps, vals, color=color, label=label, lw=lw, alpha=0.9)

# ---------- M2 injection curves ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
arms = [("input", "#2d6f9f"), ("y", "#c4493d"), ("v", "#b67524"), ("nogram", "#686d73")]
for name, color in arms:
    rid = f"nglab1x_{name}_v5_128x_freq10"
    s, rows = load(rid)
    if not rows:
        continue
    plot_curves(axes[0], rows, "train_loss", color, name)
    plot_curves(axes[1], rows, "val_loss", color, name)
    gaps = [(r["step"], r["train_loss"] is not None and r["val_loss"] - r["train_loss"]) for r in rows]
    steps = [g[0] for g in gaps]
    vals = [g[1] for g in gaps if g[1] is not None]
    axes[2].plot(steps, vals, color=color, label=name, lw=1.2, alpha=0.9)
for ax, t in zip(axes, ["online train loss", "fixed val loss", "gap (val - train)"]):
    ax.set_title(t)
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
fig.suptitle("M2 injection-point ablation @ table LR 128x (2000 steps, seed 42)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_v5_128x_injection_curves.png"), dpi=150)
plt.close(fig)
print("wrote fig_v5_128x_injection_curves.png")

# ---------- M2 final gap bars (vs 2x) ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
names = ["input", "y", "v", "nogram"]
g128 = []
for name in names:
    s, _ = load(f"nglab1x_{name}_v5_128x_freq10")
    g128.append(s.get("final_gap", float("nan")))
# 2x historical (from summary of old runs)
g2 = [5.741, 3.640, 2.014, 0.245]
x = np.arange(len(names))
w = 0.35
ax.bar(x - w/2, g2, w, label="2x (historical)", color="#9aa3ad")
ax.bar(x + w/2, g128, w, label="128x (new standard)", color="#2d6f9f")
for i, v in enumerate(g128):
    ax.text(i + w/2, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(g2):
    ax.text(i - w/2, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("final gap @2000 (val - train)")
ax.set_title("M2 injection point: 2x vs 128x table LR")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_v5_128x_injection_bars.png"), dpi=150)
plt.close(fig)
print("wrote fig_v5_128x_injection_bars.png")

# ---------- M5 dose gap ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
doses = ["0_25x", "0_5x", "0_75x", "1_5x", "2x", "2_5x", "3x", "4x", "5x", "6x", "8x"]
labels = ["0.25x", "0.5x", "0.75x", "1.5x", "2x", "2.5x", "3x", "4x", "5x", "6x", "8x"]
gaps = []
for d in doses:
    s, _ = load(f"nglab{d}_input_v5_128x_freq10")
    gaps.append(s.get("final_gap", float("nan")))
ax.plot(range(len(labels)), gaps, "o-", color="#c4493d", lw=1.5)
for i, v in enumerate(gaps):
    ax.text(i, v + 0.25, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
ax.axhline(0.2273, color="#686d73", ls="--", lw=1, label="nogram (0.23)")
ax.set_xlabel("train shard dose")
ax.set_ylabel("final gap @2000")
ax.set_title("M5 dose scan @ table LR 128x (2000 steps, seed 42)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_v5_128x_dose_gap.png"), dpi=150)
plt.close(fig)
print("wrote fig_v5_128x_dose_gap.png")

# ---------- Causal 6-arm gap ----------
fig, ax = plt.subplots(figsize=(11, 4.5))
causal = [
    ("none", 2.724), ("freeze_table_e1", 3.452),
    ("freeze_backbone_e1", 1.230), ("hash_reseed_e1", 1.354),
    ("mask_low_f200_e1", 0.101), ("mask_high_f200_e1", 2.808),
]
names = [c[0] for c in causal]
vals = [c[1] for c in causal]
colors = ["#2d6f9f", "#b67524", "#b67524", "#b67524", "#0f766e", "#d97706"]
bars = ax.bar(range(len(names)), vals, color=colors)
for i, v in enumerate(vals):
    ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=9)
ax.axhline(2.724, color="#2d6f9f", ls="--", lw=0.8, label="none baseline 2.72")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("final gap @1000")
ax.set_title("Causal interventions @ table LR 128x (1000 steps, seed 42)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_v5_128x_causal_gap.png"), dpi=150)
plt.close(fig)
print("wrote fig_v5_128x_causal_gap.png")

# ---------- X2 row width gap ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
dims = [12, 48, 192, 768]
gaps = []
for dim in dims:
    s, _ = load(f"ctbl_dim{dim}_input_v5_128x")
    gaps.append(s.get("final_gap", float("nan")))
ax.semilogx(dims, gaps, "o-", color="#2d6f9f", lw=1.5)
for x_, v in zip(dims, gaps):
    ax.text(x_, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(dims); ax.set_xticklabels([str(d) for d in dims])
ax.set_xlabel("table row width (n_embd dims)")
ax.set_ylabel("final gap @1000")
ax.set_title("X2 table row width @ table LR 128x (1000 steps)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_v5_128x_rowwidth_gap.png"), dpi=150)
plt.close(fig)
print("wrote fig_v5_128x_rowwidth_gap.png")

# ---------- X1 optimizer gap ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
opts = ["rms", "adamw", "sgd_m0"]
names = ["RMSProp", "AdamW", "SGD (m=0)"]
gaps = []
for o in opts:
    s, _ = load(f"optv5c_{o}_s128x")
    gaps.append(s.get("final_gap", float("nan")))
bars = ax.bar(range(3), gaps, color=["#2d6f9f", "#b67524", "#686d73"])
for i, v in enumerate(gaps):
    ax.text(i, v + 0.08, f"{v:.2f}", ha="center", fontsize=10)
ax.set_xticks(range(3)); ax.set_xticklabels(names)
ax.set_ylabel("final gap @1000")
ax.set_title("X1 table optimizer @ table LR 128x (1000 steps)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_v5_128x_optimizer_gap.png"), dpi=150)
plt.close(fig)
print("wrote fig_v5_128x_optimizer_gap.png")
print("ALL DONE")
