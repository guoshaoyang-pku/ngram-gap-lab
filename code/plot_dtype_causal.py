#!/usr/bin/env python3
"""Plot train/val/gap curves: a precision arm and causal controls."""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("NGLAB_RUNS_DIR", REPO_ROOT / "data" / "runs_fixed"))
REFERENCE_RUNS = Path(os.environ.get("NGLAB_REFERENCE_RUNS_DIR", RUNS))
FIGS_DIR = Path(os.environ.get("NGLAB_FIG_DIR", REPO_ROOT / "docs" / "figs" / "main"))
FP32_RUN = os.environ.get("NGLAB_FP32_RUN", "nglab1x_v10_input_fixed")
BF16_RUN = os.environ.get("NGLAB_BF16_RUN", "nglab1x_v10_input_bf16_fixed")

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows

def curve(name, run_dir):
    if not name.endswith("_fixed"):
        raise ValueError(f"non-canonical run id {name!r}; expected a *_fixed run")
    p = Path(run_dir) / name / "train_log.jsonl"
    if not p.exists():
        return None
    return load(p)

def plot_pair(ax, rows_a, rows_b, label_a, label_b, field, ylab):
    xs = [r["step"] for r in rows_a]
    ax.plot(xs, [r[field] for r in rows_a], "-", label=label_a, lw=1.5)
    if rows_b:
        ax.plot([r["step"] for r in rows_b], [r[field] for r in rows_b], "--", label=label_b, lw=1.5)
    ax.set_xlabel("step"); ax.set_ylabel(ylab); ax.legend(); ax.grid(alpha=0.3)

# ---- Panel 1: fp32 vs bf16 (main input arm, 1000 steps) ----
fp32 = curve(FP32_RUN, REFERENCE_RUNS)
bf16 = curve(BF16_RUN, RUNS)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
if fp32 and bf16:
    plot_pair(axes[0], fp32, bf16, "fp32", "bf16", "train_loss", "train loss")
    plot_pair(axes[1], fp32, bf16, "fp32", "bf16", "val_loss", "val loss")
    plot_pair(axes[2], fp32, bf16, "fp32", "bf16", "gap", "gap (val-train)")
fig.suptitle("Input-injection arm: fp32 vs bf16 (same seed 42, 1000 steps)")
fig.tight_layout()
FIGS_DIR.mkdir(parents=True, exist_ok=True)
out = FIGS_DIR / "fig_dtype_compare.png"
fig.savefig(out, dpi=130)
print("saved", out)

# ---- Panel 2: causal arms vs control (fp32) ----
control = fp32
arms = {
    "reset_table e2": ("nglab1x_input_reset_e2_fixed", "reset_table", 2),
    "reset_table e1": ("nglab1x_input_reset_e1_fixed", "reset_table", 1),
    "mask_readout e1": ("nglab1x_input_mask_e1_fixed", "mask_readout", 1),
    "freeze_table e1": ("nglab1x_input_freeze_table_e1_fixed", "freeze_table", 1),
    "freeze_backbone e1": ("nglab1x_input_freeze_backbone_e1_fixed", "freeze_backbone", 1),
}
fig2, ax2 = plt.subplots(figsize=(7.5, 5))
if control:
    ax2.plot([r["step"] for r in control], [r["gap"] for r in control],
             "-", color="k", lw=2, label="control (no intervention)")
for label, (name, _, ep) in arms.items():
    rows = curve(name, RUNS)
    if rows is None:
        print(f"missing {name}")
        continue
    ax2.plot([r["step"] for r in rows], [r["gap"] for r in rows],
             "-", lw=1.5, label=f"{label}")
    # mark intervention point
    ep_step = None
    for r in rows:
        if r["epoch"] > ep:
            ep_step = r["step"]; break
    if ep_step:
        ax2.axvline(ep_step, color="gray", ls=":", lw=0.8)
ax2.set_xlabel("step"); ax2.set_ylabel("gap (val-train)")
ax2.set_title("Causal arms (input injection) vs control")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
fig2.tight_layout()
out2 = FIGS_DIR / "fig_causal_arms.png"
fig2.savefig(out2, dpi=130)
print("saved", out2)
