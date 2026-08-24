#!/usr/bin/env python3
"""fp32 vs bf16 (same hyperparams) train/val/gap comparison."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("NGLAB_RUNS_DIR", REPO_ROOT / "data" / "runs_fixed"))
REFERENCE_RUNS = Path(os.environ.get("NGLAB_REFERENCE_RUNS_DIR", RUNS))
FIGS_DIR = Path(os.environ.get("NGLAB_FIG_DIR", REPO_ROOT / "docs" / "figs"))
FP32_RUN = os.environ.get("NGLAB_FP32_RUN", "nglab1x_v10_input_fixed")
BF16_RUN = os.environ.get("NGLAB_BF16_RUN", "nglab1x_v10_input_bf16_fixed")


def load(name, base):
    if not name.endswith("_fixed"):
        raise ValueError(f"non-canonical run id {name!r}; expected a *_fixed run")
    run_dir = base / name
    summary_path = run_dir / "summary.json"
    path = run_dir / "train_log.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"missing run summary: {summary_path}; set NGLAB_*_RUNS_DIR or NGLAB_*_RUN"
        )
    with summary_path.open() as handle:
        summary = json.load(handle)
    if summary.get("run_id") != name:
        raise ValueError(
            f"run contract mismatch: directory={name!r}, summary.run_id="
            f"{summary.get('run_id')!r}"
        )
    config = summary.get("config", {})
    if config.get("val_interval_steps") != 10:
        raise ValueError(f"{name}: val_interval_steps must be 10")
    if config.get("table_betas") != [0.0, 0.99]:
        raise ValueError(f"{name}: table_betas must be [0.0, 0.99]")
    if config.get("table_lr_scale") != 2.0:
        raise ValueError(f"{name}: table_lr_scale must be 2.0")
    if not path.exists():
        raise FileNotFoundError(
            f"missing run log: {path}; set NGLAB_*_RUNS_DIR or NGLAB_*_RUN"
        )
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows


fp32 = load(FP32_RUN, REFERENCE_RUNS)
bf16 = load(BF16_RUN, RUNS)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, field, ylab in zip(axes, ["train_loss", "val_loss", "gap"],
                           ["train loss", "val loss", "gap (val-train)"]):
    ax.plot([r["step"] for r in fp32], [r[field] for r in fp32], "-", color="tab:blue", lw=1.6, label="fp32")
    ax.plot([r["step"] for r in bf16], [r[field] for r in bf16], "--", color="tab:red", lw=1.6, label="bf16 (same HP)")
    ax.set_xlabel("step"); ax.set_ylabel(ylab); ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("Input-injection arm 1000 steps: fp32 vs bf16, identical hyperparams (seed 42)")
fig.tight_layout()
FIGS_DIR.mkdir(parents=True, exist_ok=True)
out = FIGS_DIR / "fig_fp32_vs_bf16_samehp.png"
fig.savefig(out, dpi=130)
print("saved", out)
