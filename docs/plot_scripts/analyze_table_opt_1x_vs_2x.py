#!/usr/bin/env python3
"""Compare the same table-optimizer arms at 1x epoch vs 2x epoch (2000 steps).

1x arms: data/runs_fixed/nglab1x_opt_* (seed 42) + nglab1x_v10_input_fixed as RMSProp 1x.
2x arms: data/runs_fixed/nglab2x_opt_* + nglab2x_input_v10_fv_fixed as RMSProp 1x.
Writes docs/figs/table_opt/fig_table_opt_1x_vs_2x.{svg,png}.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.environ.get("NGLAB_RUNS_DIR", os.path.join(REPO_ROOT, "data", "runs_fixed"))
FIGS_DIR = os.environ.get("NGRAM_GAP_V10_FIGS_DIR", os.path.join(REPO_ROOT, "docs", "figs", "table_opt"))
RMS_KEY = "bigram.layer_01.table_0.rms"

PAIRS = [
    ("rmsprop_1x", "RMSProp 1x", "#4CAF50"),
    ("rmsprop_2x", "RMSProp 2x", "#8BC34A"),
    ("rmsprop_4x", "RMSProp 4x", "#33691E"),
]
B2_ARMS = ["rmsprop_1x_b2_09999", "rmsprop_2x_b2_09999",
           "rmsprop_4x_b2_09999", "rmsprop_2x_b2_099999"]
B2_LABEL = {
    "rmsprop_1x_b2_09999": "RMSProp 1x b2=.9999",
    "rmsprop_2x_b2_09999": "RMSProp 2x b2=.9999",
    "rmsprop_4x_b2_09999": "RMSProp 4x b2=.9999",
    "rmsprop_2x_b2_099999": "RMSProp 2x b2=.99999",
}


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def find(dirname, run_id):
    d = os.path.join(dirname, run_id)
    return d if os.path.exists(os.path.join(d, "summary.json")) else None


def series_for(path):
    tl = {p["step"]: p for p in load_jsonl(os.path.join(path, "train_log.jsonl"))}
    tn = {p["step"]: p for p in load_jsonl(os.path.join(path, "table_norm.jsonl"))}
    steps = sorted(tl)
    return steps, [tn.get(s, {}).get(RMS_KEY, np.nan) for s in steps], [tl[s].get("gap", np.nan) for s in steps]


def plot_pair(ax_norm, ax_gap, arm, label, color, refs, ls1="solid", ls2="dashed"):
    for is2x, (d, ls) in enumerate([(refs[arm][0], ls1), (refs[arm][1], ls2)]):
        if not d:
            continue
        steps, norms, gaps = series_for(d)
        tag = "2x" if is2x else "1x"
        ax_norm.plot(steps, norms, color=color, ls=ls, lw=2.2,
                     label=f"{label} · {tag}" if arm == "rmsprop_1x" or is2x == 0 else None)
        ax_gap.plot(steps, gaps, color=color, ls=ls, lw=2.2,
                    label=f"{label} · {tag}" if arm == "rmsprop_1x" or is2x == 0 else None)


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    refs = {}  # arm -> (1x_dir, 2x_dir)
    for arm, _, _ in PAIRS:
        one = "nglab1x_v10_input_fixed" if arm == "rmsprop_1x" else f"nglab1x_opt_{arm}_fixed"
        two = "nglab2x_input_v10_fv_fixed" if arm == "rmsprop_1x" else f"nglab2x_opt_{arm}_fixed"
        refs[arm] = (find(RUNS_DIR, one), find(RUNS_DIR, two))
    b2refs = {}
    for arm in B2_ARMS:
        b2refs[arm] = (None, find(RUNS_DIR, f"nglab2x_opt_{arm}_fixed"))

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for arm, label, color in PAIRS:
        plot_pair(axes[0, 0], axes[0, 1], arm, label, color, refs)
    for arm in B2_ARMS:
        d = b2refs[arm][1]
        if not d:
            continue
        steps, norms, gaps = series_for(d)
        axes[1, 0].plot(steps, norms, lw=2.0, label=B2_LABEL[arm])
        axes[1, 1].plot(steps, gaps, lw=2.0, label=B2_LABEL[arm])

    axes[0, 0].set_title("Norm · LR dose (solid 1x epoch / dashed 2x epoch)")
    axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("param RMS"); axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Gap · LR dose")
    axes[0, 1].axhline(0, color="k", lw=0.8)
    axes[0, 1].set_xlabel("step"); axes[0, 1].set_ylabel("gap"); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].set_title("Norm · beta2 sweep @ 2x epoch")
    axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("param RMS"); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set_title("Gap · beta2 sweep @ 2x epoch")
    axes[1, 1].axhline(0, color="k", lw=0.8)
    axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("gap"); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Table optimizer: 1x vs 2x epoch, 2000 steps (seed 42)", fontsize=14, y=1.0)
    fig.tight_layout()
    for ext in ["svg", "png"]:
        out = os.path.join(FIGS_DIR, f"fig_table_opt_1x_vs_2x.{ext}")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
