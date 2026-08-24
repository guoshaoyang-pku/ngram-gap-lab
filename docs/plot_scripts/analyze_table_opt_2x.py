#!/usr/bin/env python3
"""Analyze the 2x-epoch table optimizer ablation (input injection, train shards 1,2).

Auto-discovers data/runs_fixed/nglab2x_opt_* plus the RMSProp 1x reference
(nglab2x_input_v10_fv_fixed, the §6 fixed-val run). Prints a table and writes
docs/figs/table_opt/fig_table_opt_2x.{svg,png}.
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

COLORS = {
    "rmsprop_1x": "#4CAF50", "rmsprop_2x": "#8BC34A", "rmsprop_4x": "#33691E",
    "rmsprop_1x_b2_09999": "#81D4FA", "rmsprop_2x_b2_09999": "#0288D1",
    "rmsprop_4x_b2_09999": "#01579B", "rmsprop_2x_b2_099999": "#004D40",
}
ARM_LABEL = {
    "rmsprop_1x": "RMSProp 1x (b2=.999)",
    "rmsprop_2x": "RMSProp 2x (b2=.999)",
    "rmsprop_4x": "RMSProp 4x (b2=.999)",
    "rmsprop_1x_b2_09999": "RMSProp 1x (b2=.9999)",
    "rmsprop_2x_b2_09999": "RMSProp 2x (b2=.9999)",
    "rmsprop_4x_b2_09999": "RMSProp 4x (b2=.9999)",
    "rmsprop_2x_b2_099999": "RMSProp 2x (b2=.99999)",
}
ORDER = ["rmsprop_1x", "rmsprop_2x", "rmsprop_4x",
         "rmsprop_1x_b2_09999", "rmsprop_2x_b2_09999",
         "rmsprop_4x_b2_09999", "rmsprop_2x_b2_099999"]


def load_jsonl(path):
    pts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pts.append(json.loads(line))
    return pts


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    runs = {}
    for name in sorted(os.listdir(RUNS_DIR)):
        if not name.startswith("nglab2x_opt_"):
            continue
        d = os.path.join(RUNS_DIR, name)
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        arm = name.replace("nglab2x_opt_", "")
        if arm.endswith("_fixed"):
            arm = arm[:-6]
        runs[arm] = d
    # RMSProp 1x reference (the §6 fixed-val run)
    ref = os.path.join(RUNS_DIR, "nglab2x_input_v10_fv_fixed")
    if os.path.exists(os.path.join(ref, "summary.json")):
        runs.setdefault("rmsprop_1x", ref)

    rows = []
    for arm in ORDER:
        d = runs.get(arm)
        if not d:
            continue
        summary = json.load(open(os.path.join(d, "summary.json")))
        tl = {p["step"]: p for p in load_jsonl(os.path.join(d, "train_log.jsonl"))}
        tn = {p["step"]: p for p in load_jsonl(os.path.join(d, "table_norm.jsonl"))}
        g500 = tl.get(500, {}).get("gap", float("nan"))
        g1000 = tl.get(1000, {}).get("gap", float("nan"))
        g2000 = tl.get(2000, {}).get("gap", float("nan"))
        n500 = tn.get(500, {}).get(RMS_KEY, float("nan"))
        n1000 = tn.get(1000, {}).get(RMS_KEY, float("nan"))
        n2000 = tn.get(2000, {}).get(RMS_KEY, float("nan"))
        rows.append({"arm": arm, "label": ARM_LABEL[arm], "steps": summary["steps"],
                     "final_gap": summary["final_gap"],
                     "g500": g500, "g1000": g1000, "g2000": g2000,
                     "n500": n500, "n1000": n1000, "n2000": n2000})

    hdr = (f"{'arm':26s} {'steps':>5s} {'gap@500':>8s} {'gap@1000':>9s} {'gap@2000':>9s} "
           f"{'norm@500':>9s} {'norm@1000':>9s} {'norm@2000':>9s}")
    print(hdr)
    for r in rows:
        print(f"{r['label']:26s} {r['steps']:5d} {r['g500']:+8.3f} {r['g1000']:+9.3f} "
              f"{r['g2000']:+9.3f} {r['n500']:9.4f} {r['n1000']:9.4f} {r['n2000']:9.4f}")

    # Figure: norm + gap + trajectory
    fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))
    for arm in ORDER:
        d = runs.get(arm)
        if not d:
            continue
        color = COLORS.get(arm, "#666")
        tl = {p["step"]: p for p in load_jsonl(os.path.join(d, "train_log.jsonl"))}
        tn = {p["step"]: p for p in load_jsonl(os.path.join(d, "table_norm.jsonl"))}
        steps = sorted(tl)
        norms = [tn.get(s, {}).get(RMS_KEY, np.nan) for s in steps]
        gaps = [tl[s].get("gap", np.nan) for s in steps]
        label = ARM_LABEL.get(arm, arm)
        axes[0].plot(steps, norms, color=color, lw=2.2, label=label)
        axes[1].plot(steps, gaps, color=color, lw=2.2, label=label)
        axes[2].plot(norms, gaps, color=color, lw=2.0, marker="o", ms=3, label=label)
        axes[2].annotate("", xy=(norms[-1], gaps[-1]), xytext=(norms[-4], gaps[-4]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    axes[0].set_title("2x epoch · Table norm vs step")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("param RMS"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("2x epoch · Gap (val - train) vs step")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("gap"); axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[2].set_title("2x epoch · Trajectory: norm -> gap")
    axes[2].set_xlabel("table norm"); axes[2].set_ylabel("gap"); axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.suptitle("Table optimizer ablation @ 2x epoch (train shards 1,2), 2000 steps",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    for ext in ["svg", "png"]:
        out = os.path.join(FIGS_DIR, f"fig_table_opt_2x.{ext}")
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
