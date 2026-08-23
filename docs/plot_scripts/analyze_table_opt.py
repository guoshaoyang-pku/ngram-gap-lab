#!/usr/bin/env python3
"""Analyze the n-gram table optimizer ablation (input injection).

Auto-discovers data/runs/nglab1x_opt_* plus the RMSProp 1x reference
(nglab1x_v10_input_fixed). Seed-42 arms are compared at step 1000; s43/s44
repeats are summarized as mean ± std. Outputs a printed table and
docs/figs_v10/fig_table_opt.{svg,png}.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_fixed")
FIGS_DIR = os.environ.get("NGRAM_GAP_V10_FIGS_DIR", os.path.join(REPO_ROOT, "docs", "figs_v10"))
RMS_KEY = "bigram.layer_01.table_0.rms"

COLORS = {
    "rmsprop_1x": "#4CAF50", "rmsprop_2x": "#8BC34A", "rmsprop_4x": "#33691E",
    "adamw_090999": "#FF9800", "adamw_080950": "#F44336", "sgd_09": "#9C27B0",
}
ARM_LABEL = {
    "rmsprop_1x": "RMSProp 1x", "rmsprop_2x": "RMSProp 2x", "rmsprop_4x": "RMSProp 4x",
    "adamw_090999": "AdamW (0.9,0.999)", "adamw_080950": "AdamW (0.8,0.95)", "sgd_09": "SGD mom0.9",
}


def load_jsonl(path):
    pts = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pts.append(json.loads(line))
    return pts


def parse_run(run_id):
    """Return (arm, seed) from run ids like nglab1x_opt_adamw_090999_s43_fixed."""
    body = run_id.replace("nglab1x_opt_", "")
    if body.endswith("_fixed"):
        body = body[:-6]
    seed = 42
    if body.endswith("_s43"):
        seed = 43; body = body[:-4]
    elif body.endswith("_s44"):
        seed = 44; body = body[:-4]
    return body, seed


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    runs = {}
    for name in sorted(os.listdir(RUNS_DIR)):
        if not name.startswith("nglab1x_opt_"):
            continue
        d = os.path.join(RUNS_DIR, name)
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        arm, seed = parse_run(name)
        runs.setdefault(arm, {})[seed] = d
    # RMSProp 1x reference (the v10 standard run)
    ref = os.path.join(RUNS_DIR, "nglab1x_v10_input_fixed")
    if os.path.exists(os.path.join(ref, "summary.json")):
        runs.setdefault("rmsprop_1x", {})[42] = ref

    rows = []
    for arm in sorted(runs):
        label = ARM_LABEL.get(arm, arm)
        for seed, d in sorted(runs[arm].items()):
            summary = json.load(open(os.path.join(d, "summary.json")))
            tl = {p["step"]: p for p in load_jsonl(os.path.join(d, "train_log.jsonl"))}
            tn = {p["step"]: p for p in load_jsonl(os.path.join(d, "table_norm.jsonl"))}
            g1000 = tl.get(1000, {}).get("gap", float("nan"))
            n10 = tn.get(10, {}).get(RMS_KEY, float("nan"))
            n1000 = tn.get(1000, {}).get(RMS_KEY, float("nan"))
            nlast = tn.get(summary["steps"], {}).get(RMS_KEY, float("nan"))
            rows.append({
                "arm": arm, "label": label, "seed": seed, "steps": summary["steps"],
                "final_gap": summary["final_gap"], "gap1000": g1000,
                "norm10": n10, "norm1000": n1000, "norm_last": nlast,
                "growth": (n1000 - n10) / n10 * 100 if n10 else float("nan"),
            })

    hdr = (f"{'arm':20s} {'seed':>4s} {'steps':>5s} {'gap@1000':>9s} {'final_gap':>9s} "
           f"{'norm@10':>8s} {'norm@1000':>9s} {'growth':>8s}")
    print(hdr)
    for r in rows:
        print(f"{r['label']:20s} {r['seed']:4d} {r['steps']:5d} {r['gap1000']:+9.3f} "
              f"{r['final_gap']:+9.3f} {r['norm10']:8.4f} {r['norm1000']:9.4f} "
              f"{r['growth']:+7.1f}%")

    # seed-42 mean summary for multi-seed arms
    print("\n-- multi-seed summary (mean over s42/s43/s44) --")
    for arm in ["rmsprop_2x", "adamw_090999"]:
        vals = [r["norm1000"] for r in rows if r["arm"] == arm and r["seed"] in (42, 43, 44)]
        gaps = [r["gap1000"] for r in rows if r["arm"] == arm and r["seed"] in (42, 43, 44)]
        if len(vals) >= 2:
            print(f"{ARM_LABEL[arm]:20s} norm@1000 = {np.mean(vals):.4f} ± {np.std(vals):.4f} "
                  f"(n={len(vals)}) | gap@1000 = {np.mean(gaps):+.3f} ± {np.std(gaps):.3f}")

    # Figure: norm + gap panels (seed-42 solid lines; shaded std for multi-seed arms)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for arm, color in COLORS.items():
        if arm not in runs:
            continue
        seed_dirs = runs[arm]
        series = []
        for seed, d in seed_dirs.items():
            tl = {p["step"]: p for p in load_jsonl(os.path.join(d, "train_log.jsonl"))}
            tn = {p["step"]: p for p in load_jsonl(os.path.join(d, "table_norm.jsonl"))}
            steps = sorted(tl)
            series.append((steps, [tn.get(s, {}).get(RMS_KEY, float("nan")) for s in steps],
                           [tl[s]["gap"] for s in steps]))
        # align on step 10..max
        max_step = max(s[-1] for s, _, _ in series)
        all_steps = list(range(10, max_step + 1, 10))
        norms = np.full((len(series), len(all_steps)), np.nan)
        gaps = np.full((len(series), len(all_steps)), np.nan)
        for i, (steps, n, g) in enumerate(series):
            smap = dict(zip(steps, n)); gmap = dict(zip(steps, g))
            for j, st in enumerate(all_steps):
                norms[i, j] = smap.get(st, np.nan)
                gaps[i, j] = gmap.get(st, np.nan)
        mean_n, std_n = np.nanmean(norms, axis=0), np.nanstd(norms, axis=0)
        mean_g, std_g = np.nanmean(gaps, axis=0), np.nanstd(gaps, axis=0)
        label = ARM_LABEL.get(arm, arm)
        axes[0].plot(all_steps, mean_n, color=color, lw=2, label=label)
        if len(seed_dirs) > 1:
            axes[0].fill_between(all_steps, mean_n - std_n, mean_n + std_n, color=color, alpha=0.15)
        axes[1].plot(all_steps, mean_g, color=color, lw=2, label=label)
        if len(seed_dirs) > 1:
            axes[1].fill_between(all_steps, mean_g - std_g, mean_g + std_g, color=color, alpha=0.15)
    axes[0].set_title("Table norm (bigram layer-1 table RMS)")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("param RMS")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("Gap (val - train)")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("gap")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    for ext in ["svg", "png"]:
        out = os.path.join(FIGS_DIR, f"fig_table_opt.{ext}")
        fig.savefig(out, dpi=110, bbox_inches="tight")
        print(f"[fig] {out}")


if __name__ == "__main__":
    main()
