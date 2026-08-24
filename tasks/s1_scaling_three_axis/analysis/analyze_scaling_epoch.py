#!/usr/bin/env python3
"""ngram-gap-lab · epoch-length scaling analysis (plan §3).

Reads data/runs_scaling/<run>/:
  - train_log.jsonl        : {step, epoch, train_loss, val_loss, gap}
  - exact_freq_loss.jsonl  : per-exact-f train/val/shared stats
  - summary.json           : run metadata (epoch_batches, module flags, betas)

Produces:
  - fig: online-gap vs step, per epoch length (fixed-step alignment)
  - fig: online-gap vs epoch, per epoch length (fixed-epoch alignment)
  - fig: delta-G (module - no-ngram) vs step / epoch
  - table: final online-gap per (L, module, alignment)

Usage: python3 docs/plot_scripts/analyze_scaling_epoch.py [runs_dir]
"""
import json
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
RUNS_DIR = os.environ.get(
    "NGLAB_SCALING_RUNS_DIR",
    os.environ.get("NGLAB_RUNS_DIR", os.path.join(REPO_ROOT, "data", "runs_scaling")),
)
if len(sys.argv) > 1:
    RUNS_DIR = sys.argv[1]
FIGS_DIR = os.path.join(
    REPO_ROOT, "docs", "appendices", "s1_scaling_three_axis", "figs"
)
os.makedirs(FIGS_DIR, exist_ok=True)

EPB = {"L1": 42, "L2": 84, "L3": 168, "L4": 337}  # L4 = full shard 1 (user 2026-08-24)
MODULES = {
    "bigram": {"bigram": 1, "trigram": 0, "label": "bigram-only", "color": "#9C27B0"},
    "trigram": {"bigram": 0, "trigram": 1, "label": "trigram-only", "color": "#FF9800"},
    "both": {"bigram": 1, "trigram": 1, "label": "both", "color": "#2196F3"},
    "nogram": {"bigram": 0, "trigram": 0, "label": "no-ngram", "color": "#4CAF50"},
}
ALIGN = {"fs": "fixed-step", "fe": "fixed-epoch"}
SEED_ORDER = (42, 43, 44)


def load_online(run_dir):
    path = os.path.join(run_dir, "train_log.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            out.append({
                "step": int(e["step"]),
                "epoch": float(e["epoch"]),
                "train": float(e["train_loss"]),
                "val": float(e["val_loss"]),
                "gap": float(e["gap"]),
            })
    return out


def load_summary(run_dir):
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def is_current_scaling_summary(summary, physical_id):
    config = summary.get("config", {})
    dense_monitor = (
        config.get("val_interval_steps") == 10
        and config.get("table_norm_interval_steps") == 10
        and not config.get("val_steps")
    )
    sparse_monitor = bool(config.get("val_steps"))
    return (
        summary.get("run_id") == physical_id
        and config.get("table_optimizer") == "rmsprop"
        and config.get("table_lr_scale") == 2.0
        and config.get("table_betas") == [0.0, 0.99]
        and (dense_monitor or sparse_monitor)
        and summary.get("compute_dtype") == "bf16"
        and summary.get("torch_compile") is True
    )


def canonical_run_dirs(runs_dir):
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue
        physical_id = os.path.basename(run_dir)
        if not physical_id.endswith("_fixed"):
            continue
        yield physical_id[:-len("_fixed")], physical_id, run_dir


def match_run(run_id):
    """Parse run_id into (L, module, alignment, seed)."""
    m = {"seed": 42}
    suffix = None
    if run_id.endswith(tuple(f"_s{s}" for s in SEED_ORDER if s != 42)):
        base, suffix = run_id.rsplit("_s", 1)
        run_id = base
        try:
            m["seed"] = int(suffix)
        except ValueError:
            return {}
    for k in EPB:
        if f"_{k}_" in run_id:
            m["L"] = k
            break
    for mod in MODULES:
        if f"_{mod}_" in run_id:
            m["module"] = mod
            break
    for al in ALIGN:
        if run_id.endswith(f"_{al}"):
            m["align"] = al
            break
    return m


def collect(runs_dir):
    """Return {run_id: {online, summary, meta}} for all scaling epoch runs."""
    out = {}
    legacy_count = 0
    rejected_count = 0
    for run_id, physical_id, run_dir in canonical_run_dirs(runs_dir):
        meta = match_run(run_id)
        if not meta.get("L") or not meta.get("module") or not meta.get("align"):
            continue
        summary = load_summary(run_dir)
        if not is_current_scaling_summary(summary, physical_id):
            rejected_count += 1
            continue
        if int(summary.get("seed", meta.get("seed", 42))) != int(meta.get("seed", 42)):
            rejected_count += 1
            continue
        online = load_online(run_dir)
        if not online:
            continue
        out[run_id] = {
            "online": online,
            "summary": summary,
            "meta": meta,
            "run_id": physical_id,
            "run_dir": run_dir,
        }
    for run_dir in glob.glob(os.path.join(runs_dir, "*")):
        if os.path.isdir(run_dir) and not os.path.basename(run_dir).endswith("_fixed"):
            legacy_count += 1
    if legacy_count:
        print(f"ignored {legacy_count} non-canonical scaling directories (expected *_fixed)")
    if rejected_count:
        print(f"ignored {rejected_count} scaling directories with an invalid run contract")
    return out


def final_gap(run):
    online = run["online"]
    if not online:
        return None
    return online[-1]["gap"]


def base_run_id(L, mod, align, seed):
    suffix = "" if seed == 42 else f"_s{seed}"
    return f"ep_{L}_{mod}_{align}{suffix}"


def main():
    runs = collect(RUNS_DIR)
    if not runs:
        print(f"no runs found under {RUNS_DIR}")
        return
    print(f"found {len(runs)} runs")

    available_seeds = sorted({r["meta"]["seed"] for r in runs.values()})

    # ---- fixed-step / fixed-epoch curves (seed 42 canonical view) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for align, ax in (("fs", axes[0]), ("fe", axes[1])):
        for L in ["L1", "L2", "L3", "L4"]:
            for mod in ["bigram", "trigram", "both", "nogram"]:
                run_id = base_run_id(L, mod, align, 42)
                if run_id not in runs:
                    continue
                online = runs[run_id]["online"]
                x = [p["step"] for p in online] if align == "fs" else [p["epoch"] for p in online]
                y = [p["gap"] for p in online]
                ax.plot(x, y, label=f"{L} {mod}", color=MODULES[mod]["color"],
                        linestyle="-", marker=".", markersize=3, alpha=0.9)
        ax.set_xlabel("step" if align == "fs" else "epoch")
        ax.set_ylabel("online gap (val - train)")
        ax.set_title(f"{ALIGN[align]} alignment (seed 42)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("Epoch-length scaling: online train/validation gap")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_gap_by_alignment.png"), dpi=150)
    plt.close(fig)
    print(f"saved {FIGS_DIR}/epoch_gap_by_alignment.png")

    # ---- multi-seed final ΔG summary: module - no-ngram, fixed-step ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
    offsets = {"bigram": -0.22, "trigram": 0.0, "both": 0.22}
    for L in ["L1", "L2", "L3", "L4"]:
        for mod in ["bigram", "trigram", "both"]:
            vals = []
            for seed in available_seeds:
                mod_id = base_run_id(L, mod, "fs", seed)
                base_id = base_run_id(L, "nogram", "fs", seed)
                if mod_id not in runs or base_id not in runs:
                    continue
                g_mod = final_gap(runs[mod_id])
                g_base = final_gap(runs[base_id])
                if g_mod is None or g_base is None:
                    continue
                vals.append((seed, g_mod - g_base))
            if not vals:
                continue
            xs = [x_positions[L] + offsets[mod]] * len(vals)
            ys = [v for _, v in vals]
            ax.scatter(xs, ys, color=MODULES[mod]["color"], alpha=0.75, s=28)
            mean_y = float(np.mean(ys))
            ax.plot([x_positions[L] + offsets[mod]], [mean_y], marker="_",
                    markersize=14, markeredgewidth=2.5, color=MODULES[mod]["color"])
    for mod in ["bigram", "trigram", "both"]:
        ax.scatter([], [], color=MODULES[mod]["color"], label=mod)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["L1", "L2", "L3", "L4"])
    ax.set_xlabel("epoch length")
    ax.set_ylabel("final ΔG = G(module) − G(no-ngram)")
    ax.set_title("Fixed-step final ΔG by seed (points = seeds, tick = mean)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_deltaG_fs_multiseed.png"), dpi=150)
    plt.close(fig)
    print(f"saved {FIGS_DIR}/epoch_deltaG_fs_multiseed.png")

    # ---- seed 42 delta-G curve (kept for continuity with existing report) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in ["L1", "L2", "L3", "L4"]:
        base_id = base_run_id(L, "nogram", "fs", 42)
        if base_id not in runs:
            continue
        base = runs[base_id]["online"]
        base_map = {p["step"]: p["gap"] for p in base}
        for mod in ["bigram", "trigram", "both"]:
            run_id = base_run_id(L, mod, "fs", 42)
            if run_id not in runs:
                continue
            online = runs[run_id]["online"]
            xs, ys = [], []
            for p in online:
                if p["step"] in base_map:
                    xs.append(p["step"])
                    ys.append(p["gap"] - base_map[p["step"]])
            ax.plot(xs, ys, label=f"{L} {mod}", color=MODULES[mod]["color"],
                    marker=".", markersize=3, alpha=0.9)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("ΔG = G(module) − G(no-ngram)")
    ax.set_title("Fixed-step: table-induced online gap vs epoch length (seed 42)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_deltaG_fs.png"), dpi=150)
    plt.close(fig)
    print(f"saved {FIGS_DIR}/epoch_deltaG_fs.png")

    # ---- summary table ----
    rows = []
    for run_id, run in sorted(runs.items()):
        g = final_gap(run)
        if g is None:
            continue
        m = run["meta"]
        rows.append({
            "run": run["run_id"],
            "L": m["L"],
            "module": m["module"],
            "align": m["align"],
            "seed": m["seed"],
            "final_gap": round(g, 4),
            "epb": run["summary"].get("epoch_batches"),
        })
    if rows:
        header = ",".join(rows[0].keys())
        lines = [header] + [",".join(str(r[k]) for k in rows[0].keys()) for r in rows]
        out_csv = os.path.join(FIGS_DIR, "epoch_final_gap.csv")
        with open(out_csv, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"saved {out_csv}")
        print("final gaps:")
        for r in sorted(rows, key=lambda x: (x["align"], x["L"], x["module"], x["seed"])):
            print(f"  {r['run']:<34} seed={r['seed']} gap={r['final_gap']:+.4f}")


if __name__ == "__main__":
    main()
