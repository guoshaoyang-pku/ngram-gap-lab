#!/usr/bin/env python3
"""ngram-gap-lab · epoch-length scaling analysis (plan §3).

Reads data/runs_scaling/<run>/:
  - fixed_train_loss.jsonl : {step, epoch, fixed_train_loss, fixed_val_loss, fixed_gap}
  - exact_freq_loss.jsonl  : per-exact-f train/val/shared stats
  - train_log.jsonl        : online loss + epoch (diagnostic)
  - summary.json           : run metadata (epoch_batches, module flags, betas)

Produces:
  - fig: fixed-gap vs step, per epoch length (fixed-step alignment)
  - fig: fixed-gap vs epoch, per epoch length (fixed-epoch alignment)
  - fig: delta-G (module - no-ngram) vs step / epoch
  - table: final fixed-gap per (L, module, alignment)

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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.environ.get(
    "NGLAB_SCALING_RUNS_DIR",
    os.environ.get("NGLAB_RUNS_DIR", os.path.join(REPO_ROOT, "data", "runs_scaling")),
)
if len(sys.argv) > 1:
    RUNS_DIR = sys.argv[1]
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs", "scaling_epoch")
os.makedirs(FIGS_DIR, exist_ok=True)

EPB = {"L1": 42, "L2": 84, "L3": 168, "L4": 336}
MODULES = {
    "bigram": {"bigram": 1, "trigram": 0, "label": "bigram-only", "color": "#9C27B0"},
    "trigram": {"bigram": 0, "trigram": 1, "label": "trigram-only", "color": "#FF9800"},
    "both": {"bigram": 1, "trigram": 1, "label": "both", "color": "#2196F3"},
    "nogram": {"bigram": 0, "trigram": 0, "label": "no-ngram", "color": "#4CAF50"},
}
ALIGN = {"fs": "fixed-step", "fe": "fixed-epoch"}


def load_probe(run_dir):
    path = os.path.join(run_dir, "fixed_train_loss.jsonl")
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
                "fixed_train": float(e["fixed_train_loss"]),
                "fixed_val": float(e["fixed_val_loss"]),
                "fixed_gap": float(e["fixed_gap"]),
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
    return (
        summary.get("run_id") == physical_id
        and config.get("table_optimizer") == "rmsprop"
        and config.get("table_lr_scale") == 2.0
        and config.get("table_betas") == [0.0, 0.99]
        and config.get("val_interval_steps") == 10
        and config.get("table_norm_interval_steps") == 10
        and summary.get("probe_eval_interval") == 10
        and summary.get("compute_dtype") == "bf16"
        and summary.get("torch_compile") is False
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
    """Parse run_id into (L, module, alignment)."""
    m = {}
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
    """Return {run_id: {probe, summary, meta}} for all scaling epoch runs."""
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
        probe = load_probe(run_dir)
        if not probe:
            continue
        out[run_id] = {
            "probe": probe,
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
    probe = run["probe"]
    if not probe:
        return None
    return probe[-1]["fixed_gap"]


def main():
    runs = collect(RUNS_DIR)
    if not runs:
        print(f"no runs found under {RUNS_DIR}")
        return
    print(f"found {len(runs)} runs")

    # ---- fixed-step: gap vs step ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for align, ax in (("fs", axes[0]), ("fe", axes[1])):
        for L in ["L1", "L2", "L3", "L4"]:
            for mod in ["bigram", "trigram", "both", "nogram"]:
                run_id = f"ep_{L}_{mod}_{align}"
                if run_id not in runs:
                    continue
                probe = runs[run_id]["probe"]
                x = [p["step"] for p in probe] if align == "fs" else [p["epoch"] for p in probe]
                y = [p["fixed_gap"] for p in probe]
                ax.plot(x, y, label=f"{L} {mod}", color=MODULES[mod]["color"],
                        linestyle="-", marker=".", markersize=3, alpha=0.9)
        ax.set_xlabel("step" if align == "fs" else "epoch")
        ax.set_ylabel("fixed gap (val - train probe)")
        ax.set_title(f"{ALIGN[align]} alignment")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle("Epoch-length scaling: fixed-train-probe gap")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "epoch_gap_by_alignment.png"), dpi=150)
    plt.close(fig)
    print(f"saved {FIGS_DIR}/epoch_gap_by_alignment.png")

    # ---- delta-G: module - no-ngram (fixed-step) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for L in ["L1", "L2", "L3", "L4"]:
        base_id = f"ep_{L}_nogram_fs"
        if base_id not in runs:
            continue
        base = runs[base_id]["probe"]
        base_map = {p["step"]: p["fixed_gap"] for p in base}
        for mod in ["bigram", "trigram", "both"]:
            run_id = f"ep_{L}_{mod}_fs"
            if run_id not in runs:
                continue
            probe = runs[run_id]["probe"]
            xs, ys = [], []
            for p in probe:
                if p["step"] in base_map:
                    xs.append(p["step"])
                    ys.append(p["fixed_gap"] - base_map[p["step"]])
            ax.plot(xs, ys, label=f"{L} {mod}", color=MODULES[mod]["color"],
                    marker=".", markersize=3, alpha=0.9)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("ΔG = G(module) − G(no-ngram)")
    ax.set_title("Fixed-step: table-induced gap vs epoch length")
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
        for r in sorted(rows, key=lambda x: (x["align"], x["L"], x["module"])):
            print(f"  {r['run']:<28} gap={r['final_gap']:+.4f}")


if __name__ == "__main__":
    main()
