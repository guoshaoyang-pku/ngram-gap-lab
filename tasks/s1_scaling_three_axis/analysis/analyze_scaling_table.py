#!/usr/bin/env python3
"""ngram-gap-lab · table-size scaling analysis (plan §5).

Reads data/runs_scaling/<run>/:
  - fixed_train_loss.jsonl : final fixed gap per run
  - table_occupancy.json   : per-branch/layer/hash occupancy + collision
  - summary.json           : table_mult, params

Plots:
  - final ΔG vs logical addresses 2R (log-log)
  - ΔG vs measured collision rate / occupancy
  - per-branch curves

Usage: python3 docs/plot_scripts/analyze_scaling_table.py [runs_dir]
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

TABLE_MULT_TO_2R = {64: 1048576, 32: 524288, 16: 262144, 8: 131072,
                    4: 65536, 2: 32768, 1: 16384}


def load_probe_final(run_dir):
    path = os.path.join(run_dir, "fixed_train_loss.jsonl")
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            last = e
    if not last:
        return None
    return {"fixed_gap": float(last["fixed_gap"]),
            "fixed_train": float(last["fixed_train_loss"]),
            "fixed_val": float(last["fixed_val_loss"])}


def load_occupancy(run_dir):
    path = os.path.join(run_dir, "table_occupancy.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def canonical_run_dirs(runs_dir):
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue
        physical_id = os.path.basename(run_dir)
        if not physical_id.endswith("_fixed"):
            continue
        yield physical_id[:-len("_fixed")], physical_id, run_dir


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
        and summary.get("torch_compile") is True
    )


def main():
    runs = {}
    legacy_count = 0
    rejected_count = 0
    for run_id, physical_id, run_dir in canonical_run_dirs(RUNS_DIR):
        if not run_id.startswith("tbl_"):
            continue
        parts = run_id.split("_")
        if len(parts) < 3:
            continue
        # tbl_<mult>_<module> or tbl_pilot_<...>
        if parts[1] == "pilot":
            # tbl_pilot_<addr>_<module>
            mult = None
            mod = parts[-1]
            addr = parts[2]
            addr_map = {"1M": 1048576, "128K": 131072, "16K": 16384}
            if addr not in addr_map:
                continue
            logical = addr_map[addr]
        else:
            mult = int(parts[1])
            mod = parts[2]
            logical = TABLE_MULT_TO_2R.get(mult)
            if logical is None:
                continue
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            rejected_count += 1
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        if not is_current_scaling_summary(summary, physical_id):
            rejected_count += 1
            continue
        probe = load_probe_final(run_dir)
        occ = load_occupancy(run_dir)
        runs[run_id] = {
            "run_id": physical_id,
            "logical": logical, "mult": mult, "module": mod,
            "probe": probe, "occ": occ,
        }
    for run_dir in glob.glob(os.path.join(RUNS_DIR, "*")):
        if os.path.isdir(run_dir) and not os.path.basename(run_dir).endswith("_fixed"):
            legacy_count += 1
    if legacy_count:
        print(f"ignored {legacy_count} non-canonical scaling directories (expected *_fixed)")
    if rejected_count:
        print(f"ignored {rejected_count} scaling directories with an invalid run contract")

    if not runs:
        print(f"no tbl_* runs under {RUNS_DIR}")
        return
    print(f"found {len(runs)} table runs")

    # ---- final gap vs logical addresses ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod, color in (("bigram", "#9C27B0"), ("trigram", "#FF9800"), ("both", "#2196F3")):
        pts = [(r["logical"], r["probe"]["fixed_gap"])
               for r in runs.values() if r["module"] == mod and r["probe"]]
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=mod, color=color)
    ax.set_xscale("log")
    ax.set_xlabel("logical addresses 2R (per n-gram, per layer)")
    ax.set_ylabel("final fixed gap")
    ax.set_title("Table-size scaling: final gap vs logical addresses")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "table_gap_vs_2R.png"), dpi=150)
    plt.close(fig)
    print("saved table_gap_vs_2R.png")

    # ---- ΔG vs collision rate (bigram layer 0 hash 0) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for mod, color in (("bigram", "#9C27B0"), ("trigram", "#FF9800"), ("both", "#2196F3")):
        pts = []
        for r in runs.values():
            if r["module"] != mod or not r["occ"] or not r["probe"]:
                continue
            branch = "bigram"
            coll = r["occ"]["branches"][branch]["0"][0]["collision_rate"]
            pts.append((coll, r["probe"]["fixed_gap"]))
        if not pts:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=mod, color=color)
    ax.set_xlabel("measured collision rate (bigram L0 h0)")
    ax.set_ylabel("final fixed gap")
    ax.set_title("Table-size scaling: gap vs collision rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "table_gap_vs_collision.png"), dpi=150)
    plt.close(fig)
    print("saved table_gap_vs_collision.png")

    # ---- table summary csv ----
    rows = []
    for run_id, r in sorted(runs.items()):
        rows.append({
            "run": r["run_id"], "module": r["module"], "logical_2R": r["logical"],
            "mult": r["mult"],
            "final_gap": round(r["probe"]["fixed_gap"], 4) if r["probe"] else None,
            "collision": round(r["occ"]["branches"]["bigram"]["0"][0]["collision_rate"], 4)
            if r["occ"] else None,
            "occupancy": round(r["occ"]["branches"]["bigram"]["0"][0]["occupancy"], 4)
            if r["occ"] else None,
        })
    if rows:
        out_csv = os.path.join(FIGS_DIR, "table_summary.csv")
        with open(out_csv, "w") as f:
            f.write(",".join(rows[0].keys()) + "\n")
            for r in rows:
                f.write(",".join(str(r[k]) for k in rows[0].keys()) + "\n")
        print(f"saved {out_csv}")
        for r in sorted(rows, key=lambda x: (x["module"], x["logical_2R"])):
            print(f"  {r['run']:<20} 2R={r['logical_2R']:>8} gap={r['final_gap']:+.4f} "
                  f"coll={r['collision']:.4f} occ={r['occupancy']:.4f}")


if __name__ == "__main__":
    main()
