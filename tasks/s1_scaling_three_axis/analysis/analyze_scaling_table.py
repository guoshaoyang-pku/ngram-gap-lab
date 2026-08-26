#!/usr/bin/env python3
"""ngram-gap-lab · table-size scaling analysis (plan §5).

Reads data/runs_scaling/<run>/:
  - train_log.jsonl        : final online train/validation gap per dense run
  - table_occupancy.json   : per-branch/layer/hash occupancy + collision
  - summary.json           : table_mult, params

Plots (all log-log by default):
  - final gap vs logical addresses 2R (log-log), per module
  - final gap vs measured collision rate / occupancy (log-log)
  - same, with a "scatter + fitted power-law" view (raw points are few, so
    we overlay a log-log linear fit to guide the eye; the fit is explicit,
    not a claim of a scaling law)

Design choices for the table grid:
  - show all measured sizes as points; do NOT fabricate intermediate sizes.
  - log-log axes make a power-law G ~ (2R)^k look linear.
  - lines are optional (set SHOW_LINES=False to show only markers).

Usage:
  python3 tasks/s1_scaling_three_axis/analysis/analyze_scaling_table.py [runs_dir]
  ... --modules bigram,both --x-scale linear --y-scale log --no-lines
"""
import json
import os
import sys
import glob
import math
import argparse

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
FIGS_DIR = os.path.join(
    REPO_ROOT, "docs", "appendices", "s1_scaling_three_axis", "figs"
)
os.makedirs(FIGS_DIR, exist_ok=True)

# skill palette
PAPER = "#f7f5ef"
BORDER = "#c8c1b6"
MUTED = "#686d73"
ANCHOR = "#353d79"
MOD_COLORS = {"bigram": "#2d6f9f", "trigram": "#c4493d", "both": "#353d79"}
SEED_MARKERS = {42: "o", 43: "s", 44: "^"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot final table-size gaps with static and interactive views."
    )
    parser.add_argument(
        "runs_dir", nargs="?", default=RUNS_DIR,
        help="directory containing *_fixed scaling runs",
    )
    parser.add_argument(
        "--modules", default="bigram,trigram,both",
        help="comma-separated curves to show (bigram,trigram,both)",
    )
    parser.add_argument(
        "--x-scale", choices=("log", "linear"), default="log",
        help="x-axis scale for static figures",
    )
    parser.add_argument(
        "--y-scale", choices=("log", "linear"), default="log",
        help="y-axis scale for static figures",
    )
    parser.add_argument(
        "--no-lines", action="store_true",
        help="show markers only in static figures",
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="skip the interactive Plotly HTML outputs",
    )
    parser.add_argument(
        "--gap", choices=("online", "fixed"), default="online",
        help="which gap to plot (online = train_log last online gap; "
             "fixed = final_fixed_gap from summary.json)",
    )
    args = parser.parse_args()
    args.modules = tuple(m.strip() for m in args.modules.split(",") if m.strip())
    allowed = set(MOD_COLORS)
    if not args.modules or set(args.modules) - allowed:
        parser.error("--modules must contain one or more of: bigram,trigram,both")
    return args


def load_online_final(run_dir):
    path = os.path.join(run_dir, "train_log.jsonl")
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
    return {"online_gap": float(last["gap"]),
            "online_train": float(last["train_loss"]),
            "online_val": float(last["val_loss"])}


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
    dense_monitor = config.get("val_interval_steps") == 10 and not config.get("val_steps")
    sparse_monitor = bool(config.get("val_steps"))
    return (
        summary.get("run_id") == physical_id
        and config.get("table_optimizer") == "rmsprop"
        and config.get("table_lr_scale") == 2.0
        and config.get("table_betas") == [0.0, 0.99]
        and (dense_monitor or sparse_monitor)
        and summary.get("compute_dtype") == "bf16"
        and summary.get("torch_compile") is False
    )


def style_ax(ax):
    ax.set_facecolor(PAPER)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(MUTED)
    ax.grid(alpha=0.3, which="both", color=BORDER)


def write_interactive_html(rows, path, title, x_key, x_title, note):
    traces = []
    for mod, color in MOD_COLORS.items():
        pts = [r for r in rows if r["module"] == mod]
        pts.sort(key=lambda r: r[x_key])
        if x_key == "logical_2R":
            xs = [r["logical_2R"] for r in pts]
        else:
            xs = [1.0 - r["collision"] for r in pts]
        traces.append({
            "x": xs,
            "y": [r["final_gap"] for r in pts],
            "mode": "lines+markers",
            "name": mod,
            "line": {"color": color, "width": 2},
            "marker": {"color": color, "size": 8},
        })
    payload = json.dumps(traces, ensure_ascii=False)
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body{margin:0;background:__PAPER__;color:__MUTED__;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
#plot{width:100%;height:650px}
.note{max-width:1000px;margin:8px auto;padding:0 18px;font-size:14px}
</style></head><body>
<div id="plot"></div>
<p class="note">__NOTE__ Legend clicks hide/show curves. The buttons switch x and y
axes independently between linear and logarithmic scales.</p>
<script>
const traces = __PAYLOAD__;
const layout = {
  title: __TITLE_JSON__,
  paper_bgcolor: __PAPER_JSON__,
  plot_bgcolor: __PAPER_JSON__,
  font: {color: __MUTED_JSON__},
  xaxis: {title: __X_TITLE_JSON__, type: "log",
          gridcolor: __BORDER_JSON__},
  yaxis: {title: "final fixed gap", type: "log",
          gridcolor: __BORDER_JSON__},
  legend: {orientation: "h", x: 0.02, y: 1.08},
  margin: {l: 75, r: 35, t: 90, b: 70},
  updatemenus: [{
    type: "buttons", direction: "right", x: 0, y: 1.17,
    buttons: [
      {label: "x log", method: "relayout", args: [{"xaxis.type": "log"}]},
      {label: "x linear", method: "relayout", args: [{"xaxis.type": "linear"}]},
      {label: "y log", method: "relayout", args: [{"yaxis.type": "log"}]},
      {label: "y linear", method: "relayout", args: [{"yaxis.type": "linear"}]},
      {label: "both log", method: "relayout", args: [{"xaxis.type": "log", "yaxis.type": "log"}]}
    ]
  }]
};
Plotly.newPlot("plot", traces, layout, {responsive: true, displaylogo: false});
</script></body></html>"""
    html = (template
            .replace("__TITLE__", title)
            .replace("__PAPER__", PAPER)
            .replace("__MUTED__", MUTED)
            .replace("__NOTE__", note)
            .replace("__PAYLOAD__", payload)
            .replace("__TITLE_JSON__", json.dumps(title, ensure_ascii=False))
            .replace("__PAPER_JSON__", json.dumps(PAPER))
            .replace("__MUTED_JSON__", json.dumps(MUTED))
            .replace("__X_TITLE_JSON__", json.dumps(x_title, ensure_ascii=False))
            .replace("__BORDER_JSON__", json.dumps(BORDER)))
    with open(path, "w") as f:
        f.write(html)


def final_gap(run_dir, summary, gap_mode):
    if gap_mode == "fixed":
        v = summary.get("final_fixed_gap")
        if v is not None:
            return float(v)
    online = load_online_final(run_dir)
    if online:
        return online["online_gap"]
    # Sparse table runs intentionally omit train_log.jsonl; their final
    # online gap is still recorded in summary.json.
    v = summary.get("final_gap")
    return float(v) if v is not None else None


def parse_table_run(run_id):
    seed = 42
    if run_id.endswith("_s43") or run_id.endswith("_s44"):
        base, seed_text = run_id.rsplit("_s", 1)
        run_id = base
        seed = int(seed_text)
    parts = run_id.split("_")
    if len(parts) < 3:
        return None
    if parts[1] == "pilot":
        mult = None
        mod = parts[-1]
        addr = parts[2]
        addr_map = {"1M": 1048576, "128K": 131072, "16K": 16384}
        if addr not in addr_map:
            return None
        logical = addr_map[addr]
    else:
        mult = int(parts[1])
        mod = parts[2]
        logical = 16384 * mult
    if mod not in MOD_COLORS:
        return None
    return {"mult": mult, "module": mod, "logical": logical, "seed": seed}


def main():
    args = parse_args()
    runs_dir = args.runs_dir
    runs = {}
    legacy_count = 0
    rejected_count = 0
    for run_id, physical_id, run_dir in canonical_run_dirs(runs_dir):
        if not run_id.startswith("tbl_"):
            continue
        parsed = parse_table_run(run_id)
        if parsed is None:
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
        if int(summary.get("seed", parsed["seed"])) != int(parsed["seed"]):
            rejected_count += 1
            continue
        online = load_online_final(run_dir)
        occ = load_occupancy(run_dir)
        runs[run_id] = {
            "run_id": physical_id,
            "logical": parsed["logical"], "mult": parsed["mult"],
            "module": parsed["module"], "seed": parsed["seed"],
            "online": online, "occ": occ, "summary": summary,
            "_dir": run_dir,
        }
    for run_dir in glob.glob(os.path.join(runs_dir, "*")):
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

    # ---- final gap vs logical addresses (log-log), per module/seed ----
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(PAPER)
    seed_values = sorted({r["seed"] for r in runs.values()})
    for mod, color in MOD_COLORS.items():
        if mod not in args.modules:
            continue
        for seed in seed_values:
            pts = [(r["logical"], final_gap(r["_dir"], r["summary"], args.gap))
                   for r in runs.values() if r["module"] == mod and r["seed"] == seed
                   and final_gap(r["_dir"], r["summary"], args.gap) is not None]
            if not pts:
                continue
            pts.sort()
            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.array([p[1] for p in pts], dtype=float)
            marker = SEED_MARKERS.get(seed, "o")
            label = f"{mod} s{seed} (n={len(pts)})"
            ax.plot(xs, ys, marker=marker, markersize=6,
                    linewidth=0 if args.no_lines else 1.5,
                    label=label, color=color,
                    alpha=0.95 if seed == 42 else 0.65)
            pos = ys > 0
            if (pos.sum() >= 3 and not args.no_lines
                    and args.x_scale == "log" and args.y_scale == "log"):
                k, b = np.polyfit(np.log(xs[pos]), np.log(ys[pos]), 1)
                xf = np.linspace(xs[pos].min(), xs[pos].max(), 100)
                ax.plot(xf, np.exp(b) * xf ** k, "--", linewidth=1.0, alpha=0.5, color=color)
    ax.set_xscale(args.x_scale)
    ax.set_yscale(args.y_scale)
    ax.set_xlabel(f"logical addresses 2R (per n-gram, per layer; {args.x_scale})")
    gap_label = "final fixed-probe gap" if args.gap == "fixed" else "final online gap"
    ax.set_ylabel(f"{gap_label} ({args.y_scale})")
    fit_note = (
        "dashed = log-log linear fit per seed (guide, not a scaling-law claim)"
        if args.x_scale == "log" and args.y_scale == "log" and not args.no_lines
        else "measured final points"
    )
    ax.set_title("Table-size scaling: final gap vs logical addresses\n" + fit_note)
    ax.legend(fontsize=8, ncol=2)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "table_gap_vs_2R.png"), dpi=150)
    plt.close(fig)
    print("saved table_gap_vs_2R.png")

    # ---- ΔG vs collision rate (bigram L0 h0), log-x, log-y ----
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(PAPER)
    for mod, color in MOD_COLORS.items():
        if mod not in args.modules:
            continue
        for seed in seed_values:
            pts = []
            for r in runs.values():
                if r["module"] != mod or r["seed"] != seed or not r["occ"]:
                    continue
                g = final_gap(r["_dir"], r["summary"], args.gap)
                if g is None:
                    continue
                coll = r["occ"]["branches"]["bigram"]["0"][0]["collision_rate"]
                pts.append((1.0 - coll, g))
            if not pts:
                continue
            pts.sort()
            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.array([p[1] for p in pts], dtype=float)
            marker = SEED_MARKERS.get(seed, "o")
            ax.plot(xs, ys, marker=marker, markersize=6,
                    linewidth=0 if args.no_lines else 1.5,
                    label=f"{mod} s{seed} (n={len(pts)})", color=color,
                    alpha=0.95 if seed == 42 else 0.65)
            pos = ys > 0
            if (pos.sum() >= 3 and not args.no_lines
                    and args.x_scale == "log" and args.y_scale == "log"):
                k, b = np.polyfit(np.log(xs[pos]), np.log(ys[pos]), 1)
                xf = np.linspace(xs[pos].min(), xs[pos].max(), 100)
                ax.plot(xf, np.exp(b) * xf ** k, "--", linewidth=1.0, alpha=0.5, color=color)
    ax.set_xscale(args.x_scale)
    ax.set_yscale(args.y_scale)
    ax.set_xlabel(
        f"1 − collision rate (bigram L0 h0; {args.x_scale}; larger = fewer collisions)"
    )
    ax.set_ylabel(f"{gap_label} ({args.y_scale})")
    ax.set_title("Table-size scaling: gap vs (1 − collision rate)\n" + fit_note)
    ax.legend(fontsize=8, ncol=2)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "table_gap_vs_collision.png"), dpi=150)
    plt.close(fig)
    print("saved table_gap_vs_collision.png")

    # ---- table summary csv ----
    rows = []
    for run_id, r in sorted(runs.items()):
        g = final_gap(r["_dir"], r["summary"], args.gap)
        rows.append({
            "run": r["run_id"], "seed": r["seed"], "module": r["module"],
            "logical_2R": r["logical"], "mult": r["mult"],
            "final_gap": round(g, 4) if g is not None else None,
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
        for r in sorted(rows, key=lambda x: (x["module"], x["seed"], x["logical_2R"])):
            if r["final_gap"] is None:
                print(f"  {r['run']:<24} s{r['seed']} 2R={r['logical_2R']:>8} gap=missing")
            else:
                print(f"  {r['run']:<24} s{r['seed']} 2R={r['logical_2R']:>8} gap={r['final_gap']:+.4f} "
                      f"coll={r['collision']:.4f} occ={r['occupancy']:.4f}")
        if not args.no_html:
            html_rows = [
                r for r in rows
                if r["module"] in args.modules and r["final_gap"] is not None
                and r["collision"] is not None
            ]
            write_interactive_html(
                html_rows,
                os.path.join(FIGS_DIR, "table_gap_vs_2R.html"),
                "Table-size scaling: final gap vs logical addresses",
                "logical_2R",
                "logical addresses 2R",
                "Interactive view of all measured table points.",
            )
            write_interactive_html(
                html_rows,
                os.path.join(FIGS_DIR, "table_gap_vs_collision.html"),
                "Table-size scaling: final gap vs 1 − collision rate",
                "collision",
                "1 − collision rate (bigram L0 h0)",
                "The collision view uses 1 − collision rate because collision rate is near 1.",
            )
            print("saved table_gap_vs_2R.html")
            print("saved table_gap_vs_collision.html")


if __name__ == "__main__":
    main()
