#!/usr/bin/env python3
"""ngram-gap-lab · docs/plot_scripts/gen_all_figures.py

Generate all figures for the blog from training outputs:
  1. fig_gap.svg / fig_loss.svg — v/y/input gap and loss curves
  2. fig_table_norm.svg / fig_input_alignment.svg — norm and loss alignment
  3. fig_freq_*.svg — per-frequency-bin loss, gap, and contribution
  4. fig_hitcount_dist.svg — full bigram/trigram hit-count distribution

Reads from data/runs/<run_id>/train_log.jsonl, table_norm.jsonl, freq_bin_loss.jsonl.
Outputs to docs/figs/*.svg and docs/figs/*.html.
"""
import json
import os
import re
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs")
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs")
MIRROR_FIGS_DIR = os.environ.get("NGRAM_GAP_BLOG_FIGS_DIR")
FALLBACK_BLOG_DIR = os.environ.get(
    "NGRAM_GAP_BLOG_DIR",
    os.path.join(os.path.dirname(REPO_ROOT), "guoshaoyang-pku.github.io",
                 "blogs", "ngram-gap-mechanism-guide"),
)

RUNS = {
    "v": {"label": "v (ResFormer, add to V)", "color": "#2196F3", "dir": "nglab_v"},
    "y": {"label": "y (post-attn residual)", "color": "#F44336", "dir": "nglab_y"},
    "input": {"label": "input (over-encoding)", "color": "#4CAF50", "dir": "nglab_input"},
}


def load_jsonl(path):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pts.append(json.loads(line))
    return pts


def extract_js_value(path, variable):
    if not os.path.exists(path):
        return None
    text = open(path).read()
    match = re.search(rf"var {variable} = (.*?);\n", text, re.S)
    return json.loads(match.group(1)) if match else None


def extract_js_traces(path, variable):
    if not os.path.exists(path):
        return []
    text = open(path).read()
    match = re.search(rf"var {variable} = (.*?);\n", text, re.S)
    if not match:
        return []
    block = match.group(1)
    traces = []
    for x_text, y_text, name in re.findall(
            r"x:\s*(\[.*?\]),\s*y:\s*(\[.*?\]),\s*mode:\s*\"[^\"]*\",\s*"
            r"name:\s*\"([^\"]+)\"", block, re.S):
        traces.append({"x": json.loads(x_text), "y": json.loads(y_text), "name": name})
    return traces


def load_fallback_blog_data():
    """Recover the committed figure data when local run artifacts are absent."""
    gap_data = extract_js_value(os.path.join(FALLBACK_BLOG_DIR, "fig_gap_loss.html"),
                                "gapData")
    loss_data = extract_js_value(os.path.join(FALLBACK_BLOG_DIR, "fig_gap_loss.html"),
                                 "lossData")
    norm_data = extract_js_traces(os.path.join(FALLBACK_BLOG_DIR, "fig_loss_norm.html"),
                                  "normTraces")
    freq_data = extract_js_value(os.path.join(FALLBACK_BLOG_DIR, "fig_gap_by_freq.html"),
                                 "series")
    dist_data = extract_js_value(os.path.join(FALLBACK_BLOG_DIR, "fig_hitcount_dist.html"),
                                 "barData")
    if not gap_data or not loss_data or not freq_data:
        return {}
    if not dist_data:
        dist_data = {}
        for branch in ["bigram", "trigram"]:
            dist_data[branch] = [
                {
                    "bucket": bucket,
                    "train_frac": freq_data[branch][bucket]["train_frac"][-1],
                    "val_frac": freq_data[branch][bucket]["val_frac"][-1],
                }
                for bucket in BUCKET_ORDER
            ]

    data = {}
    for key in RUNS:
        gap_trace = next((trace for trace in gap_data if trace["name"].startswith(key)), None)
        train_trace = next((trace for trace in loss_data if trace["name"].startswith(key)
                            and "(train)" in trace["name"]), None)
        val_trace = next((trace for trace in loss_data if trace["name"].startswith(key)
                          and "(val)" in trace["name"]), None)
        if not gap_trace or not train_trace or not val_trace:
            continue
        data[key] = {
            "info": RUNS[key],
            "train_log": [
                {"step": step, "gap": gap, "train_loss": train_loss, "val_loss": val_loss}
                for step, gap, train_loss, val_loss in zip(
                    gap_trace["x"], gap_trace["y"], train_trace["y"], val_trace["y"])
            ],
            "table_norm": [],
            "freq_bin": [],
        }

    if "input" not in data:
        return {}
    input_data = data["input"]
    if norm_data:
        norm_steps = norm_data[0]["x"]
        input_data["table_norm"] = [
            {"step": step, "bigram.layer_01.table_0.rms": bigram,
             "trigram.layer_01.table_0.rms": trigram}
            for step, bigram, trigram in zip(norm_steps, norm_data[0]["y"], norm_data[1]["y"])
        ]

    input_data["freq_bin"] = []
    for index, step in enumerate(freq_data["bigram"]["novel"]["steps"]):
        point = {"step": step, "train": {}, "val": {}}
        for branch in ["bigram", "trigram"]:
            point["train"][branch] = {}
            point["val"][branch] = {}
            fractions = {row["bucket"]: row for row in dist_data[branch]}
            for bucket in BUCKET_ORDER:
                series_row = freq_data[branch][bucket]
                fraction = fractions[bucket]
                train_loss = series_row["train_loss"][index]
                val_loss = series_row["val_loss"][index]
                point["train"][branch][bucket] = {
                    "frac": fraction["train_frac"], "mean_loss": train_loss,
                    "total_contrib": fraction["train_frac"] * train_loss, "token_count": 1,
                }
                point["val"][branch][bucket] = {
                    "frac": fraction["val_frac"], "mean_loss": val_loss,
                    "total_contrib": fraction["val_frac"] * val_loss, "token_count": 1,
                }
        input_data["freq_bin"].append(point)
    return data


def load_all():
    """Load all run data into a dict."""
    data = {}
    for key, info in RUNS.items():
        run_dir = os.path.join(RUNS_DIR, info["dir"])
        data[key] = {
            "info": info,
            "train_log": load_jsonl(os.path.join(run_dir, "train_log.jsonl")),
            "table_norm": load_jsonl(os.path.join(run_dir, "table_norm.jsonl")),
            "freq_bin": load_jsonl(os.path.join(run_dir, "freq_bin_loss.jsonl")),
        }
    if not any(d["train_log"] for d in data.values()):
        fallback = load_fallback_blog_data()
        if fallback:
            data.update(fallback)
    return data


PLOTLY_HEAD = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

HTML_WRAP = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 12px; background: #fafafa; }}
  h2 {{ font-size: 1.1em; }}
  .chart {{ width: 100%; max-width: 900px; height: 480px; margin: 12px 0; }}
  .controls {{ margin: 8px 0; font-size: 0.9em; }}
  .controls button {{ padding: 4px 10px; margin: 2px; cursor: pointer; border: 1px solid #ccc; border-radius: 4px; background: #fff; }}
  .controls button.active {{ background: #333; color: #fff; }}
  .note {{ font-size: 0.85em; color: #666; margin: 8px 0; }}
</style></head><body>
<h2>{title}</h2>
<p class="note">{note}</p>
{body}
{plotly}
<script>{script}</script>
</body></html>"""


def gen_fig_gap_loss(data):
    """Figure 1: v/y/input gap + loss curves."""
    out = os.path.join(FIGS_DIR, "fig_gap_loss.html")
    traces_gap = []
    traces_loss = []
    for key, d in data.items():
        info = d["info"]
        pts = d["train_log"]
        if not pts:
            continue
        x = [p["step"] for p in pts]
        gap = [p["gap"] for p in pts]
        traces_gap.append({"x": x, "y": gap, "mode": "lines+markers", "name": info["label"],
                            "line": {"color": info["color"], "width": 2}})
        traces_loss.append({"x": x, "y": [p["train_loss"] for p in pts], "mode": "lines",
                            "name": info["label"] + " (train)", "line": {"color": info["color"], "width": 1.5, "dash": "dash"}})
        traces_loss.append({"x": x, "y": [p["val_loss"] for p in pts], "mode": "lines",
                            "name": info["label"] + " (val)", "line": {"color": info["color"], "width": 2}})

    epoch_shapes = [
        {"type": "line", "x0": 337, "x1": 337, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#ccc", "dash": "dot"}},
        {"type": "line", "x0": 686, "x1": 686, "y0": 0, "y1": 1, "yref": "paper", "line": {"color": "#ccc", "dash": "dot"}},
    ]
    body = '<div id="gap_chart" class="chart"></div><div id="loss_chart" class="chart"></div>'
    script = f"""
    var gapData = {json.dumps(traces_gap)};
    var lossData = {json.dumps(traces_loss)};
    var shapes = {json.dumps(epoch_shapes)};
    var annots = [
      {{x: 337, y: 0.95, yref: "paper", text: "epoch2", showarrow: false, font: {{size: 10, color: "#999"}}}},
      {{x: 686, y: 0.95, yref: "paper", text: "epoch3", showarrow: false, font: {{size: 10, color: "#999"}}}}
    ];
    Plotly.newPlot("gap_chart", gapData, {{
        title: "Train/Val Gap (val - train)", xaxis: {{title: "step"}}, yaxis: {{title: "gap"}},
        margin: {{l:60,r:30,t:50,b:50}}, shapes: shapes, annotations: annots
    }});
    Plotly.newPlot("loss_chart", lossData, {{
        title: "Train/Val Loss", xaxis: {{title: "step"}}, yaxis: {{title: "loss"}},
        margin: {{l:60,r:30,t:50,b:50}}, shapes: shapes
    }});
    """
    html = HTML_WRAP.format(title="注入点消融：Gap + Loss", note="v/y/input 三注入点对比，vanilla nanoGPT + bigram+trigram + seed42 + 1000步",
                            body=body, plotly=PLOTLY_HEAD, script=script)
    with open(out, "w") as f:
        f.write(html)
    print(f"[fig] {out}")


def gen_fig_loss_norm(data):
    """Figure 2: table norm ↔ loss time alignment (input run as main)."""
    out = os.path.join(FIGS_DIR, "fig_loss_norm.html")
    d = data.get("input", {})
    if not d.get("train_log"):
        print("[fig] fig_loss_norm: no input data, skipping")
        return
    train_pts = d["train_log"]
    norm_pts = d["table_norm"]
    info = d["info"]

    x_loss = [p["step"] for p in train_pts]
    gap = [p["gap"] for p in train_pts]
    train_loss = [p["train_loss"] for p in train_pts]
    val_loss = [p["val_loss"] for p in train_pts]

    x_norm = [p["step"] for p in norm_pts]
    # use bigram layer_01 table_0 rms
    bg_rms = [p.get("bigram.layer_01.table_0.rms", 0) for p in norm_pts]
    tg_rms = [p.get("trigram.layer_01.table_0.rms", 0) for p in norm_pts]

    body = '<div id="norm_chart" class="chart"></div><div id="align_chart" class="chart"></div>'
    script = f"""
    var normTraces = [
      {{x: {json.dumps(x_norm)}, y: {json.dumps(bg_rms)}, mode: "lines", name: "bigram table RMS", line: {{color: "#2196F3", width: 2}}}},
      {{x: {json.dumps(x_norm)}, y: {json.dumps(tg_rms)}, mode: "lines", name: "trigram table RMS", line: {{color: "#F44336", width: 2}}}}
    ];
    var lossTraces = [
      {{x: {json.dumps(x_loss)}, y: {json.dumps(train_loss)}, mode: "lines", name: "train loss", line: {{color: "#4CAF50", width: 1.5, dash: "dash"}}}},
      {{x: {json.dumps(x_loss)}, y: {json.dumps(val_loss)}, mode: "lines", name: "val loss", line: {{color: "#FF9800", width: 2}}}},
      {{x: {json.dumps(x_loss)}, y: {json.dumps(gap)}, mode: "lines", name: "gap", line: {{color: "#9C27B0", width: 2}}, yaxis: "y2"}}
    ];
    var shapes = [
      {{type: "line", x0: 337, x1: 337, y0: 0, y1: 1, yref: "paper", line: {{color: "#ccc", dash: "dot"}}}},
      {{type: "line", x0: 686, x1: 686, y0: 0, y1: 1, yref: "paper", line: {{color: "#ccc", dash: "dot"}}}}
    ];
    Plotly.newPlot("norm_chart", normTraces, {{
        title: "N-gram Table Param RMS (input run)", xaxis: {{title: "step"}}, yaxis: {{title: "RMS"}},
        margin: {{l:60,r:30,t:50,b:50}}, shapes: shapes
    }});
    Plotly.newPlot("align_chart", lossTraces, {{
        title: "Loss + Gap (input run, dual axis)", xaxis: {{title: "step"}},
        yaxis: {{title: "loss", side: "left"}},
        yaxis2: {{title: "gap", side: "right", overlaying: "y"}},
        margin: {{l:60,r:60,t:50,b:50}}, shapes: shapes, legend: {{x: 0.02, y: 0.98}}
    }});
    """
    html = HTML_WRAP.format(title="Table Norm × Loss 对齐 (input 注入)", note="table RMS 增长与 gap 出现的时间对齐关系",
                            body=body, plotly=PLOTLY_HEAD, script=script)
    with open(out, "w") as f:
        f.write(html)
    print(f"[fig] {out}")


BUCKET_ORDER = ["novel", "1", "2", "3", "4", "5", "6-10", "11-20", "21-50",
                "51-100", "101-200", "201-500", "501-1k", "1k-5k", "5k+"]
BUCKET_COLORS = {"novel": "#E91E63", "1": "#F44336", "2": "#FF5722", "3": "#FF9800",
                 "4": "#FFC107", "5": "#FFEB3B", "6-10": "#CDDC39", "11-20": "#8BC34A",
                 "21-50": "#4CAF50", "51-100": "#009688", "101-200": "#00BCD4",
                 "201-500": "#03A9F4", "501-1k": "#2196F3", "1k-5k": "#3F51B5", "5k+": "#673AB7"}

PAPER = "#f7f5ef"
PANEL = "#fffdf8"
INK = "#232426"
MUTED = "#686d73"
LINE = "#c8c1b6"
ANCHOR = "#353d79"
RUN_COLORS = {"v": "#b67524", "y": "#c4493d", "input": "#2d6f9f"}


def style_axis(ax):
    ax.set_facecolor(PANEL)
    ax.grid(axis="y", color=LINE, linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def add_epoch_lines(ax):
    for step, label in [(337, "epoch 2"), (686, "epoch 3")]:
        ax.axvline(step, color=LINE, linestyle=":", linewidth=1.2)
        ax.text(step + 8, 0.96, label, transform=ax.get_xaxis_transform(),
                color=MUTED, fontsize=8, va="top")


def save_svg(fig, name):
    fig.savefig(os.path.join(FIGS_DIR, name), format="svg", facecolor=PAPER,
                bbox_inches="tight")
    if MIRROR_FIGS_DIR:
        os.makedirs(MIRROR_FIGS_DIR, exist_ok=True)
        fig.savefig(os.path.join(MIRROR_FIGS_DIR, name), format="svg",
                    facecolor=PAPER, bbox_inches="tight")
    plt.close(fig)


def gen_static_loss_figures(data):
    fig, ax = plt.subplots(figsize=(10.8, 4.8), facecolor=PAPER)
    style_axis(ax)
    for key, info in RUNS.items():
        pts = data[key]["train_log"]
        if not pts:
            continue
        x = [p["step"] for p in pts]
        ax.plot(x, [p["gap"] for p in pts], color=RUN_COLORS[key], linewidth=2.2,
                marker="o", markersize=2.8, label=info["label"])
    add_epoch_lines(ax)
    ax.set_title("Train / validation gap", loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("val loss − train loss")
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
    fig.tight_layout()
    save_svg(fig, "fig_gap.svg")

    fig, ax = plt.subplots(figsize=(10.8, 5.3), facecolor=PAPER)
    style_axis(ax)
    for key, info in RUNS.items():
        pts = data[key]["train_log"]
        if not pts:
            continue
        x = [p["step"] for p in pts]
        color = RUN_COLORS[key]
        ax.plot(x, [p["train_loss"] for p in pts], color=color, linewidth=1.5,
                linestyle="--", alpha=0.72, label=f"{key} train")
        ax.plot(x, [p["val_loss"] for p in pts], color=color, linewidth=2.1,
                label=f"{key} val")
    add_epoch_lines(ax)
    ax.set_title("Train / validation loss", loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save_svg(fig, "fig_loss.svg")


def gen_static_norm_figures(data):
    d = data.get("input", {})
    if not d.get("train_log"):
        return
    norm_pts = d.get("table_norm", [])
    fig, ax = plt.subplots(figsize=(10.8, 4.8), facecolor=PAPER)
    style_axis(ax)
    if norm_pts:
        x = [p["step"] for p in norm_pts]
        for prefix, label, color in [
            ("bigram", "bigram table RMS", "#2d6f9f"),
            ("trigram", "trigram table RMS", "#c4493d"),
        ]:
            key = next((k for k in norm_pts[0] if k.startswith(prefix) and k.endswith("table_0.rms")), None)
            if key:
                ax.plot(x, [p.get(key, 0) for p in norm_pts], color=color,
                        linewidth=2.2, label=label)
    add_epoch_lines(ax)
    ax.set_title("N-gram table norm", loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("RMS")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    save_svg(fig, "fig_table_norm.svg")

    pts = d["train_log"]
    x = [p["step"] for p in pts]
    fig, ax = plt.subplots(figsize=(10.8, 4.8), facecolor=PAPER)
    style_axis(ax)
    ax.plot(x, [p["train_loss"] for p in pts], color="#2d6f9f", linewidth=1.6,
            linestyle="--", label="train loss")
    ax.plot(x, [p["val_loss"] for p in pts], color="#c4493d", linewidth=2.1,
            label="val loss")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.plot(x, [p["gap"] for p in pts], color=ANCHOR, linewidth=2.0,
             label="gap")
    ax2.set_ylabel("gap", color=ANCHOR)
    ax2.tick_params(colors=ANCHOR, labelsize=9)
    ax2.spines["right"].set_color(ANCHOR)
    add_epoch_lines(ax)
    ax.set_title("Input run: loss and gap alignment", loc="left",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("step")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False,
              loc="upper left", fontsize=9)
    fig.tight_layout()
    save_svg(fig, "fig_input_alignment.svg")


def final_bucket_values(point, branch):
    values = {}
    for bucket in BUCKET_ORDER:
        train = point["train"][branch].get(bucket, {})
        val = point["val"][branch].get(bucket, {})
        values[bucket] = {
            "train_loss": train.get("mean_loss", 0),
            "val_loss": val.get("mean_loss", 0),
            "gap": val.get("mean_loss", 0) - train.get("mean_loss", 0),
            "train_contrib": train.get("total_contrib", 0),
            "val_contrib": val.get("total_contrib", 0),
            "train_frac": train.get("frac", 0),
            "val_frac": val.get("frac", 0),
        }
    return values


def gen_static_frequency_figures(data):
    points = data.get("input", {}).get("freq_bin", [])
    if not points:
        return
    last = points[-1]
    positions = np.arange(len(BUCKET_ORDER))
    width = 0.38
    for branch in ["bigram", "trigram"]:
        values = final_bucket_values(last, branch)
        train_loss = [values[b]["train_loss"] for b in BUCKET_ORDER]
        val_loss = [values[b]["val_loss"] for b in BUCKET_ORDER]
        gap = [values[b]["gap"] for b in BUCKET_ORDER]
        train_contrib = [values[b]["train_contrib"] for b in BUCKET_ORDER]
        val_contrib = [values[b]["val_contrib"] for b in BUCKET_ORDER]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True,
                                 facecolor=PAPER,
                                 gridspec_kw={"height_ratios": [1.3, 1, 1]})
        for ax in axes:
            style_axis(ax)
        axes[0].bar(positions - width / 2, train_loss, width, color="#2d6f9f",
                    alpha=0.82, label="train mean loss")
        axes[0].bar(positions + width / 2, val_loss, width, color="#c4493d",
                    alpha=0.82, label="val mean loss")
        axes[0].set_ylabel("mean loss")
        axes[0].set_title(f"{branch.capitalize()} frequency decomposition · step {last['step']}",
                          loc="left", fontsize=15, fontweight="bold")
        axes[0].legend(frameon=False, loc="upper right", fontsize=9)
        axes[1].bar(positions, gap, color=[BUCKET_COLORS[b] for b in BUCKET_ORDER],
                    alpha=0.88)
        axes[1].axhline(0, color=LINE, linewidth=1)
        axes[1].set_ylabel("val − train")
        axes[2].bar(positions - width / 2, train_contrib, width, color="#2d6f9f",
                    alpha=0.82, label="train frac × loss")
        axes[2].bar(positions + width / 2, val_contrib, width, color="#c4493d",
                    alpha=0.82, label="val frac × loss")
        axes[2].set_ylabel("total contribution")
        axes[2].legend(frameon=False, loc="upper right", fontsize=9)
        axes[2].set_xticks(positions)
        axes[2].set_xticklabels(BUCKET_ORDER, rotation=42, ha="right")
        axes[2].set_xlabel("training hit-count bucket")
        fig.tight_layout(h_pad=1.1)
        save_svg(fig, f"fig_freq_{branch}.svg")


def gen_static_distribution_figure(data):
    points = data.get("input", {}).get("freq_bin", [])
    if not points:
        return
    last = points[-1]
    positions = np.arange(len(BUCKET_ORDER))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.8), sharex=True,
                             facecolor=PAPER)
    for ax, branch in zip(axes, ["bigram", "trigram"]):
        style_axis(ax)
        values = final_bucket_values(last, branch)
        train = [values[b]["train_frac"] for b in BUCKET_ORDER]
        val = [values[b]["val_frac"] for b in BUCKET_ORDER]
        ax.bar(positions - width / 2, train, width, color="#2d6f9f",
               alpha=0.82, label="train fraction")
        ax.bar(positions + width / 2, val, width, color="#c4493d",
               alpha=0.82, label="val fraction")
        twin = ax.twinx()
        twin.plot(positions, np.cumsum(train), color="#2d6f9f",
                  linestyle=":", marker="o", markersize=2.5, linewidth=1.6)
        twin.plot(positions, np.cumsum(val), color="#c4493d",
                  linestyle=":", marker="o", markersize=2.5, linewidth=1.6)
        twin.set_ylim(0, 1.04)
        twin.set_ylabel("cumulative", color=MUTED)
        twin.tick_params(colors=MUTED, labelsize=8)
        twin.spines["right"].set_color(LINE)
        ax.set_ylabel("token fraction")
        ax.set_title(f"{branch.capitalize()} context frequency", loc="left",
                     fontsize=14, fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(BUCKET_ORDER, rotation=42, ha="right")
    axes[-1].set_xlabel("training hit-count bucket")
    fig.suptitle("Context frequency distribution · all buckets", x=0.06,
                 ha="left", y=0.995, fontsize=16, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_svg(fig, "fig_hitcount_dist.svg")


def gen_fig_gap_by_freq(data):
    """Figure 3: per-frequency-bin train/val/gap loss (KEY figure).

    Interactive: switch between bigram/trigram, per-token/total, train+val/gap views.
    Uses input run's freq_bin_loss.jsonl.
    """
    out = os.path.join(FIGS_DIR, "fig_gap_by_freq.html")
    d = data.get("input", {})
    fb_pts = d.get("freq_bin", [])
    if not fb_pts:
        print("[fig] fig_gap_by_freq: no freq_bin data, skipping")
        return

    # Extract time series per bucket for each branch
    series = {}  # branch -> bucket -> {steps, train_loss, val_loss, train_frac, val_frac}
    for branch in ["bigram", "trigram"]:
        series[branch] = {}
        for b in BUCKET_ORDER:
            series[branch][b] = {"steps": [], "train_loss": [], "val_loss": [],
                                 "train_frac": [], "val_frac": []}
    for pt in fb_pts:
        step = pt["step"]
        for branch in ["bigram", "trigram"]:
            for b in BUCKET_ORDER:
                td = pt["train"][branch].get(b, {"token_count": 0, "frac": 0, "mean_loss": 0, "total_contrib": 0})
                vd = pt["val"][branch].get(b, {"token_count": 0, "frac": 0, "mean_loss": 0, "total_contrib": 0})
                s = series[branch][b]
                s["steps"].append(step)
                s["train_loss"].append(td["mean_loss"])
                s["val_loss"].append(vd["mean_loss"])
                s["train_frac"].append(td["frac"])
                s["val_frac"].append(vd["frac"])

    body = """
    <div class="controls">
      <b>曲线:</b> <button class="active" onclick="setCurve('both')">train+val</button>
      <button onclick="setCurve('gap')">gap</button>
      <b>指标:</b> <button class="active" onclick="setMetric('per_token')">per-token loss</button>
      <button onclick="setMetric('total')">总贡献</button>
      <b>context:</b> <button class="active" onclick="setBranch('bigram')">bigram</button>
      <button onclick="setBranch('trigram')">trigram</button>
    </div>
    <div id="freq_chart" class="chart"></div>
    """
    script = f"""
    var series = {json.dumps(series)};
    var bucketOrder = {json.dumps(BUCKET_ORDER)};
    var bucketColors = {json.dumps(BUCKET_COLORS)};
    var curCurve = "both";
    var curMetric = "per_token";
    var curBranch = "bigram";

    function getTraces() {{
      var traces = [];
      var data = series[curBranch];
      for (var i = 0; i < bucketOrder.length; i++) {{
        var b = bucketOrder[i];
        var d = data[b];
        if (d.steps.length === 0) continue;
        var y;
        if (curCurve === "gap") {{
          y = d.steps.map(function(s, j) {{ return d.val_loss[j] - d.train_loss[j]; }});
        }} else if (curMetric === "per_token") {{
          // both: show val only for clarity, plus train as dashed
          y = d.val_loss;
        }} else {{
          // total contrib
          y = d.steps.map(function(s, j) {{ return d.val_frac[j] * d.val_loss[j]; }});
        }}
        traces.push({{
          x: d.steps, y: y, mode: "lines+markers", name: b,
          line: {{color: bucketColors[b], width: 2}},
          visible: curCurve === "both" ? true : true
        }});
        if (curCurve === "both" && curMetric === "per_token") {{
          traces.push({{
            x: d.steps, y: d.train_loss, mode: "lines", name: b + " (train)",
            line: {{color: bucketColors[b], width: 1, dash: "dash"}}, visible: false
          }});
        }}
      }}
      return traces;
    }}

    function plot() {{
      var traces = getTraces();
      var shapes = [
        {{type: "line", x0: 337, x1: 337, y0: 0, y1: 1, yref: "paper", line: {{color: "#ccc", dash: "dot"}}}},
        {{type: "line", x0: 686, x1: 686, y0: 0, y1: 1, yref: "paper", line: {{color: "#ccc", dash: "dot"}}}}
      ];
      var title = curBranch + " | " + (curCurve === "gap" ? "gap (val-train)" : (curMetric === "per_token" ? "val loss (per-token)" : "val total contrib"));
      Plotly.newPlot("freq_chart", traces, {{
        title: title, xaxis: {{title: "step"}}, yaxis: {{title: curMetric === "total" ? "frac × loss" : "loss"}},
        margin: {{l:60,r:30,t:50,b:50}}, shapes: shapes, legend: {{x: 0.02, y: 0.98, font: {{size: 9}}}}
      }});
    }}

    function setCurve(c) {{ curCurve = c; document.querySelectorAll('.controls button').forEach(function(b){{b.classList.remove('active')}}); plot(); }}
    function setMetric(m) {{ curMetric = m; plot(); }}
    function setBranch(b) {{ curBranch = b; plot(); }}
    plot();
    """
    html = HTML_WRAP.format(title="频率 bin 分解：train / val / gap", note="input 注入 run。每条线 = 一个频率桶。novel = train 中未出现的 context（仅 val）。epoch2@337, epoch3@686。",
                            body=body, plotly=PLOTLY_HEAD, script=script)
    with open(out, "w") as f:
        f.write(html)
    print(f"[fig] {out}")


def gen_fig_hitcount_dist(data):
    """Figure 4: final per-frequency-bin gap for both branches."""
    out = os.path.join(FIGS_DIR, "fig_hitcount_dist.html")
    d = data.get("input", {})
    fb_pts = d.get("freq_bin", [])
    if not fb_pts:
        print("[fig] fig_hitcount_dist: no freq_bin data, skipping")
        return
    last = fb_pts[-1]

    body = """
    <div class="controls">
      <b>context:</b>
      <button class="active" onclick="plotDist('bigram', this)">bigram</button>
      <button onclick="plotDist('trigram', this)">trigram</button>
    </div>
    <div id="dist_chart" class="chart"></div>
    """
    gap_data = {"bigram": [], "trigram": []}
    for branch in ["bigram", "trigram"]:
        for b in BUCKET_ORDER:
            vd = last["val"][branch].get(b, {})
            td = last["train"][branch].get(b, {})
            train_loss = td.get("mean_loss", 0)
            val_loss = vd.get("mean_loss", 0)
            gap_data[branch].append({
                "bucket": b,
                "gap": val_loss - train_loss,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_frac": td.get("frac", 0),
                "val_frac": vd.get("frac", 0),
            })

    script = f"""
    var gapData = {json.dumps(gap_data)};
    var bucketOrder = {json.dumps(BUCKET_ORDER)};

    function plotDist(branch, button) {{
      if (button) {{
        document.querySelectorAll('.controls button').forEach(function(b) {{
          b.classList.remove('active');
        }});
        button.classList.add('active');
      }}
      var d = gapData[branch];
      var trace = {{
        x: d.map(function(x){{return x.bucket}}),
        y: d.map(function(x){{return x.gap}}),
        customdata: d.map(function(x){{return [x.train_loss, x.val_loss, x.train_frac, x.val_frac]}}),
        type: "bar",
        name: branch,
        marker: {{color: branch === "bigram" ? "#353d79" : "#c4493d"}},
        hovertemplate: "bucket=%{{x}}<br>gap=%{{y:.3f}}<br>train loss=%{{customdata[0]:.3f}}<br>val loss=%{{customdata[1]:.3f}}<br>train fraction=%{{customdata[2]:.2%}}<br>val fraction=%{{customdata[3]:.2%}}<extra></extra>"
      }};
      Plotly.newPlot("dist_chart", [trace], {{
        title: "末态频率桶 gap · " + branch,
        xaxis: {{title: "hit count bucket", type: "category", categoryorder: "array",
                categoryarray: bucketOrder, tickangle: -42}},
        yaxis: {{title: "final gap = val loss − train loss", zeroline: true,
                 zerolinecolor: "#232426", zerolinewidth: 1}},
        margin: {{l:70,r:30,t:50,b:80}}, showlegend: true
      }});
    }}
    plotDist("bigram", document.querySelector('.controls button'));
    """
    html = HTML_WRAP.format(title="各命中频次桶的末态 gap", note="最后一个评估 checkpoint 的 per-bucket gap：val loss − train loss。悬停可查看桶占比和 train/val loss。",
                            body=body, plotly=PLOTLY_HEAD, script=script)
    with open(out, "w") as f:
        f.write(html)
    print(f"[fig] {out}")


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    data = load_all()
    for key, d in data.items():
        print(f"{key}: train_log={len(d['train_log'])} table_norm={len(d['table_norm'])} freq_bin={len(d['freq_bin'])}")
    gen_static_loss_figures(data)
    gen_static_norm_figures(data)
    gen_static_frequency_figures(data)
    gen_static_distribution_figure(data)
    gen_fig_gap_loss(data)
    gen_fig_loss_norm(data)
    gen_fig_gap_by_freq(data)
    gen_fig_hitcount_dist(data)
    if MIRROR_FIGS_DIR:
        os.makedirs(MIRROR_FIGS_DIR, exist_ok=True)
        for name in ["fig_gap_loss.html", "fig_loss_norm.html",
                     "fig_gap_by_freq.html", "fig_hitcount_dist.html"]:
            source = os.path.join(FIGS_DIR, name)
            target = os.path.join(MIRROR_FIGS_DIR, name)
            if os.path.exists(source):
                with open(source) as source_file, open(target, "w") as target_file:
                    target_file.write(source_file.read())
    print("[done] all figures generated")


if __name__ == "__main__":
    main()
