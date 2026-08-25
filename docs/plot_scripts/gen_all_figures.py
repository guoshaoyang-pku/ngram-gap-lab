#!/usr/bin/env python3
"""ngram-gap-lab · docs/plot_scripts/gen_all_figures.py

Generate the canonical figures for the blog from training outputs:
  1. fig_gap.svg / fig_loss.svg — v/y/input gap and loss curves
  2. fig_table_norm.svg / fig_input_alignment.svg — norm and loss alignment
  3. fig_freq_*.svg — dual-y-axis figure: train/val token fraction bars (left) + train/val mean loss & final gap curves (right)
  4. fig_loss_norm.html — one combined loss/gap/table-RMS Plotly figure
  5. fig_gap_by_freq.html plus Log-x and Log-log frequency-to-gap views

The hit-count distribution generator remains available as a standalone
diagnostic, but is not part of the canonical public-guide output.

Reads from data/runs_fixed/<run_id>/train_log.jsonl, table_norm.jsonl, freq_bin_loss.jsonl.
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
RUNS_DIR = os.environ.get("NGLAB_RUNS_DIR", os.path.join(REPO_ROOT, "data", "runs_fixed"))
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs", "main")
MIRROR_FIGS_DIR = os.environ.get("NGRAM_GAP_BLOG_FIGS_DIR")
FALLBACK_BLOG_DIR = os.environ.get(
    "NGRAM_GAP_BLOG_DIR",
    os.path.join(os.path.dirname(REPO_ROOT), "guoshaoyang-pku.github.io",
                 "blogs", "ngram-gap-mechanism-guide"),
)

# v2 new standard (table β₂=0.99 · table_lr_scale=2.0 · bf16+compile).
# Colors follow the ngram-gap-plotting skill RUN_COLORS.
RUNS = {
    "v": {"label": "v (ResFormer, add to V)", "color": "#b67524", "dir": "nglab1x_v_v2_fixed"},
    "y": {"label": "y (post-attn residual)", "color": "#c4493d", "dir": "nglab1x_y_v2_fixed"},
    "input": {"label": "input (over-encoding)", "color": "#2d6f9f", "dir": "nglab1x_input_v2_fixed"},
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
  .chart {{ width: 100%; max-width: 900px; height: 320px; margin: 10px 0; }}
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
        color = info["color"]
        gap_clean, gap_bad = clean_series([p["gap"] for p in pts])
        train_clean, train_bad = clean_series([p["train_loss"] for p in pts])
        val_clean, val_bad = clean_series([p["val_loss"] for p in pts])
        gap_raw = [p["gap"] for p in pts]
        train_raw = [p["train_loss"] for p in pts]
        val_raw = [p["val_loss"] for p in pts]
        traces_gap.append({"x": x, "y": smooth(gap_clean), "mode": "lines",
                           "name": info["label"], "line": {"color": color, "width": 2}})
        traces_gap.append({"x": x, "y": raw_scatter(gap_raw, gap_bad), "mode": "markers",
                           "name": info["label"] + " (raw)", "showlegend": False,
                           "marker": {"color": color, "size": 3.5, "opacity": 0.35}})
        traces_loss.append({"x": x, "y": smooth(train_clean), "mode": "lines",
                            "name": info["label"] + " (train)",
                            "line": {"color": color, "width": 1.5, "dash": "dash"}})
        traces_loss.append({"x": x, "y": raw_scatter(train_raw, train_bad), "mode": "markers",
                            "name": info["label"] + " (train raw)", "showlegend": False,
                            "marker": {"color": color, "size": 3.5, "opacity": 0.35}})
        traces_loss.append({"x": x, "y": smooth(val_clean), "mode": "lines",
                            "name": info["label"] + " (val)",
                            "line": {"color": color, "width": 2}})
        traces_loss.append({"x": x, "y": raw_scatter(val_raw, val_bad), "mode": "markers",
                            "name": info["label"] + " (val raw)", "showlegend": False,
                            "marker": {"color": color, "size": 3.5, "opacity": 0.35}})

    eb = epoch_boundary_pairs(next(iter(data.values()))["train_log"])
    epoch_shapes = [
        {"type": "line", "x0": step, "x1": step, "y0": 0, "y1": 1,
         "yref": "paper", "line": {"color": "#ccc", "dash": "dot"}}
        for step, _ in eb
    ]
    epoch_annots = [
        {"x": step, "y": 0.95, "yref": "paper", "text": label,
         "showarrow": False, "font": {"size": 10, "color": "#999"}}
        for step, label in eb
    ]
    body = '<div id="gap_chart" class="chart"></div><div id="loss_chart" class="chart"></div>'
    script = f"""
    var gapData = {json.dumps(traces_gap)};
    var lossData = {json.dumps(traces_loss)};
    var shapes = {json.dumps(epoch_shapes)};
    var annots = {json.dumps(epoch_annots)};
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
    gap_raw = [p["gap"] for p in train_pts]
    train_raw = [p["train_loss"] for p in train_pts]
    val_raw = [p["val_loss"] for p in train_pts]
    gap_clean, gap_bad = clean_series(gap_raw)
    train_clean, train_bad = clean_series(train_raw)
    val_clean, val_bad = clean_series(val_raw)
    gap = smooth(gap_clean)
    train_loss = smooth(train_clean)
    val_loss = smooth(val_clean)

    x_norm = [p["step"] for p in norm_pts]
    # use bigram layer_01 table_0 rms
    bg_raw = [p.get("bigram.layer_01.table_0.rms", 0) for p in norm_pts]
    tg_raw = [p.get("trigram.layer_01.table_0.rms", 0) for p in norm_pts]
    bg_clean, bg_bad = clean_series(bg_raw)
    tg_clean, tg_bad = clean_series(tg_raw)
    bg_rms = smooth(bg_clean)
    tg_rms = smooth(tg_clean)

    epoch_shapes = [
        {"type": "line", "x0": step, "x1": step, "y0": 0, "y1": 1,
         "yref": "paper", "line": {"color": "#ccc", "dash": "dot"}}
        for step, _ in epoch_boundary_pairs(train_pts)
    ]
    traces = [
        {"x": x_loss, "y": train_loss, "mode": "lines", "name": "train loss",
         "line": {"color": "#3c8d5a", "width": 1.8, "dash": "dash"}, "yaxis": "y"},
        {"x": x_loss, "y": raw_scatter(train_raw, train_bad), "mode": "markers",
         "name": "train loss (raw)", "showlegend": False,
         "marker": {"color": "#3c8d5a", "size": 3.5, "opacity": 0.35}, "yaxis": "y"},
        {"x": x_loss, "y": val_loss, "mode": "lines", "name": "val loss",
         "line": {"color": "#d97932", "width": 2}, "yaxis": "y"},
        {"x": x_loss, "y": raw_scatter(val_raw, val_bad), "mode": "markers",
         "name": "val loss (raw)", "showlegend": False,
         "marker": {"color": "#d97932", "size": 3.5, "opacity": 0.35}, "yaxis": "y"},
        {"x": x_loss, "y": gap, "mode": "lines", "name": "gap",
         "line": {"color": "#353d79", "width": 2}, "yaxis": "y2"},
        {"x": x_loss, "y": raw_scatter(gap_raw, gap_bad), "mode": "markers",
         "name": "gap (raw)", "showlegend": False,
         "marker": {"color": "#353d79", "size": 3.5, "opacity": 0.35}, "yaxis": "y2"},
        {"x": x_norm, "y": bg_rms, "mode": "lines", "name": "bigram table RMS",
         "line": {"color": "#2d6f9f", "width": 2}, "yaxis": "y3"},
        {"x": x_norm, "y": raw_scatter(bg_raw, bg_bad), "mode": "markers",
         "name": "bigram table RMS (raw)", "showlegend": False,
         "marker": {"color": "#2d6f9f", "size": 3.5, "opacity": 0.35}, "yaxis": "y3"},
        {"x": x_norm, "y": tg_rms, "mode": "lines", "name": "trigram table RMS",
         "line": {"color": "#c4493d", "width": 2}, "yaxis": "y3"},
        {"x": x_norm, "y": raw_scatter(tg_raw, tg_bad), "mode": "markers",
         "name": "trigram table RMS (raw)", "showlegend": False,
         "marker": {"color": "#c4493d", "size": 3.5, "opacity": 0.35}, "yaxis": "y3"},
    ]
    body = '<div id="norm_chart" class="chart" style="height: 320px"></div>'
    script = f"""
    var traces = {json.dumps(traces)};
    var shapes = {json.dumps(epoch_shapes)};
    Plotly.newPlot("norm_chart", traces, {{
        title: "Loss, Gap, and N-gram Table RMS (input run)",
        xaxis: {{title: "step"}},
        yaxis: {{title: "loss", side: "left", domain: [0, 1]}},
        yaxis2: {{title: "gap", side: "right", overlaying: "y", position: 1}},
        yaxis3: {{title: "table RMS", side: "right", overlaying: "y", position: 0.94, anchor: "free"}},
        height: 320, margin: {{l:65,r:105,t:42,b:45}}, shapes: shapes, legend: {{x: 0.02, y: 0.98}}
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


def _parse_bound(bucket):
    if bucket == "novel":
        return (0, 0)
    if bucket.endswith("+"):  # open-ended, e.g. "5k+" / "10k+"
        base = bucket[:-1]
        v = int(float(base[:-1]) * 1000) if base.endswith("k") else int(base)
        return (v, v * 2)
    if "-" not in bucket:
        v = int(bucket)
        return (v, v)
    lo_s, hi_s = bucket.split("-")

    def val(tok):
        if tok.endswith("k"):
            return int(float(tok[:-1]) * 1000)
        return int(tok)

    lo = val(lo_s)
    if hi_s.endswith("k+"):
        hi = int(float(hi_s[:-2]) * 1000) * 2
    elif hi_s.endswith("k"):
        hi = int(float(hi_s[:-1]) * 1000)
    else:
        hi = int(hi_s)
    return (lo, hi)


def sync_buckets(data):
    """Adopt the bucket scheme actually present in the run data (v2 wave uses
    24 finer buckets instead of the historical 15)."""
    global BUCKET_ORDER, BUCKET_BOUNDS, BUCKET_COLORS
    d = data.get("input", {})
    pts = d.get("freq_bin", [])
    if not pts:
        return
    branches = pts[-1].get("train", {})
    order = list(branches.get("bigram", {}).keys())
    if not order or order == BUCKET_ORDER:
        return
    BUCKET_ORDER = order
    BUCKET_BOUNDS = {b: _parse_bound(b) for b in order}
    ramp = plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(order) - 1)))
    colors = {"novel": "#E91E63"}
    for i, b in enumerate(order[1:]):
        r, g, bl, _ = ramp[i]
        colors[b] = f"#{int(r*255):02X}{int(g*255):02X}{int(bl*255):02X}"
    BUCKET_COLORS = colors
    print(f"[fig] bucket scheme synced from data: {len(order)} buckets")

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


def epoch_boundary_pairs(train_log):
    """(step, label) pairs where a new training epoch starts, from the epoch field."""
    pairs = []
    prev = None
    for p in train_log:
        ep = p.get("epoch")
        if prev is not None and ep is not None and ep != prev:
            pairs.append((p["step"], f"epoch {ep}"))
        prev = ep
    return pairs


def smooth(pts, window=7):
    """Centered moving average to soften single-step noise / early spikes."""
    n = len(pts)
    if n == 0:
        return pts
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(pts[lo:hi]) / (hi - lo))
    return out


def clean_series(values, window=7, max_dev=1.0):
    """Replace isolated outliers (deviation from local median > max_dev) with
    the local median, so single-step spikes do not distort smoothing or the
    axis range. Returns (cleaned, outlier_indices)."""
    n = len(values)
    half = window // 2
    cleaned = list(values)
    bad = []
    for i, v in enumerate(values):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        win = values[lo:hi]
        med = sorted(win)[len(win) // 2]
        if abs(v - med) > max_dev:
            cleaned[i] = med
            bad.append(i)
    return cleaned, bad


def raw_scatter(values, bad):
    """Raw values with outlier slots replaced by None (skipped by Plotly)."""
    return [None if i in bad else v for i, v in enumerate(values)]


def add_epoch_lines(ax, boundaries=None):
    boundaries = boundaries or [(337, "epoch 2"), (686, "epoch 3")]
    for step, label in boundaries:
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
        raw = [p["gap"] for p in pts]
        cleaned, bad = clean_series(raw)
        ax.scatter(x, [np.nan if i in bad else v for i, v in enumerate(raw)],
                   s=8, color=RUN_COLORS[key], alpha=0.3, linewidths=0, zorder=2)
        ax.plot(x, smooth(cleaned), color=RUN_COLORS[key], linewidth=2.2,
                marker="o", markersize=2.8, label=info["label"])
    add_epoch_lines(ax, epoch_boundary_pairs(data["v"]["train_log"]))
    ax.set_title("Train / validation gap (train = online batch loss)", loc="left",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("val (fixed) − train (online)")
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
        tr_raw = [p["train_loss"] for p in pts]
        val_raw = [p["val_loss"] for p in pts]
        tr_clean, tr_bad = clean_series(tr_raw)
        val_clean, val_bad = clean_series(val_raw)
        ax.scatter(x, [np.nan if i in tr_bad else v for i, v in enumerate(tr_raw)],
                   s=6, color=color, alpha=0.22, linewidths=0, zorder=2)
        ax.scatter(x, [np.nan if i in val_bad else v for i, v in enumerate(val_raw)],
                   s=6, color=color, alpha=0.22, linewidths=0, zorder=2)
        ax.plot(x, smooth(tr_clean), color=color, linewidth=1.5,
                linestyle="--", alpha=0.72, label=f"{key} train")
        ax.plot(x, smooth(val_clean), color=color, linewidth=2.1,
                label=f"{key} val")
    add_epoch_lines(ax, epoch_boundary_pairs(data["v"]["train_log"]))
    ax.set_title("Train / validation loss (train dashed = online batch loss)", loc="left",
                 fontsize=15, fontweight="bold")
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
                raw = [p.get(key, 0) for p in norm_pts]
                cleaned, bad = clean_series(raw)
                ax.scatter(x, [np.nan if i in bad else v for i, v in enumerate(raw)],
                           s=6, color=color, alpha=0.22, linewidths=0, zorder=2)
                ax.plot(x, smooth(cleaned), color=color,
                        linewidth=2.2, label=label)
    add_epoch_lines(ax, epoch_boundary_pairs(d["train_log"]))
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
    tr_raw = [p["train_loss"] for p in pts]
    val_raw = [p["val_loss"] for p in pts]
    gap_raw = [p["gap"] for p in pts]
    tr_clean, tr_bad = clean_series(tr_raw)
    val_clean, val_bad = clean_series(val_raw)
    gap_clean, gap_bad = clean_series(gap_raw)
    ax.scatter(x, [np.nan if i in tr_bad else v for i, v in enumerate(tr_raw)],
               s=6, color="#2d6f9f", alpha=0.22, linewidths=0, zorder=2)
    ax.scatter(x, [np.nan if i in val_bad else v for i, v in enumerate(val_raw)],
               s=6, color="#c4493d", alpha=0.22, linewidths=0, zorder=2)
    ax.plot(x, smooth(tr_clean), color="#2d6f9f", linewidth=1.6,
            linestyle="--", label="train loss")
    ax.plot(x, smooth(val_clean), color="#c4493d", linewidth=2.1,
            label="val loss")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.scatter(x, [np.nan if i in gap_bad else v for i, v in enumerate(gap_raw)],
                s=6, color=ANCHOR, alpha=0.22, linewidths=0, zorder=2)
    ax2.plot(x, smooth(gap_clean), color=ANCHOR, linewidth=2.0,
             label="gap")
    ax2.set_ylabel("gap", color=ANCHOR)
    ax2.tick_params(colors=ANCHOR, labelsize=9)
    ax2.spines["right"].set_color(ANCHOR)
    add_epoch_lines(ax, epoch_boundary_pairs(d["train_log"]))
    ax.set_title("Input run: loss and gap alignment (train = online batch loss)",
                 loc="left", fontsize=15, fontweight="bold")
    ax.set_xlabel("step")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False,
              loc="upper left", fontsize=9)
    fig.tight_layout()
    save_svg(fig, "fig_input_alignment.svg")


def gen_static_combined_norm_figure(data):
    d = data.get("input", {})
    if not d.get("train_log"):
        return
    train_pts = d["train_log"]
    norm_pts = d.get("table_norm", [])
    x_loss = [p["step"] for p in train_pts]
    tr_raw = [p["train_loss"] for p in train_pts]
    val_raw = [p["val_loss"] for p in train_pts]
    gap_raw = [p["gap"] for p in train_pts]
    tr_clean, tr_bad = clean_series(tr_raw)
    val_clean, val_bad = clean_series(val_raw)
    gap_clean, gap_bad = clean_series(gap_raw)
    train_loss = smooth(tr_clean)
    val_loss = smooth(val_clean)
    gap = smooth(gap_clean)
    fig, ax = plt.subplots(figsize=(10.8, 5.0), facecolor=PAPER)
    style_axis(ax)
    ax.scatter(x_loss, [np.nan if i in tr_bad else v for i, v in enumerate(tr_raw)],
               s=6, color="#3c8d5a", alpha=0.22, linewidths=0, zorder=2)
    ax.scatter(x_loss, [np.nan if i in val_bad else v for i, v in enumerate(val_raw)],
               s=6, color="#d97932", alpha=0.22, linewidths=0, zorder=2)
    ax.plot(x_loss, train_loss, color="#3c8d5a", linewidth=1.7,
            linestyle="--", label="train loss")
    ax.plot(x_loss, val_loss, color="#d97932", linewidth=2.1,
            label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax2 = ax.twinx()
    ax2.scatter(x_loss, [np.nan if i in gap_bad else v for i, v in enumerate(gap_raw)],
                s=6, color=ANCHOR, alpha=0.22, linewidths=0, zorder=2)
    ax2.plot(x_loss, gap, color=ANCHOR, linewidth=2.0, label="gap")
    ax2.set_ylabel("gap", color=ANCHOR)
    ax2.tick_params(colors=ANCHOR, labelsize=9)
    ax2.spines["right"].set_color(ANCHOR)
    if norm_pts:
        x_norm = [p["step"] for p in norm_pts]
        for prefix, label, color in [
            ("bigram", "bigram table RMS", "#2d6f9f"),
            ("trigram", "trigram table RMS", "#c4493d"),
        ]:
            key = next((k for k in norm_pts[0]
                        if k.startswith(prefix) and k.endswith("table_0.rms")), None)
            if key:
                ax3 = ax.twinx()
                ax3.spines["right"].set_position(("axes", 1.10 if prefix == "bigram" else 1.18))
                rms_raw = [p.get(key, 0) for p in norm_pts]
                rms_clean, rms_bad = clean_series(rms_raw)
                ax3.scatter(x_norm, [np.nan if i in rms_bad else v
                                     for i, v in enumerate(rms_raw)],
                            s=5, color=color, alpha=0.2, linewidths=0, zorder=2)
                ax3.plot(x_norm, smooth(rms_clean),
                         color=color, linewidth=2.0, label=label)
                ax3.set_ylabel("table RMS", color=color)
                ax3.tick_params(colors=color, labelsize=8)
                ax3.spines["right"].set_color(color)
    add_epoch_lines(ax, epoch_boundary_pairs(train_pts))
    ax.set_title("Input run: loss, gap, and n-gram table RMS (train = online)",
                 loc="left", fontsize=15, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles += handles2
    labels += labels2
    for axis in fig.axes[2:]:
        h, l = axis.get_legend_handles_labels()
        handles += h
        labels += l
    ax.legend(handles, labels, frameon=False, ncol=3,
              loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save_svg(fig, "fig_loss_norm.svg")


def gen_static_log_figures(data):
    points = data.get("input", {}).get("freq_bin", [])
    if not points:
        return
    last = points[-1]
    bounds = BUCKET_BOUNDS
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), facecolor=PAPER)
    for ax, y_log, title in [
        (axes[0], False, "Final gap vs frequency · log-x"),
        (axes[1], True, "Final gap vs frequency · log-log"),
    ]:
        style_axis(ax)
        for branch, color in [("bigram", "#353d79"), ("trigram", "#c4493d")]:
            xs, ys = [], []
            for bucket in BUCKET_ORDER:
                if bucket == "novel":
                    continue
                low, high = bounds[bucket]
                train = last["train"][branch].get(bucket, {})
                val = last["val"][branch].get(bucket, {})
                if not train.get("token_count") or not val.get("token_count"):
                    continue
                gap = val.get("mean_loss", 0) - train.get("mean_loss", 0)
                if gap <= 0:
                    continue
                xs.append((low * high) ** 0.5)
                ys.append(gap)
            ax.plot(xs, ys, "o-", color=color, linewidth=1.8,
                    markersize=4, label=branch)
        ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")
        ax.set_xlabel("training hit count")
        ax.set_ylabel("final gap")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(w_pad=2.0)
    save_svg(fig, "fig_gap_vs_frequency.svg")

    for y_log, name, title in [
        (False, "fig_gap_vs_frequency_logx.svg", "Final gap vs frequency · log-x"),
        (True, "fig_gap_vs_frequency_loglog.svg", "Final gap vs frequency · log-log"),
    ]:
        fig, ax = plt.subplots(figsize=(10.8, 4.8), facecolor=PAPER)
        style_axis(ax)
        for branch, color in [("bigram", "#353d79"), ("trigram", "#c4493d")]:
            xs, ys = [], []
            for bucket in BUCKET_ORDER:
                if bucket == "novel":
                    continue
                low, high = bounds[bucket]
                train = last["train"][branch].get(bucket, {})
                val = last["val"][branch].get(bucket, {})
                if not train.get("token_count") or not val.get("token_count"):
                    continue
                gap = val.get("mean_loss", 0) - train.get("mean_loss", 0)
                if gap <= 0:
                    continue
                xs.append((low * high) ** 0.5)
                ys.append(gap)
            ax.plot(xs, ys, "o-", color=color, linewidth=1.9,
                    markersize=4, label=branch)
        ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")
        ax.set_xlabel("training hit count")
        ax.set_ylabel("final gap")
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
        ax.legend(frameon=False, loc="best", fontsize=9)
        fig.tight_layout()
        save_svg(fig, name)


SLICE_STEPS = [300, 500, 750, 1000, 1250, 2000]
SLICE_BRANCH_COLORS = {
    "bigram": ["#c6cfe8", "#94a3d1", "#7484bd", "#55619f", "#3f4985", "#353d79"],
    "trigram": ["#efcfcb", "#e5a8a1", "#dc8a81", "#d26b60", "#c9564a", "#b53a2e"],
}
BUCKET_BOUNDS = {
    "novel": (0, 0), "1": (1, 1), "2": (2, 2), "3": (3, 3),
    "4": (4, 4), "5": (5, 5), "6-10": (6, 10), "11-20": (11, 20),
    "21-50": (21, 50), "51-100": (51, 100), "101-200": (101, 200),
    "201-500": (201, 500), "501-1k": (501, 1000),
    "1k-5k": (1000, 5000), "5k+": (5000, 10000),
}


def fit_power_law(xs, ys):
    lx = np.log(np.asarray(xs, float))
    ly = np.log(np.asarray(ys, float))
    alpha, logc = np.polyfit(lx, ly, 1)
    return alpha, np.exp(logc)


def gen_static_step_slice_figures(data):
    """Gap vs frequency at multiple step slices (log-log), to test whether the
    power-law exponent is stable across training."""
    points = data.get("input", {}).get("freq_bin", [])
    if not points:
        return
    by_step = {p["step"]: p for p in points}
    missing = [s for s in SLICE_STEPS if s not in by_step]
    if missing:
        print(f"[fig] fig_gap_vs_frequency_steps: missing steps {missing}, skipping")
        return
    branches = ["bigram", "trigram"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), facecolor=PAPER)
    for ax, branch in zip(axes, branches):
        style_axis(ax)
        handles = []
        for i, step in enumerate(SLICE_STEPS):
            pt = by_step[step]
            xs, ys = [], []
            for bucket in BUCKET_ORDER:
                if bucket == "novel":
                    continue
                low, high = BUCKET_BOUNDS[bucket]
                train = pt["train"][branch].get(bucket, {})
                val = pt["val"][branch].get(bucket, {})
                if not train.get("token_count") or not val.get("token_count"):
                    continue
                gap = val.get("mean_loss", 0) - train.get("mean_loss", 0)
                if gap <= 0:
                    continue
                xs.append((low * high) ** 0.5)
                ys.append(gap)
            color = SLICE_BRANCH_COLORS[branch][i]
            if len(xs) >= 3:
                alpha, _ = fit_power_law(xs, ys)
                label = f"step {step} · α={alpha:.2f}"
            else:
                label = f"step {step} · no positive gap yet" if not xs else f"step {step}"
            handle, = ax.plot(xs, ys, "o-", color=color, linewidth=1.8,
                              markersize=3.8, label=label)
            handles.append(handle)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training hit count")
        ax.set_ylabel("gap = val loss − train loss")
        ax.set_title(f"Gap vs frequency · {branch}", loc="left",
                     fontsize=13, fontweight="bold")
        ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=3, columnspacing=1.2,
                  title="slice · power-law α")
        ax.get_legend().get_title().set_fontsize(8.5)
        ax.get_legend().get_title().set_color(MUTED)
    fig.tight_layout(w_pad=2.2, rect=(0, 0.06, 1, 1))
    save_svg(fig, "fig_gap_vs_frequency_steps.svg")


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
        train_frac = [values[b]["train_frac"] for b in BUCKET_ORDER]
        val_frac = [values[b]["val_frac"] for b in BUCKET_ORDER]
        train_loss = [values[b]["train_loss"] for b in BUCKET_ORDER]
        val_loss = [values[b]["val_loss"] for b in BUCKET_ORDER]
        gap = [values[b]["gap"] for b in BUCKET_ORDER]
        # train side has no tokens in "novel" => no train loss / gap there
        no_train = [
            last["train"][branch].get(b, {}).get("token_count", 0) == 0
            for b in BUCKET_ORDER
        ]
        train_loss = [np.nan if missing else v
                      for missing, v in zip(no_train, train_loss)]
        gap = [np.nan if missing else v
               for missing, v in zip(no_train, gap)]

        fig, ax = plt.subplots(figsize=(12, 4.5), facecolor=PAPER)
        style_axis(ax)
        ax.bar(positions - width / 2, train_frac, width, color="#2d6f9f",
               alpha=0.82, label="train fraction")
        ax.bar(positions + width / 2, val_frac, width, color="#c4493d",
               alpha=0.82, label="val fraction")
        ax.set_ylabel("token fraction")
        ax.set_title(f"{branch.capitalize()} frequency decomposition · step {last['step']}"
                     " (train loss = online window)",
                     loc="left", fontsize=14, fontweight="bold")
        ax2 = ax.twinx()
        ax2.plot(positions, train_loss, color="#2d6f9f", linewidth=1.5,
                 linestyle="--", label="train mean loss (online)")
        ax2.plot(positions, val_loss, color="#c4493d", linewidth=2.1,
                 label="val mean loss (fixed)")
        ax2.plot(positions, gap, "o-", color=ANCHOR, linewidth=1.9,
                 markersize=4, label="final gap")
        ax2.axhline(0, color=LINE, linewidth=1)
        ax2.set_ylabel("mean loss / gap", color=ANCHOR)
        ax2.tick_params(colors=ANCHOR, labelsize=9)
        ax2.spines["right"].set_color(ANCHOR)
        ax.set_xticks(positions)
        ax.set_xticklabels(BUCKET_ORDER, rotation=42, ha="right")
        ax.set_xlabel("training hit-count bucket")
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, frameon=False,
                  loc="upper right", fontsize=9)
        fig.tight_layout()
        save_svg(fig, f"fig_freq_{branch}.svg")


def gen_static_distribution_figure(data):
    points = data.get("input", {}).get("freq_bin", [])
    if not points:
        return
    last = points[-1]
    positions = np.arange(len(BUCKET_ORDER))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True,
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

    epoch_shapes = [
        {"type": "line", "x0": step, "x1": step, "y0": 0, "y1": 1,
         "yref": "paper", "line": {"color": "#ccc", "dash": "dot"}}
        for step, _ in epoch_boundary_pairs(next(iter(data.values()))["train_log"])
    ]
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
      var shapes = {json.dumps(epoch_shapes)};
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
    eb_note = "、".join(f"epoch{label.replace('epoch ', '')}@{step}"
                        for step, label in epoch_boundary_pairs(
                            next(iter(data.values()))["train_log"]))
    html = HTML_WRAP.format(title="频率 bin 分解：train / val / gap", note="input 注入 run。每条线 = 一个频率桶。novel = train 中未出现的 context（仅 val）。" + eb_note + "。",
                            body=body, plotly=PLOTLY_HEAD, script=script)
    with open(out, "w") as f:
        f.write(html)
    print(f"[fig] {out}")


def gen_fig_hitcount_dist(data):
    """Figure 4: frequency histogram with final gap on the line axis."""
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
      var trainTrace = {{
        x: d.map(function(x){{return x.bucket}}),
        y: d.map(function(x){{return x.train_frac}}),
        customdata: d.map(function(x){{return [x.train_loss, x.val_loss, x.gap]}}),
        type: "bar",
        name: "train fraction",
        marker: {{color: "#2d6f9f"}},
        hovertemplate: "bucket=%{{x}}<br>train fraction=%{{y:.2%}}<br>train loss=%{{customdata[0]:.3f}}<br>val loss=%{{customdata[1]:.3f}}<br>gap=%{{customdata[2]:.3f}}<extra></extra>"
      }};
      var valTrace = {{
        x: d.map(function(x){{return x.bucket}}),
        y: d.map(function(x){{return x.val_frac}}),
        customdata: d.map(function(x){{return [x.train_loss, x.val_loss, x.gap]}}),
        type: "bar",
        name: "val fraction",
        marker: {{color: "#c4493d"}},
        hovertemplate: "bucket=%{{x}}<br>val fraction=%{{y:.2%}}<br>train loss=%{{customdata[0]:.3f}}<br>val loss=%{{customdata[1]:.3f}}<br>gap=%{{customdata[2]:.3f}}<extra></extra>"
      }};
      var gapTrace = {{
        x: d.map(function(x){{return x.bucket}}),
        y: d.map(function(x){{return x.gap}}),
        customdata: d.map(function(x){{return [x.train_frac, x.val_frac, x.train_loss, x.val_loss]}}),
        mode: "lines+markers",
        name: "final gap",
        line: {{color: "#353d79", width: 3}},
        marker: {{color: "#353d79", size: 7}},
        yaxis: "y2",
        hovertemplate: "bucket=%{{x}}<br>final gap=%{{y:.3f}}<br>train fraction=%{{customdata[0]:.2%}}<br>val fraction=%{{customdata[1]:.2%}}<br>train loss=%{{customdata[2]:.3f}}<br>val loss=%{{customdata[3]:.3f}}<extra></extra>"
      }};
      Plotly.newPlot("dist_chart", [trainTrace, valTrace, gapTrace], {{
        title: "命中频次分布 + 末态 gap · " + branch,
        barmode: "group",
        xaxis: {{title: "hit count bucket", type: "category", categoryorder: "array",
                categoryarray: bucketOrder, tickangle: -42}},
        yaxis: {{title: "token fraction", rangemode: "tozero"}},
        yaxis2: {{title: "final gap = val loss − train loss", side: "right",
                 overlaying: "y", zeroline: true, zerolinecolor: "#232426",
                 zerolinewidth: 1}},
        margin: {{l:70,r:85,t:50,b:90}}, showlegend: true
      }});
    }}
    plotDist("bigram", document.querySelector('.controls button'));
    """
    html = HTML_WRAP.format(title="命中频次分布 + 末态 gap", note="柱：train/val token fraction；曲线：最后一个 checkpoint 的 per-bucket gap = val loss − train loss。",
                            body=body, plotly=PLOTLY_HEAD, script=script)
    with open(out, "w") as f:
        f.write(html)
    print(f"[fig] {out}")


def gen_fig_gap_log(data, output_name="fig_gap_vs_frequency_log.html", y_log=True):
    """Figure 5: final gap against hit count on a logarithmic x-axis."""
    out = os.path.join(FIGS_DIR, output_name)
    d = data.get("input", {})
    fb_pts = d.get("freq_bin", [])
    if not fb_pts:
        print("[fig] fig_gap_vs_frequency_log: no freq_bin data, skipping")
        return
    last = fb_pts[-1]
    bounds = BUCKET_BOUNDS
    log_data = {}
    for branch in ["bigram", "trigram"]:
        rows = []
        for bucket in BUCKET_ORDER:
            if bucket == "novel":
                continue
            low, high = bounds[bucket]
            td = last["train"][branch].get(bucket, {})
            vd = last["val"][branch].get(bucket, {})
            train_loss = td.get("mean_loss", 0)
            val_loss = vd.get("mean_loss", 0)
            train_count = td.get("token_count", 0)
            val_count = vd.get("token_count", 0)
            rows.append({
                "bucket": bucket,
                "x": (low * high) ** 0.5,
                "x_low": low,
                "x_high": high,
                "gap": val_loss - train_loss if train_count > 0 and val_count > 0 else None,
                "train_count": train_count,
                "val_count": val_count,
                "train_frac": td.get("frac", 0),
                "val_frac": vd.get("frac", 0),
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
        log_data[branch] = [
            row for row in rows if row["gap"] is not None and row["gap"] > 0
        ]

    body = """
    <div class="controls">
      <b>context:</b>
      <button class="active" onclick="setVisible('both', this)">both</button>
      <button onclick="setVisible('bigram', this)">bigram</button>
      <button onclick="setVisible('trigram', this)">trigram</button>
    </div>
    <div id="log_gap_chart" class="chart"></div>
    """
    y_title = "final gap = val loss − train loss" + (" (log scale)" if y_log else "")
    y_axis_config = 'type: "log"' if y_log else "zeroline: true"
    script = f"""
    var logData = {json.dumps(log_data)};
    var colors = {{bigram: "#353d79", trigram: "#c4493d"}};
    function makeTrace(branch) {{
      var d = logData[branch];
      return {{
        x: d.map(function(x){{return x.x}}),
        y: d.map(function(x){{return x.gap}}),
        customdata: d.map(function(x){{return [x.bucket, x.x_low, x.x_high, x.train_count, x.val_count, x.train_frac, x.val_frac, x.train_loss, x.val_loss]}}),
        mode: "lines+markers",
        name: branch,
        line: {{color: colors[branch], width: 2.5}},
        marker: {{color: colors[branch], size: 8}},
        error_x: {{
          type: "data",
          symmetric: false,
          array: d.map(function(x){{return x.x_high - x.x}}),
          arrayminus: d.map(function(x){{return x.x - x.x_low}}),
          color: colors[branch],
          thickness: 1.2,
          width: 4
        }},
        hovertemplate: "bucket=%{{customdata[0]}}<br>frequency range=%{{customdata[1]}}–%{{customdata[2]}}<br>gap=%{{y:.3f}}<br>train tokens=%{{customdata[3]:,}}<br>val tokens=%{{customdata[4]:,}}<br>train fraction=%{{customdata[5]:.2%}}<br>val fraction=%{{customdata[6]:.2%}}<br>train mean token loss=%{{customdata[7]:.3f}}<br>val mean token loss=%{{customdata[8]:.3f}}<extra></extra>"
      }};
    }}
    function setVisible(which, button) {{
      if (button) {{
        document.querySelectorAll('.controls button').forEach(function(b){{b.classList.remove('active')}});
        button.classList.add('active');
      }}
      var visibility = which === "both" ? [true, true] :
        (which === "bigram" ? [true, "legendonly"] : ["legendonly", true]);
      Plotly.restyle("log_gap_chart", {{visible: visibility}});
    }}
    Plotly.newPlot("log_gap_chart", [makeTrace("bigram"), makeTrace("trigram")], {{
      title: "Final per-bucket gap vs training hit count",
      xaxis: {{title: "training hit count (log scale)", type: "log", dtick: 1}},
      yaxis: {{title: "{y_title}", {y_axis_config}}},
      margin: {{l:70,r:30,t:50,b:65}}, showlegend: true,
      legend: {{x: 0.02, y: 0.98}}
    }});
    """
    title = "Log–log frequency → gap" if y_log else "Log-x frequency → gap"
    note = (
        "使用原始 15 个真实频率桶；novel 被排除，因为 train hit count=0 时没有 train token loss，"
        "无法定义同桶 gap。x、y 两轴均为对数尺度，仅显示正 gap。"
        if y_log else
        "使用原始 15 个真实频率桶；novel 被排除，因为 train hit count=0 时没有 train token loss，"
        "无法定义同桶 gap。x 轴为对数尺度，y 轴为线性尺度。"
    )
    html = HTML_WRAP.format(
        title=title,
        note=note,
        body=body, plotly=PLOTLY_HEAD, script=script)
    with open(out, "w") as f:
        f.write(html)
    print(f"[fig] {out}")


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    data = load_all()
    sync_buckets(data)
    for key, d in data.items():
        print(f"{key}: train_log={len(d['train_log'])} table_norm={len(d['table_norm'])} freq_bin={len(d['freq_bin'])}")
    gen_static_loss_figures(data)
    gen_static_norm_figures(data)
    gen_static_combined_norm_figure(data)
    gen_static_frequency_figures(data)
    gen_static_log_figures(data)
    gen_static_step_slice_figures(data)
    gen_fig_gap_loss(data)
    gen_fig_loss_norm(data)
    gen_fig_gap_by_freq(data)
    gen_fig_hitcount_dist(data)
    gen_fig_gap_log(data, "fig_gap_vs_frequency_loglog.html", y_log=True)
    gen_fig_gap_log(data, "fig_gap_vs_frequency_logx.html", y_log=False)
    gen_fig_gap_log(data, "fig_gap_vs_frequency_log.html", y_log=False)
    if MIRROR_FIGS_DIR:
        os.makedirs(MIRROR_FIGS_DIR, exist_ok=True)
        for name in ["fig_gap_loss.html", "fig_loss_norm.html",
                     "fig_gap_by_freq.html",
                     "fig_gap_vs_frequency_loglog.html",
                     "fig_gap_vs_frequency_logx.html"]:
            source = os.path.join(FIGS_DIR, name)
            target = os.path.join(MIRROR_FIGS_DIR, name)
            if os.path.exists(source):
                with open(source) as source_file, open(target, "w") as target_file:
                    target_file.write(source_file.read())
    print("[done] all figures generated")


if __name__ == "__main__":
    main()
