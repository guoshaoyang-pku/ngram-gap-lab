#!/usr/bin/env python3
"""ngram-gap-lab · docs/plot_scripts/gen_all_figures.py

Generate all figures for the blog from training outputs:
  1. fig_gap_loss.html       — v/y/input gap + loss curves
  2. fig_loss_norm.html       — table norm ↔ loss time alignment
  3. fig_gap_by_freq.html     — per-frequency-bin train/val/gap (KEY figure)
  4. fig_hitcount_dist.html   — hit count distribution + cumulative

Reads from data/runs/<run_id>/train_log.jsonl, table_norm.jsonl, freq_bin_loss.jsonl.
Outputs to docs/figs/*.html (interactive plotly).
"""
import json
import os
import sys
import glob
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs")
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs")

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
    """Figure 4: hit count distribution (bar + cumulative) from freq_index.

    Uses the last freq_bin point to get token fractions per bucket.
    """
    out = os.path.join(FIGS_DIR, "fig_hitcount_dist.html")
    d = data.get("input", {})
    fb_pts = d.get("freq_bin", [])
    if not fb_pts:
        print("[fig] fig_hitcount_dist: no freq_bin data, skipping")
        return
    last = fb_pts[-1]

    body = '<div id="dist_chart" class="chart"></div>'
    bar_data = {"bigram": [], "trigram": []}
    for branch in ["bigram", "trigram"]:
        for b in BUCKET_ORDER:
            vd = last["val"][branch].get(b, {"frac": 0})
            td = last["train"][branch].get(b, {"frac": 0})
            bar_data[branch].append({"bucket": b, "train_frac": td["frac"], "val_frac": vd["frac"]})

    script = f"""
    var barData = {json.dumps(bar_data)};
    var bucketOrder = {json.dumps(BUCKET_ORDER)};
    var bucketColors = {json.dumps(BUCKET_COLORS)};

    function plotDist(branch) {{
      var d = barData[branch];
      var trainTrace = {{x: d.map(function(x){{return x.bucket}}), y: d.map(function(x){{return x.train_frac}}), type: "bar", name: "train", marker: {{color: "#2196F3"}}}};
      var valTrace = {{x: d.map(function(x){{return x.bucket}}), y: d.map(function(x){{return x.val_frac}}), type: "bar", name: "val", marker: {{color: "#F44336"}}}};
      // cumulative
      var cumTrain = []; var cumVal = []; var s1=0; var s2=0;
      for (var i=0; i<d.length; i++) {{ s1 += d[i].train_frac; s2 += d[i].val_frac; cumTrain.push(s1); cumVal.push(s2); }}
      var cumTrainTrace = {{x: d.map(function(x){{return x.bucket}}), y: cumTrain, mode: "lines+markers", name: "train (cumul)", line: {{color: "#2196F3", dash: "dot"}}, yaxis: "y2"}};
      var cumValTrace = {{x: d.map(function(x){{return x.bucket}}), y: cumVal, mode: "lines+markers", name: "val (cumul)", line: {{color: "#F44336", dash: "dot"}}, yaxis: "y2"}};
      Plotly.newPlot("dist_chart", [trainTrace, valTrace, cumTrainTrace, cumValTrace], {{
        title: branch + " context 频次分布 (train vs val)", barmode: "group",
        xaxis: {{title: "hit count bucket"}}, yaxis: {{title: "token fraction"}},
        yaxis2: {{title: "cumulative", side: "right", overlaying: "y"}},
        margin: {{l:60,r:60,t:50,b:50}}, legend: {{x: 0.02, y: 0.98}}
      }});
    }}
    plotDist("bigram");
    """
    html = HTML_WRAP.format(title="命中频次分布", note="train vs val 的 n-gram context 频次分布（bar = 占比，虚线 = 累积分布）",
                            body=body, plotly=PLOTLY_HEAD, script=script)
    with open(out, "w") as f:
        f.write(html)
    print(f"[fig] {out}")


def main():
    os.makedirs(FIGS_DIR, exist_ok=True)
    data = load_all()
    for key, d in data.items():
        print(f"{key}: train_log={len(d['train_log'])} table_norm={len(d['table_norm'])} freq_bin={len(d['freq_bin'])}")
    gen_fig_gap_loss(data)
    gen_fig_loss_norm(data)
    gen_fig_gap_by_freq(data)
    gen_fig_hitcount_dist(data)
    print("[done] all figures generated")


if __name__ == "__main__":
    main()
