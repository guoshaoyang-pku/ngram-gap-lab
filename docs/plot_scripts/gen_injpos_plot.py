#!/usr/bin/env python3
"""ngram-gap-lab · docs/plot_scripts/gen_injpos_plot.py

Generate injection-point ablation plot from train_log.jsonl files.
Reads data/runs/<run_id>/train_log.jsonl and table_norm.jsonl,
produces docs/figs/injpos_ablation.html with gap + loss + table norm.
"""
import json
import os
import sys
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs_fixed")
OUT_HTML = os.path.join(REPO_ROOT, "docs", "figs", "injpos_ablation.html")

RUNS = {
    "v (ResFormer, add to V)": {
        "dir": "nglab_v",
        "color": "#2196F3",
    },
    "y (post-attn residual)": {
        "dir": "nglab_y",
        "color": "#F44336",
    },
    "input (over-encoding, add to wte)": {
        "dir": "nglab_input",
        "color": "#4CAF50",
    },
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


def load_table_norm(path):
    pts = []
    if not os.path.exists(path):
        return pts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pts.append(json.loads(line))
    return pts


data = {}
table_data = {}
for name, info in RUNS.items():
    run_dir = os.path.join(RUNS_DIR, info["dir"])
    train_pts = load_jsonl(os.path.join(run_dir, "train_log.jsonl"))
    table_pts = load_table_norm(os.path.join(run_dir, "table_norm.jsonl"))
    if train_pts:
        data[name] = train_pts
        print(f"{name}: {len(train_pts)} train points, final gap={train_pts[-1].get('gap', 0):.4f}")
    if table_pts:
        table_data[name] = table_pts
        print(f"{name}: {len(table_pts)} table_norm points")

# Build HTML
html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>n-gram 注入点消融：v vs y vs input</title>
<style>
  body { font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; }
  h1 { font-size: 1.4em; }
  .subtitle { color: #666; margin-bottom: 20px; }
  .chart { width: 100%; max-width: 900px; height: 450px; margin: 20px 0; }
  table { border-collapse: collapse; margin: 15px 0; font-size: 0.9em; }
  td, th { border: 1px solid #ddd; padding: 6px 12px; text-align: right; }
  th { background: #f5f5f5; }
  th:first-child, td:first-child { text-align: left; }
  .note { background: #fff3cd; padding: 12px; border-radius: 6px; margin: 15px 0; }
  .summary { display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }
  .card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 200px; }
  .card h3 { margin: 0 0 10px 0; font-size: 1em; }
  .card .gap { font-size: 1.8em; font-weight: bold; }
</style>
</head>
<body>
<h1>n-gram 注入点消融：v vs y vs input</h1>
<p class="subtitle">Vanilla nanoGPT + bigram+trigram + RMSProp(table)/AdamW(backbone) + 1000 steps + seed42.
唯一变量：n-gram value 注入点。数据来自 ngram-gap-lab 干净复现。</p>

<div class="note">
<b>最简复现 setting</b>：vanilla nanoGPT + bigram+trigram + <b>input 注入</b>（over-encoding 风格，加到 wte）+ RMSProp table optimizer。
不需要 current shell / Muon / current optimizer grouping / RoPE / RMSNorm。
</div>

<div class="summary">"""

# Summary cards
for name, info in RUNS.items():
    if name in data:
        final = data[name][-1]
        color = info["color"]
        html += f'<div class="card"><h3 style="color:{color}">{name}</h3>'
        html += f'<div class="gap" style="color:{color}">{final.get("gap",0):+.3f}</div>'
        html += f'<div>gap @ step {final.get("step","?")}</div></div>'
html += "</div>"

html += """
<div id="gap_chart" class="chart"></div>
<div id="loss_chart" class="chart"></div>
<div id="table_norm_chart" class="chart"></div>

<table id="data_table">
<tr><th>step</th><th>v gap</th><th>y gap</th><th>input gap</th><th>v train</th><th>v val</th><th>y train</th><th>y val</th><th>input train</th><th>input val</th></tr>
</table>

<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
const allData = """ + json.dumps(data, indent=2) + """;
const tableData = """ + json.dumps(table_data, indent=2) + """;

const colors = {
  "v (ResFormer, add to V)": "#2196F3",
  "y (post-attn residual)": "#F44336",
  "input (over-encoding, add to wte)": "#4CAF50"
};
const names = Object.keys(allData);

// Gap chart
const gapTraces = [];
const lossTraces = [];
for (const name of names) {
    const pts = allData[name];
    const x = pts.map(p => p.step);
    const gapY = pts.map(p => p.gap);
    gapTraces.push({x: x, y: gapY, mode: "lines+markers", name: name, line: {color: colors[name], width: 2}});
    const trainY = pts.map(p => p.train_loss);
    const valY = pts.map(p => p.val_loss);
    lossTraces.push({x: x, y: trainY, mode: "lines", name: name + " (train)", line: {color: colors[name], width: 1.5, dash: "dash"}});
    lossTraces.push({x: x, y: valY, mode: "lines", name: name + " (val)", line: {color: colors[name], width: 2}});
}

const epochShapes = [
  {type: "line", x0: 337, x1: 337, y0: 0, y1: 1, yref: "paper", line: {color: "#ccc", dash: "dot"}},
  {type: "line", x0: 686, x1: 686, y0: 0, y1: 1, yref: "paper", line: {color: "#ccc", dash: "dot"}}
];
const epochAnnots = [
  {x: 337, y: 0.95, yref: "paper", text: "epoch2", showarrow: false, font: {size: 10, color: "#999"}},
  {x: 686, y: 0.95, yref: "paper", text: "epoch3", showarrow: false, font: {size: 10, color: "#999"}}
];

Plotly.newPlot("gap_chart", gapTraces, {
    title: "Train/Val Gap (val_loss - train_loss)",
    xaxis: {title: "step"},
    yaxis: {title: "gap", gridcolor: "#eee"},
    margin: {l: 60, r: 30, t: 50, b: 50},
    legend: {x: 0.02, y: 0.98},
    shapes: epochShapes,
    annotations: epochAnnots
});

Plotly.newPlot("loss_chart", lossTraces, {
    title: "Train/Val Loss",
    xaxis: {title: "step"},
    yaxis: {title: "loss", gridcolor: "#eee"},
    margin: {l: 60, r: 30, t: 50, b: 50},
    legend: {x: 0.02, y: 0.98},
    shapes: epochShapes
});

// Table norm chart
const tableTraces = [];
const tableNames = Object.keys(tableData);
for (const name of tableNames) {
    const pts = tableData[name];
    const x = pts.map(p => p.step);
    // use bigram.layer_01.table_0.rms if available, else first key
    const yKey = Object.keys(pts[0] || {}).find(k => k.includes("bigram") && k.includes("layer_01"));
    if (yKey) {
      const y = pts.map(p => p[yKey] || 0);
      tableTraces.push({x: x, y: y, mode: "lines", name: name, line: {color: colors[name], width: 2}});
    }
}
if (tableTraces.length > 0) {
  Plotly.newPlot("table_norm_chart", tableTraces, {
      title: "Bigram Table RMS (layer_01, table_0)",
      xaxis: {title: "step"},
      yaxis: {title: "param RMS", gridcolor: "#eee"},
      margin: {l: 60, r: 30, t: 50, b: 50},
      shapes: epochShapes
  });
}

// Fill table
const table = document.getElementById("data_table");
const allSteps = new Set();
for (const name of names) {
  for (const p of allData[name]) allSteps.add(p.step);
}
const sortedSteps = [...allSteps].sort((a,b) => a-b);
for (const s of sortedSteps) {
    const row = table.insertRow();
    row.insertCell(0).textContent = s;
    for (const name of names) {
        const pt = allData[name].find(p => p.step === s);
        row.insertCell(-1).textContent = pt ? pt.gap.toFixed(4) : "-";
    }
    for (const name of names) {
        const pt = allData[name].find(p => p.step === s);
        if (pt) {
            row.insertCell(-1).textContent = pt.train_loss.toFixed(4);
            row.insertCell(-1).textContent = pt.val_loss.toFixed(4);
        } else {
            row.insertCell(-1).textContent = "-";
            row.insertCell(-1).textContent = "-";
        }
    }
}
</script>
</body>
</html>
"""

os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, "w") as f:
    f.write(html)
print(f"Written to {OUT_HTML}")
