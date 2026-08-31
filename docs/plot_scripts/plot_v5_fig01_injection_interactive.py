#!/usr/bin/env python3
"""Fig 1 (interactive v2): one arm at a time, train+val+gap on a single axis.

View = one panel; traces = online train loss, fixed val loss, gap (all on the
same y axis) for a single injection arm.  One dropdown switches among the 16
arm x setting combos (input default / y / v / nogram) x
(128x @ 1k default / 128x @ 2k / 2x @ 1k / 2x @ 2k).

Raw eval records (every 10 steps) are points; a thin 3-point moving average is
the connecting line.  Self-contained HTML fragment (plotly via CDN) embedded
into the report via iframe.

Set NGLAB_RUNS_FIXED to the mirror containing the eight runs.
"""
import json
import os
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("NGLAB_RUNS_FIXED", ROOT / "data" / "runs_fixed"))
OUT = ROOT / "docs" / "figs" / "main" / "fig_v5_injection_interactive.html"

# (combo label, run dir name in the mirror, max step)
COMBOS = [
    ("input · 128× · 1k", "nglab1x_input_v5_128x_freq10_fixed", 1000),
    ("input · 128× · 2k", "nglab1x_input_v5_128x_freq10_fixed", 2000),
    ("input · 2× · 1k",   "nglab1x_input_v5_fixed", 1000),
    ("input · 2× · 2k",   "nglab1x_input_v5_fixed", 2000),
    ("y · 128× · 1k",     "nglab1x_y_v5_128x_freq10_fixed", 1000),
    ("y · 128× · 2k",     "nglab1x_y_v5_128x_freq10_fixed", 2000),
    ("y · 2× · 1k",       "nglab1x_y_v5_fixed", 1000),
    ("y · 2× · 2k",       "nglab1x_y_v5_fixed", 2000),
    ("v · 128× · 1k",     "nglab1x_v_v5_128x_freq10_fixed", 1000),
    ("v · 128× · 2k",     "nglab1x_v_v5_128x_freq10_fixed", 2000),
    ("v · 2× · 1k",       "nglab1x_v_v5_fixed", 1000),
    ("v · 2× · 2k",       "nglab1x_v_v5_fixed", 2000),
    ("nogram · 128× · 1k", "nglab1x_nogram_v5_128x_freq10_fixed", 1000),
    ("nogram · 128× · 2k", "nglab1x_nogram_v5_128x_freq10_fixed", 2000),
    ("nogram · 2× · 1k",   "nglab1x_nogram_v5_fixed", 1000),
    ("nogram · 2× · 2k",   "nglab1x_nogram_v5_fixed", 2000),
]

TRAIN_C = "#14736f"   # bigram teal
VAL_C = "#b3402e"     # val red
GAP_C = "#444444"
MEAN = "#000000"


def load(run_id, cap):
    tr_s, tr, val_s, val = [], [], [], []
    with open(MIRROR / run_id / "train_log.jsonl") as fh:
        for ln in fh:
            e = json.loads(ln)
            if e.get("kind") == "train" and e["step"] <= cap:
                tr_s.append(e["step"]); tr.append(e["loss"])
            elif e.get("kind") == "val" and e["step"] <= cap:
                val_s.append(e["step"]); val.append(e["loss"])
    # gap at matched steps
    tm = dict(zip(tr_s, tr))
    g_s = [s for s in val_s if s in tm]
    gap = [v - tm[s] for s, v in zip(val_s, val) if s in tm]
    return tr_s, tr, val_s, val, g_s, gap


def movavg(y, w=3):
    out = []
    for i in range(len(y)):
        seg = y[max(0, i - w // 2):i + w // 2 + 1]
        out.append(sum(seg) / len(seg))
    return out


fig = go.Figure()
for label, rid, cap in COMBOS:
    tr_s, tr, val_s, val, g_s, gap = load(rid, cap)
    # raw points (faint)
    for x, y, name, c in [(tr_s, tr, "train (online)", TRAIN_C),
                          (val_s, val, "val (fixed)", VAL_C),
                          (g_s, gap, "gap", GAP_C)]:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", visible=False, showlegend=False,
            marker=dict(size=3.5, color=c, opacity=0.35),
            legendgroup=name, name=name, hoverinfo="skip"))
    # smoothed lines
    for x, y, name, c in [(tr_s, movavg(tr), "train (online)", TRAIN_C),
                          (val_s, movavg(val), "val (fixed)", VAL_C),
                          (g_s, movavg(gap), "gap", GAP_C)]:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines", visible=False, legendgroup=name, name=name,
            line=dict(color=c, width=2.0),
            hovertemplate="step %{x}<br>%{y:.3f}<extra>" + name + "</extra>"))

PER = 6  # 3 raw + 3 lines per combo
vis = []
for i in range(len(COMBOS)):
    v = [False] * (len(COMBOS) * PER)
    for j in range(i * PER, (i + 1) * PER):
        v[j] = True
    vis.append(v)
for j in range(PER):
    fig.data[j].visible = True  # default: input · 128x · 1k

buttons = []
for (label, _, _), v in zip(COMBOS, vis):
    buttons.append(dict(label=label, method="update", args=[{"visible": v}]))

fig.update_layout(
    updatemenus=[dict(active=0, buttons=buttons, x=0.02, xanchor="left",
                      y=1.16, yanchor="top", bgcolor="#fff", bordercolor="#ccc")],
    annotations=[dict(text="臂 · table LR · 步数预算", x=0.02, xref="paper",
                      xanchor="left", y=1.30, yref="paper", showarrow=False,
                      font=dict(size=12, color="#666"))],
    xaxis=dict(title="step", gridcolor="#e6e6e6", zeroline=False),
    yaxis=dict(title="loss / gap", gridcolor="#e6e6e6", zeroline=False),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="-apple-system, 'Helvetica Neue', Arial, sans-serif", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.35),
    margin=dict(l=55, r=20, t=90, b=45), height=460,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="fig1x")
OUT.write_text(
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<style>body{margin:0;background:#fff}</style></head>"
    f"<body>{div}</body></html>")
print(f"[saved] {OUT}")
