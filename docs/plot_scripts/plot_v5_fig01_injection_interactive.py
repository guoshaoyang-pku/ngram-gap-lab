#!/usr/bin/env python3
"""Fig 1 (interactive v3): arms overlaid on one panel, train/val/gap per arm.

Layout: one panel.  For the selected LR x budget combo (dropdown), all four
injection arms are overlaid; each arm draws three traces in the arm's color:
  * train  = solid line (arm color), raw records as faint points
  * val    = solid line, same hue semi-transparent (alpha .5)
  * gap    = dashed line (arm color)
Arms are shown/hidden by clicking the legend: input and nogram are on by
default, y and v start as legend-only.

Axis lines are drawn (showline), matching the static v5 figures.  Raw eval
records (every 10 steps) are points; thin lines are a 3-point moving average.
Self-contained HTML fragment (plotly via CDN) embedded via iframe.

Set NGLAB_RUNS_FIXED to the mirror containing the eight runs.
"""
import json
import os
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
MIRROR = Path(os.environ.get("NGLAB_RUNS_FIXED", ROOT / "data" / "runs_fixed"))
OUT = ROOT / "docs" / "figs" / "main" / "fig_v5_injection_interactive.html"

ARM_COLORS = {
    "input": "#2d6f9f",
    "y": "#c4493d",
    "v": "#c58a0b",
    "nogram": "#686d73",
}
ARMS = ["input", "y", "v", "nogram"]
DEFAULT_ON = {"input", "nogram"}

# (combo label, run dir pattern, max step)
COMBOS = [
    ("128× · 1k steps", "nglab1x_{arm}_v5_128x_freq10_fixed", 1000),
    ("128× · 2k steps", "nglab1x_{arm}_v5_128x_freq10_fixed", 2000),
    ("2× · 1k steps", "nglab1x_{arm}_v5_fixed", 1000),
    ("2× · 2k steps", "nglab1x_{arm}_v5_fixed", 2000),
]


def rgba(hexc, a):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def load(run_dir, cap):
    s, tr, val, gap = [], [], [], []
    with open(MIRROR / run_dir / "train_log.jsonl") as fh:
        for ln in fh:
            e = json.loads(ln)
            if e["step"] <= cap:
                s.append(e["step"])
                tr.append(e["train_loss"])
                val.append(e["val_loss"])
                gap.append(e["gap"])
    return s, tr, val, gap


def movavg(y, w=3):
    out = []
    for i in range(len(y)):
        seg = y[max(0, i - w // 2):i + w // 2 + 1]
        out.append(sum(seg) / len(seg))
    return out


fig = go.Figure()
combo_vis = []  # per combo: flat visibility list over all traces

for ci, (label, pattern, cap) in enumerate(COMBOS):
    vis = []
    for arm in ARMS:
        s, tr, val, gap = load(pattern.format(arm=arm), cap)
        c = ARM_COLORS[arm]
        on = True if arm in DEFAULT_ON else "legendonly"
        first = (ci == 0)
        specs = [
            (s, tr, "markers", dict(size=3.5, color=c, opacity=0.22), None, False),
            (s, movavg(tr), "lines", None, dict(color=c, width=2.0), True),
            (s, val, "markers", dict(size=3.5, color=c, opacity=0.18), None, False),
            (s, movavg(val), "lines", None, dict(color=rgba(c, 0.5), width=1.8), True),
            (s, gap, "markers", dict(size=3.5, color=c, opacity=0.22), None, False),
            (s, movavg(gap), "lines", None, dict(color=c, width=1.6, dash="dash"), True),
        ]
        kinds = ["train", "train", "val", "val", "gap", "gap"]
        for (x, y, mode, marker, line, is_line), kind in zip(specs, kinds):
            fig.add_trace(go.Scatter(
                x=x, y=y, mode=mode, visible=on if ci == 0 else False,
                showlegend=bool(is_line and first),
                legendgroup=f"{arm}-{kind}", name=f"{arm} · {kind}",
                marker=marker or {}, line=line or {},
                hoverinfo="skip" if not is_line else None,
                hovertemplate=("step %{x}<br>%{y:.3f}<extra>" +
                               f"{arm} {kind}</extra>") if is_line else None))
            vis.append(on)
    combo_vis.append(vis)

buttons = [
    dict(label=label, method="update", args=[{"visible": v}])
    for (label, _, _), v in zip(COMBOS, combo_vis)
]

axis = dict(showline=True, linecolor="#555", linewidth=1.0, ticks="outside",
            gridcolor="#e6e6e6", zeroline=False)
fig.update_layout(
    updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor="left",
                      y=1.20, yanchor="top", bgcolor="#fff",
                      bordercolor="#ccc")],
    annotations=[dict(text="学习率 × 步数", x=0.0, xref="paper", xanchor="left",
                      y=1.32, yref="paper", showarrow=False,
                      font=dict(size=12, color="#666"))],
    xaxis=dict(title="step", **axis),
    yaxis=dict(title="loss / gap", **axis),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="-apple-system, 'Helvetica Neue', Arial, sans-serif",
              size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.20,
                title=dict(text="臂（点击显示/隐藏）", font=dict(size=11))),
    margin=dict(l=60, r=20, t=105, b=45), height=500,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="fig1x")
OUT.write_text(
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<style>body{margin:0;background:#fff}</style></head>"
    f"<body>{div}</body></html>")
print(f"[saved] {OUT}")
