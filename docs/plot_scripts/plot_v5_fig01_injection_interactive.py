#!/usr/bin/env python3
"""Fig 1 (interactive v4): arms overlaid, custom controls, no CDN dependency.

One panel.  Controls above the plot:
  * a <select> for LR x budget (128x/2x x 1k/2k steps);
  * four checkboxes for the injection arms (input + nogram on by default) --
    arms are overlaid, any subset can be shown.
Each arm draws train = solid arm color, val = same hue alpha .5,
gap = dashed arm color; raw records (every 10 steps) are faint points, lines
are 3-point moving averages.

The plotly legend is reduced to a single vertical column of 3 style entries
(train / val / gap) and is non-interactive; arm visibility is driven only by
the checkboxes (a small inline script calls Plotly.restyle).  plotly.js is
referenced as the sibling file `plotly.min.js` (no CDN, works offline and on
GitHub Pages).

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
KINDS = ["train", "val", "gap"]

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

# --- 3 dummy legend traces (always visible, style guide only) ---
for name, line in [
    ("train", dict(color="#555", width=2.0)),
    ("val", dict(color="rgba(85,85,85,0.5)", width=1.8)),
    ("gap", dict(color="#555", width=1.6, dash="dash")),
]:
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name=name,
                             line=line, showlegend=True, hoverinfo="skip"))

# --- data traces: combo -> arm -> kind -> (points, line) ---
for label, pattern, cap in COMBOS:
    for arm in ARMS:
        s, tr, val, gap = load(pattern.format(arm=arm), cap)
        c = ARM_COLORS[arm]
        series = {"train": (tr, dict(color=c, width=2.0)),
                  "val": (val, dict(color=rgba(c, 0.5), width=1.8)),
                  "gap": (gap, dict(color=c, width=1.6, dash="dash"))}
        for kind in KINDS:
            y, line = series[kind]
            fig.add_trace(go.Scatter(
                x=s, y=y, mode="markers", showlegend=False, hoverinfo="skip",
                marker=dict(size=3.5, color=c, opacity=0.22)))
            fig.add_trace(go.Scatter(
                x=s, y=movavg(y), mode="lines", showlegend=False,
                line=line, hovertemplate=("step %{x}<br>%{y:.3f}<extra>" +
                                          f"{arm} {kind}</extra>")))

N_DATA = len(COMBOS) * len(ARMS) * len(KINDS) * 2
init = [True] * 3 + [False] * N_DATA
for ai, arm in enumerate(ARMS):
    if arm in DEFAULT_ON:
        for k in range(len(KINDS)):
            for p in range(2):
                init[3 + ((0 * len(ARMS) + ai) * len(KINDS) + k) * 2 + p] = True
for i, t in enumerate(fig.data):
    t.visible = init[i]

axis = dict(showline=True, linecolor="#555", linewidth=1.0, ticks="outside",
            gridcolor="#e6e6e6", zeroline=False)
fig.update_layout(
    xaxis=dict(title="step", **axis),
    yaxis=dict(title="loss / gap", **axis),
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(family="-apple-system, 'Helvetica Neue', Arial, sans-serif",
              size=12),
    legend=dict(orientation="v", x=1.01, y=1.0, xanchor="left",
                itemclick=False, itemdoubleclick=False),
    margin=dict(l=60, r=80, t=20, b=45), height=470,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
div = fig.to_html(full_html=False, include_plotlyjs="plotly.min.js",
                  div_id="fig1x")

options = "".join(
    f'<option value="{i}"{" selected" if i == 0 else ""}>{lb}</option>'
    for i, (lb, _, _) in enumerate(COMBOS))
boxes = "".join(
    f'<label style="color:{ARM_COLORS[a]};font-weight:600;margin-right:10px">'
    f'<input type="checkbox" class="fig1-arm" value="{ai}"'
    f'{" checked" if a in DEFAULT_ON else ""}> {a}</label>'
    for ai, a in enumerate(ARMS))

n_arm, n_kind = len(ARMS), len(KINDS)
script = f"""
<script>
(function() {{
  var gd = document.getElementById('fig1x');
  if (typeof Plotly === 'undefined') {{
    gd.innerHTML = '<p style="color:#c00">plotly.min.js 未加载（应与 html 同目录）。</p>';
    return;
  }}
  var NARM = {n_arm}, NKIND = {n_kind};
  function apply() {{
    var c = +document.getElementById('fig1-combo').value;
    var arms = [];
    document.querySelectorAll('.fig1-arm:checked').forEach(function(x) {{
      arms.push(+x.value);
    }});
    var vis = [true, true, true];
    for (var cc = 0; cc < {len(COMBOS)}; cc++)
      for (var aa = 0; aa < NARM; aa++)
        for (var kk = 0; kk < NKIND; kk++) {{
          var on = (cc === c && arms.indexOf(aa) >= 0);
          vis.push(on, on);
        }}
    Plotly.restyle(gd, 'visible', vis);
  }}
  document.getElementById('fig1-combo').addEventListener('change', apply);
  document.querySelectorAll('.fig1-arm').forEach(function(x) {{
    x.addEventListener('change', apply);
  }});
}})();
</script>
"""

OUT.write_text(
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<style>body{margin:0;background:#fff;font-family:-apple-system,"
    "'Helvetica Neue',Arial,sans-serif;font-size:13px}"
    "#fig1-controls{padding:6px 10px 0}"
    "#fig1-controls select{font-size:13px;margin-left:4px}</style></head>"
    "<body><div id='fig1-controls'>"
    "学习率 × 步数：<select id='fig1-combo'>" + options + "</select>"
    "&nbsp;&nbsp;臂（可叠加）：" + boxes +
    "</div>" + div + script + "</body></html>")
print(f"[saved] {OUT}")
