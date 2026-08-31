#!/usr/bin/env python3
"""Fig 1 (interactive): injection-point trajectories with a setting switcher.

Three stacked panels (online train loss / fixed val loss / gap) x four arms
(input, y, v, nogram); a dropdown switches between:
  128x table LR @ 1k steps (default) / 128x @ 2k / 2x historical @ 1k / 2x @ 2k.

Raw eval records (every 10 steps) are shown as points; a thin 3-point moving
average is drawn as the connecting line.  Output is a self-contained HTML
fragment (plotly via CDN) embedded into the report via iframe.

Set NGLAB_RUNS_FIXED to the mirror containing the eight runs.
"""
import json
import os
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
RUNS_FIXED = Path(os.environ.get("NGLAB_RUNS_FIXED", ROOT / "data" / "runs_fixed"))
OUT = ROOT / "docs" / "figs" / "main"

ARMS = [("input", "#2d6f9f"), ("y", "#c4493d"), ("v", "#c58a0b"), ("nogram", "#686d73")]
SETTINGS = [
    ("128x · 1k steps", "nglab1x_{arm}_v5_128x_freq10", 1000),
    ("128x · 2k steps", "nglab1x_{arm}_v5_128x_freq10", 2000),
    ("2x historical · 1k steps", "nglab1x_{arm}_v5", 1000),
    ("2x historical · 2k steps", "nglab1x_{arm}_v5", 2000),
]
PANELS = [("train_loss", "online train loss"), ("val_loss", "fixed validation loss"),
          ("gap", "gap = fixed val - online train")]
EPOCH = 337


def smooth3(vals):
    out = list(vals)
    for i in range(1, len(vals) - 1):
        out[i] = (vals[i - 1] + vals[i] + vals[i + 1]) / 3.0
    return out


def load(run_id):
    rows = [json.loads(l) for l in (RUNS_FIXED / f"{run_id}_fixed" / "train_log.jsonl").open()]
    return rows


def main():
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        subplot_titles=[p[1] for p in PANELS])

    # traces: setting -> list of trace indices
    setting_traces = []
    for label, pattern, cutoff in SETTINGS:
        idxs = []
        for arm, color in ARMS:
            rows = [r for r in load(pattern.format(arm=arm)) if r["step"] <= cutoff]
            steps = [r["step"] for r in rows]
            for row_i, (key, _pname) in enumerate(PANELS):
                vals = [r[key] for r in rows]
                # thin smoothed line
                fig.add_trace(go.Scatter(
                    x=steps, y=smooth3(vals), mode="lines",
                    line=dict(color=color, width=1.1),
                    legendgroup=arm, name=arm, showlegend=(row_i == 0),
                    hovertemplate=f"{arm} %{{x}}: %{{y:.3f}}<extra></extra>",
                ), row=row_i + 1, col=1)
                idxs.append(len(fig.data) - 1)
                # raw points
                fig.add_trace(go.Scatter(
                    x=steps, y=vals, mode="markers",
                    marker=dict(color=color, size=3, opacity=0.35),
                    legendgroup=arm, name=arm, showlegend=False,
                    hovertemplate=f"{arm} raw %{{x}}: %{{y:.3f}}<extra></extra>",
                ), row=row_i + 1, col=1)
                idxs.append(len(fig.data) - 1)
        setting_traces.append(idxs)

    total = len(fig.data)
    buttons = []
    for si, ((label, _p, cutoff), idxs) in enumerate(zip(SETTINGS, setting_traces)):
        vis = [False] * total
        for i in idxs:
            vis[i] = True
        layout_update = {
            "xaxis3.range": [0, cutoff * 1.02],
            "xaxis.range": [0, cutoff * 1.02],
            "xaxis2.range": [0, cutoff * 1.02],
        }
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": vis}, layout_update]))

    # default: setting 0
    default_vis = [False] * total
    for i in setting_traces[0]:
        default_vis[i] = True
    for tr, v in zip(fig.data, default_vis):
        tr.visible = v

    # epoch boundaries on all panels (up to 2k; 1k view crops by range)
    shapes = []
    for row in (1, 2, 3):
        yref = "y" + ("" if row == 1 else str(row))
        xref = "x" + ("" if row == 1 else str(row))
        for e in range(EPOCH, 2001, EPOCH):
            shapes.append(dict(type="line", x0=e, x1=e, y0=0, y1=1,
                               xref=xref, yref=f"{yref} domain",
                               line=dict(color="#b0b7bf", width=0.7, dash="dot")))

    fig.update_layout(
        height=760, margin=dict(l=60, r=20, t=60, b=40),
        template="plotly_white",
        updatemenus=[dict(buttons=buttons, direction="down", showactive=True,
                          x=0.0, xanchor="left", y=1.16, yanchor="top",
                          bgcolor="#f0f4f8", bordercolor="#dde5ec")],
        shapes=shapes,
        legend=dict(orientation="h", x=0.3, xanchor="left", y=1.13, yanchor="top"),
        annotations=[dict(text="setting:", x=0, xref="paper", y=1.19, yref="paper",
                          showarrow=False, font=dict(size=12), xanchor="left")] + [
            dict(text="<b>online train loss</b>", xref="paper", yref="paper",
                 x=0.0, y=1.0, showarrow=False, xanchor="left", font=dict(size=12))],
    )
    fig.update_xaxes(title_text="optimizer step", row=3, col=1, range=[0, 1020])
    fig.update_xaxes(range=[0, 1020])
    fig.update_annotations(font=dict(size=12))
    fig.layout.annotations[-1].text = ""

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "fig_v5_injection_interactive.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True,
                   config=dict(displaylogo=False, responsive=True))
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
