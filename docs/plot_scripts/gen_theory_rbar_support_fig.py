#!/usr/bin/env python3
"""X3 corpus-side support-width figure: rbar(f) and Good-Turing missing mass.

Reads only data/runs_fixed/corpus_rbar_freq_v1_fixed/rbar_bins_{bigram,trigram}.json
(run_id corpus_rbar_freq_v1, docs/experiment-log.md §29).  Zero GPU, read-only.
Outputs docs/figs/theory/fig_theory_rbar_support.svg (static, self-contained).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "runs_fixed" / "corpus_rbar_freq_v1_fixed"
OUT = ROOT / "docs" / "figs" / "theory"
OUT.mkdir(parents=True, exist_ok=True)


def load(branch: str):
    with (SRC / f"rbar_bins_{branch}.json").open() as f:
        d = json.load(f)
    return d["bins"]


def collect(branch: str):
    bins = load(branch)
    f, r, mgt, nctx = [], [], [], []
    for b in bins[1:]:  # skip f=[1,2] degenerate bin (r=1, mgt undefined)
        flo, fhi = b["f_lo"], b["f_hi"]
        fmid = np.sqrt(flo * fhi)
        f.append(fmid)
        r.append(b["mean_r"])
        mgt.append(b["mean_mgt_missing"])
        nctx.append(b["contexts"])
    return np.array(f), np.array(r), np.array(mgt), np.array(nctx)


def loglog_slope(x, y, lo=0.5, hi=1.0):
    xs, ys = np.log(x), np.log(y)
    k0, k1 = int(len(xs) * lo), int(len(xs) * hi)
    if k1 - k0 < 3:
        return float("nan")
    return float(np.polyfit(xs[k0:k1], ys[k0:k1], 1)[0])


def main():
    fb, rb, mb, nb = collect("bigram")
    ft, rt, mt, nt = collect("trigram")

    sb_r = loglog_slope(fb, rb)
    st_r = loglog_slope(ft, rt)
    sb_m = loglog_slope(fb, mb)
    st_m = loglog_slope(ft, mt)

    # SVG dimensions
    W, H = 1200, 820
    pad_l, pad_r, pad_t, pad_b = 90, 30, 30, 60
    pw, ph = W - pad_l - pad_r, 300
    gap = 70
    top_y = pad_t
    bot_y = pad_t + ph + gap

    def xlog(f):
        return pad_l + pw * (np.log10(f) - np.log10(1.0)) / (np.log10(2e5) - np.log10(1.0))

    def ylog_r(r):
        return top_y + ph * (1 - (np.log10(r) - np.log10(1.0)) / (np.log10(5e3) - np.log10(1.0)))

    def ylog_m(m):
        return bot_y + ph * (1 - (np.log10(m) - np.log10(1e-4)) / (np.log10(1.0) - np.log10(1e-4)))

    def grid_path(xs, yvals, yfn):
        pts = " ".join(f"{xlog(x):.1f},{yfn(y):.1f}" for x, y in zip(xs, yvals))
        return f'<polyline points="{pts}" fill="none" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'

    # axis ticks
    xticks = []
    for e in range(1, 6):
        for m in (1, 2, 5):
            v = m * 10**e
            if 1 <= v <= 2e5:
                xticks.append(v)
    yticks_r = [1, 3, 10, 30, 100, 300, 1000, 3000]
    yticks_m = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif">')
    svg.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # gridlines
    for v in xticks:
        x = xlog(v)
        svg.append(f'<line x1="{x:.1f}" y1="{top_y}" x2="{x:.1f}" y2="{top_y+ph:.1f}" stroke="#eeeeee" stroke-width="1"/>')
        svg.append(f'<line x1="{x:.1f}" y1="{bot_y}" x2="{x:.1f}" y2="{bot_y+ph:.1f}" stroke="#eeeeee" stroke-width="1"/>')
    for v in yticks_r:
        y = ylog_r(v)
        svg.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r:.1f}" y2="{y:.1f}" stroke="#eeeeee" stroke-width="1"/>')
    for v in yticks_m:
        y = ylog_m(v)
        svg.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-pad_r:.1f}" y2="{y:.1f}" stroke="#eeeeee" stroke-width="1"/>')

    # series: rbar
    svg.append(grid_path(fb, rb, ylog_r).replace('stroke-width="2.5"', 'stroke-width="3"') if False else
               f'<polyline points="{" ".join(f"{xlog(x):.1f},{ylog_r(y):.1f}" for x,y in zip(fb,rb))}" fill="none" stroke="#1f77b4" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
    svg.append(f'<polyline points="{" ".join(f"{xlog(x):.1f},{ylog_r(y):.1f}" for x,y in zip(ft,rt))}" fill="none" stroke="#ff7f0e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')

    # rbar = f guide (slope 1)
    fg = np.logspace(0, np.log10(2e5), 50)
    svg.append(f'<polyline points="{" ".join(f"{xlog(x):.1f},{ylog_r(x):.1f}" for x in fg)}" fill="none" stroke="#999999" stroke-width="1.5" stroke-dasharray="6,4"/>')

    # mgt series (right axis, log scale 1e-4..1)
    svg.append(f'<polyline points="{" ".join(f"{xlog(x):.1f},{ylog_m(y):.1f}" for x,y in zip(fb,mb))}" fill="none" stroke="#2ca02c" stroke-width="2.5" stroke-dasharray="1,0" stroke-linecap="round" stroke-linejoin="round"/>')
    svg.append(f'<polyline points="{" ".join(f"{xlog(x):.1f},{ylog_m(y):.1f}" for x,y in zip(ft,mt))}" fill="none" stroke="#d62728" stroke-width="2.5" stroke-dasharray="1,0" stroke-linecap="round" stroke-linejoin="round"/>')

    # mgt ~ f^-0.3 guide (dashed)
    mref = [1e-3 * (f / 1e4) ** (-0.3) for f in fg]
    svg.append(f'<polyline points="{" ".join(f"{xlog(x):.1f},{ylog_m(y):.1f}" for x,y in zip(fg,mref))}" fill="none" stroke="#555555" stroke-width="1.5" stroke-dasharray="2,4"/>')

    # labels
    svg.append(f'<text x="{W/2:.0f}" y="{H-12}" font-size="17" text-anchor="middle" fill="#222">exact train hit count f (log)</text>')
    svg.append(f'<text x="18" y="{top_y+ph/2:.0f}" font-size="16" text-anchor="middle" fill="#222" transform="rotate(-90 18 {top_y+ph/2:.0f})">distinct successors r̄(f) (log)</text>')
    svg.append(f'<text x="18" y="{bot_y+ph/2:.0f}" font-size="16" text-anchor="middle" fill="#222" transform="rotate(-90 18 {bot_y+ph/2:.0f})">Good–Turing missing mass s₁/f (log)</text>')
    svg.append(f'<text x="{pad_l}" y="{top_y-10}" font-size="16" fill="#222">Support width: r̄(f) grows sublinearly, not r̄=f</text>')
    svg.append(f'<text x="{pad_l}" y="{bot_y-10}" font-size="16" fill="#222">Unresolved mass: s₁/f remains non-negligible</text>')

    # x ticks
    for v in xticks:
        svg.append(f'<text x="{xlog(v):.1f}" y="{H-pad_b+18}" font-size="11" text-anchor="middle" fill="#555">{v}</text>')

    # y ticks rbar
    for v in yticks_r:
        svg.append(f'<text x="{pad_l-8}" y="{ylog_r(v)+4:.1f}" font-size="11" text-anchor="end" fill="#1f77b4">{v}</text>')
    for v in yticks_m:
        svg.append(f'<text x="{pad_l-8}" y="{ylog_m(v)+4:.1f}" font-size="11" text-anchor="end" fill="#2ca02c">{v:.0e}</text>')

    # legend
    lx, ly = W - 310, top_y + 20
    legend = [
        ("#1f77b4", "bigram r̄(f)", 3, 0),
        ("#ff7f0e", "trigram r̄(f)", 3, 0),
        ("#999999", "r̄ = f (slope 1)", 1.5, 1),
        ("#2ca02c", "bigram s₁/f", 2.5, 2),
        ("#d62728", "trigram s₁/f", 2.5, 2),
        ("#555555", "s₁/f ∝ f^−0.3", 1.5, 1),
    ]
    for i, (color, label, sw, style) in enumerate(legend):
        y = ly + i * 24
        if style == 0:
            svg.append(f'<line x1="{lx}" y1="{y}" x2="{lx+34}" y2="{y}" stroke="{color}" stroke-width="{sw}"/>')
        elif style == 1:
            svg.append(f'<line x1="{lx}" y1="{y}" x2="{lx+34}" y2="{y}" stroke="{color}" stroke-width="{sw}" stroke-dasharray="6,4"/>')
        else:
            svg.append(f'<line x1="{lx}" y1="{y}" x2="{lx+34}" y2="{y}" stroke="{color}" stroke-width="{sw}" stroke-dasharray="1,0"/>')
        svg.append(f'<text x="{lx+42}" y="{y+4}" font-size="13" fill="#222">{label}</text>')

    # annotations
    svg.append(f'<text x="{xlog(3e3):.0f}" y="{ylog_r(2e2)-8:.0f}" font-size="14" fill="#1f77b4">bigram slope ≈ +0.54</text>')
    svg.append(f'<text x="{xlog(2e3):.0f}" y="{ylog_r(9e1)-20:.0f}" font-size="14" fill="#ff7f0e">trigram slope ≈ +0.32</text>')
    svg.append(f'<text x="{xlog(8e3):.0f}" y="{ylog_m(1e-2)-10:.0f}" font-size="14" fill="#2ca02c">mgt declines slowly overall</text>')
    svg.append(f'<text x="{xlog(1.5e4):.0f}" y="{bot_y+ph-16}" font-size="13" fill="#333">even f≈2×10⁵ leaves unresolved successors</text>')

    svg.append("</svg>")

    out = OUT / "fig_theory_rbar_support.svg"
    out.write_text("\n".join(svg))
    print(f"wrote {out}")
    print(f"bigram rbar slope={sb_r:.3f}  mgt slope={sb_m:.3f}")
    print(f"trigram rbar slope={st_r:.3f}  mgt slope={st_m:.3f}")


if __name__ == "__main__":
    main()
