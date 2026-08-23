#!/usr/bin/env python3
"""Strict-Zipf reweighting analysis (2026-08-07).

Question: if the toy's ngram frequency distribution were *strict Zipf*
(N_r ~ 1/r^2, classic rank exponent 1) instead of the current anti-Zipf
design (N_r ~ 1/r), does the gap-vs-frequency double-log plot become linear?

Answer structure (theory: docs/theory_notes/toy-gap-frequency-distributions.md):
  * per-bucket gap g(r) is a property of training dynamics + val protocol and
    is INDEPENDENT of N_r (separability) -> strict Zipf does not change the
    per-bucket log-log fit;
  * Zipf changes only the WEIGHTS -> aggregate / cumulative curves become
    clean power laws.

Figures -> docs/figs/toy/fig_zipf_gap_analysis.{png,svg}
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs")
FIGS_DIR = os.path.join(REPO_ROOT, "docs", "figs", "toy")
os.makedirs(FIGS_DIR, exist_ok=True)

TOY_META = "/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap/toy/runs/t5b_beta_000_999_low/run_meta.json"
TOY_META_OLD = "/Users/guoshaoyang/Desktop/workdir/OPHIS/OPHIS_gap/toy/run_meta_table_t5.json"

BG = "#f7f5ef"; BORDER = "#c8c1b6"; TEXT = "#686d73"; ANCHOR = "#353d79"
TRAIN = "#3c8d5a"; VAL = "#d97932"; RED = "#C44E52"


def style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(BORDER)
    ax.tick_params(colors=TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(alpha=0.25, color=BORDER)


def loglog_fit(xs, ys):
    x = np.log(np.asarray(xs, float)); y = np.log(np.asarray(ys, float))
    ok = np.isfinite(x) & np.isfinite(y) & (y > np.log(1e-9))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return None
    slope, inter = np.polyfit(x, y, 1)
    pred = slope * x + inter
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2, len(x)


def num(s):
    s = s.strip()
    return int(s[:-1]) * 1000 if s.endswith("k") else int(s)


def bin_mid(label):
    if "-" in label:
        a, b = label.split("-"); return math.sqrt(num(a) * num(b))
    if "+" in label:
        return num(label.replace("+", "")) * 2
    return num(label)


def strict_zipf_buckets(n_keys=32768, s=2.0, r_max=256):
    """N_r = round(C / r^s), C set so sum N_r = n_keys (classic Zipf density
    s=2 => rank exponent 1)."""
    rs = np.arange(1, r_max + 1)
    w = rs ** (-s)
    C = n_keys / w.sum()
    ns = np.maximum(1, np.round(C * w)).astype(int)
    ns = (ns / ns.sum() * n_keys).round().astype(int)
    ns[-1] += n_keys - ns.sum()
    return rs, ns


def main():
    # ---- measured per-bucket g(r) from the toy (final exact eval) ----
    m = json.load(open(TOY_META))
    eg = m["exact_r_gap"]
    rs_toy = np.array(sorted(int(k) for k in eg), float)
    gs_toy = np.array([eg[str(int(r))] for r in rs_toy], float)

    # ---- old t5_on_low (clean coincidental/shared kink) ----
    t5 = json.load(open(TOY_META_OLD))
    old = t5["runs"].get("t5_on_low_s42", {})
    eg_old = old.get("exact_r_gap", {})
    rs_old = np.array(sorted(int(k) for k in eg_old), float)
    gs_old = np.array([eg_old[str(int(r))] for r in rs_old], float)

    # ---- real model per-bin gap (1x run) ----
    recs = [json.loads(l) for l in open(os.path.join(RUNS_DIR, "nglab1x_e6", "freq_bin_loss.jsonl")) if l.strip()]
    last = recs[-1]
    real_x, real_y = [], []
    for gram in ("bigram", "trigram"):
        tr, va = last["train"][gram], last["val"][gram]
        for k in tr:
            if k == "novel" or tr[k]["token_count"] == 0 or va[k]["token_count"] == 0:
                continue
            g = va[k]["mean_loss"] - tr[k]["mean_loss"]
            if g > 1e-9:
                real_x.append(bin_mid(k)); real_y.append(g)

    # ---- current vs strict-Zipf N_r ----
    cur_rs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], float)
    cur_ns = np.array([16384, 8192, 4096, 2048, 1024, 512, 256, 128, 128], float)
    zr, zn = strict_zipf_buckets()
    keep = zn > 0
    zr, zn = zr[keep], zn[keep]

    # ---- Zipf-weighted aggregates using the MEASURED toy g(r) ----
    # map measured g(r) onto the strict-Zipf bucket grid (same r values)
    gmap = {r: g for r, g in zip(rs_toy, gs_toy)}
    grid_rs = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], float)
    ggrid = np.array([gmap.get(r, np.nan) for r in grid_rs])
    # token-weighted contribution per bucket under strict Zipf
    zn_grid = np.array([float(zn[zr == r][0]) if (zr == r).any() else 0.0 for r in grid_rs])
    contrib = zn_grid * ggrid * grid_rs          # r * N_r * g(r)
    gcum = np.cumsum(contrib)                     # threshold cumulative (token-weighted)
    valid = np.isfinite(ggrid)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.plot(np.log2(cur_rs), np.log2(cur_ns), "o-", color=TRAIN, lw=1.6,
            label=f"current toy (slope {np.polyfit(np.log(cur_rs), np.log(cur_ns), 1)[0]:+.2f})")
    ax.plot(np.log2(zr[zr <= 256]), np.log2(zn[zr <= 256]), "s-", color=RED, lw=1.6,
            label="strict Zipf N_r ~ 1/r²")
    ax.set_xlabel("log2(frequency r)"); ax.set_ylabel("log2(# keys)")
    ax.set_title("A. ngram frequency distribution: current (anti-Zipf) vs strict Zipf")
    ax.legend(fontsize=8); style(ax)

    ax = axes[0, 1]
    ax.plot(np.log2(rs_toy), np.log2(np.maximum(gs_toy, 1e-12)), "o-", color=RED, lw=1.8,
            label="toy measured g(r) (this week)")
    ax.plot(np.log2(rs_old), np.log2(np.maximum(gs_old, 1e-12)), "s--", color=ANCHOR, lw=1.6,
            label="toy t5_on_low g(r)")
    for lab, xx, yy in [("bigram", real_x, real_y)]:
        ax.plot(np.log2([x for x in xx if x >= 1]), np.log2([y for x, y in zip(xx, yy) if x >= 1]),
                ".", color=VAL, ms=4, alpha=0.6, label=f"real model {lab} (per-bin)")
    ax.set_xlabel("log2(frequency)"); ax.set_ylabel("log2(gap)")
    ax.set_title("B. per-bucket gap g(r): shape is distribution-INDEPENDENT")
    ax.legend(fontsize=7); style(ax)

    ax = axes[1, 0]
    f = loglog_fit(grid_rs[valid], ggrid[valid])
    ax.plot(np.log2(grid_rs[valid]), np.log2(ggrid[valid]), "o-", color=RED, lw=1.8,
            label=f"measured g(r) — loglog R²={f[1]:.2f}" if f else "measured g(r)")
    ax.set_xlabel("log2(frequency)"); ax.set_ylabel("log2(gap)")
    ax.set_title("C. same g(r), reweighted by strict Zipf N_r (per-bucket view unchanged)")
    ax.legend(fontsize=8); style(ax)

    ax = axes[1, 1]
    ok = valid & (contrib > 0) & np.isfinite(contrib)
    ax.plot(np.log2(grid_rs[ok]), np.log2(contrib[ok]), "o-", color=TRAIN, lw=1.8,
            label="r·N_r·g(r) per bucket (token-weighted)")
    f2 = loglog_fit(grid_rs[ok], contrib[ok])
    if f2:
        ax.plot([], [], " ", label=f"contribution slope {f2[0]:+.2f}, R²={f2[1]:.2f}")
    ax.set_xlabel("log2(frequency)"); ax.set_ylabel("log2(gap tokens)")
    ax.set_title("D. under strict Zipf: token-weighted gap contribution + cumulative")
    ax.legend(fontsize=8); style(ax)
    axb = ax.twinx()
    axb.plot(np.log2(grid_rs[ok]), np.log2(gcum[ok]), "s--", color=ANCHOR, lw=1.5,
             label="cumulative G_T (log-log)")
    axb.set_ylabel("log2(cumulative gap)", color=TEXT)
    axb.tick_params(colors=TEXT)
    for s in axb.spines.values(): s.set_color(BORDER)
    axb.legend(fontsize=8, loc="lower left")

    fig.suptitle("Strict-Zipf reweighting of the toy: distribution changes weights, "
                 "not the per-bucket gap curve", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for name in ("fig_zipf_gap_analysis",):
        fig.savefig(os.path.join(FIGS_DIR, name + ".png"), dpi=150)
        fig.savefig(os.path.join(FIGS_DIR, name + ".svg"))
    plt.close(fig)

    print("[zipf] wrote", os.path.join(FIGS_DIR, "fig_zipf_gap_analysis.png"))
    print(f"  toy measured g(r): r={rs_toy.tolist()}")
    print(f"    gaps = {[round(float(g), 2) for g in gs_toy]}")
    fg = loglog_fit(rs_toy, gs_toy)
    print(f"  toy per-bucket log-log fit: slope={fg[0]:+.3f} R2={fg[1]:.3f} n={fg[2]}")
    fg_old = loglog_fit(rs_old, gs_old)
    print(f"  t5_on_low per-bucket log-log fit: slope={fg_old[0]:+.3f} R2={fg_old[1]:.3f} n={fg_old[2]}")
    for gram, x, y in [("bigram", real_x, real_y), ("trigram", real_x, real_y)]:
        pass
    # separate bigram/trigram fits
    for gram in ("bigram", "trigram"):
        tr, va = last["train"][gram], last["val"][gram]
        xx, yy = [], []
        for k in tr:
            if k == "novel" or tr[k]["token_count"] == 0 or va[k]["token_count"] == 0:
                continue
            g = va[k]["mean_loss"] - tr[k]["mean_loss"]
            if g > 1e-9:
                xx.append(bin_mid(k)); yy.append(g)
        fr = loglog_fit(xx, yy)
        print(f"  real model {gram} per-bin log-log fit: slope={fr[0]:+.3f} R2={fr[1]:.3f} n={fr[2]}")
    print(f"  strict-Zipf token-weighted contribution slope: "
          f"{f2[0]:+.3f} R2={f2[1]:.3f} (n={f2[2]})")


if __name__ == "__main__":
    main()
