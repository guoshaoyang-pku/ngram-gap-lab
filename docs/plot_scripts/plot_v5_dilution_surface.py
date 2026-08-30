#!/usr/bin/env python3
"""Empirical dilution surface s_emp(f,R) from single-table bigram runs.

For each s1v5_128_tbl_bi1_R{R} run, bin the last exact_freq_loss record into
geometric f bins (train probe vs val, token_count >= 200 on both sides), then
normalize each bin by the value at the reference (largest) R:
    s_emp(f,R) = gap(f,R) / gap(f,R_ref)
Questions answered directly from data, no model assumed:
  Q1 is suppression f-dependent at fixed R (beta drift) ?
  Q2 does s_emp collapse on the mean-field coordinate x = f/(f+T/R) ?
  Q3 or on the load coordinate K/R alone (f-independent decoding threshold) ?

Output: docs/figs/theory/fig_v5_dilution_surface.png + theory_dilution_surface.csv
"""
import csv
import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.normpath(os.path.join(ROOT, "..", "data", "runs_scaling"))
NPZ = os.path.normpath(os.path.join(ROOT, "..", "data", "freq_index.npz"))
OUT = os.path.join(ROOT, "figs", "theory")
T = 49_660_000  # bigram-branch traffic approx; refined below from index
EDGES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 32768])


def binned_gap(path):
    last = None
    for line in open(path):
        last = line
    rec = json.loads(last)
    tr, va = rec["train"]["bigram"], rec["val"]["bigram"]
    out = {}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        tw = tv = gv = gt = 0.0
        for k, v in va.items():
            f = int(k)
            if lo <= f < hi and k in tr:
                tv += v["token_count"]
                gv += v["loss_sum"]
        for k, v in tr.items():
            f = int(k)
            if lo <= f < hi and k in va:
                tw += v["token_count"]
                gt += v["loss_sum"]
        if tv >= 400 and tw >= 400:
            out[(lo, hi)] = (gv / tv - gt / tw, tv)
    return out, rec["step"]


def main():
    z = np.load(NPZ)
    K = int(z["bigram_counts"].size)
    Tt = int(z["bigram_counts"].sum())
    runs = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "s1v5_128_tbl_bi1_R*_fixed"))):
        m = re.search(r"_R(\d+)_fixed$", d)
        p = os.path.join(d, "exact_freq_loss.jsonl")
        if not m or not os.path.exists(p):
            continue
        R = int(m.group(1))
        try:
            g, step = binned_gap(p)
        except Exception as e:
            print("skip", d, e)
            continue
        if g:
            runs[R] = g
    Rs = sorted(runs)
    print("runs loaded:", Rs)
    Rref = Rs[-1]
    ref = runs[Rref]
    rows = []
    for R in Rs:
        for b, (gap, wt) in runs[R].items():
            if b in ref and ref[b][0] > 0.05:
                fbar = np.sqrt(b[0] * (b[1] - 1))
                rows.append(dict(R=R, f_lo=b[0], f_hi=b[1], fbar=round(fbar, 1),
                                 gap=round(gap, 5), gap_ref=round(ref[b][0], 5),
                                 s_emp=round(gap / ref[b][0], 5),
                                 x_meanfield=round(fbar / (fbar + Tt / R), 5),
                                 load=round(K / R, 3), val_tokens=int(wt)))
    with open(os.path.join(OUT, "theory_dilution_surface.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
    cmap = plt.cm.viridis
    Rsel = [R for R in Rs if R >= 512]
    # panel 1: s_emp vs f, one line per R
    ax = axes[0]
    for i, R in enumerate(Rsel):
        pts = sorted((r["fbar"], r["s_emp"]) for r in rows if r["R"] == R)
        if len(pts) < 3:
            continue
        xs, ys = zip(*pts)
        ax.semilogx(xs, ys, "o-", ms=3.5, lw=1,
                    color=cmap(i / max(len(Rsel) - 1, 1)), label=f"R={R:,}")
    ax.set_xlabel("f (bin geometric mean)"); ax.set_ylabel("s_emp = gap(f,R)/gap(f,R_ref)")
    ax.set_title("Q1: f-dependence of suppression"); ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    # panel 2: collapse on mean-field coordinate
    ax = axes[1]
    sc = ax.scatter([r["x_meanfield"] for r in rows], [r["s_emp"] for r in rows],
                    c=np.log10([r["R"] for r in rows]), cmap=cmap, s=18)
    xg = np.logspace(-5, 0, 60)
    ax.plot(xg, xg, "k--", lw=1, label="s = x (pure mean-field)")
    ax.set_xscale("log"); ax.set_xlabel("x = f/(f + T/R)")
    ax.set_title("Q2: mean-field collapse test"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8); plt.colorbar(sc, ax=ax, label="log10 R")
    # panel 3: s_emp vs load, colored by f
    ax = axes[2]
    sc = ax.scatter([r["load"] for r in rows], [r["s_emp"] for r in rows],
                    c=np.log10([max(r["fbar"], 1) for r in rows]), cmap="plasma", s=18)
    ax.set_xscale("log"); ax.set_xlabel("load K/R"); ax.set_title("Q3: load-only test")
    ax.grid(alpha=0.3, which="both"); plt.colorbar(sc, ax=ax, label="log10 f")
    fig.suptitle(f"Empirical dilution surface, bigram single-table runs (ref R={Rref:,}, step 1000, seed 42)",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_v5_dilution_surface.png")
    fig.savefig(out, dpi=160)
    print("saved", out, f"rows={len(rows)} K={K} T={Tt}")


if __name__ == "__main__":
    main()
