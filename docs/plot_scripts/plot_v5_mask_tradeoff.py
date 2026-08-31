#!/usr/bin/env python3
"""Mask-low-frequency trade-off: gap removed vs contexts kept, per branch.

For each branch (trigram left, bigram right), as a function of the mask
threshold t (contexts with train hit-count f <= t are masked):
  * solid line  = static gap attribution removed: cumulative share of the
    step-1000 online gap contributed by f <= t tokens (per-bucket gap x
    full-epoch mass, positive parts only);
  * dashed line = token mass kept: share of epoch tokens whose context has
    f > t (they still receive n-gram input);
  * gray squares = DYNAMIC removal measured in the causalv5m sweep
    (mask_low from epoch 2, BOTH branches masked, gap at step 1000 vs the
    2.724 control) -- plotted on both panels as a reference; it is not
    branch-separated.  The trigram-only dynamic sweep (netv5t) fills this in.

Marks: vertical dotted line where mass kept crosses 90% (bigram t~5-6;
trigram has no such point above f=1), horizontal at 95% removal.

Data: docs/figs/theory/freq_train_mass_hist.csv (from the shard-1 frequency
index) + step-1000 record of the main run's freq_bin_loss.jsonl.
"""
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import v5_style as S

ROOT = Path(__file__).resolve().parents[2]
RUNS_FIXED = Path(os.environ.get("NGLAB_RUNS_FIXED", ROOT / "data" / "runs_fixed"))
FREQ_JSONL = RUNS_FIXED / "nglab1x_input_v5_128x_freq10_fixed" / "freq_bin_loss.jsonl"
HIST = ROOT / "docs" / "figs" / "theory" / "freq_train_mass_hist.csv"
OUT = ROOT / "docs" / "figs" / "main"

# dynamic joint-mask sweep (both branches, mask from e1, step-1000 gap;
# control = 2.724) from experiment-log §35.8.1
DYN = {0: 2.945, 1: 1.578, 2: 1.320, 4: 1.046, 8: 0.765, 200: 0.101}
CONTROL = 2.724


def bkey(lo, hi):
    if lo == hi:
        return str(lo)
    if hi >= 10**12:
        return f"{lo}+"
    fmt = lambda x: f"{x // 1000}k" if x >= 1000 and x % 1000 == 0 else str(x)
    return f"{fmt(lo)}-{fmt(hi)}"


def load():
    hist = {}
    for r in csv.DictReader(HIST.open()):
        hi = int(r["f_hi"]) if r["f_hi"] else 10**12
        hist.setdefault(r["branch"], []).append(
            (int(r["f_lo"]), hi, int(r["n_contexts"]), int(r["token_mass"])))
    rec = next(json.loads(l) for l in FREQ_JSONL.open()
               if json.loads(l)["step"] == 1000)
    out = {}
    for branch, rows in hist.items():
        tb, vb = rec["train"][branch], rec["val"][branch]
        edges, removed, kept_mass, kept_type = [], [], [], []
        tot_mass = sum(x[3] for x in rows)
        tot_type = sum(x[2] for x in rows)
        contrib = []
        for lo, hi, nc, tm in rows:
            g = vb.get(bkey(lo, hi), {}).get("mean_loss", 0) - \
                tb.get(bkey(lo, hi), {}).get("mean_loss", 0)
            contrib.append((lo, tm * max(g, 0)))
        tot_c = sum(c for _, c in contrib)
        for lo, hi, nc, tm in rows:
            t = lo - 1  # masking f <= lo-1 removes everything below this bucket
            c_le = sum(c for l2, c in contrib if l2 <= t)
            edges.append(max(t, 0))
            removed.append(c_le / tot_c)
            kept_mass.append(sum(x[3] for x in rows if x[0] > t) / tot_mass)
            kept_type.append(sum(x[2] for x in rows if x[0] > t) / tot_type)
        # final point: mask up through the last bucket edge
        t_last = rows[-1][0]
        out[branch] = (np.array(edges), np.array(removed),
                       np.array(kept_mass), np.array(kept_type))
    return out


def main():
    S.apply_style()
    data = load()
    dyn_x = np.array(sorted(DYN))
    dyn_y = np.array([(CONTROL - DYN[t]) / CONTROL for t in dyn_x])

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
    for ax, branch in zip(axes, ("trigram", "bigram")):
        edges, removed, kept_mass, kept_type = data[branch]
        x = np.maximum(edges, 0.5)
        ax.plot(x, removed * 100, "-", color=S.BRANCH_COLORS[branch], lw=1.8,
                marker="o", ms=4, label="static gap removed (f ≤ t share)")
        ax.plot(x, kept_mass * 100, "--", color="#2a8c62", lw=1.8,
                marker="s", ms=4, label="token mass kept (f > t)")
        ax.plot(dyn_x[:-1] + 0.5, dyn_y[:-1] * 100, ":", color="#888888",
                lw=1.2, marker="D", ms=4,
                label="dynamic removal (joint mask, both branches)")
        ax.plot([200.5], [dyn_y[-1] * 100], "D", color="#888888", ms=4)
        ax.axhline(95, color="#bbbbbb", lw=0.8, ls=":")
        ax.axhline(90, color="#dddddd", lw=0.8, ls=":")
        ax.text(205, 96.2, "95%", fontsize=8, color="#999")
        ax.text(205, 91.2, "90%", fontsize=8, color="#bbb")
        ax.set_xscale("log")
        ax.set_xlim(0.45, 400)
        ax.set_ylim(-12, 105)
        ax.set_xticks([0.5, 1, 2, 4, 8, 20, 50, 200],
                      ["0", "1", "2", "4", "8", "20", "50", "200"])
        ax.set_xlabel("mask threshold t (contexts with f ≤ t masked)")
        ax.set_title(branch)
        ax.axhline(0, color="#999", lw=0.6)
    axes[0].set_ylabel("%  (gap removed / mass kept)")
    axes[0].legend(fontsize=8, loc="center left", frameon=False)
    fig.suptitle(
        "mask-low-frequency trade-off per branch · step-1000 online gap × shard-1 epoch mass · 128×\n"
        "trigram: gap AND mass both spread over f — no threshold keeps 90% mass; "
        "bigram: mass concentrated, window exists",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    png, svg = S.save(fig, OUT, "fig_v5_mask_freq_tradeoff")
    print(png.relative_to(ROOT))
    print(svg.relative_to(ROOT))


if __name__ == "__main__":
    main()
