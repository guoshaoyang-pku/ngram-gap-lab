#!/usr/bin/env python3
"""Numerical test of the interference-dilution (soft ownership) model:

  Gap_pred(R) = C * sum_c  mass_c * g0(f_c) * f_c / (f_c + T/R)

where f_c are the exact train hit counts from data/freq_index.npz, T/R is the
mean-field interfering traffic per row, and g0 is the per-context gap kernel.
We ask: which kernel exponent beta0 (g0 = f^-beta0) reproduces the measured
local table-size slopes (net-gap 0.576 bigram / 0.665 trigram) inside the same
R windows? Also overlays the best kernel prediction on the measured points.

Inputs: data/freq_index.npz, docs/appendices/s1_scaling_three_axis/s1_table_size_points.csv
Output: docs/figs/theory/fig_v5_interference_model_vs_data.png
        docs/figs/theory/theory_interference_scan.csv
"""
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.normpath(os.path.join(ROOT, "..", "data", "freq_index.npz"))
TBL = os.path.join(ROOT, "appendices", "s1_scaling_three_axis", "s1_table_size_points.csv")
OUT = os.path.join(ROOT, "figs", "theory")
WINDOW = {"bigram": (2e3, 2e5), "trigram": (1e5, 9.3e5)}
MEAS = {"bigram": 0.5761, "trigram": 0.6648}
FLOOR = 0.02


def local_slope(R, G, lo, hi):
    m = (R >= lo) & (R <= hi) & (G > 0)
    a, _ = np.polyfit(np.log(R[m]), np.log(G[m]), 1)
    return a


def main():
    z = np.load(NPZ)
    Rgrid = np.logspace(2, 7, 61)
    scan_rows = []
    # measured points
    meas = {"bigram": [], "trigram": []}
    with open(TBL) as f:
        for r in csv.DictReader(f):
            meas[r["branch"]].append((float(r["R"]), float(r["final_gap"])))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, br in zip(axes, ("bigram", "trigram")):
        counts = z[br + "_counts"].astype(np.int64)
        T = counts.sum()
        cc = np.bincount(counts)          # n(f)
        f = np.arange(cc.size, dtype=np.float64)
        n_f = cc.astype(np.float64)
        mask = (f >= 1) & (n_f > 0)
        f, n_f = f[mask], n_f[mask]
        massf = f * n_f / T               # token-mass fraction at count f
        best = None
        for b0 in (0.0, 0.25, 0.318, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            g0 = f ** (-b0)
            # soft interference share
            Gp = np.array([(massf * g0 * f / (f + T / R)).sum() for R in Rgrid])
            s = local_slope(Rgrid, Gp, *WINDOW[br])
            # hard rank-threshold variant: keep top-R ranks
            c_sorted = np.sort(z[br + "_counts"])[::-1].astype(np.float64)
            csum = np.cumsum(c_sorted) / T
            gh = c_sorted ** (-b0)
            wgap = c_sorted * gh / T
            cw = np.cumsum(wgap)
            Gh = np.array([cw[min(int(R), len(cw)) - 1] for R in Rgrid])
            sh = local_slope(Rgrid, Gh, *WINDOW[br])
            scan_rows.append(dict(branch=br, beta0=b0, slope_soft=round(s, 4),
                                  slope_hard=round(sh, 4), meas=MEAS[br]))
            print(f"{br} beta0={b0:4.2f}: soft-slope={s:.3f} hard-slope={sh:.3f} "
                  f"(measured {MEAS[br]})")
            if best is None or abs(s - MEAS[br]) < abs(best[1] - MEAS[br]):
                best = (b0, s, Gp.copy())
        b0, s, Gp = best
        pts = np.array([(x, g) for x, g in meas[br] if x > 0 and g - FLOOR > 0])
        if len(pts):
            lo, hi = WINDOW[br]
            msel = (pts[:, 0] >= lo) & (pts[:, 0] <= hi)
            anchor_meas = np.exp(np.mean(np.log(pts[msel, 1] - FLOOR)))
            anchor_pred = np.exp(np.mean(np.log(np.interp(pts[msel, 0], Rgrid, Gp))))
            scale = anchor_meas / anchor_pred
            ax.loglog(pts[:, 0], pts[:, 1] - FLOOR, "o", ms=5, color="#1f77b4",
                      label="measured net gap (G-0.02), 31 raw runs")
        else:
            scale = 1.0
        ax.loglog(Rgrid, Gp * scale, "-", color="#d62728", lw=1.6,
                  label=f"interference model, kernel f^-{b0:g}\n"
                        f"(window slope {s:.3f} vs measured {MEAS[br]:.3f})")
        lo, hi = WINDOW[br]
        ax.axvspan(lo, hi, color="orange", alpha=0.15)
        ax.set_title(f"{br}-only table axis (seed 42, step 1000)")
        ax.set_xlabel("clean table rows R")
        ax.set_ylabel("net gap")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Interference-dilution model vs measured table-size axis "
                 "(single free amplitude per branch)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT, "fig_v5_interference_model_vs_data.png")
    fig.savefig(out, dpi=160)
    print("saved", out)
    with open(os.path.join(OUT, "theory_interference_scan.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(scan_rows[0].keys()))
        w.writeheader()
        w.writerows(scan_rows)


if __name__ == "__main__":
    main()
