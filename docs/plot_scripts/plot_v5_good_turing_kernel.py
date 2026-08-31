#!/usr/bin/env python3
"""Figure: measured gap vs the Good-Turing missing-mass kernel C*M(f).

Panel A (bigram, aligned shape test): measured binned per-f gap against
C*M(f) with a single fitted amplitude; both series share the same matched
geometric bins and validation-token weights.  The 100<fbar<=721.2 region is
a holdout: C is refit on fbar<=100 only and the dashed curve predicts it.

Panel B (trigram, cross-check): exact-f M(f) from continuation counts with
the registered 7-bin gap-fit slope overlaid as a reference power law; the
windows/weights differ, so this is a slope comparison, not a fitted overlay.

All numbers are recomputed from committed CSVs, mirroring
summarize_good_turing_kernel.py.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
THEORY = ROOT / "docs" / "figs" / "theory"
BIGRAM = THEORY / "theory_missing_mass_bigram.csv"
TRIGRAM = THEORY / "theory_missing_mass_trigram.csv"
SCALING_FITS = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_scaling_fits.csv"
OUTPUT = THEORY / "fig_v5_good_turing_kernel.png"


def weighted_log_slope(x, y, w):
    lx, ly = np.log(x), np.log(y)
    xb = np.average(lx, weights=w)
    yb = np.average(ly, weights=w)
    return float(np.sum(w * (lx - xb) * (ly - yb)) / np.sum(w * (lx - xb) ** 2))


def main():
    rows = list(csv.DictReader(BIGRAM.open()))
    f = np.array([float(r["fbar"]) for r in rows])
    gap = np.array([float(r["gap_meas"]) for r in rows])
    miss = np.array([float(r["M"]) for r in rows])
    w = np.array([float(r["val_tokens"]) for r in rows])
    keep = (f <= 721.2) & (gap > 0) & (miss > 0)
    f, gap, miss, w = (v[keep] for v in (f, gap, miss, w))

    beta_gap = -weighted_log_slope(f, gap, w)
    beta_m = -weighted_log_slope(f, miss, w)
    amp = float(np.exp(np.average(np.log(gap) - np.log(miss), weights=w)))
    pred = amp * miss
    mean_gap = np.average(gap, weights=w)
    r2 = 1.0 - float(np.sum(w * (gap - pred) ** 2) / np.sum(w * (gap - mean_gap) ** 2))

    train = f <= 100
    amp_hold = float(np.exp(np.average(np.log(gap[train]) - np.log(miss[train]), weights=w[train])))
    pred_hold = amp_hold * miss[~train]
    mean_hold = np.average(gap[~train], weights=w[~train])
    r2_hold = 1.0 - float(
        np.sum(w[~train] * (gap[~train] - pred_hold) ** 2)
        / np.sum(w[~train] * (gap[~train] - mean_hold) ** 2)
    )

    tri = list(csv.DictReader(TRIGRAM.open()))
    ft = np.array([float(r["f"]) for r in tri])
    mt = np.array([float(r["M"]) for r in tri])
    nt = np.array([float(r["n_contexts"]) for r in tri])
    kt = (ft >= 1) & (ft <= 100) & (mt > 0) & (nt > 0)
    beta_m_tri = -weighted_log_slope(ft[kt], mt[kt], nt[kt])
    fits = list(csv.DictReader(SCALING_FITS.open()))
    beta_gap_tri = abs(float(next(
        r for r in fits if r["family"] == "frequency_exact" and r["branch"] == "trigram"
    )["slope"]))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.9))

    sizes = 25 + 55 * w / w.max()
    axa.scatter(f, gap, s=sizes, color="#2d6f9f", zorder=3,
                label="measured binned gap (size ~ val tokens)")
    ff = np.geomspace(f.min(), f.max(), 200)
    mm = np.interp(np.log(ff), np.log(f), np.log(miss))
    axa.plot(ff, amp * np.exp(mm), color="#1a7f37", lw=1.6,
             label=f"C*M(f), C={amp:.2f} (fit on full window)")
    axa.plot(ff[ff > 100], amp_hold * np.exp(mm[ff > 100]), color="#c4493d",
             lw=1.6, ls="--", label=f"C={amp_hold:.2f} fit on f<=100 only (holdout)")
    axa.axvspan(100, f.max(), color="#c4493d", alpha=0.05)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlabel("exact train context count f (bigram branch)")
    axa.set_ylabel("gap (val - train probe)")
    axa.set_title("A · bigram: aligned shape test, one amplitude only")
    axa.legend(fontsize=8.5, loc="lower left")
    axa.grid(alpha=0.25, which="both")
    axa.text(0.03, 0.30,
             f"beta_gap = {beta_gap:.4f}\nbeta_M   = {beta_m:.4f}\n"
             f"weighted R2 = {r2:.4f}\nholdout R2 (f>100) = {r2_hold:.4f}",
             transform=axa.transAxes, fontsize=9, color="#1c2733",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f0f6fb", ec="#dde5ec"))

    axb.scatter(ft, mt, s=8, color="#2d6f9f", alpha=0.6, label="M(f) from counts")
    fref = np.geomspace(1, 100, 100)
    mref = np.exp(np.average(np.log(mt[kt]) + beta_gap_tri * np.log(ft[kt]), weights=nt[kt]))
    axb.plot(fref, mref * fref ** (-beta_gap_tri), color="#c4493d", lw=1.6,
             label=f"reference slope -{beta_gap_tri:.3f} (registered gap fit)")
    axb.set_xscale("log")
    axb.set_yscale("log")
    axb.set_xlabel("exact train context count f (trigram branch)")
    axb.set_ylabel("M(f) = N1(f) / (f * n(f))")
    axb.set_title("B · trigram: cross-check (windows/weights differ)")
    axb.legend(fontsize=8.5, loc="lower left")
    axb.grid(alpha=0.25, which="both")
    axb.text(0.03, 0.14,
             f"beta_M (f in [1,100], n(f) weighted) = {beta_m_tri:.4f}\n"
             f"beta_gap (registered 7-bin fit)      = {beta_gap_tri:.4f}",
             transform=axb.transAxes, fontsize=9, color="#1c2733",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f0f6fb", ec="#dde5ec"))

    fig.suptitle("gap(f) = C * M(f): the Good-Turing kernel fits with a single amplitude", y=1.0)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT.relative_to(ROOT))
    print(f"bigram: C={amp:.4f} beta_gap={beta_gap:.4f} beta_M={beta_m:.4f} "
          f"R2={r2:.4f} holdout(C={amp_hold:.4f}) R2={r2_hold:.4f}")
    print(f"trigram: beta_M={beta_m_tri:.4f} beta_gap={beta_gap_tri:.4f}")


if __name__ == "__main__":
    main()
