#!/usr/bin/env python3
"""Recompute the Good--Turing kernel exponents quoted by the main report.

Inputs are committed analysis artifacts, not hand-entered report numbers:
  * theory_missing_mass_bigram.csv: matched geometric bins of measured gap and M(f)
  * theory_missing_mass_trigram.csv: exact trigram-context continuation counts
  * s1_scaling_fits.csv: registered trigram exact-frequency gap fit

The trigram row is intentionally labelled a cross-check: its gap and M(f)
slopes use different finite windows/weights, so their numerical agreement is
not an apples-to-apples fitted-shape test.
"""
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
THEORY = ROOT / "docs" / "figs" / "theory"
BIGRAM = THEORY / "theory_missing_mass_bigram.csv"
TRIGRAM = THEORY / "theory_missing_mass_trigram.csv"
SCALING_FITS = (
    ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_scaling_fits.csv"
)
OUTPUT = THEORY / "theory_good_turing_exponent_summary.csv"


def weighted_log_slope(x, y, weights):
    lx = np.log(x)
    ly = np.log(y)
    xbar = np.average(lx, weights=weights)
    ybar = np.average(ly, weights=weights)
    return float(
        np.sum(weights * (lx - xbar) * (ly - ybar))
        / np.sum(weights * (lx - xbar) ** 2)
    )


def bigram_summary():
    rows = list(csv.DictReader(BIGRAM.open()))
    f = np.asarray([float(row["fbar"]) for row in rows])
    gap = np.asarray([float(row["gap_meas"]) for row in rows])
    missing = np.asarray([float(row["M"]) for row in rows])
    weights = np.asarray([float(row["val_tokens"]) for row in rows])
    keep = (f <= 721.2) & (gap > 0) & (missing > 0)
    f, gap, missing, weights = (
        values[keep] for values in (f, gap, missing, weights)
    )
    beta_gap = -weighted_log_slope(f, gap, weights)
    beta_missing = -weighted_log_slope(f, missing, weights)
    log_amplitude = np.average(np.log(gap) - np.log(missing), weights=weights)
    amplitude = float(np.exp(log_amplitude))
    prediction = amplitude * missing
    mean_gap = np.average(gap, weights=weights)
    r_squared = 1.0 - float(
        np.sum(weights * (gap - prediction) ** 2)
        / np.sum(weights * (gap - mean_gap) ** 2)
    )
    multiplicative_rmse = float(
        np.exp(
            np.sqrt(
                np.average((np.log(gap) - np.log(prediction)) ** 2, weights=weights)
            )
        )
        - 1.0
    )

    train = f <= 100
    test = f > 100
    holdout_log_amplitude = np.average(
        np.log(gap[train]) - np.log(missing[train]), weights=weights[train]
    )
    holdout_prediction = np.exp(holdout_log_amplitude) * missing[test]
    holdout_mean = np.average(gap[test], weights=weights[test])
    holdout_r_squared = 1.0 - float(
        np.sum(weights[test] * (gap[test] - holdout_prediction) ** 2)
        / np.sum(weights[test] * (gap[test] - holdout_mean) ** 2)
    )
    holdout_multiplicative_rmse = float(
        np.exp(
            np.sqrt(
                np.average(
                    (np.log(gap[test]) - np.log(holdout_prediction)) ** 2,
                    weights=weights[test],
                )
            )
        )
        - 1.0
    )
    return {
        "branch": "bigram",
        "beta_gap": beta_gap,
        "beta_M": beta_missing,
        "alpha_effective": 1.0 / (1.0 - beta_missing),
        "gap_window": "matched geometric bins, 1<=fbar<=721.2",
        "M_window": "same matched geometric bins",
        "gap_weighting": "validation token count per bin",
        "M_weighting": "same validation token count per bin",
        "comparison_status": "aligned shape test",
        "amplitude_C": amplitude,
        "weighted_linear_R2": r_squared,
        "multiplicative_RMSE": multiplicative_rmse,
        "holdout_rule": "fit C on fbar<=100; test 100<fbar<=721.2",
        "holdout_R2": holdout_r_squared,
        "holdout_multiplicative_RMSE": holdout_multiplicative_rmse,
        "source_gap": str(BIGRAM.relative_to(ROOT)),
        "source_M": str(BIGRAM.relative_to(ROOT)),
    }


def trigram_summary():
    rows = list(csv.DictReader(TRIGRAM.open()))
    f = np.asarray([float(row["f"]) for row in rows])
    missing = np.asarray([float(row["M"]) for row in rows])
    weights = np.asarray([float(row["n_contexts"]) for row in rows])
    keep = (f >= 1) & (f <= 100) & (missing > 0) & (weights > 0)
    beta_missing = -weighted_log_slope(f[keep], missing[keep], weights[keep])

    fits = list(csv.DictReader(SCALING_FITS.open()))
    matches = [
        row
        for row in fits
        if row["family"] == "frequency_exact" and row["branch"] == "trigram"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one registered trigram gap fit, found {len(matches)}")
    fit = matches[0]
    return {
        "branch": "trigram",
        "beta_gap": abs(float(fit["slope"])),
        "beta_M": beta_missing,
        "alpha_effective": 1.0 / (1.0 - beta_missing),
        "gap_window": "registered 7-bin positive-gap geometric fit",
        "M_window": "exact f in [1,100]",
        "gap_weighting": "token-mass weighted geometric bins",
        "M_weighting": "number of contexts n(f)",
        "comparison_status": "cross-check only; finite windows/weights differ",
        "amplitude_C": "",
        "weighted_linear_R2": "",
        "multiplicative_RMSE": "",
        "holdout_rule": "",
        "holdout_R2": "",
        "holdout_multiplicative_RMSE": "",
        "source_gap": str(SCALING_FITS.relative_to(ROOT)),
        "source_M": str(TRIGRAM.relative_to(ROOT)),
    }


def main():
    rows = [bigram_summary(), trigram_summary()]
    fieldnames = list(rows[0])
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            row["branch"],
            f"beta_gap={row['beta_gap']:.6f}",
            f"beta_M={row['beta_M']:.6f}",
            f"alpha_eff={row['alpha_effective']:.6f}",
            row["comparison_status"],
        )
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
