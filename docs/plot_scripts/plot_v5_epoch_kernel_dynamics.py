#!/usr/bin/env python3
"""20-epoch replay dynamics: verdict on the one-state plateau (§41).

Primary inputs are the 20-epoch epoch-boundary records committed in
docs/appendices/s1_scaling_three_axis/s1_epoch_long_replay_points.csv
(run_ids containing "20ep").  The plotted net n-gram gap subtracts the
matched no-gram control at the same epoch.

The dashed curve is the one-state recurrence fit

    G_e = G_1 + A * (1 - q ** (e - 1)),  0 < q < 1,

fit over e=1..20.  Verdict: it fits in-window (R^2 ~ .997) but its plateau is
window-dependent (12.66 fit on e=1..10 -> 23.67 fit on e=1..20), and the
measured increments are still ~0.5 at e=20.  No plateau is observed; growth
is sublinear.  The earlier "plateau 13.58" extrapolation is withdrawn.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "docs"
    / "appendices"
    / "s1_scaling_three_axis"
    / "s1_epoch_long_replay_points.csv"
)
OUT_DIR = ROOT / "docs" / "figs" / "theory"
FIGURE = OUT_DIR / "fig_v5_epoch_kernel_dynamics.png"
FIGURE_SVG = OUT_DIR / "fig_v5_epoch_kernel_dynamics.svg"
FIT_CSV = OUT_DIR / "theory_epoch_kernel_fit.csv"
N_EPOCHS = 20


def load_epoch_series():
    rows = list(csv.DictReader(SOURCE.open()))
    by_arm = {}
    for row in rows:
        if "20ep" not in row["run_id"]:
            continue
        by_arm.setdefault(row["arm"], {})[int(row["epoch"])] = float(row["gap"])
    required = ("trigram-only", "nogram")
    missing = [arm for arm in required if arm not in by_arm]
    if missing:
        raise RuntimeError(f"missing 20ep arms in {SOURCE}: {missing}")
    epochs = np.asarray(sorted(set(by_arm[required[0]]) & set(by_arm[required[1]])))
    if not np.array_equal(epochs, np.arange(1, N_EPOCHS + 1)):
        raise RuntimeError(f"expected epochs 1..{N_EPOCHS}, got {epochs.tolist()}")
    trigram = np.asarray([by_arm["trigram-only"][int(e)] for e in epochs])
    nogram = np.asarray([by_arm["nogram"][int(e)] for e in epochs])
    return epochs.astype(float), trigram, nogram, trigram - nogram


def fit_recurrence(epochs, values):
    """Fit q by golden-section search; G_1 is fixed to the measured first point."""
    baseline = float(values[0])

    def solve_amplitude(q):
        basis = 1.0 - q ** (epochs - 1.0)
        denominator = float(np.dot(basis, basis))
        amplitude = float(np.dot(basis, values - baseline) / denominator)
        prediction = baseline + amplitude * basis
        sse = float(np.sum((values - prediction) ** 2))
        return sse, amplitude, prediction

    left, right = 1e-6, 0.999999
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    x1 = right - (right - left) / phi
    x2 = left + (right - left) / phi
    f1 = solve_amplitude(x1)[0]
    f2 = solve_amplitude(x2)[0]
    for _ in range(100):
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - (right - left) / phi
            f1 = solve_amplitude(x1)[0]
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + (right - left) / phi
            f2 = solve_amplitude(x2)[0]
    q = (left + right) / 2.0
    sse, amplitude, prediction = solve_amplitude(q)
    total = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - sse / total
    tau = -1.0 / np.log(q)
    plateau = baseline + amplitude
    return q, tau, plateau, r_squared, prediction


def write_fit_csv(epochs, trigram, nogram, net, prediction, q, tau, plateau, r_squared):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with FIT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "trigram_gap",
                "nogram_gap",
                "net_ngram_gap",
                "recurrence_fit",
                "q",
                "tau_epochs_extrapolated",
                "plateau_extrapolated",
                "r_squared_in_window",
                "source",
            ]
        )
        for e, tri, no, observed, fitted in zip(
            epochs, trigram, nogram, net, prediction
        ):
            writer.writerow(
                [
                    int(e),
                    f"{tri:.9f}",
                    f"{no:.9f}",
                    f"{observed:.9f}",
                    f"{fitted:.9f}",
                    f"{q:.9f}",
                    f"{tau:.6f}",
                    f"{plateau:.6f}",
                    f"{r_squared:.9f}",
                    str(SOURCE.relative_to(ROOT)),
                ]
            )


def plot(epochs, trigram, nogram, net, prediction, q, plateau, r_squared):
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.65))
    left, right = axes

    left.plot(
        epochs,
        trigram,
        marker="o",
        markersize=4.2,
        linewidth=0.9,
        color="#2d6f9f",
        label="trigram-only gap (20ep run)",
    )
    left.plot(
        epochs,
        nogram,
        marker="o",
        markersize=4.2,
        linewidth=0.9,
        color="#777777",
        label="matched no-gram gap (20ep run)",
    )
    left.plot(
        epochs,
        net,
        marker="o",
        markersize=4.5,
        linewidth=1.15,
        color="#c4493d",
        label="net n-gram gap (difference)",
    )
    left.plot(
        epochs,
        prediction,
        linestyle="--",
        linewidth=1.15,
        color="#2a8c62",
        label=f"one-state fit e1–20: q={q:.3f}, R²={r_squared:.4f}",
    )
    left.axhline(0.0, color="#9ca3af", linewidth=0.6, linestyle=":")
    left.set_xlabel("completed replay pass / epoch e")
    left.set_ylabel("gap at epoch boundary")
    left.set_title("20 passes: net gap still grows, sublinearly")
    left.legend(frameon=False, fontsize=8.2)

    observed_increment = np.diff(net)
    fitted_increment = np.diff(prediction)
    boundaries = epochs[1:]
    right.plot(
        boundaries,
        observed_increment,
        marker="o",
        markersize=4.5,
        linewidth=1.0,
        color="#c4493d",
        label="measured net increment",
    )
    right.plot(
        boundaries,
        fitted_increment,
        marker="o",
        markersize=3.4,
        linestyle="--",
        linewidth=1.0,
        color="#2a8c62",
        label="one-state fit increment",
    )
    right.axhline(0.0, color="#9ca3af", linewidth=0.6, linestyle=":")
    right.set_xlabel("epoch boundary e-1 → e")
    right.set_ylabel("increment in net n-gram gap")
    right.set_title("increments decay slowly, still ≈ +0.5 at e20")
    right.legend(frameon=False, fontsize=8.2)

    for axis in axes:
        axis.grid(alpha=0.22)
        axis.set_xticks(np.arange(2, N_EPOCHS + 1, 2))

    figure.suptitle(
        "V5 replay dynamics · seed 42 · 337 steps/pass · 20-epoch verdict\n"
        f"no plateau observed; one-state plateau is window-dependent "
        f"(12.66 fit on e1–10 → {plateau:.2f} fit on e1–20)",
        fontsize=10.5,
    )
    figure.tight_layout()
    figure.savefig(FIGURE, dpi=190, bbox_inches="tight")
    figure.savefig(FIGURE_SVG, bbox_inches="tight")
    plt.close(figure)


def main():
    epochs, trigram, nogram, net = load_epoch_series()
    q, tau, plateau, r_squared, prediction = fit_recurrence(epochs, net)
    write_fit_csv(
        epochs,
        trigram,
        nogram,
        net,
        prediction,
        q,
        tau,
        plateau,
        r_squared,
    )
    plot(epochs, trigram, nogram, net, prediction, q, plateau, r_squared)
    print(FIGURE.relative_to(ROOT))
    print(FIT_CSV.relative_to(ROOT))
    print(
        f"q={q:.9f} tau={tau:.4f} plateau(extrapolated)={plateau:.4f} "
        f"R2(in-window)={r_squared:.9f}"
    )


if __name__ == "__main__":
    main()
