#!/usr/bin/env python3
"""Generate traceable plots for the theory-Zipf run with frequency diagnostics."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")
(HERE / ".mplconfig").mkdir(exist_ok=True)
(HERE / ".cache").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_LOG = HERE / "train_log.jsonl"
TABLE_LOG = HERE / "table_norm.jsonl"
EXACT_LOG = HERE / "exact_freq_loss.jsonl"
SUMMARY = HERE / "summary.json"


def read_jsonl(path: Path) -> list[dict]:
    """Read non-empty JSONL records in file order."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_summary(path: Path) -> dict:
    """Read the recorded training summary."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_loss_gap(train_records: list[dict], summary: dict) -> Path:
    """Plot online train loss, fixed validation loss, and global online gap."""
    steps = [row["step"] for row in train_records]
    train_loss = [row["train_loss"] for row in train_records]
    val_loss = [row["val_loss"] for row in train_records]
    gap = [row["gap"] for row in train_records]

    fig, (ax_loss, ax_gap) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_loss.plot(steps, train_loss, label="Online train loss", color="#1f77b4", linewidth=1.8)
    ax_loss.plot(steps, val_loss, label="Fixed validation loss", color="#d62728", linewidth=1.8)
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title(f"Theory-Zipf toy: loss and gap ({summary['run_id']}, seed {summary['seed']})")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(frameon=False, ncol=2)

    ax_gap.plot(steps, gap, color="#2ca02c", linewidth=1.8, label="Online gap")
    ax_gap.axhline(0.0, color="black", linewidth=0.8)
    ax_gap.set_xlabel("Logged step")
    ax_gap.set_ylabel("Val − train")
    ax_gap.grid(alpha=0.25)
    ax_gap.legend(frameon=False)
    fig.text(
        0.01,
        0.01,
        "Gap = fixed validation loss − current-batch online training loss; "
        f"endpoint step {summary['steps']}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output = HERE / "loss_gap_curve.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_gap_table_rms(
    train_records: list[dict], table_records: list[dict], summary: dict
) -> Path:
    """Plot global gap and table RMS when table-RMS fields are recorded.

    A no-ngram run still emits one table-norm record per logged step, but it
    intentionally has no bigram/trigram RMS fields.  In that case the figure
    remains useful as a gap-only diagnostic and states that no tables exist.
    """
    train_by_step = {row["step"]: row for row in train_records}
    table_by_step = {row["step"]: row for row in table_records}
    steps = sorted(set(train_by_step) & set(table_by_step))
    gap = [train_by_step[step]["gap"] for step in steps]

    fig, ax_gap = plt.subplots(figsize=(10, 5.5))
    gap_line = ax_gap.plot(steps, gap, color="#2ca02c", linewidth=1.8, label="Online gap")
    ax_gap.axhline(0.0, color="black", linewidth=0.8)
    ax_gap.set_xlabel("Logged step")
    ax_gap.set_ylabel("Val − train", color="#2ca02c")
    has_table_rms = bool(steps) and all(
        "bigram.layer_01.table_0.rms" in table_by_step[step]
        and "trigram.layer_01.table_0.rms" in table_by_step[step]
        for step in steps
    )
    if has_table_rms:
        ax_rms = ax_gap.twinx()
        bigram_rms = [table_by_step[step]["bigram.layer_01.table_0.rms"] for step in steps]
        trigram_rms = [table_by_step[step]["trigram.layer_01.table_0.rms"] for step in steps]
        bi_line = ax_rms.plot(
            steps, bigram_rms, color="#1f77b4", linewidth=1.6, label="Bigram table RMS"
        )
        tri_line = ax_rms.plot(
            steps, trigram_rms, color="#ff7f0e", linewidth=1.6, label="Trigram table RMS"
        )
        ax_rms.set_ylabel("Table RMS", color="#555555")
        lines = gap_line + bi_line + tri_line
        labels = [line.get_label() for line in lines]
        title = f"Gap and n-gram table RMS ({summary['run_id']})"
        note = f"Recorded table norms; seed {summary['seed']}, endpoint step {summary['steps']}"
    else:
        lines = gap_line
        labels = ["Online gap"]
        title = f"No-ngram global gap ({summary['run_id']})"
        note = (
            f"N-gram tables disabled; table_norm.jsonl has no RMS fields; "
            f"seed {summary['seed']}, endpoint step {summary['steps']}"
        )
    ax_gap.set_title(title)
    ax_gap.grid(alpha=0.25)
    ax_gap.legend(lines, labels, frameon=False, loc="upper left")
    fig.text(0.01, 0.01, note, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output = HERE / "gap_table_rms_curve.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def final_shared_rows(exact_records: list[dict], branch: str) -> list[dict]:
    """Extract final-step shared-context exact-f rows for one branch."""
    if not exact_records:
        return []
    final = exact_records[-1]
    rows = final["shared"][branch].get("per_f", {})
    return [
        {
            "f": int(frequency),
            "gap": float(row["gap"]),
            "shared_contexts": int(row["shared_contexts"]),
        }
        for frequency, row in rows.items()
    ]


def geometric_bins(rows: list[dict], min_contexts: int = 5) -> list[dict]:
    """Pool exact-f rows into fixed geometric bins using context-count weights."""
    edges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    pooled = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = [
            row for row in rows
            if lower <= row["f"] < upper
            and row["gap"] > 0
            and row["shared_contexts"] >= min_contexts
            and math.isfinite(row["gap"])
        ]
        if not selected:
            continue
        weight = sum(row["shared_contexts"] for row in selected)
        log_f = sum(row["shared_contexts"] * math.log(row["f"]) for row in selected) / weight
        log_gap = sum(row["shared_contexts"] * math.log(row["gap"]) for row in selected) / weight
        pooled.append({
            "f": math.exp(log_f),
            "gap": math.exp(log_gap),
            "weight": weight,
            "range": f"{lower}–{upper - 1}",
        })
    return pooled


def fit_power_law(points: list[dict]) -> tuple[float, float]:
    """Fit gap proportional to f**(-beta) and return (beta, R^2)."""
    if len(points) < 3:
        return float("nan"), float("nan")
    x = [math.log(point["f"]) for point in points]
    y = [math.log(point["gap"]) for point in points]
    weights = [point["weight"] for point in points]
    weight_sum = sum(weights)
    x_bar = sum(weight * value for weight, value in zip(weights, x)) / weight_sum
    y_bar = sum(weight * value for weight, value in zip(weights, y)) / weight_sum
    slope = sum(weight * (x_i - x_bar) * (y_i - y_bar) for weight, x_i, y_i in zip(weights, x, y))
    slope /= sum(weight * (x_i - x_bar) ** 2 for weight, x_i in zip(weights, x))
    predicted = [y_bar + slope * (x_i - x_bar) for x_i in x]
    residual = sum(weight * (y_i - y_hat) ** 2 for weight, y_i, y_hat in zip(weights, y, predicted))
    total = sum(weight * (y_i - y_bar) ** 2 for weight, y_i in zip(weights, y))
    return -slope, 1.0 - residual / total if total > 0 else float("nan")


def plot_gap_frequency(exact_records: list[dict], summary: dict, branch: str) -> Path:
    """Plot final shared-context gap against exact train hit count on log-log axes."""
    rows = final_shared_rows(exact_records, branch)
    points = geometric_bins(rows)
    beta, r_squared = fit_power_law(points)
    raw = [row for row in rows if row["f"] > 0 and row["gap"] > 0 and row["shared_contexts"] >= 5]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(
        [row["f"] for row in raw],
        [row["gap"] for row in raw],
        s=8,
        alpha=0.18,
        color="#999999",
        label="Exact f (positive gap, n≥5)",
    )
    ax.plot(
        [point["f"] for point in points],
        [point["gap"] for point in points],
        "o-",
        color="#1f77b4",
        linewidth=1.8,
        markersize=4,
        label="Geometric-bin weighted mean",
    )
    if len(points) >= 2 and math.isfinite(beta):
        anchor = points[0]
        fit_x = [point["f"] for point in points]
        fit_y = [anchor["gap"] * (value / anchor["f"]) ** (-beta) for value in fit_x]
        ax.plot(fit_x, fit_y, "--", color="#d62728", linewidth=1.4, label=f"fit β={beta:.3f}, R²={r_squared:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Exact train context hit count f")
    ax.set_ylabel("Shared-context gap (val − train)")
    ax.set_title(f"{branch.capitalize()} gap(f), final step {summary['steps']} ({summary['run_id']})")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.text(
        0.01,
        0.01,
        "f=0 novel contexts excluded; positive shared-context gaps only; "
        "bins weighted by shared-context count",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output = HERE / f"gap_vs_frequency_{branch}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"{branch}: exact_rows={len(rows)} bins={len(points)} beta={beta:.6f} R2={r_squared:.6f}")
    return output


def main() -> None:
    """Read recorded artifacts and generate all four figures."""
    train_records = read_jsonl(TRAIN_LOG)
    table_records = read_jsonl(TABLE_LOG)
    exact_records = read_jsonl(EXACT_LOG)
    summary = read_summary(SUMMARY)
    outputs = [
        plot_loss_gap(train_records, summary),
        plot_gap_table_rms(train_records, table_records, summary),
        plot_gap_frequency(exact_records, summary, "bigram"),
        plot_gap_frequency(exact_records, summary, "trigram"),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
