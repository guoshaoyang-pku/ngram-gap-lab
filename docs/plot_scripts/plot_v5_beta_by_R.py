#!/usr/bin/env python3
"""Challenge a branch-constant frequency exponent using existing table-R runs.

For each formal single-branch table-size run, this script applies the same
seven-bin, shared-token-mass-weighted exact-frequency fit used by the S1
frequency diagnostic.  It writes the full audit table and a gated summary plot.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data" / "runs_scaling"
FIGURE = ROOT / "docs" / "figs" / "theory" / "fig_v5_beta_by_table_R.png"
TABLE = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "beta_by_table_R.csv"


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def final_jsonl(path):
    with path.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return max(records, key=lambda record: int(record["step"]))


def exact_rows(record, branch):
    rows = []
    per_frequency = record.get("shared", {}).get(branch, {}).get("per_f", {})
    for frequency_text, values in per_frequency.items():
        frequency = int(frequency_text)
        contexts = int(values.get("shared_contexts", 0))
        gap = values.get("gap")
        if frequency > 0 and contexts >= 32 and gap is not None and math.isfinite(float(gap)):
            rows.append(
                {
                    "f": frequency,
                    "gap": float(gap),
                    "weight": frequency * contexts,
                    "contexts": contexts,
                }
            )
    return rows


def pool_rows(rows, bins=7):
    if not rows:
        return []
    frequencies = np.asarray([row["f"] for row in rows], dtype=float)
    bounds = np.unique(
        np.geomspace(frequencies.min(), frequencies.max() + 1, bins + 1).astype(int)
    )
    pooled = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        members = [row for row in rows if low <= row["f"] < high]
        if not members:
            continue
        weight = sum(row["weight"] for row in members)
        gap = sum(row["gap"] * row["weight"] for row in members) / weight
        if gap > 0:
            pooled.append((math.sqrt(low * max(low, high - 1)), gap))
    return pooled


def fit_power_law(pooled):
    if len(pooled) < 2:
        return float("nan"), float("nan")
    log_f = np.log([point[0] for point in pooled])
    log_gap = np.log([point[1] for point in pooled])
    exponent, intercept = np.polyfit(log_f, log_gap, 1)
    prediction = intercept + exponent * log_f
    total = np.sum((log_gap - log_gap.mean()) ** 2)
    r_squared = 1 - np.sum((log_gap - prediction) ** 2) / total if total else float("nan")
    return -float(exponent), float(r_squared)


def collect():
    output = []
    for branch, prefix in (
        ("bigram", "s1v5_128_tbl_bi1_R"),
        ("trigram", "s1v5_128_tbl_tri1_R"),
    ):
        for run_dir in RUNS.glob(f"{prefix}*_fixed"):
            rows = int(run_dir.name.removeprefix(prefix).removesuffix("_fixed"))
            record = final_jsonl(run_dir / "exact_freq_loss.jsonl")
            pooled = pool_rows(exact_rows(record, branch))
            beta, r_squared = fit_power_law(pooled)
            summary = read_json(run_dir / "summary.json")
            final_gap = float(summary["final_gap"])
            identifiable = len(pooled) >= 5 and r_squared >= 0.90 and final_gap >= 0.25
            output.append(
                {
                    "branch": branch,
                    "R": rows,
                    "run_id": run_dir.name.removesuffix("_fixed"),
                    "step": int(record["step"]),
                    "seed": int(summary["seed"]),
                    "beta": beta,
                    "r2": r_squared,
                    "positive_bins": len(pooled),
                    "final_gap": final_gap,
                    "identifiable": identifiable,
                    "source": str((run_dir / "exact_freq_loss.jsonl").relative_to(ROOT)),
                }
            )
    return sorted(output, key=lambda row: (row["branch"], row["R"]))


def write_table(rows):
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot(rows):
    colors = {"bigram": "#0f766e", "trigram": "#d97706"}
    figure, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)
    for branch in ("bigram", "trigram"):
        branch_rows = [row for row in rows if row["branch"] == branch]
        for identifiable, marker, alpha, suffix in (
            (False, "o", 0.28, "low-signal / weak fit"),
            (True, "o", 0.95, "identified"),
        ):
            selected = [row for row in branch_rows if row["identifiable"] is identifiable]
            if not selected:
                continue
            axes[0].scatter(
                [row["R"] for row in selected],
                [row["beta"] for row in selected],
                color=colors[branch],
                alpha=alpha,
                marker=marker,
                s=35,
                label=f"{branch} · {suffix}",
            )
        axes[1].plot(
            [row["R"] for row in branch_rows],
            [row["r2"] for row in branch_rows],
            color=colors[branch],
            marker="o",
            markersize=3.5,
            linewidth=0.9,
            label=branch,
        )
    axes[0].set_ylabel("fitted β in gap(f) ∝ f⁻ᵝ")
    axes[0].set_title("Frequency exponent is R-dependent, not branch-constant")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].axhline(0.90, color="#64748b", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("log-log fit R²")
    axes[1].set_xlabel("physical table rows R (log scale)")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.23, which="both")
    figure.text(
        0.5,
        0.01,
        "62 formal single-branch runs · same 7-bin exact-frequency diagnostic · seed 42 · step 1000\n"
        "Identified gate: ≥5 positive bins, R²≥0.90, final global gap≥0.25; faded points are not interpreted.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main():
    rows = collect()
    write_table(rows)
    plot(rows)
    print(TABLE.relative_to(ROOT))
    print(FIGURE.relative_to(ROOT))


if __name__ == "__main__":
    main()
