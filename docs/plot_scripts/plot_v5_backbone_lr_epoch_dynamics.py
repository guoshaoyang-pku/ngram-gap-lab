#!/usr/bin/env python3
"""Backbone-LR replay dynamics with absolute table LR fixed at 0.0768.

The script discovers the 14 ``blrabs_*_fixed`` runs from their summaries,
pairs input/no-gram arms by backbone LR, and reads every plotted value from
``train_log.jsonl``.  It produces:

* raw input, no-gram, and paired net gaps against replay exposure;
* paired net gap against cumulative backbone optimizer dose;
* the pass-2 one-state recurrence fit requested in experiment-log §42; and
* a train-benefit / validation-penalty decomposition for the longest pair.

The recurrence plateau is an extrapolated fit parameter, not an observed
asymptote.  Epoch points are the last logged checkpoint in each epoch; the
blrabs cadence is 50 steps and one replay pass is 337 steps.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from v5_style import apply_style, save


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = ROOT / "data" / "runs_fixed"
DEFAULT_OUT_DIR = ROOT / "docs" / "figs" / "theory"
EXPECTED_LRS = (0.00006, 0.0001, 0.0003, 0.0006, 0.001, 0.002, 0.004)
ABS_TABLE_LR = 0.0768
EPOCH_STEPS = 337


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def close(a, b, atol=1e-10):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=atol)


def normalize_generated_text(path):
    """Keep generated SVG/CSV friendly to git diff --check."""
    text = path.read_text()
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")


def cumulative_lr_multiplier(step, warmup_steps=100, start=0.25):
    """Exact sum of the step-anchored warmup_constant multipliers."""
    warm_n = min(int(step), int(warmup_steps))
    if warmup_steps <= 1:
        warm_sum = float(warm_n)
    else:
        warm_sum = (
            warm_n * start
            + (1.0 - start)
            * warm_n
            * (warm_n - 1)
            / (2.0 * (warmup_steps - 1))
        )
    return warm_sum + max(0, int(step) - int(warmup_steps))


def discover_runs(runs_dir):
    runs = {}
    for directory in sorted(runs_dir.glob("blrabs_*_fixed")):
        summary_path = directory / "summary.json"
        log_path = directory / "train_log.jsonl"
        if not summary_path.is_file() or not log_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        cfg = summary["config"]
        lr = float(cfg["nanogpt_adam_lr"])
        is_input = bool(cfg["enable_bigram_ve"] and cfg["enable_trigram_ve"])
        arm = "input" if is_input else "nogram"
        if (lr, arm) in runs:
            raise RuntimeError(f"duplicate pair member for lr={lr:g}, arm={arm}")
        if not close(lr * float(cfg["table_lr_scale"]), ABS_TABLE_LR):
            raise RuntimeError(f"absolute table LR is not locked in {directory.name}")
        if summary.get("steps") != cfg.get("max_steps"):
            raise RuntimeError(f"incomplete summary in {directory.name}")
        rows = read_jsonl(log_path)
        if not rows or rows[-1]["step"] != cfg["max_steps"]:
            raise RuntimeError(f"incomplete train log in {directory.name}")
        runs[(lr, arm)] = {
            "run_id": summary["run_id"],
            "summary": summary,
            "rows": rows,
        }

    observed_lrs = tuple(sorted({key[0] for key in runs}))
    if observed_lrs != EXPECTED_LRS:
        raise RuntimeError(
            f"expected LR grid {EXPECTED_LRS}, found {observed_lrs} in {runs_dir}"
        )
    missing = [
        (lr, arm)
        for lr in EXPECTED_LRS
        for arm in ("input", "nogram")
        if (lr, arm) not in runs
    ]
    if missing:
        raise RuntimeError(f"missing paired runs: {missing}")
    return runs


def pair_rows(runs):
    paired = {}
    for lr in EXPECTED_LRS:
        input_rows = {row["step"]: row for row in runs[(lr, "input")]["rows"]}
        nogram_rows = {row["step"]: row for row in runs[(lr, "nogram")]["rows"]}
        common_steps = sorted(set(input_rows) & set(nogram_rows))
        if common_steps != sorted(input_rows) or common_steps != sorted(nogram_rows):
            raise RuntimeError(f"input/no-gram cadence mismatch at lr={lr:g}")
        rows = []
        for step in common_steps:
            inp = input_rows[step]
            no = nogram_rows[step]
            input_gap = float(inp["gap"])
            nogram_gap = float(no["gap"])
            train_benefit = float(no["train_loss"] - inp["train_loss"])
            val_penalty = float(inp["val_loss"] - no["val_loss"])
            net_gap = input_gap - nogram_gap
            if not close(net_gap, train_benefit + val_penalty, atol=2e-6):
                raise RuntimeError(f"paired decomposition mismatch at lr={lr:g}, step={step}")
            rows.append(
                {
                    "lr": lr,
                    "step": int(step),
                    "pass": float(step) / EPOCH_STEPS,
                    "epoch": int(inp["epoch"]),
                    "dose": lr * cumulative_lr_multiplier(step),
                    "input_gap": input_gap,
                    "nogram_gap": nogram_gap,
                    "net_gap": net_gap,
                    "train_benefit": train_benefit,
                    "val_penalty": val_penalty,
                }
            )
        paired[lr] = rows
    return paired


def epoch_endpoints(rows):
    by_epoch = {}
    for row in rows:
        by_epoch[row["epoch"]] = row
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def recurrence_fit(endpoint_rows):
    rows = [row for row in endpoint_rows if row["epoch"] >= 2]
    epochs = np.asarray([row["epoch"] for row in rows], dtype=float)
    values = np.asarray([row["net_gap"] for row in rows], dtype=float)
    if len(rows) < 4:
        raise RuntimeError("need at least four epoch endpoints for recurrence fit")
    g2 = float(values[0])

    def solve(q):
        carry = q ** (epochs - 2.0)
        basis = 1.0 - carry
        denominator = float(np.dot(basis, basis))
        if denominator <= 0:
            return float("inf"), float("nan"), np.full_like(values, np.nan)
        g_star = float(np.dot(basis, values - g2 * carry) / denominator)
        prediction = g2 * carry + g_star * basis
        sse = float(np.sum((values - prediction) ** 2))
        return sse, g_star, prediction

    left, right = 1e-5, 0.99999
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    x1 = right - (right - left) / phi
    x2 = left + (right - left) / phi
    f1, f2 = solve(x1)[0], solve(x2)[0]
    for _ in range(120):
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - (right - left) / phi
            f1 = solve(x1)[0]
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + (right - left) / phi
            f2 = solve(x2)[0]
    q = (left + right) / 2.0
    sse, g_star, prediction = solve(q)
    total = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - sse / total if total > 0 else float("nan")
    tau = -1.0 / math.log(q)
    return {
        "q": q,
        "tau_epochs": tau,
        "g_star": g_star,
        "r_squared": r_squared,
        "epochs": epochs,
        "values": values,
        "prediction": prediction,
    }


def interpolate_at_dose(rows, target):
    x = np.asarray([row["dose"] for row in rows])
    y = np.asarray([row["net_gap"] for row in rows])
    p = np.asarray([row["pass"] for row in rows])
    if target < x[0] or target > x[-1]:
        raise RuntimeError(f"dose {target:g} outside observed range [{x[0]}, {x[-1]}]")
    return float(np.interp(target, x, y)), float(np.interp(target, x, p))


def write_csvs(paired, fits, out_dir):
    points_path = out_dir / "theory_backbone_lr_epoch_points.csv"
    fits_path = out_dir / "theory_backbone_lr_epoch_fits.csv"
    dose_path = out_dir / "theory_backbone_lr_matched_dose.csv"

    with points_path.open("w", newline="") as handle:
        fields = list(next(iter(paired.values()))[0])
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for lr in EXPECTED_LRS:
            writer.writerows(paired[lr])

    with fits_path.open("w", newline="") as handle:
        fields = [
            "backbone_lr",
            "n_epoch_points",
            "q",
            "tau_epochs",
            "minus_log_q_over_lr",
            "g_star_extrapolated_full",
            "g_star_extrapolated_half_window",
            "g_star_window_shift_fraction",
            "r_squared_in_window",
            "observed_final_epoch",
            "observed_final_net_gap",
            "observed_last_increment",
            "fit_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for lr in EXPECTED_LRS:
            endpoint = epoch_endpoints(paired[lr])
            full = fits[lr]
            usable = [row for row in endpoint if row["epoch"] >= 2]
            half_n = max(4, len(usable) // 2)
            half = recurrence_fit(usable[:half_n])
            g_shift = abs(full["g_star"] - half["g_star"]) / max(
                abs(full["g_star"]), 1e-12
            )
            status = "descriptive_only_no_observed_plateau"
            if full["r_squared"] < 0.95:
                status = "poor_fit_optimizer_failure"
            writer.writerow(
                {
                    "backbone_lr": f"{lr:.8g}",
                    "n_epoch_points": len(usable),
                    "q": f"{full['q']:.9f}",
                    "tau_epochs": f"{full['tau_epochs']:.6f}",
                    "minus_log_q_over_lr": f"{-math.log(full['q']) / lr:.6f}",
                    "g_star_extrapolated_full": f"{full['g_star']:.6f}",
                    "g_star_extrapolated_half_window": f"{half['g_star']:.6f}",
                    "g_star_window_shift_fraction": f"{g_shift:.6f}",
                    "r_squared_in_window": f"{full['r_squared']:.9f}",
                    "observed_final_epoch": usable[-1]["epoch"],
                    "observed_final_net_gap": f"{usable[-1]['net_gap']:.6f}",
                    "observed_last_increment": f"{usable[-1]['net_gap'] - usable[-2]['net_gap']:.6f}",
                    "fit_status": status,
                }
            )

    matched_lrs = (0.0003, 0.0006, 0.001, 0.002)
    target_dose = min(paired[lr][-1]["dose"] for lr in matched_lrs)
    matched = []
    for lr in matched_lrs:
        gap, pass_count = interpolate_at_dose(paired[lr], target_dose)
        matched.append((lr, gap, pass_count))
    gap_values = np.asarray([item[1] for item in matched])
    cv = float(np.std(gap_values, ddof=0) / np.mean(gap_values))
    with dose_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "backbone_lr",
                "matched_cumulative_dose",
                "interpolated_passes",
                "interpolated_net_gap",
                "across_lr_gap_cv",
            ]
        )
        for lr, gap, pass_count in matched:
            writer.writerow(
                [
                    f"{lr:.8g}",
                    f"{target_dose:.8f}",
                    f"{pass_count:.6f}",
                    f"{gap:.6f}",
                    f"{cv:.6f}",
                ]
            )
    return points_path, fits_path, dose_path, target_dose, matched, cv


def plot_dynamics(paired, out_dir, target_dose, matched, cv):
    apply_style()
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(EXPECTED_LRS)))
    color_by_lr = dict(zip(EXPECTED_LRS, colors))
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2))
    ax_input, ax_no, ax_net, ax_dose, ax_match, ax_decomp = axes.flat

    for lr in EXPECTED_LRS:
        rows = paired[lr]
        pass_count = [row["pass"] for row in rows]
        dose = [row["dose"] for row in rows]
        color = color_by_lr[lr]
        linestyle = "--" if lr == 0.004 else "-"
        label = f"{lr:g}"
        ax_input.plot(pass_count, [row["input_gap"] for row in rows], color=color, lw=1.25, ls=linestyle, label=label)
        ax_no.plot(pass_count, [row["nogram_gap"] for row in rows], color=color, lw=1.25, ls=linestyle, label=label)
        ax_net.plot(pass_count, [row["net_gap"] for row in rows], color=color, lw=1.35, ls=linestyle, label=label)
        ax_dose.plot(dose, [row["net_gap"] for row in rows], color=color, lw=1.35, ls=linestyle, label=label)

    for axis, title, ylabel in (
        (ax_input, "Input arm", "input gap"),
        (ax_no, "Matched no-gram arm", "no-gram gap"),
        (ax_net, "Paired difference", "net n-gram gap"),
    ):
        axis.axhline(0.0, color="#9ca3af", lw=0.7, ls=":")
        axis.set_xlabel("replay exposure (step / 337)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    ax_input.legend(title="backbone LR", ncol=2, fontsize=7.5, title_fontsize=8)

    ax_dose.axhline(0.0, color="#9ca3af", lw=0.7, ls=":")
    ax_dose.axvline(target_dose, color="#c4493d", lw=0.9, ls=":")
    ax_dose.set_xlabel(r"cumulative backbone dose $D=\sum_t \eta_B(t)$")
    ax_dose.set_ylabel("net n-gram gap")
    ax_dose.set_title("Dose does not collapse the curves")

    passes = [item[2] for item in matched]
    gaps = [item[1] for item in matched]
    ax_match.plot(passes, gaps, color="#c4493d", marker="o", lw=1.2)
    for lr, gap, pass_count in matched:
        ax_match.annotate(f"LR={lr:g}", (pass_count, gap), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax_match.set_xlabel(f"passes at matched dose D={target_dose:.3f}")
    ax_match.set_ylabel("interpolated net gap")
    ax_match.set_title(f"Same dose: more replays, larger gap (CV={cv:.2f})")

    long_lr = 0.0003
    endpoints = epoch_endpoints(paired[long_lr])
    epochs = [row["epoch"] for row in endpoints]
    ax_decomp.plot(epochs, [row["train_benefit"] for row in endpoints], color="#2d6f9f", marker="o", ms=2.8, lw=1.0, label="train benefit")
    ax_decomp.plot(epochs, [row["val_penalty"] for row in endpoints], color="#c4493d", marker="o", ms=2.8, lw=1.0, label="validation penalty")
    ax_decomp.plot(epochs, [row["net_gap"] for row in endpoints], color="#1a7f37", lw=1.25, label="sum = net gap")
    ax_decomp.axhline(0.0, color="#9ca3af", lw=0.7, ls=":")
    ax_decomp.set_xlabel("logged epoch number")
    ax_decomp.set_ylabel("paired loss difference")
    ax_decomp.set_title("Long arm: late growth is validation damage")
    ax_decomp.legend()

    fig.suptitle(
        "Backbone-LR sweep with absolute table LR fixed at 0.0768\n"
        "14 paired fixed-replay runs · seed 42 · 50-step diagnostics",
        fontsize=11.5,
    )
    fig.tight_layout()
    paths = save(fig, out_dir, "fig_v5_backbone_lr_epoch_dynamics")
    normalize_generated_text(paths[1])
    return paths


def plot_fit_diagnostics(paired, fits, out_dir):
    apply_style()
    stable_lrs = EXPECTED_LRS[:-1]
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(stable_lrs)))
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.35))
    ax_fit, ax_rate, ax_star = axes

    for lr, color in zip(stable_lrs, colors):
        endpoint = [row for row in epoch_endpoints(paired[lr]) if row["epoch"] >= 2]
        fit = fits[lr]
        epochs = np.asarray([row["epoch"] for row in endpoint])
        values = np.asarray([row["net_gap"] for row in endpoint])
        ax_fit.plot(epochs, values, color=color, marker="o", ms=2.8, lw=0.9, label=f"{lr:g}")
        ax_fit.plot(fit["epochs"], fit["prediction"], color=color, ls="--", lw=0.9)
    ax_fit.set_xlabel("logged epoch number")
    ax_fit.set_ylabel("net n-gram gap")
    ax_fit.set_title("One-state fits are descriptive in-window")
    ax_fit.legend(title="backbone LR", ncol=2, fontsize=7.2, title_fontsize=8)

    lrs = np.asarray(stable_lrs)
    rates = np.asarray([-math.log(fits[lr]["q"]) / lr for lr in stable_lrs])
    ax_rate.plot(lrs, rates, color="#c4493d", marker="o", lw=1.1)
    ax_rate.set_xscale("log")
    ax_rate.set_yscale("log")
    ax_rate.set_xlabel("backbone LR")
    ax_rate.set_ylabel(r"$-\log(q)/\eta_B$")
    ax_rate.set_title("Not constant: LR-only time scaling fails")

    full = np.asarray([fits[lr]["g_star"] for lr in stable_lrs])
    half = []
    observed = []
    for lr in stable_lrs:
        endpoint = [row for row in epoch_endpoints(paired[lr]) if row["epoch"] >= 2]
        half_n = max(4, len(endpoint) // 2)
        half.append(recurrence_fit(endpoint[:half_n])["g_star"])
        observed.append(endpoint[-1]["net_gap"])
    ax_star.plot(lrs, full, color="#2d6f9f", marker="o", lw=1.0, label="G* full-window fit")
    ax_star.plot(lrs, half, color="#c58a0b", marker="o", lw=1.0, label="G* half-window fit")
    ax_star.plot(lrs, observed, color="#1a7f37", marker="o", lw=1.0, label="last observed gap")
    ax_star.set_xscale("log")
    ax_star.set_xlabel("backbone LR")
    ax_star.set_ylabel("gap")
    ax_star.set_title("Fitted balance is window-dependent")
    ax_star.legend(fontsize=7.5)

    fig.suptitle(
        "Pass-2 recurrence audit · fitted G* is extrapolation, not an observed plateau",
        fontsize=11.2,
    )
    fig.tight_layout()
    paths = save(fig, out_dir, "fig_v5_backbone_lr_recurrence_diagnostics")
    normalize_generated_text(paths[1])
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if not args.runs_dir.is_dir():
        raise RuntimeError(
            f"runs directory is unavailable: {args.runs_dir}; pass --runs-dir explicitly"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.runs_dir)
    paired = pair_rows(runs)
    fits = {lr: recurrence_fit(epoch_endpoints(paired[lr])) for lr in EXPECTED_LRS}
    points, fit_csv, dose_csv, target_dose, matched, cv = write_csvs(
        paired, fits, args.out_dir
    )
    dynamics = plot_dynamics(paired, args.out_dir, target_dose, matched, cv)
    diagnostics = plot_fit_diagnostics(paired, fits, args.out_dir)

    for path in (*dynamics, *diagnostics, points, fit_csv, dose_csv):
        print(path.relative_to(ROOT))
    print(f"matched_dose={target_dose:.8f} across_lr_gap_cv={cv:.6f}")


if __name__ == "__main__":
    main()
