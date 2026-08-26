from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figs" / "theory"
S1_FIGS = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2f63a6"
RED = "#bb4b4b"
GREEN = "#3d8b68"
GOLD = "#b07d24"
INK = "#26364d"
MUTED = "#6b7280"


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)


def read_csv(name: str) -> list[dict[str, str]]:
    with (S1_FIGS / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def fit_loglog(rows: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    values = np.asarray(rows, dtype=float)
    log_x = np.log(values[:, 0])
    log_y = np.log(values[:, 1])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    residual = log_y - (slope * log_x + intercept)
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((log_y - log_y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst else float("nan")
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return float(slope), float(r2), rmse, float(np.exp(intercept))


def parse_epoch_rows() -> list[dict[str, object]]:
    parsed = []
    for row in read_csv("epoch_final_gap.csv"):
        run = text(row, "run")
        match = re.match(r"ep_(L[1-4])_(.+)_(fs|fe)(?:_s(43|44))?_fixed$", run)
        gap = numeric(row, "final_gap")
        if not match or gap is None:
            continue
        parsed.append(
            {
                "run": run,
                "L": match.group(1),
                "module": match.group(2),
                "align": match.group(3),
                "seed": match.group(4) or "42",
                "gap": gap,
            }
        )
    return parsed


def write_table_slope_audit() -> list[dict[str, object]]:
    rows = read_csv("table_summary.csv")
    windows = (
        ("mult_1_16", lambda mult: mult <= 16),
        ("mult_1_32", lambda mult: mult <= 32),
        ("mult_8_64", lambda mult: 8 <= mult <= 64),
        ("mult_12_48", lambda mult: 12 <= mult <= 48),
        ("mult_44_256", lambda mult: 44 <= mult <= 256),
        ("all", lambda mult: True),
    )
    audit = []
    for module in ("bigram", "trigram", "both"):
        for window_name, predicate in windows:
            for seed in ("42", "43", "44"):
                selected = sorted(
                    (
                        (float(row["logical_2R"]), float(row["final_gap"]))
                        for row in rows
                        if row["module"] == module
                        and row["seed"] == seed
                        and float(row["final_gap"]) > 0
                        and predicate(int(row["mult"]))
                    )
                )
                if len(selected) < 3:
                    continue
                slope, r2, rmse, amplitude = fit_loglog(selected)
                audit.append(
                    {
                        "module": module,
                        "window": window_name,
                        "seed": seed,
                        "n": len(selected),
                        "slope": slope,
                        "r2": r2,
                        "rmse_log": rmse,
                        "amplitude": amplitude,
                    }
                )
    out = OUT / "theory_table_slope_audit.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    return audit


def plot_table_slopes() -> None:
    audit = write_table_slope_audit()
    selected = [
        row
        for row in audit
        if row["window"] in {"mult_1_16", "mult_44_256"}
        and row["module"] in {"bigram", "trigram", "both"}
    ]
    labels = sorted(
        {f"{row['module']}/{row['window']}" for row in selected},
        key=lambda label: (label.split("/")[0], label.split("/")[1]),
    )
    fig, ax = plt.subplots(figsize=(10.2, 4.7))
    for index, label in enumerate(labels):
        module, window = label.split("/")
        values = [row for row in selected if row["module"] == module and row["window"] == window]
        by_seed = {row["seed"]: row for row in values}
        seeds = sorted(by_seed)
        x_values = np.linspace(index - 0.18, index + 0.18, len(seeds))
        y_values = [by_seed[seed]["slope"] for seed in seeds]
        ax.scatter(x_values, y_values, color=BLUE, s=38, zorder=3)
        if y_values:
            ax.errorbar(
                index,
                float(np.mean(y_values)),
                yerr=float(np.std(y_values)),
                color=RED,
                fmt="o",
                capsize=4,
                lw=1.4,
            )
    ax.axhline(0, color="#aeb7c2", lw=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        [label.replace("mult_", "m=").replace("_", "–") for label in labels],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("log–log slope of final gap vs logical table addresses")
    ax.set_title("S1 table axis: local slopes are window- and module-dependent")
    ax.text(
        0.01,
        0.02,
        "points = seeds 42/43/44; whisker = seed SD; guide, not a universal law",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
    )
    save(fig, "fig_theory_table_slopes")


def plot_epoch_audit() -> None:
    rows = parse_epoch_rows()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
    colors = {"bigram": BLUE, "trigram": RED, "both": GOLD, "nogram": GREEN}
    x_values = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
    for align, ax in zip(("fs", "fe"), axes):
        for module in ("bigram", "trigram", "both", "nogram"):
            for seed in ("42", "43", "44"):
                selected = [
                    row
                    for row in rows
                    if row["align"] == align
                    and row["module"] == module
                    and row["seed"] == seed
                ]
                selected.sort(key=lambda row: x_values[row["L"]])
                if not selected:
                    continue
                ax.plot(
                    [x_values[row["L"]] for row in selected],
                    [row["gap"] for row in selected],
                    marker="o",
                    ms=3.5,
                    lw=1.0,
                    alpha=0.62,
                    color=colors[module],
                    label=f"{module} s{seed}",
                )
        ax.set_xticks([1, 2, 3, 4], ["L1", "L2", "L3", "L4"])
        ax.set_xlabel("nested-prefix epoch length")
        ax.set_ylabel("online final gap")
        ax.set_title("fixed-step, 1000 steps" if align == "fs" else "fixed-epoch, replay aligned")
        ax.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, fontsize=7, ncol=2)
    fig.suptitle(
        "S1 epoch axis: seed-resolved trajectories, not a single monotone scaling curve",
        color=INK,
    )
    save(fig, "fig_theory_epoch_multiseed")


def final_m2_frequency_rows(arm: str = "input") -> list[dict[str, object]]:
    path = ROOT / "data" / "runs_fixed" / f"nglab1x_{arm}_v2_fixed" / "freq_bin_loss.jsonl"
    final = None
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("step") == 2000:
                final = row
    if final is None:
        raise FileNotFoundError(f"no step=2000 record in {path}")
    output = []
    for branch in ("bigram", "trigram"):
        train = final["train"][branch]
        val = final["val"][branch]
        for bucket, train_row in train.items():
            if bucket == "novel" or bucket not in val:
                continue
            val_row = val[bucket]
            train_count = int(train_row.get("token_count", 0))
            val_count = int(val_row.get("token_count", 0))
            if train_count == 0 or val_count == 0:
                continue
            output.append(
                {
                    "arm": arm,
                    "branch": branch,
                    "bucket": bucket,
                    "train_fraction": float(train_row["frac"]),
                    "val_fraction": float(val_row["frac"]),
                    "train_loss": float(train_row["mean_loss"]),
                    "val_loss": float(val_row["mean_loss"]),
                    "bucket_gap": float(val_row["mean_loss"] - train_row["mean_loss"]),
                    "train_contribution": float(train_row["total_contrib"]),
                    "val_contribution": float(val_row["total_contrib"]),
                    "gap_contribution": float(
                        val_row["total_contrib"] - train_row["total_contrib"]
                    ),
                    "train_tokens": train_count,
                    "val_tokens": val_count,
                }
            )
    return output


def plot_m2_frequency_decomposition() -> None:
    rows = final_m2_frequency_rows()
    out = OUT / "theory_m2_frequency_decomposition.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bucket_order = [
        "1", "2", "3", "4", "5", "6-8", "9-12", "13-20", "21-30",
        "31-50", "51-75", "76-100", "101-150", "151-200", "201-300",
        "301-500", "501-750", "751-1k", "1k-2k", "2k-5k", "5k-10k", "10k+",
    ]
    branches = ("bigram", "trigram")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True)
    for ax, branch in zip(axes, branches):
        selected = [row for row in rows if row["branch"] == branch]
        selected.sort(key=lambda row: bucket_order.index(row["bucket"]))
        labels = [row["bucket"] for row in selected]
        x = np.arange(len(selected))
        ax.bar(
            x - 0.18,
            [row["train_fraction"] for row in selected],
            width=0.36,
            color=BLUE,
            alpha=0.55,
            label="train token fraction",
        )
        ax.bar(
            x + 0.18,
            [row["val_fraction"] for row in selected],
            width=0.36,
            color=RED,
            alpha=0.55,
            label="val token fraction",
        )
        ax2 = ax.twinx()
        ax2.plot(
            x,
            [row["bucket_gap"] for row in selected],
            "o-",
            color=INK,
            lw=1.6,
            label="bucket mean-loss gap",
        )
        ax2.plot(
            x,
            [row["gap_contribution"] for row in selected],
            "s--",
            color=GOLD,
            lw=1.3,
            label="weighted gap contribution",
        )
        ax.axhline(0, color="#aeb7c2", lw=0.8)
        ax2.axhline(0, color="#aeb7c2", lw=0.8)
        ax.set_ylabel("token fraction")
        ax2.set_ylabel("loss gap / weighted contribution")
        ax.set_title(f"M2-v2 input · {branch} · final frequency evaluation @ step 2000")
        ax.grid(axis="y", alpha=0.2)
        handles, labels_ = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels_ + labels2, frameon=False, fontsize=8, loc="upper left")
    axes[-1].set_xticks(np.arange(len(bucket_order)), bucket_order, rotation=42, ha="right")
    fig.suptitle(
        "M2 frequency bins: per-bin gap is not the same as contribution to global gap",
        color=INK,
    )
    save(fig, "fig_theory_m2_frequency_decomposition")


def bucket_midpoint(bucket: str) -> float | None:
    """Geometric midpoint of a hit-count bucket label; None when unbounded."""

    def one(token: str) -> float:
        token = token.strip()
        if token[-1] in "kK":
            return float(token[:-1]) * 1000.0
        return float(token)

    if bucket in ("novel", "") or bucket.endswith("+"):
        return None
    if "-" in bucket:
        low, high = bucket.split("-")
        return math.sqrt(one(low) * one(high))
    return one(bucket)


def freq_record_at(run: str, step: int) -> dict:
    """Freq-bin record at one logged step of a fixed run."""
    path = ROOT / "data" / "runs_fixed" / run / "freq_bin_loss.jsonl"
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("step") == step:
                return row
    raise FileNotFoundError(f"no freq-bin record at step={step} in {path}")


def paired_decomposition_rows(
    arm: str = "nglab1x_input_v2_fixed",
    control: str = "nglab1x_nogram_v2_fixed",
    step: int = 2000,
) -> list[dict[str, object]]:
    """Difference-in-differences split of the bucket gap against the nogram arm.

    M(f) = L_train^control - L_train^arm  (memorisation gain, train side)
    X(f) = L_val^arm      - L_val^control (generalisation cost, val side)
    G_arm(f) - G_control(f) = M(f) + X(f) exactly; the shared backbone cancels.
    """
    rec_a, rec_c = freq_record_at(arm, step), freq_record_at(control, step)
    rows = []
    for branch in ("bigram", "trigram"):
        ta, va = rec_a["train"][branch], rec_a["val"][branch]
        tc, vc = rec_c["train"][branch], rec_c["val"][branch]
        for bucket in ta:
            f = bucket_midpoint(bucket)
            if f is None or bucket not in va or bucket not in tc or bucket not in vc:
                continue
            cells = (ta[bucket], va[bucket], tc[bucket], vc[bucket])
            if min(c["token_count"] for c in cells) < 1:
                continue
            gain = tc[bucket]["mean_loss"] - ta[bucket]["mean_loss"]
            cost = va[bucket]["mean_loss"] - vc[bucket]["mean_loss"]
            rows.append({
                "branch": branch,
                "bucket": bucket,
                "f_mid": f,
                "train_arm": ta[bucket]["mean_loss"],
                "train_control": tc[bucket]["mean_loss"],
                "val_arm": va[bucket]["mean_loss"],
                "val_control": vc[bucket]["mean_loss"],
                "memorisation_gain": gain,
                "generalisation_cost": cost,
                "gap_arm": va[bucket]["mean_loss"] - ta[bucket]["mean_loss"],
                "gap_control": vc[bucket]["mean_loss"] - tc[bucket]["mean_loss"],
                "net_table_effect": gain + cost,
            })
    rows.sort(key=lambda item: (item["branch"], item["f_mid"]))
    return rows


def plot_paired_decomposition() -> None:
    rows = paired_decomposition_rows()
    with (OUT / "theory_paired_decomposition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    colors = {"bigram": BLUE, "trigram": RED}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    ax = axes[0]
    for branch, color in colors.items():
        sel = [r for r in rows if r["branch"] == branch]
        f = [r["f_mid"] for r in sel]
        ax.plot(f, [r["gap_arm"] for r in sel], "o-", color=color, lw=1.5, ms=4,
                label=f"{branch} · input arm")
        ax.plot(f, [r["gap_control"] for r in sel], "s--", color=color, lw=1.1, ms=3.5,
                alpha=0.5, label=f"{branch} · nogram control")
    ax.set_xscale("log")
    ax.set_xlabel("train hit count f")
    ax.set_ylabel("bucket gap (nats)")
    ax.set_title("Control arm: no frequency structure at all")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    for branch, color in colors.items():
        sel = [r for r in rows if r["branch"] == branch]
        f = [r["f_mid"] for r in sel]
        ax.plot(f, [r["memorisation_gain"] for r in sel], "o-", color=color, lw=1.5, ms=4,
                label=f"{branch} · M = train-side gain")
        ax.plot(f, [r["generalisation_cost"] for r in sel], "^--", color=color, lw=1.3, ms=4,
                alpha=0.7, label=f"{branch} · X = val-side cost")
    ax.axhline(0.0, color=MUTED, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("train hit count f")
    ax.set_ylabel("nats")
    ax.set_title("What the table buys (M) vs what it costs (X)")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    for branch, color in colors.items():
        sel = [r for r in rows if r["branch"] == branch]
        f = np.array([r["f_mid"] for r in sel])
        net = np.array([r["net_table_effect"] for r in sel])
        keep = (f <= 900) & (net > 0)
        slope, r2, _, amp = fit_loglog(list(zip(f[keep], net[keep])))
        ax.plot(f, net, "o", color=color, ms=5,
                label=f"{branch} · M+X   α={slope:.3f} (R²={r2:.3f}, f≤900)")
        grid = np.logspace(0, math.log10(900), 40)
        ax.plot(grid, amp * grid ** slope, "-", color=color, lw=1.3, alpha=0.85)
    grid = np.logspace(0, math.log10(900), 40)
    ax.plot(grid, 11.4 * grid ** -1.0, ":", color=MUTED, lw=1.5,
            label="slope −1 · linear response (variance law)")
    ax.plot(grid, 11.4 * grid ** -0.5, "-.", color=GOLD, lw=1.5,
            label="slope −1/2 · sign / saturating response")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("train hit count f")
    ax.set_ylabel("net table effect M+X (nats)")
    ax.set_title("Net table effect vs response references")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    fig.suptitle(
        "Paired difference-in-differences · input vs nogram · v2 wave · step 2000 · seed 42 "
        "· historical bf16+compile",
        color=INK,
    )
    save(fig, "fig_theory_paired_decomposition")


OPT_ARMS = [
    ("SGD m=0.9, scale 1.0", "nglab1x_opt_sgd_09_fixed", GREEN),
    ("AdamW (0.8,0.95), scale 1.0", "nglab1x_opt_adamw_080950_fixed", GOLD),
    ("AdamW (0.9,0.999), scale 1.0", "nglab1x_opt_adamw_090999_fixed", BLUE),
    ("RMSProp (0,0.99), scale 2.0", "nglab1x_input_v2_fixed", RED),
]


def optimizer_gap_points(run: str, branch: str, step: int = 1000,
                         f_max: float = 900.0) -> list[tuple[float, float]]:
    rec = freq_record_at(run, step)
    train, val = rec["train"][branch], rec["val"][branch]
    pts = []
    for bucket in train:
        f = bucket_midpoint(bucket)
        if f is None or f > f_max or bucket not in val:
            continue
        if train[bucket]["token_count"] < 1 or val[bucket]["token_count"] < 1:
            continue
        gap = val[bucket]["mean_loss"] - train[bucket]["mean_loss"]
        if gap > 0:
            pts.append((f, gap))
    pts.sort()
    return pts


def optimizer_exponent_rows() -> list[dict[str, object]]:
    rows = []
    for label, run, _ in OPT_ARMS:
        for branch in ("bigram", "trigram"):
            pts = optimizer_gap_points(run, branch)
            if len(pts) < 5:
                rows.append({"optimizer": label, "run_id": run, "branch": branch,
                             "n_bins": len(pts), "slope": None, "r2": None,
                             "mean_gap": None, "implied_a": None})
                continue
            slope, r2, _, _ = fit_loglog(pts)
            rows.append({
                "optimizer": label,
                "run_id": run,
                "branch": branch,
                "n_bins": len(pts),
                "slope": slope,
                "r2": r2,
                "mean_gap": float(np.mean([p[1] for p in pts])),
                "implied_a": -2.0 * slope - 1.0,
            })
    return rows


def plot_optimizer_exponent() -> None:
    rows = optimizer_exponent_rows()
    with (OUT / "theory_optimizer_exponent.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9))

    ax = axes[0]
    for label, run, color in OPT_ARMS:
        pts = optimizer_gap_points(run, "bigram")
        row = next(r for r in rows if r["run_id"] == run and r["branch"] == "bigram")
        if row["r2"] is None or row["r2"] < 0.7:
            tag = f"no power law (R²={row['r2']:.2f})" if row["r2"] else "no power law"
        else:
            tag = f"α={row['slope']:.3f} (R²={row['r2']:.2f})"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, lw=1.4, ms=4,
                label=f"{label} · {tag}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("train hit count f")
    ax.set_ylabel("bigram bucket gap (nats)")
    ax.set_title("Step 1000, seed 42: the table optimizer sets level and shape")
    ax.grid(alpha=0.2, which="both")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    ax = axes[1]
    valid = [r for r in rows if r["slope"] is not None and r["r2"] >= 0.7]
    x = np.arange(len(valid))
    ax.bar(x, [r["implied_a"] for r in valid], alpha=0.85,
           color=[next(c for _, run, c in OPT_ARMS if run == r["run_id"]) for r in valid])
    ax.axhline(1.0, color=MUTED, lw=1.3, ls="--")
    ax.text(0.02, 1.05, "a = 1 · linear response → slope −1 (variance law)",
            transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED)
    ax.axhline(0.0, color=GOLD, lw=1.3, ls="-.")
    ax.text(0.02, 0.06, "a = 0 · sign / saturating response → slope −1/2",
            transform=ax.get_yaxis_transform(), fontsize=8, color=GOLD)
    for i, r in enumerate(valid):
        ax.annotate(f"{r['implied_a']:+.2f}", (i, r["implied_a"] + 0.06), ha="center",
                    fontsize=8, color=INK)
    ax.set_xticks(x)
    def short(name: str) -> str:
        return (name.replace("AdamW ", "AdamW\n").replace("RMSProp ", "RMSProp\n")
                .replace(", scale ", "\ns"))
    ax.set_xticklabels([f"{short(r['optimizer'])}\n{r['branch']}" for r in valid], fontsize=7)
    ax.set_ylabel("implied response exponent  a = −2α − 1")
    ax.set_ylim(-1.15, 1.4)
    ax.set_title("No arm reaches the linear-response regime a = 1")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        "Table-optimizer ablation · step 1000 · seed 42 · f ≤ 900\n"
        "SGD arm has no resolvable power law · RMSProp arm also carries scale 2.0 (confounded)",
        color=INK,
    )
    save(fig, "fig_theory_optimizer_exponent")


def numeric(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def text(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return "unknown"


def collect_fit_records(value: object) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if isinstance(value, dict):
        if "beta" in value and ("branch" in value or "module" in value):
            records.append(value)
        for child in value.values():
            records.extend(collect_fit_records(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(collect_fit_records(child))
    return records


def plot_sampling_regimes() -> None:
    frequency = np.logspace(0, 5, 500)
    resolved = 7.0 / frequency
    probabilities = np.arange(1, 129, dtype=float) ** -1
    probabilities /= probabilities.sum()
    unseen_mass = np.sum(
        probabilities[:, None] * (1.0 - probabilities[:, None]) ** frequency[None, :],
        axis=0,
    )
    unresolved = unseen_mass * np.maximum(np.log(frequency / (1e-3 * 128.0)), 1e-3)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1))
    axes[0].loglog(frequency, resolved, color=BLUE, lw=2, label="resolved: (K−1)/f, K=8")
    axes[0].loglog(frequency, 1.0 / frequency, color=INK, ls="--", lw=1.2, label="−1 guide")
    axes[0].axvspan(1e1, 1e5, color=GREEN, alpha=0.08, label="resolved schematic region")
    axes[0].set_title("Finite support: the −1 law has a domain")
    axes[0].set_xlabel("exact context hits f")
    axes[0].set_ylabel("expected gap")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].loglog(frequency, unresolved, color=RED, lw=2, label="unseen-mass term")
    axes[1].loglog(frequency, resolved, color=INK, ls="--", lw=1.2, label="full-support (K−1)/f")
    axes[1].axvspan(1, 80, color=RED, alpha=0.08, label="unresolved tail")
    axes[1].set_title("Long tail: unseen mass changes the slope")
    axes[1].set_xlabel("exact context hits f")
    axes[1].set_ylabel("gap contribution")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Sampling-law regimes: L3 analytic/numerical boundary", color=INK)
    save(fig, "fig_theory_sampling_regimes")


def plot_bias_split() -> None:
    frequency = np.logspace(0.7, 4.5, 300)
    k = 8
    half_bias = (k - 1.0) / (2.0 * frequency)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.semilogx(frequency, half_bias, color=RED, lw=2, label="E[CE_val] − H = +(K−1)/(2f)")
    ax.semilogx(frequency, -half_bias, color=BLUE, lw=2, label="E[CE_train] − H = −(K−1)/(2f)")
    ax.semilogx(frequency, 2.0 * half_bias, color=INK, ls="--", lw=1.5, label="gap = (K−1)/f")
    ax.axhline(0, color="#aeb7c2", lw=0.8)
    ax.set_title("Train/validation second-order bias split, K=8")
    ax.set_xlabel("exact context hits f (log)")
    ax.set_ylabel("excess cross-entropy (nats)")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.03, 0.04, "analytic resolved-support result; H(P) cancels", transform=ax.transAxes, color=MUTED, fontsize=9)
    save(fig, "fig_theory_bias_split")


def plot_beta_seed() -> None:
    fit_path = S1_FIGS / "fit_manifest.json"
    manifest = json.loads(fit_path.read_text())
    records = collect_fit_records(manifest)
    if not records:
        rows = read_csv("frequency_snapshot_fit.csv")
        records = [
            {
                "branch": text(row, "branch"),
                "module": text(row, "module"),
                "seed": text(row, "seed"),
                "beta": numeric(row, "beta"),
            }
            for row in rows
            if numeric(row, "beta") is not None
        ]
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        beta = record.get("beta")
        if beta is None:
            continue
        try:
            beta_value = float(beta)
        except (TypeError, ValueError):
            continue
        key = (str(record.get("branch", "unknown")), str(record.get("module", "unknown")))
        grouped[key][str(record.get("seed", "unknown"))].append(beta_value)
    labels = sorted(grouped)
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    for index, key in enumerate(labels):
        seed_values = grouped[key]
        means = [(seed, float(np.mean(values))) for seed, values in sorted(seed_values.items())]
        x_values = np.linspace(index - 0.18, index + 0.18, max(1, len(means)))
        y_values = [value for _, value in means]
        ax.scatter(x_values, y_values, color=BLUE, s=36, zorder=3)
        if y_values:
            ax.errorbar(index, float(np.mean(y_values)), yerr=float(np.std(y_values)), color=RED, fmt="o", capsize=4, lw=1.4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{branch}/{module}" for branch, module in labels], rotation=18, ha="right")
    ax.set_ylabel("fitted β")
    ax.set_title("S1 frequency fits: seed points and mean ± dispersion")
    ax.text(0.01, 0.02, "historical bf16 + torch.compile; β only is comparatively seed-stable", transform=ax.transAxes, color=MUTED, fontsize=9)
    save(fig, "fig_theory_s1_beta_seed")


def plot_exposure_frequency() -> None:
    epoch_rows = parse_epoch_rows()
    frequency_rows = read_csv("frequency_snapshot_fit.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    colors = {"bigram": BLUE, "trigram": RED, "both": GOLD, "nogram": GREEN}
    x_values = {"L1": 1, "L2": 2, "L3": 3, "L4": 4}
    for align in ("fs", "fe"):
        for module in ("bigram", "trigram", "both", "nogram"):
            selected = [
                row
                for row in epoch_rows
                if row["align"] == align and row["module"] == module
            ]
            for seed in ("42", "43", "44"):
                points = sorted(
                    (x_values[row["L"]], row["gap"])
                    for row in selected
                    if row["seed"] == seed
                )
                if not points:
                    continue
                axes[0].plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    ms=3,
                    lw=1.0,
                    alpha=0.62,
                    color=colors[module],
                    label=f"{align}/{module} s{seed}",
                )
    axes[0].set_title("S1 epoch/exposure audit")
    axes[0].set_xlabel("nested-prefix epoch length L")
    axes[0].set_ylabel("final gap")
    axes[0].set_xticks([1, 2, 3, 4], ["L1", "L2", "L3", "L4"])
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
    snapshot_groups: dict[str, list[float]] = defaultdict(list)
    for row in frequency_rows:
        beta = numeric(row, "beta")
        if beta is not None:
            snapshot_groups[text(row, "snapshot", "epoch", "step")].append(beta)
    snapshot_items = sorted(
        snapshot_groups.items(),
        key=lambda item: int(re.search(r"\d+", item[0]).group()) if re.search(r"\d+", item[0]) else item[0],
    )
    axes[1].errorbar(
        range(len(snapshot_items)),
        [np.mean(values) for _, values in snapshot_items],
        yerr=[np.std(values) for _, values in snapshot_items],
        color=GOLD,
        marker="o",
        capsize=4,
        lw=1.5,
    )
    axes[1].set_xticks(range(len(snapshot_items)))
    axes[1].set_xticklabels([label for label, _ in snapshot_items], rotation=20, ha="right")
    axes[1].set_title("Frequency fit β across exposure snapshots")
    axes[1].set_xlabel("snapshot")
    axes[1].set_ylabel("β mean ± within-snapshot spread")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Exposure × frequency: observational audit, not causal identification", color=INK)
    save(fig, "fig_theory_exposure_frequency")


def main() -> None:
    for required in (
        S1_FIGS / "epoch_final_gap.csv",
        S1_FIGS / "frequency_snapshot_fit.csv",
        S1_FIGS / "fit_manifest.json",
        S1_FIGS / "table_summary.csv",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    plot_sampling_regimes()
    plot_bias_split()
    plot_beta_seed()
    plot_table_slopes()
    plot_epoch_audit()
    plot_m2_frequency_decomposition()
    plot_paired_decomposition()
    plot_optimizer_exponent()
    plot_exposure_frequency()
    print(f"wrote theory figures to {OUT}")


if __name__ == "__main__":
    main()