#!/usr/bin/env python3
"""128× standard figures for the restored main experiment document.

Generates the 128× versions of the figures that the registry script still
reads from 2× run ids:
  fig_v5_128x_injection_frequency_bigram.png   M2 4-arm per-bin gap (128×)
  fig_v5_128x_injection_frequency_trigram.png
  fig_v5_128x_causal_losses.png                6-arm train/val/gap curves (128×)
  fig_v5_128x_causal_frequency_effect.png      none vs mask_low vs mask_high (128×)

Data source: data/runs_fixed/*_128x_fixed / causalv5c_*_128x_fixed.
Points are raw records; thin lines are 3-point visual connectors.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS_FIXED = ROOT / "data" / "runs_fixed"
OUT = ROOT / "docs" / "figs" / "main"
OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()] if path.exists() else []


def read_run(run_id):
    run_dir = RUNS_FIXED / f"{run_id}_fixed"
    summary = json.loads((run_dir / "summary.json").read_text())
    return summary, read_jsonl(run_dir / "train_log.jsonl")


def final_record(run_id, name="freq_bin_loss.jsonl"):
    path = RUNS_FIXED / f"{run_id}_fixed" / name
    records = read_jsonl(path)
    return max(records, key=lambda row: int(row.get("step", -1))) if records else None


def smooth(values):
    if len(values) < 3:
        return np.asarray(values), np.arange(len(values))
    return np.convolve(values, np.ones(3) / 3, mode="valid"), np.arange(1, len(values) - 1)


def add_raw_and_smoothed(axis, rows, metric, color, label=None):
    steps = np.asarray([row["step"] for row in rows])
    values = np.asarray([row.get(metric, float("nan")) for row in rows])
    axis.scatter(steps, values, color=color, s=3.5, alpha=0.25, zorder=2)
    averaged, offsets = smooth(values)
    axis.plot(steps[offsets], averaged, color=color, linewidth=0.9,
              label=label, zorder=3)


def ordered_frequency_buckets(stats):
    def lower_bound(label):
        if label == "novel":
            return -1
        token = label.split("-")[0].rstrip("+")
        multiplier = 1
        if token.endswith("k"):
            token = token[:-1]
            multiplier = 1_000
        elif token.endswith("m"):
            token = token[:-1]
            multiplier = 1_000_000
        return int(float(token) * multiplier)
    return sorted(stats, key=lower_bound)


def frequency_gaps(record, branch):
    train = record["train"][branch]
    val = record["val"][branch]
    return {
        bucket: (
            float(val[bucket]["mean_loss"]) - float(train[bucket]["mean_loss"])
            if train[bucket]["token_count"] > 0
            else np.nan
        )
        for bucket in train
        if bucket != "novel" and bucket in val
    }


def add_finite_smoothed_segments(axis, x, values, color, label=None):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    starts = np.flatnonzero(valid & np.concatenate(([True], ~valid[:-1])))
    ends = np.flatnonzero(valid & np.concatenate((~valid[1:], [True]))) + 1
    for index, (start, end) in enumerate(zip(starts, ends)):
        averaged, offsets = smooth(values[start:end])
        axis.plot(
            x[start:end][offsets],
            averaged,
            color=color,
            linewidth=0.8,
            label=label if index == 0 else None,
        )


def save_figure(figure, filename):
    path = OUT / filename
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(path.relative_to(ROOT))


# ---------- S1 clean table-size fit ----------
def plot_s1_table_size_clean():
    path = ROOT / "docs" / "appendices" / "s1_scaling_three_axis" / "s1_table_size_points.csv"
    rows = list(csv.DictReader(path.open()))
    figure, axis = plt.subplots(figsize=(9.8, 6.4))
    # Fit convention (2026-08-29 revision): the raw R>=16000 fit mixed the
    # small-R collapse regime with the large-R saturation regime and biased the
    # exponents low. Each branch is now fitted only over its middle linear
    # window, on gap - floor so the table contribution is isolated. The raw-gap
    # fit over the same window is kept in the legend for sensitivity.
    styles = {
        "bigram": (
            "#14736f", "bigram-only", (2e3, 2e5), "2e3-2e5",
            ("bigram saturation: local slope\ndecays beyond R ≈ 3e5", 4.2e5, 0.30),
        ),
        "trigram": (
            "#c56c0b", "trigram-only", (1.0e5, 9.3e5), "1e5-9e5",
            ("trigram stays linear through R ≈ 1e6;\nsaturates beyond R ≈ 1.2e6", 1.15e6, 1.75),
        ),
    }
    # Small-R extension batch (2026-08-29): R <= 10000, drawn as open markers.
    small_r_cutoff = 10000
    # no-gram floor = backbone-only val-train gap at step 1000
    # (nogram run 0.0234; mean of the small-R table plateau 0.0196)
    no_gram_floor = 0.02
    for branch, (color, label, (fit_lo, fit_hi), window_note, saturation) in styles.items():
        branch_rows = sorted(
            [row for row in rows if row["branch"] == branch],
            key=lambda row: float(row["R"]),
        )
        x = np.asarray([float(row["R"]) for row in branch_rows])
        y = np.asarray([float(row["final_gap"]) for row in branch_rows])
        large = x > small_r_cutoff
        small = ~large
        axis.scatter(
            x[large], y[large], color=color, s=24, alpha=0.82,
            label=f"{label} raw (formal points)",
        )
        axis.scatter(
            x[small], y[small], facecolors="none", edgecolors=color, s=30,
            alpha=0.85, linewidths=1.1, label=f"{label} raw (small-R extension)",
        )
        averaged, offsets = smooth(y)
        axis.plot(x[offsets], averaged, color=color, linewidth=0.9)
        fit_mask = (x >= fit_lo) & (x <= fit_hi) & (y - no_gram_floor > 0)
        slope, intercept = np.polyfit(
            np.log(x[fit_mask]), np.log(y[fit_mask] - no_gram_floor), 1
        )
        raw_slope, _ = np.polyfit(np.log(x[fit_mask]), np.log(y[fit_mask]), 1)
        fit_x = np.geomspace(x[fit_mask].min(), x[fit_mask].max(), 160)
        axis.plot(
            fit_x,
            no_gram_floor + np.exp(intercept) * fit_x**slope,
            color=color,
            linestyle="--",
            linewidth=1.15,
            label=(
                f"{label}: (gap−{no_gram_floor:.2f}) ∝ R^{slope:.2f} "
                f"[R {window_note}; raw-gap fit {raw_slope:.2f}]"
            ),
        )
        saturation_text, sat_x, sat_y = saturation
        axis.text(sat_x, sat_y, saturation_text, fontsize=8.5, color="#444444")
    axis.axvspan(1, small_r_cutoff, color="#888888", alpha=0.10)
    axis.annotate(
        "small-R collapse regime:\ngap → no-gram floor (≈0.02 at step 1000)",
        xy=(60, 0.045),
        xytext=(250, 0.006),
        fontsize=8.5,
        color="#444444",
        arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 0.8},
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("physical rows R of the varied clean table (log scale)")
    axis.set_ylabel("final online gap at step 1000 (log scale)")
    axis.grid(alpha=0.22, which="both")
    axis.legend(fontsize=8.5, frameon=False, loc="upper left")
    figure.suptitle(
        "V5 S1 clean table-size scaling · single-table branches · clean-window power-law fits"
    )
    total_runs = len(rows)
    figure.text(
        0.5,
        0.015,
        f"Raw endpoint points from {total_runs} formal runs (thin line = 3-point connector); "
        f"dashed fits use gap−{no_gram_floor:.2f} over each branch's middle linear window "
        "(collapse and saturation regimes excluded; K_train = 3.54e6 bigram / 1.90e7 trigram "
        "distinct contexts); raw-gap fits over the same windows give 0.50 / 0.65.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(figure, "fig_v5_s1_table_size_loglog_clean.png")


# ---------- M2 2× vs 128× trajectories ----------
def plot_injection_comparison():
    arms = (
        ("input", "#2d6f9f"),
        ("y", "#c4493d"),
        ("v", "#b67524"),
        ("nogram", "#686d73"),
    )
    standards = (
        ("2× historical v5", "nglab1x_{arm}_v5", 2.0, "--"),
        ("128× current v5", "nglab1x_{arm}_v5_128x_freq10", 128.0, "-"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(13.0, 8.0), sharex="col")
    metrics = (
        ("train_loss", "online train loss"),
        ("val_loss", "fixed validation loss"),
        ("gap", "online gap = fixed val − online train"),
    )
    for column, (standard_label, pattern, _, linestyle) in enumerate(standards):
        for arm, color in arms:
            run_id = pattern.format(arm=arm)
            _, rows = read_run(run_id)
            if not rows or max(row["step"] for row in rows) != 2000:
                print(f"skip injection comparison: {run_id} incomplete")
                return
            label = f"{arm} · {standard_label}"
            steps = np.asarray([row["step"] for row in rows])
            for row_index, (metric, ylabel) in enumerate(metrics):
                values = np.asarray([
                    row.get(metric, row["val_loss"] - row["train_loss"])
                    for row in rows
                ])
                axes[row_index, column].scatter(
                    steps, values, color=color, s=2.3, alpha=0.18
                )
                averaged, offsets = smooth(values)
                axes[row_index, column].plot(
                    steps[offsets],
                    averaged,
                    color=color,
                    linewidth=0.85,
                    linestyle=linestyle,
                    label=label,
                )
                axes[row_index, column].set_ylabel(ylabel)
                axes[row_index, column].grid(alpha=0.20)
            for boundary in (337, 674, 1011, 1348, 1685):
                for axis in axes[:, column]:
                    axis.axvline(
                        boundary, color="#9ca3af", linewidth=0.35,
                        linestyle=":", alpha=0.5
                    )
        axes[0, column].set_title(standard_label)
        axes[2, column].set_xlabel("optimizer step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="lower center",
                  bbox_to_anchor=(0.5, -0.01), fontsize=7.5, frameon=False)
    figure.suptitle(
        "M2 injection-point trajectories · 2× historical vs 128× current standard\n"
        "points = raw online records; thin lines = 3-point visual connector; "
        "dotted lines = epoch boundaries"
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    save_figure(figure, "fig_v5_2x_128x_injection_curves.png")


# ---------- M2 injection frequency bins (128×) ----------
def plot_injection_frequency():
    arms = [
        ("input", "#2d6f9f"),
        ("y", "#c4493d"),
        ("v", "#b67524"),
        ("nogram", "#686d73"),
    ]
    records = {}
    for arm, _ in arms:
        record = final_record(f"nglab1x_{arm}_v5_128x_freq10")
        if record is None or int(record.get("step", -1)) != 2000:
            print(f"skip: {arm} 128x freq record incomplete")
            return
        records[arm] = record

    for branch in ("bigram", "trigram"):
        buckets = ordered_frequency_buckets(records["input"]["train"][branch])
        x = np.arange(len(buckets))
        figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
        for arm, color in arms:
            record = records[arm]
            train = record["train"][branch]
            val = record["val"][branch]
            train_fraction = [train[bucket]["frac"] for bucket in buckets]
            val_fraction = [val[bucket]["frac"] for bucket in buckets]
            gap = [
                (
                    val[bucket]["mean_loss"] - train[bucket]["mean_loss"]
                    if train[bucket]["token_count"] > 0
                    else np.nan
                )
                for bucket in buckets
            ]
            if arm == "input":
                axes[0].bar(x - 0.18, train_fraction, width=0.36, color=color,
                            alpha=0.70, label="input train fraction")
                axes[0].bar(x + 0.18, val_fraction, width=0.36, color=color,
                            alpha=0.28, label="input fixed-val fraction")
            axes[1].scatter(x, gap, color=color, s=19, alpha=0.70, zorder=2)
            add_finite_smoothed_segments(axes[1], x, gap, color, arm)
        axes[0].set_ylabel("token fraction")
        axes[0].set_title("frequency coverage: input arm")
        axes[0].legend(fontsize=8, frameon=False)
        axes[1].axhline(0, color="#686d73", linewidth=0.7, linestyle=":")
        axes[1].set_ylabel("fixed-val mean loss − current-batch train mean loss")
        axes[1].set_title("per-bin gap across injection arms")
        axes[1].legend(fontsize=8, frameon=False)
        for axis in axes:
            axis.set_xticks(x, buckets, rotation=38, ha="right")
            axis.set_xlabel("train context hit-count bucket")
            axis.grid(alpha=0.22)
        figure.suptitle(
            f"M2 v5 128× current-batch frequency bins · {branch} · final step 2000\n"
            "points = raw buckets; thin line = 3-point visual connector; novel has no gap",
            fontsize=10,
        )
        figure.tight_layout()
        save_figure(figure, f"fig_v5_128x_injection_frequency_{branch}.png")


# ---------- Causal 7-arm curves (128×) ----------
def plot_causal_losses():
    run_ids = (
        "causalv5c_none_128x",
        "causalv5c_freeze_table_e1_128x",
        "causalv5c_freeze_backbone_e1_128x",
        "causalv5c_hash_reseed_e1_128x",
        "causalv5c_hash_reseed_e1e2",
        "causalv5c_mask_low_f200_e1_128x",
        "causalv5m2_mask_high_t200_e1",
    )
    labels = {
        "causalv5c_none_128x": "none",
        "causalv5c_freeze_table_e1_128x": "freeze_table e1",
        "causalv5c_freeze_backbone_e1_128x": "freeze_backbone e1",
        "causalv5c_hash_reseed_e1_128x": "hash_reseed e1",
        "causalv5c_hash_reseed_e1e2": "hash_reseed e1+e2",
        "causalv5c_mask_low_f200_e1_128x": "mask_low f≤200",
        "causalv5m2_mask_high_t200_e1": "mask_high f≥200",
    }
    curves = []
    for run_id in run_ids:
        summary, rows = read_run(run_id)
        if not rows or max(row["step"] for row in rows) != 1000:
            print(f"skip causal curve: {run_id} incomplete")
            return
        curves.append((run_id, rows))
    colors = plt.get_cmap("tab10")(np.arange(len(curves)))
    figure, axes = plt.subplots(3, 1, figsize=(10.4, 8.8), sharex=True)
    metrics = (
        ("train_loss", "online train loss"),
        ("val_loss", "fixed validation loss"),
        ("gap", "online gap = fixed val − online train"),
    )
    for color, (run_id, rows) in zip(colors, curves):
        steps = np.asarray([row["step"] for row in rows])
        label = labels[run_id]
        for axis, (metric, ylabel) in zip(axes, metrics):
            values = np.asarray(
                [row.get(metric, row["val_loss"] - row["train_loss"]) for row in rows]
            )
            axis.scatter(steps, values, color=color, s=3.2, alpha=0.25)
            averaged, offsets = smooth(values)
            axis.plot(steps[offsets], averaged, color=color, linewidth=0.75, label=label)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.20)
            for boundary in (337, 674):
                axis.axvline(
                    boundary, color="#9ca3af", linewidth=0.35,
                    linestyle=":", alpha=0.55
                )
    axes[0].legend(ncol=4, fontsize=7, frameon=False)
    axes[-1].set_xlabel("optimizer step")
    figure.suptitle(
        "V5 causal-refresh 128× · seven matched intervention arms (1000 steps)\n"
        "hash_reseed e1+e2 resees the context→row mapping at both epoch boundaries;\n"
        "mask_low f≤200 masks seen contexts with f≤200 plus novel (f=0) at train and eval;\n"
        "points = raw online records; thin lines = 3-point visual connector",
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_128x_causal_losses.png")


# ---------- Causal frequency contribution (128×) ----------
def plot_causal_frequency_effect():
    frequency_records = {
        run_id: final_record(run_id)
        for run_id in (
            "causalv5c_none_128x",
            "causalv5c_mask_low_f200_e1_128x",
            "causalv5m2_mask_high_t200_e1",
        )
    }
    if any(record is None or int(record.get("step", -1)) != 1000
           for record in frequency_records.values()):
        print("skip causal frequency figure: final frequency evidence incomplete")
        return
    labels = {
        "causalv5c_none_128x": "none",
        "causalv5c_mask_low_f200_e1_128x": "mask_low f≤200",
        "causalv5m2_mask_high_t200_e1": "mask_high f≥200",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
    for axis, branch in zip(axes, ("bigram", "trigram")):
        for run_id, color in zip(frequency_records, ("#374151", "#0f766e", "#d97706")):
            gaps = frequency_gaps(frequency_records[run_id], branch)
            buckets = ordered_frequency_buckets(gaps)
            x = np.arange(len(buckets))
            values = [gaps[bucket] for bucket in buckets]
            axis.scatter(x, values, color=color, s=20, alpha=0.75)
            add_finite_smoothed_segments(axis, x, values, color, labels[run_id])
        axis.axhline(0, color="#686d73", linewidth=0.7, linestyle=":")
        axis.set_xticks(x, buckets, rotation=38, ha="right")
        axis.set_xlabel("train context hit-count bucket")
        axis.set_title(branch)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("fixed-val mean loss − current-batch train mean loss")
    axes[0].legend(fontsize=8, frameon=False)
    figure.suptitle(
        "V5 causal frequency contribution · 128× · final step 1000\n"
        "low mask = f≤200 incl. novel (train+eval); high mask = f≥200;\n"
        "points = raw bins; thin line = 3-point connector",
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_128x_causal_frequency_effect.png")


# ---------- mask_low inclusive threshold scan (f≤t, 128×) ----------
def plot_mask_low_le_scan():
    """Gap trajectories for mask_low f≤t at t=0,1,2,4,8 (+ t=200 endpoint).

    All scan arms mask at train AND eval (the mask lives in forward), and
    novel (f=0) contexts are always masked in low mode.  t=0 therefore only
    removes the novel-context table readout at eval; t≥1 additionally removes
    the training signal of contexts seen ≤ t times."""
    run_specs = (
        ("causalv5c_none_128x", "none (control)", "#6b7280"),
        ("causalv5m_mask_low_le0_e1", "mask f≤0 (novel only)", "#8dd3c7"),
        ("causalv5m_mask_low_le1_e1", "mask f≤1", "#4c9f70"),
        ("causalv5m_mask_low_le2_e1", "mask f≤2", "#2d86b3"),
        ("causalv5m_mask_low_le4_e1", "mask f≤4", "#2660a4"),
        ("causalv5m_mask_low_le8_e1", "mask f≤8", "#1b3a6b"),
        ("causalv5c_mask_low_f200_e1_128x", "mask f≤200 (endpoint)", "#0f766e"),
    )
    curves = []
    for run_id, label, color in run_specs:
        summary, rows = read_run(run_id)
        if not rows or max(row["step"] for row in rows) != 1000:
            print(f"skip mask_low le-scan: {run_id} incomplete")
            return
        curves.append((label, color, rows))
    figure, axis = plt.subplots(figsize=(10.4, 5.6))
    for label, color, rows in curves:
        steps = np.asarray([row["step"] for row in rows])
        values = np.asarray(
            [row.get("gap", row["val_loss"] - row["train_loss"]) for row in rows]
        )
        axis.scatter(steps, values, color=color, s=3.2, alpha=0.25)
        averaged, offsets = smooth(values)
        axis.plot(steps[offsets], averaged, color=color, linewidth=0.9, label=label)
    for boundary in (337, 674):
        axis.axvline(boundary, color="#9ca3af", linewidth=0.35,
                     linestyle=":", alpha=0.55)
    axis.axhline(0, color="#686d73", linewidth=0.7, linestyle=":")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("online gap = fixed val − online train")
    axis.grid(alpha=0.20)
    axis.legend(fontsize=8, frameon=False, loc="upper left")
    figure.suptitle(
        "mask_low f≤t threshold scan · 128× · intervention at epoch-2 boundary (1000 steps)\n"
        "mask applies at train and eval; novel (f=0) contexts always masked in low mode;\n"
        "points = raw online records; thin lines = 3-point visual connector",
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_128x_mask_low_le_scan.png")


# ---------- freeze arms: forking with a frozen write/read path ----------
def plot_freeze_forking():
    """freeze_table and freeze_backbone both still develop gap after the
    epoch-2 boundary — dedicated 3-panel view with final values annotated."""
    run_specs = (
        ("causalv5c_none_128x", "none (control)", "#374151"),
        ("causalv5c_freeze_table_e1_128x", "freeze_table e1", "#d97706"),
        ("causalv5c_freeze_backbone_e1_128x", "freeze_backbone e1", "#0f766e"),
    )
    curves = []
    for run_id, label, color in run_specs:
        summary, rows = read_run(run_id)
        if not rows or max(row["step"] for row in rows) != 1000:
            print(f"skip freeze forking: {run_id} incomplete")
            return
        curves.append((label, color, rows))
    figure, axes = plt.subplots(3, 1, figsize=(10.4, 8.4), sharex=True)
    metrics = (
        ("train_loss", "online train loss"),
        ("val_loss", "fixed validation loss"),
        ("gap", "online gap = fixed val − online train"),
    )
    for label, color, rows in curves:
        steps = np.asarray([row["step"] for row in rows])
        for axis, (metric, ylabel) in zip(axes, metrics):
            values = np.asarray(
                [row.get(metric, row["val_loss"] - row["train_loss"]) for row in rows]
            )
            axis.scatter(steps, values, color=color, s=3.2, alpha=0.25)
            averaged, offsets = smooth(values)
            axis.plot(steps[offsets], averaged, color=color,
                      linewidth=0.9, label=label)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.20)
            for boundary in (337, 674):
                axis.axvline(boundary, color="#9ca3af", linewidth=0.35,
                             linestyle=":", alpha=0.55)
    gap_axis = axes[-1]
    for label, color, rows in curves:
        final = rows[-1].get("gap", rows[-1]["val_loss"] - rows[-1]["train_loss"])
        gap_axis.annotate(
            f"{label}: {final:+.2f}",
            (rows[-1]["step"], final), textcoords="offset points",
            xytext=(6, 0), fontsize=8, color=color, va="center",
        )
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[-1].set_xlabel("optimizer step")
    figure.suptitle(
        "freeze arms still fork · 128× · freeze at epoch-2 boundary (1000 steps)\n"
        "freeze_table: backbone keeps learning on frozen table content;\n"
        "freeze_backbone: table keeps writing under RMSProp 128×;\n"
        "points = raw online records; thin lines = 3-point visual connector",
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_128x_freeze_forking.png")


# ---------- S1 epoch number: gap growth within one long replay ----------
def plot_epoch_number():
    """Gap at each epoch boundary of the 20-epoch L4 long replay (§41).

    Source of truth: docs/appendices/s1_scaling_three_axis/
    s1_epoch_long_replay_points.csv (20ep run rows).  Left: raw gap vs epoch
    number for trigram-only, both-tables and the no-gram control.  Right:
    per-epoch increments -- early peak then slow decay; still positive at
    e20, i.e. sublinear growth, no observed plateau.
    """
    import csv as _csv
    csv_path = (ROOT / "docs" / "appendices" / "s1_scaling_three_axis"
                / "s1_epoch_long_replay_points.csv")
    by_arm = {}
    for row in _csv.DictReader(csv_path.open()):
        if "20ep" not in row["run_id"]:
            continue
        by_arm.setdefault(row["arm"], {})[int(row["epoch"])] = float(row["gap"])
    runs = (
        ("trigram-only", "trigram-only", "#2d6f9f"),
        ("bigram+trigram", "both-tables", "#67439b"),
        ("no-gram control", "nogram", "#686d73"),
    )
    series = {}
    for label, arm, color in runs:
        pts = sorted(by_arm.get(arm, {}).items())
        if len(pts) < 20:
            print(f"skip epoch-number figure: {arm} incomplete ({len(pts)}/20)")
            return
        series[label] = (color, pts)

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    for label, (color, epochs) in series.items():
        x = [e for e, _ in epochs]
        y = [g for _, g in epochs]
        axes[0].scatter(x, y, color=color, s=22, zorder=3)
        axes[0].plot(x, y, color=color, linewidth=0.9, alpha=0.85, label=label)
        if label != "no-gram control":
            increments = [y[i] - y[i - 1] for i in range(1, len(y))]
            axes[1].plot(x[1:], increments, color=color, linewidth=0.9,
                         marker="o", markersize=3.6, label=label)
    tri_color, tri_epochs = series["trigram-only"]
    ref = [(e, g) for e, g in tri_epochs if e >= 2]
    slope = (ref[-1][1] - ref[0][1]) / (ref[-1][0] - ref[0][0])
    axes[0].plot([ref[0][0], ref[-1][0]],
                 [ref[0][1], ref[0][1] + slope * (ref[-1][0] - ref[0][0])],
                 color="#9ca3af", linestyle="--", linewidth=0.8,
                 label=f"secant ref ≈ {slope:.2f}/epoch (e2–e20)")
    axes[0].axhline(0, color="#686d73", linewidth=0.6, linestyle=":")
    axes[0].set_xlabel("epoch number (L4 = 337 batches / epoch)")
    axes[0].set_ylabel("gap at epoch boundary")
    axes[0].set_title("gap vs epoch number · 20-epoch L4 replay")
    axes[0].set_xticks(range(1, 21, 2))
    axes[1].axhline(0, color="#686d73", linewidth=0.6, linestyle=":")
    axes[1].set_xlabel("epoch boundary (e−1 → e)")
    axes[1].set_ylabel("per-epoch gap increment")
    axes[1].set_title("increment: early peak → slow decay, still > 0 at e20")
    axes[1].set_xticks(range(2, 21, 2))
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle(
        "S1 epoch-number relation · 128× · seed 42 · fixed replay of shard 1 · 20 epochs\n"
        "points = raw boundary records (gap = fixed val − online train); "
        "sublinear growth, no plateau observed up to e20",
        fontsize=10,
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_s1_epoch_number.png")


if __name__ == "__main__":
    plot_s1_table_size_clean()
    plot_injection_comparison()
    plot_injection_frequency()
    plot_causal_losses()
    plot_causal_frequency_effect()
    plot_mask_low_le_scan()
    plot_freeze_forking()
    plot_epoch_number()
    print("DONE")
