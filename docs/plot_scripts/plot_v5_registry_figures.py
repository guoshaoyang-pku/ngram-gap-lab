#!/usr/bin/env python3
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS_FIXED = ROOT / "data" / "runs_fixed"
RUNS_SCALING = ROOT / "data" / "runs_scaling"
OUT = ROOT / "docs" / "figs" / "main"
V5_COLOR = "#0f766e"
BIGRAM_COLOR = "#0f766e"
TRIGRAM_COLOR = "#d97706"


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def maybe_read_jsonl(path):
    return read_jsonl(path) if path.exists() else []


def read_run(run_id, runs_dir=RUNS_FIXED):
    run_dir = runs_dir / f"{run_id}_fixed"
    return read_json(run_dir / "summary.json"), read_jsonl(run_dir / "train_log.jsonl")


def smooth(values):
    if len(values) < 3:
        return np.asarray(values), np.arange(len(values))
    return np.convolve(values, np.ones(3) / 3, mode="valid"), np.arange(1, len(values) - 1)


def add_raw_and_smoothed(axis, rows, metric, color, label):
    steps = np.asarray([row["step"] for row in rows])
    values = np.asarray([row[metric] for row in rows])
    axis.scatter(steps, values, color=color, s=9, alpha=0.5, zorder=2)
    averaged, offsets = smooth(values)
    axis.plot(steps[offsets], averaged, color=color, linewidth=1.0, label=label, zorder=3)


def save_figure(figure, filename):
    path = OUT / filename
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(path.relative_to(ROOT))


def plot_injection():
    arms = [
        ("input", "#2d6f9f"),
        ("y", "#c4493d"),
        ("v", "#b67524"),
        ("nogram", "#686d73"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharex=True)
    for arm, color in arms:
        _, rows = read_run(f"nglab1x_{arm}_v5")
        add_raw_and_smoothed(axes[0], rows, "gap", color, arm)
        add_raw_and_smoothed(axes[1], rows, "train_loss", color, f"{arm} train")
        values = [row["val_loss"] for row in rows]
        averaged, offsets = smooth(values)
        steps = np.asarray([row["step"] for row in rows])
        axes[1].plot(steps[offsets], averaged, color=color, linewidth=1.0, linestyle="--",
                     label=f"{arm} val")
    axes[0].axhline(0, color="#686d73", linewidth=0.7, linestyle=":")
    axes[0].set_title("V5 injection arms: online gap")
    axes[0].set_ylabel("fixed validation loss − online train loss")
    axes[1].set_title("V5 injection arms: loss")
    axes[1].set_ylabel("loss")
    for axis in axes:
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5, frameon=False, ncol=2)
    figure.suptitle(
        "V5 · seed 42 · clean bigram+trigram R=2²⁰ · warmup_constant(100) · bf16 · no compile",
        fontsize=10,
        y=1.02,
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_injection.png")


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


def add_finite_smoothed_segments(axis, x, values, color, label):
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
            linewidth=0.9,
            label=label if index == 0 else None,
        )


def plot_injection_frequency():
    arms = [
        ("input", "#2d6f9f"),
        ("y", "#c4493d"),
        ("v", "#b67524"),
        ("nogram", "#686d73"),
    ]
    finals = {}
    for arm, _ in arms:
        run_id = f"nglab1x_{arm}_v5_freq10{'_r1' if arm == 'input' else ''}"
        record = completed_frequency_record(run_id, 2000)
        if record is None:
            print("skip M2 current-batch frequency figures: four-arm freq10 batch incomplete")
            return
        finals[arm] = record

    for branch in ("bigram", "trigram"):
        buckets = ordered_frequency_buckets(finals["input"]["train"][branch])
        x = np.arange(len(buckets))
        figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.8))
        for arm, color in arms:
            record = finals[arm]
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
            f"M2 v5 current-batch frequency bins · {branch} · final step 2000\n"
            "points = raw buckets; thin line = 3-point visual connector; novel has no gap",
            fontsize=10,
        )
        figure.tight_layout()
        save_figure(figure, f"fig_v5_injection_frequency_{branch}.png")


def plot_fixed_step_dose():
    doses = [
        ("0.25×", "nglab0_25x_input_v5", 0.25),
        ("0.5×", "nglab0_5x_input_v5", 0.5),
        ("0.75×", "nglab0_75x_input_v5", 0.75),
        ("1.5×", "nglab1_5x_input_v5", 1.5),
        ("2×", "nglab2x_input_v5", 2.0),
        ("2.5×", "nglab2_5x_input_v5", 2.5),
        ("3×", "nglab3x_input_v5", 3.0),
        ("4×", "nglab4x_input_v5", 4.0),
        ("5×", "nglab5x_input_v5", 5.0),
        ("6×", "nglab6x_input_v5", 6.0),
        ("8×", "nglab8x_input_v5", 8.0),
    ]
    xs, gaps = [], []
    for _, run_id, dose in doses:
        summary, _ = read_run(run_id)
        xs.append(dose)
        gaps.append(summary["final_gap"])
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    axis.plot(xs, gaps, color="#2d6f9f", linewidth=1.1, marker="o", markersize=4)
    axis.axhline(0, color="#686d73", linewidth=0.7, linestyle=":")
    axis.set_xscale("log")
    axis.set_xlabel("train-shard dose relative to 1× (log scale)")
    axis.set_ylabel("final online gap at step 2000")
    axis.set_title("V5 fixed-step dose scan")
    axis.grid(alpha=0.25, which="both")
    for label, dose, gap in zip((item[0] for item in doses), xs, gaps):
        axis.annotate(label, (dose, gap), xytext=(0, 6), textcoords="offset points",
                      ha="center", fontsize=7)
    figure.text(
        0.5,
        0.01,
        "input · seed 42 · clean R=2²⁰ · warmup_constant(100) · bf16 · no compile",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, "fig_v5_dose_fixedstep.png")


def plot_s1_epoch():
    lengths = ["L1", "L2", "L3", "L4"]
    both, nogram = [], []
    for length in lengths:
        both_summary, _ = read_run(f"s1v5_{length}_both_fs", RUNS_SCALING)
        nogram_summary, _ = read_run(f"s1v5_{length}_nogram_fs", RUNS_SCALING)
        both.append(both_summary["final_gap"])
        nogram.append(nogram_summary["final_gap"])
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(lengths))
    axis.plot(x, both, color="#353d79", marker="o", linewidth=1.2, label="bigram + trigram")
    axis.plot(x, nogram, color="#8a8f8a", marker="o", linewidth=1.2, label="no-gram")
    axis.plot(x, np.asarray(both) - np.asarray(nogram), color="#c4493d", marker="o",
              linewidth=1.2, label="table-induced Δgap")
    axis.set_xticks(x, ["L1\n42", "L2\n84", "L3\n168", "L4\n337"])
    axis.set_xlabel("epoch-prefix length (device batches per epoch)")
    axis.set_ylabel("final online gap at step 1000")
    axis.set_title("V5 S1 epoch-prefix axis")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False)
    figure.text(
        0.5,
        0.01,
        "seed 42 · fixed-step · clean R=2²⁰ · warmup_constant(100) · bf16 · no compile",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, "fig_v5_s1_epoch_prefix.png")


def plot_table_size():
    points = []
    for run_dir in sorted(RUNS_SCALING.glob("ctbl_v5_both_*_fixed")):
        summary = read_json(run_dir / "summary.json")
        rows = int(run_dir.name.removeprefix("ctbl_v5_both_").removesuffix("_fixed"))
        points.append((rows, summary["final_gap"]))
    points.sort()
    rows, gaps = zip(*points)
    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    axis.plot(rows, gaps, color="#353d79", marker="o", markersize=3.5, linewidth=1.1)
    axis.set_xscale("log")
    axis.set_xlabel("physical rows R per clean table (log scale)")
    axis.set_ylabel("final online gap at step 1000")
    axis.set_title("V5 clean double-table size scan")
    axis.grid(alpha=0.25, which="both")
    figure.text(
        0.5,
        0.01,
        "bigram and trigram enlarged together · seed 42 · warmup_constant(100) · bf16 · no compile",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, "fig_v5_s1_table_size.png")

    positive_points = [(row, gap) for row, gap in points if gap > 0]
    if positive_points:
        log_rows, log_gaps = zip(*positive_points)
        figure, axis = plt.subplots(figsize=(7.8, 4.8))
        axis.plot(log_rows, log_gaps, color="#353d79", marker="o", markersize=3.5,
                  linewidth=1.0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("physical rows R per clean table (log scale)")
        axis.set_ylabel("final online gap at step 1000 (log scale)")
        axis.set_title("V5 clean double-table size scan · positive-gap log–log view")
        axis.grid(alpha=0.25, which="both")
        figure.text(
            0.5,
            0.01,
            "Raw endpoints only. Non-positive gaps are excluded because log(y) is undefined.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        save_figure(figure, "fig_v5_s1_table_size_loglog.png")

    load_points = {"bigram": [], "trigram": []}
    for run_dir in sorted(RUNS_SCALING.glob("ctbl_v5_both_*_fixed")):
        occupancy_path = run_dir / "table_occupancy.json"
        if not occupancy_path.exists():
            continue
        payload = read_json(occupancy_path)
        branches = payload.get("branches", payload)
        for branch in load_points:
            values = branches.get(branch, {})
            if isinstance(values, dict) and "0" in values:
                values = values["0"][0]
            distinct = values.get("distinct_contexts_K", values.get("distinct_contexts"))
            physical = values.get("physical_rows_R", values.get("table_size"))
            collision = values.get("collision_rate")
            if distinct is None or physical is None or collision is None:
                continue
            load_points[branch].append(
                (float(physical), float(distinct) / float(physical), float(collision))
            )
    if any(load_points.values()):
        figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharex=True)
        for branch, color in (("bigram", BIGRAM_COLOR), ("trigram", TRIGRAM_COLOR)):
            values = sorted(load_points[branch])
            if not values:
                continue
            xs, loads, collisions = zip(*values)
            axes[0].plot(xs, loads, marker="o", markersize=3.6, linewidth=1.0,
                         color=color, label=branch)
            axes[1].plot(xs, collisions, marker="o", markersize=3.6, linewidth=1.0,
                         color=color, label=branch)
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlabel("physical rows R per branch (log scale)")
            axis.grid(alpha=0.25, which="both")
            axis.legend(frameon=False)
        axes[0].set_ylabel("load ratio K/R")
        axes[0].set_title("K/R is table load, not collision")
        axes[1].set_ylabel("measured collision rate = (K − occupied) / K")
        axes[1].set_title("Actual hash collision rate")
        figure.suptitle("V5 clean double-table occupancy accounting")
        figure.tight_layout()
        save_figure(figure, "fig_v5_s1_table_load_collision.png")


def final_record(path):
    records = maybe_read_jsonl(path)
    return max(records, key=lambda row: int(row.get("step", -1))) if records else None


def completed_curve(run_id, expected_step):
    summary_path = RUNS_FIXED / f"{run_id}_fixed" / "summary.json"
    rows = read_curve(run_id)
    if not summary_path.exists() or not rows:
        return None
    if max(int(row["step"]) for row in rows) != expected_step:
        return None
    return rows


def completed_frequency_record(run_id, expected_step):
    run_dir = RUNS_FIXED / f"{run_id}_fixed"
    if not (run_dir / "summary.json").exists():
        return None
    record = final_record(run_dir / "freq_bin_loss.jsonl")
    if record is None or int(record.get("step", -1)) != expected_step:
        return None
    return record


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


def intervention_steps(run_id):
    summary_path = RUNS_FIXED / f"{run_id}_fixed" / "summary.json"
    if not summary_path.exists():
        return []
    events = read_json(summary_path).get("intervention", {}).get("events", [])
    return [
        int(event["step"])
        for event in events
        if isinstance(event.get("step"), (int, float))
    ]


def exact_frequency_rows_from_shared(record, branch):
    per_frequency = record.get("shared", {}).get(branch, {}).get("per_f", {})
    rows = []
    for frequency_text, values in per_frequency.items():
        frequency = int(frequency_text)
        if frequency <= 0 or not values.get("shared_contexts"):
            continue
        gap = values.get("gap")
        if gap is None:
            continue
        rows.append(
            {
                "f": frequency,
                "gap": float(gap),
                "shared_token_mass": float(values["shared_contexts"]) * frequency,
                "contexts": int(values["shared_contexts"]),
            }
        )
    return sorted(rows, key=lambda row: row["f"])


def pool_exact_frequency_gap(rows, bins=12):
    usable = [
        row for row in rows
        if np.isfinite(row["gap"]) and row["f"] > 0 and row["contexts"] >= 32
    ]
    if not usable:
        return []
    values = np.asarray([row["f"] for row in usable], dtype=float)
    bounds = np.unique(np.geomspace(values.min(), values.max() + 1, bins + 1).astype(int))
    output = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        members = [row for row in usable if low <= row["f"] < high]
        if not members:
            continue
        shared_token_mass = sum(row["shared_token_mass"] for row in members)
        output.append(
            {
                "f_mid": math.sqrt(low * max(low + 1, high - 1)),
                "gap": (
                    sum(row["gap"] * row["shared_token_mass"] for row in members)
                    / shared_token_mass
                ),
                "contexts": sum(row["contexts"] for row in members),
                "label": f"{low}–{high - 1}",
            }
        )
    return output


def exact_frequency_mass_from_index(index_path, branch, bins=12):
    if not index_path.exists():
        return []
    with np.load(index_path) as index:
        counts = np.asarray(index[f"{branch}_counts"], dtype=np.int64)
    positive = counts[counts > 0]
    if positive.size == 0:
        return []
    bounds = np.unique(
        np.geomspace(positive.min(), positive.max() + 1, bins + 1).astype(int)
    )
    output = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        members = positive[(positive >= low) & (positive < high)]
        if not members.size:
            continue
        output.append(
            {
                "token_mass": int(members.sum()),
                "label": f"{low}–{high - 1}",
            }
        )
    return output


def plot_s1_frequency():
    records = {
        branch: final_record(
            RUNS_SCALING / f"s1v5_freq_{branch}_fixed" / "exact_freq_loss.jsonl"
        )
        for branch in ("bigram", "trigram")
    }
    if not any(records.values()):
        print("skip S1 frequency figures: exact-frequency log unavailable")
        return
    by_branch = {
        branch: exact_frequency_rows_from_shared(record, branch)
        for branch, record in records.items()
        if record is not None
    }
    if not any(by_branch.values()):
        print("skip S1 frequency figures: no context-matched exact-frequency rows")
        return
    summary = read_json(RUNS_SCALING / "s1v5_freq_bigram_fixed" / "summary.json")
    index_path = ROOT / "data" / Path(summary.get("freq_index", "")).name

    figure, axis = plt.subplots(figsize=(8.5, 4.9))
    for branch, color in (("bigram", BIGRAM_COLOR), ("trigram", TRIGRAM_COLOR)):
        rows = by_branch[branch]
        if not rows:
            continue
        raw_rows = [row for row in rows if row["contexts"] >= 32]
        axis.scatter([row["f"] for row in raw_rows], [row["gap"] for row in raw_rows],
                     s=9, alpha=0.30, color=color, label=f"{branch} raw f (≥32 shared)")
        pooled = pool_exact_frequency_gap(rows)
        axis.plot([row["f_mid"] for row in pooled], [row["gap"] for row in pooled],
                  marker="o", markersize=3.6, linewidth=1.0, color=color,
                  label=f"{branch} shared-token-mass pooled")
    axis.set_xscale("log")
    axis.set_xlabel("exact train hit-count per context f (log scale)")
    axis.set_ylabel("diagnostic context-matched gap = val loss − train loss")
    final_steps = sorted({record["step"] for record in records.values() if record is not None})
    axis.set_title(
        f"S1 diagnostic exact-frequency gap at final logged step {final_steps[-1]}"
    )
    axis.grid(alpha=0.25, which="both")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.text(
        0.5,
        0.01,
        "Fixed-probe diagnostic: points and pooled bins require ≥32 shared contexts. "
        "Thin lines pool f intervals by shared-context token mass only. "
        "Novel f=0 contexts have no train loss, therefore no gap.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, "fig_v5_s1_frequency_exact_f.png")

    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.8), sharex=True)
    for axis, branch, color in zip(axes, ("bigram", "trigram"), (BIGRAM_COLOR, TRIGRAM_COLOR)):
        mass_rows = exact_frequency_mass_from_index(index_path, branch)
        if not mass_rows:
            axis.set_visible(False)
            continue
        x = np.arange(len(mass_rows))
        axis.bar(x, [row["token_mass"] for row in mass_rows], color=color, alpha=0.78)
        axis.set_yscale("log")
        axis.set_ylabel("Σf over train contexts\n(log)")
        axis.set_title(branch)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x, [row["label"] for row in mass_rows], rotation=35, ha="right")
    if not any(
        exact_frequency_mass_from_index(index_path, branch)
        for branch in ("bigram", "trigram")
    ):
        plt.close(figure)
        print("skip S1 frequency token-mass figure: training frequency index unavailable")
        return
    axes[-1].set_xlabel("exact-f interval; bar height is total token exposure Σf")
    figure.suptitle("S1 frequency coverage: token exposure, not distinct-context count")
    figure.tight_layout()
    save_figure(figure, "fig_v5_s1_frequency_token_mass.png")


def read_curve(run_id):
    records = maybe_read_jsonl(RUNS_FIXED / f"{run_id}_fixed" / "train_log.jsonl")
    return [
        row for row in records
        if {"step", "train_loss", "val_loss"}.issubset(row)
    ]


def plot_curve_grid(run_ids, filename, title, boundary_runs=()):
    available = [(run_id, read_curve(run_id)) for run_id in run_ids]
    available = [(run_id, rows) for run_id, rows in available if rows]
    if not available:
        print(f"skip {filename}: curves unavailable")
        return
    figure, axes = plt.subplots(3, len(available), figsize=(4.3 * len(available), 8.1),
                                sharex="col", squeeze=False)
    metrics = (
        ("train_loss", "online train loss"),
        ("val_loss", "fixed validation loss"),
        ("gap", "online gap"),
    )
    for column, (run_id, rows) in enumerate(available):
        steps = np.asarray([row["step"] for row in rows])
        for row_index, (metric, label) in enumerate(metrics):
            values = np.asarray(
                [row.get(metric, row["val_loss"] - row["train_loss"]) for row in rows]
            )
            axis = axes[row_index, column]
            axis.scatter(steps, values, color=V5_COLOR, s=4.2, alpha=0.40)
            averaged, offsets = smooth(values)
            axis.plot(steps[offsets], averaged, color=V5_COLOR, linewidth=0.8)
            if run_id in boundary_runs:
                first_epoch_two = next(
                    (row["step"] for row in rows if int(row.get("epoch", 0)) >= 2),
                    None,
                )
                if first_epoch_two is not None:
                    axis.axvline(first_epoch_two, color="#374151", linewidth=0.75, linestyle="--")
            axis.set_ylabel(label)
            axis.grid(alpha=0.20)
        axes[0, column].set_title(run_id.replace("nglab1x_", "").replace("_v5", ""))
        axes[-1, column].set_xlabel("optimizer step")
    figure.suptitle(title + "\npoints = raw online records; thin line = 3-point visual connector")
    figure.tight_layout()
    save_figure(figure, filename)


def plot_causal_refresh():
    run_ids = (
        "causalv5c_none",
        "causalv5c_reset_table_e1",
        "causalv5c_reset_table_e2",
        "causalv5c_mask_readout_e1",
        "causalv5c_freeze_table_e1",
        "causalv5c_freeze_backbone_e1",
        "causalv5c_hash_reseed_e1",
        "causalv5c_mask_low_f200_e1",
        "causalv5c_mask_high_f200_e1",
    )
    curves = [(run_id, completed_curve(run_id, 1000)) for run_id in run_ids]
    if any(rows is None for _, rows in curves):
        print("skip causal-refresh figures: nine-arm batch incomplete")
        return
    colors = plt.get_cmap("tab10")(np.arange(len(curves)))
    figure, axes = plt.subplots(3, 1, figsize=(10.4, 8.8), sharex=True)
    metrics = (
        ("train_loss", "online train loss"),
        ("val_loss", "fixed validation loss"),
        ("gap", "online gap = fixed val − online train"),
    )
    for color, (run_id, rows) in zip(colors, curves):
        steps = np.asarray([row["step"] for row in rows])
        label = run_id.replace("causalv5c_", "")
        for axis, (metric, ylabel) in zip(axes, metrics):
            values = np.asarray(
                [row.get(metric, row["val_loss"] - row["train_loss"]) for row in rows]
            )
            axis.scatter(steps, values, color=color, s=3.2, alpha=0.28)
            averaged, offsets = smooth(values)
            axis.plot(steps[offsets], averaged, color=color, linewidth=0.75, label=label)
            for step in intervention_steps(run_id):
                axis.axvline(step, color=color, linewidth=0.45, linestyle=":", alpha=0.55)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.20)
    axes[0].legend(ncol=3, fontsize=7, frameon=False)
    axes[-1].set_xlabel("optimizer step")
    figure.suptitle(
        "V5 causal-refresh · nine matched intervention arms\n"
        "points = raw online records; thin lines = 3-point visual connector; dotted lines = recorded intervention step"
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_causal_losses.png")

    frequency_records = {
        run_id: completed_frequency_record(run_id, 1000)
        for run_id in ("causalv5c_none", "causalv5c_mask_low_f200_e1", "causalv5c_mask_high_f200_e1")
    }
    if any(record is None for record in frequency_records.values()):
        print("skip causal frequency figure: final frequency evidence incomplete")
        return
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
    for axis, branch in zip(axes, ("bigram", "trigram")):
        for run_id, color in zip(frequency_records, ("#374151", "#0f766e", "#d97706")):
            gaps = frequency_gaps(frequency_records[run_id], branch)
            buckets = ordered_frequency_buckets(gaps)
            x = np.arange(len(buckets))
            values = [gaps[bucket] for bucket in buckets]
            axis.scatter(x, values, color=color, s=20, alpha=0.75)
            add_finite_smoothed_segments(axis, x, values, color, run_id.replace("causalv5c_", ""))
        axis.axhline(0, color="#686d73", linewidth=0.7, linestyle=":")
        axis.set_xticks(x, buckets, rotation=38, ha="right")
        axis.set_xlabel("train context hit-count bucket")
        axis.set_title(branch)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("fixed-val mean loss − current-batch train mean loss")
    axes[0].legend(fontsize=8, frameon=False)
    figure.suptitle(
        "V5 causal frequency contribution · final step 1000\n"
        "low mask = f≤200; high mask = f>200; points = raw bins; thin line = 3-point connector"
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_causal_frequency_effect.png")


def plot_optimizer_refresh():
    groups = (
        (
            "table LR scale",
            (
                "optv5c_rms_b099_s0p5",
                "optv5c_rms_b099_s1p0",
                "optv5c_rms_b099_s2p0_r1",
                "optv5c_rms_b099_s3p0",
                "optv5c_rms_b099_s4p0",
            ),
        ),
        (
            "RMSProp β₂",
            (
                "optv5c_rms_b095_s2p0",
                "optv5c_rms_b098_s2p0",
                "optv5c_rms_b099_s2p0_r1",
                "optv5c_rms_b0995_s2p0",
                "optv5c_rms_b0999_s2p0",
            ),
        ),
        (
            "table optimizer",
            (
                "optv5c_rms_b099_s2p0_r1",
                "optv5c_adamw_b099_s2p0",
                "optv5c_sgd_m0_s2p0",
            ),
        ),
    )
    all_run_ids = tuple(dict.fromkeys(run_id for _, run_ids in groups for run_id in run_ids))
    curves = {run_id: completed_curve(run_id, 1000) for run_id in all_run_ids}
    if any(rows is None for rows in curves.values()):
        print("skip optimizer-refresh figures: eleven-arm batch incomplete")
        return
    figure, axes = plt.subplots(3, 3, figsize=(13.8, 8.6), sharex="col")
    metrics = (
        ("train_loss", "online train loss"),
        ("val_loss", "fixed validation loss"),
        ("gap", "online gap"),
    )
    for column, (title, run_ids) in enumerate(groups):
        colors = plt.get_cmap("tab10")(np.arange(len(run_ids)))
        for color, run_id in zip(colors, run_ids):
            rows = curves[run_id]
            steps = np.asarray([row["step"] for row in rows])
            label = run_id.replace("optv5c_", "")
            for row_index, (metric, ylabel) in enumerate(metrics):
                axis = axes[row_index, column]
                values = np.asarray(
                    [row.get(metric, row["val_loss"] - row["train_loss"]) for row in rows]
                )
                axis.scatter(steps, values, color=color, s=2.8, alpha=0.22)
                averaged, offsets = smooth(values)
                axis.plot(steps[offsets], averaged, color=color, linewidth=0.75, label=label)
                axis.set_ylabel(ylabel)
                axis.grid(alpha=0.20)
        axes[0, column].set_title(title)
        axes[0, column].legend(fontsize=6.6, frameon=False)
        axes[-1, column].set_xlabel("optimizer step")
    figure.suptitle(
        "V5 clean table optimizer refresh · full 1000-step curves\n"
        "points = raw online records; thin lines = 3-point visual connector"
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_optimizer_full_curves.png")

    candidate_ids = (
        "optv5c_rms_b099_s1p0",
        "optv5c_rms_b099_s2p0_r1",
        "optv5c_rms_b099_s3p0",
    )
    records = {
        run_id: completed_frequency_record(run_id, 1000)
        for run_id in candidate_ids
    }
    if any(record is None for record in records.values()):
        print("skip optimizer frequency figure: final frequency evidence incomplete")
        return
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
    for axis, branch in zip(axes, ("bigram", "trigram")):
        for run_id, color in zip(records, ("#2d6f9f", "#0f766e", "#c4493d")):
            gaps = frequency_gaps(records[run_id], branch)
            buckets = ordered_frequency_buckets(gaps)
            x = np.arange(len(buckets))
            values = [gaps[bucket] for bucket in buckets]
            axis.scatter(x, values, color=color, s=20, alpha=0.75)
            add_finite_smoothed_segments(axis, x, values, color, run_id.replace("optv5c_", ""))
        axis.axhline(0, color="#686d73", linewidth=0.7, linestyle=":")
        axis.set_xticks(x, buckets, rotation=38, ha="right")
        axis.set_xlabel("train context hit-count bucket")
        axis.set_title(branch)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("fixed-val mean loss − current-batch train mean loss")
    axes[0].legend(fontsize=8, frameon=False)
    figure.suptitle(
        "V5 table-LR frequency diagnostic · final step 1000\n"
        "points = raw bins; thin line = 3-point visual connector"
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_optimizer_frequency.png")


def plot_dose_refresh():
    doses = (
        ("0.25×", "nglab0_25x_input_v5_freq10"),
        ("0.5×", "nglab0_5x_input_v5_freq10"),
        ("0.75×", "nglab0_75x_input_v5_freq10"),
        ("1×", "nglab1x_input_v5_freq10_r1"),
        ("1.5×", "nglab1_5x_input_v5_freq10"),
        ("2×", "nglab2x_input_v5_freq10"),
        ("2.5×", "nglab2_5x_input_v5_freq10"),
        ("3×", "nglab3x_input_v5_freq10"),
        ("4×", "nglab4x_input_v5_freq10"),
        ("5×", "nglab5x_input_v5_freq10"),
        ("6×", "nglab6x_input_v5_freq10"),
        ("8×", "nglab8x_input_v5_freq10"),
    )
    curves = [(label, run_id, completed_curve(run_id, 2000)) for label, run_id in doses]
    if any(rows is None for _, _, rows in curves):
        print("skip dose-refresh figures: twelve-dose evidence incomplete")
        return
    figure, axes = plt.subplots(3, 1, figsize=(10.1, 8.7), sharex=True)
    colors = plt.get_cmap("viridis", len(curves))(np.arange(len(curves)))
    metrics = (
        ("train_loss", "online train loss"),
        ("val_loss", "fixed validation loss"),
        ("gap", "online gap"),
    )
    for color, (label, _, rows) in zip(colors, curves):
        steps = np.asarray([row["step"] for row in rows])
        boundary = next(
            (row["step"] for row in rows if int(row.get("epoch", 0)) >= 2),
            None,
        )
        for axis, (metric, ylabel) in zip(axes, metrics):
            values = np.asarray(
                [row.get(metric, row["val_loss"] - row["train_loss"]) for row in rows]
            )
            axis.scatter(steps, values, color=color, s=2.5, alpha=0.20)
            averaged, offsets = smooth(values)
            axis.plot(steps[offsets], averaged, color=color, linewidth=0.70, label=label)
            if boundary is not None:
                axis.axvline(boundary, color=color, linewidth=0.35, linestyle=":", alpha=0.45)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.20)
    axes[0].legend(ncol=4, fontsize=6.8, frameon=False)
    axes[-1].set_xlabel("optimizer step")
    figure.suptitle(
        "V5 shard-dose refresh · complete 2000-step trajectories\n"
        "points = raw online records; thin lines = 3-point visual connector; dotted lines = epoch-2 boundary"
    )
    figure.tight_layout()
    save_figure(figure, "fig_v5_dose_trajectories.png")

    frequency_records = {
        label: completed_frequency_record(run_id, 2000)
        for label, run_id in doses
    }
    if any(record is None for record in frequency_records.values()):
        print("skip dose frequency heatmap: final frequency evidence incomplete")
        return
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.1), sharey=True)
    for axis, branch in zip(axes, ("bigram", "trigram")):
        bucket_names = ordered_frequency_buckets(
            frequency_gaps(next(iter(frequency_records.values())), branch)
        )
        matrix = np.asarray(
            [
                [frequency_gaps(record, branch).get(bucket, np.nan) for bucket in bucket_names]
                for record in frequency_records.values()
            ],
            dtype=float,
        )
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", interpolation="none")
        axis.set_xticks(np.arange(len(bucket_names)), bucket_names, rotation=38, ha="right")
        axis.set_yticks(np.arange(len(doses)), [label for label, _ in doses])
        axis.set_xlabel("train context hit-count bucket")
        axis.set_title(branch)
        figure.colorbar(image, ax=axis, label="fixed-val mean loss − current-batch train mean loss")
    axes[0].set_ylabel("train-shard dose")
    figure.suptitle("V5 dose × frequency-bin gap · final step 2000 · raw bins only")
    figure.tight_layout()
    save_figure(figure, "fig_v5_dose_frequency_heatmap.png")


def plot_existing_causal():
    plot_curve_grid(
        (
            "nglab1x_reset_e1_v5",
            "nglab1x_reset_e2_v5",
            "nglab1x_mask_e1_v5",
            "nglab1x_freeze_table_e1_v5",
            "nglab1x_freeze_backbone_e1_v5",
        ),
        "fig_v5_causal_existing_losses.png",
        "Existing v5 causal arms (precursor batch; causal-refresh will supersede it)",
        {
            "nglab1x_reset_e1_v5",
            "nglab1x_reset_e2_v5",
            "nglab1x_mask_e1_v5",
            "nglab1x_freeze_table_e1_v5",
            "nglab1x_freeze_backbone_e1_v5",
        },
    )


def plot_existing_optimizer():
    plot_curve_grid(
        (
            "optv5_rms_b098_s2p0_curve",
            "optv5_rms_b099_s1p0_curve",
            "optv5_rms_b099_s3p0_curve",
        ),
        "fig_v5_optimizer_existing_curves.png",
        "Existing v5 optimizer curves (three precursor arms; clean 11-arm batch is planned)",
    )


def plot_existing_dose_trajectories():
    run_ids = (
        "nglab0_25x_input_v5",
        "nglab0_5x_input_v5",
        "nglab0_75x_input_v5",
        "nglab1x_input_v5",
        "nglab1_5x_input_v5",
        "nglab2x_input_v5",
        "nglab2_5x_input_v5",
        "nglab3x_input_v5",
        "nglab4x_input_v5",
        "nglab5x_input_v5",
        "nglab6x_input_v5",
        "nglab8x_input_v5",
    )
    available = [(run_id, read_curve(run_id)) for run_id in run_ids]
    available = [(run_id, rows) for run_id, rows in available if rows]
    if not available:
        print("skip dose trajectories: curves unavailable")
        return
    figure, axes = plt.subplots(3, 1, figsize=(8.8, 8.2), sharex=True)
    colors = plt.get_cmap("viridis", len(available))(np.arange(len(available)))
    for color, (run_id, rows) in zip(colors, available):
        steps = np.asarray([row["step"] for row in rows])
        label = run_id.replace("nglab", "").replace("_input_v5", "")
        for axis, metric, ylabel in zip(
            axes,
            ("train_loss", "val_loss", "gap"),
            ("online train loss", "fixed validation loss", "online gap"),
        ):
            values = np.asarray(
                [row.get(metric, row["val_loss"] - row["train_loss"]) for row in rows]
            )
            axis.scatter(steps, values, color=color, s=3.4, alpha=0.22)
            averaged, offsets = smooth(values)
            axis.plot(steps[offsets], averaged, color=color, linewidth=0.75, label=label)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.20)
    axes[0].legend(ncol=3, fontsize=7, frameon=False)
    axes[-1].set_xlabel("optimizer step")
    figure.suptitle("Existing v5 fixed-step dose trajectories\n"
                    "points = raw online records; thin line = 3-point visual connector")
    figure.tight_layout()
    save_figure(figure, "fig_v5_dose_trajectories_existing.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_injection()
    plot_injection_frequency()
    plot_causal_refresh()
    plot_optimizer_refresh()
    plot_dose_refresh()
    plot_fixed_step_dose()
    plot_s1_epoch()
    plot_table_size()
    plot_s1_frequency()
    plot_existing_causal()
    plot_existing_optimizer()
    plot_existing_dose_trajectories()


if __name__ == "__main__":
    main()