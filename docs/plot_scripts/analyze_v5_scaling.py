#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUNS_FIXED = ROOT / "data" / "runs_fixed"
RUNS_SCALING = ROOT / "data" / "runs_scaling"
OUT = ROOT / "docs" / "appendices" / "s1_scaling_three_axis"
SOURCE_REVISION = "7583ae3222ffb4bbfb13262295a6a828e1f08d3f"


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summary(run_id, scaling=False):
    base = RUNS_SCALING if scaling else RUNS_FIXED
    return read_json(base / f"{run_id}_fixed" / "summary.json")


def run_rows(run_id, scaling=False):
    base = RUNS_SCALING if scaling else RUNS_FIXED
    return read_jsonl(base / f"{run_id}_fixed" / "train_log.jsonl")


def final_frequency(run_id, scaling=False):
    base = RUNS_SCALING if scaling else RUNS_FIXED
    rows = read_jsonl(base / f"{run_id}_fixed" / "freq_bin_loss.jsonl")
    return max(rows, key=lambda row: int(row["step"]))


def final_exact_frequency(run_id, scaling=False):
    base = RUNS_SCALING if scaling else RUNS_FIXED
    rows = read_jsonl(base / f"{run_id}_fixed" / "exact_freq_loss.jsonl")
    return max(rows, key=lambda row: int(row["step"]))


def fit_line(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) < 2:
        return {"n": len(x), "slope": "", "intercept": "", "r2": ""}
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - predicted) ** 2) / total if total else float("nan")
    return {"n": len(x), "slope": float(slope), "intercept": float(intercept), "r2": float(r2)}


def log_log_fit(x_values, y_values):
    usable = [(x, y) for x, y in zip(x_values, y_values) if x > 0 and y > 0]
    fit = fit_line(np.log([x for x, _ in usable]), np.log([y for _, y in usable]))
    fit["x_transform"] = "ln(x)"
    fit["y_transform"] = "ln(y)"
    return fit


def rank_correlation(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    if len(x) < 2:
        return float("nan")
    x_rank = np.argsort(np.argsort(x)).astype(float)
    y_rank = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def log_at_steps(run_id, targets):
    values = {int(row["step"]): row for row in run_rows(run_id, scaling=True)}
    output = {}
    for target in targets:
        row = values.get(int(target))
        output[int(target)] = row.get("gap", "") if row else ""
    return output


def frequency_gap(record, branch, bucket):
    train = record["train"][branch].get(bucket, {})
    val = record["val"][branch].get(bucket, {})
    if not train or not val or train.get("token_count", 0) <= 0:
        return ""
    return float(val["mean_loss"]) - float(train["mean_loss"])


def exact_rows(record, branch):
    rows = []
    for frequency_text, values in record.get("shared", {}).get(branch, {}).get("per_f", {}).items():
        frequency = int(frequency_text)
        shared_contexts = int(values.get("shared_contexts", 0))
        gap = values.get("gap")
        if frequency <= 0 or shared_contexts <= 0 or gap is None:
            continue
        rows.append(
            {
                "f": frequency,
                "gap": float(gap),
                "weight": frequency * shared_contexts,
                "contexts": shared_contexts,
            }
        )
    return rows


def pooled_exact_rows(rows, bins=7):
    usable = [row for row in rows if row["contexts"] >= 32 and np.isfinite(row["gap"])]
    if not usable:
        return []
    frequencies = np.asarray([row["f"] for row in usable], dtype=float)
    bounds = np.unique(np.geomspace(frequencies.min(), frequencies.max() + 1, bins + 1).astype(int))
    pooled = []
    for low, high in zip(bounds[:-1], bounds[1:]):
        members = [row for row in usable if low <= row["f"] < high]
        if not members:
            continue
        weight = sum(row["weight"] for row in members)
        pooled.append(
            {
                "f_low": low,
                "f_high": high - 1,
                "f_mid": math.sqrt(low * max(low, high - 1)),
                "gap": sum(row["gap"] * row["weight"] for row in members) / weight,
                "contexts": sum(row["contexts"] for row in members),
                "per_f_count": len(members),
            }
        )
    return pooled


def analyze_table_size():
    index_path = ROOT / "data" / "freq_index.npz"
    with np.load(index_path) as index:
        context_counts = {
            "bigram": int(index["bigram_keys"].size),
            "trigram": int(index["trigram_keys"].size),
        }
    rows = []
    fit_rows = []
    for branch, prefix, fixed_branch in (
        ("bigram", "s1v5_128_tbl_bi1_R", "disabled"),
        ("trigram", "s1v5_128_tbl_tri1_R", "disabled"),
    ):
        branch_rows = []
        for path in sorted(RUNS_SCALING.glob(f"{prefix}*_fixed")):
            run_id = path.name.removesuffix("_fixed")
            physical_rows = int(run_id.removeprefix(prefix))
            current = summary(run_id, scaling=True)
            checkpoints = log_at_steps(run_id, (337, 674, 1000))
            row = {
                "axis": "table_size",
                "branch": branch,
                "run_id": run_id,
                "seed": current["seed"],
                "step": current["steps"],
                "R": physical_rows,
                "K": context_counts[branch],
                "K_over_R": context_counts[branch] / physical_rows,
                "collision_rate": "",
                "final_train_loss": current["final_train_loss"],
                "final_val_loss": current["final_val_loss"],
                "final_gap": current["final_gap"],
                "gap_337": checkpoints[337],
                "gap_674": checkpoints[674],
                "gap_1000": checkpoints[1000],
                "freq_index_sha256": current["freq_index_sha256"],
                "source_code_revision": SOURCE_REVISION,
                "source": f"data/runs_scaling/{run_id}_fixed",
            }
            rows.append(row)
            branch_rows.append((physical_rows, current["final_gap"]))
        branch_rows.sort()
        fit = log_log_fit(
            [x for x, y in branch_rows],
            [y for x, y in branch_rows],
        )
        fit.update(
            {
                "family": "table_size",
                "branch": branch,
                "x_name": "physical rows R",
                "y_name": "final gap at step 1000",
                "rank_corr_R_gap": rank_correlation(
                    [x for x, _ in branch_rows],
                    [y for _, y in branch_rows],
                ),
                "K": context_counts[branch],
                "collision_rate_status": "not measured; do not infer from K/R",
                "fixed_branch": fixed_branch,
            }
        )
        fit_rows.append(fit)
    return rows, fit_rows


def analyze_epoch():
    specs = [
        (0.125, 42, "s1v5_128_ep_tri_0p125xL4_3ep"),
        (0.1667, 56, "s1v5_128_ep_tri_0p1667xL4_3ep"),
        (0.25, 84, "s1v5_128_ep_tri_0p25xL4_3ep"),
        (0.3333, 112, "s1v5_128_ep_tri_0p3333xL4_3ep"),
        (0.5, 168, "s1v5_128_ep_tri_0p5xL4_3ep"),
        (0.6667, 224, "s1v5_128_ep_tri_0p6667xL4_3ep"),
        (0.75, 253, "s1v5_128_ep_tri_0p75xL4_3ep"),
        (1.0, 337, "s1v5_128_ep_tri_1p0xL4_3ep"),
        (1.25, 421, "s1v5_128_ep_tri_1p25xL4_3ep"),
        (1.5, 506, "s1v5_128_ep_tri_1p5xL4_3ep"),
        (1.75, 590, "s1v5_128_ep_tri_1p75xL4_3ep"),
        (2.0, 674, "s1v5_128_ep_tri_2p0xL4_3ep"),
    ]
    rows = []
    multipliers = []
    gaps = []
    for multiplier, epoch_batches, run_id in specs:
        current = summary(run_id, scaling=True)
        checkpoint_steps = tuple(epoch_batches * epoch for epoch in (1, 2, 3))
        checkpoints = log_at_steps(run_id, checkpoint_steps)
        row = {
            "axis": "epoch_length",
            "run_id": run_id,
            "seed": current["seed"],
            "epoch_multiplier_L4": multiplier,
            "epoch_batches": epoch_batches,
            "target_steps": current["steps"],
            "actual_steps": current["steps"],
            "final_train_loss": current["final_train_loss"],
            "final_val_loss": current["final_val_loss"],
            "final_gap": current["final_gap"],
            "gap_epoch_1": checkpoints[checkpoint_steps[0]],
            "gap_epoch_2": checkpoints[checkpoint_steps[1]],
            "gap_epoch_3": checkpoints[checkpoint_steps[2]],
            "freq_index_sha256": current["freq_index_sha256"],
            "source_code_revision": SOURCE_REVISION,
            "source": f"data/runs_scaling/{run_id}_fixed",
        }
        rows.append(row)
        multipliers.append(multiplier)
        gaps.append(current["final_gap"])
    log_multiplier = np.log(np.asarray(multipliers))
    quadratic = np.polyfit(log_multiplier, np.asarray(gaps), 2)
    vertex_log = -quadratic[1] / (2 * quadratic[0])
    prediction = np.polyval(quadratic, log_multiplier)
    total = np.sum((np.asarray(gaps) - np.mean(gaps)) ** 2)
    r2 = 1 - np.sum((np.asarray(gaps) - prediction) ** 2) / total if total else float("nan")
    fit_rows = [
        {
            "family": "epoch_length",
            "branch": "trigram-only",
            "model": "quadratic gap vs ln(multiplier)",
            "n": len(rows),
            "quadratic_a": float(quadratic[0]),
            "quadratic_b": float(quadratic[1]),
            "quadratic_c": float(quadratic[2]),
            "estimated_min_multiplier_L4": float(np.exp(vertex_log)),
            "estimated_min_gap": float(np.polyval(quadratic, vertex_log)),
            "r2": float(r2),
            "rank_corr_multiplier_gap": rank_correlation(multipliers, gaps),
        }
    ]
    long_rows = []
    for label, run_id in (
        ("trigram-only", "s1v5_128_ep_tri_1xL4_10ep"),
        ("nogram", "s1v5_128_ep1xL4_10ep_nogram"),
    ):
        current = summary(run_id, scaling=True)
        values = {int(row["step"]): row for row in run_rows(run_id, scaling=True)}
        for epoch in range(1, 11):
            step = 337 * epoch
            row = values.get(step)
            if row is not None:
                long_rows.append(
                    {
                        "run_id": run_id,
                        "arm": label,
                        "epoch": epoch,
                        "step": step,
                        "train_loss": row["train_loss"],
                        "val_loss": row["val_loss"],
                        "gap": row["gap"],
                        "seed": current["seed"],
                        "source": f"data/runs_scaling/{run_id}_fixed",
                    }
                )
    return rows, fit_rows, long_rows


def analyze_dose():
    doses = (
        (0.25, "nglab0_25x_input_v5_freq10"),
        (0.5, "nglab0_5x_input_v5_freq10"),
        (0.75, "nglab0_75x_input_v5_freq10"),
        (1.0, "nglab1x_input_v5_freq10_r1"),
        (1.5, "nglab1_5x_input_v5_freq10"),
        (2.0, "nglab2x_input_v5_freq10"),
        (2.5, "nglab2_5x_input_v5_freq10"),
        (3.0, "nglab3x_input_v5_freq10"),
        (4.0, "nglab4x_input_v5_freq10"),
        (5.0, "nglab5x_input_v5_freq10"),
        (6.0, "nglab6x_input_v5_freq10"),
        (8.0, "nglab8x_input_v5_freq10"),
    )
    rows = []
    frequency_rows = []
    for dose, run_id in doses:
        current = summary(run_id)
        record = final_frequency(run_id)
        for branch in ("bigram", "trigram"):
            for bucket, values in record["train"][branch].items():
                frequency_rows.append(
                    {
                        "dose": dose,
                        "run_id": run_id,
                        "step": record["step"],
                        "branch": branch,
                        "bucket": bucket,
                        "train_token_count": values["token_count"],
                        "train_fraction": values["frac"],
                        "val_token_count": record["val"][branch][bucket]["token_count"],
                        "val_fraction": record["val"][branch][bucket]["frac"],
                        "gap": frequency_gap(record, branch, bucket),
                        "seed": current["seed"],
                        "source": f"data/runs_fixed/{run_id}_fixed",
                    }
                )
        config = current["config"]
        rows.append(
            {
                "axis": "dose",
                "run_id": run_id,
                "seed": current["seed"],
                "dose": dose,
                "steps": current["steps"],
                "final_train_loss": current["final_train_loss"],
                "final_val_loss": current["final_val_loss"],
                "final_gap": current["final_gap"],
                "train_shards": ",".join(map(str, config["train_shards"])),
                "freq_index_sha256": current["freq_index_sha256"],
                "source_code_revision": SOURCE_REVISION,
                "source": f"data/runs_fixed/{run_id}_fixed",
            }
        )
    positive_low = [row for row in rows if row["dose"] <= 5 and row["final_gap"] > 0]
    fit_rows = [dict(log_log_fit([row["dose"] for row in positive_low], [row["final_gap"] for row in positive_low]),
                     family="dose", branch="input", model="positive gap; dose <= 5x only",
                     rank_corr_dose_gap=rank_correlation(
                         [row["dose"] for row in positive_low],
                         [row["final_gap"] for row in positive_low],
                     ),
                     sign_change_bracket="5x to 6x")]
    return rows, frequency_rows, fit_rows


def analyze_frequency():
    run_id = "s1v5_128_frequency_main"
    current = summary(run_id, scaling=True)
    record = final_exact_frequency(run_id, scaling=True)
    point_rows = []
    fit_rows = []
    for branch in ("bigram", "trigram"):
        exact = exact_rows(record, branch)
        pooled = pooled_exact_rows(exact)
        for row in exact:
            point_rows.append(
                {
                    "run_id": run_id,
                    "branch": branch,
                    "step": record["step"],
                    "f": row["f"],
                    "gap": row["gap"],
                    "shared_token_mass": row["weight"],
                    "shared_contexts": row["contexts"],
                    "seed": current["seed"],
                    "source": f"data/runs_scaling/{run_id}_fixed",
                }
            )
        positive = [row for row in pooled if row["gap"] > 0]
        fit = log_log_fit(
            [row["f_mid"] for row in positive],
            [row["gap"] for row in positive],
        )
        fit.update(
            {
                "family": "frequency_exact",
                "branch": branch,
                "model": "token-mass-weighted geometric-bin fit; positive gap only",
                "step": record["step"],
                "run_id": run_id,
                "rank_corr_f_gap": rank_correlation(
                    [row["f_mid"] for row in positive],
                    [row["gap"] for row in positive],
                ),
                "source": f"data/runs_scaling/{run_id}_fixed/exact_freq_loss.jsonl",
            }
        )
        fit_rows.append(fit)
    return point_rows, fit_rows


def analyze_optimizer():
    groups = {
        "table_lr_scale": (
            "optv5c_rms_b099_s0p5",
            "optv5c_rms_b099_s1p0",
            "optv5c_rms_b099_s2p0_r1",
            "optv5c_rms_b099_s3p0",
            "optv5c_rms_b099_s4p0",
        ),
        "beta2": (
            "optv5c_rms_b095_s2p0",
            "optv5c_rms_b098_s2p0",
            "optv5c_rms_b099_s2p0_r1",
            "optv5c_rms_b0995_s2p0",
            "optv5c_rms_b0999_s2p0",
        ),
        "optimizer": (
            "optv5c_rms_b099_s2p0_r1",
            "optv5c_adamw_b099_s2p0",
            "optv5c_sgd_m0_s2p0",
        ),
    }
    rows = []
    for group, run_ids in groups.items():
        for run_id in run_ids:
            current = summary(run_id)
            rows.append(
                {
                    "family": group,
                    "run_id": run_id,
                    "seed": current["seed"],
                    "steps": current["steps"],
                    "final_train_loss": current["final_train_loss"],
                    "final_val_loss": current["final_val_loss"],
                    "final_gap": current["final_gap"],
                    "table_optimizer": current["config"]["table_optimizer"],
                    "table_betas": ",".join(map(str, current["config"]["table_betas"])),
                    "table_lr_scale": current["config"]["table_lr_scale"],
                    "source": f"data/runs_fixed/{run_id}_fixed",
                }
            )
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table_rows, table_fits = analyze_table_size()
    epoch_rows, epoch_fits, long_rows = analyze_epoch()
    dose_rows, dose_frequency_rows, dose_fits = analyze_dose()
    frequency_rows, frequency_fits = analyze_frequency()
    optimizer_rows = analyze_optimizer()
    write_csv(
        OUT / "s1_table_size_points.csv",
        table_rows,
        [
            "axis", "branch", "run_id", "seed", "step", "R", "K", "K_over_R",
            "collision_rate", "final_train_loss", "final_val_loss", "final_gap",
            "gap_337", "gap_674", "gap_1000", "freq_index_sha256",
            "source_code_revision", "source",
        ],
    )
    write_csv(
        OUT / "s1_epoch_length_points.csv",
        epoch_rows,
        [
            "axis", "run_id", "seed", "epoch_multiplier_L4", "epoch_batches",
            "target_steps", "actual_steps", "final_train_loss", "final_val_loss",
            "final_gap", "gap_epoch_1", "gap_epoch_2", "gap_epoch_3",
            "freq_index_sha256", "source_code_revision", "source",
        ],
    )
    write_csv(
        OUT / "s1_epoch_long_replay_points.csv",
        long_rows,
        ["run_id", "arm", "epoch", "step", "train_loss", "val_loss", "gap", "seed", "source"],
    )
    write_csv(
        OUT / "s1_dose_points.csv",
        dose_rows,
        [
            "axis", "run_id", "seed", "dose", "steps", "final_train_loss",
            "final_val_loss", "final_gap", "train_shards", "freq_index_sha256",
            "source_code_revision", "source",
        ],
    )
    write_csv(
        OUT / "s1_dose_frequency_gap.csv",
        dose_frequency_rows,
        [
            "dose", "run_id", "step", "branch", "bucket", "train_token_count",
            "train_fraction", "val_token_count", "val_fraction", "gap", "seed", "source",
        ],
    )
    write_csv(
        OUT / "s1_frequency_exact_points.csv",
        frequency_rows,
        [
            "run_id", "branch", "step", "f", "gap", "shared_token_mass",
            "shared_contexts", "seed", "source",
        ],
    )
    write_csv(
        OUT / "v5_optimizer_points.csv",
        optimizer_rows,
        [
            "family", "run_id", "seed", "steps", "final_train_loss", "final_val_loss",
            "final_gap", "table_optimizer", "table_betas", "table_lr_scale", "source",
        ],
    )
    fit_rows = table_fits + epoch_fits + dose_fits + frequency_fits
    write_csv(
        OUT / "s1_scaling_fits.csv",
        fit_rows,
        sorted({key for row in fit_rows for key in row}),
    )
    summary_lines = [
        "# V5 scaling statistics",
        "",
        f"- Source revision for the training batch: `{SOURCE_REVISION}`.",
        "- Every row below is a single seed-42 run; these are descriptive fits, not uncertainty intervals.",
        "- Gap is fixed validation loss minus the same-step current-batch online train loss.",
        "",
        "## Table-size axis",
    ]
    for fit in table_fits:
        summary_lines.append(
            f"- {fit['branch']}: log-log slope `{fit['slope']:.6f}`, "
            f"R² `{fit['r2']:.6f}`, rank correlation `{fit['rank_corr_R_gap']:.6f}`, "
            f"n=`{fit['n']}`; K=`{fit['K']}`, and K/R is a load ratio only. "
            "Collision rate was not measured."
        )
    epoch_fit = epoch_fits[0]
    summary_lines.extend(
        [
            "",
            "## Epoch-length axis",
            f"- Quadratic descriptive fit in ln(L4 multiplier): "
            f"vertex `{epoch_fit['estimated_min_multiplier_L4']:.6f}×L4`, "
            f"predicted gap `{epoch_fit['estimated_min_gap']:.6f}`, "
            f"R² `{epoch_fit['r2']:.6f}`; this summarizes the observed U-shape and is not a mechanistic law.",
            "",
            "## Dose axis",
        ]
    )
    dose_fit = dose_fits[0]
    summary_lines.append(
        f"- Positive-gap points through 5× only: log-log slope `{dose_fit['slope']:.6f}`, "
        f"R² `{dose_fit['r2']:.6f}`, rank correlation `{dose_fit['rank_corr_dose_gap']:.6f}`; "
        "the sign change is bracketed between 5× and 6×, so no global power law is reported."
    )
    summary_lines.extend(
        [
            "",
            "## Exact-frequency axis",
        ]
    )
    for fit in frequency_fits:
        summary_lines.append(
            f"- {fit['branch']}: token-mass-weighted geometric-bin log-log slope "
            f"`{fit['slope']:.6f}`, R² `{fit['r2']:.6f}`, rank correlation "
            f"`{fit['rank_corr_f_gap']:.6f}`, n=`{fit['n']}`; "
            "positive-gap bins only, with exact-f points retained in the CSV."
        )
    summary_lines.extend(
        [
            "",
            "## Files",
            "- `s1_table_size_points.csv`: one row per formal bigram-R/trigram-R run.",
            "- `s1_epoch_length_points.csv`: one row per formal 3-epoch point.",
            "- `s1_epoch_long_replay_points.csv`: epoch-boundary values for both/no-gram 10-epoch replay.",
            "- `s1_dose_points.csv`: final dose endpoints.",
            "- `s1_dose_frequency_gap.csv`: raw final frequency-bin rows, including token fractions.",
            "- `s1_frequency_exact_points.csv`: shared-context exact-f rows from the formal frequency run.",
            "- `s1_scaling_fits.csv`: descriptive fit coefficients and rank correlations.",
            "- `v5_optimizer_points.csv`: final metrics for the clean 11-arm optimizer refresh.",
        ]
    )
    (OUT / "s1_scaling_analysis.md").write_text("\n".join(summary_lines) + "\n")
    print(OUT / "s1_scaling_analysis.md")


if __name__ == "__main__":
    main()