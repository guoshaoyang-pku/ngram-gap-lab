#!/usr/bin/env python3
"""Exact finite-sample tests for the residual--response view of loss gap.

This task is deliberately independent of code/train.py and has no random draw:
every expectation is evaluated by enumerating a binomial distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


RUN_IDS = {
    "counttable": "l6_counttable_freq_exact_v1",
    "responses": "l6_response_moments_exact_v1",
}
FREQUENCIES = tuple(2**power for power in range(2, 13))


def binomial_weights(f: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(f + 1, dtype=float)
    log_w = np.array(
        [
            math.lgamma(f + 1)
            - math.lgamma(int(ki) + 1)
            - math.lgamma(f - int(ki) + 1)
            + ki * math.log(p)
            + (f - ki) * math.log1p(-p)
            for ki in k
        ],
        dtype=float,
    )
    weights = np.exp(log_w - log_w.max())
    weights /= weights.sum()
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    return k, weights


def local_slopes(rows: list[dict[str, float | str]], group_key: str) -> None:
    previous: dict[str, tuple[float, float]] = {}
    for row in rows:
        group = str(row[group_key])
        f = float(row["f"])
        gap = float(row["exact_gap"])
        if group in previous:
            prev_f, prev_gap = previous[group]
            row["local_slope"] = math.log(gap / prev_gap) / math.log(f / prev_f)
        else:
            row["local_slope"] = ""
        previous[group] = (f, gap)


def fitted_slopes(
    rows: list[dict[str, float | str]], group_key: str, fit_min_f: int
) -> dict[str, float]:
    groups = sorted({str(row[group_key]) for row in rows})
    answer = {}
    for group in groups:
        chosen = [
            row
            for row in rows
            if str(row[group_key]) == group and float(row["f"]) >= fit_min_f
        ]
        x = np.log([float(row["f"]) for row in chosen])
        y = np.log([float(row["exact_gap"]) for row in chosen])
        answer[group] = float(np.polyfit(x, y, 1)[0])
    return answer


def counttable_rows(alpha: float) -> tuple[list[dict[str, float | str]], dict]:
    rows: list[dict[str, float | str]] = []
    for p in (0.50, 0.20, 0.05):
        for f in FREQUENCIES:
            k, weights = binomial_weights(f, p)
            delta = k / f - p
            q = (k + alpha) / (f + 2.0 * alpha)
            learned_log_odds = np.log(q) - np.log1p(-q)
            exact_gap = float(np.sum(weights * delta * learned_log_odds))

            q0 = (f * p + alpha) / (f + 2.0 * alpha)
            response_scale = f / (f + 2.0 * alpha)
            g1 = 1.0 / (q0 * (1.0 - q0))
            g2 = -1.0 / q0**2 + 1.0 / (1.0 - q0) ** 2
            g3 = 2.0 / q0**3 + 2.0 / (1.0 - q0) ** 3
            m2 = float(np.sum(weights * delta**2))
            m3 = float(np.sum(weights * delta**3))
            m4 = float(np.sum(weights * delta**4))
            term2 = g1 * response_scale * m2
            term3 = 0.5 * g2 * response_scale**2 * m3
            term4 = (g3 / 6.0) * response_scale**3 * m4
            rows.append(
                {
                    "run_id": RUN_IDS["counttable"],
                    "p": p,
                    "f": f,
                    "exact_gap": exact_gap,
                    "variance_term": term2,
                    "third_moment_term": term3,
                    "fourth_moment_term": term4,
                    "through_fourth": term2 + term3 + term4,
                    "local_slope": "",
                }
            )
    local_slopes(rows, "p")
    fit_min_f = 512
    summary = {
        "run_id": RUN_IDS["counttable"],
        "status": "done",
        "method": "exact binomial enumeration",
        "seed": None,
        "alpha": alpha,
        "fit_min_f": fit_min_f,
        "fitted_slopes": fitted_slopes(rows, "p", fit_min_f),
        "claim_boundary": (
            "The -1 exponent is an asymptotic resolved-count-table result; "
            "the variance contribution is only the leading response-expansion term."
        ),
    }
    return rows, summary


def response_rows() -> tuple[list[dict[str, float | str]], dict]:
    rows: list[dict[str, float | str]] = []
    response_specs = (
        ("linear", lambda delta: delta, lambda f: 1.0 / f),
        ("sign", np.sign, lambda f: math.sqrt(2.0 / (math.pi * f))),
        ("cubic", lambda delta: delta**3, lambda f: 3.0 / f**2 - 2.0 / f**3),
    )
    for response, response_fn, theory_fn in response_specs:
        for f in tuple(value for value in FREQUENCIES if value >= 8):
            k, weights = binomial_weights(f, 0.5)
            delta = (2.0 * k - f) / f
            exact_gap = float(np.sum(weights * delta * response_fn(delta)))
            rows.append(
                {
                    "run_id": RUN_IDS["responses"],
                    "response": response,
                    "f": f,
                    "exact_gap": exact_gap,
                    "theory_reference": theory_fn(f),
                    "local_slope": "",
                }
            )
            if response == "linear":
                assert abs(exact_gap - 1.0 / f) < 2e-12
            if response == "cubic":
                assert abs(exact_gap - theory_fn(f)) < 2e-12
    rows.sort(key=lambda row: (str(row["response"]), int(row["f"])))
    local_slopes(rows, "response")
    fit_min_f = 128
    summary = {
        "run_id": RUN_IDS["responses"],
        "status": "done",
        "method": "exact Rademacher/binomial enumeration",
        "seed": None,
        "fit_min_f": fit_min_f,
        "fitted_slopes": fitted_slopes(rows, "response", fit_min_f),
        "claim_boundary": (
            "The exponent depends on the learned response to the sampling residual: "
            "linear, sign, and cubic responses select different moments."
        ),
    }
    return rows, summary


def write_result(output_root: Path, run_id: str, config: dict, rows: list[dict], summary: dict) -> Path:
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(RUN_IDS), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    if args.experiment == "counttable":
        rows, summary = counttable_rows(args.alpha)
        config = {
            "run_id": RUN_IDS["counttable"],
            "experiment": "counttable",
            "probabilities": [0.50, 0.20, 0.05],
            "frequencies": list(FREQUENCIES),
            "alpha": args.alpha,
            "enumeration": "binomial exact",
            "seed": None,
        }
    else:
        rows, summary = response_rows()
        config = {
            "run_id": RUN_IDS["responses"],
            "experiment": "responses",
            "responses": ["linear", "sign", "cubic"],
            "frequencies": [value for value in FREQUENCIES if value >= 8],
            "enumeration": "Rademacher/binomial exact",
            "seed": None,
        }
    output_dir = write_result(args.output_root, config["run_id"], config, rows, summary)
    print(output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
