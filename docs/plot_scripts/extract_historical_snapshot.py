#!/usr/bin/env python3
"""Freeze the legacy interactive-figure data before retiring standalone HTML.

The source pages are historical generated artifacts whose original run folders are
no longer present in this repository. This one-way extractor keeps only the data
needed to reproduce their seven Plotly charts in the consolidated report.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SOURCE_FILES = {
    "injection": "fig_gap_loss.html",
    "norm": "fig_loss_norm.html",
    "frequency": "fig_gap_by_freq.html",
    "distribution": "fig_hitcount_dist.html",
    "log_frequency": "fig_gap_vs_frequency_log.html",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def extract_json_variable(text: str, variable: str) -> object:
    match = re.search(rf"\bvar\s+{re.escape(variable)}\s*=\s*(.*?);\s*\n", text, re.S)
    require(match is not None, f"missing JSON variable {variable}")
    return json.loads(match.group(1))


def extract_trace_array(text: str, variable: str) -> list[dict]:
    match = re.search(rf"\bvar\s+{re.escape(variable)}\s*=\s*(.*?);\s*\n", text, re.S)
    require(match is not None, f"missing trace variable {variable}")
    traces = []
    for x_text, y_text, name in re.findall(
        r"x:\s*(\[.*?\]),\s*y:\s*(\[.*?\]),\s*mode:\s*\"[^\"]*\",\s*"
        r"name:\s*\"([^\"]+)\"",
        match.group(1),
        re.S,
    ):
        traces.append({"x": json.loads(x_text), "y": json.loads(y_text), "name": name})
    require(bool(traces), f"no traces extracted from {variable}")
    return traces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path, required=True,
        help="directory containing the five retired standalone HTML source pages",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("docs/data/historical-figures.json")
    )
    args = parser.parse_args()

    sources = {}
    for key, filename in SOURCE_FILES.items():
        path = args.source_dir / filename
        require(path.is_file() and path.stat().st_size > 0, f"missing source {path}")
        sources[key] = path.read_text(encoding="utf-8")

    snapshot = {
        "schema_version": 1,
        "provenance": {
            "description": "Frozen data extracted from the retired standalone historical Plotly pages.",
            "source_files": [f"docs/figs/{name}" for name in SOURCE_FILES.values()],
        },
        "charts": {
            "injection_gap": {
                "traces": extract_json_variable(sources["injection"], "gapData")
            },
            "injection_loss": {
                "traces": extract_json_variable(sources["injection"], "lossData")
            },
            "table_norm": {
                "traces": extract_trace_array(sources["norm"], "normTraces")
            },
            "input_alignment": {
                "traces": extract_trace_array(sources["norm"], "lossTraces")
            },
            "frequency_bins": {
                "series": extract_json_variable(sources["frequency"], "series")
            },
            "hitcount_distribution": {
                "series": extract_json_variable(sources["distribution"], "gapData")
            },
            "gap_vs_frequency_log": {
                "series": extract_json_variable(sources["log_frequency"], "logData")
            },
        },
    }
    require(len(snapshot["charts"]) == 7, "historical snapshot must contain seven charts")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out} with {len(snapshot['charts'])} historical charts")


if __name__ == "__main__":
    main()
