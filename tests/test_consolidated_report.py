import json
import math
import re
import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPORT = DOCS / "frequency-gap-by-hit-count.html"
STAGE2A_RUN_IDS = {
    "nglab_s2a_b0500_lr0250_s42",
    "nglab_s2a_b0500_lr0500_s42",
    "nglab_s2a_b0500_lr1000_s42",
    "nglab_s2a_b0900_lr0250_s42",
    "nglab_s2a_b0900_lr0500_s42",
    "nglab_s2a_b0900_lr1000_s42",
    "nglab_s2a_b0999_lr0000_s42_r1",
    "nglab_s2a_b0999_lr0125_s42",
    "nglab_s2a_b0999_lr0250_s42",
    "nglab_s2a_b0999_lr0375_s42",
    "nglab_s2a_b0999_lr0625_s42",
    "nglab_s2a_b0999_lr0750_s42",
    "nglab_s2a_b0999_lr0875_s42",
}
STAGE3R_RUN_IDS = {
    "nglab_s3r2_sequential_s42",
    "nglab_s3r2_reshuffle_s42_p101",
    "nglab_s3r3_random_frozen_s42_p101",
    "nglab_s3r3_random_reshuffle_s42_p101",
}
FREQUENCY_MASK_LABELS = (
    "none", "0", "1", "2", "5", "6", "7", "8", "9", "10", "11", "12",
    "13", "14", "15", "16", "17", "18", "19", "20", "25", "30", "35",
    "40", "45", "50", "60", "70", "80", "90", "100", "110", "120",
    "130", "140", "150", "160", "170", "180", "190", "200", "210",
    "500", "1k", "2k", "5k", "20k", "100k", "all",
)


class IdCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.append(attributes["id"])


class ConsolidatedReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = REPORT.read_text(encoding="utf-8")

    def test_docs_contains_exactly_one_html(self):
        html_files = sorted(path.relative_to(DOCS).as_posix() for path in DOCS.rglob("*.html"))
        self.assertEqual(html_files, ["frequency-gap-by-hit-count.html"])

    def test_report_has_one_plotly_import_and_unique_ids(self):
        self.assertEqual(self.document.count("cdn.plot.ly/plotly-2.27.0.min.js"), 1)
        parser = IdCollector()
        parser.feed(self.document)
        ids = parser.ids
        self.assertEqual(len(ids), len(set(ids)))
        for anchor in (
            "historical-injection", "historical-frequency", "rmsprop-stage1-results",
            "rmsprop-stage2a-results", "reshuffle-stage3r-results",
            "frequency-mask-sweep",
        ):
            self.assertIn(anchor, ids)
        for chart_id in (
            "stage2a-gap-lr-chart",
            "stage2a-probe1-lr-freq-chart",
            "stage2a-probe2-lr-freq-chart",
            "stage3r-full-online-loss-chart",
            "stage3r-global-gap-chart",
            "frequency-mask-gap-chart",
        ):
            self.assertIn(chart_id, ids)

    def test_stage_selector_and_result_table_sizes(self):
        selector = re.search(
            r'<select id="stage-run-select">(.*?)</select>', self.document, re.S
        )
        self.assertIsNotNone(selector)
        self.assertEqual(selector.group(1).count("<option"), 9)

        convergence = re.search(
            r'<table class="result-table" id="stage-convergence-table">.*?'
            r'<tbody>(.*?)</tbody>', self.document, re.S
        )
        edge = re.search(
            r'<table class="result-table" id="stage-edge-table">.*?'
            r'<tbody>(.*?)</tbody>', self.document, re.S
        )
        self.assertIsNotNone(convergence)
        self.assertIsNotNone(edge)
        self.assertEqual(convergence.group(1).count("<tr"), 9)
        self.assertEqual(edge.group(1).count("<tr"), 18)
        self.assertIn("+0.6866", convergence.group(1))
        self.assertIn("+0.9154", convergence.group(1))

    def test_stage2a_selector_payload_and_result_table_sizes(self):
        selector = re.search(
            r'<select id="stage2a-run-select">(.*?)</select>', self.document, re.S
        )
        self.assertIsNotNone(selector)
        options = dict(
            re.findall(r'<option value="([^"]+)"(?: selected)?>(.*?)</option>', selector.group(1))
        )
        self.assertEqual(set(options), STAGE2A_RUN_IDS)
        self.assertEqual(len(options), 13)
        self.assertIn("LR x0.125", options["nglab_s2a_b0999_lr0125_s42"])
        self.assertIn("LR x0.875", options["nglab_s2a_b0999_lr0875_s42"])

        convergence = re.search(
            r'<table class="result-table" id="stage2a-convergence-table">.*?'
            r'<tbody>(.*?)</tbody>', self.document, re.S
        )
        edge = re.search(
            r'<table class="result-table" id="stage2a-edge-table">.*?'
            r'<tbody>(.*?)</tbody>', self.document, re.S
        )
        self.assertIsNotNone(convergence)
        self.assertIsNotNone(edge)
        self.assertEqual(convergence.group(1).count("<tr"), 13)
        self.assertEqual(edge.group(1).count("<tr"), 26)
        self.assertIn("+0.0977", convergence.group(1))
        self.assertIn("+0.9608", convergence.group(1))
        self.assertIn("0.000500", convergence.group(1))
        for run_id in STAGE2A_RUN_IDS:
            self.assertEqual(convergence.group(1).count(f'data-run-id="{run_id}"'), 1)
            self.assertEqual(edge.group(1).count(f'data-run-id="{run_id}"'), 2)
            self.assertIn(f'"runId":"{run_id}"', self.document)

    def test_stage3r_pair_selector_tables_and_payload(self):
        selector = re.search(
            r'<select id="stage3r-run-select">(.*?)</select>', self.document, re.S
        )
        self.assertIsNotNone(selector)
        options = dict(
            re.findall(r'<option value="([^"]+)"(?: selected)?>(.*?)</option>', selector.group(1))
        )
        self.assertEqual(set(options), STAGE3R_RUN_IDS)
        self.assertIn("epoch 1 original order · no shuffle", options["nglab_s3r2_sequential_s42"])
        self.assertIn("epoch 1 original order · shuffle", options["nglab_s3r2_reshuffle_s42_p101"])
        self.assertIn("epoch 1 random order · no shuffle", options["nglab_s3r3_random_frozen_s42_p101"])
        self.assertIn("epoch 1 random order · shuffle", options["nglab_s3r3_random_reshuffle_s42_p101"])

        convergence = re.search(
            r'<table class="result-table" id="stage3r-convergence-table">.*?'
            r'<tbody>(.*?)</tbody>', self.document, re.S
        )
        edge = re.search(
            r'<table class="result-table" id="stage3r-edge-table">.*?'
            r'<tbody>(.*?)</tbody>', self.document, re.S
        )
        self.assertIsNotNone(convergence)
        self.assertIsNotNone(edge)
        self.assertEqual(convergence.group(1).count("<tr"), 4)
        self.assertEqual(edge.group(1).count("<tr"), 8)
        self.assertIn("+0.9212", convergence.group(1))
        self.assertIn("+0.5727", convergence.group(1))
        self.assertIn("+1.2695", convergence.group(1))
        self.assertIn("+0.7882", convergence.group(1))
        self.assertIn("+0.2389", edge.group(1))
        self.assertIn("+0.4362", edge.group(1))
        self.assertIn("+0.3072", edge.group(1))
        self.assertIn("+0.5273", edge.group(1))
        for run_id in STAGE3R_RUN_IDS:
            self.assertEqual(convergence.group(1).count(f'data-run-id="{run_id}"'), 1)
            self.assertEqual(edge.group(1).count(f'data-run-id="{run_id}"'), 2)
            self.assertIn(f'"runId":"{run_id}"', self.document)

        match = re.search(
            r"const stage3rAnalysis = (\{.*?\});\nconst frequencyMaskAnalysis",
            self.document,
            re.S,
        )
        self.assertIsNotNone(match)
        analysis = json.loads(match.group(1))
        self.assertEqual(analysis["epochStarts"], [338, 675])
        self.assertEqual(len(analysis["onlineEdgeMetrics"]), 8)
        self.assertEqual(len(analysis["fixedGramEdgeChanges"]), 8)
        self.assertEqual(set(analysis["pairingAudit"]), {"original", "random"})
        self.assertEqual(analysis["pairingAudit"]["original"]["forkStep"], 337)
        self.assertEqual(analysis["pairingAudit"]["random"]["forkStep"], 337)
        self.assertTrue(analysis["pairingAudit"]["original"]["prefixRowsIdentical"])
        self.assertTrue(analysis["pairingAudit"]["random"]["prefixRowsIdentical"])
        self.assertEqual(
            analysis["pairingAudit"]["original"]["sharedParameterStateSha256"],
            "cc487a7acd2042f70d893d789bb7331ebbd5c303464dbe17496f0cdc8c0946ca",
        )
        self.assertEqual(
            analysis["pairingAudit"]["random"]["sharedParameterStateSha256"],
            "3a616d781269cf7879ce8fd1959a16f5bf1ccbb4406dfd77d9fb86797a831665",
        )
        self.assertEqual(self.document.count('class="stage3r-loss-toggle"'), 4)
        self.assertIn('id="stage3r-dense-sampling" checked', self.document)
        self.assertNotIn('id="stage3r-full-gap-chart"', self.document)
        self.assertNotIn('id="stage3r-online-edge-chart"', self.document)
        self.assertNotIn('id="stage3r-fixedgram-edge-chart"', self.document)
        self.assertIn('(step-1)%epochSteps===0', self.document)

    def test_report_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.html"
            command = [
                sys.executable,
                str(DOCS / "generate_report.py"),
                "--runs-root", str(ROOT / "data" / "runs"),
                "--historical-data", str(DOCS / "data" / "historical-figures.json"),
                "--out", str(output),
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            first = output.read_bytes()
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertEqual(first, output.read_bytes())
            self.assertEqual(first, REPORT.read_bytes())

    def test_lr_analysis_payload_and_source_values(self):
        match = re.search(
            r"const lrAnalysis = (\{.*?\});\nconst stage3rAnalysis", self.document, re.S
        )
        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))
        self.assertEqual(payload["epoch2Step"], 674)
        self.assertEqual(
            payload["probeWindows"],
            {
                "first": {"before": 164, "read_start": 169, "read_end": 172, "after": 174},
                "second": {"before": 501, "read_start": 506, "read_end": 509, "after": 511},
            },
        )
        self.assertEqual(set(payload["groups"]), {"0.500", "0.900", "0.999"})
        self.assertEqual(
            {key: len(group["points"]) for key, group in payload["groups"].items()},
            {"0.500": 3, "0.900": 3, "0.999": 9},
        )
        run_ids = set()
        for group in payload["groups"].values():
            learning_rates = [point["lrScale"] for point in group["points"]]
            self.assertEqual(learning_rates, sorted(learning_rates))
            self.assertEqual(len(learning_rates), len(set(learning_rates)))
            for point in group["points"]:
                self.assertTrue(math.isfinite(point["onlineGapStep674"]))
                self.assertNotIn(point["runId"], run_ids)
                run_ids.add(point["runId"])
                for read_name in ("first", "second"):
                    for branch in ("bigram", "trigram"):
                        buckets = point["probeRead"][read_name][branch]
                        self.assertEqual(len(buckets), 15)
                        for values in buckets.values():
                            self.assertTrue(all(math.isfinite(values[key]) for key in ("before", "after", "delta")))
                            self.assertAlmostEqual(values["delta"], values["after"] - values["before"])

        beta999 = {point["lrScale"]: point for point in payload["groups"]["0.999"]["points"]}
        self.assertEqual(beta999[0.5]["runId"], "nglab_rms_b0999_lr050_s1")
        self.assertEqual(beta999[1.0]["runId"], "nglab_baseline_input_midprobe_sparse_20260812")
        source_run = ROOT / "data" / "runs" / "nglab_s2a_b0999_lr0375_s42"
        online_rows = [json.loads(line) for line in (source_run / "online_frequency_gap_contribution.jsonl").read_text(encoding="utf-8").splitlines() if line]
        online = next(row for row in online_rows if row["step"] == 674)
        self.assertAlmostEqual(beta999[0.375]["onlineGapStep674"], online["online_val_loss"] - online["train_writer_loss"])
        fixed_rows = [json.loads(line) for line in (source_run / "fixed_probe_frequency_gap_contribution.jsonl").read_text(encoding="utf-8").splitlines() if line]
        before = next(row for row in fixed_rows if row["step"] == 164)
        after = next(row for row in fixed_rows if row["step"] == 174)
        expected = after["gap_contribution"]["bigram"]["1"]["contribution"] - before["gap_contribution"]["bigram"]["1"]["contribution"]
        self.assertAlmostEqual(beta999[0.375]["probeRead"]["first"]["bigram"]["1"]["delta"], expected)

    def test_frequency_mask_sweep_payload_and_source_values(self):
        match = re.search(
            r"const frequencyMaskAnalysis = (\{.*?\});\nconst buckets",
            self.document,
            re.S,
        )
        self.assertIsNotNone(match)
        payload = json.loads(match.group(1))
        self.assertEqual(payload["metric"], "online validation loss - writer train loss")
        self.assertEqual(payload["timing"], "pre_optimizer_step")
        self.assertEqual(
            payload["axis"],
            {"defaultScale": "log1p", "allEquivalentThreshold": 195964, "noneControlX": 0},
        )
        self.assertEqual(
            payload["epochEnds"],
            [{"epoch": 1, "step": 337}, {"epoch": 2, "step": 674}, {"epoch": 3, "step": 1011}],
        )
        points = payload["points"]
        self.assertEqual(len(points), 49)
        self.assertEqual(tuple(point["label"] for point in points), FREQUENCY_MASK_LABELS)
        self.assertEqual([point["ordinal"] for point in points], list(range(49)))
        self.assertEqual(len({point["runId"] for point in points}), 49)
        self.assertNotIn("nglab_freqmask_none_s42_h200360", {point["runId"] for point in points})
        self.assertEqual(points[0]["threshold"], None)
        self.assertIsNone(points[0]["functionX"])
        self.assertEqual(points[1]["threshold"], 0)
        self.assertEqual(points[1]["functionX"], 0)
        self.assertEqual(points[-1]["threshold"], "all")
        self.assertEqual(points[-1]["functionX"], 195964)
        self.assertEqual(
            len([
                point for point in points
                if point["functionX"] is not None and point["functionX"] <= 210
            ]),
            41,
        )
        for point in points:
            self.assertEqual(set(point["epochGaps"]), {"1", "2", "3"})
            self.assertTrue(all(math.isfinite(value) for value in point["epochGaps"].values()))
            self.assertGreater(point["wallSeconds"], 0)
            self.assertEqual(set(point["maskedOccurrenceFraction"]), {"bigram", "trigram"})
        self.assertEqual(points[0]["maskedOccurrenceFraction"], {"bigram": 0.0, "trigram": 0.0})
        self.assertEqual(points[-1]["maskedOccurrenceFraction"], {"bigram": 1.0, "trigram": 1.0})

        source_run = ROOT / "data" / "runs" / "nglab_freqmask_x000060_s42"
        source_rows = [
            json.loads(line)
            for line in (source_run / "online_gap.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        point60 = next(point for point in points if point["threshold"] == 60)
        for epoch, step in ((1, 337), (2, 674), (3, 1011)):
            source = next(row for row in source_rows if row["step"] == step)
            self.assertAlmostEqual(point60["epochGaps"][str(epoch)], source["gap"])
        audit = payload["bridgeAudit"]
        self.assertEqual(audit["primaryRunId"], "nglab_freqmask_none_s42")
        self.assertEqual(audit["bridgeRunId"], "nglab_freqmask_none_s42_h200360")
        self.assertAlmostEqual(
            audit["maxAbsoluteDelta"],
            max(abs(value) for value in audit["bridgeMinusPrimary"].values()),
        )
        self.assertIn('id="frequency-mask-x-scale"', self.document)
        self.assertIn('id="frequency-mask-y-view"', self.document)
        self.assertIn('value="absolute" selected>3 lines', self.document)
        self.assertIn('value="increment">2 lines', self.document)
        self.assertRegex(
            self.document,
            r'<input type="number" id="frequency-mask-x-max"[^>]*value="210"',
        )
        self.assertIn('continuous log10(x + 1)', self.document)
        self.assertIn('none control (not connected)', self.document)
        self.assertIn('row.epochGaps["2"]-row.epochGaps["1"]', self.document)
        self.assertIn('row.epochGaps["3"]-row.epochGaps["1"]', self.document)
        self.assertIn('row.functionX<=xMax', self.document)
        self.assertNotIn('categoryarray:labels', self.document)

    def test_historical_snapshot_covers_seven_charts(self):
        snapshot = json.loads(
            (DOCS / "data" / "historical-figures.json").read_text(encoding="utf-8")
        )
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(len(snapshot["provenance"]["source_files"]), 5)
        self.assertEqual(
            set(snapshot["charts"]),
            {
                "injection_gap", "injection_loss", "table_norm", "input_alignment",
                "frequency_bins", "hitcount_distribution", "gap_vs_frequency_log",
            },
        )

    def test_no_active_standalone_html_generator_remains(self):
        svg_generator = (DOCS / "plot_scripts" / "gen_all_figures.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(".html", svg_generator)
        for launcher in (ROOT / "code" / "cluster").glob("*.sh"):
            text = launcher.read_text(encoding="utf-8")
            self.assertNotIn("frequency-gap-by-hit-count-", text)


if __name__ == "__main__":
    unittest.main()
