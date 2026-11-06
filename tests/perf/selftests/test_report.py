"""
Tests for `tests.perf.report`.
"""

from __future__ import annotations

import io
import json
import re
import tempfile
import unittest
import urllib.parse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from tests.perf.report.page import main, render_page, version_label
from tests.perf.report.results import (
    COMPARISON_COLUMNS,
    BenchmarkMeasurement,
    BenchmarkRunResults,
    ConfidenceInterval,
)

# ------------------------------------------------------------------------------ results


MEASUREMENT_TABLE_COLUMNS = [
    "measurement_id",
    "group",
    "benchmark_name",
    "target",
    "label",
    "arguments",
    "tier",
    "variant",
    "timing_mode",
    "observation_count",
    "workload_id",
    "mean_seconds",
    "mean_seconds_low",
    "mean_seconds_high",
    "median_seconds",
    "median_seconds_low",
    "median_seconds_high",
    "stddev_seconds",
    "stddev_seconds_low",
    "stddev_seconds_high",
    "median_absolute_deviation_seconds",
    "median_absolute_deviation_seconds_low",
    "median_absolute_deviation_seconds_high",
    "observations_per_second",
    "peak_memory_bytes",
]


def a_payload(entry_count: int = 1) -> dict:
    """
    A pytest-benchmark payload with fully stamped `extra_info`.

    Parameters
    ----------
    entry_count : int
        How many benchmark entries the payload should hold.

    Returns
    -------
    dict
        A payload of the shape `--benchmark-json` writes.
    """
    return {
        "datetime": "2026-01-01T00:00:00",
        "machine_info": {
            "node": "somewhere",
            "cpu": {"vendor": "x"},
            "macrosynergy": {"hardware_id": "abcd1234"},
        },
        "benchmarks": [
            {
                "fullname": f"t.py::test_target[{index}]",
                "name": f"test_target[{index}]",
                "options": {"min_rounds": 5},
                "stats": {
                    "data": [0.10, 0.11, 0.09, 0.105, 0.095],
                    "mean": 0.1,
                    "rounds": 5,
                },
                "extra_info": {
                    "group": "qdf",
                    "benchmark_name": f"target_{index}",
                    "label": "Measure a target.",
                    "target": f"package.module:target_{index}",
                    "call": {"df": "DataFrame 3,000×4"},
                    "variant": {"dtype": "categorical"},
                    "timing_mode": "calibrated",
                    "panel_size": {"tier": "tiny", "observation_count": 3000},
                    "peak_memory_bytes": 1024,
                },
            }
            for index in range(entry_count)
        ],
    }


def leaves(node, prefix: str = "") -> dict:
    """
    Every scalar in a nested payload, keyed by its dotted path.

    Parameters
    ----------
    node : Any
        A payload, or any part of one.
    prefix : str
        Dotted path of `node` within the payload it came from.

    Returns
    -------
    dict
        Scalar value per dotted path.
    """
    found = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if not isinstance(value, (dict, list)):
                found[prefix + key] = value
            found.update(leaves(value, prefix + key + "."))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.update(leaves(value, f"{prefix}{index}."))
    return found


class TestConfidenceInterval(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [1.0, 1.1, 0.9, 1.05, 0.95]

    def test_interval_contains_its_point_value(self):
        interval = ConfidenceInterval.from_samples(self.samples, np.mean)
        self.assertLessEqual(interval.low, interval.value)
        self.assertLessEqual(interval.value, interval.high)

    def test_interval_is_reproducible_from_the_seed(self):
        first = ConfidenceInterval.from_samples(self.samples, np.mean)
        self.assertEqual(first, ConfidenceInterval.from_samples(self.samples, np.mean))

    def test_a_single_sample_has_no_bounds(self):
        interval = ConfidenceInterval.from_samples([0.5], np.mean)
        self.assertEqual(
            (interval.value, interval.low, interval.high), (0.5, None, None)
        )


class TestBenchmarkMeasurement(unittest.TestCase):
    def test_identifier_ignores_the_pytest_node_id(self):
        entry = a_payload()["benchmarks"][0]
        original = BenchmarkMeasurement.from_payload_entry(entry).identifier
        entry["fullname"] = "elsewhere.py::renamed[9]"
        entry["name"] = "renamed[9]"
        self.assertEqual(
            BenchmarkMeasurement.from_payload_entry(entry).identifier, original
        )
        self.assertEqual(original, "qdf/target_0/tiny/dtype=categorical")

    def test_retuning_a_tier_keeps_the_identifier_but_moves_the_workload_id(self):
        before = BenchmarkMeasurement.from_payload_entry(a_payload()["benchmarks"][0])
        entry = a_payload()["benchmarks"][0]
        entry["extra_info"]["panel_size"]["observation_count"] = 6000
        after = BenchmarkMeasurement.from_payload_entry(entry)
        self.assertEqual(after.identifier, before.identifier)
        self.assertNotEqual(after.workload_id, before.workload_id)

    def test_changing_the_call_moves_the_workload_id(self):
        before = BenchmarkMeasurement.from_payload_entry(a_payload()["benchmarks"][0])
        entry = a_payload()["benchmarks"][0]
        entry["extra_info"]["call"]["ticks"] = "list[6]"
        after = BenchmarkMeasurement.from_payload_entry(entry)
        self.assertEqual(after.identifier, before.identifier)
        self.assertNotEqual(after.workload_id, before.workload_id)

    def test_runs_recorded_without_a_call_still_load(self):
        entry = a_payload()["benchmarks"][0]
        for key in ("label", "target", "call"):
            del entry["extra_info"][key]
        measurement = BenchmarkMeasurement.from_payload_entry(entry)
        self.assertIsNone(measurement.target)
        self.assertEqual(measurement.arguments, "")

    def test_a_measurement_requires_its_metadata(self):
        entry = a_payload()["benchmarks"][0]
        del entry["extra_info"]["group"]
        with self.assertRaisesRegex(ValueError, "extra_info is missing"):
            BenchmarkMeasurement.from_payload_entry(entry)

    def test_throughput_divides_observations_by_the_mean(self):
        measurement = BenchmarkMeasurement.from_payload_entry(
            a_payload()["benchmarks"][0]
        )
        self.assertAlmostEqual(
            measurement.observations_per_second, 3000 / 0.1, places=6
        )


class TestBenchmarkRunResults(unittest.TestCase):
    def test_measurement_table_has_one_row_per_measurement(self):
        table = BenchmarkRunResults(a_payload(3)).measurement_table()
        self.assertEqual(list(table.columns), MEASUREMENT_TABLE_COLUMNS)
        self.assertEqual(len(table), 3)
        self.assertEqual(
            list(table["measurement_id"]),
            [f"qdf/target_{index}/tiny/dtype=categorical" for index in range(3)],
        )

    def test_measurement_table_bounds_bracket_their_statistic(self):
        row = BenchmarkRunResults(a_payload()).measurement_table().iloc[0]
        for name in ("mean_seconds", "median_seconds", "stddev_seconds"):
            self.assertLessEqual(row[f"{name}_low"], row[name])
            self.assertLessEqual(row[name], row[f"{name}_high"])
        self.assertEqual(row["observation_count"], 3000)
        self.assertEqual(row["peak_memory_bytes"], 1024)

    def test_derived_statistics_preserve_every_original_key(self):
        payload = a_payload(2)
        before = deepcopy(payload)
        after = BenchmarkRunResults(payload).with_derived_statistics()
        original, enriched = leaves(before), leaves(after)
        self.assertLessEqual(set(original), set(enriched))
        for key, value in original.items():
            self.assertEqual(enriched[key], value)

    def test_derived_statistics_come_from_the_measurement_table(self):
        results = BenchmarkRunResults(a_payload(2))
        expected = results.measurement_table().to_dict("records")
        enriched = results.with_derived_statistics()
        for entry, row in zip(enriched["benchmarks"], expected):
            self.assertEqual(set(entry["macrosynergy"]), set(row))
            self.assertEqual(
                entry["macrosynergy"]["measurement_id"], row["measurement_id"]
            )

    def test_derived_statistics_are_json_serialisable(self):
        enriched = BenchmarkRunResults(a_payload(2)).with_derived_statistics()
        self.assertIsInstance(json.dumps(enriched), str)

    def test_derived_statistics_record_the_run_block(self):
        enriched = BenchmarkRunResults(a_payload()).with_derived_statistics()
        run = enriched["macrosynergy"]
        self.assertEqual(run["run_id"], "2026-01-01T00:00:00")
        self.assertEqual(run["tiers_measured"], ["tiny"])
        self.assertEqual(run["bootstrap"]["confidence"], 0.95)

    def test_derived_statistics_are_idempotent(self):
        payload = a_payload()
        once = BenchmarkRunResults(payload).with_derived_statistics()
        twice = BenchmarkRunResults(deepcopy(once)).with_derived_statistics()
        self.assertEqual(
            json.dumps(once, sort_keys=True), json.dumps(twice, sort_keys=True)
        )

    def test_round_timings_can_be_dropped(self):
        after = BenchmarkRunResults(a_payload()).with_derived_statistics(
            drop_round_timings=True
        )
        self.assertNotIn("data", after["benchmarks"][0]["stats"])
        self.assertEqual(after["benchmarks"][0]["stats"]["mean"], 0.1)

    def test_repeated_measurement_ids_are_rejected(self):
        payload = a_payload(2)
        first = payload["benchmarks"][0]["extra_info"]["benchmark_name"]
        payload["benchmarks"][1]["extra_info"]["benchmark_name"] = first
        with self.assertRaisesRegex(ValueError, "Repeated measurement_id"):
            BenchmarkRunResults(payload)

    def test_a_faster_run_reads_as_improved(self):
        faster = a_payload()
        faster["benchmarks"][0]["stats"]["data"] = [0.01, 0.011, 0.009, 0.0105, 0.0095]
        comparison = BenchmarkRunResults(faster).compare_against(
            BenchmarkRunResults(a_payload())
        )
        self.assertEqual(list(comparison["verdict"]), ["improved"])
        self.assertLess(comparison["relative_change_high"].iloc[0], 0)

    def test_a_slower_run_reads_as_regressed(self):
        slower = a_payload()
        slower["benchmarks"][0]["stats"]["data"] = [1.0, 1.1, 0.9, 1.05, 0.95]
        comparison = BenchmarkRunResults(slower).compare_against(
            BenchmarkRunResults(a_payload())
        )
        self.assertEqual(list(comparison["verdict"]), ["regressed"])
        self.assertGreater(comparison["relative_change_low"].iloc[0], 0)

    def test_an_identical_run_reads_as_no_change(self):
        comparison = BenchmarkRunResults(a_payload()).compare_against(
            BenchmarkRunResults(a_payload())
        )
        self.assertEqual(list(comparison.columns), COMPARISON_COLUMNS)
        self.assertEqual(comparison["verdict"].iloc[0], "no change")
        self.assertGreater(comparison["p_value"].iloc[0], 0.5)

    def test_a_significant_change_inside_the_noise_band_reads_as_within_noise(self):
        generator = np.random.default_rng(0)
        before, after = a_payload(), a_payload()
        before["benchmarks"][0]["stats"]["data"] = list(
            0.1 + generator.normal(0, 1e-4, 500)
        )
        after["benchmarks"][0]["stats"]["data"] = list(
            0.101 + generator.normal(0, 1e-4, 500)
        )
        comparison = BenchmarkRunResults(after).compare_against(
            BenchmarkRunResults(before)
        )
        self.assertLess(comparison["p_value"].iloc[0], 0.05)
        self.assertEqual(comparison["verdict"].iloc[0], "within noise")

    def test_a_retuned_tier_reads_as_workload_changed(self):
        retuned = a_payload()
        retuned["benchmarks"][0]["extra_info"]["panel_size"]["observation_count"] = 6000
        comparison = BenchmarkRunResults(retuned).compare_against(
            BenchmarkRunResults(a_payload())
        )
        self.assertEqual(comparison["verdict"].iloc[0], "workload changed")
        self.assertTrue(np.isnan(comparison["relative_change"].iloc[0]))

    def test_runs_on_different_hardware_are_refused(self):
        elsewhere = a_payload()
        elsewhere["machine_info"]["macrosynergy"]["hardware_id"] = "ffff0000"
        with self.assertRaisesRegex(ValueError, "different hardware"):
            BenchmarkRunResults(a_payload()).compare_against(
                BenchmarkRunResults(elsewhere)
            )

    def test_comparison_thresholds_are_validated(self):
        run = BenchmarkRunResults(a_payload())
        with self.assertRaises(TypeError):
            run.compare_against(run, noise_threshold="5%")
        with self.assertRaises(ValueError):
            run.compare_against(run, noise_threshold=-0.1)
        with self.assertRaises(ValueError):
            run.compare_against(run, significance_level=1.5)

    def test_comparing_runs_with_no_shared_measurement_gives_no_rows(self):
        other = a_payload()
        other["benchmarks"][0]["extra_info"]["benchmark_name"] = "elsewhere"
        comparison = BenchmarkRunResults(a_payload()).compare_against(
            BenchmarkRunResults(other)
        )
        self.assertTrue(comparison.empty)


# --------------------------------------------------------------------------------- page


def a_two_tier_payload() -> dict:
    """
    A payload holding one benchmark at two tiers, so its scaling chart is drawn.

    Returns
    -------
    dict
        A payload whose two entries share a benchmark name and differ in tier.
    """
    payload = a_payload(entry_count=2)
    second = payload["benchmarks"][1]["extra_info"]
    second["benchmark_name"] = "target_0"
    second["panel_size"] = {"tier": "small", "observation_count": 104000}
    return payload


def csv_rows(page: str) -> pd.DataFrame:
    """
    The rows behind a page's CSV download.

    Parameters
    ----------
    page : str
        A rendered page.

    Returns
    -------
    pd.DataFrame
        The CSV the download link carries.
    """
    link = re.search(r"href='data:text/csv;charset=utf-8,([^']*)'", page)
    return pd.read_csv(io.StringIO(urllib.parse.unquote(link.group(1))))


class TestRenderPage(unittest.TestCase):
    def setUp(self) -> None:
        self.page = render_page(BenchmarkRunResults(a_payload(entry_count=2)))

    def test_title_names_the_package_and_its_version(self):
        self.assertIn(
            "<title>Macrosynergy Package performance benchmarks unknown version</title>",
            self.page,
        )

    def test_each_method_gets_a_tile_naming_its_target_and_label(self):
        self.assertEqual(self.page.count("<article class=tile"), 2)
        self.assertIn("<code title='package.module'>target_0</code>", self.page)
        self.assertIn("<p class=label>Measure a target.</p>", self.page)

    def test_tile_table_puts_the_variant_parameter_in_the_header(self):
        self.assertIn("<th>dtype</th>", self.page)
        self.assertIn("<th scope=row>categorical</th>", self.page)

    def test_every_case_opens_a_dialog_with_its_call(self):
        for index in range(2):
            dialog = f"case-qdf-target-{index}-tiny-dtype-categorical"
            self.assertIn(f"data-dialog='{dialog}'", self.page)
            self.assertIn(f"<dialog id='{dialog}'", self.page)
        self.assertIn("<td>DataFrame 3,000×4</td>", self.page)

    def test_each_chart_type_is_drawn_once_per_method(self):
        for metric in ("time", "throughput", "memory"):
            self.assertEqual(self.page.count(f"class='chart chart-{metric}'"), 2)
        self.assertNotIn("chart-change", self.page)

    def test_without_a_baseline_there_is_no_regressions_section(self):
        self.assertNotIn("Regressions", self.page)
        self.assertIn("<h2>Fastest</h2>", self.page)
        self.assertIn("<h2>Environment</h2>", self.page)

    def test_chart_colours_follow_the_page_theme(self):
        self.assertIn("var(--series-1)", self.page)
        self.assertNotIn("#2a78d6", self.page.split("</style>", 1)[1])

    def test_csv_download_holds_one_row_per_case(self):
        rows = csv_rows(self.page)
        self.assertEqual(len(rows), 2)
        self.assertIn("median_seconds", rows.columns)
        self.assertIn("Download results as CSV", self.page)


class TestRenderComparison(unittest.TestCase):
    def setUp(self) -> None:
        slower = a_two_tier_payload()
        slower["benchmarks"][0]["stats"]["data"] = [1.0, 1.1, 0.9, 1.05, 0.95]
        self.page = render_page(
            BenchmarkRunResults(slower), BenchmarkRunResults(a_two_tier_payload())
        )

    def test_regressions_come_first_and_list_the_slower_case(self):
        regressions = self.page.index("<h2>Regressions")
        self.assertLess(regressions, self.page.index("<h2 id=methods>"))
        self.assertIn("<span class=count>1</span>", self.page)
        self.assertIn("No case is faster than the baseline.", self.page)

    def test_change_chart_joins_the_switcher(self):
        self.assertIn("<label for=show-change>Change</label>", self.page)
        self.assertEqual(self.page.count("class='chart chart-change'"), 1)
        # Each compared case gets a change panel in its dialog.
        self.assertEqual(self.page.count("change in mean (%)"), 2)

    def test_environment_shows_both_runs(self):
        self.assertIn("<th>Baseline</th><th>Current</th>", self.page)

    def test_csv_carries_the_comparison_columns(self):
        self.assertEqual(
            sorted(csv_rows(self.page)["verdict"]), ["no change", "regressed"]
        )


class TestVersionLabel(unittest.TestCase):
    def test_label_joins_release_build_kind_and_commit(self):
        machine = {
            "package_version": "1.8.1dev0+d21d6ab",
            "commit_sha": "f0ca6970" + "0" * 32,
        }
        self.assertEqual(version_label(machine), "v1.8.1.dev+f0ca6970")

    def test_uncommitted_package_changes_are_marked(self):
        machine = {
            "package_version": "1.8.1",
            "commit_sha": "a" * 40,
            "package_diff_hash": "x",
        }
        self.assertEqual(version_label(machine), "v1.8.1+aaaaaaaa.dirty")


class TestMain(unittest.TestCase):
    def test_writes_the_page_beside_the_enriched_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(json.dumps(a_payload()))
            self.assertEqual(main([str(path)]), 0)
            self.assertIn("macrosynergy", json.loads(path.read_text()))
            self.assertTrue(
                path.with_suffix(".html").read_text().startswith("<!doctype")
            )

    def test_baseline_option_adds_the_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            path, baseline = Path(directory) / "run.json", Path(directory) / "old.json"
            path.write_text(json.dumps(a_payload()))
            baseline.write_text(json.dumps(a_payload()))
            main([str(path), "--baseline", str(baseline)])
            self.assertIn("<h2>Regressions", path.with_suffix(".html").read_text())


if __name__ == "__main__":
    unittest.main()
