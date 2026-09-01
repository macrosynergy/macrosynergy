"""Tests for the objects in results.py."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy

import numpy as np

from tests.perf.results import (
    BenchmarkMeasurement,
    BenchmarkRunResults,
    ConfidenceInterval,
)

MEASUREMENT_TABLE_COLUMNS = [
    "measurement_id",
    "group",
    "benchmark_name",
    "tier",
    "variant",
    "timing_mode",
    "observation_count",
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
            "macrosynergy": {"fingerprint": "abcd1234"},
        },
        "benchmarks": [
            {
                "fullname": f"t.py::test_target[{index}]",
                "name": f"test_target[{index}]",
                "options": {"min_rounds": 5},
                "stats": {"data": [0.10, 0.11, 0.09, 0.105, 0.095], "mean": 0.1, "rounds": 5},
                "extra_info": {
                    "group": "qdf",
                    "benchmark_name": f"target_{index}",
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
            self.assertEqual(entry["macrosynergy"]["measurement_id"], row["measurement_id"])

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
        self.assertLess(comparison["percent_change"].iloc[0], 0)

    def test_a_slower_run_reads_as_regressed(self):
        slower = a_payload()
        slower["benchmarks"][0]["stats"]["data"] = [1.0, 1.1, 0.9, 1.05, 0.95]
        comparison = BenchmarkRunResults(slower).compare_against(
            BenchmarkRunResults(a_payload())
        )
        self.assertEqual(list(comparison["verdict"]), ["regressed"])
        self.assertGreater(comparison["percent_change"].iloc[0], 0)

    def test_overlapping_intervals_read_as_no_change(self):
        comparison = BenchmarkRunResults(a_payload()).compare_against(
            BenchmarkRunResults(a_payload())
        )
        self.assertEqual(
            list(comparison.columns),
            [
                "measurement_id",
                "baseline_seconds",
                "current_seconds",
                "percent_change",
                "verdict",
            ],
        )
        self.assertEqual(comparison["verdict"].iloc[0], "no change")

    def test_comparing_runs_with_no_shared_measurement_gives_no_rows(self):
        other = a_payload()
        other["benchmarks"][0]["extra_info"]["benchmark_name"] = "elsewhere"
        comparison = BenchmarkRunResults(a_payload()).compare_against(
            BenchmarkRunResults(other)
        )
        self.assertTrue(comparison.empty)


if __name__ == "__main__":
    unittest.main()
