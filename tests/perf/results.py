"""
What a benchmark run produced. `BenchmarkRunResults` turns a pytest-benchmark payload
into `BenchmarkMeasurement` objects, `measurement_table` is the single definition of the
per-case statistics, and every statistic carries percentile-bootstrap bounds so that
run-to-run noise reads as no change rather than as a regression.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 12345
BOOTSTRAP_CONFIDENCE = 0.95


def _sample_stddev(values, axis=None):
    """
    Standard deviation with one degree of freedom removed.

    Parameters
    ----------
    values : np.ndarray
        Timings, either one sample or a stack of bootstrap resamples.
    axis : Optional[int]
        Axis to reduce, or None to reduce everything.

    Returns
    -------
    np.ndarray
        The sample standard deviation.
    """
    return np.std(values, axis=axis, ddof=1)


def _median_absolute_deviation(values, axis=None):
    """
    Median of the absolute deviations from the median.

    Parameters
    ----------
    values : np.ndarray
        Timings, either one sample or a stack of bootstrap resamples.
    axis : Optional[int]
        Axis to reduce, or None to reduce everything.

    Returns
    -------
    np.ndarray
        The median absolute deviation.
    """
    median = np.median(values, axis=axis, keepdims=axis is not None)
    return np.median(np.abs(values - median), axis=axis)


def _variant_label(variant: Tuple[Tuple[str, str], ...]) -> str:
    """
    The variant of a measurement as one comparable string.

    Parameters
    ----------
    variant : Tuple[Tuple[str, str], ...]
        Sorted parameter-value pairs other than the panel size.

    Returns
    -------
    str
        Comma-separated `name=value` pairs, empty when there is no variant.
    """
    return ",".join(f"{name}={value}" for name, value in variant)


@dataclass(frozen=True)
class ConfidenceInterval:
    """
    A statistic with percentile-bootstrap bounds.

    Parameters
    ----------
    value : float
        The statistic computed on the observed timings.
    low : Optional[float]
        Lower bound, or None when there were too few timings to resample.
    high : Optional[float]
        Upper bound, or None when there were too few timings to resample.
    """

    value: float
    low: Optional[float] = None
    high: Optional[float] = None

    @classmethod
    def from_samples(
        cls, samples: Sequence[float], statistic: Callable, seed: int = BOOTSTRAP_SEED
    ) -> "ConfidenceInterval":
        """
        Bootstrap a statistic over timings; a single timing yields no bounds.

        Parameters
        ----------
        samples : Sequence[float]
            The observed per-round timings, in seconds.
        statistic : Callable
            Reducer accepting the samples, and accepting an `axis` keyword so it can
            reduce a stack of resamples in one call.
        seed : int
            Seed for the resampling generator, recorded with the run so an interval is
            reproducible from the timings alone.

        Returns
        -------
        ConfidenceInterval
            The statistic and its bounds, or the statistic alone for one sample.
        """
        observed = np.asarray(samples, dtype=float)
        point = float(statistic(observed))
        if observed.size < 2:
            return cls(point)
        resampled = np.random.default_rng(seed).choice(
            observed, size=(BOOTSTRAP_RESAMPLES, observed.size)
        )
        tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0 * 100.0
        low, high = np.percentile(statistic(resampled, axis=1), [tail, 100.0 - tail])
        return cls(point, float(low), float(high))


@dataclass(frozen=True)
class BenchmarkMeasurement:
    """
    One benchmark run at one panel size, with its raw per-round timings.

    Parameters
    ----------
    group : str
        The `perf_group` label the benchmark was marked with.
    benchmark_name : str
        The test function name without its `test_` prefix.
    tier : str
        Name of the panel size tier that was measured.
    variant : Tuple[Tuple[str, str], ...]
        Sorted parameter-value pairs other than the panel size.
    panel_size : Dict[str, Any]
        The size block the run recorded.
    round_seconds : Tuple[float, ...]
        Per-round timings, in seconds.
    timing_mode : str
        "calibrated" for the default path, "pedantic" where the target mutates its
        input.
    peak_memory_bytes : Optional[int]
        Peak allocation over one untimed call of the target.
    """

    group: str
    benchmark_name: str
    tier: str
    variant: Tuple[Tuple[str, str], ...]
    panel_size: Dict[str, Any]
    round_seconds: Tuple[float, ...]
    timing_mode: str = "calibrated"
    peak_memory_bytes: Optional[int] = None

    @classmethod
    def from_payload_entry(cls, entry: Dict[str, Any]) -> "BenchmarkMeasurement":
        """
        Read one `benchmarks[]` entry of a pytest-benchmark payload.

        Parameters
        ----------
        entry : Dict[str, Any]
            One benchmark entry, whose `extra_info` the perf conftest stamped.

        Returns
        -------
        BenchmarkMeasurement
            The measurement described by the entry.

        Raises
        ------
        ValueError
            If `extra_info` lacks `group`, `benchmark_name` or `panel_size`, which means
            the benchmark ran without the conftest that records them.
        """
        recorded = entry.get("extra_info") or {}
        missing = [
            key
            for key in ("group", "benchmark_name", "panel_size")
            if key not in recorded
        ]
        if missing:
            raise ValueError(f"{entry.get('fullname')}: extra_info is missing {missing}")
        return cls(
            group=recorded["group"],
            benchmark_name=recorded["benchmark_name"],
            tier=recorded["panel_size"]["tier"],
            variant=tuple(sorted((recorded.get("variant") or {}).items())),
            panel_size=recorded["panel_size"],
            round_seconds=tuple(entry["stats"]["data"]),
            timing_mode=recorded.get("timing_mode", "calibrated"),
            peak_memory_bytes=recorded.get("peak_memory_bytes"),
        )

    @property
    def identifier(self) -> str:
        """
        Stable name, from recorded metadata rather than the pytest node id.

        Renaming a file or reordering decorators changes the node id but not this, so a
        measurement keeps its history across refactoring.

        Returns
        -------
        str
            Group, benchmark name, tier and variant, joined by slashes.
        """
        parts = (self.group, self.benchmark_name, self.tier, _variant_label(self.variant))
        return "/".join(part for part in parts if part)

    @property
    def mean_seconds(self) -> ConfidenceInterval:
        """
        Mean round time.

        Returns
        -------
        ConfidenceInterval
            The mean, in seconds, with bootstrap bounds.
        """
        return ConfidenceInterval.from_samples(self.round_seconds, np.mean)

    @property
    def median_seconds(self) -> ConfidenceInterval:
        """
        Median round time.

        Returns
        -------
        ConfidenceInterval
            The median, in seconds, with bootstrap bounds.
        """
        return ConfidenceInterval.from_samples(self.round_seconds, np.median)

    @property
    def stddev_seconds(self) -> ConfidenceInterval:
        """
        Spread of the round times.

        Returns
        -------
        ConfidenceInterval
            The sample standard deviation, in seconds, with bootstrap bounds.
        """
        return ConfidenceInterval.from_samples(self.round_seconds, _sample_stddev)

    @property
    def median_absolute_deviation_seconds(self) -> ConfidenceInterval:
        """
        Outlier-resistant spread of the round times.

        Returns
        -------
        ConfidenceInterval
            The median absolute deviation, in seconds, with bootstrap bounds.
        """
        return ConfidenceInterval.from_samples(
            self.round_seconds, _median_absolute_deviation
        )

    @property
    def observations_per_second(self) -> Optional[float]:
        """
        Throughput; flat across tiers means the target scales linearly.

        Returns
        -------
        Optional[float]
            Observations divided by the mean round time, or None when that time is zero.
        """
        mean = float(np.mean(self.round_seconds))
        return self.panel_size["observation_count"] / mean if mean else None


class BenchmarkRunResults:
    """
    Every measurement of one run, the machine it ran on, and the payload it came from.

    Parameters
    ----------
    payload : Dict[str, Any]
        A pytest-benchmark JSON payload, as written by `--benchmark-json`.

    Raises
    ------
    ValueError
        If two measurements share an identifier, or if an entry's `extra_info` is
        incomplete.
    """

    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        self.machine = (payload.get("machine_info") or {}).get("macrosynergy")
        self.measurements = [
            BenchmarkMeasurement.from_payload_entry(entry)
            for entry in payload.get("benchmarks", [])
        ]
        identifiers = [m.identifier for m in self.measurements]
        repeated = sorted({i for i in identifiers if identifiers.count(i) > 1})
        if repeated:
            raise ValueError(f"Repeated measurement_id: {repeated}")

    @classmethod
    def load(cls, path: Any) -> "BenchmarkRunResults":
        """
        Read a run from a pytest-benchmark JSON file.

        Parameters
        ----------
        path : Any
            Path to the file, as anything `pathlib.Path` accepts.

        Returns
        -------
        BenchmarkRunResults
            The run described by the file.
        """
        return cls(json.loads(Path(path).read_text()))

    def by_identifier(self) -> Dict[str, BenchmarkMeasurement]:
        """
        The measurements, keyed for lookup.

        Returns
        -------
        Dict[str, BenchmarkMeasurement]
            Every measurement under its identifier.
        """
        return {m.identifier: m for m in self.measurements}

    def measurement_table(self) -> pd.DataFrame:
        """
        The per-case statistics, one row per measurement.

        This is the single definition of what the suite computes for a case; the derived
        block written into a payload is built from these rows.

        Returns
        -------
        pd.DataFrame
            One row per measurement. Timing columns are seconds and carry `_low` and
            `_high` companions holding their bootstrap bounds; `peak_memory_bytes` is
            bytes.
        """
        rows = []
        for measurement in self.measurements:
            row = {
                "measurement_id": measurement.identifier,
                "group": measurement.group,
                "benchmark_name": measurement.benchmark_name,
                "tier": measurement.tier,
                "variant": _variant_label(measurement.variant),
                "timing_mode": measurement.timing_mode,
                "observation_count": measurement.panel_size["observation_count"],
            }
            for name in (
                "mean_seconds",
                "median_seconds",
                "stddev_seconds",
                "median_absolute_deviation_seconds",
            ):
                interval = getattr(measurement, name)
                row[name] = interval.value
                row[f"{name}_low"] = interval.low
                row[f"{name}_high"] = interval.high
            row["observations_per_second"] = measurement.observations_per_second
            row["peak_memory_bytes"] = measurement.peak_memory_bytes
            rows.append(row)
        return pd.DataFrame(rows)

    def with_derived_statistics(self, drop_round_timings: bool = False) -> Dict[str, Any]:
        """
        The payload with derived keys added; nothing existing is changed or removed.

        The additions live under one `macrosynergy` key, at the top level and per
        benchmark, so `pytest-benchmark compare` still reads the file.

        Parameters
        ----------
        drop_round_timings : bool
            Whether to omit the per-round samples from `stats.data`.

        Returns
        -------
        Dict[str, Any]
            A copy of the payload carrying the run block and one derived block per
            benchmark.
        """
        enriched = deepcopy(self.payload)
        enriched.setdefault("macrosynergy", {}).update(
            {
                "run_id": enriched.get("datetime"),
                "tiers_measured": sorted({m.tier for m in self.measurements}),
                "bootstrap": {
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "confidence": BOOTSTRAP_CONFIDENCE,
                },
            }
        )
        table = self.measurement_table()
        records = table.astype(object).where(table.notna(), None).to_dict("records")
        for entry, record in zip(enriched.get("benchmarks", []), records):
            entry["macrosynergy"] = record
            if drop_round_timings:
                entry["stats"].pop("data", None)
        return enriched

    def compare_against(self, baseline: "BenchmarkRunResults") -> pd.DataFrame:
        """
        How every measurement present in both runs moved.

        The verdict comes from whether the two mean intervals overlap rather than from a
        percentage threshold, so run-to-run noise reads as no change.

        Parameters
        ----------
        baseline : BenchmarkRunResults
            The earlier run to compare against.

        Returns
        -------
        pd.DataFrame
            One row per shared measurement, with columns `measurement_id`,
            `baseline_seconds`, `current_seconds`, `percent_change` and `verdict`. The
            verdict is "improved", "regressed" or "no change".
        """
        current, previous = self.by_identifier(), baseline.by_identifier()
        rows = []
        for key in sorted(current.keys() & previous.keys()):
            before = previous[key].mean_seconds
            after = current[key].mean_seconds
            percent = (
                (after.value - before.value) / before.value * 100.0
                if before.value
                else 0.0
            )
            bounds = (before.low, before.high, after.low, after.high)
            overlapping = None in bounds or (
                after.low <= before.high and before.low <= after.high
            )
            if overlapping:
                verdict = "no change"
            else:
                verdict = "improved" if after.value < before.value else "regressed"
            rows.append(
                {
                    "measurement_id": key,
                    "baseline_seconds": before.value,
                    "current_seconds": after.value,
                    "percent_change": percent,
                    "verdict": verdict,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "measurement_id",
                "baseline_seconds",
                "current_seconds",
                "percent_change",
                "verdict",
            ],
        )
