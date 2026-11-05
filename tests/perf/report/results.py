"""
Loading, statistics and comparison of benchmark results.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 12345
BOOTSTRAP_CONFIDENCE = 0.95
# A change is reported only when it is significant at this level and its whole bootstrap
# interval sits outside +/- the noise threshold, as Criterion.rs decides it.
SIGNIFICANCE_LEVEL = 0.05
# ponytail: one threshold for every runner; calibrate per runner from repeated runs of
# identical code, so that those read as no change.
NOISE_THRESHOLD = 0.05
COMPARISON_COLUMNS = [
    "measurement_id",
    "baseline_mean_seconds",
    "current_mean_seconds",
    "relative_change",
    "relative_change_low",
    "relative_change_high",
    "p_value",
    "verdict",
]
# Bump when the layout of the derived block changes, so readers can tell old files apart.
SCHEMA_VERSION = 2


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


def _welch_t(current, baseline, axis=None):
    """
    Welch's t statistic for the difference between two mean timings.

    Parameters
    ----------
    current : np.ndarray
        Timings of the later run, either one sample or a stack of resamples.
    baseline : np.ndarray
        Timings of the earlier run, shaped like `current` apart from the sample axis.
    axis : Optional[int]
        Axis holding the samples, or None for one sample each.

    Returns
    -------
    np.ndarray
        The t statistic; positive when `current` is slower.
    """
    difference = np.mean(current, axis=axis) - np.mean(baseline, axis=axis)
    spread = (
        np.var(current, axis=axis, ddof=1) / np.shape(current)[-1]
        + np.var(baseline, axis=axis, ddof=1) / np.shape(baseline)[-1]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return difference / np.sqrt(spread)


def bootstrap_mean_changes(
    current: Sequence[float], baseline: Sequence[float], generator: np.random.Generator
) -> np.ndarray:
    """
    Relative change in mean timing over independent bootstrap resamples of two runs.

    Parameters
    ----------
    current : Sequence[float]
        Per-round timings of the later run, in seconds.
    baseline : Sequence[float]
        Per-round timings of the earlier run, in seconds.
    generator : np.random.Generator
        Source of the resampling, so a caller seeding it alike gets the same draws.

    Returns
    -------
    np.ndarray
        `BOOTSTRAP_RESAMPLES` changes, each a fraction of the resampled baseline mean.
    """
    after = np.asarray(current, dtype=float)
    before = np.asarray(baseline, dtype=float)
    resampled_after = generator.choice(after, size=(BOOTSTRAP_RESAMPLES, after.size))
    resampled_before = generator.choice(before, size=(BOOTSTRAP_RESAMPLES, before.size))
    return resampled_after.mean(axis=1) / resampled_before.mean(axis=1) - 1.0


def mean_change(
    current: Sequence[float], baseline: Sequence[float], seed: int = BOOTSTRAP_SEED
) -> Tuple[ConfidenceInterval, Optional[float]]:
    """
    Relative change in mean timing, with bootstrap bounds and a bootstrap p-value.

    The bounds resample each run independently. The p-value is a bootstrap t-test: both
    runs are resampled from their pooled timings, which is what "no difference" would
    look like, and the p-value is the share of those resamples whose t statistic is at
    least as extreme as the observed one.

    Parameters
    ----------
    current : Sequence[float]
        Per-round timings of the later run, in seconds.
    baseline : Sequence[float]
        Per-round timings of the earlier run, in seconds.
    seed : int
        Seed for the resampling generator.

    Returns
    -------
    Tuple[ConfidenceInterval, Optional[float]]
        The change as a fraction of the baseline mean (0.1 is 10% slower), and the
        p-value. With fewer than two timings in either run there are no bounds and the
        p-value is None.
    """
    after = np.asarray(current, dtype=float)
    before = np.asarray(baseline, dtype=float)
    change = after.mean() / before.mean() - 1.0
    if after.size < 2 or before.size < 2:
        return ConfidenceInterval(float(change)), None

    generator = np.random.default_rng(seed)
    changes = bootstrap_mean_changes(after, before, generator)
    tail = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0 * 100.0
    low, high = np.percentile(changes, [tail, 100.0 - tail])

    observed = _welch_t(after, before)
    if not np.isfinite(observed):
        # Both runs are constant: any difference is exact, none is no difference.
        p_value = 0.0 if change else 1.0
    else:
        pooled = np.concatenate([after, before])
        null = _welch_t(
            generator.choice(pooled, size=(BOOTSTRAP_RESAMPLES, after.size)),
            generator.choice(pooled, size=(BOOTSTRAP_RESAMPLES, before.size)),
            axis=1,
        )
        p_value = float(np.mean(np.abs(null) >= abs(observed)))
    return ConfidenceInterval(float(change), float(low), float(high)), p_value


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
    target : Optional[str]
        The callable measured, as `module:qualname`, or None for runs recorded before
        targets were.
    label : Optional[str]
        One line saying what the benchmark measures, from its docstring.
    call : Dict[str, str]
        Each argument of the measured call, described, keyed by parameter name.
    setup : Dict[str, Dict[str, str]]
        Calls made to prepare the target, such as a constructor, described the same way
        and keyed by the callable's name.
    """

    group: str
    benchmark_name: str
    tier: str
    variant: Tuple[Tuple[str, str], ...]
    panel_size: Dict[str, Any]
    round_seconds: Tuple[float, ...]
    timing_mode: str = "calibrated"
    peak_memory_bytes: Optional[int] = None
    target: Optional[str] = None
    label: Optional[str] = None
    call: Dict[str, str] = field(default_factory=dict)
    setup: Dict[str, Dict[str, str]] = field(default_factory=dict)

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
            raise ValueError(
                f"{entry.get('fullname')}: extra_info is missing {missing}"
            )
        return cls(
            group=recorded["group"],
            benchmark_name=recorded["benchmark_name"],
            tier=recorded["panel_size"]["tier"],
            variant=tuple(sorted((recorded.get("variant") or {}).items())),
            panel_size=recorded["panel_size"],
            round_seconds=tuple(entry["stats"]["data"]),
            timing_mode=recorded.get("timing_mode", "calibrated"),
            peak_memory_bytes=recorded.get("peak_memory_bytes"),
            target=recorded.get("target"),
            label=recorded.get("label"),
            call=recorded.get("call") or {},
            setup=recorded.get("setup") or {},
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
        parts = (
            self.group,
            self.benchmark_name,
            self.tier,
            _variant_label(self.variant),
        )
        return "/".join(part for part in parts if part)

    @property
    def workload_id(self) -> str:
        """
        Hash of the work measured, so a retuned tier or a changed call that keeps its
        name reads as new work.

        Returns
        -------
        str
            The first eight hex digits of the SHA-256 of the size block, variant, target,
            call and setup.
        """
        blob = json.dumps(
            [self.panel_size, self.variant, self.target, self.call, self.setup],
            sort_keys=True,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:8]

    @property
    def arguments(self) -> str:
        """
        The measured call's arguments as one line.

        Returns
        -------
        str
            `name=description` pairs joined by semicolons, empty when none were recorded.
        """
        return "; ".join(f"{name}={value}" for name, value in self.call.items())

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
                "target": measurement.target,
                "label": measurement.label,
                "arguments": measurement.arguments,
                "tier": measurement.tier,
                "variant": _variant_label(measurement.variant),
                "timing_mode": measurement.timing_mode,
                "observation_count": measurement.panel_size["observation_count"],
                "workload_id": measurement.workload_id,
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

    def with_derived_statistics(
        self, drop_round_timings: bool = False
    ) -> Dict[str, Any]:
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
                "schema_version": SCHEMA_VERSION,
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

    def compare_against(
        self,
        baseline: "BenchmarkRunResults",
        noise_threshold: float = NOISE_THRESHOLD,
        significance_level: float = SIGNIFICANCE_LEVEL,
    ) -> pd.DataFrame:
        """
        How every measurement present in both runs moved, decided as Criterion.rs does.

        A change is "regressed" or "improved" only when its p-value is below
        `significance_level` and its whole bootstrap interval lies beyond
        `noise_threshold`. A significant change whose interval reaches into the noise
        band is "within noise"; anything else is "no change". A case whose size block or
        variant differs between the runs is "workload changed" and gets no statistics.

        Parameters
        ----------
        baseline : BenchmarkRunResults
            The earlier run to compare against.
        noise_threshold : float
            Relative change treated as noise, as a fraction (0.05 is 5%).
        significance_level : float
            Largest p-value that counts as a change.

        Returns
        -------
        pd.DataFrame
            One row per shared measurement, with columns `measurement_id`,
            `baseline_mean_seconds`, `current_mean_seconds`, `relative_change`,
            `relative_change_low`, `relative_change_high`, `p_value` and `verdict`.

        Raises
        ------
        TypeError
            If `noise_threshold` or `significance_level` is not a number.
        ValueError
            If `noise_threshold` is negative, `significance_level` is not in (0, 1), or
            both runs record a hardware id and the ids differ.
        """
        for name, value in (
            ("noise_threshold", noise_threshold),
            ("significance_level", significance_level),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a number.")
        if noise_threshold < 0:
            raise ValueError("`noise_threshold` must be non-negative.")
        if not 0 < significance_level < 1:
            raise ValueError("`significance_level` must be between 0 and 1.")
        hardware = [(run.machine or {}).get("hardware_id") for run in (baseline, self)]
        if all(hardware) and hardware[0] != hardware[1]:
            raise ValueError(
                f"Runs on different hardware are not comparable: {hardware[0]} != "
                f"{hardware[1]}."
            )

        current, previous = self.by_identifier(), baseline.by_identifier()
        rows = []
        for key in sorted(current.keys() & previous.keys()):
            after, before = current[key], previous[key]
            row = {
                "measurement_id": key,
                "baseline_mean_seconds": float(np.mean(before.round_seconds)),
                "current_mean_seconds": float(np.mean(after.round_seconds)),
            }
            if after.workload_id != before.workload_id:
                rows.append({**row, "verdict": "workload changed"})
                continue
            change, p_value = mean_change(after.round_seconds, before.round_seconds)
            if p_value is None or p_value >= significance_level:
                verdict = "no change"
            elif change.low > noise_threshold:
                verdict = "regressed"
            elif change.high < -noise_threshold:
                verdict = "improved"
            else:
                verdict = "within noise"
            rows.append(
                {
                    **row,
                    "relative_change": change.value,
                    "relative_change_low": change.low,
                    "relative_change_high": change.high,
                    "p_value": p_value,
                    "verdict": verdict,
                }
            )
        return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
