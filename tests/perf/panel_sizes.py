"""
The data sizes a benchmark can run against. `PANEL_SIZES` names the tiers, a `PanelSize`
describes one of them and builds the DataFrame for it, and the builders are cached so a
frame is constructed once per run and shared between benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytest

from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (
    _map_to_business_day_frequency,
    get_cid,
    get_xcat,
    qdf_to_ticker_df,
)

FIRST_DATE = "2000-01-01"
VALUE_STYLE = "linear"

QUANTAMENTAL_INDEX_COLUMNS = ("cid", "xcat", "real_date")


def _generated_cids(count: int) -> List[str]:
    """
    Distinct three-letter cross-sectional identifiers.

    Parameters
    ----------
    count : int
        How many identifiers to generate.

    Returns
    -------
    List[str]
        `count` identifiers, in generation order.
    """
    return [
        f"{chr(65 + i // 676 % 26)}{chr(65 + i // 26 % 26)}{chr(65 + i % 26)}"
        for i in range(count)
    ]


def _generated_xcats(count: int) -> List[str]:
    """
    Distinct extended-category names.

    Parameters
    ----------
    count : int
        How many categories to generate.

    Returns
    -------
    List[str]
        `count` category names, in generation order.
    """
    return [f"XCAT{i:03d}" for i in range(count)]


def _business_days(count: int) -> pd.DatetimeIndex:
    """
    Business days starting at `FIRST_DATE`.

    A tier's date count has to be exact, so the range is built by period count rather
    than by end date, after confirming the package still treats daily data as business
    daily.

    Parameters
    ----------
    count : int
        How many business days to generate.

    Returns
    -------
    pd.DatetimeIndex
        `count` consecutive business days.

    Raises
    ------
    RuntimeError
        If the package no longer maps the daily frequency to "B".
    """
    if _map_to_business_day_frequency("D") != "B":
        raise RuntimeError("Package daily frequency is no longer 'B'.")
    return pd.bdate_range(start=FIRST_DATE, periods=count)


@dataclass(frozen=True)
class PanelSize:
    """
    How much synthetic panel data a benchmark runs against, and how long to measure it.

    Tickers are given either as an explicit tuple or as `cid_count` and `xcat_count`,
    never both, matching the contract of `make_test_df`. Every count and shape is
    derived from those fields rather than stored, so a size cannot contradict itself.

    Parameters
    ----------
    tier : str
        Name of the tier, used as the pytest parameter id and recorded with a
        measurement.
    date_count : int
        Number of business days the panel spans.
    min_rounds : int
        Least number of measured rounds pytest-benchmark should complete.
    max_seconds : float
        Wall-clock budget pytest-benchmark may spend on the benchmark.
    cid_count : Optional[int]
        Number of cross-sections, when tickers are given as parts.
    xcat_count : Optional[int]
        Number of extended categories, when tickers are given as parts.
    tickers : Optional[Tuple[str, ...]]
        Explicit tickers, as an alternative to `cid_count` and `xcat_count`.
    metrics : Tuple[str, ...]
        Metric columns the long DataFrame carries.
    is_ticker_df : bool
        Whether this size was selected for the wide format rather than the long one.
    extra : Tuple[Tuple[str, Any], ...]
        Additional key-value pairs recorded with a measurement.

    Raises
    ------
    ValueError
        If both ticker forms are given, or neither, or only one half of the parts form;
        if `tickers` repeats a ticker; or if `date_count` is not positive.
    """

    tier: str
    date_count: int
    min_rounds: int
    max_seconds: float
    cid_count: Optional[int] = None
    xcat_count: Optional[int] = None
    tickers: Optional[Tuple[str, ...]] = None
    metrics: Tuple[str, ...] = ("value",)
    is_ticker_df: bool = False
    extra: Tuple[Tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        by_parts = self.cid_count is not None or self.xcat_count is not None
        if by_parts == (self.tickers is not None):
            raise ValueError(
                "Give either `tickers` or both `cid_count` and `xcat_count`, "
                "not both forms and not neither. Received "
                f"`tickers`={self.tickers!r}, `cid_count`={self.cid_count!r}, "
                f"`xcat_count`={self.xcat_count!r}."
            )
        if by_parts and (self.cid_count is None or self.xcat_count is None):
            raise ValueError(
                "Give both `cid_count` and `xcat_count`, or neither. Received "
                f"`cid_count`={self.cid_count!r}, `xcat_count`={self.xcat_count!r}."
            )
        if self.tickers is not None and len(set(self.tickers)) != len(self.tickers):
            raise ValueError(
                f"`tickers` must not contain duplicates. Received {self.tickers!r}."
            )
        if self.date_count <= 0:
            raise ValueError(
                f"`date_count` must be positive. Received {self.date_count!r} instead."
            )

    @property
    def ticker_count(self) -> int:
        """
        Number of tickers the panel holds.

        Returns
        -------
        int
            The length of `tickers`, or `cid_count * xcat_count`.
        """
        if self.tickers is not None:
            return len(self.tickers)
        return self.cid_count * self.xcat_count

    @property
    def observation_count(self) -> int:
        """
        Number of data points, the size axis every benchmark is comparable on.

        Returns
        -------
        int
            `ticker_count * date_count`.
        """
        return self.ticker_count * self.date_count

    @property
    def cids(self) -> List[str]:
        """
        Cross-sectional identifiers the panel contains.

        Returns
        -------
        List[str]
            Generated identifiers for the parts form, or the sorted distinct cids of
            `tickers`.
        """
        if self.tickers is None:
            return _generated_cids(self.cid_count)
        return sorted(set(get_cid(list(self.tickers))))

    @property
    def xcats(self) -> List[str]:
        """
        Extended categories the panel contains.

        Returns
        -------
        List[str]
            Generated categories for the parts form, or the sorted distinct xcats of
            `tickers`.
        """
        if self.tickers is None:
            return _generated_xcats(self.xcat_count)
        return sorted(set(get_xcat(list(self.tickers))))

    @property
    def qdf_shape(self) -> Tuple[int, int]:
        """
        Shape of the long DataFrame, which holds one row per ticker and date.

        Returns
        -------
        Tuple[int, int]
            Row count and column count.
        """
        columns = len(QUANTAMENTAL_INDEX_COLUMNS) + len(self.metrics)
        return self.observation_count, columns

    @property
    def ticker_df_shape(self) -> Tuple[int, int]:
        """
        Shape of the wide DataFrame, which holds one row per date and one column per
        ticker.

        Returns
        -------
        Tuple[int, int]
            Row count and column count.
        """
        return self.date_count, self.ticker_count

    @property
    def shape(self) -> Tuple[int, int]:
        """
        Shape of the format this size was selected for.

        Returns
        -------
        Tuple[int, int]
            `ticker_df_shape` when `is_ticker_df` is set, otherwise `qdf_shape`.
        """
        if self.is_ticker_df:
            return self.ticker_df_shape
        return self.qdf_shape

    @property
    def df_format(self) -> str:
        """
        Name of the format this size was selected for.

        Returns
        -------
        str
            "wide" when `is_ticker_df` is set, otherwise "long".
        """
        return "wide" if self.is_ticker_df else "long"

    def as_qdf(self, categorical: bool = False) -> pd.DataFrame:
        """
        The long DataFrame for this size, shared between callers.

        Parameters
        ----------
        categorical : bool
            Whether to return the frame as a categorical `QuantamentalDataFrame`.

        Returns
        -------
        pd.DataFrame
            One row per ticker and date, with a column per metric. The same object is
            returned on every call until the cache is cleared.
        """
        return _cached_qdf(self, categorical)

    def as_ticker_df(self) -> pd.DataFrame:
        """
        The wide DataFrame for this size, shared between callers.

        Returns
        -------
        pd.DataFrame
            One row per date and one column per ticker, indexed by `real_date`.
        """
        return _cached_ticker_df(self)

    def as_qdf_copy(self, categorical: bool = False) -> pd.DataFrame:
        """
        An independent long DataFrame, for targets that modify what they are given.

        Parameters
        ----------
        categorical : bool
            Whether to return the frame as a categorical `QuantamentalDataFrame`.

        Returns
        -------
        pd.DataFrame
            A fresh copy of `as_qdf`, safe to mutate.
        """
        return self.as_qdf(categorical=categorical).copy()

    def describe(self) -> Dict[str, Any]:
        """
        The size block recorded alongside a measurement.

        Returns
        -------
        Dict[str, Any]
            The stored fields plus the derived counts and shape, in the plain types
            pytest-benchmark's `extra_info` channel can serialise.
        """
        row_count, column_count = self.shape
        return {
            "tier": self.tier,
            "df_format": self.df_format,
            "ticker_count": self.ticker_count,
            "cid_count": self.cid_count,
            "xcat_count": self.xcat_count,
            "tickers": list(self.tickers) if self.tickers is not None else None,
            "date_count": self.date_count,
            "metrics": list(self.metrics),
            "observation_count": self.observation_count,
            "row_count": row_count,
            "column_count": column_count,
            "extra": dict(self.extra),
        }

    def __str__(self) -> str:
        return self.tier


def _check_shape(
    size: PanelSize, df: pd.DataFrame, expected: Tuple[int, int]
) -> None:
    """
    Confirm a freshly built DataFrame has the shape the size that asked for it derives.

    Parameters
    ----------
    size : PanelSize
        The size the DataFrame was built for.
    df : pd.DataFrame
        The DataFrame that was built.
    expected : Tuple[int, int]
        The row and column counts derived from `size`.

    Raises
    ------
    AssertionError
        If the shapes disagree, which means a recorded size would misdescribe its data.
    """
    if df.shape != expected:
        raise AssertionError(
            f"{size.tier} {size.df_format} frame is {df.shape}, expected {expected}"
        )


@lru_cache(maxsize=None)
def _cached_qdf(size: PanelSize, categorical: bool) -> pd.DataFrame:
    """
    Build the long DataFrame for a size, once per distinct request.

    Parameters
    ----------
    size : PanelSize
        The size to build. Frozen, so it serves as the cache key.
    categorical : bool
        Whether to return the frame as a categorical `QuantamentalDataFrame`.

    Returns
    -------
    pd.DataFrame
        One row per ticker and date, with a column per metric.

    Raises
    ------
    AssertionError
        If the built frame's shape or date count disagrees with `size`.
    """
    dates = _business_days(size.date_count)
    shared = dict(
        metrics=list(size.metrics),
        start=str(dates[0].date()),
        end=str(dates[-1].date()),
        style=VALUE_STYLE,
    )
    if size.tickers is not None:
        df = make_test_df(tickers=list(size.tickers), cids=None, xcats=None, **shared)
    else:
        df = make_test_df(tickers=None, cids=size.cids, xcats=size.xcats, **shared)

    _check_shape(size, df, size.qdf_shape)
    built_dates = df["real_date"].nunique()
    if built_dates != size.date_count:
        raise AssertionError(
            f"{size.tier} frame holds {built_dates} dates, expected {size.date_count}"
        )
    return QuantamentalDataFrame(df, categorical=True) if categorical else df


@lru_cache(maxsize=None)
def _cached_ticker_df(size: PanelSize) -> pd.DataFrame:
    """
    Build the wide DataFrame for a size, once per size.

    Parameters
    ----------
    size : PanelSize
        The size to build. Frozen, so it serves as the cache key.

    Returns
    -------
    pd.DataFrame
        One row per date and one column per ticker, indexed by `real_date`.

    Raises
    ------
    AssertionError
        If the built frame's shape disagrees with `size`.
    """
    long_df = replace(size, is_ticker_df=False).as_qdf()
    wide = qdf_to_ticker_df(long_df, value_column=size.metrics[0])
    _check_shape(size, wide, size.ticker_df_shape)
    return wide


def clear_df_cache() -> None:
    """
    Release every DataFrame built so far.

    Returns
    -------
    None
    """
    _cached_qdf.cache_clear()
    _cached_ticker_df.cache_clear()


class PanelSizeCatalog:
    """
    The named panel sizes, and which of them this run measures.

    Parameters
    ----------
    sizes : Dict[str, PanelSize]
        The sizes, keyed by tier name.
    default_tiers : Sequence[str]
        Tiers measured when the run does not select any.
    """

    def __init__(self, sizes: Dict[str, PanelSize], default_tiers: Sequence[str]):
        self._sizes = dict(sizes)
        self._default_tiers = tuple(default_tiers)
        self.tier_names = tuple(sizes)
        self.selected_tiers = tuple(default_tiers)

    def __getitem__(self, tier: str) -> PanelSize:
        return self._sizes[tier]

    def select_tiers(self, comma_separated: Optional[str]) -> None:
        """
        Choose the tiers this run measures; empty input restores the default.

        Parameters
        ----------
        comma_separated : Optional[str]
            Tier names separated by commas, or None to restore the default selection.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If a named tier is not in the catalogue, or if the input names no tier at
            all.
        """
        if not comma_separated:
            self.selected_tiers = self._default_tiers
            return
        requested = tuple(
            part.strip() for part in comma_separated.split(",") if part.strip()
        )
        unknown = [tier for tier in requested if tier not in self._sizes]
        if unknown:
            raise ValueError(
                f"Unknown tier(s) {unknown}; choose from {list(self._sizes)}."
            )
        if not requested:
            raise ValueError(
                f"No tiers selected. Received `comma_separated`={comma_separated!r}."
            )
        self.selected_tiers = requested

    def qdf_sizes(self, *only_tiers: str) -> List[Any]:
        """
        Parametrisation for benchmarks that consume a long DataFrame.

        Parameters
        ----------
        *only_tiers : str
            Tiers this benchmark is limited to. Omit to allow every selected tier.

        Returns
        -------
        List[Any]
            One `pytest.param` per measured tier, each carrying its measurement budget.
        """
        return self._parameters(is_ticker_df=False, only_tiers=only_tiers)

    def ticker_df_sizes(self, *only_tiers: str) -> List[Any]:
        """
        Parametrisation for benchmarks that consume a wide DataFrame.

        Parameters
        ----------
        *only_tiers : str
            Tiers this benchmark is limited to. Omit to allow every selected tier.

        Returns
        -------
        List[Any]
            One `pytest.param` per measured tier, each carrying its measurement budget.
        """
        return self._parameters(is_ticker_df=True, only_tiers=only_tiers)

    def _parameters(self, is_ticker_df: bool, only_tiers: Tuple[str, ...]) -> List[Any]:
        """
        Build the pytest parameters for the selected tiers in one format.

        Parameters
        ----------
        is_ticker_df : bool
            Whether the benchmark consumes the wide format.
        only_tiers : Tuple[str, ...]
            Tiers this benchmark is limited to, empty for no limit.

        Returns
        -------
        List[Any]
            One `pytest.param` per measured tier, or a single explained skip when the
            limit and the selection share no tier.
        """
        tiers = [t for t in self.selected_tiers if not only_tiers or t in only_tiers]
        if not tiers:
            reason = (
                f"this benchmark is limited to {only_tiers}, and none of those tiers is in "
                f"the selected set {self.selected_tiers}"
            )
            return [
                pytest.param(None, marks=pytest.mark.skip(reason=reason), id="skipped")
            ]
        return [
            self._as_parameter(replace(self._sizes[t], is_ticker_df=is_ticker_df))
            for t in tiers
        ]

    @staticmethod
    def _as_parameter(size: PanelSize) -> Any:
        """
        One pytest parameter carrying a size and its measurement budget.

        Parameters
        ----------
        size : PanelSize
            The size to wrap.

        Returns
        -------
        Any
            A `pytest.param` identified by the tier name and marked with
            `pytest.mark.benchmark`.
        """
        return pytest.param(
            size,
            id=size.tier,
            marks=pytest.mark.benchmark(
                min_rounds=size.min_rounds, max_time=size.max_seconds
            ),
        )


PANEL_SIZES = PanelSizeCatalog(
    {
        "tiny": PanelSize(
            "tiny", date_count=250, min_rounds=25, max_seconds=1.0, cid_count=4, xcat_count=3
        ),
        "small": PanelSize(
            "small", date_count=1300, min_rounds=15, max_seconds=3.0, cid_count=10, xcat_count=8
        ),
        "medium": PanelSize(
            "medium", date_count=3500, min_rounds=8, max_seconds=10.0, cid_count=20, xcat_count=15
        ),
        "large": PanelSize(
            "large", date_count=5200, min_rounds=5, max_seconds=30.0, cid_count=40, xcat_count=30
        ),
    },
    default_tiers=("small", "medium"),
)

TARGET_OBSERVATION_COUNTS = {
    "tiny": 3_000,
    "small": 100_000,
    "medium": 1_000_000,
    "large": 6_000_000,
}
