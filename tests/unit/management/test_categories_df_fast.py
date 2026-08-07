import unittest
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (
    categories_df,
    categories_df_fast,
    categories_df_fast_loop,
)

CIDS: List[str] = ["AUD", "CAD", "GBP", "JPY", "USD"]
XCATS: List[str] = ["FTR1", "FTR2", "FTR3", "RET"]
BLACKLIST: Dict[str, Any] = {
    "JPY": ("2005-01-01", "2006-06-30"),
    "GBP_1": ("2004-01-01", "2004-12-31"),
    "GBP_2": ("2010-01-01", "2010-12-31"),
}


def make_panel(
    seed: int = 0,
    cids: Optional[List[str]] = None,
    xcats: Optional[List[str]] = None,
    start: str = "2003-01-01",
    end: str = "2012-12-31",
) -> pd.DataFrame:
    """A plain long panel of business-day observations with reproducible values."""

    rng = np.random.default_rng(seed)
    cids = CIDS if cids is None else cids
    xcats = XCATS if xcats is None else xcats
    dates = pd.bdate_range(start, end)
    df = pd.DataFrame(
        [(d, c, x) for c in cids for x in xcats for d in dates],
        columns=["real_date", "cid", "xcat"],
    )
    df["value"] = rng.normal(size=len(df))
    return df.reset_index(drop=True)


def panels() -> Dict[str, pd.DataFrame]:
    """One panel per structural hazard the two implementations have to agree on."""

    rng = np.random.default_rng(7)
    full = make_panel()
    out: Dict[str, pd.DataFrame] = {
        "categorical": QuantamentalDataFrame(full),
        "object_dtype": full,
        "single_cid": QuantamentalDataFrame(make_panel(cids=["AUD"])),
        "two_xcats": QuantamentalDataFrame(make_panel(xcats=["FTR1", "RET"])),
    }

    out["holes"] = QuantamentalDataFrame(
        full.drop(rng.choice(len(full), len(full) // 8, replace=False)).reset_index(
            drop=True
        )
    )

    all_nan = full.copy()
    all_nan.loc[all_nan["xcat"] == "FTR2", "value"] = np.nan
    out["all_nan_category"] = QuantamentalDataFrame(all_nan)

    nan_month = full.copy()
    in_month = (nan_month["cid"] == "AUD") & (
        nan_month["real_date"].dt.to_period("M") == pd.Period("2010-05")
    )
    nan_month.loc[in_month, "value"] = np.nan
    out["nan_month"] = QuantamentalDataFrame(nan_month)

    out["ragged"] = QuantamentalDataFrame(
        pd.concat(
            [
                make_panel(cids=[cid], start=start, end=end)
                for cid, start, end in [
                    ("AUD", "2003-01-01", "2012-12-31"),
                    ("CAD", "2007-01-01", "2012-12-31"),
                    ("GBP", "2003-01-01", "2008-12-31"),
                    ("JPY", "2011-01-01", "2012-12-31"),
                    ("USD", "2003-01-01", "2012-12-31"),
                ]
            ]
        ).reset_index(drop=True)
    )

    # a category carried by a single cross section, and a multi-year date gap
    lopsided = full[~((full["xcat"] == "FTR3") & (full["cid"] != "USD"))]
    out["lopsided_category"] = QuantamentalDataFrame(lopsided.reset_index(drop=True))
    gap = full[~full["real_date"].dt.year.isin([2006, 2007, 2008])]
    out["date_gap"] = QuantamentalDataFrame(gap.reset_index(drop=True))

    out["exact_duplicates"] = QuantamentalDataFrame(
        pd.concat([full, full.sample(200, random_state=1)]).reset_index(drop=True)
    )
    out["conflicting_duplicates"] = QuantamentalDataFrame(
        pd.concat(
            [full, full.sample(50, random_state=2).assign(value=999.0)]
        ).reset_index(drop=True)
    )

    # unused categories left behind by an external filter
    stale = QuantamentalDataFrame(full)
    out["stale_categories"] = stale[stale["cid"] != "USD"].reset_index(drop=True)

    extreme = full.copy()
    extreme.loc[extreme.index[:400], "value"] = np.inf
    extreme.loc[extreme.index[400:800], "value"] = -np.inf
    extreme.loc[extreme.index[800:1200], "value"] = 1e300
    out["extreme_values"] = QuantamentalDataFrame(extreme)

    return out


def argsets() -> List[Dict[str, Any]]:
    """The single-call argument matrix; `xcats` and `cids` default to the full lists."""

    return [
        dict(),
        dict(freq="M", lag=1, xcat_aggs=["last", "sum"]),
        dict(freq="D", lag=0, xcat_aggs=["last", "sum"]),
        dict(freq="W", lag=1, xcat_aggs=["mean", "mean"]),
        dict(freq="Q", lag=2, xcat_aggs=["mean", "sum"]),
        dict(freq="A", lag=1, xcat_aggs=["first", "last"]),
        dict(freq="M", lag=1, xcat_aggs=["median", "std"]),
        dict(freq="M", lag=1, xcat_aggs=["min", "max"]),
        dict(freq="M", lag=1, xcat_aggs=["sum", "sum"]),
        dict(freq="M", lag=1, fwin=3),
        dict(freq="M", lag=1, fwin=12),
        dict(freq="M", lag=24),
        dict(freq="M", lag=1000),
        dict(freq="M", lag=1, start="2007-01-01"),
        dict(freq="M", lag=1, end="2007-01-01"),
        dict(freq="M", lag=1, start="2005-01-01", end="2010-12-31"),
        dict(freq="M", lag=1, blacklist=BLACKLIST),
        dict(freq="M", lag=1, cids=None),
        dict(freq="M", lag=1, cids=[CIDS[0]]),
        dict(freq="M", lag=1, cids=CIDS + ["ZZZ"]),
        dict(freq="M", lag=1, xcats=["FTR1", "RET"]),
        dict(freq="M", lag=1, xcats=["FTR1", "ZZZ_NOPE", "RET"]),
        dict(freq="M", lag=1, xcats=["RET", "FTR1", "FTR2"]),
        dict(freq="M", lag=1, xcats=["FTR1", "FTR1", "RET"]),
    ]


def call(fn, *args, **kwargs):
    """Run `fn`, returning its result or the exception it raised."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the exception IS the result
            return exc


class ParityTestCase(unittest.TestCase):
    """Base class holding the exact-equality assertions every suite below uses."""

    def assertSameResult(self, expected: Any, actual: Any) -> None:
        if isinstance(expected, BaseException) or isinstance(actual, BaseException):
            self.assertIsInstance(
                expected, BaseException, f"categories_df returned {type(expected)}"
            )
            self.assertIsInstance(
                actual, BaseException, f"expected {type(expected).__name__}"
            )
            self.assertIs(type(actual), type(expected))
            return
        self.assertIsInstance(actual, pd.DataFrame)
        self.assertEqual(list(expected.columns), list(actual.columns))
        self.assertEqual(list(expected.index.names), list(actual.index.names))
        self.assertEqual(list(expected.index.dtypes), list(actual.index.dtypes))
        pd.testing.assert_frame_equal(
            expected,
            actual,
            check_exact=True,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
        )

    def assertParity(self, df: pd.DataFrame, **kwargs: Any) -> None:
        kwargs.setdefault("xcats", list(XCATS))
        kwargs.setdefault("cids", list(CIDS))
        self.assertSameResult(
            call(categories_df, df, **kwargs), call(categories_df_fast, df, **kwargs)
        )

    def assertBatchParity(self, df: pd.DataFrame, specs: List[Dict[str, Any]]) -> None:
        expected = [call(categories_df, df, **spec) for spec in specs]
        actual = call(categories_df_fast_loop, df, specs)
        self.assertIsInstance(actual, list)
        self.assertEqual(len(expected), len(actual))
        for i, (ref, got) in enumerate(zip(expected, actual)):
            with self.subTest(spec=i):
                self.assertSameResult(ref, got)


class TestArgumentMatrix(ParityTestCase):
    """Every argument set against every panel shape."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panels = panels()

    def test_matrix(self) -> None:
        for name, panel in self.panels.items():
            for i, kwargs in enumerate(argsets()):
                with self.subTest(panel=name, argset=i):
                    self.assertParity(panel, **kwargs)

    def test_value_column_variants(self) -> None:
        base = make_panel()
        base["grading"] = np.repeat([1.0, 2.0, 3.0], len(base) // 3 + 1)[: len(base)]
        base["eop_lag"] = np.arange(len(base)) % 40  # deliberately integer dtype
        for val in ("value", "grading", "eop_lag"):
            for aggs in (["mean", "mean"], ["sum", "sum"], ["last", "sum"]):
                for frame in (base, QuantamentalDataFrame(base)):
                    with self.subTest(
                        val=val, aggs=aggs, qdf=isinstance(frame, QuantamentalDataFrame)
                    ):
                        self.assertParity(
                            frame, val=val, xcat_aggs=aggs, freq="M", lag=1
                        )

    def test_weekend_dates_present(self) -> None:
        base = make_panel(end="2005-12-31")
        weekend = base.copy()
        weekend["real_date"] = weekend["real_date"] + pd.Timedelta(days=1)
        mixed = (
            pd.concat([base, weekend])
            .drop_duplicates(subset=["real_date", "cid", "xcat"])
            .reset_index(drop=True)
        )
        for freq in ("D", "W", "M", "Q", "A"):
            with self.subTest(freq=freq):
                self.assertParity(QuantamentalDataFrame(mixed), freq=freq, lag=1)

    def test_single_date_panel(self) -> None:
        one_day = make_panel(start="2010-06-15", end="2010-06-15")
        self.assertParity(QuantamentalDataFrame(one_day), freq="M", lag=1)
        self.assertParity(QuantamentalDataFrame(one_day), freq="D", lag=0)


class TestYears(ParityTestCase):
    """The multi-year aggregation branch."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.qdf = QuantamentalDataFrame(make_panel())
        cls.plain = make_panel()

    def test_parity(self) -> None:
        for frame in (self.qdf, self.plain):
            for years in (1, 2, 3, 5, 20):
                for aggs in (["mean", "mean"], ["sum", "sum"], ["last", "first"]):
                    for start in ("2003-01-01", "2005-06-30"):
                        with self.subTest(years=years, aggs=aggs, start=start):
                            self.assertParity(
                                frame,
                                xcats=["FTR1", "RET"],
                                years=years,
                                start=start,
                                xcat_aggs=aggs,
                            )

    def test_parity_with_filters(self) -> None:
        for kwargs in (
            dict(end="2008-12-31"),
            dict(cids=["AUD", "CAD"]),
            dict(cids=None),
            dict(blacklist=BLACKLIST),
            dict(freq="Q", fwin=4),  # both are accepted and silently ignored
        ):
            with self.subTest(**kwargs):
                self.assertParity(
                    self.qdf,
                    xcats=["FTR1", "RET"],
                    years=5,
                    start="2003-01-01",
                    **kwargs,
                )

    def test_index_and_dtypes(self) -> None:
        out = categories_df_fast(
            self.qdf, ["FTR1", "RET"], CIDS, years=5, start="2003-01-01"
        )
        self.assertEqual(list(out.index.names), ["cid", "real_date"])
        self.assertEqual(out.index.dtypes["real_date"], np.dtype("O"))
        self.assertEqual(
            sorted(set(out.index.get_level_values("real_date"))),
            ["2003 - 2007", "2008 - 2012"],
        )
        self.assertTrue(all(dtype == np.float64 for dtype in out.dtypes))

    def test_sorted_column_order(self) -> None:
        """`pivot` sorts the columns, so the dependent variable need not be last."""

        out = categories_df_fast(
            self.qdf, ["RET", "FTR1"], CIDS, years=5, start="2003-01-01"
        )
        self.assertEqual(list(out.columns), ["FTR1", "RET"])

    def test_sum_of_empty_bucket_is_zero(self) -> None:
        """No `min_count` on this branch, so an all-NaN bucket sums to 0.0, not NaN."""

        panel = make_panel()
        panel.loc[(panel["cid"] == "AUD") & (panel["xcat"] == "FTR1"), "value"] = np.nan
        qdf = QuantamentalDataFrame(panel)
        self.assertParity(
            qdf,
            xcats=["FTR1", "RET"],
            years=5,
            start="2003-01-01",
            xcat_aggs=["sum", "sum"],
        )
        out = categories_df_fast(
            qdf,
            ["FTR1", "RET"],
            CIDS,
            years=5,
            start="2003-01-01",
            xcat_aggs=["sum", "sum"],
        )
        self.assertTrue((out.xs("AUD", level="cid")["FTR1"] == 0.0).all())

    def test_preconditions(self) -> None:
        for kwargs, error in [
            (dict(years=2, start="2003-01-01", lag=1), AssertionError),
            (dict(years=2, start=None), AssertionError),
            (dict(years=2, start=pd.Timestamp("2003-01-01")), AssertionError),
            (dict(years=2, start="2003-01-01", xcats=list(XCATS)), AssertionError),
            (dict(years=2.0, start="2003-01-01"), TypeError),
            (dict(years=0, start="2003-01-01"), ZeroDivisionError),
            (dict(years="two", start="2003-01-01"), TypeError),
        ]:
            kwargs.setdefault("xcats", ["FTR1", "RET"])
            with self.subTest(**kwargs):
                self.assertParity(self.qdf, **kwargs)
                with self.assertRaises(error):
                    categories_df_fast(self.qdf, cids=CIDS, **kwargs)


class TestBatch(ParityTestCase):
    """`categories_df_fast_loop` against the same calls made one at a time."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.panels = panels()

    def _scenarios(self) -> Dict[str, List[Dict[str, Any]]]:
        base = dict(cids=list(CIDS), freq="M", lag=1, xcat_aggs=["last", "sum"])
        return {
            "single": [dict(xcats=list(XCATS), **base)],
            "identical": [dict(xcats=list(XCATS), **base)] * 3,
            "multi target": [
                dict(xcats=["FTR1", "FTR2", x], **base) for x in ("RET", "FTR3")
            ],
            "feature sweep": [
                dict(xcats=XCATS[:k] + ["RET"], **base) for k in (1, 2, 3)
            ],
            "varying cids": [
                dict(xcats=list(XCATS), cids=c, freq="M", lag=1)
                for c in (CIDS, CIDS[:2], CIDS[1:3], None)
            ],
            "varying freq": [
                dict(xcats=list(XCATS), cids=list(CIDS), freq=f, lag=1)
                for f in ("D", "W", "M", "Q", "A")
            ],
            "varying lag": [
                dict(xcats=list(XCATS), cids=list(CIDS), freq="M", lag=lag)
                for lag in (0, 1, 2, 6)
            ],
            "varying window": [
                dict(xcats=list(XCATS), start=start, blacklist=bl, **base)
                for start in (None, "2007-01-01")
                for bl in (None, BLACKLIST)
            ],
            "years mixed in": [
                dict(
                    xcats=["FTR1", "RET"], cids=list(CIDS), years=2, start="2003-01-01"
                ),
                dict(xcats=list(XCATS), **base),
                dict(
                    xcats=["FTR2", "RET"], cids=list(CIDS), years=5, start="2003-01-01"
                ),
            ],
            "everything varies": [
                dict(xcats=["FTR1", "RET"], cids=list(CIDS), freq="M", lag=1),
                dict(
                    xcats=list(XCATS),
                    cids=CIDS[:3],
                    freq="Q",
                    lag=2,
                    xcat_aggs=["mean", "sum"],
                ),
                dict(
                    xcats=["FTR1", "FTR2", "RET"],
                    cids=None,
                    freq="W",
                    lag=0,
                    xcat_aggs=["sum", "mean"],
                    start="2006-01-01",
                ),
                dict(
                    xcats=["FTR3", "RET"],
                    cids=list(CIDS),
                    freq="A",
                    lag=1,
                    blacklist=BLACKLIST,
                    fwin=2,
                ),
            ],
        }

    def test_scenarios(self) -> None:
        for panel_name, panel in self.panels.items():
            for name, specs in self._scenarios().items():
                with self.subTest(panel=panel_name, scenario=name):
                    self.assertBatchParity(panel, specs)

    def test_failing_specs_are_returned_positionally(self) -> None:
        panel = self.panels["categorical"]
        specs = [
            dict(xcats=list(XCATS), cids=list(CIDS)),
            dict(xcats=["ONLY_ONE"], cids=list(CIDS)),  # fewer than two categories
            dict(xcats=list(XCATS), cids=["ZZZ"]),  # no valid cross section
            dict(xcats=list(XCATS), cids=list(CIDS), freq="NOPE"),
            dict(xcats="not a list", cids=list(CIDS)),
            dict(xcats=list(XCATS), cids=list(CIDS), val="not_a_metric"),
            dict(xcats=list(XCATS), cids=list(CIDS), xcat_aggs=["mean"]),
        ]
        self.assertBatchParity(panel, specs)
        results = categories_df_fast_loop(panel, specs)
        self.assertIsInstance(results[0], pd.DataFrame)
        self.assertEqual(
            [type(r).__name__ for r in results[1:]],
            [
                "ValueError",
                "ValueError",
                "ValueError",
                "AssertionError",
                "AssertionError",
                "AssertionError",
            ],
        )

    def test_unusable_specs(self) -> None:
        panel = self.panels["categorical"]
        results = categories_df_fast_loop(
            panel,
            [
                dict(xcats=list(XCATS), cids=list(CIDS)),
                dict(xcats=list(XCATS), nonsense=1),
                dict(cids=list(CIDS)),
            ],
        )
        self.assertIsInstance(results[0], pd.DataFrame)
        self.assertIsInstance(results[1], TypeError)
        self.assertIn("nonsense", str(results[1]))
        self.assertIsInstance(results[2], TypeError)
        self.assertIn("xcats", str(results[2]))

    def test_empty_batch(self) -> None:
        self.assertEqual(categories_df_fast_loop(self.panels["categorical"], []), [])

    def test_generator_of_specs(self) -> None:
        panel = self.panels["categorical"]
        specs = [dict(xcats=list(XCATS), cids=list(CIDS), lag=lag) for lag in (0, 1)]
        self.assertEqual(len(categories_df_fast_loop(panel, iter(specs))), 2)

    def test_batch_matches_single_entry_point(self) -> None:
        panel = self.panels["ragged"]
        spec = dict(xcats=list(XCATS), cids=list(CIDS), freq="Q", lag=2)
        self.assertSameResult(
            categories_df_fast(panel, **spec), categories_df_fast_loop(panel, [spec])[0]
        )


class TestExceptionsAndWarnings(ParityTestCase):
    """Which inputs raise, which warn, and with what type."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.qdf = QuantamentalDataFrame(make_panel())
        cls.plain = make_panel()

    def test_rejected_arguments(self) -> None:
        for kwargs, error in [
            (dict(freq="NOPE"), ValueError),
            (dict(freq=1), TypeError),
            (dict(xcats="FTR1"), AssertionError),
            (dict(xcats=[1, 2]), AssertionError),
            (dict(xcat_aggs="mean"), AssertionError),
            (dict(xcat_aggs=["mean"]), AssertionError),
            (dict(xcat_aggs=["mean", "mean", "mean"]), AssertionError),
            (dict(xcat_aggs=[np.mean, np.mean]), AssertionError),
            (dict(val="close"), AssertionError),
            (dict(val="grading"), AssertionError),  # a metric, but not a column
            (dict(xcats=["FTR1"]), ValueError),
            (dict(xcats=["NOPE1", "NOPE2"]), ValueError),
            (dict(cids=["ZZZ"]), ValueError),
            (dict(start="2099-01-01"), ValueError),
        ]:
            with self.subTest(**kwargs):
                self.assertParity(self.qdf, **kwargs)
                with self.assertRaises(error):
                    categories_df_fast(
                        self.qdf,
                        kwargs.pop("xcats", list(XCATS)),
                        kwargs.pop("cids", list(CIDS)),
                        **kwargs,
                    )

    def test_non_quantamental_frame(self) -> None:
        for bad in (
            pd.DataFrame({"a": [1]}),
            self.plain.drop(columns=["value"]),
            self.plain.rename(columns={"real_date": "date"}),
            42,
        ):
            with self.subTest(kind=type(bad).__name__):
                self.assertSameResult(
                    call(categories_df, bad, list(XCATS), list(CIDS)),
                    call(categories_df_fast, bad, list(XCATS), list(CIDS)),
                )
                with self.assertRaises(TypeError):
                    categories_df_fast(bad, list(XCATS), list(CIDS))

    def test_conflicting_duplicate_keys_raise(self) -> None:
        base = make_panel()
        clashing = pd.concat(
            [base, base.sample(20, random_state=3).assign(value=999.0)]
        ).reset_index(drop=True)
        for frame in (clashing, QuantamentalDataFrame(clashing)):
            with self.subTest(qdf=isinstance(frame, QuantamentalDataFrame)):
                self.assertParity(frame, freq="M", lag=1)
                with self.assertRaises(ValueError):
                    categories_df_fast(frame, list(XCATS), list(CIDS))

    def test_exact_duplicate_keys_are_dropped(self) -> None:
        base = make_panel()
        doubled = pd.concat([base, base.sample(200, random_state=4)]).reset_index(
            drop=True
        )
        for frame in (doubled, QuantamentalDataFrame(doubled)):
            with self.subTest(qdf=isinstance(frame, QuantamentalDataFrame)):
                self.assertParity(frame, freq="M", lag=1)

    def test_missing_categories_warn(self) -> None:
        with self.assertWarns(UserWarning) as caught:
            categories_df_fast(self.qdf, ["FTR1", "NOPE", "RET"], CIDS)
        self.assertIn("categories are missing", str(caught.warning))

    def test_missing_cross_sections_warn(self) -> None:
        with self.assertWarns(UserWarning) as caught:
            categories_df_fast(self.qdf, list(XCATS), CIDS + ["ZZZ"])
        self.assertIn("cross sections are missing", str(caught.warning))

    def test_cids_none_never_warns(self) -> None:
        """The cross-section warning is gated on `cids` having been given."""

        subset = QuantamentalDataFrame(make_panel(cids=["AUD", "CAD"]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            categories_df_fast(subset, list(XCATS), None)
        self.assertEqual([], [w for w in caught if "cross sections" in str(w.message)])


class TestBlacklistBehaviour(ParityTestCase):
    """The blacklist paths, which differ between the two `reduce_df` bodies."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.qdf = QuantamentalDataFrame(make_panel())
        cls.plain = make_panel()

    def test_wellformed(self) -> None:
        for blacklist in (
            BLACKLIST,
            {},
            {"AUD": [pd.Timestamp("2005-01-01"), pd.Timestamp("2006-01-01")]},
            {"ZZZ": ("2005-01-01", "2006-01-01")},  # matches nothing
            {"AUD": ("2099-01-01", "2099-12-31")},  # entirely outside the panel
        ):
            for frame in (self.qdf, self.plain):
                with self.subTest(blacklist=blacklist, qdf=frame is self.qdf):
                    self.assertParity(frame, blacklist=blacklist, freq="M", lag=1)

    def test_unparseable_bound_raises_per_frame_type(self) -> None:
        """The bound is compared raw on one path and parsed on the other."""

        blacklist = {"AUD": ("not-a-date", "2006-01-01")}
        self.assertParity(self.qdf, blacklist=blacklist)
        self.assertParity(self.plain, blacklist=blacklist)
        with self.assertRaises(TypeError):
            categories_df_fast(self.qdf, list(XCATS), CIDS, blacklist=blacklist)
        with self.assertRaises(ValueError):  # DateParseError, a ValueError subclass
            categories_df_fast(self.plain, list(XCATS), CIDS, blacklist=blacklist)

    def test_malformed_blacklists(self) -> None:
        for blacklist in (
            ["AUD"],
            "AUD",
            {1: ("2005-01-01", "2006-01-01")},
            {"AUD": 5},
            {"AUD": ("2005-01-01",)},
            {"AUD": ("2005-01-01", "2006-01-01", "2007-01-01")},
        ):
            for frame in (self.qdf, self.plain):
                with self.subTest(blacklist=blacklist, qdf=frame is self.qdf):
                    self.assertParity(frame, blacklist=blacklist)

    def test_malformed_blacklist_in_a_batch(self) -> None:
        """An unhashable blacklist must not be used to key the shared row cache."""

        specs = [
            dict(xcats=list(XCATS), cids=list(CIDS)),
            dict(xcats=list(XCATS), cids=list(CIDS), blacklist=["AUD"]),
            dict(xcats=list(XCATS), cids=list(CIDS), blacklist=BLACKLIST),
        ]
        for frame in (self.qdf, self.plain):
            with self.subTest(qdf=frame is self.qdf):
                self.assertBatchParity(frame, specs)


class TestSharedReshapeInvariants(ParityTestCase):
    """The properties that make one reshape safe to share across many requests."""

    def test_union_only_category_does_not_shift_the_lag(self) -> None:
        """A period only a union-only category occupies must not enter `shift(lag)`."""

        covered = pd.bdate_range("2010-01-01", "2010-06-30").union(
            pd.bdate_range("2010-09-01", "2010-12-31")
        )
        rows = [
            (d, "AUD", x, 1.0 + i)
            for x in ("FTR1", "RET")
            for i, d in enumerate(covered)
        ]
        rows += [
            (d, "AUD", "ODD", 5.0) for d in pd.bdate_range("2010-07-01", "2010-07-31")
        ]
        panel = QuantamentalDataFrame(
            pd.DataFrame(rows, columns=["real_date", "cid", "xcat", "value"])
        )
        specs = [
            dict(xcats=["FTR1", "RET"], cids=["AUD"], freq="M", lag=lag)
            for lag in (0, 1, 2)
        ]
        specs += [dict(xcats=["ODD", "RET"], cids=["AUD"], freq="M", lag=1)]
        self.assertBatchParity(panel, specs)

        # the July period exists in the union but not in the FTR1/RET request
        lagged = categories_df_fast_loop(panel, specs)[1]
        months = lagged.index.get_level_values("real_date").month
        self.assertNotIn(7, list(months))

    def test_forward_window_is_not_grouped_by_cid(self) -> None:
        """`fwin` rolls the whole dependent column, so cross sections bleed together."""

        panel = QuantamentalDataFrame(make_panel(cids=["AUD", "CAD"], end="2005-12-31"))
        self.assertParity(panel, cids=["AUD", "CAD"], fwin=3, freq="M", lag=1)
        out = categories_df_fast(panel, list(XCATS), ["AUD", "CAD"], fwin=3, freq="M")
        last_aud = out.xs("AUD", level="cid")["RET"]
        # a per-cid forward window would leave the final fwin-1 periods NaN
        self.assertFalse(last_aud.iloc[-2:].isna().any())
        last_cad = out.xs("CAD", level="cid")["RET"]
        self.assertTrue(last_cad.iloc[-2:].isna().all())

    def test_reduce_order_differs_between_frame_types(self) -> None:
        """The two `reduce_df` bodies derive the surviving categories differently."""

        shared = make_panel(
            cids=["AUD", "CAD"], xcats=["FTR1", "RET"], end="2005-12-31"
        )
        cad_only = make_panel(seed=1, cids=["CAD"], xcats=["ONLYCAD"], end="2005-12-31")
        plain = pd.concat([shared, cad_only]).reset_index(drop=True)
        qdf = QuantamentalDataFrame(plain)
        request = dict(xcats=["FTR1", "ONLYCAD", "RET"], cids=["AUD"])

        # the QDF body derives xcats before the cid filter, so ONLYCAD survives with
        # no column of its own; the plain body filters first and simply warns
        self.assertIsInstance(call(categories_df, qdf, **request), KeyError)
        self.assertIsInstance(call(categories_df, plain, **request), pd.DataFrame)
        self.assertParity(qdf, **request)
        self.assertParity(plain, **request)

    def test_bin_labels_cover_empty_periods(self) -> None:
        """A date gap must not renumber the bins that follow it."""

        base = make_panel(end="2012-12-31")
        gapped = base[~base["real_date"].dt.year.isin([2005, 2006, 2007, 2008])]
        for frame in (
            QuantamentalDataFrame(gapped.reset_index(drop=True)),
            gapped.reset_index(drop=True),
        ):
            for freq in ("D", "W", "M", "Q", "A"):
                for lag in (0, 1, 3):
                    with self.subTest(
                        freq=freq, lag=lag, qdf=isinstance(frame, QuantamentalDataFrame)
                    ):
                        self.assertParity(frame, freq=freq, lag=lag)

    def test_nan_cross_section_raises(self) -> None:
        """A NaN cid reaches `sorted(...)` in the reduction and fails there."""

        panel = make_panel(end="2004-12-31")
        panel.loc[panel.index[:50], "cid"] = np.nan
        self.assertSameResult(
            call(categories_df, panel, list(XCATS), None),
            call(categories_df_fast, panel, list(XCATS), None),
        )
        with self.assertRaises(TypeError):
            categories_df_fast(panel, list(XCATS), None)

    def test_intraday_timestamps(self) -> None:
        """Rows are addressed by day, so a timestamp with a time falls back cleanly."""

        panel = make_panel(end="2004-12-31")
        panel["real_date"] = panel["real_date"] + pd.Timedelta(hours=9)
        for frame in (panel, QuantamentalDataFrame(panel)):
            for freq in ("D", "M", "Q"):
                with self.subTest(
                    freq=freq, qdf=isinstance(frame, QuantamentalDataFrame)
                ):
                    self.assertParity(frame, freq=freq, lag=1)

    def test_non_nanosecond_and_tz_aware_dates(self) -> None:
        """Both pass the QuantamentalDataFrame duck-type but not the block's
        nanosecond addressing."""

        base = make_panel(end="2004-12-31")
        millisecond = base.assign(real_date=base["real_date"].astype("datetime64[ms]"))
        tz_aware = base.assign(real_date=base["real_date"].dt.tz_localize("UTC"))
        for name, panel in (("ms", millisecond), ("tz", tz_aware)):
            for frame in (panel, QuantamentalDataFrame(panel)):
                with self.subTest(
                    dtype=name, qdf=isinstance(frame, QuantamentalDataFrame)
                ):
                    self.assertParity(frame, freq="M", lag=1)

    def test_sparse_panel_uses_the_factorize_branch(self) -> None:
        """A panel far sparser than its row-id space still has to be exact."""

        rng = np.random.default_rng(11)
        cids = [f"C{i:03d}" for i in range(120)]
        dates = pd.bdate_range("2003-01-01", "2012-12-31")
        rows = []
        for cid in cids:
            for day in rng.choice(len(dates), size=6, replace=False):
                for xcat in ("FTR1", "RET"):
                    rows.append((dates[day], cid, xcat, rng.normal()))
        panel = pd.DataFrame(rows, columns=["real_date", "cid", "xcat", "value"])
        for frame in (panel, QuantamentalDataFrame(panel)):
            for lag in (0, 1):
                with self.subTest(
                    lag=lag, qdf=isinstance(frame, QuantamentalDataFrame)
                ):
                    self.assertSameResult(
                        call(categories_df, frame, ["FTR1", "RET"], cids, lag=lag),
                        call(categories_df_fast, frame, ["FTR1", "RET"], cids, lag=lag),
                    )

    def test_wide_row_branches_agree(self) -> None:
        """The seen-mask and factorize branches of the row index must be identical."""

        from macrosynergy.management.utils.categories_df_fast import (
            _dfw_row_positions,
        )

        rng = np.random.default_rng(5)
        n_row_ids = 4000
        for n_filled in (1, 37, 4000):
            row_id = rng.choice(n_row_ids, size=n_filled, replace=False).repeat(3)
            rng.shuffle(row_id)
            seen = _dfw_row_positions(row_id, n_row_ids, len(row_id), np.int32)
            factorized = _dfw_row_positions(row_id, n_row_ids, 1, np.int32)
            with self.subTest(filled=n_filled):
                np.testing.assert_array_equal(seen[0], factorized[0])
                np.testing.assert_array_equal(seen[1], factorized[1])
                np.testing.assert_array_equal(seen[0][seen[1]], row_id)

    def test_non_alphabetical_categorical_order(self) -> None:
        """`pivot` honours a categorical level's order only while every level is used."""

        panel = make_panel(end="2005-12-31")
        reversed_cids = pd.CategoricalDtype(sorted(CIDS, reverse=True), ordered=False)
        panel["cid"] = panel["cid"].astype(reversed_cids)
        panel["xcat"] = panel["xcat"].astype("category")
        for kwargs in (
            dict(cids=list(CIDS)),
            dict(cids=CIDS[:2]),
            dict(cids=None),
            dict(xcats=["FTR1", "RET"]),
        ):
            with self.subTest(**kwargs):
                self.assertParity(panel, freq="M", lag=1, **kwargs)

    def test_index_level_is_object_for_both_frame_types(self) -> None:
        for frame in (
            make_panel(end="2004-12-31"),
            QuantamentalDataFrame(make_panel(end="2004-12-31")),
        ):
            out = categories_df_fast(frame, list(XCATS), CIDS)
            self.assertEqual(out.index.dtypes["cid"], np.dtype("O"))
            self.assertEqual(out.index.dtypes["real_date"], np.dtype("<M8[ns]"))


class TestModuleContract(unittest.TestCase):
    """The module is a competing pathway, not a wrapper around the shipped one."""

    def test_imports_nothing_from_the_shipped_implementation(self) -> None:
        import ast
        import inspect
        import sys

        tree = ast.parse(inspect.getsource(sys.modules[categories_df_fast.__module__]))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        self.assertEqual(
            {name for name in imported if name.startswith("macrosynergy")},
            {"macrosynergy.management.types", "macrosynergy.management.utils.core"},
        )

    def test_signature_matches_categories_df(self) -> None:
        import inspect

        shipped = inspect.signature(categories_df)
        fast = inspect.signature(categories_df_fast)
        self.assertEqual(list(shipped.parameters), list(fast.parameters))
        self.assertEqual(
            [p.default for p in shipped.parameters.values()],
            [p.default for p in fast.parameters.values()],
        )


if __name__ == "__main__":
    unittest.main()
