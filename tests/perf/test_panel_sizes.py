"""Tests for the objects in panel_sizes.py."""

from __future__ import annotations

import unittest
from dataclasses import replace

import pytest
from parameterized import parameterized

from tests.perf.panel_sizes import (
    PANEL_SIZES,
    TARGET_OBSERVATION_COUNTS,
    PanelSize,
    clear_df_cache,
)

OBSERVATION_TOLERANCE = 0.10
CHEAP_TIERS = ["tiny", "small"]
EXPENSIVE_TIERS = ["medium", "large"]


def a_panel_size(**overrides) -> PanelSize:
    """
    A small `PanelSize` for tests that do not care about the exact numbers.

    Parameters
    ----------
    **overrides
        Fields to replace on the default size.

    Returns
    -------
    PanelSize
        The default size with `overrides` applied.
    """
    defaults = dict(
        tier="test", date_count=10, min_rounds=1, max_seconds=1.0, cid_count=2, xcat_count=3
    )
    return PanelSize(**{**defaults, **overrides})


class TestPanelSizeValidation(unittest.TestCase):
    def test_rejects_both_ticker_forms(self):
        with self.assertRaisesRegex(ValueError, "not both forms and not neither"):
            a_panel_size(tickers=("AAA_XCAT000",))

    def test_rejects_neither_ticker_form(self):
        with self.assertRaisesRegex(ValueError, "not both forms and not neither"):
            PanelSize(tier="test", date_count=10, min_rounds=1, max_seconds=1.0)

    @parameterized.expand(
        [
            ("missing_xcat_count", {"cid_count": 2, "xcat_count": None}),
            ("missing_cid_count", {"cid_count": None, "xcat_count": 3}),
        ]
    )
    def test_rejects_one_half_of_the_parts_form(self, _name, overrides):
        with self.assertRaisesRegex(ValueError, "both `cid_count` and `xcat_count`"):
            PanelSize(
                tier="test", date_count=10, min_rounds=1, max_seconds=1.0, **overrides
            )

    def test_rejects_duplicate_tickers(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            a_panel_size(cid_count=None, xcat_count=None, tickers=("AAA_X", "AAA_X"))

    def test_rejects_non_positive_date_count(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            a_panel_size(date_count=0)


class TestPanelSizeCounting(unittest.TestCase):
    def test_ticker_count_from_parts(self):
        self.assertEqual(a_panel_size(cid_count=2, xcat_count=3).ticker_count, 6)

    def test_ticker_count_and_observations_from_explicit_tickers(self):
        size = a_panel_size(
            cid_count=None, xcat_count=None, tickers=("AAA_X", "BBB_Y")
        )
        self.assertEqual(size.ticker_count, 2)
        self.assertEqual(size.observation_count, 20)

    def test_shape_differs_between_formats(self):
        size = a_panel_size()
        self.assertEqual(size.qdf_shape, (60, 4))
        self.assertEqual(size.ticker_df_shape, (10, 6))
        self.assertEqual(size.shape, size.qdf_shape)
        self.assertEqual(
            replace(size, is_ticker_df=True).shape, size.ticker_df_shape
        )

    def test_df_format_names_the_selected_format(self):
        self.assertEqual(a_panel_size().df_format, "long")
        self.assertEqual(replace(a_panel_size(), is_ticker_df=True).df_format, "wide")

    def test_panel_size_is_usable_as_a_cache_key(self):
        self.assertEqual(len({a_panel_size(), a_panel_size()}), 1)

    def test_describe_reports_the_selected_format(self):
        described = replace(a_panel_size(), is_ticker_df=True).describe()
        self.assertEqual(described["df_format"], "wide")
        self.assertEqual(
            (described["row_count"], described["column_count"]), (10, 6)
        )
        self.assertEqual(described["metrics"], ["value"])


class TestPanelSizeCatalog(unittest.TestCase):
    def setUp(self) -> None:
        previous = PANEL_SIZES.selected_tiers
        self.addCleanup(setattr, PANEL_SIZES, "selected_tiers", previous)

    def test_tier_names_lists_every_registered_tier(self):
        self.assertEqual(
            PANEL_SIZES.tier_names, ("tiny", "small", "medium", "large")
        )

    @parameterized.expand(list(TARGET_OBSERVATION_COUNTS))
    def test_each_tier_lands_on_its_observation_target(self, tier):
        target = TARGET_OBSERVATION_COUNTS[tier]
        actual = PANEL_SIZES[tier].observation_count
        self.assertLessEqual(abs(actual - target) / target, OBSERVATION_TOLERANCE)

    def test_parameters_carry_the_measurement_budget(self):
        parameter = PANEL_SIZES.qdf_sizes("small")[0]
        marks = {mark.name: mark.kwargs for mark in parameter.marks}
        self.assertEqual(marks["benchmark"], {"min_rounds": 15, "max_time": 3.0})

    def test_ticker_df_sizes_are_marked_as_such(self):
        self.assertTrue(PANEL_SIZES.ticker_df_sizes("small")[0].values[0].is_ticker_df)
        self.assertFalse(PANEL_SIZES.qdf_sizes("small")[0].values[0].is_ticker_df)

    def test_a_tier_limit_outside_the_selection_yields_one_explained_skip(self):
        PANEL_SIZES.select_tiers("medium")
        parameters = PANEL_SIZES.qdf_sizes("tiny")
        self.assertEqual(len(parameters), 1)
        reason = parameters[0].marks[0].kwargs["reason"]
        self.assertIn("tiny", reason)
        self.assertIn("medium", reason)

    def test_select_tiers_names_an_unknown_tier(self):
        with self.assertRaisesRegex(ValueError, "enormous"):
            PANEL_SIZES.select_tiers("enormous")

    def test_select_tiers_falls_back_to_the_default(self):
        PANEL_SIZES.select_tiers("tiny")
        PANEL_SIZES.select_tiers(None)
        self.assertEqual(PANEL_SIZES.selected_tiers, ("small", "medium"))

    def test_select_tiers_accepts_a_comma_separated_list(self):
        PANEL_SIZES.select_tiers("tiny, large")
        self.assertEqual(PANEL_SIZES.selected_tiers, ("tiny", "large"))


def check_qdf_matches_its_size(case: unittest.TestCase, tier: str) -> None:
    """
    Assert the long DataFrame for a tier has the counts that tier derives.

    Parameters
    ----------
    case : unittest.TestCase
        The running test, used for its assertion methods.
    tier : str
        Name of the tier to build.

    Returns
    -------
    None
    """
    size = PANEL_SIZES[tier]
    df = size.as_qdf()
    case.assertEqual(df.shape, size.qdf_shape)
    case.assertEqual(len(df), size.observation_count)
    case.assertEqual(df["real_date"].nunique(), size.date_count)
    tickers = df["cid"].astype(str) + "_" + df["xcat"].astype(str)
    case.assertEqual(tickers.nunique(), size.ticker_count)


def check_ticker_df_matches_its_size(case: unittest.TestCase, tier: str) -> None:
    """
    Assert the wide DataFrame for a tier has the counts that tier derives.

    Parameters
    ----------
    case : unittest.TestCase
        The running test, used for its assertion methods.
    tier : str
        Name of the tier to build.

    Returns
    -------
    None
    """
    size = replace(PANEL_SIZES[tier], is_ticker_df=True)
    df = size.as_ticker_df()
    case.assertEqual(df.shape, size.ticker_df_shape)
    case.assertEqual(df.index.nunique(), size.date_count)
    case.assertEqual(df.shape[0] * df.shape[1], size.observation_count)


class TestDataFrameConstruction(unittest.TestCase):
    @parameterized.expand(CHEAP_TIERS)
    def test_qdf_matches_its_size(self, tier):
        check_qdf_matches_its_size(self, tier)

    @parameterized.expand(CHEAP_TIERS)
    def test_ticker_df_matches_its_size(self, tier):
        check_ticker_df_matches_its_size(self, tier)

    def test_built_frames_are_shared_and_copies_are_not(self):
        size = PANEL_SIZES["tiny"]
        self.assertIs(size.as_qdf(), size.as_qdf())
        copy = size.as_qdf_copy()
        self.assertIsNot(copy, size.as_qdf())
        copy.loc[0, "value"] = -999.0
        self.assertNotEqual(size.as_qdf().loc[0, "value"], -999.0)

    def test_clearing_the_cache_rebuilds(self):
        size = PANEL_SIZES["tiny"]
        first = size.as_qdf()
        clear_df_cache()
        self.assertIsNot(size.as_qdf(), first)


@pytest.mark.perf
class TestDataFrameConstructionAtScale(unittest.TestCase):
    """
    The same construction checks at the tiers whose frames are expensive to build.

    The mark sits on the class rather than on each method: pytest marks do not survive
    `parameterized.expand`, which replaces the decorated function with generated ones.
    """

    @parameterized.expand(EXPENSIVE_TIERS)
    def test_qdf_matches_its_size(self, tier):
        check_qdf_matches_its_size(self, tier)

    @parameterized.expand(EXPENSIVE_TIERS)
    def test_ticker_df_matches_its_size(self, tier):
        check_ticker_df_matches_its_size(self, tier)


if __name__ == "__main__":
    unittest.main()
