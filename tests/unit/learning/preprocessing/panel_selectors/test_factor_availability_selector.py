import numpy as np
import pandas as pd
import pytest

from macrosynergy.learning.preprocessing.panel_selectors import (
    FactorAvailabilitySelector,
)


@pytest.fixture
def make_panel_df():
    """Factory fixture to create a MultiIndex (cid, date) DataFrame."""

    def _make(data_dict, cids, dates):
        idx = pd.MultiIndex.from_product([cids, dates], names=["cid", "real_date"])
        df = pd.DataFrame(data_dict, index=idx)
        return df

    return _make


class TestInit:
    def test_default_paraM(self):
        selector = FactorAvailabilitySelector()
        assert selector.min_cids == 2
        assert selector.min_periods == 36

    def test_custom_paraM(self):
        selector = FactorAvailabilitySelector(min_cids=5, min_periods=12)
        assert selector.min_cids == 5
        assert selector.min_periods == 12

    def test_invalid_min_cids(self):
        with pytest.raises(ValueError):
            FactorAvailabilitySelector(min_cids=-1)

    def test_invalid_min_periods(self):
        with pytest.raises(ValueError):
            FactorAvailabilitySelector(min_periods=-1)

    def test_invalid_min_cids_type(self):
        with pytest.raises(TypeError):
            FactorAvailabilitySelector(min_cids=3.33333)

    def test_invalid_min_periods_type(self):
        with pytest.raises(TypeError):
            FactorAvailabilitySelector(min_periods=7.1)


class TestDetermineFeatures:
    """Tests for the determine_features method."""

    def test_all_features_pass(self, make_panel_df):
        """All features have full coverage and should be selected."""
        cids = ["USD", "GBP", "JPY"]
        dates = pd.date_range("2000-01-01", periods=36, freq="M")

        df = make_panel_df(
            {
                "xcat1": np.random.randn(len(cids) * len(dates)),
                "xcat2": np.random.randn(len(cids) * len(dates)),
            },
            cids=cids,
            dates=dates,
        )
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=36)
        mask = selector.determine_features(df, y=None)

        assert mask.dtype == bool
        assert mask.tolist() == [True, True]

    def test_no_features_pass(self, make_panel_df):
        """No feature meets the availability threshold."""
        cids = ["USD"]
        dates = pd.date_range("2000-01-01", periods=5, freq="M")
        df = make_panel_df(
            {
                "xcat1": [np.nan] * len(cids) * len(dates),
                "xcat2": [np.nan] * len(cids) * len(dates),
            },
            cids=cids,
            dates=dates,
        )
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=3)
        mask = selector.determine_features(df, y=None)

        assert mask.tolist() == [False, False]

    def test_partial_selection(self, make_panel_df):
        """One feature passes, the other does not."""
        cids = ["USD", "GBP"]
        dates = pd.date_range("2000-01-01", periods=36, freq="M")
        n = len(cids) * len(dates)

        # xcat1: fully populated — should pass
        xcat1 = np.random.randn(n)

        # xcat2: only one cid has data — should fail with min_cids=2
        xcat2 = np.full(n, np.nan)
        xcat2[: len(dates)] = np.random.randn(len(dates))  # only "US"

        df = make_panel_df(
            {"xcat1": xcat1, "xcat2": xcat2},
            cids=cids,
            dates=dates,
        )
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=36)
        mask = selector.determine_features(df, y=None)

        assert mask.tolist() == [True, False]

    def test_min_periods_capped_by_dataset(self, make_panel_df):
        """
        If the dataset has fewer periods than min_periods, the threshold
        should be capped at the number of available periods.
        """
        cids = ["USD", "GBP"]
        dates = pd.date_range("2000-01-01", periods=10, freq="M")
        n = len(cids) * len(dates)
        df = make_panel_df(
            {"xcat1": np.random.randn(n)},
            cids=cids,
            dates=dates,
        )
        # min_periods=100 but only 10 dates exist — should still pass
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=100)
        mask = selector.determine_features(df, y=None)

        assert mask.tolist() == [True]

    def test_exact_threshold_not_met_cids(self, make_panel_df):
        """Feature has enough periods but one too few cids."""
        cids = ["USD", "GBP", "JPY"]
        dates = pd.date_range("2000-01-01", periods=10, freq="M")

        vals = []
        for cid in cids:
            for i, date in enumerate(dates):
                if cid == "USD" and i < 5:
                    vals.append(1.0)
                else:
                    vals.append(np.nan)

        df = make_panel_df({"xcat1": vals}, cids=cids, dates=dates)
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=5)
        mask = selector.determine_features(df, y=None)

        assert mask.tolist() == [False]

    def test_exact_threshold_not_met_periods(self, make_panel_df):
        """Feature has enough cids but one too few periods."""
        cids = ["USD", "GBP"]
        dates = pd.date_range("2000-01-01", periods=10, freq="M")

        vals = []
        for cid in cids:
            for i, date in enumerate(dates):
                # Both cids have data for only 4 out of 10 dates
                if i < 4:
                    vals.append(1.0)
                else:
                    vals.append(np.nan)

        df = make_panel_df({"xcat1": vals}, cids=cids, dates=dates)
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=5)
        mask = selector.determine_features(df, y=None)

        assert mask.tolist() == [False]


class TestFit:
    def test_invalid_X(self):
        with pytest.raises(TypeError):
            selector = FactorAvailabilitySelector()
            selector.fit(X=1)

        with pytest.raises(TypeError):
            selector = FactorAvailabilitySelector()
            selector.fit(X="X")

        with pytest.raises(TypeError):
            selector = FactorAvailabilitySelector()
            selector.fit(X=None)

        with pytest.raises(ValueError):
            selector = FactorAvailabilitySelector()
            selector.fit(X=pd.DataFrame(np.random.randn(10, 10)))

    def test_invalid_y(self, make_panel_df):
        dates = pd.date_range("2000-01-01", periods=5, freq="M")
        X = make_panel_df(
            {
                "xcat1": np.random.randn(len(dates)),
                "xcat2": np.random.randn(len(dates)),
            },
            cids=["CAD"],
            dates=dates,
        )

        with pytest.raises(TypeError):
            selector = FactorAvailabilitySelector()
            selector.fit(X=X, y=1)

        with pytest.raises(TypeError):
            selector = FactorAvailabilitySelector()
            selector.fit(X=X, y="y")

        with pytest.raises(TypeError):
            selector = FactorAvailabilitySelector()
            selector.fit(X=X, y=np.random.randn(20))

        with pytest.raises(ValueError):
            selector = FactorAvailabilitySelector()
            selector.fit(X=X, y=pd.Series([1] * 20))

    def test_all_features_pass(self, make_panel_df):
        """All features have full coverage and should be selected."""
        cids = ["USD", "GBP", "JPY"]
        dates = pd.date_range("2000-01-01", periods=36, freq="M")

        df = make_panel_df(
            {
                "xcat1": np.random.randn(len(cids) * len(dates)),
                "xcat2": np.random.randn(len(cids) * len(dates)),
            },
            cids=cids,
            dates=dates,
        )
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=36)
        selector.fit(X=df, y=None)
        mask = selector.mask

        assert mask.dtype == bool
        assert mask.tolist() == [True, True]

    def test_no_features_pass(self, make_panel_df):
        """No feature meets the availability threshold."""
        cids = ["USD"]
        dates = pd.date_range("2000-01-01", periods=5, freq="M")
        df = make_panel_df(
            {
                "xcat1": [np.nan] * len(cids) * len(dates),
                "xcat2": [np.nan] * len(cids) * len(dates),
            },
            cids=cids,
            dates=dates,
        )
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=3)
        selector.fit(X=df, y=None)

        assert selector.mask.tolist() == [False, False]

    def test_partial_selection(self, make_panel_df):
        """One feature passes, the other does not."""
        cids = ["USD", "GBP"]
        dates = pd.date_range("2000-01-01", periods=36, freq="M")
        n = len(cids) * len(dates)

        # xcat1: fully populated — should pass
        xcat1 = np.random.randn(n)

        # xcat2: only one cid has data — should fail with min_cids=2
        xcat2 = np.full(n, np.nan)
        xcat2[: len(dates)] = np.random.randn(len(dates))  # only "US"

        df = make_panel_df(
            {"xcat1": xcat1, "xcat2": xcat2},
            cids=cids,
            dates=dates,
        )
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=36)
        selector.fit(X=df, y=None)

        assert selector.mask.tolist() == [True, False]

    def test_min_periods_capped_by_dataset(self, make_panel_df):
        """
        If the dataset has fewer periods than min_periods, the threshold
        should be capped at the number of available periods.
        """
        cids = ["USD", "GBP"]
        dates = pd.date_range("2000-01-01", periods=10, freq="M")
        n = len(cids) * len(dates)
        df = make_panel_df(
            {"xcat1": np.random.randn(n)},
            cids=cids,
            dates=dates,
        )
        # min_periods=100 but only 10 dates exist — should still pass
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=100)
        selector.fit(X=df, y=None)

        assert selector.mask.tolist() == [True]

    def test_exact_threshold_not_met_cids(self, make_panel_df):
        """Feature has enough periods but one too few cids."""
        cids = ["USD", "GBP", "JPY"]
        dates = pd.date_range("2000-01-01", periods=10, freq="M")

        vals = []
        for cid in cids:
            for i, date in enumerate(dates):
                if cid == "USD" and i < 5:
                    vals.append(1.0)
                else:
                    vals.append(np.nan)

        df = make_panel_df({"xcat1": vals}, cids=cids, dates=dates)
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=5)
        selector.fit(X=df, y=None)

        assert selector.mask.tolist() == [False]

    def test_exact_threshold_not_met_periods(self, make_panel_df):
        """Feature has enough cids but one too few periods."""
        cids = ["USD", "GBP"]
        dates = pd.date_range("2000-01-01", periods=10, freq="M")

        vals = []
        for cid in cids:
            for i, date in enumerate(dates):
                # Both cids have data for only 4 out of 10 dates
                if i < 4:
                    vals.append(1.0)
                else:
                    vals.append(np.nan)

        df = make_panel_df({"xcat1": vals}, cids=cids, dates=dates)
        selector = FactorAvailabilitySelector(min_cids=2, min_periods=5)
        selector.fit(X=df, y=None)

        assert selector.mask.tolist() == [False]


class TestTransform:
    def test_transform_types(self, make_panel_df):
        """
        Test inputs of the transform method are checked for correctness.
        """
        dates = pd.date_range("2000-01-01", periods=5, freq="M")
        X = make_panel_df(
            {
                "xcat1": np.random.randn(len(dates)),
                "xcat2": np.random.randn(len(dates)),
            },
            cids=["CAD"],
            dates=dates,
        )

        selector = FactorAvailabilitySelector(min_cids=2, min_periods=5).fit(X)

        with pytest.raises(TypeError):
            selector.transform(X=1)

        with pytest.raises(TypeError):
            selector.transform(X="X")

        with pytest.raises(TypeError):
            selector.transform(X=X.values)

        with pytest.raises(ValueError):
            selector.transform(X.iloc[:, :-1])

        with pytest.raises(ValueError):
            selector.transform(X.reset_index())


    def test_transform_valid(self, make_panel_df):
        """
        Test that the transform method works as expected
        """
        cids = ["USD", "GBP", "JPY"]
        dates = pd.date_range("2000-01-01", periods=36, freq="M")

        X = make_panel_df(
            {
                "xcat1": np.random.randn(len(cids) * len(dates)),
                "xcat2": np.random.randn(len(cids) * len(dates)),
            },
            cids=cids,
            dates=dates,
        )

        selector = FactorAvailabilitySelector().fit(X)
        X_transformed = selector.transform(X)

        assert X_transformed.shape[1] == 2
        assert isinstance(X_transformed, pd.DataFrame)
        assert all(xcat in X.columns for xcat in X_transformed.columns)
