import numpy as np
import pandas as pd
import pytest

from macrosynergy.learning.sequential.base_panel_learner import (
    _create_long_format_df,
    _resolve_blacklists,
)
from macrosynergy.management.simulate import make_qdf


@pytest.fixture
def qdf():
    np.random.seed(42)
    cids = ["AUD", "CAD", "GBP", "USD"]
    xcats = ["XR", "CPI", "GROWTH", "RIR"]
    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2016-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2016-01-01", "2020-12-31", 0, 1]
    df_cids.loc["GBP"] = ["2016-01-01", "2020-12-31", 0, 1]
    df_cids.loc["USD"] = ["2016-01-01", "2020-12-31", 0, 1]

    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2016-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CPI"] = ["2016-01-01", "2020-12-31", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2016-01-01", "2020-12-31", 1, 2, 0.9, 1]
    df_xcats.loc["RIR"] = ["2016-01-01", "2020-12-31", -0.1, 2, 0.8, 0.3]

    df = make_qdf(df_cids, df_xcats, back_ar=0.75)
    df["value"] = df["value"].astype("float32")
    return df


class TestResolveBlacklists:
    def test_none_returns_list_of_nones(self):
        result = _resolve_blacklists(blacklist=None, n_targets=3)
        assert result == [None, None, None]

    def test_dict_is_replicated_n_targets_times(self):
        bl = {"USD": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"))}
        result = _resolve_blacklists(blacklist=bl, n_targets=2)
        assert result == [bl, bl]
        assert len(result) == 2

    def test_list_is_returned_as_is(self):
        bl1 = {"USD": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"))}
        bl2 = {"GBP": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-06-01"))}
        result = _resolve_blacklists(blacklist=[bl1, bl2], n_targets=2)
        assert result == [bl1, bl2]

    def test_single_target_with_dict(self):
        bl = {"USD": (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"))}
        result = _resolve_blacklists(blacklist=bl, n_targets=1)
        assert result == [bl]


class TestCreateLongFormatDf:
    def test_returns_dataframe_with_correct_columns_and_index(self, qdf):
        xcats = ["CPI", "GROWTH", "RIR"]

        result = _create_long_format_df(
            df=qdf,
            xcats=xcats,
            cids=["AUD", "CAD"],
            n_targets=1,
            start=None,
            end=None,
            blacklist=None,
            freq="M",
            lag=1,
            xcat_aggs=["last", "sum"],
        )

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["CPI", "GROWTH", "RIR"]
        assert result.index.names == ["cid", "real_date"]

    def test_single_target_with_blacklist(self, qdf):
        blacklist = {
            "AUD": (
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2019-01-01"),
            ),
        }

        result = _create_long_format_df(
            df=qdf,
            xcats=["CPI", "GROWTH", "RIR"],
            cids=["AUD", "CAD"],
            n_targets=1,
            start=None,
            end=None,
            blacklist=blacklist,
            freq="M",
            lag=1,
            xcat_aggs=["last", "sum"],
        )

        assert not result.empty
        assert result.loc[
            "AUD",
            "2018-01-01":"2019-01-01",
        ].empty

    def test_multi_target_columns_and_index(self, qdf):
        result = _create_long_format_df(
            df=qdf,
            xcats=["CPI", "GROWTH", "XR", "RIR"],
            cids=["AUD", "CAD"],
            n_targets=2,
            start=None,
            end=None,
            blacklist=None,
            freq="M",
            lag=1,
            xcat_aggs=["last", "sum"],
        )

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["CPI", "GROWTH", "XR", "RIR"]
        assert result.index.names == ["cid", "real_date"]

    def test_multi_target_with_single_blacklist_dict(self, qdf):
        blacklist = {
            "AUD": (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01")),
        }

        result = _create_long_format_df(
            df=qdf,
            xcats=["CPI", "GROWTH", "XR", "RIR"],
            cids=["AUD", "CAD"],
            n_targets=2,
            start=None,
            end=None,
            blacklist=blacklist,
            freq="M",
            lag=1,
            xcat_aggs=["last", "sum"],
        )

        assert not result.empty
        assert result[
            (result.index.get_level_values("cid") == "AUD")
            & (result.index.get_level_values("real_date") <= pd.Timestamp("2019-01-01"))
            & (result.index.get_level_values("real_date") >= pd.Timestamp("2018-01-01"))
        ].empty

    def test_multi_target_with_list_of_blacklists(self, qdf):
        bl1 = {
            "AUD": (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01")),
        }
        bl2 = {
            "CAD": (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-01-01")),
            "AUD": (pd.Timestamp("2018-08-01"), pd.Timestamp("2019-08-01")),
        }

        result = _create_long_format_df(
            df=qdf,
            xcats=["CPI", "GROWTH", "XR", "RIR"],
            cids=["AUD", "CAD"],
            n_targets=2,
            start=None,
            end=None,
            blacklist=[bl1, bl2],
            freq="M",
            lag=1,
            xcat_aggs=["last", "sum"],
        )

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["CPI", "GROWTH", "XR", "RIR"]

        def mask(cid, start, end):
            return (
                (result.index.get_level_values("cid") == cid)
                & (result.index.get_level_values("real_date") <= pd.Timestamp(end))
                & (result.index.get_level_values("real_date") >= pd.Timestamp(start))
            )

        assert result[mask("AUD", "2018-08-01", "2019-01-01")].empty
        assert result.loc[mask("AUD", "2018-01-01", "2018-07-31"), "XR"].isna().all()
        assert result.loc[mask("AUD", "2018-01-01", "2018-07-31"), "RIR"].notna().all()
        assert result.loc[mask("CAD", "2019-01-01", "2020-01-01"), "XR"].notna().all()
        assert result.loc[mask("CAD", "2019-01-01", "2020-01-01"), "RIR"].isna().all()
