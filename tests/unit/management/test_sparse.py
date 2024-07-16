import unittest
import unittest.mock
import pandas as pd
import warnings
import numpy as np

from typing import List, Tuple, Dict, Union, Set, Any
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import concat_single_metric_qdfs, ticker_df_to_qdf
from macrosynergy.management.utils.sparse import (
    InformationStateChanges,
    VolatilityEstimationMethods,
    create_delta_data,
    calculate_score_on_sparse_indicator,
    infer_frequency,
    _remove_insignificant_values,
    _isc_dict_to_frames,
    _get_metric_df_from_isc,
    temporal_aggregator_exponential,
    temporal_aggregator_mean,
    temporal_aggregator_period,
)
import random
import string

FREQ_STR_MAP = {
    "B": "daily",
    "W-FRI": "weekly",
    "BM": "monthly",
    "BQ": "quarterly",
    "BA": "annual",
}


def random_string(
    length: int = 4,
) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=length))


def get_end_of_period_for_date(
    date: pd.Timestamp,
    freq: str,
) -> pd.Timestamp:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.bdate_range(start=date, periods=1, freq=freq)[0]


def concat_ts_list(
    ts_list: List[pd.Series],
    metric: str,
) -> pd.DataFrame:
    df = pd.concat(ts_list, axis=1)
    df.index.name, df.columns.name = "real_date", "ticker"
    return ticker_df_to_qdf(df, metric=metric)


def get_long_format_data(
    cids: List[str] = ["USD", "EUR", "JPY", "GBP"],
    start: str = "2010-01-01",
    end: str = "2020-01-21",
    xcats: List[str] = ["GDP", "CPI", "UNEMP", "RATE"],
    num_freqs: int = 2,
) -> pd.DataFrame:
    # Map of frequency codes to their descriptive names
    freq_map = FREQ_STR_MAP.copy()
    full_date_range = pd.bdate_range(start=start, end=end)
    get_random_freq = lambda: random.choice(list(freq_map.keys()))

    # Generate ticker symbols by combining currency ids and category names
    tickers = [f"{cid}_{xc}" for cid in cids for xc in xcats]
    ticker_freq_tuples = [
        (ticker, get_random_freq())
        for ticker in tickers
        for num_freq in range(num_freqs)
    ]

    # Generate time series data for each (ticker, frequency) tuple
    values_ts_list: List[pd.Series] = []
    eop_ts_list: List[pd.Series] = []
    eoplag_ts_list: List[pd.Series] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)

        for ticker, freq in ticker_freq_tuples:
            dts = pd.bdate_range(start=start, end=end, freq=freq)
            namex = f"{ticker}_{freq_map[freq].upper()}"
            ts = pd.Series(np.random.random(len(dts)), index=dts, name=namex)
            ts = ts.reindex(full_date_range).ffill()
            ts.loc[ts.isna()] = np.random.random(ts.isna().sum())
            values_ts_list.append(ts)

            tseop = pd.Series(
                dict([(dt, get_end_of_period_for_date(dt, freq)) for dt in dts]),
                name=namex,
            )
            tseop = tseop.reindex(full_date_range).ffill()
            tseop.loc[tseop.isna()] = min(full_date_range)
            eop_ts_list.append(tseop)

            _tdf = tseop.to_frame().reset_index()
            _tdf["eop_lag"] = _tdf.apply(
                lambda x: abs((x["index"].date() - x[tseop.name].date()).days),
                axis=1,
            )
            tseop = pd.Series(
                _tdf["eop_lag"].values, index=_tdf["index"], name=tseop.name
            )
            eoplag_ts_list.append(tseop)

    return concat_single_metric_qdfs(
        [
            concat_ts_list(ts_list=values_ts_list, metric="value"),  # good
            concat_ts_list(ts_list=eop_ts_list, metric="eop"),
            concat_ts_list(ts_list=eoplag_ts_list, metric="eop_lag"),
        ]
    )


def _get_helper_col(df: QuantamentalDataFrame) -> QuantamentalDataFrame:
    assert isinstance(df, QuantamentalDataFrame)
    return df["cid"] + "-" + df["xcat"] + "-" + df["real_date"].dt.strftime("%Y-%m-%d")


class TestFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.qdf_small = get_long_format_data(end="2012-01-01")

    def test_create_delta_data(self) -> None:
        qdf = self.qdf_small.copy()
        res = create_delta_data(df=qdf, return_density_stats=True)
        self.assertTrue(isinstance(res, tuple))
        self.assertTrue(isinstance(res[0], dict))
        self.assertTrue(isinstance(res[1], pd.DataFrame))
        self.assertEqual(
            set(res[1].columns), set(["ticker", "changes_density", "date_range"])
        )

        res = create_delta_data(df=qdf, return_density_stats=False)
        self.assertTrue(isinstance(res, dict))

        with self.assertRaises(ValueError):
            create_delta_data(df=1)
        with self.assertRaises(ValueError):
            create_delta_data(df=qdf, return_density_stats="yes")

        # drop value and assert value error
        with self.assertRaises(ValueError):
            create_delta_data(df=qdf.drop(columns="value"))

        # test with missing eop_lag
        create_delta_data(qdf.drop(columns="eop_lag"))

    def test_calculate_score_on_sparse_indicator(self) -> None:
        qdf = self.qdf_small.copy()
        argsdict = dict(
            isc=create_delta_data(df=qdf),
            std="std",
            halflife=5,
            min_periods=10,
            isc_version=0,
            iis=True,
            custom_method=None,
            custom_method_kwargs={},
        )
        res = calculate_score_on_sparse_indicator(**argsdict)
        self.assertTrue(isinstance(res, dict))

        # test with custom method
        res = calculate_score_on_sparse_indicator(
            **{**argsdict, "custom_method": VolatilityEstimationMethods.std}
        )

        # try with custom method = 'banana'
        with self.assertRaises(TypeError):
            calculate_score_on_sparse_indicator(
                **{**argsdict, "custom_method": "banana"}
            )

        with self.assertRaises(TypeError):
            calculate_score_on_sparse_indicator(
                **{
                    **argsdict,
                    "custom_method": VolatilityEstimationMethods.std,
                    "custom_method_kwargs": "banana",
                }
            )

        with self.assertRaises(ValueError):
            calculate_score_on_sparse_indicator(**{**argsdict, "std": "banana"})

    def test_remove_insignificant_values(self) -> None:
        wdf = pd.DataFrame(
            columns=["A", "B", "C"],
            data=0,
            index=pd.bdate_range(start="2010-01-01", end="2012-01-01"),
        )
        rdf: pd.DataFrame = _remove_insignificant_values(wdf)
        self.assertTrue(rdf.isna().all().all())

    def test_isc_dict_to_frames(self) -> None:
        qdf = self.qdf_small.copy()
        isc = create_delta_data(df=qdf)
        res = _isc_dict_to_frames(isc)
        self.assertTrue(isinstance(res, list))
        self.assertTrue(all([isinstance(x, pd.DataFrame) for x in res]))

        isc_copy = isc.copy()

        isc_copy[list(isc_copy.keys())[0]] = isc_copy[list(isc_copy.keys())[0]].loc[
            isc_copy[list(isc_copy.keys())[0]].index > "2015-01-01"
        ]
        # check that this will warn
        with self.assertWarns(UserWarning):
            _isc_dict_to_frames(isc_copy)

        isc_copy = isc.copy()
        isc_copy[list(isc_copy.keys())[0]] = set()
        with self.assertRaises(AssertionError):
            _isc_dict_to_frames(isc_copy)

        # now try with non-existant metric
        isc_copy = isc.copy()
        with self.assertRaises(AssertionError):
            _isc_dict_to_frames(isc_copy, metric="banana")

    def test_get_metric_df_from_isc(self) -> None:
        qdf = self.qdf_small.copy()
        dtr = pd.DatetimeIndex(qdf["real_date"].unique())
        isc = create_delta_data(df=qdf)
        res: pd.DataFrame = _get_metric_df_from_isc(
            isc=isc, metric="value", date_range=dtr
        )
        self.assertTrue(isinstance(res, pd.DataFrame))
        self.assertTrue(res.index.name == "real_date")
        self.assertTrue(res.columns.name == "ticker")

        # check raises type error if fill!=int, or 'fill'
        with self.assertRaises(ValueError):
            _get_metric_df_from_isc(isc, metric="value", date_range=dtr, fill="banana")

        with self.assertRaises(TypeError):
            # pass a function instead of a str/Number
            _get_metric_df_from_isc(
                isc, metric="value", date_range=dtr, fill=create_delta_data
            )

    def test_infer_frequency(self) -> None:
        qdf = InformationStateChanges.from_qdf(self.qdf_small).to_qdf()
        # drop where
        _tickers: List[str] = (qdf["cid"] + "_" + qdf["xcat"]).unique().tolist()
        res = infer_frequency(qdf)
        self.assertTrue(isinstance(res, pd.Series))
        self.assertTrue(set(res.index) == set(_tickers))

        for tx in _tickers:
            self.assertTrue(tx in res.index)
            found_freq = tx.split("_")[-1][0].upper()
            self.assertEqual(res[tx], found_freq)


class TestTemporalAggregators(unittest.TestCase):
    def setUp(self) -> None:
        self.qdf = get_long_format_data(end="2011-01-01")

    def test_temporal_aggregator_exponential(self) -> None:
        qdf = self.qdf.copy()
        hl = 5
        res = temporal_aggregator_exponential(qdf, halflife=hl, winsorise=True)
        self.assertIsInstance(res, QuantamentalDataFrame)

        # check that all xcats end with f"EWM{halflife:d}D"
        self.assertTrue(all([str(x).endswith(f"EWM{hl:d}D") for x in res["xcat"]]))

    def test_temporal_aggregator_mean(self) -> None:
        qdf = self.qdf.copy()
        wd = 10
        res = temporal_aggregator_mean(qdf, window=wd, winsorise=True)
        self.assertIsInstance(res, QuantamentalDataFrame)

        # check that all xcats end with  f"MA{window:d}D"
        self.assertTrue(all([str(x).endswith(f"MA{wd:d}D") for x in res["xcat"]]))

    def test_temporal_aggregator_period(self) -> None:
        qdf = self.qdf.copy()
        start, end = qdf["real_date"].min(), qdf["real_date"].max()
        postfix = "_pineapple"

        isc = create_delta_data(df=qdf)
        for k, v in isc.items():
            v["zscore_norm_squared"] = np.random.random(len(v))
            isc[k] = v

        res = temporal_aggregator_period(
            isc=isc,
            start=start,
            end=end,
            postfix=postfix,
        )

        self.assertIsInstance(res, QuantamentalDataFrame)


class TestVolatilityEstimationMethods(unittest.TestCase):
    def setUp(self) -> None:
        dts = pd.bdate_range(start="2010-01-01", end="2020-01-01")
        self.ts = pd.Series(data=np.random.random(len(dts)), index=dts)
        self.methods: List[str] = ["std", "abs", "exp", "exp_abs"]

    def test_method_calls(self) -> None:
        kwrgs = {"halflife": 5, "min_periods": 10}
        for method in self.methods:
            res = VolatilityEstimationMethods[method](self.ts, **kwrgs)
            self.assertIsInstance(res, pd.Series)

        # try getting banana, should raise key error
        with self.assertRaises(KeyError):
            VolatilityEstimationMethods["banana"](self.ts, **kwrgs)


class TestInformationStateChanges(unittest.TestCase):

    def test_class_methods(self) -> None:
        # test the class methods
        qdf = get_long_format_data(end="2012-01-01")
        isc = InformationStateChanges.from_qdf(qdf)
        self.assertTrue(isinstance(isc, InformationStateChanges))
        self.assertTrue(isinstance(isc.isc_dict, dict))

        self.assertEqual(set(isc.isc_dict.keys()), set(isc.keys()))
        # can't directly check the values and items as they are not hashable
        self.assertEqual(str(isc), str(isc.isc_dict))
        self.assertEqual(repr(isc), repr(isc.isc_dict))

        first_dict_key = list(isc.isc_dict.keys())[0]
        self.assertTrue((isc[first_dict_key]).equals(isc.isc_dict[first_dict_key]))

        self.assertEqual(len(list(isc.items())), len(list(isc.isc_dict.items())))
        self.assertEqual(len(list(isc.values())), len(list(isc.isc_dict.values())))

        # test __setitem__
        isc["banana"] = pd.DataFrame()
        self.assertTrue("banana" in isc.isc_dict.keys())
        self.assertTrue(isinstance(isc.isc_dict["banana"], pd.DataFrame))

    def test_isc_object_round_trip(self) -> None:

        qdfidx = QuantamentalDataFrame.IndexCols

        df = get_long_format_data()
        dfc = df.copy()
        tdf = InformationStateChanges.from_qdf(df).to_qdf()

        diff_mask = abs(df["value"].diff())
        diff_mask: pd.Series = diff_mask > 0.0  # type: ignore
        diff_df: pd.DataFrame = tdf.loc[diff_mask, :].reset_index(drop=True)

        # create helper column which is cid-xcat-real_date
        diff_df.loc[:, "helper"] = _get_helper_col(diff_df)
        dfc.loc[:, "helper"] = _get_helper_col(dfc)

        # reduce the diff_df to only the diffs (helpers)
        dfc = dfc[dfc["helper"].isin(diff_df["helper"])].reset_index(drop=True)

        diff_df = diff_df.drop(columns=qdfidx).set_index("helper")
        dfc = dfc.drop(columns=qdfidx).set_index("helper")

        # keep only the users columns (no grading columns)
        diff_df = diff_df.loc[:, dfc.columns]

        self.assertTrue(diff_df.equals(dfc))

    def test_isc_to_qdf(self) -> None:
        df = get_long_format_data(start="2010-01-01", end="2012-01-01")
        ## Test that the grading is not output when not asked for
        tdf = InformationStateChanges.from_qdf(df).to_qdf(
            metrics=["eop"], postfix="$%A"
        )
        self.assertTrue("value" in tdf.columns)
        self.assertTrue("eop" in tdf.columns)
        self.assertTrue("eop_lag" in tdf.columns)
        self.assertTrue("grading" not in tdf.columns)

        self.assertTrue([str(u).endswith("$%A") for u in list(tdf["xcat"])])

    def test_temporal_aggregator_period(self) -> None:
        df = get_long_format_data(start="2010-01-01", end="2012-01-01")
        iscobj = InformationStateChanges.from_qdf(df)
        for k, v in iscobj.items():
            v["zscore_norm_squared"] = np.random.random(len(v))
            iscobj[k] = v

        ref = temporal_aggregator_period(
            isc=iscobj.isc_dict,
            start=iscobj._min_period,
            end=iscobj._max_period,
            winsorise=10,
        )

        test = iscobj.temporal_aggregator_period(winsorise=10)
        self.assertTrue(test.equals(ref))

    def test_calculate_score_on_sparse_indicator(self) -> None:
        qdf = get_long_format_data(end="2012-01-01")
        isc_obj = InformationStateChanges.from_qdf(qdf)

        argsdict = dict(
            std="std",
            halflife=5,
            min_periods=10,
            isc_version=0,
            iis=True,
            custom_method=None,
            custom_method_kwargs={},
        )

        # Attempt with default method adn check _calculate_score_on_sparse_indicator_for_class is called
        with unittest.mock.patch(
            "macrosynergy.management.utils.sparse._calculate_score_on_sparse_indicator_for_class"
        ) as mock:
            isc_obj.calculate_score(**argsdict)
            mock.assert_called_once()

        ## Attempt with custom method
        isc_obj: InformationStateChanges = InformationStateChanges.from_qdf(qdf)
        isc_obj.calculate_score(
            **{**argsdict, "custom_method": VolatilityEstimationMethods.std}
        )

        ## try with bad method args
        with self.assertRaises(TypeError):
            isc_obj.calculate_score(
                **{
                    **argsdict,
                    "custom_method": VolatilityEstimationMethods.std,
                    "custom_method_kwargs": "banana",
                }
            )

        ## type error with custom_method as str
        with self.assertRaises(TypeError):
            isc_obj.calculate_score(**{**argsdict, "custom_method": "banana"})

        ## type error with std as 'banana'
        with self.assertRaises(ValueError):
            isc_obj.calculate_score(**{**argsdict, "std": "banana"})


if __name__ == "__main__":
    unittest.main()
