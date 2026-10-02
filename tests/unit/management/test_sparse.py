import json
import random
import string
import unittest
import unittest.mock
import warnings
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

from macrosynergy.compat import PD_NEW_DATE_FREQ
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import (
    concat_single_metric_qdfs,
    get_cid,
    get_xcat,
    is_valid_iso_date,
    qdf_to_ticker_df,
    ticker_df_to_qdf,
)
from macrosynergy.management.utils.sparse import (
    InformationStateChanges,
    VolatilityEstimationMethods,
    _get_metric_df_from_isc,
    _isc_dict_to_frames,
    _remove_insignificant_values,
    calculate_score_on_sparse_indicator,
    create_delta_data,
    infer_frequency,
    temporal_aggregator_exponential,
    temporal_aggregator_mean,
    temporal_aggregator_period,
    weight_from_frequency,
)

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

    def get_random_freq() -> str:
        return random.choice(list(freq_map.keys()))

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
        # should warn saying 'eop_lag' not found
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            create_delta_data(df=qdf.drop(columns="eop_lag"))
            self.assertTrue(len(w) == 1)
            self.assertTrue(
                "`df` does not contain an `eop_lag` column" in str(w[-1].message)
            )

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

        with self.assertRaises(TypeError):
            infer_frequency(1)

        with self.assertRaises(ValueError):
            infer_frequency(qdf.drop(columns="eop_lag"))

        # make the eop 1000 for all
        qdfc = qdf.copy()
        qdfc["eop_lag"] = 1000
        res = infer_frequency(qdfc)
        self.assertTrue(set(res) == {"D"})  # daily is the fallback

        qdfc = qdf.copy()
        qdfc["eop_lag"] = np.random.randint(1, 1000, len(qdfc))
        res = infer_frequency(qdfc)
        self.assertTrue(set(res) == {"D"})  # daily is the fallback

    def test_weight_from_frequency(self) -> None:
        # {"D": 1, "W": 5, "M": 21, "Q": 93, "A": 252}
        fdict = {"D": 1, "W": 5, "M": 21, "Q": 93, "A": 252}

        for i in range(1, 100):
            base_num = random.randint(1, 255)
            for freq, weight in fdict.items():
                expc_res = weight / base_num
                res = weight_from_frequency(freq, base=base_num)
                self.assertEqual(res, expc_res)

        ltrs = set(string.ascii_uppercase) - set(fdict.keys())
        for ltr in ltrs:
            with self.assertRaises(AssertionError):
                weight_from_frequency(ltr)


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
        isc: InformationStateChanges = InformationStateChanges.from_qdf(qdf)
        self.assertTrue(isinstance(isc, InformationStateChanges))
        self.assertTrue(isinstance(isc.isc_dict, dict))

        self.assertEqual(set(isc.isc_dict.keys()), set(isc.keys()))
        # can't directly check the values and items as they are not hashable

        tks = (qdf["cid"] + "_" + qdf["xcat"]).unique().tolist()
        self.assertTrue(str(tks) in str(isc))
        self.assertEqual(
            repr(isc), f"InformationStateChanges object with {len(tks)} tickers"
        )
        first_dict_key = list(isc.isc_dict.keys())[0]
        self.assertTrue((isc[first_dict_key]).equals(isc.isc_dict[first_dict_key]))

        self.assertEqual(len(list(isc.items())), len(list(isc.isc_dict.items())))
        self.assertEqual(len(list(isc.values())), len(list(isc.isc_dict.values())))

        # test __setitem__
        isc["banana"] = pd.DataFrame()
        self.assertTrue("banana" in isc.isc_dict.keys())
        self.assertTrue(isinstance(isc.isc_dict["banana"], pd.DataFrame))

    def test_to_dict(self) -> None:
        df = get_long_format_data(end="2012-01-01")
        tickers = (df["cid"] + "_" + df["xcat"]).unique().tolist()
        ticker: str = random.choice(tickers)

        isc = InformationStateChanges.from_qdf(df)

        with self.assertRaises(TypeError):
            isc.to_dict()

        res: Dict[str, Any] = isc.to_dict(ticker)
        self.assertTrue(isinstance(res, dict))
        # must have data, columns, ticker, last_real_date as keys
        self.assertEqual(
            set(res.keys()), set(["data", "columns", "ticker", "last_real_date"])
        )
        self.assertTrue(res["columns"] == ("real_date", "value", "eop", "grading"))
        self.assertTrue(is_valid_iso_date(res["last_real_date"]))
        self.assertTrue(isinstance(res["data"], list))
        self.assertTrue(all([isinstance(x, tuple) for x in res["data"]]))

    def test_to_json(self) -> None:
        df = get_long_format_data(end="2012-01-01")
        tickers = (df["cid"] + "_" + df["xcat"]).unique().tolist()
        ticker: str = random.choice(tickers)

        isc = InformationStateChanges.from_qdf(df)

        with self.assertRaises(TypeError):
            isc.to_json()

        res: str = isc.to_json(ticker)
        self.assertEqual(
            json.dumps(isc.to_dict(ticker), sort_keys=True),
            json.dumps(json.loads(res), sort_keys=True),
        )

    def test_isc_object_round_trip(self) -> None:
        qdfidx = QuantamentalDataFrame.IndexCols

        df = get_long_format_data()
        dfc = df.copy()

        tdf = InformationStateChanges.from_qdf(df).to_qdf()

        diff_mask = abs(df["value"].diff())
        diff_mask: pd.Series = diff_mask > 0.0  # type: ignore
        diff_mask.iloc[0] = True
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

        self.assertTrue(diff_df.eq(dfc).all().all())

    def test_isc_object_round_trip_wide(self) -> None:
        df = get_long_format_data()

        tdf = InformationStateChanges.from_qdf(df).to_qdf()
        wdf_orig = qdf_to_ticker_df(df)
        wdf_trip = qdf_to_ticker_df(tdf)

        self.assertTrue(wdf_orig.columns.equals(wdf_trip.columns))

        for col in wdf_orig.columns:
            ots_diff: pd.Series = wdf_orig[col].diff().abs() > 0
            ots_diff.loc[wdf_orig[col].first_valid_index()] = True
            self.assertTrue((wdf_trip[col][ots_diff]).eq(wdf_orig[col][ots_diff]).all())

    def test_isc_to_qdf(self) -> None:
        df = get_long_format_data(end="2012-01-01")
        ## Test that the grading is not output when not asked for
        tdf = InformationStateChanges.from_qdf(df).to_qdf(
            metrics=["eop"], postfix="$%A"
        )
        self.assertTrue("value" in tdf.columns)
        self.assertTrue("eop" in tdf.columns)
        self.assertTrue("eop_lag" in tdf.columns)
        self.assertTrue("grading" not in tdf.columns)

        self.assertTrue([str(u).endswith("$%A") for u in list(tdf["xcat"])])

    def test_isc_to_qdf_metrics(self) -> None:
        df = get_long_format_data(end="2012-01-01")
        # metrics=None emits ALL available (non-value) metrics; metrics=[] emits none.
        tdf_all = InformationStateChanges.from_qdf(df).to_qdf(
            metrics=None, postfix="$%A"
        )
        self.assertTrue("value" in tdf_all.columns)
        self.assertTrue("eop" in tdf_all.columns)
        self.assertTrue("eop_lag" in tdf_all.columns)
        self.assertTrue("grading" in tdf_all.columns)

        tdf_none = InformationStateChanges.from_qdf(df).to_qdf(
            metrics=[], postfix="$%A"
        )
        self.assertEqual(list(tdf_none.columns), ["real_date", "cid", "xcat", "value"])

    def test_isc_to_qdf_winsorise(self) -> None:
        df = get_long_format_data(end="2012-01-01")
        ## Test that the grading is not output when not asked for
        isc: InformationStateChanges = InformationStateChanges.from_qdf(df)
        win_df = isc.to_qdf(metrics=None, thresh=0)
        self.assertTrue(np.allclose(win_df["value"], 0))
        win_df = isc.to_qdf(metrics=None, thresh=(-0.01, 0.1))
        self.assertFalse((win_df["value"] > 0.1).any())
        self.assertFalse((win_df["value"] < -0.01).any())

        # a negative scalar clips by its magnitude: thresh=-0.1 == thresh=0.1.
        neg = isc.to_qdf(metrics=None, thresh=-0.1)
        pos = isc.to_qdf(metrics=None, thresh=0.1)
        self.assertTrue(np.allclose(neg["value"], pos["value"]))

        with self.assertRaises(ValueError):
            isc.to_qdf(metrics=None, thresh="banana")

        with self.assertRaises(ValueError):
            isc.to_qdf(metrics=None, thresh=(-0.01, "banana"))

    def test_temporal_aggregator_period(self) -> None:
        df = get_long_format_data(end="2012-01-01")
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

    def test_get_releases(self):
        qdf = get_long_format_data(end="2012-01-01")
        isc_obj: InformationStateChanges = InformationStateChanges.from_qdf(qdf)
        from_date = "2010-01-01"
        to_date = "2010-10-01"
        res = isc_obj.get_releases(
            from_date=from_date,
            to_date=to_date,
            latest_only=False,
        )

        self.assertTrue(isinstance(res, pd.DataFrame))
        self.assertTrue(res["real_date"].max() <= pd.Timestamp(to_date))
        self.assertTrue(res["real_date"].min() >= pd.Timestamp(from_date))
        expc_cols = ["real_date", "ticker", "eop", "value", "change", "version"]
        missing_cols: Set[str] = set(expc_cols) - set(res.columns)
        self.assertTrue(len(missing_cols) == 0, f"Missing columns: {missing_cols}")

        # test get_releases warning when dates are swapped
        with self.assertWarns(UserWarning):
            isc_obj.get_releases(from_date=to_date, to_date=from_date)

        with self.assertRaises(TypeError):
            isc_obj.get_releases(from_date=1)

        with self.assertRaises(ValueError):
            isc_obj.get_releases(from_date="banana")

        with self.assertRaises(ValueError):
            isc_obj.get_releases(latest_only="banana")

        # test with dates=None
        res = isc_obj.get_releases(from_date=None, to_date=None)

    def test_get_releases_latest(self):
        qdf = get_long_format_data(
            start="2020-01-01",
            end=pd.Timestamp.today().normalize(),
        )
        _today = pd.Timestamp.today().normalize()
        _lbd = _today - pd.offsets.BDay(1)
        _tmin2 = _today - pd.offsets.BDay(2)
        _tmin3 = _today - pd.offsets.BDay(3)
        all_tickers = (qdf["cid"] + "_" + qdf["xcat"]).unique()
        random_tickers = random.choices(all_tickers, k=5)
        cdf = pd.DataFrame(
            [
                {
                    "cid": get_cid(ticker),
                    "xcat": get_xcat(ticker),
                    "real_date": _lbd,
                    "value": np.random.random(),
                    "eop": _lbd,
                    "eop_lag": 0,
                }
                for ticker in random_tickers
            ]
        )
        qdf = qdf[~qdf["real_date"].isin([_lbd, _today, _tmin2, _tmin3])]
        qdf = (
            pd.concat([qdf, cdf], axis=0, ignore_index=True)
            .drop_duplicates(subset=["cid", "xcat"], keep="last")
            .reset_index(drop=True)
        )

        isc_obj: InformationStateChanges = InformationStateChanges.from_qdf(qdf)
        res = isc_obj.get_releases()

        # Ensure compatibility with Pandas 1.3.5
        unique_dates = res["real_date"].unique()
        timestamp_list = [pd.Timestamp(date) for date in unique_dates]

        self.assertTrue(set(res.index) == set(random_tickers))
        self.assertTrue(timestamp_list == [_lbd])

        ## try with release calendar
        res = isc_obj.get_releases(latest_only=False, from_date=_tmin3)
        self.assertTrue(set(res["ticker"]) == set(random_tickers))
        self.assertTrue(all(res["real_date"] > _tmin3))

    def test_get_releases_excl_xcats(self):
        cids = ["USD", "EUR", "JPY", "GBP"]
        xcats = ["GDP", "CPI", "UNEMP", "FX", "BOND", "EQ", "COM"]
        start = "2020-01-01"
        end = pd.Timestamp.today().normalize()
        qdf = get_long_format_data(cids=cids, xcats=xcats, start=start, end=end)
        # update xcats - they have freq. appended as suffix
        xcats = qdf["xcat"].unique().tolist()
        isc_obj = InformationStateChanges.from_qdf(qdf)

        _today = pd.Timestamp.today().normalize()
        _lbd = _today - pd.offsets.BDay(1)
        all_tickers = (qdf["cid"] + "_" + qdf["xcat"]).unique()

        # pick 3 random xcats
        selected_xcats = random.choices(xcats, k=3)
        excl_xcats = list(set(xcats) - set(selected_xcats))
        cdf = pd.DataFrame(
            [
                {
                    "cid": get_cid(ticker),
                    "xcat": get_xcat(ticker),
                    "real_date": _lbd,
                    "value": np.random.random(),
                    "eop": _lbd,
                    "eop_lag": 0,
                }
                for ticker in all_tickers
            ]
        )
        qdf = qdf[~qdf["real_date"].isin([_lbd, _today])]
        qdf = (
            pd.concat([qdf, cdf], axis=0, ignore_index=True)
            .drop_duplicates(subset=["cid", "xcat"], keep="last")
            .reset_index(drop=True)
        )

        isc_obj: InformationStateChanges = InformationStateChanges.from_qdf(qdf)
        res = isc_obj.get_releases(excl_xcats=excl_xcats)

        # Ensure compatibility with Pandas 1.3.5
        unique_dates = res["real_date"].unique()
        timestamp_list = [pd.Timestamp(date) for date in unique_dates]

        self.assertTrue(set(get_xcat(list(set(res.index)))) == set(selected_xcats))
        self.assertTrue(timestamp_list == [_lbd])

        with self.assertRaises(TypeError):
            isc_obj.get_releases(excl_xcats="banana")

        with self.assertRaises(TypeError):
            isc_obj.get_releases(excl_xcats=[1])

    def test_calc_score_vol_forecast(self):
        qdf = get_long_format_data(end="2012-01-01")

        # check that roundtrip remains the same
        qdfa = (
            InformationStateChanges.from_qdf(qdf)
            .calculate_score(volatility_forecast=True)
            .to_qdf()
        ).sort_index()

        qdfb = (
            InformationStateChanges.from_qdf(qdf)
            .calculate_score(volatility_forecast=False)
            .to_qdf()
        ).sort_index()

        self.assertTrue(qdfa.equals(qdfb))

        ## Test the actual std in the information state changes

        isc = InformationStateChanges.from_qdf(qdf).calculate_score(
            volatility_forecast=True, min_periods=1
        )

        isc_test = InformationStateChanges.from_qdf(qdf).calculate_score(
            volatility_forecast=False, min_periods=1
        )

        self.assertTrue(set(isc.keys()) == set(isc_test.keys()))

        for ticker in isc.keys():
            dfa = isc[ticker].sort_index()
            dfb = isc_test[ticker].sort_index()
            self.assertTrue((dfa["std"]).equals(dfb["std"].shift(periods=1)))

    def test_from_isc_df(self):
        qdf = get_long_format_data(
            end="2012-01-01", cids=["USD"], xcats=["GDP"], num_freqs=1
        )
        isc_min: InformationStateChanges = InformationStateChanges.from_qdf(
            qdf, norm=False
        )
        isc_min.calculate_score()
        tickers = list(isc_min.keys())
        assert len(tickers) == 1
        test_ticker = tickers[0]
        new_isc: InformationStateChanges = InformationStateChanges.from_isc_df(
            df=isc_min[test_ticker],
            ticker=test_ticker,
        )

        self.assertTrue(isc_min == new_isc)

    def test_to_qdf_ffill_value(self):
        qdf = get_long_format_data(
            end="2012-01-01", cids=["USD"], xcats=["GDP"], num_freqs=1
        )
        mask = qdf["eop"] != qdf["eop"].shift(1)
        qdf.loc[mask, "grading"] = np.random.rand(mask.sum())
        qdf["grading"] = qdf["grading"].ffill().abs()
        qdf["grading"] = (qdf["grading"] / qdf["grading"].max()) * 3
        qdf["grading"] = qdf["grading"].round(1)

        qdf["grading"] = np.random.randint(11, 31, size=len(qdf))
        qdf["grading"] = (qdf["grading"] / 10).round(1)
        df: pd.DataFrame = InformationStateChanges.from_qdf(qdf, norm=False).to_qdf(
            ffill_value_column="ffill"
        )
        df["eop_lag"] = df["eop_lag"].astype(float)
        self.assertEqual(set(df.columns), set(qdf.columns))
        cols = sorted(df.columns)
        df = QuantamentalDataFrame(df).sort_values(by=cols).reset_index(drop=True)
        qdf = QuantamentalDataFrame(qdf).sort_values(by=cols).reset_index(drop=True)

        for col in cols:
            self.assertTrue(df[col].equals(qdf[col]), f"Column {col} does not match")

    def test_return_dtypes(self):
        # test return type - regular df
        qdf = get_long_format_data(end="2012-01-01")
        isc: InformationStateChanges = InformationStateChanges.from_qdf(qdf)
        outdf = isc.to_qdf()

        self.assertTrue(outdf["cid"].dtype.name == "object")
        self.assertTrue(outdf["xcat"].dtype.name == "object")
        self.assertTrue(isinstance(outdf, QuantamentalDataFrame))

        # test return type - categorical df (not QDF)
        qdf = get_long_format_data(end="2012-01-01")
        qdf = QuantamentalDataFrame(qdf).copy()
        self.assertTrue(
            type(qdf) is pd.DataFrame,
            "Invalid test - This should be a pd.DataFrame, not QuantamentalDataFrame",
        )
        isc: InformationStateChanges = InformationStateChanges.from_qdf(qdf)
        outdf = isc.to_qdf()

        self.assertTrue(outdf["cid"].dtype.name == "category")
        self.assertTrue(outdf["xcat"].dtype.name == "category")
        self.assertTrue(isinstance(outdf, QuantamentalDataFrame))

        # test they are categorical when QDF, also initialized as categorical
        qdf = get_long_format_data(end="2012-01-01")
        qdf = QuantamentalDataFrame(qdf, _initialized_as_categorical=True)
        isc: InformationStateChanges = InformationStateChanges.from_qdf(qdf)
        outdf = isc.to_qdf()

        self.assertTrue(outdf["cid"].dtype.name == "category")
        self.assertTrue(outdf["xcat"].dtype.name == "category")

        # test they are categorical when QDF from regular QDF
        qdf = get_long_format_data(end="2012-01-01")
        qdf = QuantamentalDataFrame(qdf, categorical=True)
        isc: InformationStateChanges = InformationStateChanges.from_qdf(qdf)
        outdf = isc.to_qdf()

        self.assertTrue(outdf["cid"].dtype.name == "object")
        self.assertTrue(outdf["xcat"].dtype.name == "object")

        #  test they are not categorical when QDF
        qdf = get_long_format_data(end="2012-01-01")
        qdf = QuantamentalDataFrame(qdf, categorical=False)
        isc = InformationStateChanges.from_qdf(qdf)
        outdf = isc.to_qdf()

        self.assertTrue(outdf["cid"].dtype.name == "object")
        self.assertTrue(outdf["xcat"].dtype.name == "object")

    def test_to_qdf_error(self):
        isc = InformationStateChanges()
        try:
            isc.to_qdf()
        except ValueError as e:
            expected_msg = "InformationStateChanges object is empty"
            self.assertTrue(expected_msg.lower() in str(e).lower())

    def test_to_qdf_error_missing_eop_lag(self):
        df = make_test_df(start="2012-01-01", end="2012-01-06")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            isc: InformationStateChanges = InformationStateChanges.from_qdf(df)
        # from_qdf may now also warn that some tickers have too few eop to weight, so
        # assert the eop_lag warning is among the warnings rather than the only one.
        expected = "`df` does not contain an `eop_lag` column"
        self.assertTrue(any(expected in str(x.message) for x in w))

        try:
            isc.to_qdf()
        except ValueError as e:
            expected_msg = "Please verify that the input data has the required"
            self.assertTrue(expected_msg.lower() in str(e).lower())


class TestInformationStateChangesScoreBy(unittest.TestCase):
    def test_score_by_diff(self):
        qdf = get_long_format_data(end="2015-01-01")
        isc: InformationStateChanges = InformationStateChanges.from_qdf(qdf)

        isc_score_by_diff = InformationStateChanges.from_qdf(qdf, score_by="diff")

        for ticker in isc.keys():
            self.assertTrue(isc[ticker].equals(isc_score_by_diff[ticker]))

    def test_score_by_level(self):
        qdf = get_long_format_data(end="2015-01-01")

        isc_score_by_level = InformationStateChanges.from_qdf(qdf, score_by="level")
        isc_score_by_diff = InformationStateChanges.from_qdf(qdf, score_by="diff")

        all_isna = []
        for ticker in isc_score_by_diff.keys():
            if isc_score_by_level[ticker]["zscore"].isna().all():
                all_isna.append(ticker)
                continue
            self.assertFalse(
                isc_score_by_diff[ticker].equals(isc_score_by_level[ticker])
            )

        if len(all_isna) == len(isc_score_by_diff.keys()):
            self.fail("All tickers have NaNs in zscore when using score_by='level'")

    def test_score_by_patch(self):
        qdf = get_long_format_data(end="2015-01-01")

        for sc_by in ["level", "diff"]:
            with unittest.mock.patch(
                "macrosynergy.management.utils.sparse._calculate_score_on_sparse_indicator_for_class"
            ) as mock:
                _ = InformationStateChanges.from_qdf(qdf, norm=True, score_by=sc_by)
                mock.assert_called_once()
                self.assertEqual(mock.call_args[1]["score_by"], sc_by)

    def test_score_by_invalid_method(self):
        qdf = get_long_format_data(end="2015-01-01")

        with self.assertRaises(ValueError):
            InformationStateChanges.from_qdf(qdf, score_by="banana")


class TestAnnualizeByReleaseFrequency(unittest.TestCase):
    # pandas >=2.2 renamed the business period-end aliases (BM->BME, BQ->BQE); pick per
    # version so the date ranges are identical on the old pandas pinned for Python 3.7.
    _BME = "BME" if PD_NEW_DATE_FREQ else "BM"
    _BQE = "BQE" if PD_NEW_DATE_FREQ else "BQ"

    @staticmethod
    def _qdf(
        values_by_ticker: Dict[str, np.ndarray], dates: pd.DatetimeIndex
    ) -> QuantamentalDataFrame:
        """Build a QDF (value + eop_lag=0 so eop == real_date) from per-ticker series."""
        val = pd.DataFrame(values_by_ticker, index=dates)
        lag = pd.DataFrame(
            {t: np.zeros(len(dates)) for t in values_by_ticker}, index=dates
        )
        for d in (val, lag):
            d.index.name, d.columns.name = "real_date", "ticker"
        return concat_single_metric_qdfs(
            [
                ticker_df_to_qdf(val, metric="value"),
                ticker_df_to_qdf(lag, metric="eop_lag"),
            ]
        )

    @classmethod
    def _isc(
        cls, values_by_ticker: Dict[str, np.ndarray], dates: pd.DatetimeIndex
    ) -> InformationStateChanges:
        """Build an ISC from per-ticker value series with eop_lag=0 (eop == real_date)."""
        return InformationStateChanges.from_qdf(cls._qdf(values_by_ticker, dates))

    @staticmethod
    def _rw(n: int, seed: int) -> np.ndarray:
        """Deterministic random-walk values. A constant-diff ramp (e.g. arange+1) has a
        zero rolling-std and yields an ``inf`` zscore, which makes a
        ``zscore_fw == zscore * weight`` check pass vacuously (inf == inf); a random walk
        gives finite, non-zero zscores so the weight is actually exercised."""
        return 100.0 + np.random.RandomState(seed).normal(size=n).cumsum()

    def _assert_weighted(self, df: pd.DataFrame, factor: int) -> None:
        z, zfw = df["zscore"], df["zscore_fw"]
        finite = np.isfinite(z) & (z != 0)
        self.assertGreater(int(finite.sum()), 0)  # guard: never a vacuous check
        self.assertTrue(np.allclose(zfw[finite], z[finite] * np.sqrt(1 / factor)))

    def _assert_clipped_weighted(
        self, df: pd.DataFrame, factor: int, lower: float, upper: float
    ) -> None:
        # zscore_fw == winsorised(zscore) * weight; guard that clipping actually bites.
        z, zfw = df["zscore"], df["zscore_fw"]
        finite = np.isfinite(z) & (z != 0)
        self.assertGreater(int(finite.sum()), 0)
        self.assertTrue(((z[finite] < lower) | (z[finite] > upper)).any())
        clipped = z[finite].clip(lower=lower, upper=upper)
        self.assertTrue(np.allclose(zfw[finite], clipped * np.sqrt(1 / factor)))

    def test_monthly_weight(self):
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        isc = self._isc({"AUD_CPIH": self._rw(len(dates), 42)}, dates)

        isc.annualize_by_release_frequency()
        # monthly cadence -> zscore_fw = zscore * sqrt(1/12), written onto the ISC.
        self._assert_weighted(isc.isc_dict["AUD_CPIH"], 12)

    def test_quarterly_weight(self):
        dates = pd.bdate_range("2010-01-01", "2019-12-31", freq=self._BQE)
        isc = self._isc({"AUD_CPIH": self._rw(len(dates), 7)}, dates)

        isc.annualize_by_release_frequency()
        # quarterly cadence -> zscore_fw = zscore * sqrt(1/4).
        self._assert_weighted(isc.isc_dict["AUD_CPIH"], 4)

    def test_multiple_tickers(self):
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        isc = self._isc(
            {"AUD_CPIH": self._rw(len(dates), 1), "AUD_GDP": self._rw(len(dates), 2)},
            dates,
        )
        isc.annualize_by_release_frequency()
        for tkr in ("AUD_CPIH", "AUD_GDP"):
            self._assert_weighted(isc.isc_dict[tkr], 12)

    def test_time_varying_weight_across_cadence_break(self):
        # marquee behaviour: a series whose release cadence changes is weighted by the
        # CONTEMPORANEOUS frequency -- quarterly before the break, monthly after it.
        brk = pd.bdate_range("2008-01-01", "2015-12-31", freq=self._BQE).union(
            pd.bdate_range("2016-01-01", "2019-12-31", freq=self._BME)
        )
        isc = self._isc({"USD_CPIH": self._rw(len(brk), 7)}, brk)
        isc.annualize_by_release_frequency()
        df = isc.isc_dict["USD_CPIH"]

        freq = df["infered_freq"]
        pre = freq[freq.index <= "2015-12-31"].dropna().astype(str).unique()
        post = freq[freq.index >= "2016-06-01"].dropna().astype(str).unique()
        self.assertEqual(set(pre), {"Q"})
        self.assertEqual(set(post), {"M"})

        # each point uses its own frequency's weight; no single global weight matches.
        weights = {"Q": np.sqrt(1 / 4), "M": np.sqrt(1 / 12)}
        w = freq.astype(str).map(weights)
        z, zfw = df["zscore"], df["zscore_fw"]
        finite = np.isfinite(z) & (z != 0)
        self.assertGreater(int(finite.sum()), 0)
        self.assertTrue(np.allclose(zfw[finite], z[finite] * w[finite]))
        self.assertFalse(np.allclose(zfw[finite], z[finite] * np.sqrt(1 / 4)))
        self.assertFalse(np.allclose(zfw[finite], z[finite] * np.sqrt(1 / 12)))

    def test_insufficient_eop_warns_and_skips(self):
        # A ticker whose release frequency cannot be inferred (a constant series collapses
        # to a single eop) must not crash the default norm=True from_qdf path: it is warned
        # about and skipped, while well-formed tickers are still weighted.
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        with self.assertWarnsRegex(UserWarning, "Cannot infer release frequency"):
            isc = self._isc(
                {
                    "AUD_CPIH": self._rw(len(dates), 3),
                    "AUD_GDP": np.full(len(dates), 5.0),
                },
                dates,
            )
        self.assertNotIn("zscore_fw", isc.isc_dict["AUD_GDP"].columns)
        self.assertIn("zscore_fw", isc.isc_dict["AUD_CPIH"].columns)

    def test_annualize_warnings_captured(self):
        # Inspect the two skip-and-warn guards via catch_warnings:
        #   - a ticker with no 'zscore' column, and
        #   - a ticker whose release frequency can't be inferred (<2 distinct eop).
        # Built without the auto-weighting so the explicit call's warnings are captured clean.
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        qdf = self._qdf(
            {
                "AUD_CPIH": self._rw(
                    len(dates), 3
                ),  # well-formed -> weighted, no warning
                "AUD_GDP": np.full(
                    len(dates), 5.0
                ),  # constant -> single eop -> uninferable
                "AUD_RATE": self._rw(
                    len(dates), 4
                ),  # 'zscore' dropped -> no-zscore guard
            },
            dates,
        )
        # norm=False -> minimal ISC; score explicitly so the guards are the only warnings.
        isc = InformationStateChanges.from_qdf(qdf, norm=False)
        isc.calculate_score()
        del isc.isc_dict["AUD_RATE"]["zscore"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            isc.annualize_by_release_frequency()

        msgs = [str(rec.message) for rec in w]
        self.assertTrue(all(issubclass(rec.category, UserWarning) for rec in w))
        self.assertTrue(any("No 'zscore'" in m and "AUD_RATE" in m for m in msgs))
        self.assertTrue(
            any("Cannot infer release frequency" in m and "AUD_GDP" in m for m in msgs)
        )
        # weighted ticker unaffected; both flagged tickers skipped (no zscore_fw).
        self.assertIn("zscore_fw", isc.isc_dict["AUD_CPIH"].columns)
        self.assertNotIn("zscore_fw", isc.isc_dict["AUD_GDP"].columns)
        self.assertNotIn("zscore_fw", isc.isc_dict["AUD_RATE"].columns)

    def test_from_qdf_respects_annualize_flag(self):
        # annualize_by_release_frequency=None (default) follows `norm`; an explicit bool
        # overrides. from_qdf must only invoke the weighting when this resolves to True.
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        qdf = self._qdf({"AUD_CPIH": self._rw(len(dates), 3)}, dates)

        def _called(norm, flag):
            with unittest.mock.patch.object(
                InformationStateChanges, "annualize_by_release_frequency"
            ) as m:
                InformationStateChanges.from_qdf(
                    qdf, norm=norm, annualize_by_release_frequency=flag
                )
            return m.called

        # default (None) follows norm -- norm=False now yields a minimal ISC, no weighting
        self.assertTrue(_called(norm=True, flag=None))
        self.assertFalse(_called(norm=False, flag=None))
        # explicit bool overrides on the norm=True path
        self.assertTrue(_called(norm=True, flag=True))
        self.assertFalse(_called(norm=True, flag=False))
        self.assertFalse(_called(norm=False, flag=False))

        # norm=False + explicit True -> forced to norm=True (with a warning) then called
        with unittest.mock.patch.object(
            InformationStateChanges, "annualize_by_release_frequency"
        ) as m:
            with self.assertWarnsRegex(UserWarning, "can only run when"):
                InformationStateChanges.from_qdf(
                    qdf, norm=False, annualize_by_release_frequency=True
                )
        self.assertTrue(m.called)

    def test_from_qdf_validates_norm_and_flag(self):
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        qdf = self._qdf({"AUD_CPIH": self._rw(len(dates), 3)}, dates)
        with self.assertRaises(ValueError):
            InformationStateChanges.from_qdf(qdf, norm="yes")
        with self.assertRaises(ValueError):
            InformationStateChanges.from_qdf(qdf, annualize_by_release_frequency="yes")

    def test_thresh_scalar_winsorises_symmetrically(self):
        # a scalar clips to (-|thresh|, |thresh|); its sign is irrelevant.
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        for t in (1.0, -1.0):
            isc = self._isc({"AUD_CPIH": self._rw(len(dates), 42)}, dates)
            isc.annualize_by_release_frequency(thresh=t)
            self._assert_clipped_weighted(isc.isc_dict["AUD_CPIH"], 12, -1.0, 1.0)

    def test_thresh_tuple_lopsided(self):
        # asymmetric bounds (one tighter than the other) must both be honoured as given.
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        for lower, upper in ((-0.5, 2.0), (-2.0, 0.5)):
            isc = self._isc({"AUD_CPIH": self._rw(len(dates), 42)}, dates)
            isc.annualize_by_release_frequency(thresh=(lower, upper))
            self._assert_clipped_weighted(isc.isc_dict["AUD_CPIH"], 12, lower, upper)

    def test_thresh_tuple_unordered_is_sorted(self):
        # min/max normalises the tuple: an inverted (high, low) clips as (low, high).
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        isc = self._isc({"AUD_CPIH": self._rw(len(dates), 42)}, dates)
        isc.annualize_by_release_frequency(thresh=(2.0, -0.5))
        self._assert_clipped_weighted(isc.isc_dict["AUD_CPIH"], 12, -0.5, 2.0)

    def test_thresh_leaves_zscore_untouched(self):
        # winsorisation feeds zscore_fw only; the stored zscore column stays intact.
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        isc = self._isc({"AUD_CPIH": self._rw(len(dates), 42)}, dates)
        z_before = isc.isc_dict["AUD_CPIH"]["zscore"].copy()
        isc.annualize_by_release_frequency(thresh=0.5)
        pd.testing.assert_series_equal(isc.isc_dict["AUD_CPIH"]["zscore"], z_before)

    def test_thresh_invalid_raises(self):
        # wrong tuple length/contents, or a non-numeric non-tuple, must raise ValueError.
        dates = pd.bdate_range("2015-01-01", "2018-12-31", freq=self._BME)
        for bad in [(1.0,), (1.0, 2.0, 3.0), ("a", "b"), "banana", [1.0, 2.0]]:
            isc = self._isc({"AUD_CPIH": self._rw(len(dates), 42)}, dates)
            with self.assertRaises(ValueError):
                isc.annualize_by_release_frequency(thresh=bad)


if __name__ == "__main__":
    unittest.main()
