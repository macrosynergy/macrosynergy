import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any

from macrosynergy.management.simulate import make_test_df
from macrosynergy.pnl.contract_signals import (
    contract_signals,
    _check_arg_types,
    _gen_contract_signals,
    _add_hedged_signals,
    _apply_hedge_ratios,
)
from macrosynergy.management.types import Numeric, QuantamentalDataFrame
from macrosynergy.management.utils import (
    is_valid_iso_date,
    standardise_dataframe,
    ticker_df_to_qdf,
    qdf_to_ticker_df,
    reduce_df,
    update_df,
    get_cid,
    get_xcat,
)


class TestContractSignals(unittest.TestCase):
    def setUp(self):
        self.cids: List[str] = ["USD", "EUR", "GBP", "AUD", "CAD"]
        self.xcats: List[str] = ["SIG", "HR"]
        self.start: str = "2000-01-01"
        self.end: str = "2020-12-31"
        self.ctypes: List[str] = ["FX", "IRS", "CDS"]
        self.cscales: List[Numeric] = [1.0, 0.5, 0.1]
        self.csigns: List[int] = [1, -1, 1]
        self.hbasket: List[str] = ["USD_EQ", "EUR_EQ"]
        self.hscales: List[Numeric] = [0.7, 0.3]
        self.sig = "SIG"
        self.hratios = "HR"

    def _testDF(self) -> QuantamentalDataFrame:
        return make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start=self.start,
            end=self.end,
        )

    def test_check_arg_types(self):
        args = [
            "df",
            "sig",
            "cids",
            "ctypes",
            "cscales",
            "csigns",
            "hbasket",
            "hscales",
            "hratios",
            "start",
            "end",
            "blacklist",
            "sname",
        ]

        for arg in args:
            self.assertFalse(_check_arg_types(**{arg: 1}))

    def test_gen_contract_signals(self):
        test_df = self._testDF()
        good_args = dict(
            df=test_df,
            sig=self.sig,
            cids=self.cids,
            ctypes=self.ctypes,
            cscales=self.cscales,
            csigns=self.csigns,
        )

        df = _gen_contract_signals(**good_args)
        self.assertIsInstance(df, QuantamentalDataFrame)
        # self.assertEqual(set(tickers_in_test_df), set(tickers_in_df))
        self.assertAlmostEqual(
            len(set(qdf_to_ticker_df(test_df).columns))
            / len(set(qdf_to_ticker_df(df).columns)),
            len(test_df) / len(df),
        )

        # should all be 0 when cscales are 0
        bad_args = good_args.copy()
        bad_args["cscales"] = [0 for _ in self.cscales]
        df = _gen_contract_signals(**bad_args)
        self.assertTrue(np.allclose(df["value"], 0))

        # should all be negative when csigns are -1
        bad_args = good_args.copy()
        bad_args["csigns"] = [-1 for _ in self.csigns]
        df = _gen_contract_signals(**bad_args)
        self.assertTrue(np.all(df["value"] < 0))

        # should all be exactly 1 when cscales are 1
        bad_args = good_args.copy()
        bad_args["df"]["value"] = 1
        bad_args["cscales"] = [1, 1, 1]
        bad_args["csigns"] = [1, 1, 1]
        df = _gen_contract_signals(**bad_args)
        self.assertTrue((df["value"] == 1.0).all())

        # only FX_CSIG should be non-zero when cscales are 1,0,0
        bad_args = good_args.copy()
        bad_args["df"]["value"] = 1
        bad_args["cscales"] = [1, 0, 0]
        bad_args["csigns"] = [1, 1, 1]
        df = _gen_contract_signals(**bad_args)
        self.assertTrue(df[df["value"].apply(bool)]["xcat"].unique()[0] == "FX_CSIG")

    def test_apply_hedge_ratios(self):
        test_df = self._testDF()
        good_args = dict(
            df=test_df,
            sig=self.sig,
            cids=self.cids,
            hbasket=self.hbasket,
            hscales=self.hscales,
            hratios=self.hratios,
        )

        df = _apply_hedge_ratios(**good_args)
        self.assertIsInstance(df, QuantamentalDataFrame)

        # should all be 0 when hscales are 0
        bad_args = good_args.copy()
        bad_args["df"]["value"] = 1
        bad_args["hscales"] = [0, 0]
        df = _apply_hedge_ratios(**bad_args)
        self.assertTrue((df["value"] == 0).all())

        # should be len(cids) * 1 (5) (adding 1 for each)
        bad_args = good_args.copy()
        bad_args["df"]["value"] = 1
        bad_args["hscales"] = [1, 1]
        df = _apply_hedge_ratios(**bad_args)
        self.assertTrue((df["value"] == len(self.cids)).all())

        bad_args = good_args.copy()
        bad_args["df"]["value"] = 1
        bad_args["hscales"] = [-1, 0]
        # bad_args["hbasket"] = ["USD_EQ", "EUR_EQ"]
        df = _apply_hedge_ratios(**bad_args)
        nz_bool = df["value"].apply(bool)
        nz_tickers = (df["cid"] + "_" + df["xcat"])[nz_bool].unique().tolist()
        self.assertTrue(len(nz_tickers) == 1)
        self.assertTrue(nz_tickers[0] == "USD_EQ_CSIG")
        self.assertTrue((df[nz_bool]["value"] == -5.0).all())

    def test_add_hedged_signals(self):
        dfcs = make_test_df(
            cids=self.cids,
            xcats=[f"{ct}_CSIG" for ct in self.ctypes],
            start=self.start,
            end=self.end,
        )
        dfcs["value"] = 1
        dfhr = make_test_df(
            cids=get_cid(self.hbasket),
            xcats=[f"{xc}_CSIG" for xc in list(set(get_xcat(self.hbasket)))],
            start=self.start,
            end=self.end,
        )
        dfhr["value"] = 1
        df = _add_hedged_signals(dfcs, dfhr)
        # all should be ones
        self.assertTrue((df["value"] == 1).all())

        self.assertTrue(dfcs.eq(_add_hedged_signals(dfcs, None)).all().all())

    def test_contract_signal_no_adjustment(self):
        p: pd.DataFrame = pd.DataFrame(
            1.0,
            columns=[f"{cid:s}_SIGNAL" for cid in ("AUD", "GBP", "EUR")],
            index=pd.date_range("2000-01-01", periods=252, freq="B"),
        )
        p.index.name = "real_date"
        p.columns.name = "ticker"
        dfx = p.stack().to_frame("value").reset_index()
        dfx[["cid", "xcat"]] = dfx.ticker.str.split("_", n=1, expand=True)

        dfc: pd.DataFrame = contract_signals(
            dfx, sig="SIGNAL", cids=["AUD", "GBP", "EUR"], ctypes=["FX"]
        )

        self.assertIsInstance(dfc, pd.DataFrame)
        # TODO check identical dfx and dfc (once adjusted for leverage)
        self.assertEqual(set(dfc.value), set([1]))

    def test_contract_signal_with_volatility_adjustment(self):
        p: pd.DataFrame = pd.DataFrame(
            1.0,
            columns=[f"{cid:s}_SIGNAL" for cid in ("AUD", "GBP", "EUR")],
            index=pd.date_range("2000-01-01", periods=252, freq="B"),
        )
        p.index.name = "real_date"
        p.columns.name = "ticker"
        dfx = p.stack().to_frame("value").reset_index()

        # TODO add monthly volatility changes and compare with contract signals...
        # Leverage: inversion of volatility targets
        p_leverage = 1 / pd.DataFrame(
            [[0.5, 1.0, 2.0]],
            index=p.index,
            columns=p.columns.str.split("_").map(lambda x: x[0] + "_FXLEV"),
        )
        p_leverage.index.name = "real_date"
        p_leverage.columns.name = "ticker"
        df_lev = p_leverage.stack().to_frame("value").reset_index()

        dfx = pd.concat((dfx, df_lev), axis=0, ignore_index=True)
        dfx[["cid", "xcat"]] = dfx.ticker.str.split("_", n=1, expand=True)

        dfc: pd.DataFrame = contract_signals(
            dfx,
            sig="SIGNAL",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            cscales=["FXLEV"],
        )

        self.assertIsInstance(dfc, pd.DataFrame)
        for cid in ["AUD", "GBP", "EUR"]:
            self.assertTrue(
                (
                    dfc.loc[dfc.cid == cid, "value"]
                    == p_leverage[f"{cid:s}_FXLEV"].iloc[0]
                ).all()
            )

    def test_contract_signal_relative_value(self):
        p: pd.DataFrame = pd.DataFrame(
            1.0,
            columns=[f"{cid:s}_SIGNAL" for cid in ("AUD", "GBP", "EUR")],
            index=pd.date_range("2000-01-01", periods=252, freq="B"),
        )
        p.index.name = "real_date"
        p.columns.name = "ticker"
        dfx = p.stack().to_frame("value").reset_index()
        dfx[["cid", "xcat"]] = dfx.ticker.str.split("_", n=1, expand=True)

        # TODO add relative value changes and compare with contract signals...
        dfc: pd.DataFrame = contract_signals(
            dfx,
            sig="SIGNAL",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            relative_value=True,
        )

        self.assertIsInstance(dfc, pd.DataFrame)
        # TODO for unit signals (same signal for all cross sections), and relative value, the contract signal should be zero position.
        self.assertEqual(set(dfc["value"]), set([0.0]))

    def test_contract_signal_relative_value_and_volatility_adjustment(self):
        p: pd.DataFrame = pd.DataFrame(
            1.0,
            columns=[f"{cid:s}_SIGNAL" for cid in ("AUD", "GBP", "EUR")],
            index=pd.date_range("2000-01-01", periods=252, freq="B"),
        )
        p.index.name = "real_date"
        p.columns.name = "ticker"
        dfx = p.stack().to_frame("value").reset_index()

        # Leverage: inversion of volatility targets
        p_leverage = 1 / pd.DataFrame(
            [[0.5, 1.0, 2.0]],
            index=p.index,
            columns=p.columns.str.split("_").map(lambda x: x[0] + "_FXLEV"),
        )
        p_leverage.index.name = "real_date"
        p_leverage.columns.name = "ticker"
        df_lev = p_leverage.stack().to_frame("value").reset_index()

        dfx = pd.concat((dfx, df_lev), axis=0, ignore_index=True)
        dfx[["cid", "xcat"]] = dfx.ticker.str.split("_", n=1, expand=True)
        # TODO add volatility to the above unit signal

        # TODO add relative value changes and compare with contract signals...
        dfc: pd.DataFrame = contract_signals(
            dfx,
            sig="SIGNAL",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            cscales=["FXLEV"],
            relative_value=True,
        )

        self.assertIsInstance(dfc, pd.DataFrame)
        # TODO for unit signals (same signal for all cross sections), and relative value, the contract signal should be zero position.
        # TODO similar: relative value is after volatility adjustment of a signal - so even when adding volatility, a unit signal should be zero position.
        self.assertEqual(set(dfc.value), set([0]))


if __name__ == "__main__":
    unittest.main()
