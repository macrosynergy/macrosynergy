import unittest
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any
from numbers import Number

from macrosynergy.management.simulate import make_test_df
from macrosynergy.pnl.contract_signals import (
    contract_signals,
    _gen_contract_signals,
    _add_hedged_signals,
    _apply_hedge_ratios,
    _check_scaling_args,
)
from macrosynergy.management.types import QuantamentalDataFrame
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
        self.cscales: List[Number] = [1.0, 0.5, 0.1]
        self.csigns: List[int] = [1, -1, 1]
        self.hbasket: List[str] = ["USD_EQ", "EUR_EQ"]
        self.hscales: List[Number] = [0.7, 0.3]
        self.sig = "SIG"
        self.hratios = "HR"
        self.sname = "tEsT_sTrAT"

    def _testDF(self) -> QuantamentalDataFrame:
        return make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start=self.start,
            end=self.end,
        )

    def test_gen_contract_signals(self):
        test_df = qdf_to_ticker_df(self._testDF())
        good_args = dict(
            df_wide=test_df,
            sig=self.sig,
            cids=self.cids,
            ctypes=self.ctypes,
            cscales=self.cscales,
            csigns=self.csigns,
        )

        df_wide = _gen_contract_signals(**good_args)
        self.assertIsInstance(df_wide, pd.DataFrame)
        # should all be 0 when cscales are 0
        bad_args = good_args.copy()
        bad_args["cscales"] = [0 for _ in self.cscales]
        df_wide = _gen_contract_signals(**bad_args)
        self.assertTrue(np.allclose(df_wide.values, 0))

        # should all be negative when csigns are -1
        bad_args = good_args.copy()
        bad_args["csigns"] = [-1 for _ in self.csigns]
        df_wide = _gen_contract_signals(**bad_args)
        self.assertTrue(np.all(df_wide.values < 0))

        # should all be exactly 1 when cscales are 1
        bad_args = good_args.copy()
        bad_args["df_wide"].loc[:, :] = 1
        bad_args["cscales"] = [1, 1, 1]
        bad_args["csigns"] = [1, 1, 1]
        df_wide = _gen_contract_signals(**bad_args)
        self.assertTrue(np.all(df_wide.values == 1.0))

        # only FX_CSIG should be non-zero when cscales are 1,0,0
        bad_args = good_args.copy()
        bad_args["df_wide"].loc[:, :] = 1
        bad_args["cscales"] = [1, 0, 0]
        bad_args["csigns"] = [1, 1, 1]
        df_wide = _gen_contract_signals(**bad_args)
        list_of_cols_with_values = list(
            set(
                get_xcat(
                    df_wide.columns[df_wide.apply(lambda x: x != 0).any()].unique()
                )
            )
        )
        self.assertTrue(list_of_cols_with_values == ["FX_CSIG"])

    def test_apply_hedge_ratios(self):
        test_df = qdf_to_ticker_df(self._testDF())
        good_args = dict(
            df_wide=test_df,
            sig=self.sig,
            cids=self.cids,
            hbasket=self.hbasket,
            hscales=self.hscales,
            hratios=self.hratios,
        )

        wide_df = _apply_hedge_ratios(**good_args)

        # should all be 0 when hscales are 0
        bad_args = good_args.copy()
        bad_args["df_wide"].loc[:, :] = 1
        bad_args["hscales"] = [0, 0]
        wide_df = _apply_hedge_ratios(**bad_args)
        self.assertTrue(np.allclose(wide_df.values, 0))

        # should be len(cids) * 1 (5) (adding 1 for each)
        bad_args = good_args.copy()
        bad_args["df_wide"].loc[:, :] = 1
        bad_args["hscales"] = [1, 1]
        wide_df = _apply_hedge_ratios(**bad_args)
        self.assertTrue(np.all(wide_df.values == len(self.cids)))

        bad_args = good_args.copy()
        bad_args["df_wide"].loc[:, :] = 1
        bad_args["hscales"] = [-1, 0]
        # bad_args["hbasket"] = ["USD_EQ", "EUR_EQ"]
        wide_df = _apply_hedge_ratios(**bad_args)
        # nz_tickers -- columns with any non-zero values
        nz_tickers = list(
            (wide_df.columns[wide_df.apply(lambda x: x != 0).any()].unique())
        )
        self.assertTrue(len(nz_tickers) == 1)
        self.assertTrue(nz_tickers[0] == "USD_EQ_CSIG")
        self.assertTrue(np.all(wide_df["USD_EQ_CSIG"].values == -5))

    def test_add_hedged_signals(self):
        dfcs = qdf_to_ticker_df(
            make_test_df(
                cids=self.cids,
                xcats=[f"{ct}_CSIG" for ct in self.ctypes],
                start=self.start,
                end=self.end,
            )
        )
        dfcs.loc[:, :] = 1
        dfhr = qdf_to_ticker_df(
            make_test_df(
                cids=get_cid(self.hbasket),
                xcats=[f"{xc}_CSIG" for xc in list(set(get_xcat(self.hbasket)))],
                start=self.start,
                end=self.end,
            )
        )
        dfhr.loc[:, :] = 1
        df = _add_hedged_signals(dfcs, dfhr)
        # all should be ones
        self.assertTrue(np.all(df.values == 1))
        self.assertTrue(np.all(dfcs.values == _add_hedged_signals(dfcs, None).values))

    def test_check_scaling_args(self):
        good_args = dict(
            ctypes=self.ctypes,
            cscales=self.cscales,
            csigns=self.csigns,
            hbasket=self.hbasket,
            hscales=self.hscales,
            hratios=self.hratios,
        ).copy()
        # full run
        _check_scaling_args(**good_args)

        # should raise error when the following pairs are not the same length
        # - cscales, csigns
        # - cscales, csigns
        # - hscales, hbasket
        for argx in ["cscales", "csigns", "hscales", "hbasket"]:
            bad_args = good_args.copy()
            bad_args[argx] = bad_args[argx][:-1]
            with self.assertRaises(ValueError):
                _check_scaling_args(**bad_args)

            if argx != "hbasket":
                bad_args[argx] = [(None, None) for _ in range(len(good_args[argx]))]
                with self.assertRaises(TypeError):
                    _check_scaling_args(**bad_args)

        bad_args = good_args.copy()
        # set hratios to None
        bad_args["hratios"] = None
        with self.assertRaises(ValueError):
            _check_scaling_args(**bad_args)

        # with cscales/csigns = None, the return cscales/csigns should be all ones
        bad_args = good_args.copy()
        bad_args["cscales"] = None
        bad_args["csigns"] = None
        cscales, csigns, *_ = _check_scaling_args(**bad_args)

        self.assertTrue(np.all(np.array(cscales) == 1))
        self.assertTrue(np.all(np.array(csigns) == 1))

    def test_contract_signal_main(self):
        good_args = dict(
            df=self._testDF(),
            sig=self.sig,
            cids=self.cids,
            ctypes=self.ctypes,
            cscales=self.cscales,
            csigns=self.csigns,
            hbasket=self.hbasket,
            hscales=self.hscales,
            hratios=self.hratios,
            sname=self.sname,
            relative_value=True,
        )
        # full run
        dfc: QuantamentalDataFrame = contract_signals(**good_args)
        self.assertIsInstance(dfc, QuantamentalDataFrame)

        for arg in good_args:
            bad_args = good_args.copy()

            bad_args[arg] = np.zeros(1)
            with self.assertRaises(TypeError):
                contract_signals(**bad_args)
        for arg in good_args:
            if isinstance(good_args[arg], (list, str, dict)):
                if isinstance(good_args[arg], list):
                    bad_args[arg] = []
                elif isinstance(good_args[arg], dict):
                    bad_args[arg] = {}
                elif isinstance(good_args[arg], str):
                    bad_args[arg] = ""
                with self.assertRaises(ValueError):
                    contract_signals(**bad_args)

        for date_var in ["start", "end"]:
            bad_args = good_args.copy()
            bad_args[date_var] = "bad_date"
            with self.assertRaises(ValueError):
                contract_signals(**bad_args)

        # test for improper qdf
        bad_args = good_args.copy()
        bad_args["df"] = bad_args["df"].rename(columns={"cid": "foo"})
        with self.assertRaises(TypeError):
            contract_signals(**bad_args)

        # test for missing tickers
        bad_args = good_args.copy()
        bad_args["df"] = bad_args["df"][
            bad_args["df"]["cid"] != bad_args["cids"][-1]
        ].reset_index(drop=True)
        with self.assertRaises(ValueError):
            contract_signals(**bad_args)

        # test for mismatch

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
