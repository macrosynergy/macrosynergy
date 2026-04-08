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
    _basket_contract_signals,
    _apply_relative_value,
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
        self.end: str = "2002-12-31"
        self.ctypes: List[str] = ["FX", "IRS", "CDS"]
        self.cscales: List[Number] = [1.0, 0.5, 0.1]
        self.csigns: List[int] = [1, -1, 1]
        self.basket_contracts: List[str] = ["USD_EQ", "EUR_EQ"]
        self.basket_weights: List[Number] = [0.7, 0.3]
        self.sig = "SIG"
        self.hedge_xcat = "HR"
        self.sname = "tEsT_sTrAT"

    def _testDF(self) -> QuantamentalDataFrame:
        return make_test_df(
            cids=self.cids,
            xcats=self.xcats,
            start=self.start,
            end=self.end,
        )

    def test_gen_contract_signals(self):
        test_df = self._testDF()
        test_df["value"] = test_df["value"].abs()
        test_df = qdf_to_ticker_df(test_df)
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
        self.assertTrue(np.all(df_wide.values <= 0))

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

    def test_basket_contract_signals(self):
        test_df = qdf_to_ticker_df(self._testDF())
        good_args = dict(
            df_wide=test_df,
            sig=self.sig,
            cids=self.cids,
            basket_contracts=self.basket_contracts,
            basket_weights=self.basket_weights,
            hedge_xcat=self.hedge_xcat,
        )

        wide_df = _basket_contract_signals(**good_args)

        # should all be 0 when basket_weights are 0
        bad_args = good_args.copy()
        bad_args["df_wide"].loc[:, :] = 1
        bad_args["basket_weights"] = [0, 0]
        wide_df = _basket_contract_signals(**bad_args)
        self.assertTrue(np.allclose(wide_df.values, 0))

        # should be len(cids) * 1 (5) (adding 1 for each)
        bad_args = good_args.copy()
        bad_args["df_wide"].loc[:, :] = 1
        bad_args["basket_weights"] = [1, 1]
        wide_df = _basket_contract_signals(**bad_args)
        self.assertTrue(np.all(wide_df.values == -len(self.cids)))

        bad_args = good_args.copy()
        bad_args["df_wide"].loc[:, :] = 1
        bad_args["basket_weights"] = [-1, 0]
        # bad_args["basket_contracts"] = ["USD_EQ", "EUR_EQ"]
        wide_df = _basket_contract_signals(**bad_args)
        # nz_tickers -- columns with any non-zero values
        nz_tickers = list(
            (wide_df.columns[wide_df.apply(lambda x: x != 0).any()].unique())
        )
        self.assertTrue(len(nz_tickers) == 1)
        self.assertTrue(nz_tickers[0] == "USD_EQ_CSIG")
        self.assertTrue(np.all(wide_df["USD_EQ_CSIG"].values == 5))

    def test_basket_contract_signals_with_timeseries_scales(self):
        test_df = self._testDF()
        test_df = qdf_to_ticker_df(test_df)
        test_df.loc[:, :] = 1.0

        w1 = pd.Series(np.linspace(0.0, 1.0, len(test_df.index)), index=test_df.index)
        w2 = pd.Series(np.linspace(1.0, 0.0, len(test_df.index)), index=test_df.index)

        for ix, cid in enumerate(self.cids):
            test_df[f"{cid}_HW1"] = w1 * (ix + 1)
            test_df[f"{cid}_HW2"] = w2 * (ix + 1)

        test_df.loc[test_df.index[0], "GBP_HW1"] = np.nan
        test_df.loc[test_df.index[1], "AUD_SIG"] = np.nan
        test_df.loc[test_df.index[2], "CAD_HR"] = np.nan

        wide_df = _basket_contract_signals(
            df_wide=test_df,
            sig=self.sig,
            cids=self.cids,
            basket_contracts=self.basket_contracts,
            basket_weights=["HW1", "HW2"],
            hedge_xcat=self.hedge_xcat,
        )

        self.assertTrue(set(wide_df.columns) == {"USD_EQ_CSIG", "EUR_EQ_CSIG"})

        for hb, hscale in zip(self.basket_contracts, ["HW1", "HW2"]):
            basket_pos = f"{hb}_CSIG"
            expected = pd.Series(0.0, index=test_df.index)
            for cid in self.cids:
                contrib = (
                    test_df[f"{cid}_{self.sig}"]
                    * test_df[f"{cid}_{self.hedge_xcat}"]
                    * test_df[f"{cid}_{hscale}"]
                ).fillna(0.0)
                expected += contrib

            self.assertTrue(np.allclose(wide_df[basket_pos].values, -expected.values))

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
                cids=get_cid(self.basket_contracts),
                xcats=[
                    f"{xc}_CSIG" for xc in list(set(get_xcat(self.basket_contracts)))
                ],
                start=self.start,
                end=self.end,
            )
        )
        dfhr.loc[:, :] = 1
        df = _add_hedged_signals(dfcs, dfhr)
        # all should be ones
        self.assertTrue(np.all(df.values == 1))
        self.assertTrue(np.all(dfcs.values == _add_hedged_signals(dfcs, None).values))

    def test_apply_relative_value(self):
        dates = pd.bdate_range("2000-01-03", periods=3)

        # Single ctype, 3 cids, hand-picked positions
        # AUD_FX_CSIG = 8, GBP_FX_CSIG = 2, EUR_FX_CSIG = 5
        # mean = (8+2+5)/3 = 5
        # RV: AUD = 8-5 = 3, GBP = 2-5 = -3, EUR = 5-5 = 0
        df = pd.DataFrame(
            {"AUD_FX_CSIG": 8.0, "GBP_FX_CSIG": 2.0, "EUR_FX_CSIG": 5.0},
            index=dates,
        )
        df.index.name = "real_date"

        result = _apply_relative_value(
            df.copy(), cids=["AUD", "GBP", "EUR"], ctypes=["FX"]
        )
        np.testing.assert_allclose(result["AUD_FX_CSIG"].values, 3.0, atol=1e-10)
        np.testing.assert_allclose(result["GBP_FX_CSIG"].values, -3.0, atol=1e-10)
        np.testing.assert_allclose(result["EUR_FX_CSIG"].values, 0.0, atol=1e-10)

        # Two ctypes — each computed independently
        # FX: AUD=8, GBP=2 → mean=5 → RV: 3, -3
        # IRS: AUD=-1, GBP=-3 → mean=-2 → RV: 1, -1
        df2 = pd.DataFrame(
            {
                "AUD_FX_CSIG": 8.0,
                "GBP_FX_CSIG": 2.0,
                "AUD_IRS_CSIG": -1.0,
                "GBP_IRS_CSIG": -3.0,
            },
            index=dates,
        )
        df2.index.name = "real_date"

        result2 = _apply_relative_value(
            df2.copy(), cids=["AUD", "GBP"], ctypes=["FX", "IRS"]
        )
        np.testing.assert_allclose(result2["AUD_FX_CSIG"].values, 3.0, atol=1e-10)
        np.testing.assert_allclose(result2["GBP_FX_CSIG"].values, -3.0, atol=1e-10)
        np.testing.assert_allclose(result2["AUD_IRS_CSIG"].values, 1.0, atol=1e-10)
        np.testing.assert_allclose(result2["GBP_IRS_CSIG"].values, -1.0, atol=1e-10)

        # NaN excludes cid from the mean for that date
        # date 0: AUD=8, GBP=NaN, EUR=4 → mean=(8+4)/2=6 → AUD=2, EUR=-2
        # date 1: AUD=8, GBP=2, EUR=4   → mean=(8+2+4)/3≈4.667
        df3 = pd.DataFrame(
            {
                "AUD_FX_CSIG": [8.0, 8.0],
                "GBP_FX_CSIG": [float("nan"), 2.0],
                "EUR_FX_CSIG": [4.0, 4.0],
            },
            index=dates[:2],
        )
        df3.index.name = "real_date"

        result3 = _apply_relative_value(
            df3.copy(), cids=["AUD", "GBP", "EUR"], ctypes=["FX"]
        )
        # date 0
        self.assertAlmostEqual(result3["AUD_FX_CSIG"].iloc[0], 2.0, places=10)
        self.assertTrue(np.isnan(result3["GBP_FX_CSIG"].iloc[0]))
        self.assertAlmostEqual(result3["EUR_FX_CSIG"].iloc[0], -2.0, places=10)
        # date 1: mean = 14/3
        mean_d1 = (8.0 + 2.0 + 4.0) / 3
        self.assertAlmostEqual(result3["AUD_FX_CSIG"].iloc[1], 8.0 - mean_d1, places=10)
        self.assertAlmostEqual(result3["GBP_FX_CSIG"].iloc[1], 2.0 - mean_d1, places=10)
        self.assertAlmostEqual(result3["EUR_FX_CSIG"].iloc[1], 4.0 - mean_d1, places=10)

        # Single tradable cid → position = 0
        df4 = pd.DataFrame(
            {
                "AUD_FX_CSIG": [7.0],
                "GBP_FX_CSIG": [float("nan")],
            },
            index=dates[:1],
        )
        df4.index.name = "real_date"

        result4 = _apply_relative_value(df4.copy(), cids=["AUD", "GBP"], ctypes=["FX"])
        self.assertAlmostEqual(result4["AUD_FX_CSIG"].iloc[0], 0.0, places=10)
        self.assertTrue(np.isnan(result4["GBP_FX_CSIG"].iloc[0]))

    def test_check_scaling_args(self):
        good_args = dict(
            ctypes=self.ctypes,
            cscales=self.cscales,
            csigns=self.csigns,
            basket_contracts=self.basket_contracts,
            basket_weights=self.basket_weights,
            hedge_xcat=self.hedge_xcat,
        ).copy()
        # full run
        _check_scaling_args(**good_args)

        # should raise error when the following pairs are not the same length
        # - cscales, csigns
        # - cscales, csigns
        # - basket_weights, basket_contracts
        for argx in ["cscales", "csigns", "basket_weights", "basket_contracts"]:
            bad_args = good_args.copy()
            bad_args[argx] = bad_args[argx][:-1]
            with self.assertRaises(ValueError):
                _check_scaling_args(**bad_args)

            if argx != "basket_contracts":
                bad_args[argx] = [(None, None) for _ in range(len(good_args[argx]))]
                with self.assertRaises(TypeError):
                    _check_scaling_args(**bad_args)

        bad_args = good_args.copy()
        # set hedge_xcat to None
        bad_args["hedge_xcat"] = None
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
            basket_contracts=self.basket_contracts,
            basket_weights=self.basket_weights,
            hedge_xcat=self.hedge_xcat,
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
        # With position-level RV: unit signals but different leverages produce
        # different-sized positions. RV subtracts the position mean, so positions
        # are non-zero but sum to zero across cids at each date.
        dfc_wide = dfc.pivot_table(index="real_date", columns="cid", values="value")
        row_sums = dfc_wide.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 0.0, atol=1e-10)


class TestRelativeValueBasket(unittest.TestCase):
    """
    Tests for relative_value=True in contract_signals.

    When relative_value is True, each cross-section's signal calls for a main
    position (sig * csign * cscale) plus an equal-weighted opposite-sign basket
    of those positions across all concurrently tradable cross-sections. The
    basket inherits csigns (negated) and cscales from the positions it offsets.

    Net position per contract type:
        pos(c, i, t) = sig_c * csign_i * cscale_{c,i}
        rv_pos(c, i, t) = pos(c, i, t) - mean_t(pos(·, i, t))
    """

    @staticmethod
    def _make_qdf(ticker_data: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Build a QDF from a dict of {ticker: value_or_array}.
        ticker format: "CID_XCAT", value is a scalar (constant) or array per date.
        """
        rows = []
        for ticker, values in ticker_data.items():
            cid, xcat = ticker.split("_", 1)
            if isinstance(values, (int, float)):
                values = [values] * len(dates)
            for date, val in zip(dates, values):
                rows.append({"cid": cid, "xcat": xcat, "real_date": date, "value": val})
        return pd.DataFrame(rows)

    def test_rv_basic_basket_decomposition(self):
        """
        With fixed cscale=1 and csign=1, pos = sig, so
        rv_pos = sig - mean(sig). Same as signal-level RV.

        Setup: sig = [AUD:3, GBP:1, EUR:2], ctype=FX, csign=1, cscale=1
        Positions: [3, 1, 2], mean = 2
        RV: AUD=1, GBP=-1, EUR=0
        """
        dates = pd.bdate_range("2000-01-03", periods=5)
        dfx = self._make_qdf({"AUD_SIG": 3.0, "GBP_SIG": 1.0, "EUR_SIG": 2.0}, dates)

        dfc = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            relative_value=True,
        )

        expected = {"AUD": 1.0, "GBP": -1.0, "EUR": 0.0}
        for cid, exp_val in expected.items():
            actual = dfc.loc[dfc["cid"] == cid, "value"].values
            np.testing.assert_allclose(
                actual,
                exp_val,
                atol=1e-10,
                err_msg=f"{cid}: expected {exp_val}, got {actual[0]}",
            )

    def test_rv_positions_sum_to_zero_fixed_cscale(self):
        """
        With fixed cscales, positions sum to zero across cids at every date.
        """
        dates = pd.bdate_range("2000-01-03", periods=20)
        np.random.seed(42)
        cids = ["AUD", "GBP", "EUR", "CAD"]
        sig_data = {f"{cid}_SIG": np.random.randn(len(dates)) for cid in cids}
        dfx = self._make_qdf(sig_data, dates)

        dfc = contract_signals(
            dfx,
            sig="SIG",
            cids=cids,
            ctypes=["FX"],
            relative_value=True,
        )

        dfc_wide = dfc.pivot_table(index="real_date", columns="cid", values="value")
        row_sums = dfc_wide.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values,
            0.0,
            atol=1e-10,
            err_msg="Positions must sum to zero (market-neutral) at each date",
        )

    def test_rv_positions_sum_to_zero_variable_cscale(self):
        """
        With variable cscales (category ticker), positions STILL sum to zero.
        This is the key difference from the old implementation which operated
        on raw signals - that approach broke sum-to-zero with variable cscales.

        Setup: sig=[4,2], cscale="LEV", AUD_LEV=2.0, GBP_LEV=0.5
        Positions: AUD=4*1*2=8, GBP=2*1*0.5=1
        Mean position = (8+1)/2 = 4.5
        RV: AUD=8-4.5=3.5, GBP=1-4.5=-3.5
        Sum = 0 (3.5 + -3.5)
        """
        dates = pd.bdate_range("2000-01-03", periods=5)
        dfx = self._make_qdf(
            {"AUD_SIG": 4.0, "GBP_SIG": 2.0, "AUD_LEV": 2.0, "GBP_LEV": 0.5},
            dates,
        )

        dfc = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP"],
            ctypes=["FX"],
            cscales=["LEV"],
            relative_value=True,
        )

        dfc_wide = dfc.pivot_table(index="real_date", columns="cid", values="value")
        row_sums = dfc_wide.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values,
            0.0,
            atol=1e-10,
            err_msg="Positions must sum to zero even with variable cscales",
        )

    def test_rv_with_variable_cscales_values(self):
        """
        Verify exact position values with variable cscales.

        Setup: sig=[4,2], cscale="LEV", AUD_LEV=2.0, GBP_LEV=0.5, csign=1
        Positions before RV: AUD_FX=4*1*2=8, GBP_FX=2*1*0.5=1
        Position mean = (8+1)/2 = 4.5
        RV: AUD_FX=8-4.5=3.5, GBP_FX=1-4.5=-3.5
        """
        dates = pd.bdate_range("2000-01-03", periods=5)
        dfx = self._make_qdf(
            {"AUD_SIG": 4.0, "GBP_SIG": 2.0, "AUD_LEV": 2.0, "GBP_LEV": 0.5},
            dates,
        )

        dfc = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP"],
            ctypes=["FX"],
            cscales=["LEV"],
            relative_value=True,
        )

        expected = {"AUD": 3.5, "GBP": -3.5}
        for cid, exp_val in expected.items():
            actual = dfc.loc[dfc["cid"] == cid, "value"].values
            np.testing.assert_allclose(
                actual,
                exp_val,
                atol=1e-10,
                err_msg=f"{cid}: expected {exp_val}",
            )

    def test_rv_blacklist_excludes_from_basket(self):
        """
        Blacklisted cids are excluded from the tradable basket.

        Setup: sig=[6,2,4], ctype=FX, csign=1, cscale=1
        GBP blacklisted for dates[0:3].

        Blacklisted dates (tradable=[AUD,EUR]):
          Positions: AUD=6, EUR=4, mean=5
          RV: AUD=1, EUR=-1

        Non-blacklisted dates (tradable=[AUD,GBP,EUR]):
          Positions: AUD=6, GBP=2, EUR=4, mean=4
          RV: AUD=2, GBP=-2, EUR=0
        """
        dates = pd.bdate_range("2000-01-03", periods=6)
        dfx = self._make_qdf({"AUD_SIG": 6.0, "GBP_SIG": 2.0, "EUR_SIG": 4.0}, dates)
        blacklist = {
            "GBP": (dates[0].strftime("%Y-%m-%d"), dates[2].strftime("%Y-%m-%d"))
        }

        dfc = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            relative_value=True,
            blacklist=blacklist,
        )

        for date in dates[:3]:
            self.assertAlmostEqual(
                dfc.loc[
                    (dfc["cid"] == "AUD") & (dfc["real_date"] == date), "value"
                ].iloc[0],
                1.0,
                places=10,
            )
            self.assertAlmostEqual(
                dfc.loc[
                    (dfc["cid"] == "EUR") & (dfc["real_date"] == date), "value"
                ].iloc[0],
                -1.0,
                places=10,
            )
            gbp = dfc.loc[(dfc["cid"] == "GBP") & (dfc["real_date"] == date), "value"]
            self.assertTrue(gbp.empty or gbp.isna().all())

        for date in dates[3:]:
            self.assertAlmostEqual(
                dfc.loc[
                    (dfc["cid"] == "AUD") & (dfc["real_date"] == date), "value"
                ].iloc[0],
                2.0,
                places=10,
            )
            self.assertAlmostEqual(
                dfc.loc[
                    (dfc["cid"] == "GBP") & (dfc["real_date"] == date), "value"
                ].iloc[0],
                -2.0,
                places=10,
            )
            self.assertAlmostEqual(
                dfc.loc[
                    (dfc["cid"] == "EUR") & (dfc["real_date"] == date), "value"
                ].iloc[0],
                0.0,
                places=10,
            )

    def test_rv_blacklisted_dates_still_market_neutral(self):
        """
        Tradable positions sum to zero at every date, even with blacklist.
        """
        dates = pd.bdate_range("2000-01-03", periods=10)
        dfx = self._make_qdf({"AUD_SIG": 6.0, "GBP_SIG": 2.0, "EUR_SIG": 4.0}, dates)
        blacklist = {
            "GBP": (dates[0].strftime("%Y-%m-%d"), dates[4].strftime("%Y-%m-%d"))
        }

        dfc = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            relative_value=True,
            blacklist=blacklist,
        )

        dfc_wide = dfc.pivot_table(index="real_date", columns="cid", values="value")
        row_sums = dfc_wide.sum(axis=1)
        np.testing.assert_allclose(
            row_sums.values,
            0.0,
            atol=1e-10,
            err_msg="Tradable positions must sum to zero even with blacklist",
        )

    def test_rv_single_tradable_cid_gives_zero(self):
        """
        When only 1 cid is tradable, the basket fully offsets: pos - pos = 0.
        """
        dates = pd.bdate_range("2000-01-03", periods=5)
        dfx = self._make_qdf(
            {
                "AUD_SIG": [5.0, 5.0, 5.0, 5.0, 5.0],
                "GBP_SIG": [float("nan"), float("nan"), 3.0, 3.0, 3.0],
                "EUR_SIG": [float("nan"), float("nan"), 1.0, 1.0, 1.0],
            },
            dates,
        )

        dfc = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            relative_value=True,
        )

        # Dates 0-1: only AUD tradable → position = 0
        for date in dates[:2]:
            aud_val = dfc.loc[
                (dfc["cid"] == "AUD") & (dfc["real_date"] == date), "value"
            ]
            self.assertAlmostEqual(aud_val.iloc[0], 0.0, places=10)

        # Dates 2-4: all tradable, positions=[5,3,1], mean=3
        for date in dates[2:]:
            self.assertAlmostEqual(
                dfc.loc[
                    (dfc["cid"] == "AUD") & (dfc["real_date"] == date), "value"
                ].iloc[0],
                2.0,
                places=10,
            )
            self.assertAlmostEqual(
                dfc.loc[
                    (dfc["cid"] == "GBP") & (dfc["real_date"] == date), "value"
                ].iloc[0],
                0.0,
                places=10,
            )
            self.assertAlmostEqual(
                dfc.loc[
                    (dfc["cid"] == "EUR") & (dfc["real_date"] == date), "value"
                ].iloc[0],
                -2.0,
                places=10,
            )

    def test_rv_multiple_ctypes_with_csigns(self):
        """
        Each ctype gets its own independent RV. With fixed cscales per ctype,
        the basket operates independently per ctype.

        sig=[AUD:4, GBP:2], ctypes=[FX,IRS], csigns=[1,-1], cscales=[1.0,0.5]

        FX positions: AUD=4*1*1=4, GBP=2*1*1=2, mean=3
          RV: AUD=1, GBP=-1
        IRS positions: AUD=4*(-1)*0.5=-2, GBP=2*(-1)*0.5=-1, mean=-1.5
          RV: AUD=-2-(-1.5)=-0.5, GBP=-1-(-1.5)=0.5
        """
        dates = pd.bdate_range("2000-01-03", periods=5)
        dfx = self._make_qdf({"AUD_SIG": 4.0, "GBP_SIG": 2.0}, dates)

        dfc = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP"],
            ctypes=["FX", "IRS"],
            csigns=[1, -1],
            cscales=[1.0, 0.5],
            relative_value=True,
        )

        expected = {
            ("AUD", "FX_STRAT_CSIG"): 1.0,
            ("AUD", "IRS_STRAT_CSIG"): -0.5,
            ("GBP", "FX_STRAT_CSIG"): -1.0,
            ("GBP", "IRS_STRAT_CSIG"): 0.5,
        }
        for (cid, xcat), exp_val in expected.items():
            actual = dfc.loc[
                (dfc["cid"] == cid) & (dfc["xcat"] == xcat), "value"
            ].values
            np.testing.assert_allclose(
                actual,
                exp_val,
                atol=1e-10,
                err_msg=f"{cid}_{xcat}: expected {exp_val}",
            )

    def test_rv_nan_signals_excluded_from_basket(self):
        """
        NaN signals produce NaN positions, excluded from the position mean.

        sig: AUD=[6,6,6], GBP=[NaN,2,2], EUR=[4,4,4], cscale=1, csign=1

        d0: tradable=[AUD,EUR], positions=[6,4], mean=5
          RV: AUD=1, EUR=-1
        d1-d2: tradable=all, positions=[6,2,4], mean=4
          RV: AUD=2, GBP=-2, EUR=0
        """
        dates = pd.bdate_range("2000-01-03", periods=3)
        dfx = self._make_qdf(
            {
                "AUD_SIG": [6.0, 6.0, 6.0],
                "GBP_SIG": [float("nan"), 2.0, 2.0],
                "EUR_SIG": [4.0, 4.0, 4.0],
            },
            dates,
        )

        dfc = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            relative_value=True,
        )

        d0 = dates[0]
        self.assertAlmostEqual(
            dfc.loc[(dfc["cid"] == "AUD") & (dfc["real_date"] == d0), "value"].iloc[0],
            1.0,
            places=10,
        )
        self.assertAlmostEqual(
            dfc.loc[(dfc["cid"] == "EUR") & (dfc["real_date"] == d0), "value"].iloc[0],
            -1.0,
            places=10,
        )
        gbp_d0 = dfc.loc[(dfc["cid"] == "GBP") & (dfc["real_date"] == d0), "value"]
        self.assertTrue(gbp_d0.empty or gbp_d0.isna().all())

        d1 = dates[1]
        self.assertAlmostEqual(
            dfc.loc[(dfc["cid"] == "AUD") & (dfc["real_date"] == d1), "value"].iloc[0],
            2.0,
            places=10,
        )
        self.assertAlmostEqual(
            dfc.loc[(dfc["cid"] == "GBP") & (dfc["real_date"] == d1), "value"].iloc[0],
            -2.0,
            places=10,
        )
        self.assertAlmostEqual(
            dfc.loc[(dfc["cid"] == "EUR") & (dfc["real_date"] == d1), "value"].iloc[0],
            0.0,
            places=10,
        )

    def test_rv_without_relative_value_gives_raw_positions(self):
        """
        Without relative_value=True, positions are just sig * csign * cscale.
        """
        dates = pd.bdate_range("2000-01-03", periods=5)
        dfx = self._make_qdf({"AUD_SIG": 3.0, "GBP_SIG": 1.0, "EUR_SIG": 2.0}, dates)

        dfc_raw = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            relative_value=False,
        )
        for cid, sig_val in [("AUD", 3.0), ("GBP", 1.0), ("EUR", 2.0)]:
            actual = dfc_raw.loc[dfc_raw["cid"] == cid, "value"].values
            np.testing.assert_allclose(actual, sig_val, atol=1e-10)

        dfc_rv = contract_signals(
            dfx,
            sig="SIG",
            cids=["AUD", "GBP", "EUR"],
            ctypes=["FX"],
            relative_value=True,
        )
        for cid, exp_val in [("AUD", 1.0), ("GBP", -1.0), ("EUR", 0.0)]:
            actual = dfc_rv.loc[dfc_rv["cid"] == cid, "value"].values
            np.testing.assert_allclose(actual, exp_val, atol=1e-10)

    def test_rv_fixed_vs_variable_cscale_diverge(self):
        """
        Prove that position-level RV differs from signal-level RV when
        cscales vary across cids.

        sig=[4,2], fixed cscale=1: positions=[4,2], mean=3
          RV: AUD=1, GBP=-1
        sig=[4,2], variable cscale AUD=2,GBP=0.5: positions=[8,1], mean=4.5
          RV: AUD=3.5, GBP=-3.5

        A signal-level implementation would give:
          (4-3)*2=2 and (2-3)*0.5=-0.5 - DIFFERENT and doesn't sum to zero.
        """
        dates = pd.bdate_range("2000-01-03", periods=5)

        # Fixed cscale
        dfx_fixed = self._make_qdf({"AUD_SIG": 4.0, "GBP_SIG": 2.0}, dates)
        dfc_fixed = contract_signals(
            dfx_fixed,
            sig="SIG",
            cids=["AUD", "GBP"],
            ctypes=["FX"],
            cscales=[1.0],
            relative_value=True,
        )

        # Variable cscale
        dfx_var = self._make_qdf(
            {"AUD_SIG": 4.0, "GBP_SIG": 2.0, "AUD_LEV": 2.0, "GBP_LEV": 0.5},
            dates,
        )
        dfc_var = contract_signals(
            dfx_var,
            sig="SIG",
            cids=["AUD", "GBP"],
            ctypes=["FX"],
            cscales=["LEV"],
            relative_value=True,
        )

        # Fixed: sum to zero
        wide_fixed = dfc_fixed.pivot_table(
            index="real_date", columns="cid", values="value"
        )
        np.testing.assert_allclose(wide_fixed.sum(axis=1).values, 0.0, atol=1e-10)

        # Variable: ALSO sum to zero (position-level RV guarantees this)
        wide_var = dfc_var.pivot_table(index="real_date", columns="cid", values="value")
        np.testing.assert_allclose(wide_var.sum(axis=1).values, 0.0, atol=1e-10)

        # Variable values are NOT what signal-level RV would give
        # Signal-level would give AUD=2.0, GBP=-0.5 (doesn't sum to zero)
        # Position-level gives AUD=3.5, GBP=-3.5
        for cid, wrong_val, right_val in [("AUD", 2.0, 3.5), ("GBP", -0.5, -3.5)]:
            actual = dfc_var.loc[dfc_var["cid"] == cid, "value"].values[0]
            self.assertNotAlmostEqual(
                actual,
                wrong_val,
                places=5,
                msg=f"{cid} should NOT equal signal-level RV {wrong_val}",
            )
            self.assertAlmostEqual(
                actual,
                right_val,
                places=10,
                msg=f"{cid} should equal position-level RV {right_val}",
            )


if __name__ == "__main__":
    unittest.main()
