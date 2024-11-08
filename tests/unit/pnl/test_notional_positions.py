import unittest
from unittest import mock
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Any, Set
from numbers import Number
from macrosynergy.pnl.notional_positions import (
    notional_positions,
    _apply_slip,
    _check_df_for_contract_signals,
    _vol_target_positions,
    _leverage_positions,
    notional_positions,
)
from macrosynergy.pnl.historic_portfolio_volatility import (
    historic_portfolio_vol,
    RETURN_SERIES_XCAT,
)
from macrosynergy.pnl.contract_signals import contract_signals
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.utils import (
    is_valid_iso_date,
    standardise_dataframe,
    ticker_df_to_qdf,
    get_sops,
    qdf_to_ticker_df,
    reduce_df,
    update_df,
    get_cid,
    get_xcat,
)


def mock_historic_portfolio_vol(
    df: pd.DataFrame,
    fids: List[str],
    sname: str,
    rstring: str,
    start: str,
    end: str,
    **kwargs,
) -> pd.DataFrame:
    rebal_dates = get_sops(start_date=start, end_date=end, est_freq="m")
    vol_df = pd.DataFrame(
        {
            "cid": sname,
            "xcat": "a",
            "real_date": rebal_dates,
            "value": 1,
        }
    )

    # create all possible tuples of 2x fids
    fid_pairs = [
        str(x).split("-")
        for x in set(["-".join(sorted([fid1, fid2]) for fid1 in fids for fid2 in fids)])
    ]
    vcv_df = pd.DataFrame(columns=["real_date", "fid1", "fid2", "value"])
    vcv_dict = {}
    for dt in rebal_dates:
        for fid1, fid2 in fid_pairs:
            vcv_dict[(dt, fid1, fid2)] = 1
    vcv_df = pd.DataFrame(vcv_dict).T.reset_index()
    vcv_df.columns = ["real_date", "fid1", "fid2", "value"]
    return vol_df, vcv_df


class TestNotionalPositions(unittest.TestCase):
    def setUp(self) -> None:
        self.cids: List[str] = ["USD", "EUR", "JPY", "GBP"]
        self.fcats: List[str] = ["FX", "CDS", "IRS"]
        self.sname: str = "STRATx"
        self.pname: str = "POSz"
        self.sig_ident: str = f"_CSIG_{self.sname}"
        self.fids: List[str] = [
            f"{cid}_{fcat}" for cid in self.cids for fcat in self.fcats
        ]
        ticker_endings = [f"{fcat}{self.sig_ident}" for fcat in self.fcats]
        self.f_tickers: List[str] = [
            f"{cid}_{te}" for cid in self.cids for te in ticker_endings
        ]

        self.mock_df = make_test_df(
            start="2019-01-01",
            end="2019-02-01",
            cids=self.cids,
            xcats=ticker_endings,
        )
        self.mock_df_wide = qdf_to_ticker_df(self.mock_df)

    def test__apply_slip(self):
        cids = ["USD", "EUR", "JPY", "GBP"]
        fcats = ["FX", "CDS", "IRS"]
        tdf = make_test_df(start="2019-01-01", end="2019-01-10", cids=cids, xcats=fcats)
        fids = [f"{cid}_{fcat}" for cid in cids for fcat in fcats]
        removed_fid = fids.pop(np.random.randint(len(fids)))
        result = _apply_slip(df=tdf, slip=4, fids=fids)
        out_tickers: List[str] = sorted(
            set(result["cid"].astype("object") + "_" + result["xcat"].astype("object"))
        )
        self.assertTrue(removed_fid not in out_tickers)
        self.assertIsInstance(result, QuantamentalDataFrame)

    def test__check_df_for_contract_signals(self):
        # Test vanilla case
        wide_df = self.mock_df_wide.copy()
        _check_df_for_contract_signals(
            df_wide=wide_df, sname=self.sname, fids=self.fids
        )
        # Test ValueError with missing column
        col_names = list(wide_df.columns)
        removed_col = col_names.pop(np.random.randint(len(col_names)))
        wide_df = wide_df.drop(columns=[removed_col])
        with self.assertRaises(ValueError):
            _check_df_for_contract_signals(
                df_wide=wide_df,
                sname=self.sname,
                fids=self.fids,
            )

    def test__leverage_positions(self):

        ## Test 1 - Test with all values as 1
        df_wide = self.mock_df_wide.copy()
        # set all values to 1
        df_wide.loc[:, :] = 1
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        _aum, _leverage = 100, 1
        result = _leverage_positions(
            df_wide=df_wide,
            sname=self.sname,
            pname=self.pname,
            fids=fx_fids,
            leverage=_leverage,
            aum=_aum,
        )
        # col names should be the FID+strat+pos
        expected_cols = [f"{fid}_{self.sname}_{self.pname}" for fid in fx_fids]
        found_cols = list(result.columns)
        self.assertEqual(set(expected_cols), set(found_cols))

        for cola, colb in zip(found_cols[:-1], found_cols[1:]):
            self.assertTrue(result[cola].equals(result[colb]))

        # get all unique values
        unique_values = set(result.values.flatten())
        self.assertEqual(len(unique_values), 1)
        expected_result_value = _aum * _leverage / len(fx_fids)
        self.assertEqual(unique_values, {expected_result_value})

        ## Test 2 - Test with a few nans
        # The tests only need to check the logic of the calculation relative to input
        # so we can set all values to 1
        df_wide = self.mock_df_wide.copy()
        df_wide.loc[:, :] = 1
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        df_wide = df_wide[
            [u for u in df_wide.columns if str(u).endswith(f"_FX_CSIG_{self.sname}")]
        ]

        for _leverage in [1, np.random.randint(1, 10), np.random.rand()]:
            for _aum in [1, np.random.randint(1, int(1e6)), np.random.rand() * 1e6]:

                # create a few nans in the dataframe randomly but record the locations
                shuffled_fx_fids = [f"{t}_CSIG_{self.sname}" for t in fx_fids]
                np.random.shuffle(shuffled_fx_fids)
                random_dates = np.random.choice(df_wide.index, 3, replace=False)
                nan_tuples = [
                    (random_dates[0], f"{shuffled_fx_fids[0]}"),
                    (random_dates[1], f"{shuffled_fx_fids[1]}"),
                    (random_dates[1], f"{shuffled_fx_fids[2]}"),
                    (random_dates[2], None),
                ]

                for date, fid in nan_tuples:
                    if fid is None:
                        df_wide.loc[date, :] = np.nan
                    else:
                        df_wide.loc[date, fid] = np.nan

                result = _leverage_positions(
                    df_wide=df_wide,
                    sname=self.sname,
                    pname=self.pname,
                    fids=fx_fids,
                    leverage=_leverage,
                    aum=_aum,
                )

                # col names should be the FID+strat+pos
                expected_cols = [f"{fid}_{self.sname}_{self.pname}" for fid in fx_fids]
                found_cols = list(result.columns)
                self.assertEqual(set(expected_cols), set(found_cols))

                # check nan-locations - should be in the same place
                for date, fidcsig in nan_tuples:
                    if fidcsig is None:
                        self.assertTrue(result.loc[date, :].isnull().all())
                    else:
                        posname: str = f"{fidcsig}_{self.pname}".replace("_CSIG", "")
                        self.assertTrue(np.isnan(result.loc[date, posname]))

                # iterate through all rows
                for date, row in result.iterrows():
                    # There should be one unique value in each row
                    na_count: int = int(row.isna().sum())
                    if na_count == len(fx_fids):
                        continue  # this is the all nan row
                    unique_values: Set = set(row.dropna().values)
                    self.assertEqual(len(unique_values), 1)
                    # the value should be LEV*AUM/NON-NAN-COUNT
                    expected_result_value = _leverage * _aum / (len(fx_fids) - na_count)
                    self.assertEqual(unique_values, {expected_result_value})

    @mock.patch(
        "macrosynergy.pnl.historic_portfolio_volatility.historic_portfolio_vol",
        side_effect=mock_historic_portfolio_vol,
    )
    def test__vol_target_positions(
        self,
        mock_historic_portfolio_vol: mock.MagicMock,
    ):
        # rename the columns, replace _CSIG_{self.sname} with _XR
        _aum = 1
        _vol_target = 0.1
        dt_range = pd.bdate_range(start="2019-01-01", end="2021-01-01")
        df_wide = pd.DataFrame(
            columns=self.mock_df_wide.columns, index=dt_range, data=1
        )
        df_wide.index.name = "real_date"
        # df_wide.loc[:, :] = 1
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        good_args = dict(
            sname=self.sname,
            pname=self.pname,
            fids=fx_fids,
            vol_target=_vol_target,
            aum=_aum,
            rebal_freq="m",
            est_freqs=["D", "W", "M"],
            est_weights=[1, 2, 3],
            lback_periods=[-1, -1, -1],
            half_life=[11, 5, 6],
            rstring="XR",
            lback_meth="xma",
            nan_tolerance=0.1,
            remove_zeros=True,
        )

        df_wide = df_wide[
            [u for u in df_wide.columns if str(u).endswith(f"_FX_CSIG_{self.sname}")]
        ]
        df_xr = df_wide.copy()
        df_xr.columns = [
            str(col).replace(f"_CSIG_{self.sname}", "XR") for col in df_xr.columns
        ]
        df_wide = pd.concat([df_wide, df_xr], axis=1)

        result: Tuple[pd.DataFrame, ...] = _vol_target_positions(
            df_wide=df_wide, **good_args
        )

        assert isinstance(result, Tuple)

    def test_main(self):
        cids: List[str] = ["USD", "EUR", "GBP", "AUD", "CAD"]
        xcats: List[str] = ["SIG", "HR"]

        start: str = "2000-01-01"
        end: str = "2002-01-01"

        df: pd.DataFrame = make_test_df(
            cids=cids,
            xcats=xcats,
            start=start,
            end=end,
        )

        df.loc[(df["cid"] == "USD") & (df["xcat"] == "SIG"), "value"] = 1.0
        ctypes = ["FX", "IRS", "CDS"]
        cscales = [1.0, 0.5, 0.1]
        csigns = [1, -1, 1]

        hbasket = ["USD_EQ", "EUR_EQ"]
        hscales = [0.7, 0.3]

        df_cs: pd.DataFrame = contract_signals(
            df=df,
            sig="SIG",
            cids=cids,
            ctypes=ctypes,
            cscales=cscales,
            csigns=csigns,
            hbasket=hbasket,
            hscales=hscales,
            hratios="HR",
        )

        fids: List[str] = [f"{cid}_{ctype}" for cid in cids for ctype in ctypes]

        df_notional: pd.DataFrame = notional_positions(
            df=df_cs,
            fids=fids,
            leverage=1.1,
            sname="STRAT",
        )
        all_args = dict(
            df=df_cs,
            fids=fids,
            leverage=1.1,
            sname="STRAT",
        )

        self.assertIsInstance(df_notional, pd.DataFrame)
        df_xr = make_test_df(
            cids=cids,
            xcats=[f"{_}XR" for _ in ctypes],
            start=start,
            end=end,
        )
        hv_args = dict(
            df=pd.concat([df_cs, df_xr], axis=0),
            fids=fids,
            sname="STRAT",
            vol_target=0.1,
            lback_meth="xma",
            lback_periods=-1,
            half_life=20,
            return_pvol=True,
            return_vcv=True,
        )
        dft = notional_positions(**hv_args)
        # this is a tuple of 3 dataframes
        self.assertIsInstance(dft, tuple)
        self.assertEqual(len(dft), 3)
        self.assertIsInstance(dft[0], QuantamentalDataFrame)
        self.assertIsInstance(dft[1], QuantamentalDataFrame)
        self.assertIsInstance(dft[2], pd.DataFrame)
        self.assertEqual(set(dft[2].columns), {"fid1", "fid2", "real_date", "value"})

        # now check with return_pvol=False
        hv_args["return_pvol"] = False
        dft = notional_positions(**hv_args)
        self.assertIsInstance(dft, tuple)
        self.assertEqual(len(dft), 2)
        self.assertIsInstance(dft[0], QuantamentalDataFrame)
        self.assertIsInstance(dft[1], pd.DataFrame)
        self.assertEqual(set(dft[1].columns), {"fid1", "fid2", "real_date", "value"})

        # now check with return_vcv=False
        hv_args["return_pvol"] = True
        hv_args["return_vcv"] = False
        dft = notional_positions(**hv_args)
        self.assertIsInstance(dft, tuple)
        self.assertEqual(len(dft), 2)
        self.assertIsInstance(dft[0], QuantamentalDataFrame)
        self.assertIsInstance(dft[1], QuantamentalDataFrame)

        # for all args pass None and see fail
        bad_args = all_args.copy().copy()
        for key in bad_args:
            bad_args[key] = None
            with self.assertRaises(ValueError):
                notional_positions(**bad_args)

        # vol and lev both should raise ValueError
        bad_args = all_args.copy()
        bad_args["vol_target"] = 1.1
        bad_args["leverage"] = 1.1
        with self.assertRaises(ValueError):
            notional_positions(**bad_args)

        dfb = all_args["df"].copy()
        dfb = dfb[~(dfb["xcat"].str.contains("_CSIG_"))]
        bad_args = all_args.copy()
        bad_args["df"] = dfb
        with self.assertRaises(ValueError):
            notional_positions(**bad_args)

        dfb = all_args["df"].copy()
        dfb = dfb[~(dfb["cid"] == "USD")]
        bad_args = all_args.copy()
        bad_args["df"] = dfb
        with self.assertRaises(ValueError):
            notional_positions(**bad_args)


if __name__ == "__main__":
    unittest.main()
