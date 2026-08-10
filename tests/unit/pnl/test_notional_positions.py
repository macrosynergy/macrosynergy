import unittest
import warnings
from typing import List, Tuple
from unittest import mock

import numpy as np
import pandas as pd

from macrosynergy.management.simulate import make_test_df
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.management.utils import get_sops, qdf_to_ticker_df
from macrosynergy.pnl.contract_signals import contract_signals
from macrosynergy.pnl.notional_positions import (
    _apply_slip,
    _check_df_for_contract_signals,
    _dollar_per_signal_positions,
    _leverage_positions,
    _mask_unavailable_positions,
    _vol_target_positions,
    notional_positions,
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


class TestMaskUnavailablePositions(unittest.TestCase):
    def setUp(self) -> None:
        self.sig_ident: str = "_CSIG_STRAT"
        self.dates = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
        self.fids: List[str] = ["AUD_FX", "EUR_FX"]
        self.signal_columns: List[str] = [f"{fid}{self.sig_ident}" for fid in self.fids]
        self.df_signals = pd.DataFrame(
            2.0, index=self.dates, columns=self.signal_columns
        )

    def tearDown(self) -> None: ...

    def _vcv(self, rows: List[list]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["real_date", "fid1", "fid2", "value"])

    def test_masks_fids_absent_from_vcv(self):
        # AUD_FX is in the estimate on every date; EUR_FX only from the 2nd date.
        rows = [[dt, "AUD_FX", "AUD_FX", 1.0] for dt in self.dates]
        for dt in self.dates[1:]:
            rows += [
                [dt, "EUR_FX", "EUR_FX", 1.0],
                [dt, "AUD_FX", "EUR_FX", 0.5],
            ]
        masked = _mask_unavailable_positions(
            self.df_signals, self._vcv(rows), self.sig_ident
        )

        # AUD_FX positioned throughout; EUR_FX blanked on the first date only.
        self.assertFalse(masked["AUD_FX_CSIG_STRAT"].isna().any())
        self.assertTrue(np.isnan(masked.loc[self.dates[0], "EUR_FX_CSIG_STRAT"]))
        self.assertFalse(masked.loc[self.dates[1:], "EUR_FX_CSIG_STRAT"].isna().any())

    def test_fid_never_in_vcv_is_fully_masked(self):
        # only AUD_FX ever appears - EUR_FX must be blanked on every date.
        rows = [[dt, "AUD_FX", "AUD_FX", 1.0] for dt in self.dates]
        masked = _mask_unavailable_positions(
            self.df_signals, self._vcv(rows), self.sig_ident
        )

        self.assertFalse(masked["AUD_FX_CSIG_STRAT"].isna().any())
        self.assertTrue(masked["EUR_FX_CSIG_STRAT"].isna().all())

    def test_full_universe_is_unchanged(self):
        # every contract present on every date - masking must be a no-op.
        rows = []
        for dt in self.dates:
            rows += [
                [dt, "AUD_FX", "AUD_FX", 1.0],
                [dt, "EUR_FX", "EUR_FX", 1.0],
                [dt, "AUD_FX", "EUR_FX", 0.5],
            ]
        masked = _mask_unavailable_positions(
            self.df_signals, self._vcv(rows), self.sig_ident
        )

        pd.testing.assert_frame_equal(masked, self.df_signals)


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
        self.rebal_freq: str = "w"

    def _rebalance_fixture(self):
        """
        An FX signal panel whose level is distinct in every
        rebalance period, so that a value carried over from a previous period is
        distinguishable from one freshly read on a rebalance date. The signal for the
        contract at index `offset` on the k-th rebalance date is `k + 1 + offset`.

        Blanks one contract's signal on the 3rd rebalance date and two on the 4th, plus
        a whole non-rebalance row - the latter must be a no-op, since values off the
        rebalance schedule are never read.
        """
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        sig_cols = [f"{fid}_CSIG_{self.sname}" for fid in fx_fids]
        df_wide = self.mock_df_wide.copy()[sig_cols]

        rebal_dates = list(get_sops(dates=df_wide.index, freq=self.rebal_freq))
        period = pd.Series(
            df_wide.index.isin(rebal_dates), index=df_wide.index
        ).cumsum()
        for offset, col in enumerate(sig_cols):
            df_wide[col] = (period + offset).astype(float)

        blanked = {rebal_dates[2]: sig_cols[:1], rebal_dates[3]: sig_cols[1:3]}
        for date, cols in blanked.items():
            df_wide.loc[date, cols] = np.nan
        non_rebal_date = df_wide.index[~df_wide.index.isin(rebal_dates)][0]
        df_wide.loc[non_rebal_date, :] = np.nan

        return df_wide, fx_fids, sig_cols, rebal_dates, period, blanked

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

        ## Test 2 - signals are read only on rebalance dates, and a signal missing on a
        ## rebalance date leaves that contract unpositioned for the whole period
        df_wide, fx_fids, sig_cols, rebal_dates, period, blanked = (
            self._rebalance_fixture()
        )
        expected_cols = [f"{fid}_{self.sname}_{self.pname}" for fid in fx_fids]

        for _leverage, _aum in [(1, 100), (3, 5e5), (0.5, 1.0)]:
            result = _leverage_positions(
                df_wide=df_wide.copy(),
                sname=self.sname,
                pname=self.pname,
                fids=fx_fids,
                rebal_freq=self.rebal_freq,
                leverage=_leverage,
                aum=_aum,
            )
            self.assertEqual(set(expected_cols), set(result.columns))

            # positions are held flat between rebalance dates
            for _, block in result.groupby(period):
                self.assertTrue((block.nunique(dropna=False) == 1).all())

            for rb_i, rb in enumerate(rebal_dates):
                # the signal on the k-th rebalance date is (k + 1 + offset)
                live = [
                    (offset, pos_col)
                    for offset, (pos_col, sig_col) in enumerate(
                        zip(expected_cols, sig_cols)
                    )
                    if sig_col not in blanked.get(rb, [])
                ]
                # the contracts that are still live absorb the full leverage budget
                rowsum = sum(rb_i + 1 + offset for offset, _ in live)
                for offset, pos_col in live:
                    np.testing.assert_allclose(
                        result.loc[rb, pos_col],
                        (rb_i + 1 + offset) * _aum * _leverage / rowsum,
                        rtol=1e-9,
                    )
                # a blanked contract is an explicit flat, not a NaN - these are all
                # interior gaps, with signals on both earlier and later rebalance dates
                for sig_col in blanked.get(rb, []):
                    pos_col = sig_col.replace(
                        f"_CSIG_{self.sname}", f"_{self.sname}_{self.pname}"
                    )
                    self.assertEqual(result.loc[rb, pos_col], 0.0)

    def test__leverage_positions_flat_only_inside_a_contracts_history(self):
        # A gap in the middle of a contract's history is an explicit flat (0), so the
        # close and the later reopen both show up as trades to the cost model. Dates
        # before the first signal and after the last are NaN - no position at all.
        idx = pd.bdate_range("2020-01-01", "2020-05-29")
        fids = ["USD_FX", "EUR_FX", "GBP_FX"]
        df_wide = pd.DataFrame(
            1.0, index=idx, columns=[f"{fid}_CSIG_STRAT" for fid in fids]
        )
        df_wide.index.name = "real_date"
        gbp = "GBP_FX_CSIG_STRAT"
        df_wide.loc[:"2020-01-31", gbp] = np.nan  # no history yet
        df_wide.loc[pd.Timestamp("2020-03-02"), gbp] = np.nan  # interior gap
        df_wide.loc["2020-05-01":, gbp] = np.nan  # history ended

        result = _leverage_positions(
            df_wide=df_wide, sname="STRAT", fids=fids, rebal_freq="m", aum=100
        )
        pos = result["GBP_FX_STRAT_POS"]
        self.assertTrue(np.isnan(pos.loc["2020-01-15"]))
        self.assertEqual(pos.loc["2020-03-16"], 0.0)
        self.assertTrue(np.isnan(pos.loc["2020-05-15"]))
        for live_date in ["2020-02-14", "2020-04-15"]:
            self.assertGreater(pos.loc[live_date], 0.0)

        # the flat period is entered and exited exactly once, as two real trades
        traded = pos.diff().abs()
        self.assertEqual(
            list(traded[traded > 0].index),
            [pd.Timestamp("2020-03-02"), pd.Timestamp("2020-04-01")],
        )
        # and the live contracts always carry the whole leverage budget
        np.testing.assert_allclose(result.abs().sum(axis=1), 100.0, rtol=1e-9)

    def test__leverage_positions_scales_by_gross_not_net(self):
        # Mixed-sign signals: leverage must scale by gross exposure
        # (sum of absolute signals), not net. With net scaling, positions
        # are too large (and can flip sign), and a market-neutral book
        # collapses to NaN because rowsums = 0 triggers div-by-zero masking.
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        sig_cols = [f"{fid}{self.sig_ident}" for fid in fx_fids]
        pos_cols = [f"{fid}_{self.sname}_{self.pname}" for fid in fx_fids]

        df_wide = self.mock_df_wide.copy()
        df_wide = df_wide[sig_cols]
        # Signals: +2, -1, +1, -2 -> gross = 6, net = 0
        signal_row = pd.Series({c: v for c, v in zip(sig_cols, [2.0, -1.0, 1.0, -2.0])})
        for col in sig_cols:
            df_wide[col] = signal_row[col]

        _aum, _leverage = 100.0, 1.0
        result = _leverage_positions(
            df_wide=df_wide.copy(),
            sname=self.sname,
            pname=self.pname,
            fids=fx_fids,
            leverage=_leverage,
            aum=_aum,
        )

        # Expected: position[fid] = signal[fid] * aum * leverage / gross_exposure
        gross = sum(abs(v) for v in signal_row.values)
        self.assertEqual(gross, 6.0)
        expected = {
            pos: signal_row[sig] * _aum * _leverage / gross
            for pos, sig in zip(pos_cols, sig_cols)
        }
        for pos_col, exp in expected.items():
            actual_vals = result[pos_col].dropna().unique()
            self.assertEqual(len(actual_vals), 1, msg=f"{pos_col} not flat")
            self.assertAlmostEqual(
                actual_vals[0],
                exp,
                places=10,
                msg=(
                    f"{pos_col}: expected {exp} (gross-exposure scaling), "
                    f"got {actual_vals[0]}. If this test fails, "
                    "_leverage_positions has reverted to net-sum scaling."
                ),
            )

        # Gross-scaled signs preserve the input sign on every leg.
        for pos_col, sig_col in zip(pos_cols, sig_cols):
            self.assertEqual(
                np.sign(result[pos_col].dropna().iloc[0]),
                np.sign(signal_row[sig_col]),
                msg=f"{pos_col} sign flipped vs input signal",
            )

    def test__leverage_positions_market_neutral_does_not_nan_out(self):
        # Long/short cancellation: net sum = 0 but gross > 0. The old
        # net-sum implementation NaN-ed the entire row to avoid div-by-zero;
        # gross-scaling keeps the row populated.
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        sig_cols = [f"{fid}{self.sig_ident}" for fid in fx_fids]
        pos_cols = [f"{fid}_{self.sname}_{self.pname}" for fid in fx_fids]

        df_wide = self.mock_df_wide.copy()[sig_cols]
        # Two longs of +1 and two shorts of -1 -> net = 0, gross = 4
        signal_vals = [1.0, -1.0, 1.0, -1.0]
        for col, val in zip(sig_cols, signal_vals):
            df_wide[col] = val

        _aum, _leverage = 100.0, 1.0
        result = _leverage_positions(
            df_wide=df_wide.copy(),
            sname=self.sname,
            pname=self.pname,
            fids=fx_fids,
            leverage=_leverage,
            aum=_aum,
        )

        # No row should be all NaN; every contract should have a finite position.
        self.assertFalse(
            result.isna().all(axis=1).any(),
            msg=(
                "Market-neutral row collapsed to NaN. "
                "_leverage_positions is using net rowsums instead of gross."
            ),
        )
        # Each position should be sign * aum * leverage / gross.
        for pos_col, sig_val in zip(pos_cols, signal_vals):
            exp = sig_val * _aum * _leverage / 4.0
            actual_vals = result[pos_col].dropna().unique()
            self.assertEqual(len(actual_vals), 1)
            self.assertAlmostEqual(actual_vals[0], exp, places=10)

    def test__dollar_per_signal_positions(self):
        ## Test 1 - Test with all values as 1
        df_wide = self.mock_df_wide.copy()
        # set all values to 1
        df_wide.loc[:, :] = 1
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        _aum, _dollar_per_signal = 1000, 2.0
        result = _dollar_per_signal_positions(
            df_wide=df_wide,
            sname=self.sname,
            pname=self.pname,
            fids=fx_fids,
            dollar_per_signal=_dollar_per_signal,
            aum=_aum,
        )
        # col names should be the FID+strat+pos (signal columns are filtered out)
        expected_cols = [f"{fid}_{self.sname}_{self.pname}" for fid in fx_fids]
        found_cols = list(result.columns)
        self.assertEqual(set(expected_cols), set(found_cols))

        # position = signal * dollar_per_signal -> every value equals _dollar_per_signal
        unique_values = set(result.values.flatten())
        self.assertEqual(len(unique_values), 1)
        self.assertEqual(unique_values, {1 * _dollar_per_signal})

        ## Test 2 - signals are read only on rebalance dates, and a signal missing on a
        ## rebalance date leaves that contract unpositioned for the whole period
        df_wide, fx_fids, sig_cols, rebal_dates, period, blanked = (
            self._rebalance_fixture()
        )
        expected_cols = [f"{fid}_{self.sname}_{self.pname}" for fid in fx_fids]

        for _dollar_per_signal, _aum in [(1, 100), (7, 5e5), (0.5, 1.0)]:
            with warnings.catch_warnings():
                # the AUM-exceed warning is irrelevant to the position formula here
                warnings.simplefilter("ignore")
                result = _dollar_per_signal_positions(
                    df_wide=df_wide.copy(),
                    sname=self.sname,
                    pname=self.pname,
                    fids=fx_fids,
                    rebal_freq=self.rebal_freq,
                    dollar_per_signal=_dollar_per_signal,
                    aum=_aum,
                )
            self.assertEqual(set(expected_cols), set(result.columns))

            # positions are held flat between rebalance dates
            for _, block in result.groupby(period):
                self.assertTrue((block.nunique(dropna=False) == 1).all())

            for rb_i, rb in enumerate(rebal_dates):
                for offset, (pos_col, sig_col) in enumerate(
                    zip(expected_cols, sig_cols)
                ):
                    if sig_col in blanked.get(rb, []):
                        # an interior gap is an explicit flat, not a NaN
                        self.assertEqual(result.loc[rb, pos_col], 0.0)
                    else:
                        # position = signal read on that rebalance date * dps
                        np.testing.assert_allclose(
                            result.loc[rb, pos_col],
                            (rb_i + 1 + offset) * _dollar_per_signal,
                            rtol=1e-9,
                        )

    def test__dollar_per_signal_positions_scales_by_signal_not_gross(self):
        # dollar_per_signal scales each signal directly: position = signal * dps.
        # Unlike leverage, there is NO normalisation by gross exposure or AUM, so a
        # contract's position depends only on its own signal, signs are preserved,
        # and a market-neutral book does not collapse to NaN.
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        sig_cols = [f"{fid}{self.sig_ident}" for fid in fx_fids]
        pos_cols = [f"{fid}_{self.sname}_{self.pname}" for fid in fx_fids]

        df_wide = self.mock_df_wide.copy()
        df_wide = df_wide[sig_cols]
        # Signals: +2, -1, +1, -2 -> net = 0, gross = 6
        signal_row = pd.Series({c: v for c, v in zip(sig_cols, [2.0, -1.0, 1.0, -2.0])})
        for col in sig_cols:
            df_wide[col] = signal_row[col]

        _aum, _dollar_per_signal = 1000.0, 3.0
        result = _dollar_per_signal_positions(
            df_wide=df_wide.copy(),
            sname=self.sname,
            pname=self.pname,
            fids=fx_fids,
            dollar_per_signal=_dollar_per_signal,
            aum=_aum,
        )

        # Expected: position[fid] = signal[fid] * dollar_per_signal, independent of
        # the other contracts (no gross-exposure or AUM scaling).
        expected = {
            pos: signal_row[sig] * _dollar_per_signal
            for pos, sig in zip(pos_cols, sig_cols)
        }
        for pos_col, exp in expected.items():
            actual_vals = result[pos_col].dropna().unique()
            self.assertEqual(len(actual_vals), 1, msg=f"{pos_col} not flat")
            self.assertAlmostEqual(
                actual_vals[0],
                exp,
                places=10,
                msg=(
                    f"{pos_col}: expected {exp} (signal * dollar_per_signal), "
                    f"got {actual_vals[0]}."
                ),
            )

        # Market-neutral book (net = 0) must not collapse to NaN.
        self.assertFalse(result.isna().all(axis=1).any())

        # Signs follow the input signal exactly on every leg.
        for pos_col, sig_col in zip(pos_cols, sig_cols):
            self.assertEqual(
                np.sign(result[pos_col].dropna().iloc[0]),
                np.sign(signal_row[sig_col]),
                msg=f"{pos_col} sign flipped vs input signal",
            )

    def test__dollar_per_signal_positions_warns_when_exceeding_aum(self):
        # When the total notional position on any date exceeds AUM the function
        # emits a UserWarning; otherwise it stays silent.
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        sig_cols = [f"{fid}{self.sig_ident}" for fid in fx_fids]
        df_wide = self.mock_df_wide.copy()[sig_cols]
        df_wide.loc[:, :] = 1.0  # gross position per date = len(fx_fids) * dps

        # dps large relative to aum -> total positions exceed aum -> warns
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _dollar_per_signal_positions(
                df_wide=df_wide.copy(),
                sname=self.sname,
                pname=self.pname,
                fids=fx_fids,
                dollar_per_signal=50.0,
                aum=10.0,
            )
        self.assertTrue(any(issubclass(c.category, UserWarning) for c in caught))
        self.assertTrue(any("exceed AUM" in str(c.message) for c in caught))

        # positions within aum -> no exceed-AUM warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _dollar_per_signal_positions(
                df_wide=df_wide.copy(),
                sname=self.sname,
                pname=self.pname,
                fids=fx_fids,
                dollar_per_signal=1.0,
                aum=1000.0,
            )
        self.assertFalse(any("exceed AUM" in str(c.message) for c in caught))

    def test__dollar_per_signal_positions_requires_number(self):
        # dollar_per_signal must be a number.
        fx_fids = [f"{cid}_FX" for cid in self.cids]
        df_wide = self.mock_df_wide.copy()
        with self.assertRaises(ValueError):
            _dollar_per_signal_positions(
                df_wide=df_wide,
                sname=self.sname,
                pname=self.pname,
                fids=fx_fids,
                dollar_per_signal="not-a-number",
            )

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

        basket_contracts = ["USD_EQ", "EUR_EQ"]
        basket_weights = [0.7, 0.3]

        df_cs: pd.DataFrame = contract_signals(
            df=df,
            sig="SIG",
            cids=cids,
            ctypes=ctypes,
            cscales=cscales,
            csigns=csigns,
            basket_contracts=basket_contracts,
            basket_weights=basket_weights,
            hedge_xcat="HR",
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
            start="1995-01-01",
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

    def _contract_signals_df(self) -> Tuple[pd.DataFrame, List[str]]:
        cids: List[str] = ["USD", "EUR", "GBP", "AUD", "CAD"]
        xcats: List[str] = ["SIG", "HR"]
        df: pd.DataFrame = make_test_df(
            cids=cids, xcats=xcats, start="2000-01-01", end="2002-01-01"
        )
        df.loc[(df["cid"] == "USD") & (df["xcat"] == "SIG"), "value"] = 1.0
        ctypes = ["FX", "IRS", "CDS"]
        df_cs: pd.DataFrame = contract_signals(
            df=df,
            sig="SIG",
            cids=cids,
            ctypes=ctypes,
            cscales=[1.0, 0.5, 0.1],
            csigns=[1, -1, 1],
            basket_contracts=["USD_EQ", "EUR_EQ"],
            basket_weights=[0.7, 0.3],
            hedge_xcat="HR",
        )
        fids: List[str] = [f"{cid}_{ctype}" for cid in cids for ctype in ctypes]
        return df_cs, fids

    def test_main_dollar_per_signal(self):
        df_cs, fids = self._contract_signals_df()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_notional: pd.DataFrame = notional_positions(
                df=df_cs,
                fids=fids,
                dollar_per_signal=0.5,
                sname="STRAT",
            )
        self.assertIsInstance(df_notional, QuantamentalDataFrame)

        # only position tickers are returned - the signal (_CSIG_) columns must
        # not leak through, matching the leverage / vol_target pathways.
        out_xcats = set(df_notional["xcat"].astype("object").unique())
        self.assertTrue(all(x.endswith("_STRAT_POS") for x in out_xcats))
        self.assertFalse(any("_CSIG_" in x for x in out_xcats))

        # on a rebalance date, position = that date's signal * dollar_per_signal; the
        # position is then held flat until the next rebalance date (slip=0 so signals
        # and positions share an index and can be compared directly).
        _dollar_per_signal, _rebal_freq = 2.0, "m"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_pos: pd.DataFrame = notional_positions(
                df=df_cs,
                fids=fids,
                dollar_per_signal=_dollar_per_signal,
                sname="STRAT",
                slip=0,
                rebal_freq=_rebal_freq,
            )
        pos_wide = qdf_to_ticker_df(df_pos)
        sig_wide = qdf_to_ticker_df(df_cs)
        rebal_dates = get_sops(dates=sig_wide.index, freq=_rebal_freq)
        period = pd.Series(
            pos_wide.index.isin(rebal_dates), index=pos_wide.index
        ).cumsum()
        held_dates = [dt for dt in rebal_dates if dt in pos_wide.index]
        for fid in fids:
            pos_col = f"{fid}_STRAT_POS"
            sig_col = f"{fid}_CSIG_STRAT"
            merged = pd.concat(
                [sig_wide.loc[held_dates, sig_col], pos_wide.loc[held_dates, pos_col]],
                axis=1,
            ).dropna()
            self.assertTrue(len(merged) > 0)
            np.testing.assert_allclose(
                merged[pos_col].values,
                merged[sig_col].values * _dollar_per_signal,
                rtol=1e-9,
            )
            # nothing moves between rebalance dates
            for _, block in pos_wide[pos_col].groupby(period):
                self.assertLessEqual(block.nunique(dropna=False), 1)

    def test_main_positioning_method_mutual_exclusivity(self):
        # Exactly one of `vol_target`, `leverage` or `dollar_per_signal` may be
        # specified - zero, two or three must raise ValueError.
        df_cs, fids = self._contract_signals_df()
        base = dict(df=df_cs, fids=fids, sname="STRAT")

        # exactly one method -> no mutual-exclusivity error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsInstance(
                notional_positions(**base, dollar_per_signal=0.5),
                QuantamentalDataFrame,
            )

        # zero methods specified -> ValueError
        with self.assertRaises(ValueError):
            notional_positions(**base)

        # two methods specified -> ValueError
        with self.assertRaises(ValueError):
            notional_positions(**base, leverage=1.1, dollar_per_signal=0.5)
        with self.assertRaises(ValueError):
            notional_positions(**base, vol_target=0.1, dollar_per_signal=0.5)

        # all three methods specified -> ValueError
        with self.assertRaises(ValueError):
            notional_positions(
                **base, leverage=1.1, vol_target=0.1, dollar_per_signal=0.5
            )


if __name__ == "__main__":
    unittest.main()
