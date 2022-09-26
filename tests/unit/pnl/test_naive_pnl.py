
from tests.simulate import make_qdf
from macrosynergy.pnl.naive_pnl import NaivePnL
from macrosynergy.management.shape_dfs import reduce_df
import unittest
import numpy as np
import pandas as pd


class TestAll(unittest.TestCase):

    def dataframe_construction(self):

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP', 'NZD', 'USD', 'EUR']
        self.__dict__['xcats'] = ['EQXR', 'CRY', 'GROWTH', 'INFL', 'DUXR']

        df_cids = pd.DataFrame(index=self.cids, columns=['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])
        df_cids.loc['AUD', :] = ['2008-01-03', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2010-01-03', '2020-11-30', 0, 1]
        df_cids.loc['GBP', :] = ['2012-01-03', '2020-11-30', -0.2, 0.5]
        df_cids.loc['NZD'] = ['2002-01-03', '2020-09-30', -0.1, 2]
        df_cids.loc['USD'] = ['2015-01-03', '2020-12-31', 0.2, 2]
        df_cids.loc['EUR'] = ['2008-01-03', '2020-12-31', 0.1, 2]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add',
                                         'sd_mult', 'ar_coef', 'back_coef'])

        df_xcats.loc['EQXR'] = ['2005-01-03', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
        df_xcats.loc['GROWTH'] = ['2010-01-03', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
        df_xcats.loc['DUXR'] = ['2000-01-01', '2020-12-31', 0.1, 0.5, 0, 0.1]

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}
        self.__dict__['blacklist'] = black

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

    def test_constructor(self):
        # Test NaivePnL's constructor and the instantiation of the respective fields.

        self.dataframe_construction()

        ret = ['EQXR']
        sigs = ['CRY', 'GROWTH', 'INFL']
        pnl = NaivePnL(self.dfd, ret=ret[0], sigs=sigs, cids=self.cids,
                       start='2000-01-01', blacklist=self.blacklist,
                       )
        # Confirm the categories held in the reduced DataFrame, on the instance's field,
        # are exclusively the return and signal category. This will occur if benchmarks
        # have not been defined.
        test_categories = list(pnl.df['xcat'].unique())
        self.assertTrue(sorted(test_categories) == sorted(ret + sigs))

        # Add "external" benchmarks to the instance: a category that is neither the
        # return field or one of the categories. The benchmarks will be added to the
        # instance's DataFrame.
        pnl = NaivePnL(self.dfd, ret=ret[0], sigs=sigs, cids=self.cids,
                       start='2000-01-01', blacklist=self.blacklist,
                       bms=["EUR_DUXR", "USD_DUXR"]
                       )
        test_categories = list(pnl.df['xcat'].unique())
        self.assertTrue(sorted(test_categories) == sorted(ret + sigs + ['DUXR']))

        # Test that both the benchmarks are held in the DataFrame. Implicitly validating
        # that add_bm() method works correctly. The benchmark series will be appended to
        # the DataFrame held on the instance: confirm their presence.
        first_bm = pnl.df[(pnl.df['cid'] == "EUR") & (pnl.df['xcat'] == "DUXR")]
        self.assertTrue(not first_bm.empty)
        second_bm = pnl.df[(pnl.df['cid'] == "USD") & (pnl.df['xcat'] == "DUXR")]
        self.assertTrue(not second_bm.empty)

        # Additionally, confirm that the benchmark dictionary has been populated
        # correctly as both benchmarks are present in the passed DataFrame.
        bm_tickers = list(pnl._bm_dict.keys())
        self.assertTrue(sorted(bm_tickers) == ["EUR_DUXR", "USD_DUXR"])

        # Confirm the values are correct. Confirm the values in each benchmark series
        # have been correctly lifted from the original, standardised DataFrame.
        eur_duxr = self.dfd[(self.dfd['cid'] == "EUR") & (self.dfd['xcat'] == "DUXR")]

        self.assertTrue(np.all(first_bm['value'].to_numpy()
                               == eur_duxr['value'].to_numpy()))

        self.assertTrue(np.all(np.squeeze(pnl._bm_dict["EUR_DUXR"].to_numpy())
                               == eur_duxr['value'].to_numpy()))

        # Confirm the benchmark functionality works when passing in a single ticker.
        # Also, the benchmark will already be present on the instance's DataFrame.
        pnl = NaivePnL(self.dfd, ret=ret[0], sigs=sigs, cids=self.cids,
                       start='2000-01-01', blacklist=self.blacklist,
                       bms="EUR_EQXR"
                       )
        bm_tickers = list(pnl._bm_dict.keys())
        self.assertTrue(sorted(bm_tickers) == ["EUR_EQXR"])

    def test_make_signal(self):

        self.dataframe_construction()
        df = self.dfd

        ret = 'EQXR'
        sigs = ['CRY', 'GROWTH', 'INFL']
        pnl = NaivePnL(self.dfd, ret=ret, sigs=sigs, cids=self.cids,
                       start='2000-01-01', blacklist=self.blacklist,
                       bms=["EUR_DUXR", "USD_DUXR"]
                       )

        # Test the method used for producing the signals. The signal is based on a single
        # category and the function allows for applying transformations to the signal to
        # determine the extent of the position.
        # For instance, distance from the neutral level measured in standard deviations.
        # Or a digital transformation: if the signal category is positive, take a unitary
        # long position.

        # Specifically chosen signal that will have leading NaN values. To test if
        # functionality incorrectly populates unrealised dates.
        sig = 'GROWTH'
        dfx = df[df['xcat'].isin([ret, sig])]

        # Adjust for any blacklist periods.
        dfx = reduce_df(df=dfx, xcats=[ret, sig], cids=self.cids,
                        blacklist=self.blacklist, out_all=False)

        # Will return a DataFrame with the transformed signal.
        dfw = pnl.__make_signal__(dfx=dfx, sig=sig, sig_op='zn_score_pan',
                                  min_obs=252, iis=True, sequential=True,
                                  neutral='zero', thresh=None)
        self.__dict__['signal_dfw'] = dfw

        # Confirm the first dates for each cross-section's signal are the expected start
        # dates. There are not any falsified signals being created. The signal is
        # 'GROWTH'.
        # Dates have been adjusted for the first business day.
        expected_start = {'AUD': '2010-01-04', 'CAD': '2010-01-04', 'GBP': '2012-01-03',
                          'NZD': '2010-01-04', 'USD': '2015-01-05', 'EUR': '2010-01-04'}
        signal_column = dfw['psig']
        signal_column = signal_column.reset_index()
        signal_column = signal_column.rename(columns={"psig": "value"})
        signal_column['xcat'] = 'psig'

        dfw_signal = signal_column.pivot(index='real_date', columns='cid',
                                         values='value')
        cross_sections = dfw_signal.columns
        # Confirms make_zn_scores does not produce any signals for non-realised dates.
        for c in cross_sections:
            column = dfw_signal.loc[:, c]
            self.assertTrue(column.first_valid_index() ==
                            pd.Timestamp(expected_start[c]))

    @staticmethod
    def diff_month(d1, d2):
        return (d1.year - d2.year) * 12 + d1.month - d2.month

    def test_rebalancing_dates(self):

        self.test_make_signal()

        dfw = self.signal_dfw
        dfw.reset_index(inplace=True)
        dfw = dfw.rename_axis(None, axis=1)

        dfw = dfw.sort_values(['cid', 'real_date'])

        sig_series = NaivePnL.rebalancing(dfw, rebal_freq='monthly')
        dfw['sig'] = np.squeeze(sig_series.to_numpy())

        dfw_signal_rebal = dfw.pivot(index='real_date', columns='cid',
                                     values='sig')

        # Confirm, on a single cross-section that re-balancing occurs on a monthly basis.
        # The number of unique values will equate to the number of months in the
        # time-series.
        dfw_signal_rebal_aud = dfw_signal_rebal.loc[:, 'AUD']
        aud_array = np.squeeze(dfw_signal_rebal_aud.to_numpy())
        unique_values_aud = set(aud_array)

        start_date = dfw_signal_rebal.index[0]
        end_date = dfw_signal_rebal.index[-1]

        no_months = self.diff_month(end_date, start_date)

        self.assertTrue(no_months - 1 == len(unique_values_aud))

    def test_make_pnl(self):

        self.test_make_signal()

        # Signal is produced daily. The calculation of the neutral level and standard
        # deviation are also calculated daily. Only for a highly volatile asset would
        # there be any value in calculating at a daily frequency. In most instances, the
        # neutral level would remain fairly constant over the duration of a week.
        dfw = self.signal_dfw

        # The PnL DataFrame is appended to the instance DataFrame.
        ret = 'EQXR'
        sigs = ['CRY', 'GROWTH', 'INFL']
        pnl = NaivePnL(self.dfd, ret=ret, sigs=sigs, cids=self.cids,
                       start='2000-01-01', blacklist=self.blacklist,
                       bms=["EUR_DUXR", "USD_DUXR"]
                       )

        pnl.make_pnl(sig='GROWTH', sig_op='zn_score_pan', rebal_freq='daily',
                     vol_scale=None, rebal_slip=0, pnl_name='PNL_GROWTH',
                     min_obs=252, iis=True, sequential=True, neutral='zero', thresh=None)

        # Test the PnL value produced across the panel. Implicitly tests the PnL values
        # for each cross-section.
        # Confirms the category for the PnL is being named correctly.
        pnl_df = pnl.df[pnl.df['xcat'] == 'PNL_GROWTH']

        # Confirm the first PnL value for each cross-section is aligned to the first date
        # of the respective signal, GROWTH. A PnL value should only be produced if the
        # signal is available for the respective date.
        expected_start = {'AUD': '2010-01-04', 'CAD': '2010-01-04', 'GBP': '2012-01-03',
                          'NZD': '2010-01-04', 'USD': '2015-01-05', 'EUR': '2010-01-04'}

        pnl_dfw = pnl_df.pivot(index='real_date', columns='cid',
                               values='value')
        cross_sections = self.cids
        for c in cross_sections:
            column = pnl_dfw.loc[:, c]
            # Adjust the expected start dates by one day to account for the shift
            # mechanism. The computed signal is used for the following day's position.
            self.assertTrue(column.first_valid_index() ==
                            pd.Timestamp(expected_start[c]) + pd.DateOffset(1))

        # Choose a quasi-random sample of dates to confirm the logic of computing the
        # PnL. Multiply each cross-section's signal by their respective return.
        # A "random" sample of dates (will be inclusive of dates where some
        # cross-sections have NaN values and blacklists have been applied.).
        fixed_dates = ["2010-01-13", "2012-01-26", "2015-01-20", "2019-01-08"]

        # Shift the signal by a single date. Replicating the logic in make_pnl().
        dfw['psig'] = dfw['psig'].groupby(level=0).shift(1)
        dfw.reset_index(inplace=True)
        dfw = dfw.rename_axis(None, axis=1)
        dfw = dfw.sort_values(['cid', 'real_date'])
        dfw = dfw.rename({'psig': 'sig'}, axis=1)

        dfw_sig = dfw.pivot(index='real_date', columns='cid',
                            values='sig')

        # Confirm the logic on a small but representative sample of dates.
        for date in fixed_dates:

            signals = dfw_sig.loc[date, :]
            signal_dict = dict(signals)
            df = self.dfd

            returns = df[(df['xcat'] == ret)]
            returns_dfw = returns.pivot(index='real_date', columns='cid',
                                        values='value')

            return_dict = dict(returns_dfw.loc[date, :])

            # Aggregate the individual cross-section's PnL to calculate the PnL across
            # the panel (weighted according to the signal).
            pnl_return_date = 0
            condition = lambda a, b: str(a) == 'nan' or str(b) == 'nan'
            for cid, value in signal_dict.items():
                # Mitigates for NaN values. Exclude from calculation - only sum on
                # realised dates.
                if condition(return_dict[cid], value):
                    pass
                else:
                    pnl_return_date += return_dict[cid] * value

            test_data = pnl_dfw['ALL'].loc[date]
            self.assertTrue(round(float(test_data), 4) == round(pnl_return_date, 4))

    def test_make_pnl_neg(self):

        self.dataframe_construction()

        # The majority of the logic for make_pnl is tested through the method
        # test_make_pnl(). Therefore, aim to isolate the application of the negative
        # signal through evaluate_pnl() method.
        # For make_pnl(), the sig_neg parameter will be set to True and the associated
        # transformed signal will be multiplied by minus one.
        # To test the negative signal, call make_pnl() on the same raw signal but set the
        # sig_neg parameter to True and False. The two produced PnL series should have an
        # inverse relationship with any benchmark.

        ret = "EQXR"
        sigs = ["CRY", "GROWTH", "INFL"]
        bms = ["EUR_DUXR", "USD_DUXR"]

        pnl = NaivePnL(
            self.dfd, ret=ret, sigs=sigs, cids=self.cids, start="2000-01-01",
            blacklist=self.blacklist, bms=bms
        )

        # Set the signal to True.
        # Will implicitly test if the PnL name, using the default mechanism, will have
        # the postfix "_NEG" appended given sig_neg is set to True.
        pnl.make_pnl(
            sig="INFL", sig_op="zn_score_pan", sig_neg=True, rebal_freq="monthly",
            vol_scale=5, rebal_slip=1, min_obs=250, thresh=2
        )

        # Same parameter but sig_neg is set to False.
        pnl.make_pnl(
            sig="INFL", sig_op="zn_score_pan", sig_neg=False, rebal_freq="monthly",
            vol_scale=5, rebal_slip=1, min_obs=250, thresh=2
        )

        # Confirm the direct negative correlation across the two PnLs. By adding the
        # correlation coefficients with the benchmarks, the value should equate to
        # zero.
        df_eval = pnl.evaluate_pnls(
            pnl_cats=["PNL_INFL", "PNL_INFL_NEG"],
        )

        bm_correl = df_eval.loc[[b + " correl" for b in bms], :]
        self.assertTrue(np.all(bm_correl.sum(axis=1).to_numpy()) == 0)

    def test_make_long_pnl(self):

        self.dataframe_construction()

        ret = 'EQXR'
        sigs = ['CRY', 'GROWTH', 'INFL']
        pnl = NaivePnL(self.dfd, ret=ret, sigs=sigs, cids=self.cids,
                       start='2000-01-01', blacklist=self.blacklist,
                       bms=["EUR_DUXR", "USD_DUXR"]
                       )

        pnl.make_pnl(sig='GROWTH', sig_op='zn_score_pan', rebal_freq='daily',
                     vol_scale=None, rebal_slip=0, pnl_name='PNL_GROWTH',
                     min_obs=252, iis=True, sequential=True, neutral='zero',
                     thresh=None)

        pnl.make_long_pnl(vol_scale=0, label="Unit_Long_EQXR")

        long_equity = pnl.df[pnl.df['xcat'] == "Unit_Long_EQXR"]
        # Long-only is naturally computed across the panel (individual cross-section's
        # returns are already present in the DataFrame). Therefore, confirm that the
        # only cross-section in 'cid' column is "ALL".
        self.assertTrue(list(long_equity['cid'].unique()) == ["ALL"])

        df = self.dfd
        return_df = df[df['xcat'] == "EQXR"]

        # Test on a random date.
        random_date = "2016-01-19"
        return_dfw = return_df.pivot(index='real_date', columns='cid',
                                     values='value')
        # Sum across the row: unitary position.
        return_calc = sum(return_dfw.loc[random_date, :])
        # Convert to a pd.Series.
        long_equity_series = long_equity.pivot(index='real_date', columns='cid',
                                               values='value')

        condition = return_calc - float(long_equity_series.loc[random_date])
        self.assertTrue(abs(condition) < 0.0001)

        # The remaining methods in NaivePnL are graphical plots which display the values
        # computed using the functions above. Therefore, if the functionality is correct
        # above, the plotting methods do not explicitly need to be tested in the Unit
        # Test as a visual assessment will be sufficient.


if __name__ == '__main__':

    unittest.main()


