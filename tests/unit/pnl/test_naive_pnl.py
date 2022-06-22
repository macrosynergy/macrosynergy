
from tests.simulate import make_qdf
from macrosynergy.pnl.naive_pnl import NaivePnL

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
        # that add_bm() method works correctly.
        first_bm = pnl.df[(pnl.df['cid'] == "EUR") & (pnl.df['xcat'] == "DUXR")]
        self.assertTrue(not first_bm.empty)
        second_bm = pnl.df[(pnl.df['cid'] == "USD") & (pnl.df['xcat'] == "DUXR")]
        self.assertTrue(not second_bm.empty)

        # Confirm the values are correct.
        eur_duxr = self.dfd[(self.dfd['cid'] == "EUR") & (self.dfd['xcat'] == "DUXR")]
        self.assertTrue(np.all(first_bm['value'] == eur_duxr['value']))

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
        # Will return a DataFrame with the transformed signal.
        dfw = pnl.make_signal(dfx=dfx, sig=sig, sig_op='zn_score_pan',
                              min_obs=252, iis=True, sequential=True,
                              neutral='zero', thresh=None)
        self.__dict__['signal_dfw'] = dfw

        # Confirm the first dates for each cross-section's signal are the expected start
        # dates. There are not any falsified signals being created.
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


if __name__ == '__main__':

    unittest.main()


