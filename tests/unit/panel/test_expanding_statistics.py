
import unittest
import numpy as np
import pandas as pd
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.expanding_statistics import expanding_mean_with_nan

class TestAll(unittest.TestCase):

    def dataframe_generator(self):
        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP', 'NZD']
        self.__dict__['xcats'] = ['XR', 'CRY', 'GROWTH', 'INFL']
        df_cids = pd.DataFrame(index=self.cids,
                               columns=['earliest', 'latest', 'mean_add', 'sd_mult'])

        df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0.1, 1]
        df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
        df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add', 'sd_mult',
                                         'ar_coef', 'back_coef'])

        df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY'] = ['2000-01-01', '2020-12-31', 1, 2, 0.95, 1]
        df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

        dfd_xr = dfd[dfd['xcat'] == 'XR']
        self.__dict__['dfd_xr'] = dfd_xr

        dfw = dfd_xr.pivot(index='real_date', columns='cid', values='value')
        self.__dict__['dfw'] = dfw
        no_rows = dfw.shape[0]

        self.__dict__['no_timestamps'] = no_rows

    def test_rolling_mean(self):

        self.dataframe_generator()

        ar_neutral = expanding_mean_with_nan(dfw=self.dfw)

        benchmark_pandas = [self.dfw.iloc[0:(i + 1), :].stack().mean()
                            for i in range(self.no_timestamps)]

        self.assertTrue(len(ar_neutral) == len(benchmark_pandas))

        for i, elem in enumerate(ar_neutral):
            bm_elem = round(benchmark_pandas[i], 4)
            self.assertTrue(round(elem, 4) == bm_elem)

        bm_expanding = self.dfw.mean(axis=1)
        bm_expanding = bm_expanding.expanding(min_periods=1).mean()

        # Test on another category to confirm the logic.
        dfd_cry = self.dfd[self.dfd['xcat'] == 'CRY']
        dfw_cry = dfd_cry.pivot(index='real_date', columns='cid', values='value')

        ar_neutral = expanding_mean_with_nan(dfw=dfw_cry)
        benchmark_pandas_cry = [dfw_cry.iloc[0:(i + 1), :].stack().mean()
                                for i in range(self.no_timestamps)]

        self.assertTrue(len(ar_neutral) == len(benchmark_pandas_cry))
        for i, elem in enumerate(ar_neutral):
            bm_elem_cry = round(benchmark_pandas_cry[i], 4)
            self.assertTrue(round(elem, 4) == bm_elem_cry)


if __name__ == '__main__':

    unittest.main()