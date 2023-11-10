import unittest
import random
import numpy as np
from macrosynergy.backend import get_current_backend

if get_current_backend() == "pandas":
    import pandas as pd
elif get_current_backend() == "modin.pandas":
    import modin.pandas as pd
import os
from typing import List
from macrosynergy.management.simulate import *


class Test_All(unittest.TestCase):

    def df_construction(self):
        cids = ['AUD', 'CAD', 'GBP']
        xcats = ['XR', 'CRY']
        df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])
        df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2011-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP', :] = ['2011-01-01', '2020-11-30', -0.2, 0.5]

        df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])

        df_xcats.loc['XR', :] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
        df_xcats.loc['CRY', :] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]

        random.seed(1)
        self.dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    def df_construct_black(self):
        cids = ['AUD', 'CAD', 'GBP']
        # The algorithm is designed to test on a singular category.
        xcats = ['XR']
        df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest'])
        df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31']
        df_cids.loc['CAD', :] = ['2011-01-01', '2021-11-25']
        df_cids.loc['GBP', :] = ['2011-01-01', '2020-11-30']

        df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest'])

        df_xcats.loc['XR', :] = ['2010-01-01', '2021-11-25']

        # Construct an arbitrary dictionary used to test the design of make_qdf_black()
        # function.
        # The last key's enddate is purposefully chosen to fall on the weekend, Saturday,
        # to examine the logic concerning weekends: the expected output should be to
        # shift the end-date to the next available trading day, Monday.
        self.blackout = {'AUD': ('2010-01-12', '2010-06-14'),
                         'CAD_1': ('2011-01-04', '2011-01-23'),
                         'CAD_2': ('2013-01-09', '2013-04-10'),
                         'CAD_3': ('2015-01-12', '2015-03-12'),
                         'CAD_4': ('2021-11-01', '2021-11-20')}

        random.seed(1)
        self.black_dfd = make_qdf_black(df_cids, df_xcats, self.blackout)

    @staticmethod
    def handle_nan(arr):
        arr = np.nan_to_num(arr)
        arr = arr[arr != 0.0]

        return arr

    def ar1_coef(self, x):

        x = self.handle_nan(x)
        arr_1 = x[:-1]
        arr_2 = x[1:]

        return np.corrcoef(np.array([arr_1, arr_2]))[0, 1]

    def cor_coef(self, df, ticker_x, ticker_y):
        x = ticker_x.split('_', 1)
        y = ticker_y.split('_', 1)
        filt_x = (df['cid'] == x[0]) & (self.dfd['xcat'] == x[1])
        filt_y = (df['cid'] == y[0]) & (self.dfd['xcat'] == y[1])
        dfd_x = self.dfd.loc[filt_x, ].set_index('real_date')['value']
        dfd_y = self.dfd.loc[filt_y, ].set_index('real_date')['value']

        dfd_xy = pd.merge(dfd_x, dfd_y, how='inner', left_index=True, right_index=True)
        return dfd_xy.corr().iloc[0, 1]

    def test_simulate_ar(self):

        random.seed(1)
        ser_ar = simulate_ar(100, mean=2, sd_mult=3, ar_coef=0.75)
        self.assertGreater(self.ar1_coef(ser_ar), 0.25)
        self.assertEqual(np.round(np.std(ser_ar), 2), 3)
        self.assertGreater(np.mean(ser_ar), 2)

    def test_qdf_starts(self):

        self.df_construction()
        # Utilise ampersand for element - wise logical - "and". Will return a Boolean
        # Pandas Series or Numpy Array.
        filt1 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'CRY')
        self.assertEqual(np.min(self.dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2011-01-03'))

        filt1 = (self.dfd['cid'] == 'GBP') & (self.dfd['xcat'] == 'XR')
        self.assertEqual(np.min(self.dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2011-01-03'))

    def test_qdf_ends(self):

        self.df_construction()
        filt1 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'CRY')
        self.assertEqual(np.max(self.dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2020-10-30'))

        filt1 = (self.dfd['cid'] == 'GBP') & (self.dfd['xcat'] == 'XR')
        self.assertEqual(np.max(self.dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2020-11-30'))

    def test_qdf_correl(self):

        self.df_construction()
        # self.assertGreater(self.cor_coef(self.dfd, 'AUD_XR', 'CAD_XR'), 0)
        # self.assertGreater(self.cor_coef(self.dfd, 'AUD_XR', 'GBP_XR'), 0)

    def test_qdf_ar(self):

        self.df_construction()
        filt1 = (self.dfd['cid'] == 'AUD') & (self.dfd['xcat'] == 'CRY')
        # self.assertGreater(self.ar1_coef(self.dfd.loc[filt1, 'value']), 0.25)

        filt1 = (self.dfd['cid'] == 'CAD') & (self.dfd['xcat'] == 'CRY')
        # self.assertGreater(self.ar1_coef(self.dfd.loc[filt1, 'value']), 0.25)

        filt1 = (self.dfd['cid'] == 'GBP') & (self.dfd['xcat'] == 'CRY')
        # self.assertGreater(self.ar1_coef(self.dfd.loc[filt1, 'value']), 0.25)

    def test_make_qdf_black(self):
        self.df_construct_black()

        # Rudimentary Unit Test to confirm the correct expected start and end date of
        # randomly chosen data series.
        filt1 = (self.black_dfd['cid'] == 'AUD') & (self.black_dfd['xcat'] == 'XR')
        self.assertEqual(np.min(self.black_dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2010-01-01'))

        filt1 = (self.black_dfd['cid'] == 'GBP') & (self.black_dfd['xcat'] == 'XR')
        self.assertEqual(np.min(self.black_dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2011-01-03'))

        filt1 = (self.black_dfd['cid'] == 'AUD') & (self.black_dfd['xcat'] == 'XR')
        self.assertEqual(np.max(self.black_dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2020-12-31'))

        filt1 = (self.black_dfd['cid'] == 'GBP') & (self.black_dfd['xcat'] == 'XR')
        self.assertEqual(np.max(self.black_dfd.loc[filt1, 'real_date']),
                         pd.to_datetime('2020-11-30'))

        # Test all values are Boolean values.
        values = self.black_dfd['value'].to_numpy()
        self.assertTrue(all(val == 0 or val == 1 for val in values))

        # The field self.blackout is the dates dictionary of blackout periods.
        aud_blackout = self.blackout['AUD']
        start = aud_blackout[0]
        end = aud_blackout[1]

        # Validate the number of timestamps, for a particular cross-section, with a
        # "value" equated to one equals the expected number which can be precomputed.
        # Only limitation is it is invariant to the actual timestamp and instead focuses
        # the longevity of the blackout period being correct.
        all_days = pd.date_range(start, end)
        work_days = all_days[all_days.weekday < 5]

        aud_df = self.black_dfd[self.black_dfd['cid'] == 'AUD']
        black_aud_df = aud_df[aud_df['value'] == 1]

        self.assertTrue(len(work_days) == black_aud_df.shape[0])

        dates_df = black_aud_df['real_date'].to_numpy()

        self.assertTrue(pd.to_datetime(start) == dates_df[0])
        self.assertTrue(pd.to_datetime(end) == dates_df[-1])

        # Test the "weekend handler" on the date '2021-11-20' (Saturday). The expected
        # end date of Canada's blackout period should be '2021-11-22'.
        cad_blackout = self.blackout['CAD_4']
        # In this test, only interested in the end-date.
        end = cad_blackout[1]

        # Isolate the relevant DataFrame.
        cad_df = self.black_dfd[self.black_dfd['cid'] == 'CAD']
        black_cad_df = cad_df[cad_df['value'] == 1]
        dates_df = black_cad_df['real_date'].to_numpy()

        # "weekend handler" will shift the date forwards.
        self.assertTrue(pd.to_datetime('2021-11-22') == dates_df[-1])
        
        
    def test_generate_lines(self):
        
        # generate_lines(sig_len : uint, style : str) -> Union[np.ndarray, Dict[str, np.ndarray]]
        test_len : int = 100
        line_styles : List[str] = generate_lines(sig_len=test_len, style='all')
        
        # assert that output is a dictionary and is not empty
        self.assertTrue(isinstance(line_styles, list) and all([isinstance(ix, str) for ix in line_styles]))
        self.assertTrue(len(line_styles) > 0)
        
        for l_style in line_styles:
            t_o : np.ndarray = generate_lines(sig_len=test_len, style=l_style)
            self.assertTrue(len(t_o) == test_len)
        
        # generate an array of random numbers in range(65, 91) of random len in range(3, 10)
        while True:
            r_str : str = ''.join([chr(i) for i in np.random.randint(65, 91, np.random.randint(3, 10))])
            if r_str not in line_styles:
                break
        
        self.assertRaises(ValueError, generate_lines, sig_len=test_len, style=r_str)
        self.assertRaises(ValueError, generate_lines, sig_len=-10, style=line_styles[0])
        self.assertRaises(ValueError, generate_lines, sig_len=0, style=line_styles[0])
        
        r : np.ndarray = generate_lines(sig_len=test_len, style=line_styles[0])
        l : List[bool] = []
        for _ in range(100):
            rx : np.ndarray = generate_lines(sig_len=test_len, style='any')
            l.append(np.array_equal(rx, r))
        
        self.assertTrue(sum(l) < len(l))
        
    def test_make_test_df(self):
        cids : List[str] = ['AUD', 'CAD', 'GBP', 'USD']
        xcats : List[str] = ['XR', 'IR']
        tickers : List[str] = [f'{cid}_{xc}' for cid in cids for xc in xcats]
        start_date : str = '2010-01-01'
        end_date : str = '2020-12-31'
        date_range : pd.DatetimeIndex = pd.bdate_range(start=start_date, end=end_date)
        line_styles : List[str] = generate_lines(sig_len=len(date_range), style='all')
        
        for ls in list(line_styles):
            df : pd.DataFrame = make_test_df(cids=cids, xcats=xcats, start=start_date, end=end_date, style=ls)
            
            self.assertTrue(isinstance(df, pd.DataFrame))
            self.assertFalse(df.empty)
            self.assertTrue(set(df.columns) == {'cid', 'xcat', 'real_date', 'value'})
            
            ebdates : pd.DatetimeIndex = pd.bdate_range(start=start_date, end=end_date)
            self.assertTrue(set(ebdates) == set(df['real_date']))
            self.assertTrue(df['real_date'].nunique() == len(generate_lines(sig_len=len(ebdates), style=ls)))
            self.assertTrue(df['real_date'].nunique() == len(ebdates))

            self.assertTrue(df['cid'].nunique() == len(cids))
            self.assertTrue(df['xcat'].nunique() == len(xcats))
            self.assertTrue(df['cid'].nunique() * df['xcat'].nunique() == len(tickers))
            self.assertTrue(set(df['cid'] + '_' + df['xcat']) == set(tickers))
            self.assertTrue(df.shape[0] == len(ebdates) * len(tickers))
            
            for cid in cids:
                for xcat in xcats:
                    t_df : pd.DataFrame = df[(df['cid'] == cid) & (df['xcat'] == xcat)]
                    self.assertTrue(set(ebdates) == set(t_df['real_date']))
                    # assert that the values are the same as the line style
                    self.assertTrue(np.array_equal(t_df['value'].to_numpy(), generate_lines(sig_len=len(ebdates), style=ls)))
        



if __name__ == '__main__':

    unittest.main()
