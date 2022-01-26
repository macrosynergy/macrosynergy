
import unittest
import numpy as np
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.panel.panel_calculator_alt2 import panel_calculator
from random import randint, choice

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

        df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY'] = ['2011-01-01', '2020-12-31', 1, 2, 0.95, 1]
        df_xcats.loc['GROWTH'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}

        self.__dict__['blacklist'] = black
        self.__dict__['start'] = '2010-01-01'
        self.__dict__['end'] = '2020-12-31'

    def test_panel_calculator(self):

        self.dataframe_generator()

        # Test the panel calculator on various testcases and validate that the computed
        # values are correct.

        # i)
        formulas = ["NEW1 = np.abs( XR ) + 0.52 + 2 * CRY",
                    "NEW2 = NEW1 / XR"]
        df_calc = panel_calculator(df=self.dfd, calcs=formulas,
                                   cids=self.cids, start=self.start, end=self.end,
                                   blacklist=self.blacklist)
        # First preliminary check is the presence of both categories.
        cats_df = list(df_calc['xcat'].unique())
        self.assertTrue(cats_df == ["NEW1", "NEW2"])

        # The dimensions of the returned dataframe should match the dimensions of the
        # input dataframe once reduced to the categories involved in the expression.
        filt_1 = (self.dfd['xcat'] == 'XR') | (self.dfd['xcat'] == 'CRY')
        filt_df = self.dfd[filt_1]
        date_column = filt_df['real_date']

        # The produced category, "NEW1", should be defined over the below date-series
        # given the alignment of pandas. It is worth noting that the two series are not
        # defined over the same time-period but pd.DataFrames will match on the index.
        first_date = np.min(date_column)
        end_date = np.max(date_column)
        filt_2 = (df_calc['xcat'] == "NEW1")
        df_new1 = df_calc[filt_2].pivot(index='real_date', columns='cid', values='value')
        date_range = list(df_new1.index)

        self.assertTrue(first_date == date_range[0])
        self.assertTrue(end_date == date_range[-1])

        dates = list(self.dfd['real_date'])
        date = choice(dates)
        # Test on Australia.
        row_value = df_new1.loc[date]['AUD']

        input_values = filt_df[filt_df['real_date'] == date]
        input_values = input_values[input_values['cid'] == 'AUD']
        xr = input_values[input_values['xcat'] == 'XR']['value']
        cry = input_values[input_values['xcat'] == 'CRY']['value']

        # Manually produce: "NEW1 = np.abs( XR ) + 0.52 + 2 * CRY".
        self.assertTrue(row_value == (np.abs(float(xr)) + 0.52 + 2 * float(cry)))


if __name__ == '__main__':

    unittest.main()