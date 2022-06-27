

import unittest
from macrosynergy.signal.signal_return import SignalReturnRelations
from macrosynergy.management.shape_dfs import reduce_df
from tests.simulate import make_qdf
import random
import pandas as pd
import numpy as np
import math

class TestAll(unittest.TestCase):

    def dataframe_generator(self):
        """
        Create a standardised DataFrame defined over the three categories.
        """

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP', 'NZD', 'USD']
        self.__dict__['xcats'] = ['XR', 'CRY', 'GROWTH', 'INFL']

        df_cids = pd.DataFrame(index=self.cids, columns=['earliest', 'latest',
                                                         'mean_add', 'sd_mult'])

        df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
        df_cids.loc['CAD'] = ['2010-01-01', '2020-12-31', 0, 2]
        df_cids.loc['GBP'] = ['2010-01-01', '2020-12-31', 0, 5]
        df_cids.loc['NZD'] = ['2010-01-01', '2020-12-31', 0, 3]
        df_cids.loc['USD'] = ['2010-01-01', '2020-12-31', 0, 4]

        df_xcats = pd.DataFrame(index=self.xcats, columns=['earliest', 'latest',
                                                           'mean_add', 'sd_mult',
                                                           'ar_coef', 'back_coef'])

        df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 0, 2, 0.95, 1]
        df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 0, 2, 0.9, 1]
        df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 0, 2, 0.8, 0.5]

        random.seed(2)
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.__dict__['dfd'] = dfd

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}

        self.__dict__['blacklist'] = black

        assert 'dfd' in vars(self).keys(), "Instantiation of DataFrame missing from " \
                                           "field dictionary."

    def test_constructor(self):

        self.dataframe_generator()
        # Test the Class's constructor.

        signal = 'CRY'
        srr = SignalReturnRelations(self.dfd, sig=signal, ret='XR',
                                    freq='D', blacklist=self.blacklist)

        # The signal will invariably be used as the explanatory variable and the return
        # as the dependent variable.
        # Confirm that the signal is lagged after applying categories_df().

        # Choose an arbitrary date and confirm that the signal in the original DataFrame
        # has been lagged by a day. Confirm on multiple cross-sections: AUD & USD.
        df_signal = self.dfd[self.dfd['xcat'] == signal]
        arbitrary_date_one = '2010-01-07'
        arbitrary_date_two = '2020-10-27'

        test_aud = df_signal[df_signal['real_date'] == arbitrary_date_one]
        test_aud = test_aud[test_aud['cid'] == 'AUD']['value']

        test_usd = df_signal[df_signal['real_date'] == arbitrary_date_two]
        test_usd = test_usd[test_usd['cid'] == 'USD']['value']

        lagged_df = srr.df
        aud_lagged = lagged_df.loc['AUD', signal]['2010-01-08']
        self.assertTrue(round(float(test_aud), 5) == round(aud_lagged, 5))

        usd_lagged = lagged_df.loc['USD', signal]['2020-10-28']
        self.assertTrue(round(float(test_usd), 5) == round(usd_lagged, 5))

        # In addition to the DataFrame returned by categories_df(), an instance of the
        # Class will hold two "tables" for each segmentation type.
        # Confirm the indices are the expected: cross-sections or years.
        test_index = list(srr.df_cs.index)[3:]
        self.assertTrue(sorted(self.cids) == sorted(test_index))

    def test_df_isolator(self):

        self.dataframe_generator()
        # Method used to confirm that the segmentation of the original DataFrame is
        # being applied correctly: either cross-sectional or yearly basis. Therefore, if
        # a specific "cs" is passed, will the DataFrame be reduced correctly ?

        signal = 'CRY'
        srr = SignalReturnRelations(self.dfd, sig=signal, ret='XR',
                                    freq='D', blacklist=self.blacklist)
        df = srr.df.dropna(how='any')

        # First, test cross-sectional basis.
        # Choose a "random" cross-section.

        df_cs = srr.df_isolator(df=df, cs='GBP', cs_type='cids')


if __name__ == "__main__":

    unittest.main()