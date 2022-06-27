

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

        df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
        df_xcats.loc['EQXR_NSA'] = ['2010-01-01', '2020-12-31', 0.5, 2, 0, 0.2]
        df_xcats.loc['SIG_NSA'] = ['2010-01-01', '2020-12-31', 0, 10, 0.4, 0.2]

        random.seed(2)
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

        self.__dict__['dfd'] = dfd

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}

        self.__dict__['blacklist'] = black

        assert 'dfd' in vars(self).keys(), "Instantiation of DataFrame missing from " \
                                           "field dictionary."


if __name__ == "__main__":

    unittest.main()