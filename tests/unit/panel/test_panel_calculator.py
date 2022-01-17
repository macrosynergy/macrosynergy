
import unittest
import numpy as np
import pandas as pd
from tests.simulate import make_qdf

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

        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}

        self.__dict__['blacklist'] = black

    def test_panel_calculator(self):

        self.dataframe_generator()

        
if __name__ == '__main__':

    unittest.main()