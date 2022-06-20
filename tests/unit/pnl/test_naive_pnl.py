
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
        df_cids.loc['AUD', :] = ['2010-01-01', '2020-12-31', 0.5, 2]
        df_cids.loc['CAD', :] = ['2010-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP', :] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
        df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]
        df_cids.loc['USD'] = ['2001-01-01', '2020-12-31', 0.2, 2]
        df_cids.loc['EUR'] = ['2001-01-01', '2020-12-31', 0.1, 2]

        df_xcats = pd.DataFrame(index=self.xcats,
                                columns=['earliest', 'latest', 'mean_add',
                                         'sd_mult', 'ar_coef', 'back_coef'])

        df_xcats.loc['EQXR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
        df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
        df_xcats.loc['DUXR'] = ['2000-01-01', '2020-12-31', 0.1, 0.5, 0, 0.1]

        # Standard df for tests.
        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd


if __name__ == '__main__':

    unittest.main()


