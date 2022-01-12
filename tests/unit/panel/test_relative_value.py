import unittest
import random
import numpy as np
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.panel.make_relative_value import make_relative_value
from macrosynergy.management.shape_dfs import reduce_df

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

    def test_relative_value(self):

        self.dataframe_generator()
        dfd = self.dfd

        with self.assertRaises(AssertionError):
            # Validate the assertion on the parameter "rel_meth".
            dfd_1 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=None,
                                        blacklist=None, rel_meth='subtraction',
                                        rel_xcats=None, postfix='RV')

        with self.assertRaises(AssertionError):
            # Validate the assertion on "xcats" parameter.
            dfd_2 = make_relative_value(dfd, xcats=('XR', 'GROWTH'), cids=None,
                                        blacklist=None, basket=['AUD', 'CAD', 'GBP'],
                                        rel_meth='subtract',
                                        rel_xcats=['XRvB3', 'GROWTHvB3', 'INFLvB3'])

        with self.assertRaises(AssertionError):
            # Validate the clause constructed around the basket parameter. The
            # cross-sections included in the "basket" must either be complete, inclusive
            # of all defined cross-sections, or a valid subset dependent on the cross-
            # sections passed into the function.
            dfd_3 = make_relative_value(dfd, xcats=['XR', 'GROWTH'], cids=['GBP'],
                                        blacklist=None, basket=['AUD', 'CAD'],
                                        rel_meth='divide',
                                        rel_xcats=['XRvB3', 'GROWTHvB3', 'INFLvB3'])

        # The first aspect of the code to validate is if the dataframe is reduced to a
        # single cross-section, and the basket is axiomatically limited to a single cross
        # section as well, the notion of computing the relative value is not appropriate.
        # Therefore, the returned dataframe, from the function, will be empty simply
        # because its functionality is not applicable.
        dfd_4 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=['AUD'],
                                    blacklist=None, basket=['AUD'],
                                    rel_meth='subtract', rel_xcats=None, postfix='RV')

        self.assertTrue(dfd_4.empty)

        # First part of the logic to validate is the stacking mechanism, and subsequent
        # dimensions of the returned dataframe. Once the reduction is accounted for, the
        # dimensions should reflect the returned input.
        xcats = self.xcats[:-2]
        cids = self.cids
        start = '2001-01-01'
        end = '2020-11-30'
        dfx = reduce_df(df=self.dfd, xcats=xcats, cids=cids, start=start, end=end,
                        blacklist=None, out_all=False)
        # To confirm the above statement, the parameter "basket" must be equated to None
        # to prevent any further reduction.
        # Further, for the dimensions of the input dataframe to match the output
        # dataframe, each date the dataframe is defined over must have greater than one
        # cross-section available for each index (date) otherwise rows with only a single
        # realised value will be removed.
        # Therefore, to achieve this, each date having at least two realised values,
        # set both the start date & end date parameters to the second earliest and
        # latest date respectively (of the defined cross-sections' realised series). Both
        # the categories are defined over the same time-period, so cross-sections will
        # delimit the dimensions.
        dfd_rl = make_relative_value(self.dfd, xcats=xcats, cids=cids, start=start,
                                     end=end, blacklist=None, basket=None,
                                     rel_meth='subtract', rel_xcats=None, postfix='RV')
        self.assertEqual(dfx.shape, dfd_rl.shape)

        # Test the proposal that any dates with only a single realised value will be
        # truncated from the dataframe.


if __name__ == '__main__':

    unittest.main()