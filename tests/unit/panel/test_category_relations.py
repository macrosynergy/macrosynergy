
import unittest
import io
import sys
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.category_relations import CategoryRelations

class TestAll(unittest.TestCase):

    def dataframe_generator(self):

        self.__dict__['cids'] = ['AUD', 'CAD', 'GBP', 'NZD', 'JPY', 'CHF']
        self.__dict__['xcats'] = ['XR', 'CRY', 'GROWTH', 'INFL']

        df_cids = pd.DataFrame(index=self.cids,
                               columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
        df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0.1, 1]
        df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
        df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
        df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]
        df_cids.loc['JPY'] = ['2002-01-01', '2020-09-30', -0.3, 3]
        df_cids.loc['CHF'] = ['2002-01-01', '2020-09-30', 0.3, 1]

        cols = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef']
        df_xcats = pd.DataFrame(index=self.xcats, columns=cols)
        df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
        df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
        df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
        df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

        dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.__dict__['dfd'] = dfd

        self.__dict__['black'] = {'AUD': ['2000-01-01', '2003-12-31'],
                                  'GBP': ['2018-01-01', '2100-01-01']}

        # Filter the DataFrame to test the Set Theory logic.
        # The category, Growth, will not be defined for the cross-section 'AUD'.
        # The purpose is to return the shared cross-sections across specific categories.
        # Therefore, if the category Growth is passed, 'AUD' will be excluded from the
        # returned list of cross-sections.
        # Same logic applies to 'INFL' and 'NZD'.
        filt1 = (dfd['xcat'] == 'GROWTH') & (
                    dfd['cid'] == 'AUD')
        filt2 = (dfd['xcat'] == 'INFL') & (dfd['cid'] == 'NZD')
        filt3 = (dfd['xcat'] == 'INFL') & (dfd['cid'] == 'JPY')
        # Reduced dataframe.

        self.__dict__['filter_tuple'] = (filt1 | filt2 | filt3)
        dfdx = dfd[~self.filter_tuple]

        self.__dict__['dfdx'] = dfdx

        # Define a List of the cross-sections that one is interested in modelling. The
        # dataframe might potentially be defined on a greater number of cross-sections.
        # The List, 'cidx', should be a subset or incorporate all cross-sections defined
        # in the dataframe.
        # The above stipulation on 'cidx' will require being validated in the defined
        # functionality.
        cidx = ['AUD', 'CAD', 'GBP']
        self.__dict__['cidx'] = cidx

    def test_constructor(self):

        self.dataframe_generator()

        # Testing the various assert statements built into the Class's Constructor.
        with self.assertRaises(AssertionError):
            # Test the restrictions placed on the frequency parameter.
            cr = CategoryRelations(self.dfdx, xcats=['GROWTH', 'INFL'], cids=self.cidx,
                                   freq='d', xcat_aggs=['mean', 'mean'], lag=1,
                                   start='2000-01-01', years=None, blacklist=self.black)

        with self.assertRaises(AssertionError):
            # Test the notion that the category List can only receive two categories.
            cr = CategoryRelations(self.dfdx, xcats=['GROWTH', 'INFL', 'XR'],
                                   cids=self.cidx, freq='M', xcat_aggs=['mean', 'mean'],
                                   lag=1, start='2000-01-01', years=None,
                                   blacklist=self.black)

    def test_intersection_cids(self):

        self.dataframe_generator()

        self.__dict__['cidx'] = ['AUD', 'CAD', 'GBP', 'USD', 'CHF']

        # The StringIO module is an in-memory file-like Object.
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        # Isolate the cross-sections available for both the corresponding categories.
        shared_cids = CategoryRelations.intersection_cids(self.dfdx, ['GROWTH', 'INFL'],
                                                          self.cidx)
        sys.stdout = sys.__stdout__
        capturedOutput.seek(0)

        print_statements = capturedOutput.read().strip('\n').split('.')
        print_statements = print_statements[:-1]

        # Split on the full stops, so subsequently removed from the Strings displayed in
        # the console if the conditions are satisfied.
        self.assertTrue(print_statements[0] == "GROWTH misses: ['AUD', 'USD']")
        # Account for the space in the console separating both print statements.
        self.assertTrue(print_statements[1][1:] == "INFL misses: ['USD']")

        # Aim to test the returned list of cross-sections.
        # Broaden the testcase to further test the accuracy.
        self.__dict__['cidx'] = ['AUD', 'CAD', 'GBP', 'USD', 'EUR', 'JPY', 'NZD', 'CHF']
        # Print statements will be returned to the console.
        shared_cids = CategoryRelations.intersection_cids(self.dfdx, ['GROWTH', 'INFL'],
                                                          self.cidx)

        self.assertTrue(sorted(shared_cids) == ['CAD', 'CHF', 'GBP'])


if __name__ == "__main__":

    unittest.main()