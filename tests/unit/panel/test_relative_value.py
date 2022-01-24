
import unittest
import numpy as np
import pandas as pd
from tests.simulate import make_qdf
from macrosynergy.panel.make_relative_value import make_relative_value
from macrosynergy.management.shape_dfs import reduce_df
from random import randint, choice
import io
import sys

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

    def test_relative_value_dimensionality(self):

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
        dfd_1 = make_relative_value(dfd, xcats=['GROWTH', 'INFL'], cids=['AUD'],
                                    blacklist=None, basket=['AUD'],
                                    rel_meth='subtract', rel_xcats=None, postfix='RV')

        self.assertTrue(dfd_1.empty)

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
        # truncated from the dataframe given understanding the relative value of a single
        # realised return is meaningless.
        # The difference between the dimensions of the input dataframe and the returned
        # dataframe should correspond to the number of indices with only a single value.

        xcats = self.xcats[0]
        cids = self.cids
        # Ensures for a period of time only a single cross-section is defined.
        start = '2000-01-01'
        end = '2020-09-30'
        dfx = reduce_df(df=self.dfd, xcats=[xcats], cids=cids, start=start,
                        end=end, blacklist=None, out_all=False)
        input_rows = dfx.shape[0]
        dfw = dfx.pivot(index='real_date', columns='cid', values='value')

        data = dfw.to_numpy()
        data = data.astype(dtype=np.float64)
        active_cross = np.sum(~np.isnan(data), axis=1)

        single_value = np.where(active_cross == 1)[0]
        no_single_values = single_value.size

        # Apply the function to understand if the logic above holds.
        dfd_rl = make_relative_value(self.dfd, xcats=xcats, cids=cids, start=start,
                                     end=end, blacklist=None, basket=None,
                                     rel_meth='divide', rel_xcats=None, postfix='RV')
        output_rows = dfd_rl.shape[0]

        self.assertTrue(output_rows == (input_rows - no_single_values))

        # Test "complete_cross" parameter.

        # Construct a dataframe containing two categories but one of the categories is
        # defined over fewer cross-sections. To be precise, the cross-sections present
        # for the aforementioned category will be a subset of the cross-sections
        # available for the secondary category. Further, the basket will be set to the
        # union of cross-sections.
        dfd = self.dfd
        xcats = ['XR', 'CRY']
        cids = self.cids
        start = '2000-01-01'
        end = '2020-12-31'
        dfx = reduce_df(df=dfd, xcats=xcats, cids=cids, start=start,
                        end=end, blacklist=None, out_all=False)

        # On the reduced dataframe, remove a single cross-section from one of the
        # categories.
        filt1 = ~((dfx['cid'] == 'AUD') & (dfx['xcat'] == 'XR'))
        dfdx = dfx[filt1]

        # Pass in the filtered dataframe, and test whether the correct print statement
        # appears in the console.
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        dfd_rl = make_relative_value(dfdx, xcats=xcats, cids=cids, start=start,
                                     end=end, blacklist=None, basket=None,
                                     complete_cross = False, rel_meth='subtract',
                                     rel_xcats=None, postfix='RV')
        sys.stdout = sys.__stdout__
        capturedOutput.seek(0)
        print_statement = capturedOutput.read()[-23:-2]
        test = "['CAD', 'GBP', 'NZD']"
        self.assertTrue(print_statement == test)

        # If the "complete_cross" parameter is set to True, the corresponding category
        # defined over an incomplete set of cross-sections will be removed from the
        # output dataframe.
        dfd_rl = make_relative_value(dfdx, xcats=xcats, cids=cids, start=start,
                                     end=end, blacklist=None, basket=None,
                                     complete_cross = True, rel_meth='subtract',
                                     rel_xcats=None, postfix='RV')
        # Assert the dataframe only contains a single category: the category with a
        # complete set of cross-sections.
        self.assertTrue(dfd_rl['xcat'].unique()[0] == 'CRYRV')

    def test_relative_value_logic(self):

        self.dataframe_generator()
        dfd = self.dfd

        # Aim to test the application of the actual relative_value method: subtract or
        # divide.
        # If the basket contains a single cross-section, the relative value is simply the
        # realised return of the respective cross-section. Therefore, the cross-section
        # chosen will consequently have a zero value for each output if the logic is
        # correct.
        basket_cid = ['AUD']
        dfd_2 = make_relative_value(dfd, xcats=['INFL'], cids=self.cids,
                                    blacklist=None, basket=basket_cid,
                                    rel_meth='subtract', rel_xcats=None, postfix='RV')

        basket_df = dfd_2[dfd_2['cid'] == basket_cid[0]]
        values = basket_df['value'].to_numpy()
        self.assertTrue((np.sum(values) - 0.0) < 0.00001)

        # Test the logic of the function if there are multiple cross-sections defined in
        # basket. First, test the relative value using subtraction and secondly test
        # relative value using division.

        # Incorporate three cross-sections for the basket.
        basket_cid = ['AUD', 'CAD', 'GBP']
        xcats = choice(self.xcats)
        start = '2001-01-01'
        end = '2020-10-30'
        dfx = reduce_df(df=self.dfd, xcats=[xcats], cids=self.cids, start=start,
                        end=end, blacklist=None, out_all=False)

        dfd_3 = make_relative_value(dfx, xcats=xcats, cids=self.cids,
                                    blacklist=self.blacklist, basket=basket_cid,
                                    rel_meth='subtract', rel_xcats=None, postfix='RV')
        # Isolate an arbitrarily chosen date and test the logic
        dfw = dfx.pivot(index='real_date', columns='cid', values='value')
        index = dfw.index
        no_rows = index.size
        range_ = (int(no_rows * 0.25), int(no_rows * 0.75))

        random_index = randint(*range_)
        date = index[random_index]

        random_row = dfw.iloc[random_index, :]
        random_row_dict = random_row.to_dict()
        values = [v for k, v in random_row_dict.items() if k in basket_cid]
        manual_mean = sum(values) / len(values)

        computed_values = (random_row - manual_mean)
        computed_values = computed_values.to_numpy()

        dfd_3_pivot = dfd_3.pivot(index='real_date', columns='cid', values='value')
        output_index = dfd_3_pivot.index
        index_val = np.where(output_index == date)[0]

        function_output = dfd_3_pivot.iloc[index_val, :]
        function_output = function_output.to_numpy()

        self.assertTrue(np.all(computed_values == function_output[0]))

        # Test the division.
        # Computing make_relative_value() on a single category that has been chosen
        # randomly.
        dfd_4 = make_relative_value(dfx, xcats=xcats, cids=self.cids,
                                    blacklist=self.blacklist, basket=basket_cid,
                                    rel_meth='divide', rel_xcats=None, postfix='RV')

        # Divide each cross-section's realised return by the mean of the basket.
        computed_values = (random_row / manual_mean)
        computed_values = computed_values.to_numpy()

        dfd_4_pivot = dfd_4.pivot(index='real_date', columns='cid', values='value')
        output_index = dfd_4_pivot.index
        index_val = np.where(output_index == date)[0]

        function_output = dfd_4_pivot.iloc[index_val, :]
        function_output = function_output.to_numpy()

        self.assertTrue(np.all(computed_values == function_output[0]))


if __name__ == '__main__':

    unittest.main()