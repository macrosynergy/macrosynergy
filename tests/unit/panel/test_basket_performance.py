
import unittest
import numpy as np
import pandas as pd
import random
import sys
from tests.simulate import make_qdf, dataframe_basket, construct_df
from macrosynergy.panel.basket_performance import *
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.historic_vol import flat_std

contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']

class TestAll(unittest.TestCase):

    # Internal method used to return a dataframe with the uniform weights and a bool
    # array indicating which rows the procedure is necessary for.
    @staticmethod
    def weight_check(df, max_weight):

        weights_bool = ~df.isnull()
        act_cross = weights_bool.sum(axis=1)
        uniform = 1 / act_cross

        weights_uni = weights_bool.multiply(uniform, axis=0)
        uni_bool = uniform > max_weight
        weights_uni[weights_uni == 0.0] = np.nan
        
        return weights_uni, uni_bool

    # Actual tests

    def test_check_weights(self):
        weights = construct_df()
        # Weight allocation exceeds 1.0: verify that the function catches
        # the constructed error.
        weights.iloc[0, :] += 0.5
        with self.assertRaises(AssertionError):
            check_weights(weights)

    def test_equal_weight(self):

        dfw_ret, dfw_cry, dfd_test = dataframe_basket()
        dfw_bool = (~dfw_ret.isnull())
        dfw_bool = dfw_bool.astype(dtype=np.uint8)
        bool_arr = dfw_bool.to_numpy()
        act_cross = dfw_bool.sum(axis=1).to_numpy()
        equal = 1 / act_cross

        weights = equal_weight(dfw_ret)
        
        self.assertEqual(dfw_ret.shape, weights.shape)
        self.assertEqual(list(dfw_ret.index), list(weights.index))

        weight_arr = weights.to_numpy()
        for i, row in enumerate(weight_arr):
            # Check the index of each value: weight allocated to non-NaN cross-sections.
            test = bool_arr[i, :] * equal[i]
            self.assertTrue(np.all(row == test))

    def test_fixed_weight(self):

        # Pass in GDP figures of the respective cross-sections as weights.
        # contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']
        gdp = [17, 17, 41, 9, 215]
        dfw_ret, dfw_cry, dfd_test = dataframe_basket()
        weights = fixed_weight(dfw_ret, gdp)

        self.assertEqual(dfw_ret.shape, weights.shape)

        for i in range(weights.shape[0]):
            ar_row = weights.iloc[i, :].to_numpy()
            ar_gdp = np.asarray(gdp)
            x1 = np.sum(ar_gdp[ar_row > 0])  # sum non-zero base values
            x2 = np.sum(ar_row * np.sum(ar_gdp[ar_row > 0]))  # sum weighted base values
            self.assertTrue(x1 == x2)

    def test_inverse_weight(self):

        dfw_ret, dfw_cry, dfd_test = dataframe_basket()

        weights = inverse_weight(dfw_ret, "ma")
        fvi = weights.first_valid_index()
        weights = weights[fvi:]
        sum_ = np.sum(weights, axis=1)

        # Check if weights add up to one
        self.assertTrue(np.all(np.abs(sum_ - np.ones(sum_.size)) < 0.000001))
        weights_arr = np.nan_to_num(weights.to_numpy())

        # Validate that the inverse weighting mechanism has been applied correctly.
        dfwa = dfw_ret.rolling(window=21).agg(flat_std, True)
        fvi = dfwa.first_valid_index()
        dfwa = dfwa[fvi:]
        self.assertEqual(dfwa.shape, weights_arr.shape)
        
        dfwa *= np.sqrt(252)
        rolling_std = np.nan_to_num(dfwa.to_numpy())

        # Used to account for NaNs (zeros) which disrupt the numerical ordering.
        max_float = sys.float_info.max
        rolling_std[rolling_std == 0.0] = max_float
        for i, row in enumerate(rolling_std):

            # Maps the value to the index.
            dict_ = dict(zip(row, range(row.size)))
            sorted_r = sorted(row)
            # Sorted indices. For instance, [14, 12, 19, 7] -> [7, 12, 14, 19].
            # Therefore, s_indices = [3, 1, 0, 2]
            # After the inverse method is applied the ascending order index would be:
            # [1/14, 1/12, 1/19, 1/7] -> [2, 0, 1, 3]. Reversed order.
            s_indices = [dict_[elem] for elem in sorted_r]

            row_weight = weights_arr[i, :]
            reverse_order = []
            # Access the row_weights in the order prescribed above. Given the inverse
            # operation, it should equate the descending order (largest first).
            for index in s_indices:
                reverse_order.append(row_weight[index])

            # Validate the assertion.
            self.assertTrue(reverse_order == sorted(row_weight, reverse=True))

    def test_values_weights(self):
        dfw_ret, dfw_cry, dfd_test = dataframe_basket()

        # Construct an additional dataframe on a different category to delimit weights.
        # Comparable to the design of basket_performance.py file which isolates a
        # category, from the dataframe produced by make_qdf(), and uses the now truncated
        # dataframe to form the weights. Will achieve the matching dimensions through a
        # pivot operation.
        cids_ = ['AUD', 'GBP', 'NZD', 'USD']

        xcats_ = ['FXWBASE_NSA', 'EQWBASE_NSA']

        df_cids_ = pd.DataFrame(index=cids_, columns=['earliest', 'latest', 'mean_add',
                                                      'sd_mult'])

        df_cids_.loc['AUD'] = ['2010-12-01', '2020-12-31', 0, 5]
        df_cids_.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 3]
        df_cids_.loc['NZD'] = ['2012-01-01', '2020-11-30', 0, 2]
        df_cids_.loc['USD'] = ['2013-01-01', '2020-09-30', 0, 2]

        df_xcats_ = pd.DataFrame(index=xcats_, columns=['earliest', 'latest', 'mean_add',
                                                       'sd_mult', 'ar_coef',
                                                       'back_coef'])

        df_xcats_.loc['FXWBASE_NSA'] = ['2010-01-01', '2020-12-31', 100, 1, 0.9, 0.5]
        df_xcats_.loc['EQWBASE_NSA'] = ['2010-01-01', '2020-12-31', 100, 1, 0.9, 0.5]

        random.seed(1)
        dfd_ = make_qdf(df_cids_, df_xcats_, back_ar=0.75)
        black_ = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01',
                                                               '2100-01-01']}

        wgt = 'WBASE_NSA'
        contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']
        ticks_wgt = [c + wgt for c in contracts]
        tickers = ticks_wgt

        dfx = reduce_df_by_ticker(dfd_, ticks=tickers, blacklist=black_)
        w_df = dfx.pivot(index="real_date", columns="ticker", values="value")

        # Test the assert statement on the received weight category.
        # Test the assertion of a String.
        with self.assertRaises(AssertionError):
            df_return = basket_performance(dfd_test, contracts=contracts,
                                           ret="XR_NSA",
                                           cry=None, blacklist=black_,
                                           weight_meth="values",
                                           wgt=[0.21, 0.19, 0.27, 0.33],
                                           max_weight=0.3, basket_tik="GLB_ALL",
                                           return_weights=False)

        weights = values_weight(dfw_ret, w_df, weight_meth="values")
        self.assertTrue(weights.shape == dfw_ret.shape)
        weights_inv = values_weight(dfw_ret, w_df, weight_meth="inv_values")
        self.assertTrue(weights_inv.shape == dfw_ret.shape)

        # Todo: check if weights correspond to original values
        bool_df = ~dfw_ret.isnull()
        bool_arr = bool_df.to_numpy()

        weights_df = w_df.multiply(bool_arr)
        weights_arr = weights_df.to_numpy()
        sum_row = np.nansum(weights_arr, axis=1)
        sum_row = sum_row[:, np.newaxis]

        weights_test = weights.to_numpy() * sum_row
        for i in range(weights_test.shape[0]):
            compare_1 = np.nan_to_num(weights_test[i, :])
            compare_2 = np.nan_to_num(weights_arr[i, :])
            assert (np.all(compare_1 - compare_2)) < 0.000001

    def test_max_weight(self):

        # Test on a randomly generated set of weights (pseudo-DataFrame).
        max_weight = 0.3

        pseudo_df = construct_df()
        weights_uni, uni_bool = self.weight_check(pseudo_df, max_weight)

        weights = max_weight_func(pseudo_df, max_weight)
        weights = weights.to_numpy()
        weights_uni = weights_uni.to_numpy()

        weights = np.nan_to_num(weights)
        weights_uni = np.nan_to_num(weights_uni)
        # Check whether the weights are evenly distributed or all are within the
        # upper-bound.
        for i, row in enumerate(weights):
            if uni_bool[i]:
                self.assertTrue(np.all(row == weights_uni[i, :]))
            else:
                # Accounts for floating point precession.
                self.assertTrue(np.all((row - (max_weight + 0.001)) < 0.00001))

        # Test on large DataFrame.
        dfw_ret, dfw_cry, dfd_test = dataframe_basket()

        # After the application of the inverse standard deviation weighting method,
        # the preceding rows up until the window has been populated will become obsolete.
        # Therefore, the rows should be removed.
        weights = inverse_weight(dfw_ret, "xma")
        fvi = weights.first_valid_index()
        weights = weights[fvi:]

        weights = max_weight_func(weights, max_weight)

        weights_uni, uni_bool = self.weight_check(weights, max_weight)
        weights = weights.to_numpy()
        weights_uni = weights_uni.to_numpy()

        # Unable to compare on NaNs. Convert to zeros.
        weights = np.nan_to_num(weights)
        weights_uni = np.nan_to_num(weights_uni)
        for i, row in enumerate(weights):
            if uni_bool[i]:
                self.assertTrue(np.all(row == weights_uni[i, :]))
            else:
                self.assertTrue(np.all((row - (max_weight + 0.001)) < 0.00001))

    def test_b_performance(self):

        dfw_ret, dfw_cry, dfd_test = dataframe_basket()
        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}

        c = contracts
        # Testing the assertion error on the return field. Expects a String.
        with self.assertRaises(AssertionError):
            df_return = basket_performance(dfd_test, contracts=['AUD_FX', 'NZD_FX'],
                                           ret=["XR_NSA"], cry="CRY_NSA",
                                           weight_meth="equal", wgt=None,
                                           max_weight=0.3, basket_tik="GLB_ALL",
                                           return_weights=False)
        # Testing the assertion error on the contracts field: List required.
        with self.assertRaises(AssertionError):
            df_return = basket_performance(dfd_test, contracts='AUD_FX',
                                           ret="XR_NSA", cry="CRY_NSA",
                                           weight_meth="equal", wgt=None,
                                           max_weight=0.45, basket_tik="GLB_ALL",
                                           return_weights=False)
        # Testing the assertion error on max_weight field: 0 < max_weight <= 1.
        with self.assertRaises(AssertionError):
            df_return = basket_performance(dfd_test, contracts=c, ret="XR_NSA",
                                           cry="CRY_NSA", weight_meth="equal", wgt=None,
                                           max_weight=1.2, basket_tik="GLB_ALL",
                                           return_weights=False)
        # Testing the weighting method "fixed": number of weights must be equal to the
        # number of contract.
        with self.assertRaises(AssertionError):
            gdp_figures = [17.0, 41.0, 9.0, 215.0, 23.0, 26.0]
            df_return = basket_performance(dfd_test, contracts=c, ret="XR_NSA",
                                           cry="CRY_NSA", weight_meth="fixed", wgt=None,
                                           weights=gdp_figures, max_weight=0.4,
                                           basket_tik="GLB_ALL", return_weights=False)

        df_return = basket_performance(dfd_test, contracts=c, ret="XR_NSA", cry=None,
                                       blacklist=black, weight_meth="equal",
                                       max_weight=1.0, basket_tik="GLB_ALL",
                                       return_weights=False)

        weights = equal_weight(dfw_ret)
        b_return = dfw_ret.multiply(weights).sum(axis=1).to_numpy()
        value = np.squeeze(df_return[['value']].to_numpy(), axis=1)

        # Tests the dimensions. The returned dataframe should have the same number of
        # rows given the basket return sums all contracts on each timestamp.
        # Only applicable if the parameter "return_weights" is set to zero.
        self.assertTrue(df_return.shape[0] == dfw_ret.shape[0])

        # Validate the basket's ticker name is GLB_ALL + "ret" if the parameter
        # "basket_tik" is not defined.
        self.assertTrue(np.all(df_return["ticker"] == "GLB_ALL" + "_" + "XR_NSA"))

        # Accounts for floating point precision.
        self.assertTrue(np.all(np.abs(b_return - value) < 0.000001))

        # The below code would require changes is weight_meth = "invsd" given the
        # removal of rows applied.
        df_return = basket_performance(dfd_test, contracts=c, ret="XR_NSA", cry=None,
                                       blacklist=None, weight_meth="equal",
                                       max_weight=0.3, basket_tik="GLB_ALL",
                                       return_weights=True)

        # Test the Ticker name.
        ticker = np.squeeze(df_return[['ticker']].to_numpy(), axis=1)
        weight_ticker = ticker[dfw_ret.shape[0]:]
        self.assertEqual(len(set(weight_ticker)), dfw_ret.shape[1])
        self.assertTrue(all([tick[-3:] == "WGT" for tick in weight_ticker]))

        # Test the concat function.
        last_return_index = dfw_ret.shape[0]
        date_column = df_return[['real_date']]
        first_date = date_column.iloc[0].values
        concat_date = date_column.iloc[last_return_index].values
        self.assertEqual(first_date, concat_date)

        # Test that all added weights are <= 1
        weight_column = df_return[['value']]
        weight_column = weight_column.iloc[last_return_index:].to_numpy()
        self.assertTrue(np.all(weight_column <= 1.0))


if __name__ == "__main__":

    unittest.main()
