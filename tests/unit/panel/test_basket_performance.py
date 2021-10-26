
import unittest
import numpy as np
import pandas as pd
import random
import sys

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.panel.basket_performance import *
from macrosynergy.management.shape_dfs import reduce_df_by_ticker
from macrosynergy.panel.historic_vol import flat_std

class TestAll(unittest.TestCase):

    def dataframe_construction(self):
        cids = ['AUD', 'GBP', 'NZD', 'USD']
        xcats = ['FX_XR', 'FX_CRY', 'EQ_XR', 'EQ_CRY']

        df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                    'sd_mult'])

        df_cids.loc['AUD'] = ['2010-12-01', '2020-12-31', 0, 1]
        df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
        df_cids.loc['NZD'] = ['2012-01-01', '2020-11-30', 0, 3]
        df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', 0, 4]
    
        df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                      'sd_mult', 'ar_coef', 'back_coef'])
        df_xcats.loc['FX_XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
        df_xcats.loc['FX_CRY'] = ['2011-01-01', '2020-10-30', 1, 1, 0.9, 0.5]
        df_xcats.loc['EQ_XR'] = ['2011-01-01', '2020-10-30', 0.5, 2, 0, 0.2]
        df_xcats.loc['EQ_CRY'] = ['2013-01-01', '2020-10-30', 1, 1, 0.9, 0.5]

        random.seed(2)
        self.dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
        self.black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01',
                                                                   '2100-01-01']}

        self.contracts = ['AUD_FX', 'NZD_FX', 'GBP_EQ', 'USD_EQ']
        ret = 'XR'
        cry = 'CRY'
        ticks_cry = [c + "_" + cry for c in self.contracts]
        
        ticks_ret = [c + "_" + ret for c in self.contracts]
        tickers = ticks_ret + ticks_cry
        dfx = reduce_df_by_ticker(self.dfd, blacklist = self.black)

        dfx["ticker"] = dfx["cid"] + "_" + dfx["xcat"]
        self.dfw_ret = dfx[dfx["ticker"].isin(ticks_ret)].pivot(index="real_date",
                                                                columns="cid", values=
                                                                "value")

        self.dfw_cry = dfx[dfx["ticker"].isin(ticks_cry)].pivot(index="real_date",
                                                                columns="cid", values=
                                                                "value")

    @staticmethod
    def construct_df():
        
        weights = [random.random() for i in range(65)]
        weights = np.array(weights)
        weights = weights.reshape((13, 5))

        weights[0:4, 0] = np.nan
        weights[-3:, 1] = np.nan
        weights[-6:, 2] = np.nan
        weights[-2:, -1] = np.nan
        weights[:3, -1] = np.nan

        sum_ = np.nansum(weights, axis=1)
        sum_ = sum_[:, np.newaxis]
        
        weights = np.divide(weights, sum_)
        cols = ['col_' + str(i + 1) for i in range(weights.shape[1])]
        pseudo_df = pd.DataFrame(data=weights, columns=cols)

        return pseudo_df

    @staticmethod
    def remove_rows(df):

        df_index = np.array(df.index)
        weights_bool = ~df.isnull()
        weights_bool = weights_bool.astype(dtype=np.uint8)
        
        act_cross = weights_bool.sum(axis=1)

        nan_rows = np.where(act_cross == 0)[0]
        nan_size = nan_rows.size

        w_matrix = df.to_numpy()
        if not nan_size == 0:
            start = nan_rows[0]
            end = nan_rows[-1]
            iterator = np.array(range(start, end + 1))

            bool_size = np.all(iterator.size == nan_size)
            if bool_size and np.all(iterator == nan_rows):
                w_matrix = w_matrix[(end + 1):, :]
                df_index = df_index[(end + 1):]
            else:
                w_matrix = np.delete(w_matrix, tuple(nan_rows), axis=0)
                df_index = np.delete(df_index, tuple(nan_rows), axis=0)

        df = pd.DataFrame(data=w_matrix, columns=list(df.columns), index=df_index)
        
        return df

    @staticmethod
    def weight_check(df, max_weight):

        weights_bool = ~df.isnull()
        weights_bool = weights_bool.astype(dtype=np.uint8)
        
        act_cross = weights_bool.sum(axis=1)
        uniform = 1 / act_cross

        weights_uni = weights_bool.multiply(uniform, axis=0)
        uni_bool = uniform > max_weight
        weights_uni[weights_uni == 0.0] = np.nan
        
        return weights_uni, uni_bool

    def test_check_weights(self):
        
        weights = self.construct_df()
        # Weight allocation exceeds 1.0: verify that the function catches
        # the constructed error.
        weights.iloc[0, :] += 0.5
        with self.assertRaises(AssertionError):
            check_weights(weights)

    def test_max_weight(self):

        ## Test on a randomly generated set of weights.
        max_weight = 0.3

        pseudo_df = self.construct_df()
        weights_uni, uni_bool = self.weight_check(pseudo_df, max_weight)

        weights = max_weight_func(pseudo_df, max_weight)
        weights = weights.to_numpy()
        weights_uni = weights_uni.to_numpy()

        weights = np.nan_to_num(weights)
        weights_uni = np.nan_to_num(weights_uni)
        for i, row in enumerate(weights):
            if uni_bool[i]:
                self.assertTrue(np.all(row == weights_uni[i, :]))
            else:
                self.assertTrue(np.all(row < max_weight + 0.001))
        
        self.dataframe_construction()
        dfw_ret = self.dfw_ret

        # After the application of the inverse standard deviation weighting method,
        # the preceding rows up until the window has been populated will become obsolete.
        ## Therefore, the rows should be removed.
        weights = inverse_weight(dfw_ret, "xma")
        weights = self.remove_rows(weights)
        
        weights = max_weight_func(weights, max_weight)

        weights_uni, uni_bool = self.weight_check(weights, max_weight)
        weights = weights.to_numpy()
        weights_uni = weights_uni.to_numpy()

        ## Unable to compare on NaNs.
        weights = np.nan_to_num(weights)
        weights_uni = np.nan_to_num(weights_uni)
        for i, row in enumerate(weights):
            if uni_bool[i]:
                self.assertTrue(np.all(row == weights_uni[i, :]))
            else:
                self.assertTrue(np.all(row < max_weight + 0.001))

    def test_equal_weight(self):

        self.dataframe_construction()
        dfw_ret = self.dfw_ret
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
            unique_vals = set(row)
            length = len(unique_vals) 
            self.assertTrue(length <= 2)
            if length == 1:
                self.assertTrue(unique_vals.pop() == equal[i])
            else:
                list_ = [0.0, equal[i]]
                self.assertTrue(unique_vals.pop() in list_)
                self.assertTrue(unique_vals.pop() in list_)

            test = bool_arr[i, :] * equal[i]
            self.assertTrue(np.all(row == test))

    def test_equal_weight(self):

        # Pass in GDP figures of the respective cross-sections as weights.
        # ['AUD', 'GBP', 'NZD', 'USD']
        gdp_figures = [17, 41, 9, 215]

        self.dataframe_construction()
        dfw_ret = self.dfw_ret

        weights = fixed_weight(dfw_ret, gdp_figures)
        self.assertEqual(dfw_ret.shape, weights.shape)
        weights_arr = weights.to_numpy()

        check = np.ones(shape=weights_arr.shape[0], dtype=np.float32)
        check = np.abs(check - np.sum(weights_arr, axis=1))

        self.assertTrue(np.all(check < 0.00001))

        cols = weights.columns
        weights[cols] = weights[cols].replace({'0': np.nan, 0.0: np.nan})
        weights_full = weights.dropna(axis=0, how='any')

        weights_full = weights_full.reset_index(drop=True)
        ratio_sum = sum(gdp_figures)
        ratio = [round(elem / ratio_sum, 5) for elem in gdp_figures]

        rows = weights_full.shape[0]
        for i in range(rows):
            row = list(weights_full.iloc[i, :].to_numpy())
            row = [round(elem, 5) for elem in row]
            self.assertEqual(row, ratio)

    def test_inverse_weight(self):

        self.dataframe_construction()
        dfw_ret = self.dfw_ret

        weights = inverse_weight(dfw_ret, "ma")
        weights = self.remove_rows(weights)

        sum_ = np.sum(weights, axis = 1)
        self.assertTrue(np.all(np.abs(sum_ - np.ones(sum_.size)) < 0.000001))
        weights_arr = np.nan_to_num(weights.to_numpy())

        # Validate that the inverse weighting mechanism has been applied correctly.
        dfwa = dfw_ret.rolling(window=21).agg(flat_std, True)
        dfwa = self.remove_rows(dfwa)
        
        dfwa *= np.sqrt(252)
        rolling_std = np.nan_to_num(dfwa.to_numpy())

        self.assertEqual(rolling_std.shape, weights_arr.shape)

        max_float = sys.float_info.max
        rolling_std[rolling_std == 0.0] = max_float
        for i, row in enumerate(rolling_std):

            dict_ = dict(zip(row, range(row.size)))
            sorted_r = sorted(row)
            s_indices = [dict_[elem] for elem in sorted_r]

            row_weight = weights_arr[i, :]
            reverse_order = []
            for index in s_indices:
                reverse_order.append(row_weight[index])

            self.assertTrue(reverse_order == sorted(row_weight, reverse=True))

    def test_b_performance(self):

        self.dataframe_construction()
        dfd = self.dfd

        # Testing the assertion error on the return field.
        with self.assertRaises(AssertionError):
            df_return = basket_performance(dfd, contracts=['AUD_FX', 'NZD_FX'],
                                           ret=["XR_NSA"], cry="CRY_NSA",
                                           weight_meth="equal", weight_xcat=None,
                                           max_weight=0.3, basket_tik="GLB_ALL",
                                           return_weights=False)
        # Testing the assertion error on the contracts field: List required..
        with self.assertRaises(AssertionError):
            df_return = basket_performance(dfd, contracts='AUD_FX',
                                           ret="XR_NSA", cry="CRY_NSA",
                                           weight_meth="equal", weight_xcat=None,
                                           max_weight=0.45, basket_tik="GLB_ALL",
                                           return_weights=False)
        # Testing the assertion error on max_weight field: 0 < max_weight <= 1.
        with self.assertRaises(AssertionError):
            df_return = basket_performance(dfd, contracts=['AUD_FX', 'NZD_FX'],
                                           ret="XR_NSA", cry="CRY_NSA",
                                           weight_meth="equal", weight_xcat=None,
                                           max_weight=1.2, basket_tik="GLB_ALL",
                                           return_weights=False)


if __name__ == "__main__":

    unittest.main()
