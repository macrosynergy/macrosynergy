
import unittest
import numpy as np
import pandas as pd
import sys
from tests.simulate import dataframe_basket, construct_df
from macrosynergy.panel.basket import Basket
from macrosynergy.management.shape_dfs import reduce_df, reduce_df_by_ticker
from macrosynergy.panel.historic_vol import flat_std
from itertools import chain
import warnings


warnings.filterwarnings("ignore")


class TestAll(unittest.TestCase):

    def dataframe_generator(self):

        self.__dict__['ret'] = 'XR_NSA'
        self.__dict__['cry'] = ['CRY_NSA', 'CRR_NSA']
        black = {'AUD': ['2000-01-01', '2003-12-31'],
                 'GBP': ['2018-01-01', '2100-01-01']}
        self.__dict__['black'] = black
        dfw_ret, dfw_cry, dfd = dataframe_basket(ret = self.ret, cry = self.cry,
                                                 black=self.black)
        self.__dict__['dfw_ret'] = dfw_ret
        self.__dict__['dfw_cry'] = dfw_cry
        self.__dict__['dfd'] = dfd
        contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']
        self.__dict__['contracts'] = contracts

        # Instantiate an instance of the Class.
        basket = Basket(df=dfd, contracts=contracts, ret="XR_NSA",
                        cry=["CRY_NSA", "CRR_NSA"], blacklist=black)
        self.__dict__['basket'] = basket

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

    # Actual tests.
    # Validate that the method catches any computed weights, in the dataframe, which
    # exceed one.
    def test_check_weights(self):
        weights_arr = np.random.rand(15, 3)
        cols = ['col_' + str(i + 1) for i in range(3)]
        weights = pd.DataFrame(data=weights_arr, columns=cols)
        # Exceeds the permitted weight limit.
        weights.iloc[0, 1] += 1.01
        with self.assertRaises(AssertionError):
            Basket.check_weights(weights)

    def test_equal_weight(self):

        self.dataframe_generator()
        dfw_bool = (~self.dfw_ret.isnull())
        dfw_bool = dfw_bool.astype(dtype=np.uint8)
        bool_arr = dfw_bool.to_numpy()
        act_cross = dfw_bool.sum(axis=1).to_numpy()
        equal = 1 / act_cross

        weights = self.basket.equal_weight(self.dfw_ret)
        
        self.assertEqual(self.dfw_ret.shape, weights.shape)
        # Defined over the same time-period.
        self.assertEqual(list(self.dfw_ret.index), list(weights.index))

        weight_arr = weights.to_numpy()
        for i, row in enumerate(weight_arr):
            # Check the index of each value: weight allocated to non-NaN cross-sections.
            test = bool_arr[i, :] * equal[i]
            self.assertTrue(np.all(np.nan_to_num(row) == test))

    def test_fixed_weight(self):

        self.dataframe_generator()
        w = [6, 12, 7, 8, 13]
        weights = self.basket.fixed_weight(df_ret=self.dfw_ret, weights=w)

        self.assertEqual(self.dfw_ret.shape, weights.shape)

        for i in range(weights.shape[0]):
            nulls = self.dfw_ret.iloc[i, :].isnull()
            orig_values = np.asarray(w)
            orig_values[nulls] = 0
            test_weights = orig_values / np.sum(orig_values)
            class_weights = weights.iloc[i, :].to_numpy()
            sum_of_diffs = np.nansum(test_weights - class_weights)
            self.assertAlmostEqual(sum_of_diffs, 0, 4)

    def test_inverse_weight(self):

        self.dataframe_generator()
        weights = self.basket.inverse_weight(self.dfw_ret, "ma")
        fvi = weights.first_valid_index()
        weights = weights[fvi:]
        sum_ = np.sum(weights, axis=1)

        # Check if weights add up to one
        self.assertTrue(np.all(np.abs(sum_ - np.ones(sum_.size)) < 0.000001))
        weights_arr = np.nan_to_num(weights.to_numpy())

        # Validate that the inverse weighting mechanism has been applied correctly.
        dfwa = self.dfw_ret.rolling(window=21).agg(flat_std, True)
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

        self.dataframe_generator()
        # Construct an additional dataframe on a different category to delimit weights.
        # Comparable to the design of basket_performance.py file which isolates a
        # category, from the dataframe produced by make_qdf(), and uses the now truncated
        # dataframe to form the weights. Will achieve the matching dimensions through a
        # pivot operation.
        dfd = self.dfd
        dfw_ret = self.dfw_ret

        xcats_ = ['FXWBASE_NSA', 'EQWBASE_NSA']
        # Dataframe consisting exclusively of the external weight categories.
        dfd_weights = reduce_df(df=dfd, xcats=xcats_)

        wgt = 'WBASE_NSA'
        contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']
        ticks_wgt = [c + wgt for c in contracts]
        tickers = ticks_wgt
        black_ = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01',
                                                               '2100-01-01']}

        dfx = reduce_df_by_ticker(dfd_weights, ticks=tickers, blacklist=black_)
        w_df = dfx.pivot(index="real_date", columns="ticker", values="value")

        weights = self.basket.values_weight(dfw_ret, w_df, weight_meth="values")
        self.assertTrue(weights.shape == dfw_ret.shape)
        weights_inv = self.basket.values_weight(dfw_ret, w_df, weight_meth="inv_values")
        self.assertTrue(weights_inv.shape == dfw_ret.shape)

        # Validate that weights correspond to original values.
        bool_df = ~dfw_ret.isnull()
        bool_arr = bool_df.to_numpy()

        weights_df = w_df.multiply(bool_arr)
        weights_arr = weights_df.to_numpy()
        sum_row = np.nansum(weights_arr, axis=1)
        sum_row = sum_row[:, np.newaxis]

        # The values weighting method will simply take the external weight category's
        # "returns" and normalise the values to obtain valid weights. Therefore, complete
        # the inverse of the procedure to validate the weights are correct.
        weights_test = weights.to_numpy() * sum_row
        for i in range(weights_test.shape[0]):
            compare_1 = np.nan_to_num(weights_test[i, :])
            compare_2 = np.nan_to_num(weights_arr[i, :])
            condition = (np.all(compare_1 - compare_2)) < 0.000001
            # self.assertTrue(condition)

    def test_max_weight(self):

        self.dataframe_generator()
        # Test on a randomly generated set of weights (pseudo-DataFrame).
        max_weight = 0.3

        pseudo_df = construct_df()
        # Uniform weight dataframe and boolean dataframe indicating which uniform weight
        # values exceed the prescribed maximum weight. If the equal weight value is
        # larger than the maximum weight, all weight values are unable to be less than
        # the defined maximum value. Therefore, in such instance, set each weight value
        # to the equal weight.
        weights_uni, uni_bool = self.weight_check(pseudo_df, max_weight)

        weights = self.basket.max_weight_func(pseudo_df, max_weight)
        weights = weights.to_numpy()
        weights_uni = weights_uni.to_numpy()

        weights = np.nan_to_num(weights)
        weights_uni = np.nan_to_num(weights_uni)
        # Check whether the weights are evenly distributed or all are within the
        # upper-bound.
        for i, row in enumerate(weights):
            if uni_bool[i]:
                # If the maximum weight is less than the uniformly distributed weight,
                # all weights are set to the uniform value. Validate the logic.
                self.assertTrue(np.all(row == weights_uni[i, :]))
            else:
                # Accounts for floating point precession.
                self.assertTrue(np.all((row - (max_weight + 0.001)) < 0.00001))

        # Test on a larger DataFrame.
        dfw_ret, dfw_cry, dfd_test = dataframe_basket()

        # After the application of the inverse standard deviation weighting method,
        # the rows up until the window has been populated will become obsolete.
        # Therefore, the rows should be removed.
        weights = self.basket.inverse_weight(dfw_ret, "xma")
        fvi = weights.first_valid_index()
        weights = weights[fvi:]

        weights = self.basket.max_weight_func(weights, max_weight)

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

    def test_basket_constructor(self):

        # Test the operations associated with the Class's Constructor. Will implicitly
        # test the methods: store_attributes(), pivot_dataframe(), date_check().

        self.dataframe_generator()
        dfw_ret = self.dfw_ret
        dfw_cry = self.dfw_cry
        dfd_test = self.dfd

        c = self.contracts

        # First, test the assertions on the fields of the instance.
        # Testing the assertion error on the return field. Expects a String.
        with self.assertRaises(AssertionError):
            basket_1 = Basket(df=dfd_test, contracts=['AUD_FX', 'NZD_FX'],
                              ret=["XR_NSA"], cry="CRY_NSA", ewgts=None,
                              blacklist=self.black)
        # Testing the assertion error on the contracts field: List required. A basket
        # consisting of a single contract would be meaningless.
        with self.assertRaises(AssertionError):
            basket_1 = Basket(df=dfd_test, contracts='AUD_FX',
                              ret="XR_NSA", cry="CRY_NSA", ewgts=None,
                              blacklist=self.black)

        # Test the assertion that the external weight category parameter must receive a
        # string or List where the external weight is present in the dataframe. The below
        # testcase passes in a pd.DataFrame instead.
        with self.assertRaises(AssertionError):
            dfd_assert = self.dfd[self.dfd['xcat'] == 'FXWBASE_NSA']
            basket_1 = Basket(df=dfd_test, contracts=self.contracts,
                              ret="XR_NSA", cry="CRY_NSA",
                              ewgts=dfd_assert, blacklist=self.black)

        # Validate the tickers held in the field "self.tickers" equate to the expected
        # tickers. The concerned operation is:
        # self.tickers = self.ticks_ret + self.ticks_cry + self.ticks_wgt.
        # The below Unit Test will also test the method "self.store_attributes()" which
        # is used to aggregate the "self.tickers" field.
        basket_1 = Basket(df=dfd_test, contracts=self.contracts,
                          ret="XR_NSA", cry=["CRY_NSA", "CRR_NSA"],
                          ewgts="WBASE_NSA", blacklist=self.black)
        test_tickers = basket_1.tickers
        # The category is appended to the contract. Therefore, split the category and
        # contract to validate all the expected categories are present in the List for
        # each respective contract.
        test_categories = [t[6:] for t in test_tickers]
        unique_categories = sorted(list(set(test_categories)))
        expected_categories = ["XR_NSA", "CRY_NSA", "CRR_NSA", "WBASE_NSA"]
        self.assertTrue(unique_categories == sorted(expected_categories))

        # Test the aggregation mechanism of involving multiple carry categories in the
        # Basket Class.
        basket_2 = Basket(df=dfd_test, contracts=self.contracts,
                          ret="XR_NSA", cry=["CRY_NSA", "CRR_NSA"],
                          blacklist=self.black)

        test_carry_tickers = basket_2.ticks_cry
        carry_contract = lambda cat: [t + cat for t in self.contracts]
        check = list(map(carry_contract, ["CRY_NSA", "CRR_NSA"]))
        check = list(chain.from_iterable(check))
        self.assertTrue(test_carry_tickers == check)

    def test_make_basket(self):
        # The main driver method. Contains the majority of the logic controlling the
        # necessary workflow: the method that connects the user's request and the Class's
        # logic.
        # However, the computed DataFrames, basket performance or weight dataframe, will
        # be stored inside the respective dictionaries: "self.dict_retcry" &
        # "self.dict_wgs".

        self.dataframe_generator()
        basket_1 = Basket(df=self.dfd, contracts=self.contracts, ret="XR_NSA",
                          cry=["CRY_NSA", "CRR_NSA"], blacklist=self.black,
                          ewgts="WBASE_NSA")

        # Testing the assertion error on max_weight field: 0 < max_weight <= 1.
        with self.assertRaises(AssertionError):
            basket_1.make_basket(weight_meth="equal", max_weight=1.2,
                                 basket_name="GLB_EQUAL")

        # Testing the weighting method parameter. String expected.
        # Only able to receive a single weighting method. If required, call the
        # make_basket() method multiple times passing in different weight methods, and
        # their respective dataframes will be held on the instance's field:
        # "self.dict_wgs".
        with self.assertRaises(AssertionError):
            basket_1.make_basket(weight_meth=["equal", "invsd"], max_weight=1.2,
                                 basket_name="GLB_EQUAL")

        # Test the weight method is one of the five options.
        with self.assertRaises(AssertionError):
            basket_1.make_basket(weight_meth="inverse", max_weight=0.55,
                                 basket_name="GLB_EQUAL")

        # Test that the external weight category, held by the parameter "ewgt", has been
        # included in the instantiation of the instance (defined in the field "ewgts").
        # The external weight category is required if the chosen weight method is either
        # "values" or "inv_values".
        with self.assertRaises(AssertionError):
            basket_1.make_basket(weight_meth="inv_values", ewgt="FXWBASE_NSA",
                                 max_weight=0.55, remove_zeros=True,
                                 basket_name="GLB_INV_VALUES")

        # First, check the dictionaries on the instance are populated correctly.
        basket_2 = Basket(df=self.dfd, contracts=self.contracts, ret="XR_NSA",
                          cry=["CRY_NSA", "CRR_NSA"], blacklist=self.black,
                          ewgts="WBASE_NSA")
        basket_2.make_basket(weight_meth="equal", max_weight=0.6,
                             basket_name="GLB_EQUAL")

        baskets = basket_2.dict_retcry
        basket_names = list(baskets.keys())
        self.assertTrue(next(iter(basket_names)) == "GLB_EQUAL")

        # Secondly, test the dimensions of the concatenated dataframe.
        # The returned dataframe should have the same number of rows given the basket
        # return sums all contracts on each timestamp.
        no_rets = len(self.dfw_ret.index)
        no_cry = 0
        for carry in self.dfw_cry:
            assert isinstance(carry, pd.DataFrame)
            no_cry += len(carry.index)
        no_rows = no_rets + no_cry
        self.assertTrue(no_rows == basket_2.dict_retcry["GLB_EQUAL"].shape[0])

        basket_2.make_basket(weight_meth="invsd", lback_meth="ma", lback_periods=21,
                             max_weight=0.55, remove_zeros=True,
                             basket_name="GLB_INVERSE")
        weights = [1/6, 1/12, 1/12, 1/6, 1/2]
        basket_2.make_basket(weight_meth="fixed", weights=weights, max_weight=0.55,
                             basket_name="GLB_FIXED")
        basket_keys = basket_2.dict_retcry.keys()
        for b in basket_keys:
            self.assertTrue(b in ["GLB_EQUAL", "GLB_INVERSE", "GLB_FIXED"])
        weight_keys = basket_2.dict_wgs.keys()
        self.assertTrue(list(weight_keys) == list(basket_keys))

        # Confirm the computed basket calculations are as expected.
        # To test the basket performance, choose an arbitrary index and isolate the
        # corresponding rows in the weight dataframe and contract returns dataframe, and
        # subsequently complete the manual calculation.
        dfw_ret = self.dfw_ret
        basket_2 = Basket(df=self.dfd, contracts=self.contracts, ret="XR_NSA",
                          cry=["CRY_NSA", "CRR_NSA"], blacklist=self.black,
                          ewgts="WBASE_NSA")
        basket_2.make_basket(weight_meth="equal", max_weight=0.45,
                             basket_name="GLB_EQUAL")
        dfw_wgs = basket_2.dict_wgs["GLB_EQUAL"]
        basket_df = basket_2.dict_retcry["GLB_EQUAL"]
        basket_df_dates = basket_df['real_date'].to_numpy()

        random_index_date = dfw_ret.index[1017]
        index = np.where(basket_df_dates == random_index_date)
        index_xr = index[0][0]
        assert index_xr == 1017

        weight_row = dfw_wgs.loc[random_index_date]
        weight_row = weight_row.sort_index(axis=0)
        w_row_array = weight_row.to_numpy()

        return_row = dfw_ret.loc[random_index_date]
        return_row = return_row.sort_index(axis=0)

        manual_calculation = return_row.multiply(w_row_array).sum(axis=0)
        manual_value = round(manual_calculation, 5)
        # Test on the return category: 'XR_NSA'.
        basket_xr = basket_df[basket_df['ticker'] == 'GLB_EQUAL_XR_NSA']
        basket_value = round(basket_xr.iloc[index_xr]['value'], 5)
        self.assertTrue(manual_value == basket_value)

        # Complete the same logic for a carry category.
        dfw_cry = self.dfw_cry
        # List of dataframes. Extract the first dataframe corresponding to the category
        # CRY_NSA.
        assert isinstance(dfw_cry, list)

        dfw_cry = dfw_cry[0]
        carry_row = dfw_cry.loc[random_index_date].to_numpy()
        # Account for the different column names across the two dataframes.
        manual_value = round(np.sum(carry_row * weight_row.to_numpy()), 5)
        basket_cry = basket_df[basket_df['ticker'] == 'GLB_EQUAL_CRY_NSA']
        index_cry = np.where(basket_cry['real_date'] == random_index_date)[0][0]
        basket_value = round(basket_cry.iloc[index_cry]['value'], 5)
        self.assertTrue(manual_value == basket_value)

    def test_weight_visualiser(self):

        # Will exclusively test the assert statements in the method. The various
        # visualisation tools require certain parameters to be defined (dependency
        # between parameters.

        self.dataframe_generator()
        basket_1 = Basket(df=self.dfd, contracts=self.contracts, ret="XR_NSA",
                          cry=["CRY_NSA", "CRR_NSA"], blacklist=self.black,
                          ewgts=None)
        basket_1.make_basket(weight_meth="equal", max_weight=0.45,
                             basket_name="GLB_EQUAL")
        # Pivoted dataframe. The simplest test would be to chose the first dates outside
        # the defined time-period, index.
        equal_dfw = basket_1.dict_wgs["GLB_EQUAL"]
        date_index = equal_dfw.index

        f_date = date_index[0] - pd.Timedelta(days=1)
        l_date = date_index[-1] + pd.Timedelta(days=1)

        # The method allows the user to truncate the weight dataframe to analyse over a
        # specific period of time (highly volatile periods etc) by defining the
        # "start_date" & "end_date" parameters. However, the dates passed must be within
        # the time-period.
        with self.assertRaises(AssertionError):
            # Both the date parameters expect string objects.
            basket_1.weight_visualiser("GLB_EQUAL", start_date=str(f_date),
                                       end_date=str(l_date), all_tickers=True,
                                       percentage_change=False)

        # The method allows for two options: all the tickers are included in the display
        # graphic, or only a single ticker is involved. Therefore, if the "all_tickers"
        # parameter is set to False then the parameter "single_ticker" must be defined
        # with a ticker present in the weight dataframe.
        with self.assertRaises(AssertionError):
            basket_1.weight_visualiser("GLB_EQUAL", start_date=str(f_date),
                                       end_date=str(l_date), all_tickers=False,
                                       single_ticker=None,
                                       percentage_change=False)

        # Check the ticker passed is a ticker included in the contracts.
        with self.assertRaises(AssertionError):
            unavailable_ticker = 'GBP_FX'
            assert unavailable_ticker not in self.contracts
            basket_1.weight_visualiser("GLB_EQUAL", all_tickers=False,
                                       single_ticker=unavailable_ticker,
                                       percentage_change=False)

        # The optionality of percentage change, as one of the display graphics, is only
        # applied to single tickers. Therefore, if the parameter "percentage_change" is
        # set to True, "all_tickers" must be set to False and, axiomatically, the
        # "single_ticker" parameter must be defined to reduce the weight dataframe.
        with self.assertRaises(AssertionError):
            basket_1.weight_visualiser("GLB_EQUAL", all_tickers=True,
                                       percentage_change=True)

        with self.assertRaises(AssertionError):
            unavailable_ticker = 'GBP_FX'
            basket_1.weight_visualiser("GLB_EQUAL", all_tickers=False,
                                       single_ticker=unavailable_ticker,
                                       percentage_change=True)

    def test_return_basket(self):

        # Method used to return the computed dataframes.
        self.dataframe_generator()
        basket_1 = Basket(df=self.dfd, contracts=self.contracts, ret="XR_NSA",
                          cry=["CRY_NSA", "CRR_NSA"], blacklist=self.black,
                          ewgts=None)
        basket_1.make_basket(weight_meth="equal", max_weight=0.45,
                             basket_name="GLB_EQUAL")
        basket_1.make_basket(weight_meth="invsd", lback_meth="ma", lback_periods=21,
                             max_weight=0.55, remove_zeros=True,
                             basket_name="GLB_INVERSE")
        weights = [1/6, 1/12, 1/12, 1/6, 1/2]
        basket_1.make_basket(weight_meth="fixed", weights=weights, max_weight=0.55,
                             basket_name="GLB_FIXED")

        # Test the feature that if a basket_name is not specified by the user, default
        # is equal to None, then all of the computed baskets will be returned. In this
        # instance that would involve: "GLB_EQUAL", "GLB_INVERSE" & "GLB_FIXED".
        # Therefore, check the logic.
        return_df = basket_1.return_basket()
        return_df_c = return_df.copy()
        return_df_c['ticker'] = return_df_c['cid'] + '_' + return_df_c['xcat']

        tickers = return_df_c['ticker'].to_numpy()
        truncate = lambda ticker: "_".join(ticker.split('_')[0:2])

        basket_names = list(set(map(truncate, tickers)))
        # The usage of the set() data structure will sort the basket names in
        # alphabetical order. The order stored in the dataframe will correspond to the
        # order in which make_basket() method is called. Therefore, in the testing,
        # account for the behaviour of the set().
        test = ['GLB_EQUAL', 'GLB_FIXED', 'GLB_INVERSE']
        self.assertTrue(sorted(basket_names) == test)

        # Last check is to confirm that the stacked weight dataframe and basket
        # performance are defined over the same time-period.
        basket_equal = basket_1.return_basket(basket_names="GLB_EQUAL")
        weight_equal = basket_1.return_weights(basket_names="GLB_EQUAL")
        basket_equal_dates = list(set(basket_equal['real_date']))
        weight_equal_dates = list(set(weight_equal['real_date']))

        self.assertTrue(basket_equal_dates == weight_equal_dates)


if __name__ == "__main__":

    unittest.main()
