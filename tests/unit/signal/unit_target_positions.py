

import unittest
from macrosynergy.signal.target_positions import *
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.basket import Basket
from tests.simulate import make_qdf
import random
import pandas as pd
import numpy as np
import math

class TestAll(unittest.TestCase):

    def dataframe_generator(self):
        """Create  standardised dataframe defined over the three categories"""

        self.__dict__['cids'] = ['AUD', 'GBP', 'JPY', 'NZD', 'USD']
        # Two meaningful fields and a third contrived category.
        self.__dict__['xcats'] = ['FXXR_NSA', 'EQXR_NSA', 'SIG_NSA']
        self.__dict__['ctypes'] = ['FX', 'EQ']
        self.__dict__['xcat_sig'] = 'FXXR_NSA'

        df_cids = pd.DataFrame(index=self.cids, columns=['earliest', 'latest',
                                                         'mean_add', 'sd_mult'])

        df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
        df_cids.loc['GBP'] = ['2010-01-01', '2020-12-31', 0, 2]
        df_cids.loc['JPY'] = ['2010-01-01', '2020-12-31', 0, 5]
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

        # Exclude the blacklist from the creation of the dataframe. All dates are used
        # for calculating the evolving volatility for the volatility targeting mechanism.
        dfd_reduced = reduce_df(df=self.dfd, xcats=[self.xcat_sig], cids=self.cids,
                                start='2012-01-01', end='2020-10-30',
                                blacklist=None)
        self.__dict__['dfd_reduced'] = dfd_reduced
        self.__dict__['df_pivot'] = dfd_reduced.pivot(index="real_date", columns="cid",
                                                      values="value")

    def test_weight_dataframes(self):
        """
        Tests separability and consistency of returns and weights.
        """

        # The original dataframe target_positions() receives will be a single,
        # concatenated dataframe which will include the basket weight dataframes if,
        # axiomatically, baskets are included.
        # Therefore, the method weight_dataframes() will return the pivoted weight
        # dataframes and a dictionary of the basket's name and the associated
        # constituents.

        self.dataframe_generator()
        dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                        start=None, end=None, blacklist=None)

        # Make the baskets to append to the standardised dataframe.
        west_contracts = ['GBP_FX', 'USD_FX']
        apc_contracts = ['AUD_EQ', 'JPY_EQ', 'NZD_EQ']
        basket_1 = Basket(df=dfd, contracts=west_contracts, ret="XR_NSA",
                          cry=None)

        basket_1.make_basket(weight_meth="invsd", lback_meth="ma", lback_periods=21,
                             max_weight=0.55, remove_zeros=True,
                             basket_name="WST_FX")
        df_weight_1 = basket_1.return_weights("WST_FX")

        basket_2 = Basket(df=dfd, contracts=apc_contracts, ret="XR_NSA",
                          cry=None)
        basket_2.make_basket(weight_meth="equal", max_weight=0.55,
                             basket_name="APC_EQ")
        df_weight_2 = basket_2.return_weights("APC_EQ")
        df_weight = [df_weight_1, df_weight_2]
        dfd_concat = pd.concat([dfd] + df_weight)

        b_names = ["WST_FX", "APC_EQ"]
        df_c_wgts, b_dict = weight_dataframes(df=dfd_concat, basket_names=b_names)
        # Confirm the dictionary's keys equates to the name of the two baskets.
        self.assertTrue(list(b_dict.keys()) == b_names)
        column_names = [west_contracts, apc_contracts]

        # Confirm that the columns of the weight dataframe match the constituents of the
        # associated basket.
        for i, df in enumerate(df_c_wgts):
            self.assertTrue(list(df.columns) == column_names[i])

    def test_modify_signals(self):
        """
        Test if unit positions are correct.
        """

        self.dataframe_generator()
        xcat_sig = 'SIG_NSA'
        dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                        start='2012-01-02', end='2020-10-30',
                        blacklist=None)
        dfd = dfd.reset_index(drop=True)

        with self.assertRaises(AssertionError):
            # Test that scale parameter outside ['prop', 'dig'] throws assertion error.
            scale = 'noise'
            df_unit_pos = modify_signals(df=dfd, cids=self.cids, xcat_sig=xcat_sig,
                                         start='2012-01-01', end='2020-10-30',
                                         scale=scale)

        # A. Test unit positions without zeroes and NAs.

        df_unit_pos = modify_signals(df=dfd, cids=self.cids, xcat_sig=xcat_sig,
                                     start='2012-01-01', end='2020-10-30', scale='dig')

        # Test that output has only one signal category and that it is right one
        self.assertTrue(df_unit_pos['xcat'].unique().size == 1)
        self.assertTrue(np.all(df_unit_pos['xcat'] == xcat_sig))

        # Check that all (digital) position values are exactly 1
        self.assertTrue(np.unique(np.abs(df_unit_pos['value'])) == 1)

        # Reduce the dataframe to the signal & check the logic is correct.
        dfd_sig = dfd[dfd['xcat'] == xcat_sig]
        dfd_sig = dfd_sig.reset_index(drop=True)
        condition = np.where(dfd_sig['value'].to_numpy() < 0)[0]
        first_negative_index = next(iter(condition))
        val_column = df_unit_pos['value'].to_numpy()
        self.assertTrue(val_column[first_negative_index] == -1)

        # B. Test unit positions with 0s.
        # Often a position will only be taken if the signal is outside a predetermined
        # range. Therefore, say if the value is within one standard deviation of the
        # neutral level, a position will not be taken as the magnitude of the signal is
        # not large enough and the subsequent output should be a zero value representing
        # a modified signal stating a position should not be taken for that timestamp.

        # Further, a distinction between NA and zero must be made. A NA value represents
        # a signal not being available for that timestamp, so the corresponding modified
        # signal will also be NA. The modified signal of NA will be converted to a zero
        # position.

        dfd_copy = dfd.copy()
        # The signal will be the last category in the dataframe.
        dfd_copy.iloc[-100:, -1] = 0
        df_unit_poz = modify_signals(df=dfd_copy, cids=self.cids,
                                     xcat_sig=xcat_sig, start='2012-01-01',
                                     end='2020-10-30', scale='dig')

        # Test that all absolute signal values are 0s and 1s
        unique_values = np.unique(np.abs(df_unit_poz['value']))
        self.assertTrue(np.isin(unique_values, [0, 1]).all())

        # Test that zero signals translate into zero modified signals
        presumed_zeroes = df_unit_poz['value'][-100:]
        self.assertTrue(np.unique(np.abs(presumed_zeroes)) == 0)

        # C. Test unit positions with zeroes and NAs.
        # Confirm the distinction between the two inputs. Only in target_positions() will
        # the output be the same (both zero positions).

        dfd_copy_two = dfd.copy()
        # Isolate the value column.
        dfd_copy_two.iloc[-50:, -1] = 0
        dfd_copy_two.iloc[-100:-51, -1] = np.nan
        df_unit_poz_two = modify_signals(df=dfd_copy_two, cids=self.cids,
                                         xcat_sig=xcat_sig, start='2012-01-01',
                                         end='2020-10-30', scale='dig')
        presumed_zeroes = df_unit_poz_two['value'][-50:]
        self.assertTrue(np.unique(presumed_zeroes) == 0)

        # As discussed, the modified signal will be NA given data is not available for
        # the respective timestamps.
        presumed_nas = df_unit_poz_two['value'][-100:-51]
        for nan in presumed_nas:
            self.assertTrue(math.isnan(nan))

        # D. Test proportionate signal values.

        # Modify the dataframe to include np.nan and zero. Confirm the logic still
        # stands.
        dfd_signal_copy = dfd[dfd['xcat'] == xcat_sig]
        # Change the first two timestamps for Australia signal: [2012-01-02, 2012-01-03].
        dfd_signal_copy.iloc[0, -1] = 0
        dfd_signal_copy.iloc[1, -1] = np.nan
        dfd_fxxr = dfd[dfd['xcat'] == 'FXXR_NSA']
        dfd_eqxr = dfd[dfd['xcat'] == 'EQXR_NSA']
        dfd_modified = pd.concat([dfd_fxxr, dfd_eqxr, dfd_signal_copy])

        df_unit_pos = modify_signals(df=dfd_modified, cids=self.cids, xcat_sig=xcat_sig,
                                     start='2012-01-01', end='2020-10-30', scale='prop',
                                     min_obs=0, thresh=2.5)

        # Test shape of output dataframe.
        # Accounts for the NaN value assigned to Australia. When pivoting out the NaN
        # value will be populated back.
        self.assertTrue(df_unit_pos.shape[0] == (dfd_signal_copy.shape[0] - 1))

        # Test if proportionate signals have correct values.

        df_unit_pos = modify_signals(df=dfd_modified, cids=self.cids, xcat_sig=xcat_sig,
                                     start='2012-01-01', end='2020-10-30', scale='prop',
                                     min_obs=0, thresh=5)
        df_unit_pos_w = df_unit_pos.pivot(index="real_date", columns="cid",
                                          values="value")
        output_rows = df_unit_pos_w.iloc[0:5, :]

        # Test on the first five rows which will include the inserted zero and np.nan
        # values.
        dfd_sig_w = dfd_signal_copy.pivot(index="real_date", columns="cid",
                                          values="value")
        test_rows = dfd_sig_w.iloc[0:5, :]
        # The neutral value is zero.
        numerator = test_rows - np.zeros(test_rows.shape)
        std = []
        for i in range(test_rows.shape[0]):
            # Rolling standard deviation across the panel (same as the neutral level
            # across the panel).
            row = np.abs(numerator).iloc[0:(i + 1)].to_numpy()
            row = row.reshape(row.shape[0] * row.shape[1])
            mean = np.nanmean(row)
            std.append(mean)

        std = np.array(std)
        row_vector = std[:, np.newaxis]
        test_zn_scores = numerator.div(row_vector, axis='rows')
        condition = test_zn_scores.to_numpy() - output_rows.to_numpy()
        # Convert the NaN value to zero for testing purposes only.
        condition = np.nan_to_num(condition)
        self.assertTrue(np.all(condition < 0.0001))

    @staticmethod
    def row_return(dfd, date, c_return, sigrel):

        df_c_ret = dfd[dfd['xcat'] == c_return]
        df_c_ret = df_c_ret.pivot(index="real_date", columns="cid", values="value")
        df_c_ret *= sigrel

        return df_c_ret.loc[date]

    def test_cs_unit_returns(self):
        # The method is required for volatility targeting to adjust the respective
        # positions.
        self.dataframe_generator()

        sigrels = [1, -1]
        ret = 'XR_NSA'
        contract_returns = [c + ret for c in self.ctypes]
        xcat_sig = 'FXXR_NSA'

        dates = self.df_pivot.index
        dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                        start='2012-01-02', end='2020-10-30',
                        blacklist=None)

        # The main purpose of cs_unit_returns is to aggregate the returns across the
        # respective panels. The signal, used to take positions, can determine the
        # position across multiple asset classes: Equity, Foreign Exchange etc.
        # Therefore, if there is the intention of volatility targeting, the volatility
        # should be computed across the panels (portfolio) which requires summing the
        # respective returns, and subsequently adjusting the position.

        # Test the number of sigrels matches the number of panels held in the parameter
        # "contract_returns".
        with self.assertRaises(AssertionError):
            sigrels_copy = [1, -1, 1]
            df_pos_vt = cs_unit_returns(df=dfd, contract_returns=contract_returns,
                                        sigrels=sigrels_copy, ret=ret)

        # The next test is to confirm the above logic. To achieve testing the aggregation
        # method, choose a date and confirm the return dataframe completes the binary
        # operation (including multiplying by the sigrels).

        df_pos_vt = cs_unit_returns(df=dfd, contract_returns=contract_returns,
                                    sigrels=sigrels, ret=ret)
        dates = df_pos_vt['real_date']
        date = dates.iloc[int(dates.shape[0] / 2)]
        # Returns across asset class.
        test_row = df_pos_vt[df_pos_vt['real_date'] == date]
        test_row = test_row[['cid', 'value']]
        test_dict = dict(zip(test_row['cid'], test_row['value']))

        fx_row = self.row_return(dfd=dfd, date=date, c_return=contract_returns[0],
                                 sigrel=sigrels[0])
        eq_row = self.row_return(dfd=dfd, date=date, c_return=contract_returns[1],
                                 sigrel=sigrels[1])
        manual_calculation = fx_row + eq_row
        manual_calculation = manual_calculation.to_dict()

        # The key will correspond to the cross-section and the value the associated
        # return across both assets.
        for k, v in test_dict.items():
            self.assertTrue(v == manual_calculation[k])

        # The final test is to validate that the first available date for the cross-asset
        # returns is "2012-01-02". If any dates are not shared across the two assets, the
        # respective dates will be removed from the return dataframe (used to compute
        # the volatility across the positions).
        first_date = str(dates.iloc[0].strftime("%Y-%m-%d"))
        self.assertTrue(first_date == "2012-01-02")

    def test_basket_handler(self):

        self.dataframe_generator()

        reduced_dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                                blacklist=None)
        # The positions are computed across the panel (for every cross-section available
        # for that asset class). However, baskets can also be passed into the function
        # and the basket of contracts will represent a subset of the panel. Further, each
        # contract within the basket has a corresponding weight to adjust the position.
        # Therefore, test that the basket_handler() produces the correct weight-adjusted
        # positions for the contracts in the basket.

        xcat_sig = 'SIG_NSA'
        scale = 'prop'
        df_mods = modify_signals(df=reduced_dfd, cids=self.cids, xcat_sig=xcat_sig,
                                 scale=scale)

        # The position computed using the signal is used for multiple assets. The only
        # difference is the signal.
        df_mods_w = df_mods.pivot(index="real_date", columns="cid", values="value")

        # Create the basket: the basket will be a subset of the cross-sections present in
        # the panel.
        apc_contracts = ['AUD_FX', 'NZD_FX']
        basket_1 = Basket(df=reduced_dfd, contracts=apc_contracts, ret="XR_NSA",
                          cry=None)
        basket_1.make_basket(weight_meth="equal", max_weight=0.55,
                             basket_name="APC_FX")
        # Required wide, pivoted dataframe.
        dfw_wgs = basket_1.dict_wgs["APC_FX"]

        df_mods_w_output = basket_handler(df_mods_w=df_mods_w, df_c_wgts=dfw_wgs,
                                          contracts=apc_contracts)

        # The first test is to confirm that the position dataframe, used for the basket,
        # contains the same number of contracts, columns, as the basket itself.
        self.assertTrue(df_mods_w_output.shape[1] == len(apc_contracts))

        # Complete the manual calculation to test the method (weight-adjusted positions).
        df_mods_w = df_mods_w.reindex(sorted(df_mods_w.columns), axis=1)

        split = lambda b: b.split('_')[0]
        df_mods_w_basket = df_mods_w[list(map(split, apc_contracts))]

        # Confirm alignment of contracts.
        dfw_wgs = dfw_wgs.reindex(sorted(dfw_wgs.columns), axis=1)
        df_mods_w_basket = df_mods_w_basket.multiply(dfw_wgs.to_numpy())

        self.assertTrue(np.all(df_mods_w_basket.index == df_mods_w_basket.index))
        condition = df_mods_w_output.to_numpy() == df_mods_w_basket.to_numpy()
        self.assertTrue(np.all(condition))

    @staticmethod
    def basket_weight(reduced_dfd, df_mods_w, local_contracts, basket_name):
        """
        Method used to produce the weight adjusted positions for a basket. The weighting
        method chosen for the basket is fixed to inverse of the standard deviation.

        :param <pd.DataFrame> reduced_dfd: standardised dataframe.
        :param <pd.DataFrame> df_mods_w: target positions across the panel (the signal).
        :param <List[str]> local_contracts: contracts the basket is defined over.
        :param <str> basket_name: name of the respective basket.

        :return <pd.DataFrame> df_basket_pos: the weight-adjusted basket positions.
        """

        basket_1 = Basket(df=reduced_dfd, contracts=local_contracts, ret="XR_NSA",
                          cry=None)
        basket_1.make_basket(weight_meth="invsd", lback_meth="ma", lback_periods=21,
                             max_weight=0.55, remove_zeros=True,
                             basket_name=basket_name)
        # Pivoted weight dataframe.
        df_c_wgts = basket_1.dict_wgs[basket_name]

        # Return the weight adjusted target positions for the contract. This will be the
        # benchmark dataframe being tested.
        df_basket_pos = basket_handler(df_mods_w=df_mods_w, df_c_wgts=df_c_wgts,
                                       contracts=local_contracts)
        return df_basket_pos

    def test_consolidation_help(self):

        self.dataframe_generator()
        # If a basket of contracts are defined, their respective positions will be weight
        # adjusted. After the weight-adjusted positions are computed for the basket,
        # consolidate the positions on the shared contracts: intersection between the
        # panel and respective basket.
        # Therefore, check whether the above logic is implemented.

        # Further, it is worth noting that the basket should be a subset of the panel:
        # the panel is the complete set. Therefore, additional assertions are not
        # required.
        reduced_dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                                blacklist=None)

        # Establish the targeted positions using the modified returns of the signal.
        xcat_sig = 'SIG_NSA'
        scale = 'prop'

        # The two defined sigrels are used for the panel FX and the basket
        # "apc_contracts."
        sigrels = [1, -1]

        # "ZN" postfix is the default postfix for "make_zn_scores()". Will return a
        # standardised dataframe.
        df_mods = modify_signals(df=reduced_dfd, cids=self.cids, xcat_sig=xcat_sig,
                                 scale=scale)
        df_mods_w = df_mods.pivot(index="real_date", columns="cid", values="value")

        cross_sections = lambda c: c.split('_')[0]
        apc_contracts = ['AUD_FX', 'NZD_FX']
        contract_cids = list(map(cross_sections, apc_contracts))
        non_basket_cids = [c for c in self.cids if c not in contract_cids]

        df_basket_pos = self.basket_weight(reduced_dfd, df_mods_w, apc_contracts,
                                           basket_name="APC_FX")

        # Convert both the panel & basket dataframes to standardised dataframes.
        df_basket_stack = df_basket_pos.stack().to_frame("value").reset_index()
        # Test dataframe.
        consolidate_df = consolidation_help(panel_df=df_mods, basket_df=df_basket_stack)

        # The consolidation will only be applicable on the intersection of contracts.
        # Therefore, check the other contract's positions remain the same.
        # Analyse on an arbitrary date.
        test_index = 1000
        test_date = df_basket_pos.index.to_numpy()[test_index]
        test_values = consolidate_df[consolidate_df['real_date'] == test_date]

        # Original positions: based on the panel, so columns will correspond to
        # cross-sections.
        df_mods_w_test_date = df_mods_w.loc[test_date]
        # Test date for the basket positions.
        df_basket_pos_test_date = df_basket_pos.loc[test_date]

        for c in non_basket_cids:
            logic = test_values[test_values['cid'] == c]['value']
            t_val = df_mods_w_test_date[c]
            logic = float(logic)
            self.assertTrue(abs(t_val - logic) < 0.000001)

        # Consolidate the positions computed for the basket contracts and the respective
        # panel.
        for c in contract_cids:
            logic = test_values[test_values['cid'] == c]['value']
            logic = float(logic)
            panel_val = df_mods_w_test_date[c]
            basket_val = df_basket_pos_test_date[c]
            # Consolidation operation.
            test = panel_val + basket_val
            self.assertTrue(abs(logic - test) < 0.000001)

    def test_consolidate_positions(self):

        self.dataframe_generator()
        # Conceals the function above and applies the logic to multiple baskets which are
        # consolidated to the associated number of panels.
        # Confirm the application of multiple baskets works.

        # Required multiple panels and baskets.

        reduced_dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                                blacklist=None)

        # Establish the targeted positions using the modified returns of the signal.
        xcat_sig = 'SIG_NSA'
        scale = 'dig'
        # Modified the signal returns to digital: removes the magnitude of the position,
        # and exclusively classifies whether to take a long or short position in the
        # contract.
        df_mods = modify_signals(df=reduced_dfd, cids=self.cids, xcat_sig=xcat_sig,
                                 scale=scale)
        df_mods_w = df_mods.pivot(index="real_date", columns="cid", values="value")

        # The four defined sigrels are used for the two panels FX & EQ, and the baskets
        # "apc_contracts." and "west_contracts".
        sigrels = [1, -1, -0.5, 1.5]
        west_contracts = ['GBP_FX', 'USD_FX']
        apc_contracts = ['AUD_EQ', 'JPY_EQ', 'NZD_EQ']

        # The signal panel is used to establish positions in both panels. The two
        # dataframes below are the weight-adjusted target positions for the respective
        # baskets.
        df_basket_west = self.basket_weight(reduced_dfd, df_mods_w, west_contracts,
                                            basket_name="WST_FX")
        df_basket_apc = self.basket_weight(reduced_dfd, df_mods_w, apc_contracts,
                                           basket_name="APC_EQ")
        baskets = [df_basket_west, df_basket_apc]
        b_names = ["WST_FX", "APC_EQ"]
        dict_ = dict(zip(b_names, baskets))

        # The below two For Loops will test the workflow of target_positions() which will
        # complete the operations inside a single loop.
        # Contract Types: FX & EQ.
        df_pos_cons = []
        for i, c in enumerate(self.ctypes):

            df_mods_copy = df_mods_w.copy()
            df_mods_copy *= sigrels[i]
            df_posi = df_mods_copy.stack().to_frame("value").reset_index()
            df_posi['xcat'] = c
            df_posi = df_posi.sort_values(['cid', 'xcat', 'real_date'])
            df_pos_cons.append(df_posi)

        # Two separate loops to make the intention explicit and confirm the logic. The
        # list, "df_pos_cons", after the iteration, will contain both the panel
        # dataframes and the basket dataframes.
        i = 0
        for k, v in dict_.items():
            sig = sigrels[i + len(self.ctypes)]
            v = v * sig
            df_posi = v.stack().to_frame("value").reset_index()
            df_posi['xcat'] = k
            df_posi = df_posi.sort_values(['cid', 'xcat', 'real_date'])
            df_pos_cons.append(df_posi)
            i += 1

        # Assert the consolidation has occurred correctly across both panels.
        df_pos_cons_output = consolidate_positions(df_pos_cons, self.ctypes)
        # The number of dataframes in the returned list should correspond to the number
        # of panels: the basket's position is added to the panel dataframe given the
        # basket is a subset of the panel.
        self.assertTrue(len(df_pos_cons_output) == len(self.ctypes))
        # Further, the expected category name, "xcat", of the dataframes inside the
        # returned list should be set to one of the contract types (each basket
        # dataframe will be subsumed under the panel dataframes: the contracts are the
        # same, so the positions are aggregated).
        for df in df_pos_cons_output:
            test_xcat = ''.join(df['xcat'].unique())
            self.assertTrue(test_xcat in self.ctypes)

        # Test the consolidation across the baskets.
        test_index = 1000
        test_date = df_mods_w.index.to_numpy()[test_index]

        # The basket's signal is -0.5.
        df_basket_west_test = df_basket_west.loc[test_date]
        # Basket position for its respective constituents.
        df_basket_west_test *= -0.5
        df_basket_west_test = df_basket_west_test.to_dict()

        # Panel position. Sigrel has already been applied.
        fx_panel = df_pos_cons[0]
        fx_panel_test = fx_panel[fx_panel['real_date'] == test_date]
        fx_panel_values = dict(zip(fx_panel_test['cid'], fx_panel_test['value']))

        for k, v in fx_panel_values.items():
            if k in df_basket_west_test.keys():
                fx_panel_values[k] += df_basket_west_test[k]

        fx_output, eq_output = df_pos_cons_output
        benchmark = fx_output[fx_output['real_date'] == test_date]
        benchmark_dict = dict(zip(benchmark['cid'], benchmark['value']))
        for k, v in benchmark_dict.items():
            self.assertTrue(abs(v - fx_panel_values[k]) < 0.00001)

        # Repeat the operation above but for the Equity Panel.
        # The basket's signal is 1.5.
        df_basket_apc_test = df_basket_apc.loc[test_date]
        df_basket_apc_test *= 1.5
        df_basket_apc_test = df_basket_apc_test.to_dict()

        eq_panel = df_pos_cons[1]
        eq_panel_test = eq_panel[eq_panel['real_date'] == test_date]
        eq_panel_values = dict(zip(eq_panel_test['cid'], eq_panel_test['value']))

        for k, v in eq_panel_values.items():
            if k in df_basket_apc_test.keys():
                eq_panel_values[k] += df_basket_apc_test[k]

        # Check the values against the function consolidate_positions().
        benchmark = eq_output[eq_output['real_date'] == test_date]
        benchmark_dict = dict(zip(benchmark['cid'], benchmark['value']))
        for k, v in benchmark_dict.items():
            self.assertTrue(abs(v - eq_panel_values[k]) < 0.00001)

    def test_target_positions(self):

        # The majority of the logic held within target_positions() has already been
        # tested through the above methods. The remaining operations, following on from
        # consolidating across the panel & basket, is to concatenate the panel dataframes
        # into a single dataframe and subsequently add the position name to the category
        # column.
        # The workflow and logic has already been examined individually with the
        # final method dependent on the previous steps being executed correctly.

        self.dataframe_generator()

        reduced_dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                                blacklist=None)

        with self.assertRaises(AssertionError):
            # Test the assertion that the signal field must be present in the defined
            # dataframe. Will throw an assertion.
            xcat_sig = 'INTGRWTH_NSA'
            position_df = target_positions(df=reduced_dfd, cids=self.cids,
                                           xcat_sig=xcat_sig,
                                           ctypes=['FX', 'EQ'], sigrels=[1, -1],
                                           ret='XR_NSA', start='2012-01-01',
                                           end='2020-10-30', scale='prop',
                                           cs_vtarg=0.1, posname='POS')

        with self.assertRaises(AssertionError):
            # Test the assertion that the volatility target must be a numerical value.
            xcat_sig = 'FXXR_NSA'
            sigrels = [-1, 1]
            output_df = target_positions(df=reduced_dfd, cids=self.cids,
                                         xcat_sig=xcat_sig, ctypes=self.ctypes,
                                         sigrels=sigrels, ret='XR_NSA',
                                         start=None, end='2020-12-31', scale='prop',
                                         cs_vtarg="1 STD", posname='POS')

        with self.assertRaises(AssertionError):
            # Testing the assertion on the scale parameter: required ['prop', 'dig'].
            # Pass in noise.
            scale = 'vtarg'
            position_df = target_positions(df=reduced_dfd, cids=self.cids,
                                           xcat_sig=xcat_sig,
                                           ctypes=['FX', 'EQ'], sigrels=[1, -1],
                                           ret='XR_NSA', start='2012-01-01',
                                           end='2020-10-30', scale=scale,
                                           cs_vtarg=0.1, posname='POS')

        with self.assertRaises(AssertionError):
            # Length of "sigrels" and "ctypes" must be the same. The number of "sigrels"
            # corresponds to the number of "ctypes" plus the number of baskets.

            sigrels = [1, -1, 0.5, 0.25]
            ctypes = ['FX', 'EQ']
            position_df = target_positions(df=reduced_dfd, cids=self.cids,
                                           xcat_sig=xcat_sig,
                                           ctypes=ctypes, sigrels=sigrels,
                                           ret='XR_NSA',  start='2012-01-01',
                                           end='2020-10-30', scale=scale,
                                           cs_vtarg=0.1, posname='POS')

        with self.assertRaises(AssertionError):
            # Test the notion that the received dataframe must be a standardised
            # dataframe with the columns ['cid', 'xcat', 'real_date', 'value'].
            ctypes = ['FX', 'EQ']
            sigrels = [-1, 1]
            temp_df = reduced_dfd[reduced_dfd['xcat'] == 'FX']
            temp_df = temp_df.pivot(index="real_date", columns="cid", values="value")
            xcat_sig = 'SIG_NSA'
            position_df = target_positions(df=temp_df, cids=self.cids,
                                           xcat_sig=xcat_sig,
                                           ctypes=ctypes, sigrels=sigrels,
                                           ret='XR_NSA',  start='2012-01-01',
                                           end='2020-10-30', scale=scale,
                                           cs_vtarg=0.1, posname='POS')

        # The only aspect of target_position()'s workflow which is not examined in the
        # above methods is the application of volatility targeting.
        # Test on a single category.
        ctype = self.ctypes[0]  # FX.
        output_df_vol = target_positions(df=reduced_dfd, cids=self.cids,
                                     xcat_sig=xcat_sig, ctypes=ctype,
                                     sigrels=[-1], ret='XR_NSA',
                                     start=None, end='2020-12-31', scale='prop',
                                     cs_vtarg=0.4, posname='POS')
        # Test on a single category without volatility targeting.
        output_df = target_positions(df=reduced_dfd, cids=self.cids,
                                     xcat_sig=xcat_sig, ctypes=ctype,
                                     sigrels=[-1], ret='XR_NSA',
                                     start=None, end='2020-12-31', scale='prop',
                                     cs_vtarg=None, posname='POS')
        # The first aspect to confirm is the application of volatility targeting will
        # truncate the returned dataframe by the number of cross_sections multiplied by
        # the lookback period (default is 21, so the first twenty days will not be
        # populated with volatility values).
        # By confirming the dimensions of the returned dataframe follow the above logic,
        # it shows the conditional constructs are being applied correctly.
        vol_no_rows = output_df_vol.shape[0]

        self.assertTrue(vol_no_rows == output_df.shape[0] - (20 * len(self.cids)))

        xcat_sig = 'SIG_NSA'
        sigrels = [-1, 1]

        # A limited but valid test to determine the logic is correct for the volatility
        # target is to set the value equal to zero, and consequently all the
        # corresponding positions should also be zero. If zero volatility is required,
        # unable to take a position in an asset that has non-zero standard deviation.
        output_df = target_positions(df=reduced_dfd, cids=self.cids,
                                     xcat_sig=xcat_sig, ctypes=self.ctypes,
                                     sigrels=sigrels, ret='XR_NSA',
                                     start=None, end='2020-12-31', scale='prop',
                                     cs_vtarg=0.0, posname='POS')
        self.assertTrue(np.all(output_df['value'] == 0.0))

        # Simple examples to test on the returned target positions.
        # In this permutation of target_positions, the position value is exclusively
        # predicated on the zn-score and the associated sigrels. The zn_scores method has
        # already been tested. Therefore, confirm the application of the sigrel.
        output_df_2 = target_positions(df=reduced_dfd, cids=self.cids, xcat_sig=xcat_sig,
                                       ctypes=self.ctypes, sigrels=[1, -1], ret='XR_NSA',
                                       start='2010-01-01', end='2020-12-31',
                                       scale='prop', cs_vtarg=None,
                                       posname='POS')

        # To test, given 'FX' & 'EQ' have opposing sigrels, add their value and confirm
        # the addition equates to zero.
        split = lambda xc: xc.split('_')[0]

        # Remove the "posname" from the category.
        output_df_2['xcat'] = np.array(list(map(split, output_df_2['xcat'])))
        eq = output_df_2[output_df_2['xcat'] == 'EQ']
        fx = output_df_2[output_df_2['xcat'] == 'FX']
        eq = eq.pivot(index="real_date", columns="cid", values="value")
        fx = fx.pivot(index="real_date", columns="cid", values="value")

        test = eq.add(fx)
        self.assertTrue(np.all(test.to_numpy() < 0.00001))

        reduced_dfd = reduce_df(df=self.dfd, xcats=self.xcats, cids=self.cids,
                                blacklist=None)
        # Modify the dataframe to include np.nan and zero in fixed, known positions.
        # Confirm target_positions will convert the modified signal of NaNs and zeros to
        # a zero position (a position is not taken).
        dfd_signal_copy = reduced_dfd[reduced_dfd['xcat'] == xcat_sig]
        # Change the first two timestamps for Australia signal: [2012-01-02, 2012-01-03].
        dfd_signal_copy.iloc[0, -1] = 0
        dfd_signal_copy.iloc[1, -1] = np.nan
        dfd_fxxr = reduced_dfd[reduced_dfd['xcat'] == 'FXXR_NSA']
        dfd_eqxr = reduced_dfd[reduced_dfd['xcat'] == 'EQXR_NSA']
        dfd_modified = pd.concat([dfd_fxxr, dfd_eqxr, dfd_signal_copy])
        output_df_3 = target_positions(df=dfd_modified, cids=self.cids, xcat_sig=xcat_sig,
                                       ctypes=self.ctypes, sigrels=[1, -1], ret='XR_NSA',
                                       start='2010-01-01', end='2020-12-31',
                                       scale='prop', cs_vtarg=None,
                                       posname='POS')
        # Confirm the date '2012-01-03' has a np.nan for Australia in both categories.
        output_df_eq = output_df_3[output_df_3['xcat'] == "EQ_POS"]
        output_df_fx = output_df_3[output_df_3['xcat'] == "FX_POS"]
        output_df_eq_aud = output_df_eq[output_df_eq['cid'] == 'AUD']
        output_df_fx_aud = output_df_fx[output_df_fx['cid'] == 'AUD']

        # The first two position values should both be equal to zero (the modified signal
        # was zero and np.nan respectively).
        self.assertTrue(np.all(np.abs(output_df_eq_aud.iloc[0:2, -1]) == 0.0))
        self.assertTrue(np.all(np.abs(output_df_fx_aud.iloc[0:2, -1]) == 0.0))


if __name__ == "__main__":

    unittest.main()