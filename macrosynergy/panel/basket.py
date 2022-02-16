
import numpy as np
import pandas as pd
import random
from typing import List
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.shape_dfs import reduce_df_by_ticker
from macrosynergy.panel.converge_row import ConvergeRow
from macrosynergy.management.simulate_quantamental_data import make_qdf


class Basket(object):

    def __init__(self, df: pd.DataFrame, contracts: List[str], ret: str = "XR_NSA",
                 cry: List[str] = None, start: str = None, end: str = None,
                 blacklist: dict = None, ewgts: List[str] = None):

        """
        Calculates the returns and carries of baskets of financial contracts using
        various weighting methods.

        :param <pd.Dataframe> df: standardized DataFrame with following columns: 'cid',
            'xcats', 'real_date' and 'value'.
        :param <List[str]> contracts: base tickers (combinations of cross-sections and
            base categories) denoting contracts that go into the basket.
        :param <str> ret: return category postfix; default is "XR_NSA".
        :param <List[str] or str> cry: carry category postfix; default is None. The field
            can either be a single carry or multiple carries defined in a List.
        :param <str> start: earliest date in ISO 8601 format. Default is None.
        :param <str> end: latest date in ISO 8601 format. Default is None.
        :param <dict> blacklist: cross-sections with date ranges that should be excluded
            from the dataframe. If one cross-section has several blacklist periods append
            numbers to the cross-section code.
        :param List[str] ewgts: one or more postfixes that may identify exogenous weight
            categories. Similar to return postfixes they are appended to base tickers.

        """

        assert isinstance(contracts, list)
        assert all(isinstance(c, str) for c in contracts), \
            "`contracts` must be list of strings"
        self.contract = contracts  # Todo: must be self.contracts, but hard to refactor

        assert isinstance(ret, str), "`ret`must be a string"
        self.ret = ret
        self.ticks_ret = [con + ret for con in contracts]
        self.dfw_ret = self.pivot_dataframe(df, self.ticks_ret)

        if cry is not None:
            error = "`cry` must be a string or a list of strings"
            assert isinstance(cry, (list, str)), error
        cry = [cry] if isinstance(cry, str) else cry  # remove ambiguity of type
        self.cry = cry
        self.ticklists_cry = {}
        self.dfws_cry = {}
        self.ticks_cry = []
        if cry is not None:
            for cr in cry:
                ticks = [con + cr for con in contracts]
                self.ticklists_cry[cr] = ticks
                self.ticks_cry = self.ticks_cry + ticks
                self.dfws_cry[cr] = self.pivot_dataframe(df, self.ticklists_cry[cr])

        if ewgts is not None:
            error = "`ewgts` must be a string or a list of strings"
            assert isinstance(ewgts, (list, str)), error
        wgt = [ewgts] if isinstance(ewgts, str) else ewgts  # remove ambiguity of type
        self.wgt = wgt
        if wgt is not None:
            self.ticks_wgt = [con + wg for con in contracts for wg in wgt]
        else:
            self.ticks_wgt = []

        self.tickers = self.ticks_ret + self.ticks_cry + self.ticks_wgt
        self.start = self.date_check(start)
        self.end = self.date_check(end)
        self.dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=self.tickers,
                                       blacklist=blacklist)

    @staticmethod
    def pivot_dataframe(df, tick_list):
        """Makes a wide dataframe with time index"""
        df['ticker'] = df['cid'] + '_' + df['xcat']
        dfx = df[df["ticker"].isin(tick_list)]
        dfw = dfx.pivot(index="real_date", columns="ticker", values="value")
        return dfw

    @staticmethod
    def date_check(date_string):
        """Checks if string can be converted into date format"""
        date_error = "Expected form of string: '%Y-%m-%d'."
        if date_string is not None:
            try:
                pd.Timestamp(date_string).strftime("%Y-%m-%d")
            except ValueError:
                raise AssertionError(date_error)

    @staticmethod
    def check_weights(weight: pd.DataFrame):
        """
        Checks if all rows in dataframe add up to roughly 1

        :param <pd.DataFrame> weight: weight dataframe.
        """
        check = weight.sum(axis=1)
        c = ~((abs(check - 1) < 1e-6) | (abs(check) < 1e-6))
        assert not any(c), f"weights must sum to one (or zero), not: {check[c]}"

    def equal_weight(self, df_ret: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates dataframe of equal weights based on available return data.

        :param <pd.DataFrame> df_ret: wide time-indexed data frame of returns.

        :return <pd.DataFrame>: dataframe of weights.

        Note: The method determines the  number of non-NA cross-sections per timestamp,
        and subsequently distributes the weights evenly across non-NA cross-sections.
        """

        act_cross = (~df_ret.isnull())
        uniform = (1 / act_cross.sum(axis=1)).values
        uniform = uniform[:, np.newaxis]

        broadcast = np.repeat(uniform, df_ret.shape[1], axis=1)

        weight = act_cross.multiply(broadcast)
        self.check_weights(weight=weight)

        return weight

    def fixed_weight(self, df_ret: pd.DataFrame, weights: List[float]):
        """
        Calculates fixed weights based on a single list of values and a corresponding
        return panel dataframe.

        :param <pd.DataFrame> df_ret: Return series matrix. Multidimensional.
        :param <List[float]> weights: List of floats determining weight allocation.

        :return <pd.DataFrame>: panel of weights
        """

        act_cross = (~df_ret.isnull())

        weights = np.array(weights, dtype=np.float32)
        rows = act_cross.shape[0]
        broadcast = np.tile(weights, (rows, 1))  # Constructs Array by row repetition.

        # Replaces weight factors with zeroes if concurrent return unavailable.
        weight = act_cross.multiply(broadcast)
        weight_arr = weight.to_numpy()  # convert df to np array.
        weight[weight.columns] = weight_arr / np.sum(weight_arr, axis=1)[:, np.newaxis]
        self.check_weights(weight)

        return weight

    def inverse_weight(self, dfw_ret: pd.DataFrame, lback_meth: str = "xma",
                       lback_periods: int = 21, remove_zeros: bool = True):
        """
        Calculates weights inversely proportionate to recent return standard deviations.

        :param <pd.DataFrame> dfw_ret: panel dataframe of returns.
        :param <str> lback_meth: Lookback method for "invsd" weighting method. Default is
            "xma".
        :param <int> lback_periods: Lookback periods. Default is 21.  Half-time for "xma"
            and full lookback period for "ma".
        :param <Bool> remove_zeros: Any returns that are exact zeros will not be included
            in the lookback window and prior non-zero values are added to the window
            instead.

        :return <pd.DataFrame>: Dataframe of weights.

        N.B.: The rolling standard deviation will be calculated either using the standard
        moving average (ma) or the exponential moving average (xma). Both will require
        returns before a first weight can be computed.
        """

        if lback_meth == "ma":
            dfwa = dfw_ret.rolling(window=lback_periods).agg(flat_std, remove_zeros)
            dfwa *= np.sqrt(252)

        else:

            half_life = lback_periods
            weights = expo_weights(lback_periods * 2, half_life)
            dfwa = dfw_ret.rolling(window=lback_periods * 2).agg(expo_std, w=weights,
                                                                 remove_zeros=remove_zeros)

        df_isd = 1 / dfwa
        df_wgts = df_isd / df_isd.sum(axis=1).values[:, np.newaxis]
        self.check_weights(df_wgts)

        return df_wgts

    def values_weight(self, dfw_ret: pd.DataFrame, dfw_wgt: pd.DataFrame,
                      weight_meth: str):
        """
        Returns weights based on an external weighting category.

        :param <pd.DataFrame> dfw_ret: Standard wide dataframe of returns across time and
            contracts.
        :param <pd.DataFrame> dfw_wgt: Standard wide dataframe of weight category values
            across time and contracts.
        :param <str> weight_meth:

        :return <pd.DataFrame>: Dataframe of weights.
        """

        negative_condition = np.any((dfw_wgt < 0).to_numpy())
        if negative_condition:
            dfw_wgt[dfw_wgt < 0] = 0.0
            print("Negative values in the weight matrix set to zero.")

        exo_array = dfw_wgt.to_numpy()
        df_bool = ~dfw_ret.isnull()

        weights_df = df_bool.multiply(exo_array)
        cols = weights_df.columns

        # Zeroes treated as NaNs.
        weights_df[cols] = weights_df[cols].replace({'0': np.nan, 0: np.nan})

        if weight_meth != "values":
            weights_df = 1 / weights_df

        weights = weights_df.divide(weights_df.sum(axis=1), axis=0)
        self.check_weights(weights)

        return weights

    @staticmethod
    def max_weight_func(weights: pd.DataFrame, max_weight: float):
        """
        Enforces maximum weight caps or - if impossible applies equal weight.

        :param <pd.DataFrame> weights: Corresponding weight matrix. Multidimensional.
        :param <float> max_weight: Upper-bound on the weight allowed for each
            cross-section.

        :return <pd.DataFrame>: Will return the modified weight DataFrame.

        N.B.: If the maximum weight is less than the equal weight weight, this replaces
        the computed weight with the equal weight. For instance,
        [np.nan, 0.63, np.nan, np.nan, 0.27] becomes [np.nan, 0.5, np.nan, np.nan, 0.5].
        Otherwise, the function calls the ConvergeRow Class to ensure all weights
        "converge" to a value within the upper-bound. Allow for a margin of error set to
        0.001.
        """

        dfw_wgs = weights.to_numpy()

        for i, row in enumerate(dfw_wgs):
            row = ConvergeRow.application(row, max_weight)
            weights.iloc[i, :] = row

        return weights

    def weight_df_helper(self, weight_meth: str = "equal", weights: List[float] = None,
                         lback_meth: str = "xma", lback_periods: int = 21,
                         max_weight: float = 1.0, remove_zeros: bool = True):
        """
        Subroutine used to compute the weights for the basket of returns. Will
        instantiate the computed weight dataframe on the instance using the name of the
        weighting method as the field's name.

        :param <str> weight_meth:
        :param <List[float]> weights:
        :param <str> lback_meth:
        :param <int> lback_periods:
        :param <float> max_weight:
        :param <bool> remove_zeros:

        """
        assert 0.0 < max_weight <= 1.0
        assert weight_meth in ['equal', 'fixed', 'values', 'inv_values', 'invsd']

        # C. Apply the appropriate weighting method.
        if not self.wgt_flag:
            dfx_ticks_wgt = self.dfx[self.dfx["ticker"].isin(self.tickers)]
        else:
            dfx_ticks_wgt = self.dfx[self.dfx["ticker"].isin(self.ticks_wgt)]

        dfw_ret = dfx_ticks_wgt.pivot(index="real_date", columns="ticker",
                                      values="value")

        if weight_meth == "equal":
            dfw_wgs = self.equal_weight(df_ret=dfw_ret)

        elif weight_meth == "fixed":
            message = "Expects a list of weights."
            message_2 = "List of weights must be equal to the number of contracts."
            assert isinstance(weights, list), message
            assert dfw_ret.shape[1] == len(weights), message_2
            assert all(isinstance(w, (int, float)) for w in weights)

            dfw_wgs = self.fixed_weight(df_ret=dfw_ret, weights=weights)

        elif weight_meth == "invsd":
            error_message = "Two options for the inverse-weighting method are 'ma' and " \
                            "'xma'."
            assert lback_meth in ["xma", "ma"], error_message
            assert isinstance(lback_periods, int), "Expects <int>."
            dfw_wgs = self.inverse_weight(dfw_ret=dfw_ret, lback_meth=lback_meth,
                                          lback_periods=lback_periods,
                                          remove_zeros=remove_zeros)

        elif weight_meth in ["values", "inv_values"]:
            dfw_wgt = dfw_ret.shift(1)  # Lag by one day to be used as weights.
            cols = sorted(dfw_wgt.columns)
            dfw_ret = dfw_wgt.reindex(cols, axis=1)
            dfw_wgt = dfw_wgt.reindex(cols, axis=1)
            dfw_wgs = self.values_weight(dfw_ret, dfw_wgt)

        else:
            raise NotImplementedError(f"Weight method unknown {weight_meth}")

        # D. Remove leading NA rows.

        fvi = max(dfw_wgs.first_valid_index(), dfw_ret.first_valid_index())
        dfw_wgs, dfw_ret = dfw_wgs[fvi:], dfw_ret[fvi:]

        # E. Impose cap on cross-section weight.

        if max_weight < 1.0:
            dfw_wgs = self.max_weight_func(weights=dfw_wgs, max_weight=max_weight)

        self.__dict__[weight_meth] = dfw_wgs

    def weight_dataframe(self, weight_meth: List[str] = "equal",
                         weights: List[float] = None,
                         lback_meth: str = "xma", lback_periods: int = 21,
                         max_weight: float = 1.0, remove_zeros: bool = True):
        """
        Wrapper function concealing the logic of the primary method. The purpose of the
        wrapper function is to handle either a single weight method being received or
        multiple weight methods. The latter will be held inside a List.

        :param <str or List[str]> weight_meth: method used for weighting constituent
            returns and carry. The parameter can receive either a single weight method or
            multiple weighting methods. The options are as follows:
        [1] "equal": all constituents with non-NA returns have the same weight.
            This is the default.
        [2] "fixed": weights are proportionate to single list of values (corresponding to
            contracts) provided passed to argument `weights`.
        [3] "invsd": weights based on inverse to standard deviations of recent returns.
        [4] "values": weights proportionate to a panel of values of exogenous weight
            category.
        [5] "inv_values": weights are inversely proportionate to of values of exogenous
            weight category.
        :param <List[float]> weights: single list of weights corresponding to the base
            tickers in `contracts` argument. This is only relevant for the fixed weight
            method.
        :param <str> lback_meth: lookback method for "invsd" weighting method. Default is
            Exponential MA, "ema". The alternative is simple moving average, "ma".
        :param <int> lback_periods:
        :param <float> max_weight: maximum weight of a single contract. Default is 1, i.e
            zero restrictions. The purpose of the restriction is to limit concentration
            within the basket.
        :param <bool> remove_zeros: removes the zeros. Default is set to True.
        """
        assertion_error = "String or List expected."
        assert isinstance(weight_meth, (list, str)), assertion_error
        if isinstance(weight_meth, str):
            self.weight_df_helper(weight_meth, weights, lback_meth, lback_periods,
                                  max_weight, remove_zeros)
        else:
            for method in weight_meth:
                self.weight_df_helper(method, weights, lback_meth, lback_periods,
                                      max_weight, remove_zeros)

    @staticmethod
    def carry_handler(dfw_wgs: pd.DataFrame, dfw_cry: List[pd.DataFrame], cry: List[str],
                      no_cry: int, name: str):
        """
        Method designed to handle multiple carries. If multiple carries are defined, each
        output dataframe will be stored in a List.

        :param <pd.DataFrame> dfw_wgs:
        :param <List[pd.DataFrame> dfw_cry:
        :param <str> cry:
        :param <int> no_cry:
        :param <str> name: postfix for the basket plus the respective weighting method.

        return <List[pd.DataFrame]>: list of the basket carry returns.
        """

        cry_list = []
        select = ["ticker", "real_date", "value"]
        if no_cry == 1:
            dfw_cry = [dfw_cry]
            cry = [cry]

        for i, df_c in enumerate(dfw_cry):
            dfcry = df_c.multiply(dfw_wgs).sum(axis=1)
            dfcry = dfcry.to_frame("value").reset_index()
            dfcry = dfcry.assign(ticker=name + cry[i])[select]
            cry_list.append(dfcry)

        return cry_list

    @staticmethod
    def bp_helper(weight_meth: str, dfw_wgs: pd.DataFrame, dfw_ret: pd.DataFrame,
                  dfw_cry: List[pd.DataFrame], ret: str = "XR_NSA", cry: str = None,
                  cry_flag: bool = False, no_cry: int = 0, basket_tik: str = "GLB_ALL",
                  return_weights: bool = False):

        """
        Helper function hosting the source code. The parameter, "weight_meth", will
        delimit the weighting method used to compute the basket returns, as each
        potential weight dataframe will be held as a field on the instance.

        :param <str> weight_meth:
        :param <pd.DataFrame> dfw_wgs: weight dataframe used in computation.
        :param <pd.DataFrame> dfw_ret: return dataframe.
        :param <List[pd.DataFrame]> dfw_cry: potentially a list of carry dataframes, or
            a single pivoted dataframe.
        :param <str> ret:
        :param <str> cry:
        :param <bool> cry_flag: carry flag.
        :param <int> no_cry: number of carries defined. Default is set to zero reflecting
        :param <str> basket_tik:
        :param <bool> return_weights:

        :return <pd.Dataframe>:
        """

        # F. Calculate and store weighted average returns.
        select = ["ticker", "real_date", "value"]
        dfxr = dfw_ret.multiply(dfw_wgs).sum(axis=1)
        dfxr = dfxr.to_frame("value").reset_index()
        name = basket_tik + "_" + weight_meth.upper() + "_"
        dfxr = dfxr.assign(ticker=name + ret)[select]
        store = [dfxr]

        if cry_flag:
            cry_list = Basket.carry_handler(dfw_wgs, dfw_cry, cry, no_cry, name)
            store += cry_list
        if return_weights:
            dfw_wgs.columns.name = "cid"
            w = dfw_wgs.stack().to_frame("value").reset_index()

            func = lambda c: c[:-len(ret)] + "WGT_" + weight_meth
            contracts_ = list(map(func, w["cid"].to_numpy()))
            contracts_ = np.array(contracts_)
            w["ticker"] = contracts_
            w = w.sort_values(['ticker', 'real_date'])[['ticker', 'real_date', 'value']]
            w = w.loc[w.value > 0, select]
            store.append(w)

        # Concatenate along the date index, and subsequently drop to restore natural
        # index.
        df = pd.concat(store, axis=0, ignore_index=True)

        return df

    def make_basket(self, weight_meth: str, basket_name: str = "GLB_ALL"):
        """
        Calculates all basket performance categories

        :param <str> weight_meth: one of the following # Todo: options
        :param <str> basket_name: name of basket base ticker (analogous to contract name)
            to be used for return and (possibly) carry are calculated. Default is
            "GLB_ALL".
        """

        assert isinstance(weight_meth, str), "`weight_meth` must be string"
        # Todo: should just make/store single method's performance data and weights

        if isinstance(weight_meth, str):
            dfw_wgs = self.__dict__[weight_meth]  # Todo: ??
            df = self.bp_helper(weight_meth, dfw_wgs, self.dfw_ret, self.dfws_cry,
                                ret=self.ret, cry=self.cry, cry_flag=self.cry_flag,
                                no_cry=self.no_cry, basket_name=basket_name,
                                return_weights=return_weights)
        else:  # Todo: not needed
            df = []
            for w in weight_meth:
                assertion_error = f"The method, weight_dataframe(), must be called, " \
                                  f"using the associated weight method {w}, prior to " \
                                  "basket_performance()."
                fields = self.__dict__.keys()
                assert w in fields, assertion_error
                dfw_wgs = self.__dict__[w]
                key = "basket_" + w
                if key not in fields:
                    df_w = self.bp_helper(w, dfw_wgs, self.dfw_ret, self.dfws_cry,
                                          self.ret, self.cry, self.cry_flag,
                                          self.no_cry, basket_name, return_weights)
                    self.__dict__[key] = df_w

                else:
                    df_w = self.__dict__[key]

                df.append(df_w)
            df = pd.concat(df)

        return df

    def return_basket(self, basket_names: List[str]):
        """
        Return standardized dataframe with one or more basket performance data,

        :param basket_names: single basket name or list for which performance data
            are to be returned.

        :return <pd.Dataframe>: standardized DataFrame with the basket return and
            (possibly) carry data in standard form, i.e. columns 'cid', 'xcats',
            'real_date' and 'value'.
        """
        # Todo: simple function, no more than 3 lines
        pass

    def return_weights(self, basket_name: str):
        """
        Return standardized dataframe with one or more basket performance data,

        :param basket_name: single basket name or list for which performance data
            are to be returned.

        :return <pd.Dataframe>: standardized DataFrame with basket weights.
        """
        # Todo: simple function, no more than 3 lines
        pass


if __name__ == "__main__":

    cids = ['AUD', 'GBP', 'NZD', 'USD']
    xcats = ['FXXR_NSA', 'FXCRY_NSA', 'FXCRR_NSA', 'EQXR_NSA', 'EQCRY_NSA', 'EQCRR_NSA']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0, 1]
    df_cids.loc['GBP'] = ['2011-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2012-01-01', '2020-12-31', 0, 3]
    df_cids.loc['USD'] = ['2010-01-01', '2020-12-31', 0, 4]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['FXXR_NSA'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.2]
    df_xcats.loc['FXCRY_NSA'] = ['2010-01-01', '2020-12-31', 1, 1, 0.9, 0.2]
    df_xcats.loc['FXCRR_NSA'] = ['2010-01-01', '2020-12-31', 0.5, 0.8, 0.9, 0.2]
    df_xcats.loc['EQXR_NSA'] = ['2012-01-01', '2020-12-31', 0.5, 2, 0, 0.2]
    df_xcats.loc['EQCRY_NSA'] = ['2010-01-01', '2020-12-31', 2, 1.5, 0.9, 0.5]
    df_xcats.loc['EQCRR_NSA'] = ['2010-01-01', '2020-12-31', 1.5, 1.5, 0.9, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}
    contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']
    gdp_figures = [17.0, 17.0, 41.0, 9.0, 250.0]

    contracts_1 = ['AUD_FX', 'GBP_FX', 'NZD_FX', 'USD_EQ']
    basket_1 = Basket(dfd, contracts=contracts_1,
                      ret='XR_NSA', cry=['CRY_NSA', 'CRR_NSA'])
    df = basket_1.make_basket(weight_meth='equal', basket_name='GLB_EQUAL')
