
import numpy as np
import pandas as pd
from typing import List
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.shape_dfs import reduce_df_by_ticker
from macrosynergy.panel.converge_row import ConvergeRow

class Basket(object):

    def __init__(self, df: pd.DataFrame, contracts: List[str], ret: str = "XR_NSA",
                 cry: str = None, start: str = None, end: str = None,
                 blacklist: dict = None, wgt: str = None):

        """
        Class' Constructor. The Class' purpose is to calculate the returns and carries of
        baskets of financial contracts using various weight methods.

        :param <pd.Dataframe> df: standardized DataFrame with following columns: 'cid',
            'xcats', 'real_date' and 'value'.
        :param <List[str]> contracts: base tickers (combinations of cross-sections and
            base categories) denoting contracts that go into the basket.
        :param <str> ret: return category postfix; default is "XR_NSA".
        :param <str> cry: carry category postfix; default is None.
        :param <str> start: earliest date in ISO 8601 format. Default is None.
        :param <str> end: latest date in ISO 8601 format. Default is None.
        :param <dict> blacklist: cross-sections with date ranges that should be excluded
            from the dataframe. If one cross-section has several blacklist periods append
            numbers to the cross-section code.
        :param <str> wgt: postfix used to identify exogenous weight category. Analogously
            to carry and return postfixes this should be added to base tickers to
            identify the values that denote contract weights. Only applicable for the
            weight methods 'values' or 'inv_values'.

        """

        assert isinstance(contracts, list)
        error = "contracts must be string list."
        assert all(isinstance(c, str) for c in contracts), error
        assert isinstance(ret, str), "return category must be a <str>."
        if cry is not None:
            assert isinstance(cry, str), "carry category must be a <str>."

        self.tickers = self.ticker_list(contracts, ret, cry, wgt)
        self.dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=self.tickers,
                                       blacklist=blacklist)
        self.contract = contracts
        self.ret = ret
        self.cry = cry
        self.start = self.date_check(start)
        self.end = self.date_check(end)
        self.cry_flag = (cry is not None)
        self.wgt_flag = (wgt is not None)
        self.dfw_ret = self.pivot_dataframe(self.ticks_ret)
        if self.cry_flag:
            self.dfw_cry = self.pivot_dateframe(self.ticks_cry)
        else:
            self.dfw_cry = None
        self.wgt = wgt

    def pivot_dataframe(self, tick_list):
        dfx_ticks_list = self.dfx[self.dfx["ticker"].isin(tick_list)]
        dfw = dfx_ticks_list.pivot(index="real_date", columns="ticker", values="value")
        return dfw

    @staticmethod
    def date_check(date_string):
        date_error = "Expected form of string: '%Y-%m-%d'."
        if date_string is not None:
            try:
                pd.Timestamp(date_string).strftime("%Y-%m-%d")
            except ValueError:
                raise AssertionError(date_error)

    def ticker_list(self, contracts: List[str], ret: str, cry: str, wgt: str):
        """
        Method used to establish the list of tickers involved in the computation. The
        list will potentially consist of the return categories, the carry categories and
        an exogenous weight category used for the weight dataframe.

        :param <List[str]> contracts:
        :param <str> ret:
        :param <str> cry:
        :param <str> wgt:

        :return <List[str]>: list of tickers.
        """

        ticks_ret = [c + ret for c in contracts]
        self.__dict__['ticks_ret'] = ticks_ret
        tickers = ticks_ret.copy()

        # Boolean for carry being used.
        if cry is not None:
            ticks_cry = [c + cry for c in contracts]
            self.__dict__['ticks_cry'] = ticks_cry
            tickers += ticks_cry

        if wgt is not None:
            error = f"'wgt' must be a string. Received: {type(wgt)}."
            assert isinstance(wgt, str), error
            ticks_wgt = [c + wgt for c in contracts]
            self.__dict__['ticks_wgt'] = ticks_wgt
            tickers += ticks_wgt

        return tickers

    @staticmethod
    def check_weights(weight: pd.DataFrame):
        """
        Validates that the weights computed on each timestamp sum to one accounting for
        floating point error.

        :param <pd.DataFrame> weight: weight dataframe.
        """
        check = weight.sum(axis=1)
        c = ~((abs(check - 1) < 1e-6) | (abs(check) < 1e-6))
        assert not any(c), f"weights must sum to one (or zero), not: {check[c]}"

    def equal_weight(self, df_ret: pd.DataFrame) -> pd.DataFrame:
        """
        Equal weight function: receives the pivoted return DataFrame and determines the
        number of non-NA cross-sections per timestamp, and subsequently distribute the
        weight evenly across non-NA cross-sections.

        :param <pd.DataFrame> df_ret: data-frame with returns.

        :return <pd.DataFrame>: dataframe of weights.
        """

        act_cross = (~df_ret.isnull())
        uniform = (1 / act_cross.sum(axis=1)).values
        uniform = uniform[:, np.newaxis]
        # Apply equal value to all cross sections.
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

    def weight_dataframe(self, weight_meth: str or List[str] = "equal",
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
    def bp_helper(weight_meth: str, dfw_wgs: pd.DataFrame, dfw_ret: pd.DataFrame,
                  dfw_cry: pd.DataFrame, ret: str = "XR_NSA", cry: str = None,
                  cry_flag: bool = False, basket_tik: str = "GLB_ALL",
                  return_weights: bool = False):

        """
        Helper function hosting the source code. The parameter, "weight_meth", will
        delimit the weighting method used to compute the basket returns, as each
        potential weight dataframe will be held as a field on the instance.

        :param <str> weight_meth:
        :param <pd.DataFrame> dfw_wgs: weight dataframe used in computation.
        :param <pd.DataFrame> dfw_ret: return dataframe.
        :param <pd.DataFrame> dfw_cry: carry dataframe.
        :param <str> ret:
        :param <str> cry:
        :param <bool> cry_flag: carry flag.
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
            dfcry = dfw_cry.multiply(dfw_wgs).sum(axis=1)
            dfcry = dfcry.to_frame("value").reset_index()
            dfcry = dfcry.assign(ticker=name + cry)[select]
            store.append(dfcry)
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

    def basket_performance(self, weight_meth: str or List[str],
                           basket_tik: str = "GLB_ALL",
                           return_weights: bool = False):
        """
        Produces a full dataframe of all basket performance categories (inclusive of both
        returns and carries). If more than a single weighting method is passed, the
        function will return the corresponding number of basket performance dataframes
        which are concatenated into a single dataframe.

        :param <List[str]> weight_meth:
        :param <str> basket_tik: name of basket base ticker (analogous to contract name)
            to be used for return and (possibly) carry are calculated. Default is
            "GLB_ALL".
        :param <bool> return_weights: if True add cross-section weights to output
            dataframe with 'WGT' postfix. Default is False.

        :return <pd.Dataframe>: standardized DataFrame with the basket return and
            (possibly) carry data in standard form, i.e. columns 'cid', 'xcats',
            'real_date' and 'value'.
        """

        assertion_error = "String or List expected."
        assert isinstance(weight_meth, (list, str)), assertion_error

        if isinstance(weight_meth, str):
            dfw_wgs = self.__dict__[weight_meth]
            df = self.bp_helper(weight_meth, dfw_wgs, self.dfw_ret, self.dfw_cry,
                                ret=self.ret, cry=self.cry, cry_flag=self.cry_flag,
                                basket_tik=basket_tik, return_weights=return_weights)
        else:
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
                    df_w = self.bp_helper(w, dfw_wgs, self.dfw_ret, self.dfw_cry,
                                          self.ret, self.cry, self.cry_flag, basket_tik,
                                          return_weights=return_weights)
                    self.__dict__[key] = df_w

                else:
                    df_w = self.__dict__[key]

                df.append(df_w)
            df = pd.concat(df)

        return df
