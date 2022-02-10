
import numpy as np
import pandas as pd
import random
from typing import List
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.shape_dfs import reduce_df_by_ticker
from macrosynergy.panel.converge_row import ConvergeRow

class Basket(object):

    def __init__(self, df: pd.DataFrame, contracts: List[str], ret: str = "XR_NSA",
                 weight_meth: str = 'equal', cry: str = None,
                 start: str = None, end: str = None, blacklist: dict = None,
                 wgt: str = None):

        """
        Class' Constructor.

        :param <pd.Dataframe> df: standardized DataFrame with following columns: 'cid',
            'xcats', 'real_date' and 'value'.
        :param <List[str]> contracts: base tickers (combinations of cross-sections and
            base categories) denoting contracts that go into the basket.
        :param <str> ret: return category postfix; default is "XR_NSA".
        :param <str> weight_meth: method used for weighting constituent returns and carry.
            Options are as follows:
        [1] "equal": all constituents with non-NA returns have the same weight.
            This is the default.
        [2] "fixed": weights are proportionate to single list of values (corresponding to
            contracts) provided passed to argument `weights`.
        [3] "invsd": weights based on inverse to standard deviations of recent returns.
        [4] "values": weights proportionate to a panel of values of exogenous weight
            category.
        [5] "inv_values": weights are inversely proportionate to of values of exogenous
            weight category.
        :param <str> cry: carry category postfix; default is None.
        :param <str> start: earliest date in ISO 8601 format. Default is None.
        :param <str> end: latest date in ISO 8601 format. Default is None.
        :param <dict> blacklist: cross-sections with date ranges that should be excluded
            from the dataframe. If one cross-section has several blacklist periods append
            numbers to the cross-section code.

        """

        assert isinstance(contracts, list)
        error = "contracts must be string list."
        assert all(isinstance(c, str) for c in contracts), error
        assert isinstance(ret, str), "return category must be a <str>."
        if cry is not None:
            assert isinstance(cry, str), "carry category must be a <str>."

        self.dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=self.tickers,
                                       blacklist=blacklist)
        self.contract = contracts
        self.tickers = self.ticker_list(self)
        self.ret = ret
        self.weight_meth = weight_meth
        self.cry = cry
        self.start = self.date_check(start)
        self.end = self.date_check(end)
        self.cry_flag = (cry is not None)
        self.wgt_flag = (wgt is not None) and (weight_meth in ["values", "inv_values"])
        self.dfw_ret = self.pivot_dateframe(self.ticks_ret)
        if self.cry_flag:
            self.dfw_cry = self.pivot_dateframe(self.ticks_cry)
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

    def ticker_list(self):

        ticks_ret = [c + self.ret for c in self.contracts]
        self.__dict__['ticks_ret'] == ticks_ret
        tickers = ticks_ret.copy()

        # Boolean for carry being used.
        if self.cry_flag:
            ticks_cry = [c + self.cry for c in self.contracts]
            self.__dict__['ticks_cry'] = ticks_cry
            tickers += ticks_cry

        if self.wgt_flag:
            error = f"'wgt' must be a string. Received: {type(self.wgt)}."
            assert isinstance(self.wgt, str), error
            ticks_wgt = [c + self.wgt for c in self.contracts]
            self.__dict__['ticks_wgt'] = ticks_wgt
            tickers += ticks_wgt

        return tickers

    @staticmethod
    def check_weights(weight: pd.DataFrame):
        check = weight.sum(axis=1)
        c = ~((abs(check - 1) < 1e-6) | (abs(check) < 1e-6))
        assert not any(c), f"weights must sum to one (or zero), not: {check[c]}"

    def equal_weight(self, df_ret: pd.DataFrame) -> pd.DataFrame:
        """
        Equal weight function: receives the pivoted return DataFrame and determines the
        number of non-NA cross-sections per timestamp, and subsequently distribute the
        weight evenly across non-NA cross-sections.

        :param <pd.DataFrame> df_ret: data-frame with returns.

        :return <pd.DataFrame> : dataframe of weights.
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
        weight_arr = weight.to_numpy()  # convert df to np array
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

    def values_weight(self, dfw_ret: pd.DataFrame, dfw_wgt: pd.DataFrame):
        """
        Returns weights based on an external weighting category.

        :param <pd.DataFrame> dfw_ret: Standard wide dataframe of returns across time and
            contracts.
        :param <pd.DataFrame> dfw_wgt: Standard wide dataframe of weight category values
            across time and contracts.

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

        if self.weight_meth != "values":
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

    def weight_dateframe(self, weights: List[float] = None, lback_meth: str = "xma",
                         lback_periods: int = 21, max_weight: float = 1.0,
                         remove_zeros: bool = True):
        """
        Subroutine used to compute the weights for the basket of returns.

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

        :return <pd.DataFrame>: Will return the weight DataFrame.
        """
        assert 0.0 < max_weight <= 1.0
        weight_meth = self.weight_meth
        assert self.weight_meth in ['equal', 'fixed', 'values', 'inv_values', 'invsd']

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

        self.__dict__['dfw_wgs'] = dfw_wgs
        return dfw_wgs

    def basket_performance(self, dfw_wgs: pd.DataFrame, basket_tik: str = "GLB_ALL",
                           return_weights: bool = False):

        """
        Basket performance returns a single approximate return and - optionally - carry
        series for the basket of underlying contracts.

        :param <pd.DataFrame> dfw_wgs: Weight DataFrame for the basket's constituents.
        :param <str> basket_tik: name of basket base ticker (analogous to contract name)
            to be used for return and (possibly) carry are calculated. Default is
            "GLB_ALL".
        :param <bool> return_weights: if True add cross-section weights to output
            dataframe with 'WGT' postfix. Default is False.

        :return <pd.Dataframe>: standardized DataFrame with the basket return and
            (possibly) carry data in standard form, i.e. columns 'cid', 'xcats',
            'real_date' and 'value'.
        """

        # F. Calculate and store weighted average returns.
        select = ["ticker", "real_date", "value"]
        dfxr = self.dfw_ret.multiply(dfw_wgs).sum(axis=1)
        dfxr = dfxr.to_frame("value").reset_index()
        dfxr = dfxr.assign(ticker=basket_tik + "_" + self.ret)[select]
        store = [dfxr]

        if self.cry_flag:
            dfcry = self.dfw_cry.multiply(dfw_wgs).sum(axis=1)
            dfcry = dfcry.to_frame("value").reset_index()
            dfcry = dfcry.assign(ticker=basket_tik + "_" + self.cry)[select]
            store.append(dfcry)
        if return_weights:
            dfw_wgs.columns.name = "cid"
            w = dfw_wgs.stack().to_frame("value").reset_index()
            func = lambda c: c[:-len(self.ret)] + "WGT"
            contracts_ = list(map(func, w["cid"].to_numpy()))
            contracts_ = np.array(contracts_)
            w["ticker"] = contracts_
            w = w.loc[w.value > 0, select]
            store.append(w)

        # Concatenate along the date index, and subsequently drop to restore natural
        # index.
        df = pd.concat(store, axis=0, ignore_index=True)
        return df