
import numpy as np
import pandas as pd
import random
from typing import List
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.shape_dfs import reduce_df_by_ticker
from macrosynergy.panel.converge_row import ConvergeRow
from macrosynergy.management.simulate_quantamental_data import make_qdf

class Basket(object):

    def __init__(self, df:pd.DataFrame, contracts: List[str], ret: str = "XR_NSA",
                 weight_meth: str = 'equal', cry: str = None,
                 start: str = None, end: str = None, blacklist: dict = None,
                 wgt: str = None, basket_tik: str = "GLB_ALL"):

        self.dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=self.tickers,
                                       blacklist=blacklist)
        self.contract = contracts
        self.tickers = self.ticker_list(self)
        self.ret = ret
        self.cry = cry
        self.cry_flag = (cry is not None)
        self.wgt_flag = (wgt is not None) and (weight_meth in ["values", "inv_values"])
        self.wgt = wgt
        self.basket_tik = basket_tik

    def ticker_list(self):

        ticks_ret = [c + self.ret for c in self.contracts]
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
        number of non-NA cross-sections per timestamp, and subsequently distribute the weight
        evenly across non-NA cross-sections.

        :param <pd.DataFrame> df_ret: data-frame with returns.

        :return <pd.DataFrame> : dataframe of weights.
        """

        act_cross = (~df_ret.isnull())  # df with 1s/0s for non-NA/NA returns
        uniform = (1 / act_cross.sum(axis=1)).values  # single equal value
        uniform = uniform[:, np.newaxis]
        # Apply equal value to all cross sections
        broadcast = np.repeat(uniform, df_ret.shape[1], axis=1)

        weight = act_cross.multiply(broadcast)
        self.check_weights(weight=weight)

        return weight

    def fixed_weight(self, df_ret: pd.DataFrame, weights: List[float]):
        """
        Calculates fixed weights based on a single list of values and a corresponding return
        panel dataframe.

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
        :param <Bool> remove_zeros: Any returns that are exact zeros will not be included in
            the lookback window and prior non-zero values are added to the window instead.

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
        :param <str> weight_meth: Weighting method. must be one of "values" or "inv_values".

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
        :param <float> max_weight: Upper-bound on the weight allowed for each cross-section.

        :return <pd.DataFrame>: Will return the modified weight DataFrame.

        N.B.: If the maximum weight is less than the equal weight weight, this replaces the
        computed weight with the equal weight. For instance,
        [np.nan, 0.63, np.nan, np.nan, 0.27] becomes [np.nan, 0.5, np.nan, np.nan, 0.5].
        Otherwise, the function calls the ConvergeRow Class to ensure all weights "converge"
        to a value within the upper-bound. Allow for a margin of error set to 0.001.
        """

        dfw_wgs = weights.to_numpy()

        for i, row in enumerate(dfw_wgs):
            row = ConvergeRow.application(row, max_weight)
            weights.iloc[i, :] = row

        return weights

    def weight_dateframe(self, weight_meth: str, weights: List[float],
                         lback_meth: str = "xma", lback_periods: int = 21,
                         max_weight: float = 1.0, remove_zeros: bool = True):

        """
        Subroutine used to compute the weights for the basket of returns.

        :param <str> weight_meth:
        :param <List[float]> weights:
        :param <str> lback_meth:
        :param <int> lback_periods:
        :param <float> max_weight:
        :param <bool> remove_zeros:

        :return <pd.DataFrame>: Will return the weight DataFrame.
        """

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
            for w in weights:
                try:
                    int(w)
                except ValueError:
                    print(f"List, {weights}, must be all numerical values.")
                    raise

            dfw_wgs = self.fixed_weight(df_ret=dfw_ret, weights=weights)

        elif weight_meth == "invsd":
            dfw_wgs = self.inverse_weight(dfw_ret=dfw_ret, lback_meth=lback_meth,
                                          lback_periods=lback_periods,
                                          remove_zeros=remove_zeros)

        elif weight_meth in ["values", "inv_values"]:
            dfw_wgt = dfw_ret.shift(1)  # Lag by one day to be used as weights.
            cols = sorted(dfw_wgt.columns)
            dfw_ret = dfw_wgt.reindex(cols, axis=1)
            dfw_wgt = dfw_wgt.reindex(cols, axis=1)
            dfw_wgs = self.values_weight(dfw_ret, dfw_wgt, weight_meth)

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