
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
        various weighting methods. Each instance of the Class will have an associated
        standardised dataframe, and the categories it will be defined over are the
        return and carry categories, and external weights if required. Therefore, if
        additional return or carry categories are required, a new instance will have to
        to be instantiated by the user.

        :param <pd.Dataframe> df: standardized DataFrame with following columns: 'cid',
            'xcat', 'real_date' and 'value'.
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
        assert isinstance(ret, str), "`ret`must be a string"

        self.contracts = contracts
        self.ret = ret
        self.ticks_ret = [con + ret for con in contracts]
        self.dfw_ret = self.pivot_dataframe(df, self.ticks_ret)
        self.dfws_cry = self.category_handler(df, cry, "cry")
        self.dfws_wgt = self.category_handler(df, ewgts, "wgt")
        self.tickers = self.ticks_ret + self.ticks_cry + self.ticks_wgt
        self.start = self.date_check(start)
        self.end = self.date_check(end)
        self.dfx = reduce_df_by_ticker(df, start=start, end=end, ticks=self.tickers,
                                       blacklist=blacklist)

        self.dict_retcry = {}  # dictionary for collecting basket return/carry dfs.
        self.dict_wgs = {}  # dictionary for collecting basket return/carry dfs.

    def category_handler(self, df: pd.DataFrame, category: List[str], cat_name: str):
        """
        Handles for multiple carries or external weights. Each category will be stored in
        a dictionary where the key will be the associated postfix and the value will be
        the respective wide dataframe.

        :param <pd.DataFrame> df: original, standardised dataframe.
        :param <List[str]> category: carry category postfix.
        :param <str> cat_name: associated name of the category: carry, "cry", or external
            weight, "wgt".

        :param <dict> dfws_cry:
        """

        category_flag = category is not None
        self.__dict__[cat_name + "_flag"] = category_flag
        if category_flag:
            error = "`cry` must be a <str> or a <List[str]>."
            assert isinstance(category, (list, str)), error
            category = [category] if isinstance(category, str) else category

            self.__dict__[cat_name] = category
            dfws_category = {}
            for cat in category:
                ticks = [con + cat for con in self.contracts]
                self.__dict__["ticks_" + cat_name] = ticks
                dfws_category[cat] = self.pivot_dataframe(df, ticks)
        else:
            dfws_category = None
            self.__dict__["ticks_" + cat_name] = []

        return dfws_category

    @staticmethod
    def pivot_dataframe(df, tick_list):
        """
        Reduces the standardised dataframe to include a subset of the possible tickers
        and, subsequently returns a wide dataframe: each column corresponds to a ticker.

        :param <List[str]> tick_list: list of the respective tickers.
        :param <pd.DataFrame> df: standardised dataframe.

        :return <pd.DataFrame> dfw: wide dataframe.
        """

        df['ticker'] = df['cid'] + '_' + df['xcat']
        dfx = df[df["ticker"].isin(tick_list)]
        dfw = dfx.pivot(index="real_date", columns="ticker", values="value")
        return dfw

    @staticmethod
    def date_check(date_string):
        """
        Validates that the dates passed are valid timestamp expressions and will convert
        to the required form '%Y-%m-%d'. Will raise an assertion if not in the expected
        form.

        :param <str> date_string: valid date expression. For instance, "1st January,
            2000."
        """
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

    def make_weights(self, weight_meth: str = "equal", weights: List[float] = None,
                     lback_meth: str = "xma", lback_periods: int = 21,
                     ewgt: str = None, max_weight: float = 1.0,
                     remove_zeros: bool = True):
        """
        Returns wide dataframe of weights to be used for basket series. The method can be
        called independently on the instance and will return a wide dataframe but the
        Class's expected engagement is to call the method return_weights() to receive the
        corresponding standardised weight dataframe. The method should only be called
        from the scope of make_basket().

        :param <str> weight_meth:
        :param <List[float]> weights:
        :param <str> lback_meth:
        :param <int> lback_periods:
        :param <str> ewgt:
        :param <float> max_weight:
        :param <bool> remove_zeros:

        return: <pd.DataFrame>: wide dataframe of contract weights across time.
        """
        assert 0.0 < max_weight <= 1.0
        assert weight_meth in ['equal', 'fixed', 'values', 'inv_values', 'invsd']

        # Apply weight method.

        if weight_meth == "equal":
            dfw_wgs = self.equal_weight(df_ret=self.dfw_ret)

        elif weight_meth == "fixed":
            message = "Expects a list of weights."
            message_2 = "List of weights must be equal to the number of contracts."
            assert isinstance(weights, list), message
            assert self.dfw_ret.shape[1] == len(weights), message_2
            assert all(isinstance(w, (int, float)) for w in weights)

            dfw_wgs = self.fixed_weight(df_ret=self.dfw_ret, weights=weights)

        elif weight_meth == "invsd":
            error_message = "Two options for the inverse-weighting method are 'ma' and " \
                            "'xma'."
            assert lback_meth in ["xma", "ma"], error_message
            assert isinstance(lback_periods, int), "Expects <int>."
            dfw_wgs = self.inverse_weight(dfw_ret=self.dfw_ret, lback_meth=lback_meth,
                                          lback_periods=lback_periods,
                                          remove_zeros=remove_zeros)

        elif weight_meth in ["values", "inv_values"]:
            assert ewgt in self.wgt, f'{ewgt} is not defined on the instance.'
            # Lag by one day to be used as weights.
            dfw_wgt = self.dfws_wgt[ewgt].shift(1)
            cols = sorted(dfw_wgt.columns)
            dfw_ret = dfw_wgt.reindex(cols, axis=1)
            dfw_wgt = dfw_wgt.reindex(cols, axis=1)
            dfw_wgs = self.values_weight(dfw_ret, dfw_wgt, weight_meth)

        else:
            raise NotImplementedError(f"Weight method unknown {weight_meth}")

        # Remove leading NA rows.

        fvi = max(dfw_wgs.first_valid_index(), self.dfw_ret.first_valid_index())
        dfw_wgs = dfw_wgs[fvi:]

        # Impose cap on cross-section weight.

        if max_weight < 1.0:
            dfw_wgs = self.max_weight_func(weights=dfw_wgs, max_weight=max_weight)

        return dfw_wgs

    def make_basket(self, weight_meth: str = "equal", weights: List[float] = None,
                    lback_meth: str = "xma", lback_periods: int = 21,
                    ewgt: str = None, max_weight: float = 1.0, remove_zeros: bool = True,
                    basket_name: str = "GLB_ALL"):
        """
        Calculates all basket performance categories.

        :param <str> weight_meth: method used for weighting constituent returns and
            carry. The parameter can receive either a single weight method or
            multiple weighting methods. The options are as follows:
            [1] "equal": all constituents with non-NA returns have the same weight.
                This is the default.
            [2] "fixed": weights are proportionate to a single list of values provided
                which are passed to argument `weights` (each value corresponds to a
                single contract).
            [3] "invsd": weights based on inverse to standard deviations of recent
                returns.
            [4] "values": weights proportionate to a panel of values of exogenous weight
                category.
            [5] "inv_values": weights are inversely proportionate to of values of exogenous
                weight category.
        :param <List[float]> weights: single list of weights corresponding to the base
            tickers in `contracts` argument. This is only relevant for the fixed weight
            method.
        :param <str> lback_meth: look-back method for "invsd" weighting method. Default
            is Exponential MA, "ema". The alternative is simple moving average, "ma".
        :param <int> lback_periods: look-back periods for "invsd" weighting method.
            Default is 21.  Half-time for "xma" and full lookback period for "ma".
        :param <str> ewgt: Exogenous weight postfix that defines the weight value panel.
            Only needed for the 'values' or 'inv_values' method.
        :param <float> max_weight: maximum weight of a single contract. Default is 1, i.e
            zero restrictions. The purpose of the restriction is to limit concentration
            within the basket.
        :param <bool> remove_zeros: removes the zeros. Default is set to True.
        :param <str> basket_name: name of basket base ticker (analogous to contract name)
            to be used for return and (possibly) carry are calculated. Default is
            "GLB_ALL".
        """

        assert isinstance(weight_meth, str), "`weight_meth` must be string"

        dfw_wgs = self.make_weights(weight_meth=weight_meth, weights=weights,
                                    lback_meth=lback_meth, lback_periods=lback_periods,
                                    ewgt=ewgt, max_weight=max_weight,
                                    remove_zeros=remove_zeros)

        select = ["ticker", "real_date", "value"]
        dfw_bret = self.dfw_ret.multiply(dfw_wgs).sum(axis=1)
        dfxr = dfw_bret.to_frame("value").reset_index()
        basket_ret = basket_name + "_" + self.ret
        dfxr = dfxr.assign(ticker=basket_ret)[select]
        store = [dfxr]

        if self.cry_flag:
            cry_list = []
            for cr in self.cry:
                dfw_bcry = self.dfws_cry[cr].multiply(dfw_wgs).sum(axis=1)
                dfcry = dfw_bcry.to_frame("value").reset_index()
                basket_cry = basket_name + "_" + cr
                dfcry = dfcry.assign(ticker=basket_cry)[select]
                cry_list.append(dfcry)

            store += cry_list

        df_retcry = pd.concat(store)
        self.dict_retcry[basket_name] = df_retcry
        self.dict_wgs[basket_name] = dfw_wgs

    def return_basket(self, basket_names: List[str]):
        """
        Return standardized dataframe with one or more basket performance data. Various
        baskets can be computed on the same instance using different weighting methods.
        However, the return type or carry type, for each basket dataframe, will be the
        same. The performance data will only change from using the different weighting
        method.

        :param <List[str]> basket_names: single basket name or list for which performance
            data are to be returned.

        :return <pd.Dataframe>: standardized DataFrame with the basket return and
            (possibly) carry data in standard form, i.e. columns 'cid', 'xcats',
            'real_date' and 'value'.
        """
        basket_error = "String or List of basket names expected."
        assert isinstance(basket_error, (list, str)), basket_error
        if isinstance(basket_names, str):
            basket_names = [basket_names]

        ret_baskets = []
        for b in basket_names:
            try:
                dfw_retcry = self.dict_retcry[b]
            except KeyError as e:
                print(f"Incorrect basket name, {e}.")
            else:
                ret_baskets.append(dfw_retcry)

        return_df = pd.concat(ret_baskets)
        return return_df.reset_index(drop=True)

    def return_weights(self, basket_names: List[str]):
        """
        Return the standardised dataframe containing the corresponding weights used to
        compute the basket.

        :param basket_names: single basket name or list for which performance data
            are to be returned.

        :return <pd.Dataframe>: standardized DataFrame with basket weights.
        """
        basket_error = "String or List of basket names expected."
        assert isinstance(basket_error, (list, str)), basket_error
        if isinstance(basket_names, str):
            basket_names = [basket_names]

        weight_baskets = []
        select = ['ticker', 'real_date', 'value']
        for b in basket_names:
            try:
                dfw_wgs = self.dict_wgs[b]
            except KeyError as e:
                print(f"Incorrect basket name, {e}.")
            else:
                dfw_wgs.columns.name = "cid"
                w = dfw_wgs.stack().to_frame("value").reset_index()
                w = w.sort_values(['real_date'])
                w = w.rename(columns={'cid': 'ticker'})
                w = w[select]
                w = w.loc[w.value > 0, select]
                weight_baskets.append(w)

        return_df = pd.concat(weight_baskets)
        return return_df.reset_index(drop=True)


if __name__ == "__main__":

    cids = ['AUD', 'GBP', 'NZD', 'USD']
    xcats = ['FXXR_NSA', 'FXCRY_NSA', 'FXCRR_NSA', 'EQXR_NSA', 'EQCRY_NSA', 'EQCRR_NSA',
             'FXWBASE_NSA', 'EQWBASE_NSA']

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
    df_xcats.loc['FXWBASE_NSA'] = ['2010-01-01', '2020-12-31', 1, 1.5, 0.8, 0.5]
    df_xcats.loc['EQWBASE_NSA'] = ['2010-01-01', '2020-12-31', 1, 1.5, 0.9, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}
    contracts = ['AUD_FX', 'AUD_EQ', 'NZD_FX', 'GBP_EQ', 'USD_EQ']
    gdp_figures = [17.0, 17.0, 41.0, 9.0, 250.0]

    contracts_1 = ['AUD_FX', 'GBP_FX', 'NZD_FX', 'USD_EQ']

    # First test. Multiple carries. Equal weight method.
    # The main aspect to check in the code is that basket performance has been applied to
    # both the return category and the multiple carry categories.
    basket_1 = Basket(df=dfd, contracts=contracts_1,
                      ret="XR_NSA", cry=["CRY_NSA", "CRR_NSA"], blacklist=black)
    basket_1.make_basket(weight_meth="equal", max_weight=0.55, basket_name="GLB_EQUAL")

    df_basket = basket_1.return_basket("GLB_EQUAL")
    print(df_basket)

    df_weight = basket_1.return_weights("GLB_EQUAL")
    print(df_weight)

    # Second test. Zero carries. Inverse weight method.
    # However, call make_basket() method multiple times, using different weighting
    # methods, to understand how the basket's performance varies with different weight
    # methods. For instance, does limiting the volatility of the basket, over a period of
    # time, produce lower returns than simply taking an equal weight ?
    basket_2 = Basket(df=dfd, contracts=contracts_1,
                      ret="XR_NSA", blacklist=black)
    basket_2.make_basket(weight_meth="invsd", lback_meth="ma", lback_periods=21,
                         max_weight=0.55, remove_zeros=True, basket_name="GLB_INVERSE")
    df_basket_inv = basket_2.return_basket("GLB_INVERSE")

    basket_2.make_basket(weight_meth="equal", max_weight=0.55, basket_name="GLB_EQUAL")
    df_basket_equal = basket_2.return_basket("GLB_EQUAL")
    print(df_basket_inv)
    print(df_basket_equal)

    # Third test. One carry. Inverse values weight method.
    # Allow for multiple external weight methods being passed in. If multiple external
    # weight categories are involved in the basket calculation, pass all the categories
    # on the instance and call the make_basket() method separately using the respective
    # weight categories.

    basket_3 = Basket(df=dfd, contracts=contracts_1, ret="XR_NSA", cry=["CRY_NSA"],
                      blacklist=black, ewgts=['WBASE_NSA', 'WBASE_NSA'])

    basket_3.make_basket(weight_meth="inv_values", ewgt="WBASE_NSA", max_weight=0.55,
                         remove_zeros=True,
                         basket_name="GLB_INV_VALUES")

    df_inv_values = basket_3.return_basket("GLB_INV_VALUES")
    print(df_inv_values)
