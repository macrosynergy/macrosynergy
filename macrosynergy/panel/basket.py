"""
Basket class for calculating the returns and carries of baskets 
of financial contracts using various weighting methods.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import warnings

import random
from typing import List, Union, Tuple
from macrosynergy.panel.historic_vol import expo_weights, expo_std, flat_std
from macrosynergy.management.utils import reduce_df_by_ticker
from macrosynergy.panel.converge_row import ConvergeRow
from macrosynergy.management.simulate import make_qdf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class Basket(object):
    """
    Calculates the returns and carries of baskets of financial contracts using
    various weighting methods.

    :param <pd.Dataframe> df: standardized DataFrame containing the columns: 'cid',
        'xcat', 'real_date' and 'value'.
    :param <List[str]> contracts: base tickers (combinations of cross-sections and
        base categories) that define the contracts that go into the basket.
    :param <str> ret: return category postfix to be appended to the contract base;
        default is "XR_NSA".
    :param <List[str] or str> cry: carry category postfix; default is None. The field
        can either be a single carry or multiple carries defined in a List.
    :param <str> start: earliest date in ISO 8601 format. Default is None.
    :param <str> end: latest date in ISO 8601 format. Default is None.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded
        from the DataFrame. If one cross-section has several blacklist periods append
        numbers to the cross-section code.
    :param <List[str]> ewgts: one or more postfixes that may identify exogenous weight
        categories. Similar to return postfixes they are appended to base tickers.

    N.B.: Each instance of the class will update associated standardised DataFrames,
    containing return and carry categories, and external weights.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        contracts: List[str],
        ret: str = "XR_NSA",
        cry: Union[str, List[str]] = None,
        start: str = None,
        end: str = None,
        blacklist: dict = None,
        ewgts: List[str] = None,
    ):
        df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")
        df = df[["cid", "xcat", "real_date", "value"]]

        assert isinstance(contracts, list)
        c_error = "Contracts must be a list of strings."
        assert all(isinstance(c, str) for c in contracts), c_error
        assert isinstance(ret, str), "`ret` must be a string"

        df = reduce_df_by_ticker(
            df, start=start, end=end, ticks=None, blacklist=blacklist
        )
        self.contracts = contracts
        self.ret = ret
        self.ticks_ret = [con + ret for con in contracts]
        dfw_ret = self.pivot_dataframe(df, self.ticks_ret)
        self.dfw_ret = dfw_ret.dropna(axis=0, how="all")

        self.store_attributes(df, cry, "cry")
        self.store_attributes(df, ewgts, "wgt")

        self.tickers = self.ticks_ret + self.ticks_cry + self.ticks_wgt
        self.start = self.date_check(start)
        self.end = self.date_check(end)
        self.dfx = reduce_df_by_ticker(df, ticks=self.tickers)
        self.dict_retcry = {}  # dictionary for collecting basket return/carry dfs.
        self.dict_wgs = {}  # dictionary for collecting basket return/carry dfs.

    def store_attributes(self, df: pd.DataFrame, pfx: List[str], pf_name: str):
        """
        Adds multiple attributes to class based on postfixes that denote carry or
        external weight types.

        :param <pd.DataFrame> df: original, standardised DataFrame.
        :param <List[str]> pfx: category postfixes involved in the basket calculation.
        :param <str> pf_name: associated name of the postfix "cry" or "wgt".

        Note: These are [1] flags of existence of carry and weight strings in class,
        [2] lists of tickers related to all postfixes, [3] a dictionary of wide time
        series panel dataframes for all postfixes.
        """

        pfx_flag = pfx is not None
        self.__dict__[pf_name + "_flag"] = pfx_flag
        self.__dict__["ticks_" + pf_name] = []
        if pfx_flag:
            error = f"'{pf_name}' must be a <str> or a <List[str]>."
            assert isinstance(pfx, (list, str)), error
            pfx = [pfx] if isinstance(pfx, str) else pfx
            self.__dict__[pf_name] = pfx

            dfws_pfx = {}
            for cat in pfx:
                ticks = [con + cat for con in self.contracts]
                self.__dict__["ticks_" + pf_name] += ticks
                dfws_pfx[cat] = self.pivot_dataframe(df, ticks)
        else:
            dfws_pfx = None

        self.__dict__["dfws_" + pf_name] = dfws_pfx

    @staticmethod
    def pivot_dataframe(df, tick_list):
        """
        Reduces the standardised DataFrame to include a subset of the possible tickers
        and, subsequently returns a wide dataframe: each column corresponds to a ticker.

        :param <List[str]> tick_list: list of the respective tickers.
        :param <pd.DataFrame> df: standardised dataframe.

        :return <pd.DataFrame> dfw: wide dataframe.
        """

        df["ticker"] = df["cid"] + "_" + df["xcat"]
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
            else:
                return pd.Timestamp(date_string).strftime("%Y-%m-%d")

    @staticmethod
    def check_weights(weight: pd.DataFrame):
        """
        Checks if all rows in dataframe add up to roughly 1.

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

        act_cross = ~df_ret.isnull()
        uniform = (1 / act_cross.sum(axis=1)).values
        uniform = uniform[:, np.newaxis]

        broadcast = np.repeat(uniform, df_ret.shape[1], axis=1)

        weight = act_cross.multiply(broadcast)
        cols = weight.columns
        # Zeroes treated as NaNs.
        weight[cols] = weight[cols].replace({0: np.nan})

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

        act_cross = ~df_ret.isnull()

        weights = np.array(weights, dtype=np.float32)
        rows = act_cross.shape[0]
        broadcast = np.tile(weights, (rows, 1))  # Constructs Array by row repetition.

        # Replaces weight factors with zeroes if concurrent return unavailable.
        weight = act_cross.multiply(broadcast)
        cols = weight.columns
        # Zeroes treated as NaNs.
        weight[cols] = weight[cols].replace({0: np.nan})

        weight_arr = weight.to_numpy()  # convert df to np array.
        weight[weight.columns] = weight_arr / np.sum(weight_arr, axis=1)[:, np.newaxis]
        self.check_weights(weight)

        return weight

    def inverse_weight(
        self,
        dfw_ret: pd.DataFrame,
        lback_meth: str = "xma",
        lback_periods: int = 21,
        remove_zeros: bool = True,
    ):
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
            dfwa = dfw_ret.rolling(window=lback_periods * 2).agg(
                expo_std, w=weights, remove_zeros=remove_zeros
            )

        cols = dfwa.columns
        # Zeroes treated as NaNs.
        dfwa[cols] = dfwa[cols].replace({0: np.nan})

        df_isd = 1 / dfwa
        df_wgts = df_isd / df_isd.sum(axis=1).values[:, np.newaxis]
        self.check_weights(df_wgts)

        return df_wgts

    def values_weight(
        self, dfw_ret: pd.DataFrame, dfw_wgt: pd.DataFrame, weight_meth: str
    ):
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
            warnings.warn("Negative values in the weight matrix set to zero.")

        exo_array = dfw_wgt.to_numpy()
        df_bool = ~dfw_ret.isnull()

        weights_df = df_bool.multiply(exo_array)
        cols = weights_df.columns

        # Zeroes treated as NaNs.
        weights_df[cols] = weights_df[cols].replace({0: np.nan})

        if weight_meth != "values":
            weights_df = 1 / weights_df

        weights = weights_df.divide(weights_df.sum(axis=1), axis=0)
        self.check_weights(weights)

        return weights

    def make_weights(
        self,
        weight_meth: str = "equal",
        weights: List[float] = None,
        lback_meth: str = "xma",
        lback_periods: int = 21,
        ewgt: str = None,
        max_weight: float = 1.0,
        remove_zeros: bool = True,
    ):
        """
        Returns wide dataframe of weights to be used for basket series.

        :param <str> weight_meth: method used for weighting constituent returns and
            carry. The parameter can receive either a single weight method or
            multiple weighting methods. See `make_basket` docstring.
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

        return: <pd.DataFrame>: wide dataframe of contract weights across time.
        """
        assert 0.0 < max_weight <= 1.0
        assert weight_meth in ["equal", "fixed", "values", "inv_values", "invsd"]

        # Apply weight method.

        if weight_meth == "equal":
            dfw_wgs = self.equal_weight(df_ret=self.dfw_ret)

        elif weight_meth == "fixed":
            message = "Expects a list of weights."
            message_2 = "List of weights must be equal to the number of contracts."
            assert isinstance(weights, list), message
            assert self.dfw_ret.shape[1] == len(weights), message_2
            message_3 = "Expects a list of floating point values."
            assert all(isinstance(w, (int, float)) for w in weights), message_3

            dfw_wgs = self.fixed_weight(df_ret=self.dfw_ret, weights=weights)

        elif weight_meth == "invsd":
            error_message = "Lookback method method must be 'ma' or 'xma'."
            assert lback_meth in ["xma", "ma"], error_message
            assert isinstance(lback_periods, int), "Expects <int>."
            dfw_wgs = self.inverse_weight(
                dfw_ret=self.dfw_ret,
                lback_meth=lback_meth,
                lback_periods=lback_periods,
                remove_zeros=remove_zeros,
            )

        elif weight_meth in ["values", "inv_values"]:
            assert ewgt in self.wgt, f"{ewgt} is not defined on the instance."
            # Lag by one day to be used as weights.
            try:
                dfw_wgt = self.dfws_wgt[ewgt].shift(1)
            except KeyError as e:
                print(f"Basket not found: {e}.")
            else:
                cols = sorted(dfw_wgt.columns)
                dfw_ret = dfw_wgt.reindex(cols, axis=1)
                dfw_wgt = dfw_wgt.reindex(cols, axis=1)
                dfw_wgs = self.values_weight(dfw_ret, dfw_wgt, weight_meth)

        else:
            raise NotImplementedError(f"Weight method unknown {weight_meth}")

        # Remove leading NA rows.

        fvi = max(dfw_wgs.first_valid_index(), self.dfw_ret.first_valid_index())
        dfw_wgs = dfw_wgs[fvi:]

        # Impose cap on cross-section weights.

        if max_weight < 1.0:
            dfw_wgs = self.max_weight_func(weights=dfw_wgs, max_weight=max_weight)

        return dfw_wgs

    @staticmethod
    def column_manager(df_cat: pd.DataFrame, dfw_wgs: pd.DataFrame):
        """
        Will match the column names of the two dataframes involved in the computation:
        either the return & weight dataframes or the carry & weight dataframes. The
        pandas multiply operation requires the column names, of both dataframes involved
        in the binary operation, to be identical.

        :param <pd.DataFrame> df_cat: return or carry dataframe.
        :param <pd.DataFrame> dfw_wgs: weight dataframe.

        :return <pd.DataFrame> dfw_wgs: modified weight dataframe (column names will map
            to the other dataframe received).
        """

        df_cat = df_cat.reindex(sorted(df_cat.columns), axis=1)
        dfw_wgs = dfw_wgs.reindex(sorted(dfw_wgs.columns), axis=1)

        ret_cols = df_cat.columns
        weight_cols = dfw_wgs.columns

        if all(ret_cols != weight_cols):
            dfw_wgs.columns = ret_cols

        return dfw_wgs

    def column_weights(self, dfw_wgs: pd.DataFrame):
        """
        The weight dataframe is used to compute the basket performance for returns,
        carries etc. Therefore, with their broad application, the column names of the
        dataframe should correspond to the ticker postfix of each contract.

        :param <pd.DataFrame> dfw_wgs: weight dataframe.

        :return <pd.DataFrame> dfw_wgs: weight dataframe with updated columns names.
        """

        dfw_weight_names = lambda w_name: w_name[: w_name.find(self.w_field)]
        if self.wgt_flag and self.exo_w_postfix is not None:
            self.__dict__["w_field"] = self.exo_w_postfix
        else:
            self.__dict__["w_field"] = self.ret

        cols = list(map(dfw_weight_names, dfw_wgs.columns))
        dfw_wgs.columns = cols
        dfw_wgs.columns.name = "ticker"

        return dfw_wgs

    def make_basket(
        self,
        weight_meth: str = "equal",
        weights: List[float] = None,
        lback_meth: str = "xma",
        lback_periods: int = 21,
        ewgt: str = None,
        max_weight: float = 1.0,
        remove_zeros: bool = True,
        basket_name: str = "GLB_ALL",
    ):
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
            [5] "inv_values": weights are inversely proportionate to of values of
                exogenous weight category.
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

        self.__dict__["exo_w_postfix"] = ewgt
        dfw_wgs = self.make_weights(
            weight_meth=weight_meth,
            weights=weights,
            lback_meth=lback_meth,
            lback_periods=lback_periods,
            ewgt=ewgt,
            max_weight=max_weight,
            remove_zeros=remove_zeros,
        )
        select = ["ticker", "real_date", "value"]

        dfw_wgs_copy = self.column_manager(df_cat=self.dfw_ret, dfw_wgs=dfw_wgs)

        dfw_bret = self.dfw_ret.multiply(dfw_wgs_copy).sum(axis=1)
        dfxr = dfw_bret.to_frame("value").reset_index()
        basket_ret = basket_name + "_" + self.ret
        dfxr = dfxr.assign(ticker=basket_ret)[select]
        store = [dfxr]

        if self.cry_flag:
            cry_list = []
            for cr in self.cry:
                dfw_wgs_copy = self.column_manager(
                    df_cat=self.dfws_cry[cr], dfw_wgs=dfw_wgs
                )
                dfw_bcry = self.dfws_cry[cr].multiply(dfw_wgs_copy).sum(axis=1)
                dfcry = dfw_bcry.to_frame("value").reset_index()
                basket_cry = basket_name + "_" + cr
                dfcry = dfcry.assign(ticker=basket_cry)[select]
                cry_list.append(dfcry)

            store += cry_list

        df_retcry = pd.concat(store)
        df_retcry = df_retcry.reset_index(drop=True)
        self.dict_retcry[basket_name] = df_retcry

        self.dict_wgs[basket_name] = self.column_weights(dfw_wgs)

    def weight_visualiser(
        self,
        basket_name,
        start_date: str = None,
        end_date: str = None,
        subplots: bool = True,
        facet_grid: bool = False,
        scatter: bool = False,
        all_tickers: bool = True,
        single_ticker: str = None,
        percentage_change: bool = False,
        size: Tuple[int, int] = (7, 7),
    ):
        """
        Method used to visualise the weights associated with each contract in the basket.

        :param <str> basket_name: name of basket whose weights are visualized
        :param <str> start_date: start date of he visualisation period.
        :param <str> end_date: end date of the visualization period.
        :param <bool> subplots: contract weights are displayed on different plots (True)
            or on a single plot (False).
        :param <bool> facet_grid: parameter used to break up the plot into multiple
            cartesian coordinate systems. If the basket consists of a high number of
            contracts, using the Facet Grid is recommended.
        :param <bool> scatter: if the facet_grid parameter is set to True there are two
            options: i) scatter plot if there a lot of blacklist periods; ii) line plot
            for continuous series.
        :param <bool> all_tickers: if True (default) all weights are displayed.
            If set to False `single-ticker` must be specified.
        :param <str> single_ticker: individual ticker for further, more detailed,
            analysis.
        :param <bool> percentage_change: graphical display used to further assimilate the
            fluctuations in the contract's weight. The graphical display is limited to a
            single contract. Therefore, pass the ticker into the parameter
            "single_ticker".
        :param <Tuple[int, int]> size: size of the plot. Default is (7, 7).

        """

        date_conv = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d %X")
        try:
            dfw_wgs = self.dict_wgs[basket_name]
        except KeyError as e:
            print(f"Basket not found - call make_basket() method first: {e}.")
        else:
            if isinstance(start_date, str) and isinstance(end_date, str):
                self.date_check(start_date)
                self.date_check(end_date)
                start_date = date_conv(start_date)
                end_date = date_conv(end_date)
            elif isinstance(start_date, str):
                self.date_check(start_date)
                start_date = date_conv(start_date)
                end_date = dfw_wgs.index[-1]
            elif isinstance(end_date, str):
                self.date_check(end_date)
                start_date = dfw_wgs.index[0]
                end_date = date_conv(end_date)
            else:
                start_date = dfw_wgs.index[0]
                end_date = dfw_wgs.index[-1]

            error_1 = f"{start_date} unavailable in weight dataframe."
            c = dfw_wgs.index
            assert start_date in c, error_1
            error_2 = f"{end_date} unavailable in weight dataframe."
            assert end_date in c, error_2

            dfw_wgs = dfw_wgs.truncate(before=start_date, after=end_date)
            if not all_tickers:
                error_3 = "The parameter, 'single_ticker', must be a <str>."
                assert isinstance(single_ticker, str), error_3
                error_4 = (
                    f"Ticker not present in the weight dataframe. Available "
                    f"tickers: {list(dfw_wgs.columns)}."
                )
                assert single_ticker in dfw_wgs.columns, error_4
                dfw_wgs = dfw_wgs[[single_ticker]]

            if facet_grid:
                df_stack = dfw_wgs.stack().to_frame("value").reset_index()
                df_stack = df_stack.sort_values(["ticker", "real_date"])
                no_contracts = dfw_wgs.shape[1]
                facet_cols = 4 if no_contracts >= 8 else 3
                sns.set(rc={"figure.figsize": size})
                fg = sns.FacetGrid(
                    df_stack, col="ticker", col_wrap=facet_cols, sharey=True
                )

                scatter_error = (
                    f"Boolean object expected - instead received " f"{type(scatter)}."
                )
                assert isinstance(scatter, bool), scatter_error
                if scatter:
                    fg.map_dataframe(sns.scatterplot, x="real_date", y="value")
                else:
                    # Seaborn will linearly interpolate NaN values which is visually
                    # misleading. Therefore, aim to negate the operation.
                    fg.map_dataframe(
                        sns.lineplot,
                        x="real_date",
                        y="value",
                        hue=df_stack["value"].isna().cumsum(),
                        palette=["blue"] * df_stack["value"].isna().cumsum().nunique(),
                        estimator=None,
                        markers=True,
                    )

                equal_value = 1 / no_contracts
                fg.map(plt.axhline, y=equal_value, linestyle="--", color="gray", lw=0.5)
                # Set axes labels of individual charts.
                fg.set_axis_labels("", "")
                fg.set_titles(col_template="{col_name}")
                fg.fig.suptitle("Contract weights in basket", y=1.02)
            else:
                plt.rcParams["figure.figsize"] = size
                dfw_wgs.plot(
                    subplots=subplots, title="Weight Values Timestamp", legend=True
                )
                plt.xlabel("real_date, years")

            date_func = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")
            if percentage_change:
                error_5 = (
                    "Percentage change display is applied to a single ticker. Set "
                    "the parameter 'all_tickers' to False."
                )
                assert dfw_wgs.shape[1] == 1, error_5

                plt.rcParams["figure.figsize"] = size
                fig, ax = plt.subplots()
                dfw_pct = dfw_wgs.pct_change(periods=1) * 100
                n_index = np.array(list(map(date_func, dfw_pct.index)))
                dfw_pct = dfw_pct.set_index(keys=n_index)
                dfw_pct.plot(kind="bar", color="coral", ax=ax)
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                plt.xticks(rotation=0)
                ax.set_ylabel("Percentage Change in weight.")
                ax.legend()

            plt.show()

    @staticmethod
    def column_split(df: pd.DataFrame):
        """
        Receives a dataframe with the columns 'ticker', 'real_date' and 'value' and
        returns a standardised dataframe with the columns 'cid', 'xcat', 'real_date' and
        'value. The 'ticker' column is broken up to produce the two new columns.

        :param <pd.DataFrame> df:

        :return <pd.DataFrame> df: standardised dataframe.
        """
        select = ["cid", "xcat", "real_date", "value"]

        cid_func = lambda t: t.split("_")[0]
        xcat_func = lambda t: "_".join(t.split("_")[1:])

        cids_w_df = list(map(cid_func, df["ticker"]))
        df["cid"] = np.array(cids_w_df)

        df = df.rename(columns={"ticker": "xcat"})
        df["xcat"] = np.array(list(map(xcat_func, df["xcat"])))

        df = df[select]
        return df

    def return_basket(self, basket_names: Union[str, List[str]] = None):
        """
        Return standardized dataframe of basket performance data based on one or more
        weighting methods.

        :param <str or List[str]> basket_names: single basket name or list for which
            performance data are to be returned. If none is given all baskets added to
            the instance are selected.

        :return <pd.Dataframe>: standardized DataFrame with the basket return and
            (possibly) carry data in standard form, i.e. columns 'cid', 'xcat',
            'real_date' and 'value'.
        """

        if basket_names is None:
            basket_names = list(self.dict_retcry.keys())

        basket_error = "String or List of basket names expected."
        assert isinstance(basket_error, (list, str)), basket_error
        if isinstance(basket_names, str):
            basket_names = [basket_names]

        ret_baskets = []
        for b in basket_names:
            try:
                dfw_retcry = self.dict_retcry[b]
            except KeyError as e:
                print(f"Basket not found - call make_basket() method first: {e}.")
            else:
                dfw_retcry = self.column_split(dfw_retcry)
                dfw_retcry = dfw_retcry.sort_values(["xcat", "real_date"])
                ret_baskets.append(dfw_retcry)

        return_df = pd.concat(ret_baskets)
        return return_df.reset_index(drop=True)

    def return_weights(self, basket_names: Union[str, List[str]] = None):
        """
        Return the standardised dataframe containing the corresponding weights used to
        compute the basket.

        :param <str or List[str]> basket_names: single basket name or list for which
            performance data are to be returned. If none is given all baskets added to
            the instance are selected.

        :return <pd.Dataframe>: standardized DataFrame with basket weights.
        """

        if basket_names is None:
            basket_names = list(self.dict_wgs.keys())

        basket_error = "String or List of basket names expected."
        assert isinstance(basket_error, (list, str)), basket_error
        if isinstance(basket_names, str):
            basket_names = [basket_names]

        weight_baskets = []
        select = ["cid", "xcat", "real_date", "value"]

        for b in basket_names:
            try:
                dfw_wgs = self.dict_wgs[b]
            except KeyError as e:
                print(f"Basket not found - call make_basket() method first: {e}.")
            else:
                w = dfw_wgs.stack().to_frame("value").reset_index()
                w = self.column_split(df=w)
                w = w.sort_values(["cid", "real_date"])

                w = w.loc[w.value > 0, select]
                w["xcat"] += "_" + b + "_" + "WGT"
                weight_baskets.append(w)

        return_df = pd.concat(weight_baskets)
        return return_df.reset_index(drop=True)


if __name__ == "__main__":
    cids = ["AUD", "GBP", "NZD", "USD"]
    xcats = [
        "FXXR_NSA",
        "FXCRY_NSA",
        "FXCRR_NSA",
        "EQXR_NSA",
        "EQCRY_NSA",
        "EQCRR_NSA",
        "FXWBASE_NSA",
        "EQWBASE_NSA",
    ]

    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )

    df_cids.loc["AUD"] = ["2000-01-01", "2022-03-14", 0, 1]
    df_cids.loc["GBP"] = ["2001-01-01", "2022-03-14", 0, 2]
    df_cids.loc["NZD"] = ["2002-01-01", "2022-03-14", 0, 3]
    df_cids.loc["USD"] = ["2000-01-01", "2022-03-14", 0, 4]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["FXXR_NSA"] = ["2010-01-01", "2022-03-14", 0, 1, 0, 0.2]
    df_xcats.loc["FXCRY_NSA"] = ["2010-01-01", "2022-03-14", 1, 1, 0.9, 0.2]
    df_xcats.loc["FXCRR_NSA"] = ["2010-01-01", "2022-03-14", 0.5, 0.8, 0.9, 0.2]
    df_xcats.loc["EQXR_NSA"] = ["2010-01-01", "2022-03-14", 0.5, 2, 0, 0.2]
    df_xcats.loc["EQCRY_NSA"] = ["2010-01-01", "2022-03-14", 2, 1.5, 0.9, 0.5]
    df_xcats.loc["EQCRR_NSA"] = ["2010-01-01", "2022-03-14", 1.5, 1.5, 0.9, 0.5]
    df_xcats.loc["FXWBASE_NSA"] = ["2010-01-01", "2022-02-01", 1, 1.5, 0.8, 0.5]
    df_xcats.loc["EQWBASE_NSA"] = ["2010-01-01", "2022-02-01", 1, 1.5, 0.9, 0.5]

    random.seed(2)
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    black = {"AUD": ["2010-01-01", "2013-12-31"], "GBP": ["2010-01-01", "2013-12-31"]}
    contracts = ["AUD_FX", "AUD_EQ", "NZD_FX", "GBP_EQ", "USD_EQ"]
    gdp_figures = [17.0, 17.0, 41.0, 9.0, 250.0]

    contracts_1 = ["AUD_FX", "GBP_FX", "NZD_FX", "USD_EQ"]

    # First test. Multiple carries. Equal weight method.
    # The main aspect to check in the code is that basket performance has been applied to
    # both the return category and the multiple carry categories.
    dfd["grading"] = np.ones(dfd.shape[0])

    basket_1 = Basket(
        df=dfd,
        contracts=contracts_1,
        ret="XR_NSA",
        cry=["CRY_NSA", "CRR_NSA"],
        blacklist=black,
    )
    basket_1.make_basket(weight_meth="equal", max_weight=0.55, basket_name="GLB_EQUAL")

    basket_1.make_basket(
        weight_meth="fixed",
        max_weight=0.55,
        weights=[1 / 6, 1 / 6, 1 / 6, 1 / 2],
        basket_name="GLB_FIXED",
    )

    # show the weights of the GLB_FIXED basket

    basket_1.weight_visualiser(basket_name="GLB_FIXED", subplots=False, size=(10, 5))
