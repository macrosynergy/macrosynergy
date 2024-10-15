"""
"Naive" PnLs with limited signal options and disregarding transaction costs.
"""

from dataclasses import dataclass
import warnings
from itertools import product
from typing import Dict, List, Optional, Tuple, Union
from numbers import Number

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from macrosynergy import PYTHON_3_8_OR_LATER
from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import (
    reduce_df,
    update_df,
    _map_to_business_day_frequency,
)
from macrosynergy.panel.make_zn_scores import make_zn_scores
from macrosynergy.signal import SignalReturnRelations


class NaivePnL:
    """
    Computes and collects illustrative PnLs with limited signal options and
    disregarding transaction costs.

    :param <pd.Dataframe> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <str> ret: return category.
    :param <List[str]> sigs: signal categories. Able to pass in multiple possible signals
        to the Class' constructor and their respective vintages will be held on the
        instance's DataFrame. The signals can subsequently be referenced through the
        self.make_pnl() method which receives a single signal per call.
    :param <List[str]> cids: cross sections that are traded. Default is all in the
        dataframe.
    :param <str, List[str]> bms: list of benchmark tickers for which
        correlations are displayed against PnL strategies.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded
        from the dataframe.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        ret: str,
        sigs: List[str],
        cids: List[str] = None,
        bms: Union[str, List[str]] = None,
        start: str = None,
        end: str = None,
        blacklist: dict = None,
    ):
        df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

        # Will host the benchmarks.
        self.dfd = df

        assert isinstance(ret, str), "The return category expects a single <str>."
        self.ret = ret
        xcats = [ret] + sigs

        cols = ["cid", "xcat", "real_date", "value"]
        # Potentially excludes the benchmarks but will be held on the instance level
        # through self.dfd.
        self.df, self.xcats, self.cids = reduce_df(
            df[cols], xcats, cids, start, end, blacklist, out_all=True
        )

        self.sigs = sigs

        ticker_func = lambda t: t[0] + "_" + t[1]
        self.tickers = list(map(ticker_func, product(self.cids, self.xcats)))

        self.df["real_date"] = pd.to_datetime(self.df["real_date"])

        # Data structure used to track all of the generated PnLs from make_pnl() method.
        self.pnl_names = []
        self.signal_df = {}
        self.black = blacklist

        self.bm_bool = isinstance(bms, (str, list))
        if self.bm_bool:
            bms = [bms] if isinstance(bms, str) else bms

            # Pass in the original DataFrame; negative signal will not have been applied
            # which will corrupt the use of the benchmark categories.
            bm_dict = self.add_bm(df=self.dfd, bms=bms, tickers=self.tickers)

            self._bm_dict = bm_dict

        self.pnl_params = {}

    def add_bm(self, df: pd.DataFrame, bms: List[str], tickers: List[str]):
        """
        Returns a dictionary with benchmark return series.

        :param <pd.DataFrame> df: aggregate DataFrame passed into the Class.
        :param <List[str]> bms: benchmark return tickers.
        :param <List[str]> tickers: the available tickers held in the reduced DataFrame.
            The reduced DataFrame consists exclusively of the signal & return categories.
        """

        bm_dict = {}

        for bm in bms:
            # Accounts for appending "_NEG" to the ticker.
            bm_s = bm.split("_", maxsplit=1)
            cid = bm_s[0]
            xcat = bm_s[1]
            dfa = df[(df["cid"] == cid) & (df["xcat"] == xcat)]

            if dfa.shape[0] == 0:
                print(f"{bm} has no observations in the DataFrame.")
            else:
                df_single_bm = dfa.pivot(
                    index="real_date", columns="xcat", values="value"
                )
                df_single_bm.columns = [bm]
                bm_dict[bm] = df_single_bm
                if bm not in tickers:
                    self.df = update_df(self.df, dfa)

        return bm_dict

    @staticmethod
    def _make_signal(
        dfx: pd.DataFrame,
        sig: str,
        sig_op: str = "zn_score_pan",
        min_obs: int = 252,
        iis: bool = True,
        sequential: bool = True,
        neutral: str = "zero",
        thresh: float = None,
    ):
        """
        Helper function used to produce the raw signal that forms the basis for
        positioning.

        :param <pd.DataFrame> dfx: DataFrame defined over the return & signal category.
        :param <str> sig: name of the raw signal.
        :param <str> sig_op: signal transformation.
        :param <int> min_obs: the minimum number of observations required to calculate
            zn_scores. Default is 252.
        :param <bool> iis: if True (default) zn-scores are also calculated for the initial
            sample period defined by min_obs, on an in-sample basis, to avoid losing
            history.
        :param <bool> sequential: if True (default) score parameters are estimated
            sequentially with concurrently available information only.
        :param <str> neutral: method to determine neutral level.
        :param <float> thresh: threshold value beyond which scores are winsorized,

        """

        if sig_op == "binary":
            dfw = dfx.pivot(index=["cid", "real_date"], columns="xcat", values="value")
            dfw["psig"] = np.sign(dfw[sig])
        elif sig_op == "raw":
            dfw = dfx.pivot(index=["cid", "real_date"], columns="xcat", values="value")
            dfw["psig"] = dfw[sig]
        else:
            panw = 1 if sig_op == "zn_score_pan" else 0
            # The re-estimation frequency for the neutral level and standard deviation
            # will be the same as the re-balancing frequency. For instance, if the
            # neutral level is computed weekly, a material change in the signal will only
            # manifest along a similar timeline. Therefore, re-estimation and
            # re-balancing frequencies match.
            df_ms = make_zn_scores(
                dfx,
                xcat=sig,
                neutral=neutral,
                pan_weight=panw,
                sequential=sequential,
                min_obs=min_obs,
                iis=iis,
                thresh=thresh,
            )

            df_ms = df_ms.drop("xcat", axis=1)
            df_ms["xcat"] = "psig"

            dfx_concat = pd.concat([dfx, df_ms])
            dfw = dfx_concat.pivot(
                index=["cid", "real_date"], columns="xcat", values="value"
            )

        # Reconstruct the DataFrame to recognise the signal's start date for each
        # individual cross-section
        dfw_list = []
        for c, cid_df in dfw.groupby(level=0):
            first_date = cid_df.loc[:, "psig"].first_valid_index()
            cid_df = cid_df.loc[first_date:, :]
            dfw_list.append(cid_df)
        return pd.concat(dfw_list)

    @staticmethod
    def rebalancing(dfw: pd.DataFrame, rebal_freq: str = "daily", rebal_slip=0):
        """
        The signals are calculated daily and for each individual cross-section defined in
        the panel. However, re-balancing a position can occur more infrequently than
        daily. Therefore, produce the re-balancing values according to the more
        infrequent timeline (weekly or monthly).

        :param <pd.Dataframe> dfw: DataFrame with each category represented by a column
            and the daily signal is also included with the column name 'psig'.
        :param <str> rebal_freq: re-balancing frequency for positions according to signal
            must be one of 'daily' (default), 'weekly' or 'monthly'.
        :param <str> rebal_slip: re-balancing slippage in days.

        :return <pd.Series>: will return a pd.Series containing the associated signals
            according to the re-balancing frequency.
        """

        # The re-balancing days are the first of the respective time-periods because of
        # the shift forward by one day applied earlier in the code. Therefore, only
        # concerned with the minimum date of each re-balance period.
        dfw["year"] = dfw["real_date"].dt.year
        if rebal_freq == "monthly":
            dfw["month"] = dfw["real_date"].dt.month
            rebal_dates = dfw.groupby(["cid", "year", "month"])["real_date"].min()
        elif rebal_freq == "weekly":
            dfw["week"] = dfw["real_date"].apply(lambda x: x.week)
            rebal_dates = dfw.groupby(["cid", "year", "week"])["real_date"].min()

        # Convert the index, 'cid', to a formal column aligned to the re-balancing dates.
        r_dates_df = rebal_dates.reset_index(level=0)
        r_dates_df.reset_index(drop=True, inplace=True)
        dfw = dfw[["real_date", "psig", "cid"]]

        # Isolate the required signals on the re-balancing dates. Only concerned with the
        # respective signal on the re-balancing date. However, the produced DataFrame
        # will only be defined over the re-balancing dates. Therefore, merge the
        # aforementioned DataFrame with the original DataFrame such that all business
        # days are included. The intermediary dates, dates between re-balancing dates,
        # will initially be populated by NA values. To ensure the signal is used for the
        # duration between re-balancing dates, forward fill the computed signal over the
        # associated dates.

        # The signal is computed for each individual cross-section. Therefore, merge on
        # the real_date and the cross-section.
        rebal_merge = r_dates_df.merge(dfw, how="left", on=["real_date", "cid"])
        # Re-establish the daily date series index where the intermediary dates, between
        # the re-balancing dates, will be populated using a forward fill.
        rebal_merge = dfw[["real_date", "cid"]].merge(
            rebal_merge, how="left", on=["real_date", "cid"]
        )
        rebal_merge["psig"] = rebal_merge["psig"].ffill().shift(rebal_slip)
        rebal_merge = rebal_merge.sort_values(["cid", "real_date"])

        rebal_merge = rebal_merge.set_index("real_date")
        sig_series = rebal_merge.drop(["cid"], axis=1)

        return sig_series

    def make_pnl(
        self,
        sig: str,
        sig_op: str = "zn_score_pan",
        sig_add: float = 0,
        sig_neg: bool = False,
        pnl_name: Optional[str] = None,
        rebal_freq: str = "daily",
        rebal_slip: int = 0,
        vol_scale: Optional[float] = None,
        leverage: float = 1.0,
        min_obs: int = 261,
        iis: bool = True,
        sequential: bool = True,
        neutral: str = "zero",
        thresh: float = None,
    ):
        """
        Calculate daily PnL and add to class instance.

        :param <str> sig: name of raw signal that is basis for positioning. The signal
            is assumed to be recorded at the end of the day prior to position taking.
        :param <str> sig_op: signal transformation options; must be one of
            'zn_score_pan', 'zn_score_cs', 'binary', or 'raw'. The default is 'zn_score_pan'.
            'zn_score_pan': transforms raw signals into z-scores around zero value
            based on the whole panel. The neutral level & standard deviation will use the
            cross-section of panels.
            'zn_score_cs': transforms signals to z-scores around zero based on
            cross-section alone.
            'binary': transforms signals into uniform long/shorts (1/-1) across all
            sections.
            N.B.: zn-score here means standardized score with zero being the natural
            neutral level and standardization through division by mean absolute value.
        :param <float> sig_add: add a constant to the signal after initial transformation.
            This allows to give PnLs a long or short bias relative to the signal
            score. Default is 0.
        :param <str> sig_neg: if True the PnL is based on the negative value of the
            transformed signal. Default is False.
        :param <str> pnl_name: name of the PnL to be generated and stored.
            Default is None, i.e. a default name is given. The default name will be:
            'PNL_<signal name>[_<NEG>]', with the last part added if sig_neg has been
            set to True.
            Previously calculated PnLs of the same name will be overwritten. This means
            that if a set of PnLs are to be compared, each PnL requires a distinct name.
        :param <str> rebal_freq: re-balancing frequency for positions according to signal
            must be one of 'daily' (default), 'weekly' or 'monthly'. The re-balancing is
            only concerned with the signal value on the re-balancing date which is
            delimited by the frequency chosen.
            Additionally, the re-balancing frequency will be applied to make_zn_scores()
            if used as the method to produce the raw signals.
        :param <str> rebal_slip: re-balancing slippage in days. Default is 1 which
            means that it takes one day to re-balance the position and that the new
            positions produce PnL from the second day after the signal has been recorded.
        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given.
            This is for comparative visualization and not out-of-sample. Default is none.
        :param <float> leverage: leverage applied to the raw signal when a `vol_scale` is
            not defined. Default is 1.0, i.e., position size is 1 or 100% of implied risk
            capital.
        :param <int> min_obs: the minimum number of observations required to calculate
            zn_scores. Default is 252.
        :param <bool> iis: if True (default) zn-scores are also calculated for the initial
            sample period defined by min_obs, on an in-sample basis, to avoid losing
            history.
        :param <bool> sequential: if True (default) score parameters (neutral level and
            standard deviations) are estimated sequentially with concurrently available
            information only.
        :param <str> neutral: method to determine neutral level. Default is 'zero'.
            Alternatives are 'mean' and "median".
        :param <float> thresh: threshold value beyond which scores are winsorized,
            i.e. contained at that threshold. Therefore, the threshold is the maximum
            absolute score value that the function is allowed to produce. The minimum
            threshold is one standard deviation. Default is no threshold.

        """
        for varx, name, typex in (
            (sig, "sig", str),
            (sig_op, "sig_op", str),
            (
                sig_add,
                "sig_add",
                (Number),
            ),  # testing for number instead of (float, int)
            (sig_neg, "sig_neg", bool),
            (pnl_name, "pnl_name", (str, type(None))),
            (rebal_freq, "rebal_freq", str),
            (rebal_slip, "rebal_slip", int),
            (vol_scale, "vol_scale", (Number, type(None))),
            (leverage, "leverage", (Number)),
            (min_obs, "min_obs", int),
            (iis, "iis", bool),
            (sequential, "sequential", bool),
            (neutral, "neutral", str),
            (thresh, "thresh", (Number, type(None))),
        ):
            if not isinstance(varx, typex):
                raise TypeError(f"{name} must be a {typex}.")

        error_sig = (
            f"Signal category missing from the options defined on the class: "
            f"{self.sigs}. "
        )
        if sig not in self.sigs:
            raise ValueError(error_sig)

        sig_options = ["zn_score_pan", "zn_score_cs", "binary", "raw"]
        error_sig_method = (
            f"The signal transformation method, {sig_op}, is not one of "
            f"the options specified: {sig_options}."
        )
        if sig_op not in sig_options:
            raise ValueError(error_sig_method)

        freq_params = ["daily", "weekly", "monthly"]
        freq_error = f"Re-balancing frequency must be one of: {freq_params}."
        if rebal_freq not in freq_params:
            raise ValueError(freq_error)

        if thresh is not None and thresh < 1:
            raise ValueError("thresh must be greater than or equal to one.")

        err_lev = "`leverage` must be a numerical value greater than 0."
        if leverage <= 0:
            raise ValueError(err_lev)
        err_vol = "`vol_scale` must be a numerical value greater than 0."
        if vol_scale is not None and (vol_scale <= 0):
            raise ValueError(err_vol)

        # B. Extract DataFrame of exclusively return and signal categories in time series
        # format.
        dfx = self.df[self.df["xcat"].isin([self.ret, sig])]

        dfw = self._make_signal(
            dfx=dfx,
            sig=sig,
            sig_op=sig_op,
            min_obs=min_obs,
            iis=iis,
            sequential=sequential,
            neutral=neutral,
            thresh=thresh,
        )

        if sig_neg:
            dfw["psig"] *= -1
            neg = "_NEG"
        else:
            neg = ""

        dfw["psig"] += sig_add

        self._winsorize(df=dfw["psig"], thresh=thresh)

        # Multi-index DataFrame with a natural minimum lag applied.
        dfw["psig"] = dfw["psig"].groupby(level=0).shift(1)
        dfw.reset_index(inplace=True)
        dfw = dfw.rename_axis(None, axis=1)

        dfw = dfw.sort_values(["cid", "real_date"])

        if rebal_freq != "daily":
            sig_series = self.rebalancing(
                dfw=dfw, rebal_freq=rebal_freq, rebal_slip=rebal_slip
            )
            dfw["sig"] = np.squeeze(sig_series.to_numpy())
        else:
            dfw = dfw.rename({"psig": "sig"}, axis=1)

        # The signals are generated across the panel.
        dfw["value"] = dfw[self.ret] * dfw["sig"]

        df_pnl = dfw.loc[:, ["cid", "real_date", "value"]]

        # Compute the return across the panel. The returns are still computed daily
        # regardless of the re-balancing frequency potentially occurring weekly or
        # monthly.
        df_pnl_all = df_pnl.groupby(["real_date"]).sum(numeric_only=True)
        df_pnl_all = df_pnl_all[df_pnl_all["value"].cumsum() != 0]
        # Returns are computed for each cross-section and across the panel.
        df_pnl_all["cid"] = "ALL"
        df_pnl_all = df_pnl_all.reset_index()[df_pnl.columns]
        # Will be inclusive of each individual cross-section's signal-adjusted return and
        # the aggregated panel return.
        df_pnl = pd.concat([df_pnl, df_pnl_all])

        if vol_scale is not None:
            leverage = vol_scale * (df_pnl_all["value"].std() * np.sqrt(261)) ** (-1)

        assert isinstance(leverage, (float, int)), err_lev  # sanity check
        df_pnl["value"] = df_pnl["value"] * leverage

        pnn = ("PNL_" + sig + neg) if pnl_name is None else pnl_name
        # Populating the signal dictionary is required for the display methods:
        self.signal_df[pnn] = dfw.loc[:, ["cid", "real_date", "sig"]]

        df_pnl["xcat"] = pnn
        if pnn in self.pnl_names:
            self.df = self.df[~(self.df["xcat"] == pnn)]
        else:
            self.pnl_names = self.pnl_names + [pnn]

        agg_df = pd.concat([self.df, df_pnl[self.df.columns]])
        self.df = agg_df.reset_index(drop=True)

        self.pnl_params[pnn] = PnLParams(
            pnl_name=pnn,
            signal=sig,
            sig_op=sig_op,
            sig_add=sig_add,
            sig_neg=sig_neg,
            rebal_freq=rebal_freq,
            rebal_slip=rebal_slip,
            vol_scale=vol_scale,
            neutral=neutral,
            thresh=thresh,
        )

    def make_long_pnl(
        self,
        vol_scale: Optional[float] = None,
        label: Optional[str] = None,
        leverage: float = 1.0,
    ):
        """
        The long-only returns will be computed which act as a basis for comparison
        against the signal-adjusted returns. Will take a long-only position in the
        category passed to the parameter 'self.ret'.

        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given.
            This is for comparative visualization and not out-of-sample, and is applied
            to the long-only position. Default is None.
        :param <str> label: associated label that will be mapped to the long-only
            DataFrame. The label will be used in the plotting graphic for plot_pnls().
            If a label is not defined, the default will be the name of the return
            category.
        :param <float> leverage: leverage applied to the raw signal when a `vol_scale` is
            not defined. Default is 1.0, i.e., position size is 1 or 100% of implied risk
            capital.
        """

        if vol_scale is not None:
            vol_err = (
                "The parameter `vol_scale` must be a numerical value greater than 0."
            )
            if not isinstance(vol_scale, (float, int)):
                raise TypeError(vol_err)
            elif vol_scale <= 0:
                raise ValueError(vol_err)

        else:
            err_lev = "`leverage` must be a numerical value greater than 0."
            if not isinstance(leverage, (float, int)):
                raise TypeError(err_lev)
            elif leverage <= 0:
                raise ValueError(err_lev)

        if label is None:
            label = self.ret

        dfx = self.df[self.df["xcat"].isin([self.ret])]

        df_long = self.long_only_pnl(
            dfw=dfx, vol_scale=vol_scale, label=label, leverage=leverage
        )

        self.df: pd.DataFrame = pd.concat([self.df, df_long])

        if label not in self.pnl_names:
            self.pnl_names = self.pnl_names + [label]

        self.df = self.df.reset_index(drop=True)

    @staticmethod
    def long_only_pnl(
        dfw: pd.DataFrame,
        vol_scale: float = None,
        label: str = None,
        leverage: float = 1.0,
    ):
        """
        Method used to compute the PnL accrued from simply taking a long-only position in
        the category, 'self.ret'. The returns from the category are not predicated on any
        exogenous signal.

        :param <pd.DataFrame> dfw:
        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given.
            This is for comparative visualization and not out-of-sample. Default is none.
        :param <str> label: associated label that will be mapped to the long-only
            DataFrame.
        :param <float> leverage: leverage applied to the raw signal when a `vol_scale` is
            not defined. Default is 1.0, i.e., position size is 1 or 100% of implied risk
            capital.

        :return <pd.DataFrame> panel_pnl: standardised dataframe containing exclusively
            the return category, and the long-only panel return.
        """
        lev_err = "`leverage` must be a numerical value greater than 0."

        dfw_long = dfw.reset_index(drop=True)

        panel_pnl = dfw_long.groupby(["real_date"]).sum(numeric_only=True)
        panel_pnl = panel_pnl.reset_index(level=0)
        panel_pnl["cid"] = "ALL"
        panel_pnl["xcat"] = label

        if vol_scale is not None:
            leverage = vol_scale * (panel_pnl["value"].std() * np.sqrt(261)) ** (-1)

        if not isinstance(leverage, Number) or leverage <= 0:
            raise TypeError(lev_err)
        panel_pnl["value"] = panel_pnl["value"] * leverage

        return panel_pnl[["cid", "xcat", "real_date", "value"]]

    def plot_pnls(
        self,
        pnl_cats: List[str] = None,
        pnl_cids: List[str] = ["ALL"],
        start: str = None,
        end: str = None,
        facet: bool = False,
        ncol: int = 3,
        same_y: bool = True,
        title: str = "Cumulative Naive PnL",
        title_fontsize: int = 20,
        xcat_labels: Union[List[str], dict] = None,
        xlab: str = "",
        ylab: str = "% of risk capital, no compounding",
        share_axis_labels: bool = True,
        figsize: Tuple = (12, 7),
        aspect: float = 1.7,
        height: float = 3,
        label_adj: float = 0.05,
        title_adj: float = 0.95,
        y_label_adj: float = 0.95,
    ) -> None:
        """
        Plot line chart of cumulative PnLs, single PnL, multiple PnL types per
        cross section, or multiple cross sections per PnL type.

        :param <List[str]> pnl_cats: list of PnL categories that should be plotted.
        :param <List[str]> pnl_cids: list of cross sections to be plotted;
            default is 'ALL' (global PnL).
            Note: one can only have multiple PnL categories or multiple cross sections,
            not both.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.
        :param <bool> facet: parameter to control whether each PnL series is plotted on
            its own respective grid using Seaborn's FacetGrid. Default is False and all
            series will be plotted in the same graph.
        :param <int> ncol: number of columns in facet grid. Default is 3. If the total
            number of PnLs is less than ncol, the number of columns will be adjusted on
            runtime.
        :param <bool> same_y: if True (default) all plots in facet grid share same y axis.
        :param <str> title: allows entering text for a custom chart header.
        :param <int> title_fontsize: font size for the title. Default is 20.
        :param <List[str]> xcat_labels: custom labels to be used for the PnLs.
        :param <str> xlab: label for x-axis of the plot (or subplots if faceted),
            default is None (empty string)..
        :param <str> ylab: label for y-axis of the plot (or subplots if faceted),
            default is '% of risk capital, no compounding'.
        :param <bool> share_axis_labels: if True (default) the axis labels are shared by
            all subplots in the facet grid.
        :param <tuple> figsize: tuple of plot width and height. Default is (12 , 7).
        :param <float> aspect: width-height ratio for plots in facet. Default is 1.7.
        :param <float> height: height of plots in facet. Default is 3.
        :param <float> label_adj: parameter that sets bottom of figure to fit the label.
            Default is 0.05.
        :param <float> title_adj: parameter that sets top of figure to accommodate title.
            Default is 0.95.
        :param <float> y_label_adj: parameter that sets left of figure to fit the y-label.
        """

        if pnl_cats is None:
            pnl_cats = self.pnl_names
        else:
            pnl_cats_error = (
                f"List of PnL categories expected - received {type(pnl_cats)}."
            )
            if not isinstance(pnl_cats, list):
                raise TypeError(pnl_cats_error)

            pnl_cats_copy = pnl_cats.copy()
            pnl_cats = [pnl for pnl in pnl_cats if pnl in self.pnl_names]

            dif = set(pnl_cats_copy).difference(set(pnl_cats))
            if dif:
                print(
                    f"The PnL(s) requested, {dif}, have not been defined on the "
                    f"Class. The defined PnL(s) are {self.pnl_names}."
                )
            elif len(pnl_cats) == 0:
                raise ValueError(
                    "There are not any valid PnL(s) to display given the " "request."
                )

        error_message = "Either pnl_cats or pnl_cids must be a list of length 1"
        if not ((len(pnl_cats) == 1) | (len(pnl_cids) == 1)):
            raise ValueError(error_message)

        # adjust ncols of the facetgrid if necessary
        if max([len(pnl_cats), len(pnl_cids)]) < ncol:
            ncol = max([len(pnl_cats), len(pnl_cids)])

        dfx = reduce_df(
            self.df, pnl_cats, pnl_cids, start, end, self.black, out_all=False
        )

        if max([len(pnl_cats), len(pnl_cids)]) < ncol:
            ncol = max([len(pnl_cats), len(pnl_cids)])

        error_message = (
            "The number of custom labels must match the defined number of "
            "categories in pnl_cats."
        )

        if isinstance(xcat_labels, dict) or xcat_labels is None:
            if xcat_labels is None:
                xcat_labels = pnl_cats.copy()
            else:
                if not len(xcat_labels) == len(pnl_cats):
                    raise ValueError(error_message)
                xcat_labels = [xcat_labels[pnl] for pnl in pnl_cats]

        elif isinstance(xcat_labels, list) and all(
            isinstance(item, str) for item in xcat_labels
        ):
            warnings.warn(
                "xcat_labels should be a dictionary with keys as pnl_cats and values as "
                "the custom labels. This will be enforced in a future version.",
            )
            if len(xcat_labels) != len(pnl_cats):
                raise ValueError(error_message)
        else:
            raise TypeError(
                "xcat_labels should be a dictionary with keys as pnl_cats and values as "
                "the custom labels."
            )

        no_cids = len(pnl_cids)

        sns.set_theme(
            style="whitegrid", palette="colorblind", rc={"figure.figsize": figsize}
        )

        if no_cids == 1:
            plot_by = "xcat"
            col_order = pnl_cats
            labels = xcat_labels
            legend_title = "PnL Category(s)"
        else:
            plot_by = "cid"
            col_order = pnl_cids
            if xcat_labels is not None:
                labels = pnl_cids.copy()
            legend_title = "Cross Section(s)"

        dfx["cum_value"] = dfx.groupby(plot_by).cumsum(numeric_only=True)

        if facet:
            fg = sns.FacetGrid(
                data=dfx,
                col=plot_by,
                col_wrap=ncol,
                sharey=same_y,
                aspect=aspect,
                height=height,
                col_order=col_order,
                legend_out=True,
            )
            fg.fig.suptitle(
                title,
                fontsize=title_fontsize,
            )

            fg.fig.subplots_adjust(top=title_adj, bottom=label_adj, left=y_label_adj)

            fg.map_dataframe(
                sns.lineplot,
                x="real_date",
                y="cum_value",
                hue=plot_by,
                hue_order=col_order,
                estimator=None,
                lw=1,
            )
            for ix, ax in enumerate(fg.axes.flat):
                ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
                if no_cids == 1:
                    ax.set_title(xcat_labels[ix])

            if no_cids > 1:
                fg.set_titles(row_template="", col_template="{col_name}")

            if share_axis_labels:
                fg.set_axis_labels("", "")
                fg.fig.supxlabel(xlab)
                fg.fig.supylabel(ylab)
            else:
                fg.set_axis_labels(xlab, ylab)

        else:
            fg = sns.lineplot(
                data=dfx,
                x="real_date",
                y="cum_value",
                hue=plot_by,
                hue_order=col_order,
                estimator=None,
                lw=1,
            )
            leg = fg.axes.get_legend()
            plt.title(title, fontsize=title_fontsize)
            plt.legend(
                labels=labels,
                title=legend_title,
            )
            plt.xlabel(xlab)
            plt.ylabel(ylab)

        if no_cids == 1:
            if facet:
                labels = labels[::-1]
        else:
            labels = labels[::-1]

        plt.axhline(y=0, color="black", linestyle="--", lw=1)
        plt.show()

    def signal_heatmap(
        self,
        pnl_name: str,
        pnl_cids: List[str] = None,
        start: str = None,
        end: str = None,
        freq: str = "M",
        title: str = "Average applied signal values",
        x_label: str = "",
        y_label: str = "",
        figsize: Optional[Tuple[float, float]] = None,
    ):
        """
        Display heatmap of signals across times and cross-sections.

        :param <str> pnl_name: name of naive PnL whose signals are displayed.
            N.B.: Signal is here is the value that actually determines
            the concurrent PnL.
        :param <List[str]> pnl_cids: cross-sections. Default is all available.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.
        :param <str> freq: frequency for which signal average is displayed.
            Default is monthly ('m'). The only alternative is quarterly ('q').
        :param <str> title: allows entering text for a custom chart header.
        :param <str> x_label: label for the x-axis. Default is None.
        :param <str> y_label: label for the y-axis. Default is None.
        :param <(float, float)> figsize: width and height in inches.
            Default is (14, number of cross sections).
        """

        if not isinstance(pnl_name, str):
            raise TypeError("The method expects to receive a single PnL name.")
        error_cats = (
            f"The PnL passed to 'pnl_name' parameter is not defined. The "
            f"possible options are {self.pnl_names}."
        )
        if pnl_name not in self.pnl_names:
            raise ValueError(error_cats)

        err_msg = "Defined time-period must be monthly ('m') or quarterly ('q')"
        freq = _map_to_business_day_frequency(freq)
        if not freq.startswith(("BQ", "BM")):
            raise ValueError(err_msg)

        err_cids = f"Cross-sections not available. Available cids are:" f"{self.cids}."

        if pnl_cids is None:
            pnl_cids = self.cids
        else:
            if not set(pnl_cids) <= set(self.cids):
                raise ValueError(err_cids)

        for vname, lbl in zip(["x_label", "y_label"], [x_label, y_label]):
            if not isinstance(lbl, str):
                raise TypeError(f"<str> expected for `{vname}` - received {type(lbl)}.")

        dfx: pd.DataFrame = self.signal_df[pnl_name]
        dfw: pd.DataFrame = dfx.pivot(index="real_date", columns="cid", values="sig")
        dfw: pd.DataFrame = dfw[pnl_cids]

        if start is None:
            start = dfw.index[0]
        elif end is None:
            end = dfw.index[-1]

        dfw = dfw.truncate(before=start, after=end)

        dfw = dfw.resample(freq).mean()

        if figsize is None:
            figsize = (14, len(pnl_cids))

        fig, ax = plt.subplots(figsize=figsize)
        dfw = dfw.transpose()
        dfw.columns = [str(d.strftime("%d-%m-%Y")) for d in dfw.columns]
        sns.heatmap(dfw, cmap="vlag_r", center=0)

        ax.set(xlabel=x_label, ylabel=y_label)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(title, fontsize=14)

        plt.show()

    def agg_signal_bars(
        self,
        pnl_name: str,
        freq: str = "m",
        metric: str = "direction",
        title: str = None,
        y_label: str = "Sum of Std. across the Panel",
    ):
        """
        Display aggregate signal strength and - potentially - direction.

        :param <str> pnl_name: name of the PnL whose signal is to be visualized.
            N.B.: The referenced signal corresponds to the series that determines the
            concurrent PnL.
        :param <str> freq: frequency at which the signal is visualized. Default is
            monthly ('m'). The alternative is quarterly ('q').
        :param <str> metric: the type of signal value. Default is "direction".
            Alternative is "strength".
        :param <str> title: allows entering text for a custom chart header. Default will
            be "Directional Bar Chart of <pnl_name>.".
        :param <str> y_label: label for the y-axis. Default is the sum of standard
            deviations across the panel corresponding to the default signal
            transformation: 'zn_score_pan'.

        """

        assert isinstance(pnl_name, str), (
            "The method expects to receive a single " "PnL name."
        )
        error_cats = (
            f"The PnL passed to 'pnl_name' parameter is not defined. The "
            f"possible options are {self.pnl_names}."
        )
        assert pnl_name in self.pnl_names, error_cats

        error_time = "Defined time-period must either be monthly, m, or quarterly, q."
        freq = _map_to_business_day_frequency(freq)
        if not freq.startswith(("BQ", "BM")):
            raise ValueError(error_time)

        metric_error = "The metric must either be 'direction' or 'strength'."
        assert metric in ["direction", "strength"], metric_error

        if title is None:
            title = f"Directional Bar Chart of {pnl_name}."

        dfx: pd.DataFrame = self.signal_df[pnl_name]
        dfw: pd.DataFrame = dfx.pivot(index="real_date", columns="cid", values="sig")

        # The method allows for visually understanding the overall direction of the
        # aggregate signal but also gaining insight into the proportional exposure to the
        # respective signal by measuring the absolute value, the size of the signal.
        # Is the PnL value generated by large returns or a large signal, position ?
        if metric == "strength":
            dfw = dfw.abs()

        dfw = dfw.resample(freq).mean()
        # Sum across the timestamps to compute the aggregate signal according to the
        # down-sampling frequency.
        df_s = dfw.sum(axis=1)
        index = np.array(df_s.index)
        df_signal = pd.DataFrame(
            data=df_s.to_numpy(), columns=["aggregate_signal"], index=index
        )

        df_signal = df_signal.reset_index(level=0)
        df_signal = df_signal.rename({"index": ""}, axis="columns")
        dates = [pd.Timestamp(d) for d in df_signal[""]]
        df_signal[""] = np.array(dates)

        plt.style.use("ggplot")

        fig, ax = plt.subplots()
        df_signal.plot.bar(
            x="", y="aggregate_signal", ax=ax, title=title, ylabel=y_label, legend=False
        )

        ticklabels = [""] * len(df_signal)
        skip = len(df_signal) // 12
        ticklabels[::skip] = df_signal[""].iloc[::skip].dt.strftime("%Y-%m-%d")
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))

        fig.autofmt_xdate()

        def fmt(x, pos=0, max_i=len(ticklabels) - 1):
            i = int(x)
            i = 0 if i < 0 else max_i if i > max_i else i
            return dates[i]

        ax.fmt_xdata = fmt
        plt.show()

    def evaluate_pnls(
        self,
        pnl_cats: List[str],
        pnl_cids: List[str] = ["ALL"],
        start: Optional[str] = None,
        end: Optional[str] = None,
        label_dict: Optional[Dict[str, str]] = None,
    ):
        """
        Table of key PnL statistics.

        :param <List[str]> pnl_cats: list of PnL categories that should be plotted.
        :param <List[str]> pnl_cids: list of cross-sections to be plotted; default is
            'ALL' (global PnL).
            Note: one can only have multiple PnL categories or multiple cross-sections,
            not both.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.
        :param <dict[str, str]> label_dict: dictionary with keys as pnl_cats and values
            as new labels for the PnLs.

        :return <pd.DataFrame>: standardized DataFrame with key PnL performance
            statistics.
        """

        error_cids = "List of cross-sections expected."
        error_xcats = "List of PnL categories expected."
        assert isinstance(pnl_cids, list), error_cids
        assert isinstance(pnl_cats, list), error_xcats
        assert all([isinstance(elem, str) for elem in pnl_cids]), error_cids
        assert all([isinstance(elem, str) for elem in pnl_cats]), error_xcats

        if pnl_cats is None:
            # The field, self.pnl_names, is a data structure that stores the name of the
            # category assigned to PnL values. Each time make_pnl() method is called, the
            # computed DataFrame will have an associated category established by the
            # logical method: ('PNL_' + sig) if pnl_name is None else pnl_name. Each
            # category will be held in the data structure.
            pnl_cats = self.pnl_names
        else:
            if not set(pnl_cats) <= set(self.pnl_names):
                missing = [pnl for pnl in pnl_cats if pnl not in self.pnl_names]
                pnl_error = (
                    f"Received PnL categories have not been defined. The PnL "
                    f"category(s) which has not been defined is: {missing}. "
                    f"The produced PnL category(s) are {self.pnl_names}."
                )
                raise ValueError(pnl_error)

        assert (len(pnl_cats) == 1) | (len(pnl_cids) == 1)

        dfx = reduce_df(
            self.df, pnl_cats, pnl_cids, start, end, self.black, out_all=False
        )

        groups = "xcat" if len(pnl_cids) == 1 else "cid"
        stats = [
            "Return %",
            "St. Dev. %",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max 21-Day Draw %",
            "Max 6-Month Draw %",
            "Peak to Trough Draw %",
            "Top 5% Monthly PnL Share",
            "Traded Months",
        ]

        # If benchmark tickers have been passed into the Class and if the tickers are
        # present in self.dfd.
        benchmark_tickers = []

        if self.bm_bool and bool(self._bm_dict):
            benchmark_tickers = list(self._bm_dict.keys())
            for bm in benchmark_tickers:
                stats.insert(len(stats) - 1, f"{bm} correl")

        dfw = dfx.pivot(index="real_date", columns=groups, values="value")
        df = pd.DataFrame(columns=dfw.columns, index=stats)

        df.iloc[0, :] = dfw.mean(axis=0) * 261
        df.iloc[1, :] = dfw.std(axis=0) * np.sqrt(261)
        df.iloc[2, :] = df.iloc[0, :] / df.iloc[1, :]
        dsd = dfw.apply(lambda x: np.sqrt(np.sum(x[x < 0] ** 2) / len(x))) * np.sqrt(
            261
        )
        df.iloc[3, :] = df.iloc[0, :] / dsd
        df.iloc[4, :] = dfw.rolling(21).sum().min()
        df.iloc[5, :] = dfw.rolling(6 * 21).sum().min()

        cum_pnl = dfw.cumsum()
        high_watermark = cum_pnl.cummax()
        drawdown = high_watermark - cum_pnl

        df.iloc[6, :] = -drawdown.max()

        mfreq = _map_to_business_day_frequency("M")
        monthly_pnl = dfw.resample(mfreq).sum()
        total_pnl = monthly_pnl.sum(axis=0)
        top_5_percent_cutoff = int(np.ceil(len(monthly_pnl) * 0.05))
        top_months = pd.DataFrame(columns=monthly_pnl.columns)
        for column in monthly_pnl.columns:
            top_months[column] = (
                monthly_pnl[column]
                .nlargest(top_5_percent_cutoff)
                .reset_index(drop=True)
            )

        df.iloc[7, :] = top_months.sum() / total_pnl

        if len(benchmark_tickers) > 0:
            bm_df = pd.concat(list(self._bm_dict.values()), axis=1)
            for i, bm in enumerate(benchmark_tickers):
                index = dfw.index.intersection(bm_df.index)
                correlation = dfw.loc[index].corrwith(
                    bm_df.loc[index].iloc[:, i], axis=0, method="pearson", drop=True
                )
                df.iloc[8 + i, :] = correlation

        mfreq = _map_to_business_day_frequency("M")
        df.iloc[8 + len(benchmark_tickers), :] = dfw.resample(mfreq).sum().count()

        if label_dict is not None:
            if not isinstance(label_dict, dict):
                raise TypeError("label_dict must be a dictionary.")
            if not all([isinstance(k, str) for k in label_dict.keys()]):
                raise TypeError("Keys in label_dict must be strings.")
            if not all([isinstance(v, str) for v in label_dict.values()]):
                raise TypeError("Values in label_dict must be strings.")
            if len(label_dict) != len(df.columns):
                raise ValueError(
                    "label_dict must have the same number of keys as columns in the "
                    "DataFrame."
                )
            df.rename(index=label_dict, inplace=True)
            df = df[label_dict.values()]

        return df

    def print_pnl_names(self):
        """
        Print list of names of available PnLs in the class instance.
        """

        print(self.pnl_names)

    def pnl_df(self, pnl_names: List[str] = None, cs: bool = False):
        """
        Return dataframe with PnLs.

        :param <List[str]> pnl_names: list of names of PnLs to be returned.
            Default is 'ALL'.
        :param <bool> cs: inclusion of cross section PnLs. Default is False.

        :return <pd.DataFrame>: custom DataFrame with PnLs
        """
        selected_pnls = pnl_names if pnl_names is not None else self.pnl_names

        filter_1 = self.df["xcat"].isin(selected_pnls)
        filter_2 = self.df["cid"] == "ALL" if not cs else True

        return self.df[filter_1 & filter_2]

    def _winsorize(self, df: pd.DataFrame, thresh: float):
        if thresh is not None:
            df.clip(lower=-thresh, upper=thresh, inplace=True)


def create_results_dataframe(
    title: str,
    df: pd.DataFrame,
    ret: str,
    sigs: Union[str, List[str]],
    cids: Union[str, List[str]],
    sig_ops: Union[str, List[str]],
    sig_adds: Union[float, List[float]],
    neutrals: Union[str, List[str]],
    threshs: Union[float, List[float]],
    bm: str = None,
    sig_negs: Union[bool, List[bool]] = None,
    cosp: bool = False,
    start: str = None,
    end: str = None,
    blacklist: dict = None,
    freqs: Union[str, List[str]] = "M",
    agg_sigs: Union[str, List[str]] = "last",
    sigs_renamed: dict = None,
    fwin: int = 1,
    slip: int = 0,
):
    """
    Create a DataFrame with key performance metrics for the signals and PnLs.

    :param <str> title: title of the DataFrame.
    :param <pd.DataFrame> df: DataFrame with the data.
    :param <str> ret: name of the return signal.
    :param <Union[str, List[str]> sigs: name of the comparative signal(s).
    :param <Union[str, List[str]> cids: name of the cross-section(s).
    :param <Union[str, List[str]> sig_ops: operation(s) to be applied to the signal(s).
    :param <Union[float, List[float]> sig_adds: value(s) to be added to the signal(s).
    :param <Union[str, List[str]> neutrals: neutralization method(s) to be applied.
    :param <Union[float, List[float]> threshs: threshold(s) to be applied to the signal(s).
    :param <str> bm: name of the benchmark signal.
    :param <Union[bool, List[bool]> sig_negs: whether the signal(s) should be negated.
    :param <bool> cosp: whether the signals should be cross-sectionally standardized.
    :param <str> start: start date of the analysis.
    :param <str> end: end date of the analysis.
    :param <dict> blacklist: dictionary with the blacklisted dates.
    :param <Union[str, List[str]> freqs: frequency of the rebalancing.
    :param <Union[str, List[str]> agg_sigs: aggregation method(s) for the signal(s).
    :param <dict> sigs_renamed: dictionary with the renamed signals.
    :param <int> fwin: frequency of the rolling window.
    :param <int> slip: slippage to be applied to the PnLs.

    :return <pd.DataFrame>: DataFrame with the performance metrics.
    """
    # Get the signals table and isolate relevant performance metrics

    def check_list_type(type, var):
        return isinstance(var, list) and all(isinstance(item, type) for item in var)

    if not isinstance(ret, str):
        raise TypeError("The return signal must be a string.")
    if not isinstance(sigs, str) and not check_list_type(str, sigs):
        raise TypeError("The signals must be a string or a list of strings.")
    if not isinstance(cids, str) and not check_list_type(str, cids):
        raise TypeError("The cids must be a string or a list of strings.")
    if not isinstance(sig_ops, str) and not check_list_type(str, sig_ops):
        raise TypeError("The signal operations must be a string or a list of strings.")
    if not isinstance(sig_adds, (float, int)) and not check_list_type(
        (float, int), sig_adds
    ):
        raise TypeError("The signal additions must be a float or a list of floats.")
    if not isinstance(neutrals, str) and not check_list_type(str, neutrals):
        raise TypeError("The neutralizations must be a string or a list of strings.")
    if not isinstance(threshs, (float, int)) and not check_list_type(
        (float, int), threshs
    ):
        raise TypeError("The thresholds must be a float or a list of floats.")

    srr = SignalReturnRelations(
        df=df,
        rets=ret,
        sigs=sigs,
        cids=cids,
        sig_neg=sig_negs,
        cosp=cosp,
        start=start,
        end=end,
        blacklist=blacklist,
        freqs=freqs,
        agg_sigs=agg_sigs,
        fwin=fwin,
        slip=slip,
    )
    sigs_df = (
        srr.multiple_relations_table()
        .astype("float")[["accuracy", "bal_accuracy", "pearson", "kendall"]]
        .round(3)
    )

    # Get the evaluated PnL statistics and isolate relevant performance metrics
    freq_dict = {
        "D": "daily",
        "W": "weekly",
        "M": "monthly",
    }
    pnl_freq = lambda x: freq_dict[x] if x in freq_dict.keys() else "monthly"

    if sig_negs is None:
        sig_negs = [False] * len(sigs)

    pnl = NaivePnL(
        df=df,
        ret=ret,
        sigs=sigs,
        cids=cids,
        bms=bm,
        start=start,
        end=end,
        blacklist=blacklist,
    )
    for idx, sig in enumerate(sigs):
        pnl.make_pnl(
            sig=sig,
            sig_op=(sig_ops if isinstance(sig_ops, str) else sig_ops[idx]),
            sig_add=(sig_adds if isinstance(sig_adds, (float, int)) else sig_adds[idx]),
            sig_neg=(sig_negs if isinstance(sig_negs, bool) else sig_negs[idx]),
            pnl_name=sig,
            rebal_freq=(
                pnl_freq(freqs) if isinstance(freqs, str) else pnl_freq(freqs[idx])
            ),
            rebal_slip=slip,
            vol_scale=10,
            neutral=(neutrals if isinstance(neutrals, str) else neutrals[idx]),
            thresh=(threshs if isinstance(threshs, (float, int)) else threshs[idx]),
        )

    if bm is not None:
        pnl_df = (
            pnl.evaluate_pnls(pnl_cats=sigs)
            .transpose()[["Sharpe Ratio", "Sortino Ratio", f"{bm} correl"]]
            .astype("float")
            .round(3)
        )
    else:
        pnl_df = (
            pnl.evaluate_pnls(pnl_cats=sigs)
            .transpose()[["Sharpe Ratio", "Sortino Ratio"]]
            .astype("float")
            .round(3)
        )

    # Sort pnl_df so that it matches the order of sigs
    pnl_df = pnl_df.reindex(sigs)

    sigs_df = sigs_df.reset_index(
        level=["Return", "Frequency", "Aggregation"], drop=True
    )

    if True in sig_negs:
        for sig in pnl_df.index:
            sig_neg = sig_negs[pnl_df.index.get_loc(sig)]
            if sig_neg:
                # Ensure each index is renamed only once
                pnl_df.rename(
                    index={sig: f"{sig}_NEG"},
                    inplace=True,
                )

    # Concatenate them and clean them
    res_df = pd.concat([sigs_df, pnl_df], axis=1)

    metric_map = {
        "Sharpe Ratio": "Sharpe",
        "Sortino Ratio": "Sortino",
        "accuracy": "Accuracy",
        "bal_accuracy": "Bal. Accuracy",
        "pearson": "Pearson",
        "kendall": "Kendall",
    }
    if bm is not None:
        metric_map[f"{bm} correl"] = "Market corr."

    res_df.rename(columns=metric_map, inplace=True)

    if sigs_renamed:
        res_df.rename(index=sigs_renamed, inplace=True)

    if PYTHON_3_8_OR_LATER:
        res_df = (
            res_df.style.format("{:.3f}")
            .set_caption(title)
            .set_table_styles(
                [
                    {
                        "selector": "caption",
                        "props": [("text-align", "center"), ("font-weight", "bold")],
                    }
                ]
            )
        )

    return res_df


@dataclass
class PnLParams:
    """
    Dataclass to store the parameters for the PnL creation.
    """

    signal: str
    sig_op: str
    sig_neg: bool
    sig_add: float
    pnl_name: str
    rebal_freq: str
    rebal_slip: int
    vol_scale: int
    neutral: str
    thresh: float


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "NZD", "USD", "EUR"]
    xcats = ["EQXR_NSA", "CRY", "GROWTH", "INFL", "DUXR"]

    cols_1 = ["earliest", "latest", "mean_add", "sd_mult"]
    df_cids = pd.DataFrame(index=cids, columns=cols_1)
    df_cids.loc["AUD", :] = ["2008-01-03", "2020-12-31", 0.5, 2]
    df_cids.loc["CAD", :] = ["2010-01-03", "2020-11-30", 0, 1]
    df_cids.loc["GBP", :] = ["2012-01-03", "2020-11-30", -0.2, 0.5]
    df_cids.loc["NZD"] = ["2002-01-03", "2020-09-30", -0.1, 2]
    df_cids.loc["USD"] = ["2015-01-03", "2020-12-31", 0.2, 2]
    df_cids.loc["EUR"] = ["2008-01-03", "2020-12-31", 0.1, 2]

    cols_2 = cols_1 + ["ar_coef", "back_coef"]

    df_xcats = pd.DataFrame(index=xcats, columns=cols_2)
    df_xcats.loc["EQXR_NSA"] = ["2000-01-03", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2010-01-03", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]
    df_xcats.loc["DUXR"] = ["2000-01-01", "2020-12-31", 0.1, 0.5, 0, 0.1]

    black = {"AUD": ["2006-01-01", "2015-12-31"], "GBP": ["2022-01-01", "2100-01-01"]}
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Instantiate a new instance to test the long-only functionality.
    # Benchmarks are used to calculate correlation against PnL series.
    pnl = NaivePnL(
        dfd,
        ret="EQXR_NSA",
        sigs=["CRY", "GROWTH", "INFL"],
        cids=cids,
        start="2000-01-01",
        blacklist=black,
        bms=["EUR_EQXR_NSA", "USD_EQXR_NSA"],
    )

    pnl.make_pnl(
        sig="GROWTH",
        sig_op="zn_score_pan",
        sig_neg=True,
        sig_add=0.5,
        rebal_freq="monthly",
        vol_scale=5,
        rebal_slip=1,
        min_obs=250,
        thresh=2,
    )

    pnl.make_pnl(
        sig="INFL",
        sig_op="zn_score_pan",
        sig_neg=True,
        sig_add=0.5,
        rebal_freq="monthly",
        vol_scale=5,
        rebal_slip=1,
        min_obs=250,
        thresh=2,
    )

    pnl.make_long_pnl(vol_scale=10, label="Long")

    df_eval = pnl.evaluate_pnls(
        pnl_cats=["PNL_GROWTH_NEG", "PNL_INFL_NEG"],
        start="2015-01-01",
        end="2020-12-31",
    )

    pnl.signal_heatmap(pnl_name="PNL_GROWTH_NEG", pnl_cids=cids, freq="m")

    print(df_eval)

    pnl.agg_signal_bars(
        pnl_name="PNL_GROWTH_NEG",
        freq="m",
        metric="direction",
        title=None,
    )
    pnl.plot_pnls(
        pnl_cats=["PNL_GROWTH_NEG", "Long"],
        title_fontsize=60,
        xlab="date",
        ylab="%",
    )
    pnl.plot_pnls(
        pnl_cats=["PNL_GROWTH_NEG", "Long"],
        facet=False,
        xcat_labels=["S_1", "S_2"],
        xlab="date",
        ylab="%",
    )
    pnl.plot_pnls(
        pnl_cats=["PNL_GROWTH_NEG", "Long"], facet=True, xcat_labels=["S_1", "S_2"]
    )
    pnl.plot_pnls(
        pnl_cats=["PNL_GROWTH_NEG", "Long"],
        facet=True,
    )

    pnl.plot_pnls(pnl_cats=["PNL_GROWTH_NEG"], pnl_cids=cids, xcat_labels=None)

    pnl.plot_pnls(
        pnl_cats=["PNL_GROWTH_NEG"], pnl_cids=cids, facet=True, xcat_labels=None
    )

    pnl.plot_pnls(
        pnl_cats=["PNL_GROWTH_NEG"],
        pnl_cids=cids,
        same_y=True,
        facet=True,
        xcat_labels=None,
        share_axis_labels=False,
        xlab="Date",
        ylab="PnL",
        y_label_adj=0.1,
    )

    results_eq_ols = create_results_dataframe(
        title="Performance metrics, PARITY vs OLS, equity",
        df=dfd,
        ret="EQXR_NSA",
        sigs=["GROWTH", "INFL", "CRY", "DUXR"],
        cids=cids,
        sig_ops="zn_score_pan",
        sig_adds=0,
        neutrals="zero",
        threshs=2,
        sig_negs=[True, False, False, True],
        bm="USD_EQXR_NSA",
        cosp=True,
        start="2004-01-01",
        freqs="M",
        agg_sigs="last",
        slip=1,
    )
    print(results_eq_ols.data)
