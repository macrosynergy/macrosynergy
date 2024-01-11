"""
Module for analysing and visualizing signal and a return series.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as skm
from scipy import stats
from typing import List, Union, Tuple, Dict, Any, Optional

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import (
    apply_slip as apply_slip_util,
    reduce_df,
    categories_df,
)
import macrosynergy.visuals as msv


class SignalReturnRelations:

    """
    Class for analysing and visualizing signal and a return series.

    :param <pd.Dataframe> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value.
    :param <str, List[str]> rets: one or several target return categories.
    :param <str, List[str]> sigs: list of signal categories to be considered for which
        detailed relational statistics can be calculated.
    :param <bool, List[bool]> sig_neg: if set to True puts the signal in negative terms
        for all analysis. Default is False.
    :param <bool> cosp: If True the comparative statistics are calculated only for the
        "communal sample periods", i.e. periods and cross-sections that have values
        for all compared signals. Default is False.
    :param <str> start: earliest date in ISO format. Default is None in which case the
        earliest date available will be used.
    :param <str> end: latest date in ISO format. Default is None in which case the
        latest date in the df will be used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the data frame. If one cross-section has several blacklist periods append numbers
        to the cross-section code.
    :param <str, List[str]> freqs: letters denoting all frequencies at which the series
        may be sampled. This must be a selection of 'D', 'W', 'M', 'Q', 'A'. Default is
        only 'M'. The return series will always be summed over the sample period. The
        signal series will be aggregated according to the values of `agg_sigs`.
    :param <str, List[str]> agg_sigs: aggregation method applied to the signal values in
        down-sampling. The default is "last". Alternatives are "mean", "median" and "sum".
        If a single aggregation type is chosen for multiple signal categories it is
        applied to all of them.
    :param <int> fwin: forward window of return category in base periods. Default is 1.
        This conceptually corresponds to the holding period of a position in
        accordance with the signal.
    :param <int> slip: implied slippage of feature availability for relationship with
        the target category. This mimics the relationship between trading signals and
        returns, which is often characterized by a delay due to the setup of of positions.
        Technically, this is a negative lag (early arrival) of the target category
        in working days prior to any frequency conversion. Default is 0.
    :param <str> ret: return category. (Argument is deprecated)
    :param <str> sig: primary signal category for which detailed relational statistics
        can be calculated. (Argument is deprecated)
    :param <str, List[str]> rival_sigs: "rival signals" for which basic relational
        statistics can be calculated for comparison with the primary signal category. The
        table, if rival signals are defined, will be generated upon instantiation of the
        object. (Argument is deprecated)
        N.B.: parameters set for sig, such as sig_neg, freq, and agg_sig are equally
        applied to all rival signals.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
        The return series will always be summed over the sample period.
        The signal series will be aggregated according to the value of agg_sig.
        (Argument is deprecated)
    :param <str> agg_sig: aggregation method applied to the signal values in down-
        sampling. The default is "last".
        If defined, the additional signals will also use the same aggregation method for
        any down-sampling. (Argument is deprecated)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rets: Union[str, List[str]] = None,
        sigs: Union[str, List[str]] = None,
        ret: Union[str, List[str]] = None,
        sig: Union[str, List[str]] = None,
        rival_sigs: Union[str, List[str]] = None,
        cids: Union[str, List[str]] = None,
        sig_neg: Union[bool, List[bool]] = False,
        cosp: bool = False,
        start: str = None,
        end: str = None,
        blacklist: dict = None,
        freqs: Union[str, List[str]] = "M",
        agg_sigs: Union[str, List[str]] = "last",
        freq: str = None,
        agg_sig: str = None,
        fwin: int = 1,
        slip: int = 0,
    ):
        if rets is None:
            if ret is None:
                raise ValueError("Target return must be defined.")
            else:
                warnings.warn(
                    "Parameter 'ret' is deprecated and will be removed in v0.1.0. "
                    "Please use parameter rets instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                rets = ret
        if sigs is None:
            if sig is None:
                raise ValueError("Signal must be defined.")
            else:
                warnings.warn(
                    "Parameter 'sig' is deprecated and will be removed in v0.1.0. "
                    "Please use parameter sigs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                sigs = sig
        if freq is not None:
            warnings.warn(
                "Parameter 'freq' is deprecated and will be removed in v0.1.0. Please"
                "use parameter freqs instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            freqs = freq
        if agg_sig is not None:
            warnings.warn(
                "Parameter 'agg_sig' is deprecated and will be removed in v0.1.0. Please"
                "use parameter agg_sigs instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            agg_sigs = agg_sig

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"DataFrame expected and not {type(df)}.")

        required_columns = ["cid", "xcat", "real_date", "value"]

        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                "Dataframe columns must be of value: 'cid', 'xcat','real_date' and  \
                'value'"
            )

        df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

        self.dic_freq = {
            "D": "daily",
            "W": "weekly",
            "M": "monthly",
            "Q": "quarterly",
            "A": "annual",
        }

        freq_error = f"Frequency parameter must be one of {list(self.dic_freq.keys())}."
        if isinstance(freqs, list):
            seen = set()
            self.freqs = []
            for f in freqs:
                if not f in self.dic_freq.keys():
                    raise ValueError(freq_error)
                else:
                    if f not in seen:
                        seen.add(f)
                        self.freqs.append(f)

        else:
            if not freqs in self.dic_freq.keys():
                raise ValueError(freq_error)
            else:
                self.freqs = [freqs]

        self.metrics = [
            "accuracy",
            "bal_accuracy",
            "pos_sigr",
            "pos_retr",
            "pos_prec",
            "neg_prec",
            "pearson",
            "pearson_pval",
            "kendall",
            "kendall_pval",
        ]

        self.rets = rets
        # self.freqs = list(set(freqs))  # Remove duplicate values from freqs

        if not isinstance(cosp, bool):
            raise TypeError(f"<bool> object expected and not {type(cosp)}.")

        self.cosp = cosp
        self.start = start
        self.end = end
        self.blacklist = blacklist
        self.fwin = fwin

        if isinstance(cids, str):
            self.cids = [cids]
        else:
            self.cids = cids

        self.sigs = sigs
        self.slip = slip
        self.agg_sigs = agg_sigs
        self.xcats = list(df["xcat"].unique())
        self.df = df
        self.original_df = df.copy()
        self.rival_sigs = rival_sigs

        self.df = df.copy()

        if not self.is_list_of_strings(rets):
            self.rets = [rets]

        if not self.is_list_of_strings(sigs):
            self.sigs = [sigs]

        if not self.is_list_of_strings(agg_sigs):
            self.agg_sigs = [agg_sigs]

        if not self.is_list_of_strings(freqs):
            self.freqs = [freqs]

        for sig in self.sigs:
            assert (
                sig in self.xcats
            ), "Primary signal must be available in the DataFrame."

        for ret in self.rets:
            assert (
                ret in self.xcats
            ), "Target return must be available in the DataFrame."

        # self.xcats = self.sig + self.ret

        self.signs = sig_neg if isinstance(sig_neg, list) else [sig_neg]
        for sign in self.signs:
            if not sign in [False, True]:
                raise TypeError("Sign must be either False or True.")

        if len(self.signs) < len(self.sigs):
            self.signs.extend([False] * (len(self.sigs) - len(self.signs)))

        if len(self.signs) > len(self.sigs):
            raise ValueError("Signs must have a length less than or equal to signals")
        self.signals = self.sigs

        assert (
            self.sigs[0] in self.xcats
        ), "Primary signal must be available in the DataFrame."

        signals = [self.sigs[0]]

        if rival_sigs is not None:
            r_sigs_warning = (
                "Parameter 'rival_sigs' is deprecated and will be removed "
                "in v0.1.0. Please specify the rival signals as part of the list of "
                "feature "
                "signals in the argument 'sigs'."
            )
            warnings.warn(
                r_sigs_warning,
                DeprecationWarning,
                stacklevel=2,
            )

        if len(self.sigs) > 1:
            rival_sigs = self.sigs[1:]
        if rival_sigs is not None:
            r_sigs_error = "Signal or list of signals expected."
            assert isinstance(rival_sigs, (str, list)), r_sigs_error

            r_sigs = [rival_sigs] if isinstance(rival_sigs, str) else rival_sigs

            intersection = set(self.xcats).intersection(r_sigs)
            missing = set(r_sigs).difference(intersection)

            rival_error = (
                f"The additional signals must be present in the defined "
                f"DataFrame. It is currently missing, {missing}."
            )
            assert set(r_sigs).issubset(set(self.xcats)), rival_error
            signals += r_sigs

        self.signals = signals
        # self.xcats = self.signals + [self.ret[0]]

        self.manipulate_df(
            xcat=self.signals + [self.rets[0]],
            freq=self.freqs[0],
            agg_sig=self.agg_sigs[0],
            sig=self.sigs[0],
        )

        if len(self.signals) > 1:
            self.df_sigs = self.__rival_sigs__(self.rets[0])

        self.sigs[0] = self.new_sig[0]

        self.df_cs = self.__output_table__(
            cs_type="cids", ret=self.rets[0], sig=self.sigs[0]
        )
        self.df_ys = self.__output_table__(
            cs_type="years", ret=self.rets[0], sig=self.sigs[0]
        )

        self.sigs[0] = self.revert_negation(self.sigs[0])

    def __rival_sigs__(self, ret):
        """
        Produces the panel-level table for the additional signals.
        """

        df_out = pd.DataFrame(index=self.signals, columns=self.metrics)
        df = self.df

        for s in self.signals:
            # Entire panel will be passed in.
            df_out = self.__table_stats__(
                df_segment=df, df_out=df_out, segment=s, signal=s, ret=ret
            )

        return df_out

    def signals_table(self, sigs: List[str] = None):
        """
        Output table on relations of various signals with the target return.

        :param <List[str]> sigs: signal categories to be included in the panel-level
            table. Default is None and all present signals will be displayed. Alternative
            is a valid subset of the possible categories. Primary signal must be passed
            if to be included.

        NB.:
        Analysis will be based exclusively on the panel level. Will only return a table
        if rival signals have been defined upon instantiation. If the communal sample
        parameter has been set to True, all signals will be aligned on the individual
        cross-sectional level.
        """

        try:
            df_sigs = self.df_sigs.round(decimals=3)
        except Exception:
            error_msg = "Additional signals have not been defined on the instance."
            raise AttributeError(error_msg)
        else:
            # Set to all available signals.
            sigs = self.signals if sigs is None else sigs

            assert isinstance(sigs, list), "List of signals expected."

            sigs_error = (
                f"The requested signals must be a subset of the primary plus "
                f"additional signals received, {self.signals}."
            )
            assert set(sigs).issubset(set(self.signals)), sigs_error

            return df_sigs.loc[sigs, :]

    def cross_section_table(self):
        """Output table on relations across sections and the panel."""
        return self.df_cs.round(decimals=3)

    def yearly_table(self):
        """Output table on relations across sections and the panel."""
        return self.df_ys.round(decimals=3)

    @staticmethod
    def __yaxis_lim__(accuracy_df: pd.DataFrame):
        """Determines the range the y-axis is defined over.

        :param <pd.DataFrame> accuracy_df: two dimensional DataFrame with accuracy &
            balanced accuracy columns.

        N.B.: The returned range will always be below 0.5.
        """
        y_axis = lambda min_correl: min_correl > 0.45
        min_value = accuracy_df.min().min()
        # Ensures any accuracy statistics greater than 0.5 are more pronounced given the
        # adjusted scale.
        y_input = 0.45 if y_axis(min_value) else min_value

        return y_input

    def accuracy_bars(
        self,
        ret: str = None,
        sig: str = None,
        freq: str = None,
        type: str = "cross_section",
        title: str = None,
        title_fontsize: int = 16,
        size: Tuple[float] = None,
        legend_pos: str = "best",
    ):
        """
        Plot bar chart for the overall and balanced accuracy metrics.

        :param <str> type: type of segment over which bars are drawn. Either
            "cross_section" (default), "years" or "signals".
        :param <str> title: chart header - default will be applied if none is chosen.
        :param <int> title_fontsize: font size of chart header. Default is 16.
        :param <Tuple[float]> size: 2-tuple of width and height of plot - default will be
            applied if none is chosen.
        :param <str> legend_pos: position of legend box. Default is 'best'.
            See the documentation of matplotlib.pyplot.legend.

        """

        assert type in ["cross_section", "years", "signals"]

        if freq is None:
            freq = self.freqs[0]

        if ret is None and sig is None:
            ret = self.rets[0]
            sig = self.sigs[0]
            if type == "cross_section":
                df_xs = self.df_cs
            elif type == "years":
                df_xs = self.df_ys
            else:
                df_xs = self.df_sigs
        else:
            if ret is None:
                ret = self.rets[0]
            if sig is None:
                sig = self.sigs[0]
            self.df = self.original_df.copy()
            self.manipulate_df(
                xcat=[sig, ret],
                freq=freq,
                agg_sig=self.agg_sigs[0],
                sig=sig,
            )
            sig = self.new_sig[0]
            if type == "cross_section":
                df_xs = self.__output_table__(cs_type="cids", ret=ret, sig=sig)
            elif type == "years":
                df_xs = self.__output_table__(cs_type="years", ret=ret, sig=sig)
            else:
                df_xs = self.__rival_sigs__(ret)

        dfx = df_xs[~df_xs.index.isin(["PosRatio"])]

        if title is None:
            refsig = "various signals" if type == "signals" else self.sigs
            title = (
                f"Accuracy for sign prediction of {self.rets} based on {refsig} "
                f"at {self.dic_freq[self.freqs[0]]} frequency."
            )
        if size is None:
            size = (np.max([dfx.shape[0] / 2, 8]), 6)

        sns.set_style("darkgrid")
        plt.figure(figsize=size)
        x_indexes = np.arange(dfx.shape[0])

        w = 0.4
        plt.bar(
            x_indexes - w / 2,
            dfx["accuracy"],
            label="Accuracy",
            width=w,
            color="lightblue",
        )
        plt.bar(
            x_indexes + w / 2,
            dfx["bal_accuracy"],
            label="Balanced Accuracy",
            width=w,
            color="steelblue",
        )

        plt.xticks(ticks=x_indexes, labels=dfx.index, rotation=0)
        plt.axhline(y=0.5, color="black", linestyle="-", linewidth=0.5)

        y_input = self.__yaxis_lim__(
            accuracy_df=dfx.loc[:, ["accuracy", "bal_accuracy"]]
        )

        plt.ylim(round(y_input, 2))

        plt.title(title, fontsize=title_fontsize)
        plt.legend(loc=legend_pos)
        plt.show()

    def correlation_bars(
        self,
        ret: str = None,
        sig: str = None,
        freq: str = None,
        type: str = "cross_section",
        title: str = None,
        title_fontsize: int = 16,
        size: Tuple[float] = None,
        legend_pos: str = "best",
    ):
        """
        Plot correlation coefficients and significance.

        :param <str> ret: return category. Default is the first return category.
        :param <str> sig: signal category. Default is the first signal category.
        :param <str> type: type of segment over which bars are drawn. Either
            "cross_section" (default), "years" or "signals".
        :param <str> title: chart header. Default will be applied if none is chosen.
        :param <int> title_fontsize: font size of chart header. Default is 16.
        :param <Tuple[float]> size: 2-tuple of width and height of plot.
            Default will be applied if none is chosen.
        :param <str> legend_pos: position of legend box. Default is 'best'.
            See matplotlib.pyplot.legend.

        """
        assert type in ["cross_section", "years", "signals"]

        if freq is None:
            freq = self.freqs[0]

        if ret is None and sig is None:
            ret = self.rets[0]
            sig = self.sigs[0]
            if type == "cross_section":
                df_xs = self.df_cs
            elif type == "years":
                df_xs = self.df_ys
            else:
                df_xs = self.df_sigs
        else:
            if ret is None:
                ret = self.rets[0]
            if sig is None:
                sig = self.sigs[0]
            self.df = self.original_df.copy()
            self.manipulate_df(
                xcat=[sig, ret],
                freq=freq,
                agg_sig=self.agg_sigs[0],
                sig=sig,
            )
            sig = self.new_sig[0]
            if type == "cross_section":
                df_xs = self.__output_table__(cs_type="cids", ret=ret, sig=sig)
            elif type == "years":
                df_xs = self.__output_table__(cs_type="years", ret=ret, sig=sig)
            else:
                df_xs = self.__rival_sigs__(ret)

        # Panel plus the cs_types.
        dfx = df_xs[~df_xs.index.isin(["PosRatio", "Mean"])]

        pprobs = np.array(
            [
                (1 - pv) * (np.sign(cc) + 1) / 2
                for pv, cc in zip(dfx["pearson_pval"], dfx["pearson"])
            ]
        )
        pprobs[pprobs == 0] = 0.01
        kprobs = np.array(
            [
                (1 - pv) * (np.sign(cc) + 1) / 2
                for pv, cc in zip(dfx["kendall_pval"], dfx["kendall"])
            ]
        )

        kprobs[kprobs == 0] = 0.01

        if title is None:
            refsig = "various signals" if type == "signals" else sig
            title = (
                f"Positive correlation probability of {ret} "
                f"and lagged {refsig} at {self.dic_freq[freq]} frequency."
            )
        if size is None:
            size = (np.max([dfx.shape[0] / 2, 8]), 6)

        sns.set_style("darkgrid")
        plt.figure(figsize=size)
        x_indexes = np.arange(len(dfx.index))
        w = 0.4
        plt.bar(x_indexes - w / 2, pprobs, label="Pearson", width=w, color="lightblue")
        plt.bar(x_indexes + w / 2, kprobs, label="Kendall", width=w, color="steelblue")
        plt.xticks(ticks=x_indexes, labels=dfx.index, rotation=0)

        plt.axhline(
            y=0.95,
            color="orange",
            linestyle="--",
            linewidth=0.5,
            label="95% probability",
        )
        plt.axhline(
            y=0.99, color="red", linestyle="--", linewidth=0.5, label="99% probability"
        )

        plt.title(title, fontsize=title_fontsize)
        plt.legend(loc=legend_pos)
        plt.show()

    @staticmethod
    def __slice_df__(df: pd.DataFrame, cs: str, cs_type: str):
        """
        Slice DataFrame by year, cross-section, or use full panel.

        :param <pd.DataFrame> df: standardised DataFrame.
        :param <str> cs: individual segment, cross-section or year.
        :param <str> cs_type: segmentation type.
        """

        # Row names of cross-sections or years.
        if cs != "Panel" and cs_type == "cids":
            df_cs = df.loc[cs]
        elif cs != "Panel":
            df_cs = df[df["year"] == float(cs)]
        else:
            df_cs = df

        return df_cs

    @staticmethod
    def apply_slip(
        df: pd.DataFrame,
        slip: int,
        cids: List[str],
        xcats: List[str],
        metrics: List[str],
    ) -> pd.DataFrame:
        """
        Function used to call the apply slip method that is defined in
        management/utils.py

        :param <pd.DataFrame> df: standardised DataFrame.
        :param <int> slip: slip value to apply to df.
        :param <List[str]> cids: list of cids in df to apply slip.
        :param <List[str]> xcats: list of xcats in df to apply slip.
        :param <List[str]> metrics: list of metrics in df to apply slip.
        """
        return apply_slip_util(
            df=df, slip=slip, cids=cids, xcats=xcats, metrics=metrics, raise_error=False
        )

    @staticmethod
    def is_list_of_strings(variable: Any) -> bool:
        """
        Function used to test whether a variable is a list of strings, to avoid the
        compiler saying a string is a list of characters
        :param <Any> variable: variable to be tested.
        :return <bool>: True if variable is a list of strings, False otherwise.
        """
        return isinstance(variable, list) and all(
            isinstance(item, str) for item in variable
        )

    def manipulate_df(
        self,
        xcat: str,
        freq: str,
        agg_sig: str,
        sig: str,
        sst: bool = False,
        df_result: Optional[pd.DataFrame] = None,
    ):
        """
        Used to manipulate the DataFrame to the desired format for the analysis. Firstly
        reduces the dataframe to only include data outside of the blacklist and data that
        is relevant to xcat and sig. Then applies the slip to the dataframe. It then
        converts the dataframe to the desired format for the analysis and checks whether
        any negative signs should be introduced.

        :param <str> xcat: xcat to be analysed.
        :param <str> freq: frequency to be used in analysis.
        :param <str> agg_sig: aggregation method to be used in analysis.
        :param <str> sig: signal to be analysed.
        :param <bool> sst: Boolean that specifies whether this function is to be used for
            a single statistic table.
        :param <Optional[pd.DataFrame]> df_result: DataFrame to be used for single
            statistic table. `None` by default, and when using with `sst` set to `False`.
        """

        cids = None if self.cids is None else self.cids
        dfd = reduce_df(
            self.df,
            xcats=xcat,
            cids=cids,
            start=self.start,
            end=self.end,
            blacklist=self.blacklist,
        )
        metric_cols: List[str] = list(
            set(dfd.columns.tolist()) - set(["real_date", "xcat", "cid"])
        )
        dfd: pd.DataFrame = self.apply_slip(
            df=dfd,
            slip=self.slip,
            cids=cids,
            xcats=xcat,
            metrics=metric_cols,
        )

        if self.cosp and len(self.signals) > 1:
            dfd = self.__communal_sample__(df=dfd, signal=xcat[:-1], ret=xcat[-1])

        self.dfd = dfd

        df = categories_df(
            dfd,
            xcats=xcat,
            cids=cids,
            val="value",
            start=None,
            end=None,
            freq=freq,
            blacklist=None,
            lag=1,
            xcat_aggs=[agg_sig, "sum"],
        )
        self.df = df
        self.cids = list(np.sort(self.df.index.get_level_values(0).unique()))

        if True in self.signs and self.signs[self.sigs.index(sig)]:
            index = self.sigs.index(sig)
            if self.rival_sigs is not None:
                original_name = sig + "/" + agg_sig

                self.df.loc[:, self.signals] *= -1
                s_copy = self.signals.copy()

                self.signals = [s + "_NEG" for s in self.signals]
                sig += "_NEG"
                self.df.rename(columns=dict(zip(s_copy, self.signals)), inplace=True)
                self.new_sig = sig
            else:
                original_name = sig + "/" + agg_sig

                self.df.loc[:, self.sigs[index]] *= -1
                s_copy = self.signals.copy()

                self.signals[self.signals.index(sig)] += "_NEG"
                sig += "_NEG"
                self.df.rename(columns=dict(zip(s_copy, self.signals)), inplace=True)
                self.new_sig = sig

            if sst:
                new_name = sig + "/" + agg_sig
                df_result.rename(index={original_name: new_name}, inplace=True)

        self.new_sig = [sig]

        return df_result

    def __communal_sample__(self, df: pd.DataFrame, signal: str, ret: str):
        """
        On a multi-index DataFrame, where the outer index are the cross-sections and the
        inner index are the timestamps, exclude any row where all signals do not have
        a realised value.

        :param <pd.Dataframe> df: standardized DataFrame with the following necessary
            columns: 'cid', 'xcat', 'real_date' and 'value'.

        NB.:
        Remove the return category from establishing the intersection to preserve the
        maximum amount of signal data available (required because of the applied lag).
        """

        df_w = df.pivot(index=("cid", "real_date"), columns="xcat", values="value")

        storage = []
        for c, cid_df in df_w.groupby(level=0):
            cid_df = cid_df[signal + [ret]]

            final_df = pd.DataFrame(
                data=np.empty(shape=cid_df.shape),
                columns=cid_df.columns,
                index=cid_df.index,
            )
            final_df.loc[:, :] = np.NaN

            # Return category is preserved.
            final_df.loc[:, ret] = cid_df[ret]

            intersection_df = cid_df.loc[:, signal].droplevel(level=0)
            # Intersection exclusively across the signals.
            intersection_df = intersection_df.dropna(how="any")
            if not intersection_df.empty:
                s_date = intersection_df.index[0]
                e_date = intersection_df.index[-1]

                final_df.loc[
                    (c, s_date):(c, e_date), signal
                ] = intersection_df.to_numpy()
                storage.append(final_df)
            else:
                warnings.warn(
                    f"Cross-section {c} has no common sample periods for the signals \
                    {signal} and return {ret}."
                )

        df = pd.concat(storage)
        df = df.stack().reset_index().sort_values(["cid", "xcat", "real_date"])
        df.columns = ["cid", "real_date", "xcat", "value"]

        return df[["cid", "xcat", "real_date", "value"]]

    def __table_stats__(
        self,
        df_segment: pd.DataFrame,
        df_out: pd.DataFrame,
        segment: str,
        signal: str,
        ret: str,
    ):
        """
        Method used to compute the evaluation metrics across segments: cross-section,
        yearly or category level.

        :param <pd.DataFrame> df_segment: segmented DataFrame.
        :param <pd.DataFrame> df_out: metric DataFrame where the index will be all
            segments for the respective segmentation type.
        :param <str> segment: segment which could either be an individual cross-section,
            year or category. Will form the index of the returned DataFrame.
        :param <str> signal: signal category.
        """

        # Account for NaN values between the single respective signal and return. Only
        # applicable for rival signals panel level calculations.

        df_segment = df_segment.loc[:, [ret, signal]].dropna(axis=0, how="any")

        df_sgs = np.sign(df_segment.loc[:, [ret, signal]])
        # Exact zeroes are disqualified for sign analysis only.
        df_sgs = df_sgs[~((df_sgs.iloc[:, 0] == 0) | (df_sgs.iloc[:, 1] == 0))]

        sig_sign = df_sgs[signal]
        ret_sign = df_sgs[ret]

        df_out.loc[segment, "accuracy"] = skm.accuracy_score(sig_sign, ret_sign)
        df_out.loc[segment, "bal_accuracy"] = skm.balanced_accuracy_score(
            sig_sign, ret_sign
        )

        df_out.loc[segment, "pos_sigr"] = np.mean(sig_sign == 1)
        df_out.loc[segment, "pos_retr"] = np.mean(ret_sign == 1)
        df_out.loc[segment, "pos_prec"] = skm.precision_score(
            ret_sign, sig_sign, pos_label=1
        )
        df_out.loc[segment, "neg_prec"] = skm.precision_score(
            ret_sign, sig_sign, pos_label=-1
        )

        ret_vals, sig_vals = df_segment[ret], df_segment[signal]
        df_out.loc[segment, ["kendall", "kendall_pval"]] = stats.kendalltau(
            ret_vals, sig_vals
        )
        corr, corr_pval = stats.pearsonr(ret_vals, sig_vals)
        df_out.loc[segment, ["pearson", "pearson_pval"]] = np.array([corr, corr_pval])

        return df_out

    def __output_table__(self, cs_type: str = "cids", ret=None, sig=None, srt=False):
        """
        Creates a DataFrame with information on the signal-return relation across
        cross-sections or years and, additionally, the panel.

        :param <str> cs_type: the segmentation type.

        """

        if ret is None:
            ret = self.rets if not isinstance(self.rets, list) else self.rets[0]
        if sig is None:
            sig = self.sigs if not isinstance(self.sigs, list) else self.sigs[0]

        # Analysis completed exclusively on the primary signal.
        r = [ret]
        r.append(sig)
        df = self.df[r]

        # Will remove any timestamps where both the signal & return are not realised.
        # Applicable even if communal sampling has been applied given the alignment
        # excludes the return category.
        df = df.dropna(how="any")

        if cs_type == "cids":
            css = set(self.cids)
            unique_cids_df = set(df.index.get_level_values(0).unique())

            if not css.issubset(unique_cids_df):
                warnings.warn(
                    f"Cross-sections {css - unique_cids_df} have no corresponding xcats \
                        in the dataframe."
                )
                css = css.intersection(unique_cids_df)

            css = sorted(list(css))
        else:
            df["year"] = np.array(df.reset_index(level=1)["real_date"].dt.year)
            css = [str(y) for y in list(set(df["year"]))]
            css = sorted(css)

        statms = self.metrics
        if srt:
            css = []
            index = ["Panel"]
        else:
            index = ["Panel", "Mean", "PosRatio"] + css

        df_out = pd.DataFrame(index=index, columns=statms)

        for cs in css + ["Panel"]:
            df_cs = self.__slice_df__(df=df, cs=cs, cs_type=cs_type)
            df_out = self.__table_stats__(
                df_segment=df_cs, df_out=df_out, segment=cs, signal=sig, ret=ret
            )
        if not srt:
            df_out.loc["Mean", :] = df_out.loc[css, :].mean()

            above50s = statms[0:6]
            # Overview of the cross-sectional performance.
            df_out.loc["PosRatio", above50s] = (df_out.loc[css, above50s] > 0.5).mean()

            above0s = statms[6::2]
            pos_corr_coefs = df_out.loc[css, above0s] > 0
            df_out.loc["PosRatio", above0s] = pos_corr_coefs.mean()

            below50s = statms[7::2]
            pvals_bool = df_out.loc[css, below50s] < 0.5
            pos_pvals = np.mean(np.array(pvals_bool) * np.array(pos_corr_coefs), axis=0)
            # Positive correlation with error prob < 50%.
            df_out.loc["PosRatio", below50s] = pos_pvals

        return df_out.astype("float")

    def calculate_single_stat(
        self, stat: str, ret: str = None, sig: str = None, type: str = None
    ):
        """
        Calculates a single statistic for a given signal-return relation.

        :param <str> stat: statistic to be calculated.
        :param <str> ret: return category. Default is the first return category.
        :param <str> sig: signal category. Default is the first signal category.
        :param <str> cstype: type of segment over which bars are drawn. Either
            "panel" (default), "years" or "signals".
        """
        r = [ret]
        r.append(sig)
        df = self.df[r]

        df = df.dropna(how="any")

        if type == "panel":
            css = ["Panel"]
            cs_type = "cids"
        elif type == "mean_cids" or type == "pr_cids":
            css = set(self.cids)
            unique_cids_df = set(df.index.get_level_values(0).unique())
            if not css.issubset(unique_cids_df):
                warnings.warn(
                    f"Cross-sections {css - unique_cids_df} have no corresponding xcats \
                        in the dataframe."
                )
                css = css.intersection(unique_cids_df)
            css = sorted(list(css))
            cs_type = "cids"
        elif type == "mean_years" or type == "pr_years":
            df["year"] = np.array(df.reset_index(level=1)["real_date"].dt.year)
            css = [str(y) for y in list(set(df["year"]))]
            css = sorted(css)
            cs_type = "years"
        else:
            raise ValueError("Invalid segmentation type.")

        list_of_results = []
        for cs in css:
            df_segment = self.__slice_df__(df=df, cs=cs, cs_type=cs_type)
            df_segment = df_segment.loc[:, [ret, sig]].dropna(axis=0, how="any")

            df_sgs = np.sign(df_segment.loc[:, [ret, sig]])
            # Exact zeroes are disqualified for sign analysis only.
            df_sgs = df_sgs[~((df_sgs.iloc[:, 0] == 0) | (df_sgs.iloc[:, 1] == 0))]

            sig_sign = df_sgs[sig]
            ret_sign = df_sgs[ret]
            if stat == "accuracy":
                list_of_results.append(skm.accuracy_score(sig_sign, ret_sign))
            elif stat == "bal_accuracy":
                list_of_results.append(skm.balanced_accuracy_score(sig_sign, ret_sign))
            elif stat == "pos_sigr":
                list_of_results.append(np.mean(sig_sign == 1))
            elif stat == "pos_retr":
                list_of_results.append(np.mean(ret_sign == 1))
            elif stat == "pos_prec":
                list_of_results.append(
                    skm.precision_score(ret_sign, sig_sign, pos_label=1)
                )
            elif stat == "neg_prec":
                list_of_results.append(
                    skm.precision_score(ret_sign, sig_sign, pos_label=-1)
                )
            elif stat == "kendall":
                ret_vals, sig_vals = df_segment[ret], df_segment[sig]
                list_of_results.append(stats.kendalltau(ret_vals, sig_vals)[0])
            elif stat == "kendall_pval":
                ret_vals, sig_vals = df_segment[ret], df_segment[sig]
                list_of_results.append(stats.kendalltau(ret_vals, sig_vals)[1])
            elif stat == "pearson":
                ret_vals, sig_vals = df_segment[ret], df_segment[sig]
                list_of_results.append(stats.pearsonr(ret_vals, sig_vals)[0])
            elif stat == "pearson_pval":
                ret_vals, sig_vals = df_segment[ret], df_segment[sig]
                list_of_results.append(stats.pearsonr(ret_vals, sig_vals)[1])
            else:
                raise ValueError("Invalid statistic.")

        if type == "panel":
            return list_of_results[0]
        elif type == "mean_years" or type == "mean_cids":
            return np.mean(np.array(list_of_results))
        elif type == "pr_years" or type == "pr_cids":
            if stat in self.metrics[0:6]:
                return np.mean(np.array(list_of_results) > 0.5)
            elif stat in self.metrics[6::2]:
                return np.mean(np.array(list_of_results) > 0)
            elif stat in self.metrics[7::2]:
                return np.mean(np.array(list_of_results) < 0.5)

    def summary_table(self):
        """
        Return summary output table of signal-return relations.

        N.B.: The interpretation of the columns is generally as follows:

        accuracy refers to accuracy for binary classification, i.e. positive or negative
            return, and gives the ratio of correct prediction of the sign of returns
            against all predictions. Note that exact zero values for either signal or
            return series will not be considered for accuracy analysis.
        bal_accuracy refers to balanced accuracy. This is the average of the ratios of
            correctly detected positive returns and correctly detected negative returns.
            The denominators here are the total of actual positive and negative returns
            cases. Technically, this is the average of sensitivity and specificity.
        pos_sigr is the ratio of positive signals to all predictions. It indicates the
            long bias of the signal.
        pos_retr is the ratio of positive returns to all observed returns. It indicates
            the positive bias of the returns.
        pos_prec means positive precision, i.e. the ratio of correct positive return
            predictions to all positive predictions. It indicates how well the positive
            predictions of the signal have fared. Generally, good positive precision is
            easy to accomplish if the ratio of positive returns has been high (the signal
            can simply propose a long position for the full duration).
        neg_prec means negative precision, i.e. the ratio of correct negative return
            predictions to all negative predictions. It indicates how well the negative
            predictions of the signal have fared. Generally, good negative precision is
            hard to accomplish if the ratio of positive returns has been high.
        pearson is the Pearson correlation coefficient between signal and subsequent
            return.
        pearson_pval is the probability that the (positive) correlation has been
            accidental, assuming that returns are independently distributed. This
            statistic would be invalid for forward moving averages.
        kendall is the Kendall correlation coefficient between signal and subsequent
            return.
        kendall_pval is the probability that the (positive) correlation has been
            accidental, assuming that returns are independently distributed. This
            statistic would be invalid for forward moving averages.

        The rows have the following meaning:

        Panel refers to the the whole panel of cross-sections and sample period,
            excluding unavailable and blacklisted periods.
        Mean years is the mean of the statistic across all years.
        Mean cids is the mean of the statistic across all cross-sections.
        Positive ratio is the ratio of positive years or cross-sections for which the
            statistic was above its "neutral" level, i.e. above 0.5 for classification
            ratios and positive correlation probabilities, and above 0 for the
            correlation coefficients.
        """

        dfys = self.df_ys.round(decimals=5)
        dfcs = self.df_cs.round(decimals=5)
        dfsum = pd.concat([dfys.iloc[:3,], dfcs.iloc[1:3,]], axis=0)

        dfsum.index = [
            "Panel",
            "Mean years",
            "Positive ratio",
            "Mean cids",
            "Positive ratio",
        ]

        return dfsum

    def revert_negation(self, sig: str):
        if sig[-4:] == "_NEG":
            sig = sig[:-4]
        return sig

    def single_relation_table(
        self,
        ret: str = None,
        xcat: str = None,
        freq: str = None,
        agg_sigs: str = None,
    ):
        """
        Computes all the statistics for one specific signal-return relation:

        :param <str> ret: single target return category. Default is first in target
            return list of the class.
        :param <str> xcat: single signal category to be considered. Default is first in
            feature category list of the class.
        :param <str> freq: letter denoting single frequency at which the series will
            be sampled. This must be one of the frequencies selected for the class.
            If not specified uses the freq stored in the class.
        :param <str> agg_sigs: aggregation method applied to the signal values in
            down-sampling.
        """
        self.df = self.original_df
        if ret is None:
            ret = self.rets if not isinstance(self.rets, list) else self.rets[0]
        if freq is None:
            freq = self.freqs if not isinstance(self.freqs, list) else self.freqs[0]
        if agg_sigs is None:
            agg_sigs = (
                self.agg_sigs
                if not isinstance(self.agg_sigs, list)
                else self.agg_sigs[0]
            )
        if xcat is None:
            sig = (
                self.revert_negation(self.sigs)
                if not isinstance(self.sigs, list)
                else self.revert_negation(self.sigs[0])
            )
            if isinstance(self.sigs, list):
                self.sigs[0] = self.revert_negation(self.sigs[0])
            else:
                self.sigs = self.revert_negation(self.sigs)
            xcat = [sig, ret]
        elif isinstance(xcat, list):
            sig = xcat[0]
        elif not isinstance(xcat, str):
            raise TypeError("xcat must be a string")
        else:
            sig = xcat
            xcat = [sig, ret]

        if not isinstance(ret, str):
            raise TypeError("ret must be a string")
        if not isinstance(freq, str):
            raise TypeError("freq must be a string")
        if not isinstance(agg_sigs, str):
            raise TypeError("agg_sigs must be a string")

        self.signals = [sig]

        self.manipulate_df(xcat=xcat, freq=freq, agg_sig=agg_sigs, sig=sig)

        df_result = self.__output_table__(
            cs_type="cids", ret=ret, sig=self.new_sig[0], srt=True
        ).round(decimals=5)

        self.df = self.original_df
        sig_string = sig + "_NEG" if self.signs[self.sigs.index(sig)] else sig
        index = f"{freq}: {sig_string}/{agg_sigs} => {ret}"

        df_result.rename(index={"Panel": index}, inplace=True)

        return df_result

    def multiple_relations_table(
        self,
        rets: Union[str, List[str]] = None,
        xcats: Union[str, List[str]] = None,
        freqs: Union[str, List[str]] = None,
        agg_sigs: Union[str, List[str]] = None,
    ):
        """
        Calculates all the statistics for each return and signal category specified with
        each frequency and aggregation method, note that if none are defined it does this
        for all categories, frequencies and aggregation methods that were stored in the
        class.

        :param <str, List[str]> rets: target return category
        :param <str, List[str]> xcats: signal categories to be considered
        :param <str, List[str]> freqs: letters denoting frequency at which the series
            are to be sampled.
            This must be one of 'D', 'W', 'M', 'Q', 'A'. If not specified uses the freq
            stored in the class.
        :param <str, List[str]> agg_sigs: aggregation methods applied to the signal
            values in down-sampling.
        """
        if rets is None:
            rets = self.rets
        if freqs is None:
            freqs = self.freqs
        if agg_sigs is None:
            agg_sigs = self.agg_sigs
        if not isinstance(agg_sigs, list):
            agg_sigs = [agg_sigs]
        if xcats is None:
            xcats = self.xcats
        if not isinstance(xcats, list):
            xcats = [xcats]
        if not isinstance(rets, list):
            rets = [rets]
        if not isinstance(freqs, list):
            freqs = [freqs]

        for rets_elem in rets:
            if not rets_elem in self.xcats:
                raise ValueError(f"{rets_elem} is not a valid return category")

        for xcats_elem in xcats:
            if not xcats_elem in self.xcats:
                raise ValueError(f"{xcats_elem} is not a valid signal category")

        for freqs_elem in freqs:
            if not freqs_elem in self.freqs:
                raise ValueError(f"{freqs_elem} is not a valid frequency")

        for agg_sigs_elem in agg_sigs:
            if not agg_sigs_elem in self.agg_sigs:
                raise ValueError(f"{agg_sigs_elem} is not a valid aggregation method")

        self.sigs = [self.revert_negation(sig) for sig in self.sigs]

        xcats = [x for x in xcats if x in self.sigs]

        index = [
            (
                f"{freq}: "
                f"{xcat + '_NEG' if self.signs[self.sigs.index(xcat)] else xcat}"
                f"/{agg_sig} => {ret}"
            )
            for freq in freqs
            for agg_sig in agg_sigs
            for ret in rets
            for xcat in xcats
        ]

        df_result = pd.concat(
            [
                self.single_relation_table(
                    ret=ret, xcat=[xcat, ret], freq=freq, agg_sigs=agg_sig
                )
                for freq in freqs
                for agg_sig in agg_sigs
                for ret in rets
                for xcat in xcats
            ],
            axis=0,
        )
        df_result.index = index

        return df_result

    def single_statistic_table(
        self,
        stat: str,
        type: str = "panel",
        rows: List[str] = ["xcat", "agg_sigs"],
        columns: List[str] = ["ret", "freq"],
        show_heatmap: bool = False,
        title: Optional[str] = None,
        title_fontsize: int = 16,
        row_names: Optional[List[str]] = None,
        column_names: Optional[List[str]] = None,
        min_color: Optional[float] = None,
        max_color: Optional[float] = None,
        figsize: Tuple[float] = (14, 8),
        annotate: bool = True,
        round: int = 5,
    ):
        """
        Creates a table which shows the specified statistic for each row and
        column specified as arguments:

        :param stat: type of statistic to be displayed (this can be any of
            the column names of summary_table).
        :param type: type of the statistic displayed. This can be based on
            the overall panel ("panel", default), an
            average of annual panels (mean_years), an average of cross-sectional
            relations ("mean_cids"), the positive ratio across years("pr_years"),
            positive ratio across sections ("pr_cids").
        :param <List[str]> rows: row indices, which can be return categories,
            feature categories, frequencies and/or aggregations. The choice is
            made through a list of one or more of "xcat", "ret", "freq" and
            "agg_sigs". The default is ["xcat", "agg_sigs"] resulting in index
            strings (<agg_signs>) or if only one aggregation is available.
        :param <List[str]> columns: column indices, which can be return
            categories, feature categories, frequencies and/or aggregations. The
            choice is made through a list of one or more of "xcat", "ret", "freq"
            and "agg_sigs". The default is ["ret", "freq] resulting in index
            strings () or if only one frequency is available.
        :param <bool> show_heatmap: if True, the table is visualized as a
            heatmap. Default is False.
        :param <str> title: plot title; if none given default title is shown.
        :param <int> title_fontsize: font size of title. Default is 16.
        :param <List[str]> row_names: specifies the labels of rows in the heatmap.
            If None, the indices of the generated DataFrame are used.
        :param <List[str]> column_names: specifies the labels of columns in the
            heatmap. If None, the columns of the generated DataFrame are used.
        :param <float> min_color: minimum value of the color scale. Default
            is None, in which case the minimum value of the table is used.
        :param <float> max_color: maximum value of the color scale. Default
            is None, in which case the maximum value of the table is used.
        :param <Tuple[float]> figsize: Tuple (w, h) of width and height of graph.
        :param <bool> annotate: if True, the values are annotated in the heatmap.
        :param <int> round: number of decimals to round the values to on the
            heatmap's annotations.
        """
        self.df = self.original_df
        stat_values = [
            "accuracy",
            "bal_accuracy",
            "pos_sigr",
            "pos_retr",
            "pos_prec",
            "neg_prec",
            "kendall",
            "kendall_pval",
            "pearson",
            "pearson_pval",
        ]

        if not stat in stat_values:
            raise ValueError(f"Stat must be one of {stat_values}")

        if not isinstance(rows, list):
            raise TypeError("Rows must be a list")
        if not isinstance(columns, list):
            raise TypeError("Columns must be a list")

        if not "agg_sigs" in rows and not "agg_sigs" in columns:
            agg_sigs = ["last"]
        if not "freqs" in rows and not "freqs" in columns:
            freqs = ["Q"]

        if isinstance(self.freqs, list):
            freqs = self.freqs
        else:
            freqs = [self.freqs]
        if self.is_list_of_strings(self.agg_sigs):
            agg_sigs = self.agg_sigs
        else:
            agg_sigs = [self.agg_sigs]

        type_values = ["panel", "mean_years", "mean_cids", "pr_years", "pr_cids"]
        rows_values = ["xcat", "ret", "freq", "agg_sigs"]

        if not type in type_values:
            raise ValueError(f"Type must be one of {type_values}")

        if not all([x in rows_values for x in rows]):
            raise ValueError(f"Rows must only contain {rows_values}")

        if not all([x in rows_values for x in columns]):
            raise ValueError(f"Columns must only contain {rows_values}")

        rets = self.rets if isinstance(self.rets, list) else [self.rets]
        self.sigs = [self.revert_negation(sig) for sig in self.sigs]
        sigs = self.sigs if isinstance(self.sigs, list) else [self.sigs]

        sigs_neg = []
        for sig in sigs:
            if self.signs[self.sigs.index(sig)]:
                sigs_neg.append(sig + "_NEG")
            else:
                sigs_neg.append(sig)

        rows_dict = {"xcat": sigs_neg, "ret": rets, "freq": freqs, "agg_sigs": agg_sigs}

        df_row_names, df_column_names = self.set_df_labels(rows_dict, rows, columns)

        df_result = pd.DataFrame(
            columns=df_column_names, index=df_row_names, dtype=np.float64
        )

        loop_tuples: List[Tuple[str, str, str, str]] = [
            (ret, sig, freq, agg_sig)
            for ret in rets
            for sig in sigs
            for freq in freqs
            for agg_sig in agg_sigs
        ]

        for ret, sig, freq, agg_sig in loop_tuples:
            sig_original = sig
            if self.signs[self.sigs.index(sig)]:
                sig += "_NEG"
            hash = f"{ret}/{sig}/{freq}/{agg_sig}"
            sig = sig_original

            # Prepare xcat and manipulate DataFrame
            xcat = [sig, ret]
            self.signals = [sig]
            self.manipulate_df(
                xcat=xcat,
                freq=freq,
                agg_sig=agg_sig,
                sig=sig,
                sst=True,
                df_result=df_result,
            )

            row = self.get_rowcol(hash, rows)
            column = self.get_rowcol(hash, columns)
            df_result[column][row] = self.calculate_single_stat(
                stat, ret, self.new_sig[0], type
            )

            # Reset self.df and sig to original values
            self.df = self.original_df
            sig = sig_original

        if show_heatmap:
            if not title:
                title = f"{stat}"

            if min_color is None:
                min_color = df_result.values.min()
            if max_color is None:
                max_color = df_result.values.max()

            msv.view_table(
                df_result,
                title=title,
                title_fontsize=title_fontsize,
                min_color=min_color,
                max_color=max_color,
                figsize=figsize,
                fmt=f".{round}f",
                annot=annotate,
                xticklabels=column_names,
                yticklabels=row_names,
            )

        return df_result

    def set_df_labels(self, rows_dict: Dict, rows: List[str], columns: List[str]):
        """
        Creates two lists of strings that will be used as the row and column labels for
        the resulting dataframe.

        :param <dict> rows_dict: dictionary containing the each value for each of the
            xcat, ret, freq and agg_sigs categories.
        :param <List[str]> rows: list of strings specifying which of the categories are
            included in the rows of the dataframe.
        :param <List[str]> columns: list of strings specifying which of the categories
            are included in the columns of the dataframe.
        """
        if len(rows) == 2:
            rows_names = [
                a + "/" + b for a in rows_dict[rows[0]] for b in rows_dict[rows[1]]
            ]
            columns_names = [
                a + "/" + b
                for a in rows_dict[columns[0]]
                for b in rows_dict[columns[1]]
            ]
        elif len(rows) == 1:
            rows_names = rows_dict[rows[0]]
            columns_names = [
                a + "/" + b + "/" + c
                for a in rows_dict[columns[0]]
                for b in rows_dict[columns[1]]
                for c in rows_dict[columns[2]]
            ]
        elif len(columns) == 1:
            rows_names = [
                a + "/" + b + "/" + c
                for a in rows_dict[rows[0]]
                for b in rows_dict[rows[1]]
                for c in rows_dict[rows[2]]
            ]
            columns_names = rows_dict[columns[0]]

        return rows_names, columns_names

    def get_rowcol(self, hash: str, rowcols: List[str]):
        """
        Calculates which row/column the hash belongs to.

        :param <str> hash: hash of the statistic.
        :param <List[str]> rowcols: list of strings specifying which of the categories
        are in the rows/columns of the dataframe.
        """
        result = ""
        idx: List[str] = ["ret", "xcat", "freq", "agg_sigs"]
        assert all([x in idx for x in rowcols]), "rowcols must be a subset of idx"

        for rowcol in rowcols:
            result += hash.split("/")[idx.index(rowcol)] + "/"

        return result[:-1]


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "NZD"]
    xcats = ["XR", "XRH", "CRY", "GROWTH", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0, 1]
    df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
    df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
    df_cids.loc["NZD"] = ["2007-01-01", "2020-09-30", 0.0, 2]

    df_xcats = pd.DataFrame(
        index=xcats,
        columns=["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"],
    )
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["XRH"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 0, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 0, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 0, 2, 0.8, 0.5]

    black = {"AUD": ["2006-01-01", "2015-12-31"], "GBP": ["2012-01-01", "2100-01-01"]}

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Additional signals.
    srn = SignalReturnRelations(
        dfd,
        rets="XR",
        sigs="CRY",
        rival_sigs=None,
        sig_neg=True,
        cosp=True,
        freqs="M",
        start="2002-01-01",
    )

    dfsum = srn.summary_table()
    print(dfsum)

    r_sigs = ["INFL", "GROWTH"]
    srn = SignalReturnRelations(
        dfd,
        rets="XR",
        sigs="CRY",
        rival_sigs=r_sigs,
        sig_neg=True,
        cosp=True,
        freqs="M",
        start="2002-01-01",
    )
    dfsum = srn.summary_table()
    print(dfsum)

    df_sigs = srn.signals_table(sigs=["CRY_NEG", "INFL_NEG"])
    df_sigs_all = srn.signals_table()
    print(df_sigs)
    print(df_sigs_all)

    srn.accuracy_bars(
        type="signals",
        title="Accuracy measure between target return, XR,"
        " and the respective signals, ['CRY', 'INFL'"
        ", 'GROWTH'].",
    )

    sr = SignalReturnRelations(
        dfd,
        rets="XR",
        sigs="CRY",
        freqs="M",
        start="2002-01-01",
        agg_sigs="last",
    )

    srt = sr.single_relation_table()
    mrt = sr.multiple_relations_table()
    sst = sr.single_statistic_table(stat="accuracy", type="mean_years")

    print(srt)
    print(mrt)
    print(sst)

    # Basic Signal Returns showing for multiple input values

    sr = SignalReturnRelations(
        dfd,
        rets=["XR", "XRH"],
        sigs=["CRY", "INFL", "GROWTH"],
        sig_neg=[True, True],
        cosp=True,
        freqs=["M", "Q"],
        agg_sigs=["last", "mean"],
        blacklist=black,
    )

    sr.correlation_bars(ret="XRH", sig="INFL", freq="Q")

    srt = sr.single_relation_table()
    mrt = sr.multiple_relations_table()
    sst = sr.single_statistic_table(stat="accuracy", show_heatmap=True)

    print(srt)
    print(mrt)
    print(sst)

    # Specifying specific arguments for each of the Signal Return Functions

    srt = sr.single_relation_table(ret="XR", xcat="CRY", freq="Q", agg_sigs="last")
    print(srt)

    mrt = sr.multiple_relations_table(
        rets=["XR", "GROWTH"], xcats="INFL", freqs=["M", "Q"], agg_sigs=["last", "mean"]
    )
    print(mrt)

    sst = sr.single_statistic_table(
        stat="accuracy",
        rows=["ret", "xcat", "freq"],
        columns=["agg_sigs"],
        type="mean_cids",
    )
    print(sst)
