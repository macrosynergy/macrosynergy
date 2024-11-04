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
from typing import List, Union, Tuple, Dict, Any, Optional, Callable
import statsmodels.api as sm

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import (
    apply_slip as apply_slip_util,
    reduce_df,
    categories_df,
)
import macrosynergy.visuals as msv

# Ensure warnings are printed
warnings.simplefilter("always")


class SignalReturnRelations:
    """
    Class for analysing and visualizing signal and a return series.

    :param <pd.Dataframe> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value.
    :param <str, List[str]> rets: one or several target return categories.
    :param <str, List[str]> sigs: list of signal categories to be considered for which
        detailed relational statistics can be calculated.
    :param <bool, List[bool]> sig_neg: if set to True puts the signal in negative terms
        for all analysis. If more than one signal is tested, `sig_neg` must be a
        same-order list of the same length as the signals, containing a True for each
        signal that needs to be put in negative terms. Default is False.
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
        returns, which is often characterized by a delay due to the setup of positions.
        Technically, this is a negative lag (early arrival) of the target category
        in working days prior to any frequency conversion. Default is 0.
    :param <bool> ms_panel_test: if True the Macrosynergy Panel test is calculated. Please
        note that this is a very time-consuming operation and should be used only if you
        require the result.
    :param <List[Callable]> additional_metrics: list of additional metrics to be
        calculated and added to the output table.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rets: Union[str, List[str]] = None,
        sigs: Union[str, List[str]] = None,
        cids: Union[str, List[str]] = None,
        sig_neg: Union[bool, List[bool]] = None,
        cosp: bool = False,
        start: str = None,
        end: str = None,
        blacklist: dict = None,
        freqs: Union[str, List[str]] = "M",
        agg_sigs: Union[str, List[str]] = "last",
        fwin: int = 1,
        slip: int = 0,
        ms_panel_test: bool = False,
        additional_metrics: List[Callable] = None,
    ):
        if rets is None:
            raise ValueError("Target return must be defined.")
        if sigs is None:
            raise ValueError("Signal must be defined.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"DataFrame expected and not {type(df)}.")
        if not isinstance(cids, str) and cids is not None:
            if not isinstance(cids, list):
                raise TypeError(f"List or string expected and not {type(cids)}.")
            else:
                if not all(isinstance(cid, str) for cid in cids):
                    raise TypeError(f"List of strings expected for cids.")

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
                        warnings.warn(
                            f"Frequency {f} is repeated, dropping repeated frequency."
                        )
        else:
            if not freqs in self.dic_freq.keys():
                raise ValueError(freq_error)
            else:
                self.freqs = [freqs]

        if not isinstance(ms_panel_test, bool):
            raise TypeError(
                f"<bool> object expected for ms_panel_test and not {type(ms_panel_test)}."
            )

        self.ms_panel_test = ms_panel_test

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
            "auc",
        ]

        if self.ms_panel_test:
            self.metrics.append("map_pval")

        if additional_metrics:
            self.metrics.extend(
                metric.__name__
                for metric in additional_metrics
                if hasattr(metric, "__name__")
            )
        else:
            additional_metrics = []

        self.additional_metrics = additional_metrics

        if not isinstance(cosp, bool):
            raise TypeError(f"<bool> object expected and not {type(cosp)}.")

        if isinstance(cids, str):
            self.cids = [cids]
        else:
            self.cids = cids

        self.rets = rets
        self.slip = slip
        self.agg_sigs = agg_sigs
        self.xcats = list(df["xcat"].unique())
        self.df = df
        self.cosp = cosp
        self.start = start
        self.end = end
        self.blacklist = blacklist
        self.fwin = fwin

        if not self.is_list_of_strings(rets):
            self.rets = [rets]

        if not self.is_list_of_strings(sigs):
            self.sigs = [sigs]
        else:
            self.sigs = sigs.copy()

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

        if sig_neg is None:
            self.signs = [False for _ in self.sigs]
        else:
            self.signs = sig_neg if isinstance(sig_neg, list) else [sig_neg]

        for sign in self.signs:
            if not sign in [False, True]:
                raise TypeError("Sign must be either False or True.")

        if len(self.signs) != len(self.sigs):
            raise ValueError("Signs must have a length equal to signals")

        self.xcats = self.rets + self.sigs
        self.df = reduce_df(
            df,
            xcats=self.xcats,
            cids=self.cids,
            start=self.start,
            end=self.end,
            blacklist=self.blacklist,
        )

        for sig in self.sigs:
            if self.signs[self.sigs.index(sig)]:
                self.df.loc[self.df["xcat"] == sig, "value"] *= -1
                self.df.loc[self.df["xcat"] == sig, "xcat"] = (
                    self.df.loc[self.df["xcat"] == sig, "xcat"] + "_NEG"
                )
                self.sigs[self.sigs.index(sig)] = f"{sig}_NEG"

        self.original_df = self.df.copy()

    def __rival_sigs__(self, ret, sigs=None):
        """
        Produces the panel-level table for the additional signals.
        """

        if sigs is None:
            sigs = self.sigs

        df_out = pd.DataFrame(index=sigs, columns=self.metrics)
        df = self.df

        for s in sigs:
            # Entire panel will be passed in.
            df_out = self.__table_stats__(
                df_segment=df, df_out=df_out, segment=s, signal=s, ret=ret
            )

        return df_out

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
        sigs: Union[str, List[str]] = None,
        freq: str = None,
        agg_sig: str = None,
        type: str = "cross_section",
        title: str = None,
        title_fontsize: int = 16,
        size: Tuple[float] = None,
        legend_pos: str = "best",
    ):
        """
        Plot bar chart for the overall and balanced accuracy metrics. For types:
        cross_section and years. If sigs is not specified, then the first signal in the
        list of signals will be used.

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

        if sigs is None:
            sigs = self.sigs
        elif isinstance(sigs, str):
            if sigs not in self.sigs and sigs + "_NEG" in self.sigs:
                sigs = sigs + "_NEG"
        if isinstance(sigs, list):
            for sig in sigs:
                if sig not in self.sigs and sig + "_NEG" in self.sigs:
                    sigs[sigs.index(sig)] = sig + "_NEG"

        if isinstance(sigs, str):
            sigs = [sigs]

        for sig in sigs:
            if sig not in self.sigs:
                raise ValueError(
                    f"Signal {sig} is not defined in Signal Return Relations."
                )

        if freq is None:
            freq = self.freqs[0]

        if agg_sig is None:
            agg_sig = self.agg_sigs[0]

        if ret is None:
            ret = self.rets[0]

        self.df = self.original_df.copy()
        self.manipulate_df(xcat=sigs + [ret], freq=freq, agg_sig=agg_sig)

        for i in range(len(sigs)):
            if not sigs[i] in self.sigs:
                sigs[i] = sigs[i] + "_NEG"

        if type == "cross_section":
            df_xs = self.__output_table__(cs_type="cids", ret=ret, sig=sigs[0])
        elif type == "years":
            df_xs = self.__output_table__(cs_type="years", ret=ret, sig=sigs[0])
        else:
            df_xs = self.__rival_sigs__(ret, sigs)

        dfx = df_xs[~df_xs.index.isin(["PosRatio"])]

        if title is None:
            refsig = "various signals" if type == "signals" else sigs[0]
            title = (
                f"Accuracy for sign prediction of {ret} based on {refsig} "
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
        sigs: Union[str, List[str]] = None,
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

        if ret is None and sigs is None:
            ret = self.rets[0]
            sigs = self.sigs
        else:
            if ret is None:
                ret = self.rets[0]
            if sigs is None:
                sigs = self.sigs
            elif isinstance(sigs, str):
                if sigs not in self.sigs and sigs + "_NEG" in self.sigs:
                    sigs = sigs + "_NEG"
            if isinstance(sigs, list):
                for sig in sigs:
                    if sig not in self.sigs and sig + "_NEG" in self.sigs:
                        sigs[sigs.index(sig)] = sig + "_NEG"
            self.df = self.original_df.copy()

        if isinstance(sigs, str):
            sigs = [sigs]

        self.manipulate_df(
            xcat=sigs + [ret],
            freq=freq,
            agg_sig=self.agg_sigs[0],
        )
        for i in range(len(sigs)):
            if not sigs[i] in self.sigs:
                sigs[i] = sigs[i] + "_NEG"
        if type == "cross_section":
            df_xs = self.__output_table__(cs_type="cids", ret=ret, sig=sigs[0])
        elif type == "years":
            df_xs = self.__output_table__(cs_type="years", ret=ret, sig=sigs[0])
        else:
            df_xs = self.__rival_sigs__(ret, sigs)

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
            refsig = "various signals" if type == "signals" else sigs[0]
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

    def manipulate_df(self, xcat: str, freq: str, agg_sig: str):
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
        self.df = self.original_df.copy()

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
            set(dfd.columns.tolist()) - set(["real_date", "xcat", "cid", "ticker"])
        )
        dfd: pd.DataFrame = self.apply_slip(
            df=dfd,
            slip=self.slip,
            cids=cids,
            xcats=[xcat[-1]],
            metrics=metric_cols,
        )

        if self.cosp and len(self.sigs) > 1:
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
            fwin=self.fwin,
            xcat_aggs=[agg_sig, "sum"],
        )
        self.df = df
        self.cids = list(np.sort(self.df.index.get_level_values(0).unique()))

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

                final_df.loc[(c, s_date):(c, e_date), signal] = (
                    intersection_df.to_numpy()
                )
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
        if len(ret_sign) <= 1:
            corr, corr_pval = np.NaN, np.NaN
        else:
            corr, corr_pval = stats.pearsonr(ret_vals, sig_vals)
        df_out.loc[segment, ["pearson", "pearson_pval"]] = np.array([corr, corr_pval])

        if (ret_sign == -1.0).all() or (ret_sign == 1.0).all():
            df_out.loc[segment, "auc"] = np.NaN
            warnings.warn(
                "AUC could not be calculated, since the return category has a lack of "
                "class diversity."
            )
        else:
            df_out.loc[segment, "auc"] = skm.roc_auc_score(ret_sign, sig_sign)

        if self.ms_panel_test:
            df_out.loc[segment, "map_pval"] = self.map_pval(ret_vals, sig_vals)

        for metric in self.additional_metrics:
            df_out.loc[segment, metric.__name__] = metric(ret_vals, sig_vals)

        return df_out

    def map_pval(self, ret_vals, sig_vals):
        if (
            not "cid" in ret_vals.index.names
            or ret_vals.index.get_level_values("cid").nunique() <= 1
        ):
            warnings.warn(
                "P-value could not be calculated, since there wasn't enough datapoints."
            )
            return np.NaN
        X = sm.add_constant(ret_vals)
        y = sig_vals.copy()
        groups = ret_vals.index.get_level_values("real_date")
        mlm = sm.MixedLM(y, X, groups=groups)
        try:
            re = mlm.fit(reml=False)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Singular matrix encountered, so p-value could not be calculated."
            )
            return np.NaN
        if re.summary().tables[1].iloc[1, 3] == "":
            warnings.warn(
                "P-value could not be calculated, since there wasn't enough datapoints."
            )
            return np.NaN
        pval_string = re.summary().tables[1].iloc[1, 3]
        return float(pval_string)

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

            above50s = statms[0:6] + [statms[statms.index("auc")]]
            # Overview of the cross-sectional performance.
            df_out.loc["PosRatio", above50s] = (df_out.loc[css, above50s] > 0.5).mean()

            above0s = statms[6:9:2]
            pos_corr_coefs = df_out.loc[css, above0s] > 0
            df_out.loc["PosRatio", above0s] = pos_corr_coefs.mean()

            below50s = statms[7:10:2]
            pvals_bool = df_out.loc[css, below50s] < 0.5
            pos_pvals = np.mean(np.array(pvals_bool) * np.array(pos_corr_coefs), axis=0)
            # Positive correlation with error prob < 50%.
            df_out.loc["PosRatio", below50s] = pos_pvals
            if self.ms_panel_test:
                map_pval_bool = df_out.loc[css, "map_pval"] < 0.5
                pos_map_pval = np.mean(np.array(map_pval_bool) * np.nan)
                df_out.loc["PosRatio", "map_pval"] = pos_map_pval

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
            ret_vals, sig_vals = df_segment[ret], df_segment[sig]
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
                list_of_results.append(stats.kendalltau(ret_vals, sig_vals)[0])
            elif stat == "kendall_pval":
                list_of_results.append(stats.kendalltau(ret_vals, sig_vals)[1])
            elif stat == "pearson":
                list_of_results.append(stats.pearsonr(ret_vals, sig_vals)[0])
            elif stat == "pearson_pval":
                list_of_results.append(stats.pearsonr(ret_vals, sig_vals)[1])
            elif stat == "auc":
                if (ret_sign == -1.0).all() or (ret_sign == 1.0).all():
                    list_of_results.append(np.NaN)
                    warnings.warn(
                        "AUC could not be calculated, since the return category has a "
                        "lack of class diversity."
                    )
                else:
                    list_of_results.append(skm.roc_auc_score(ret_sign, sig_sign))
            elif stat == "map_pval" and self.ms_panel_test:
                list_of_results.append(self.map_pval(ret_vals, sig_vals))
            elif True in [
                stat == metric.__name__ for metric in self.additional_metrics
            ]:
                idx = [
                    stat == metric.__name__ for metric in self.additional_metrics
                ].index(True)
                list_of_results.append(self.additional_metrics[idx](ret_vals, sig_vals))
            else:
                raise ValueError("Invalid statistic.")

        if type == "panel":
            return list_of_results[0]
        elif type == "mean_years" or type == "mean_cids":
            return np.mean(np.array(list_of_results))
        elif type == "pr_years" or type == "pr_cids":
            if stat in self.metrics[0:6] + ["auc"]:
                return np.mean(np.array(list_of_results) > 0.5)
            elif stat in self.metrics[6:9:2]:
                return np.mean(np.array(list_of_results) > 0)
            elif stat in self.metrics[7:10:2]:
                return np.mean(np.array(list_of_results) < 0.5)

    def summary_table(self, cross_section: bool = False, years: bool = False):
        warnings.warn(
            "summary_table() has been deprecated will be removed in a subsequent "
            "version, please now use single_relation_table(table_type='summary').",
            FutureWarning,
        )
        if cross_section and years:
            raise ValueError("Both cross_section and years cannot be True")
        if not (cross_section and years):
            return self.single_relation_table(table_type="summary")
        else:
            return self.single_relation_table(
                table_type="years" if years else "cross_section"
            )

    def signals_table(self, sigs: List[str] = None):
        warnings.warn(
            "signals_table() has been deprecated will be removed in a subsequent "
            "version, please now use multiple_relations_table()",
            FutureWarning,
        )
        if sigs is None:
            sigs = self.sigs
        return self.multiple_relations_table(
            rets=self.rets[0],
            xcats=sigs,
            freqs=self.freqs[0],
            agg_sigs=self.agg_sigs[0],
        )

    def cross_section_table(self):
        warnings.warn(
            "cross_section_table() has been deprecated will be removed in a subsequent "
            "version, please now use "
            " single_relation_table(table_type='cross_section_table')",
            FutureWarning,
        )
        return self.single_relation_table(table_type="cross_section")

    def yearly_table(self):
        warnings.warn(
            "yearly_table() has been deprecated will be removed in a subsequent "
            "version, please now use single_relation_table(table_type='years')",
            FutureWarning,
        )
        return self.single_relation_table(table_type="years")

    def single_relation_table(
        self,
        ret: str = None,
        xcat: str = None,
        freq: str = None,
        agg_sigs: str = None,
        table_type: str = None,
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
        :param <str> table_type: type of table to be returned. Either "summary", "years",
            "cross_section".
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
            sig = self.sigs if not isinstance(self.sigs, list) else self.sigs[0]
            xcat = [sig, ret]
        elif not isinstance(xcat, str):
            raise TypeError("xcat must be a string")
        else:  # If xcat is a string
            if xcat not in self.sigs and xcat + "_NEG" in self.sigs:
                xcat = xcat + "_NEG"
            sig = xcat
            xcat = [sig, ret]

        if not isinstance(ret, str):
            raise TypeError("ret must be a string")
        if not isinstance(freq, str):
            raise TypeError("freq must be a string")
        if not isinstance(agg_sigs, str):
            raise TypeError("agg_sigs must be a string")

        self.manipulate_df(xcat=xcat, freq=freq, agg_sig=agg_sigs)

        if not sig in self.sigs:
            sig = sig + "_NEG"

        if table_type is not None:
            if not table_type in ["summary", "years", "cross_section"]:
                raise ValueError("Invalid table type")

        if table_type == "years":
            cs_type = "years"
        else:
            cs_type = "cids"

        if table_type == "summary":
            df_result = pd.concat(
                [
                    self.__output_table__(
                        cs_type="years", ret=ret, sig=sig, srt=False
                    ).iloc[
                        :3,
                    ],
                    self.__output_table__(
                        cs_type="cids", ret=ret, sig=sig, srt=False
                    ).iloc[
                        1:3,
                    ],
                ],
                axis=0,
            )
            df_result.index = [
                df_result.index[0],
                "Mean years",
                "Positive ratio",
                "Mean cids",
                "Positive ratio",
            ]
        else:
            df_result = self.__output_table__(
                cs_type=cs_type, ret=ret, sig=sig, srt=table_type is None
            )

        self.df = self.original_df
        index = f"{freq}: {sig}/{agg_sigs} => {ret}"

        df_result.rename(index={"Panel": index}, inplace=True)

        return df_result.round(5)

    def reindex_multindex_df(
        self, df: pd.DataFrame, desired_order: List[str], var_type: str
    ):
        df["Signal_Order"] = pd.Categorical(
            df.index.get_level_values(var_type), categories=desired_order, ordered=True
        )
        df_sorted = df.sort_values("Signal_Order")
        df_sorted.drop("Signal_Order", axis=1, inplace=True)
        return df_sorted

    def multiple_relations_table(
        self,
        rets: Union[str, List[str]] = None,
        xcats: Union[str, List[str]] = None,
        freqs: Union[str, List[str]] = None,
        agg_sigs: Union[str, List[str]] = None,
        signal_name_dict: Optional[Dict[str, str]] = None,
        return_name_dict: Optional[Dict[str, str]] = None,
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
        self.df = self.original_df
        self.xcats = list(self.df["xcat"].unique())
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
        else:
            if isinstance(xcats, str):
                if xcats not in self.sigs and xcats + "_NEG" in self.sigs:
                    xcats = xcats + "_NEG"
            if isinstance(xcats, list):
                for xcat in xcats:
                    if xcat not in self.sigs and xcat + "_NEG" in self.sigs:
                        xcats[xcats.index(xcat)] = xcat + "_NEG"
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

        xcats = [x for x in xcats if x in self.sigs]

        multiindex = pd.MultiIndex.from_tuples(
            [
                (ret, xcat, freq, agg_sig)
                for freq in freqs
                for agg_sig in agg_sigs
                for ret in rets
                for xcat in xcats
            ],
            names=["Return", "Signal", "Frequency", "Aggregation"],
        )

        df_rows = []
        for freq in freqs:
            for agg_sig in agg_sigs:
                for ret in rets:
                    self.manipulate_df(xcat=xcats + [ret], freq=freq, agg_sig=agg_sig)
                    for xcat in xcats:
                        df_rows.append(
                            self.__output_table__(
                                cs_type="cids", ret=ret, sig=xcat, srt=True
                            )
                        )

        df_result = pd.concat(df_rows, axis=0)

        df_result.index = multiindex

        if signal_name_dict is not None:
            df_result.rename(index=signal_name_dict, inplace=True)
            df_result = self.reindex_multindex_df(
                df_result, signal_name_dict.values(), "Signal"
            )

        if return_name_dict is not None:
            df_result.rename(index=return_name_dict, inplace=True)
            df_result = self.reindex_multindex_df(
                df_result, return_name_dict.values(), "Return"
            )

        self.df = self.original_df

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
        signal_name_dict: Optional[Dict[str, str]] = None,
        return_name_dict: Optional[Dict[str, str]] = None,
        min_color: Optional[float] = None,
        max_color: Optional[float] = None,
        figsize: Tuple[float] = (14, 8),
        annotate: bool = True,
        round: int = 5,
    ):
        """
        Creates a table which shows the specified statistic for each row and
        column specified as arguments:

        :param <str> stat: type of statistic to be displayed (this can be any of
            the column names of summary_table).
        :param <str> type: type of the statistic displayed. This can be based on
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

        :return <pd.DataFrame>: DataFrame with the specified statistic for each row and column
        """
        self.df = self.original_df.copy()

        if not stat in self.metrics:
            raise ValueError(f"Stat must be one of {self.metrics}")

        if not isinstance(rows, list):
            raise TypeError("Rows must be a list")
        if not isinstance(columns, list):
            raise TypeError("Columns must be a list")

        type_values = ["panel", "mean_years", "mean_cids", "pr_years", "pr_cids"]
        rows_values = ["xcat", "ret", "freq", "agg_sigs"]

        if not type in type_values:
            raise ValueError(f"Type must be one of {type_values}")

        if not all([x in rows_values for x in rows]):
            raise ValueError(f"Rows must only contain {rows_values}")

        if not all([x in rows_values for x in columns]):
            raise ValueError(f"Columns must only contain {rows_values}")

        rows_dict = {
            "xcat": self.sigs,
            "ret": self.rets,
            "freq": self.freqs,
            "agg_sigs": self.agg_sigs,
        }

        df_row_names, df_column_names = self.set_df_labels(rows_dict, rows, columns)

        df_result = pd.DataFrame(
            columns=df_column_names, index=df_row_names, dtype=np.float64
        )
        # sort index to prevent performance degradation: PerformanceWarning
        df_result.sort_index(inplace=True)

        loop_tuples: List[Tuple[str, str, str, str]] = [
            (ret, sig, freq, agg_sig)
            for ret in self.rets
            for sig in self.sigs
            for freq in self.freqs
            for agg_sig in self.agg_sigs
        ]

        # Reorder tuples

        for ret, sig, freq, agg_sig in loop_tuples:
            # Prepare xcat and manipulate DataFrame
            xcat = [sig, ret]
            self.manipulate_df(xcat=xcat, freq=freq, agg_sig=agg_sig)
            hash = f"{ret}/{sig}/{freq}/{agg_sig}"

            row = self.get_rowcol(hash, rows)
            column = self.get_rowcol(hash, columns)
            df_result.loc[row, column] = self.calculate_single_stat(
                stat, ret, sig, type
            )

            # Reset self.df and sig to original values
            self.df = self.original_df

        if signal_name_dict is not None:
            # Reorder the index according to the signal_name_dict
            if "xcat" in rows:
                df_result.rename(index=signal_name_dict, inplace=True)
                df_result = self.reindex_multindex_df(
                    df_result, signal_name_dict.values(), "Signal"
                )
            else:
                df_result.rename(columns=signal_name_dict, inplace=True)
                df_result = df_result[signal_name_dict.values()]

        if return_name_dict is not None:
            # Reorder the index according to the return_name_dict
            if "ret" in rows:
                df_result.rename(index=return_name_dict, inplace=True)
                df_result = self.reindex_multindex_df(
                    df_result, return_name_dict.values(), "Return"
                )
            else:
                df_result.rename(columns=return_name_dict, inplace=True)
                df_result = df_result[return_name_dict.values()]

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
        label_dict = {
            "xcat": "Signal",
            "ret": "Return",
            "freq": "Frequency",
            "agg_sigs": "Aggregation",
        }
        if len(rows) == 2:
            rows_names = pd.MultiIndex.from_tuples(
                [(a, b) for a in rows_dict[rows[0]] for b in rows_dict[rows[1]]],
                names=[label_dict[rows[0]], label_dict[rows[1]]],
            )
            columns_names = pd.MultiIndex.from_tuples(
                [(a, b) for a in rows_dict[columns[0]] for b in rows_dict[columns[1]]],
                names=[label_dict[columns[0]], label_dict[columns[1]]],
            )
        elif len(rows) == 1:
            rows_names = rows_dict[rows[0]]
            columns_names = pd.MultiIndex.from_tuples(
                [
                    (a, b, c)
                    for a in rows_dict[columns[0]]
                    for b in rows_dict[columns[1]]
                    for c in rows_dict[columns[2]]
                ],
                names=[
                    label_dict[columns[0]],
                    label_dict[columns[1]],
                    label_dict[columns[2]],
                ],
            )
        elif len(columns) == 1:
            rows_names = pd.MultiIndex.from_tuples(
                [
                    (a, b, c)
                    for a in rows_dict[rows[0]]
                    for b in rows_dict[rows[1]]
                    for c in rows_dict[rows[2]]
                ],
                names=[label_dict[rows[0]], label_dict[rows[1]], label_dict[rows[2]]],
            )
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

        if len(rowcols) == 1:
            result = hash.split("/")[idx.index(rowcols[0])]
        if len(rowcols) == 2:
            result = (
                hash.split("/")[idx.index(rowcols[0])],
                hash.split("/")[idx.index(rowcols[1])],
            )
        if len(rowcols) == 3:
            result = (
                hash.split("/")[idx.index(rowcols[0])],
                hash.split("/")[idx.index(rowcols[1])],
                hash.split("/")[idx.index(rowcols[2])],
            )

        return result


if __name__ == "__main__":

    cids = ["AUD", "CAD", "GBP", "NZD", "USD"]
    xcats = ["XR", "XRH", "CRY", "GROWTH", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
    df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
    df_cids.loc["BRL"] = ["2001-01-01", "2020-11-30", -0.1, 2]
    df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
    df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
    df_cids.loc["USD"] = ["2003-01-01", "2020-12-31", -0.1, 2]

    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["XRH"] = ["2000-01-01", "2020-12-31", 0.2, 1, 0, 0.25]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    # All AUD GROWTH locations.
    filt1 = (dfd["xcat"] == "GROWTH") & (dfd["cid"] == "AUD")
    filt2 = (dfd["xcat"] == "INFL") & (dfd["cid"] == "NZD")

    # Reduced DataFrame.
    dfdx = dfd[~(filt1 | filt2)].copy()
    dfdx["ERA"] = "before 2007"
    dfdx.loc[dfdx["real_date"].dt.year > 2007, "ERA"] = "from 2010"

    cidx = ["AUD", "CAD", "GBP", "USD"]

    def spearman(x, y):
        return stats.spearmanr(x, y)[0]

    from statsmodels.tsa.stattools import grangercausalitytests

    def granger(x, y):
        return grangercausalitytests(
            np.array([x, y]).T, maxlag=3, addconst=True, verbose=False
        )[1][0]["ssr_ftest"][0]

    def granger_pval(x, y):
        return grangercausalitytests(
            np.array([x, y]).T, maxlag=3, addconst=True, verbose=False
        )[1][0]["ssr_ftest"][1]

    sigs = ["CRY"]
    # Additional signals.
    srn = SignalReturnRelations(
        dfd,
        rets="XR",
        sigs=sigs,
        sig_neg=True,
        cosp=True,
        freqs="Q",
        start="2002-01-01",
        ms_panel_test=True,
        additional_metrics=[spearman, granger, granger_pval],
    )

    print(sigs)

    df_dep = srn.summary_table()
    print(df_dep)

    dfsum = srn.single_relation_table(table_type="summary")
    print(dfsum)

    srn = SignalReturnRelations(
        dfd,
        rets="XR",
        sigs=["CRY", "INFL", "GROWTH"],
        sig_neg=[True, True, True],
        cosp=True,
        freqs="M",
        start="2002-01-01",
        additional_metrics=[spearman, granger, granger_pval],
    )

    df_sigs = srn.multiple_relations_table()
    print(df_sigs)

    dfsum = srn.single_relation_table(table_type="cross_section")
    print(dfsum)

    srn.accuracy_bars(
        type="signals",
        title="Accuracy",
    )

    sst = srn.single_statistic_table(stat="granger_pval")

    print(sst)

    sr = SignalReturnRelations(
        dfd,
        rets=["XR", "XRH"],
        sigs=["CRY", "INFL", "GROWTH"],
        freqs="M",
        start="2002-01-01",
        agg_sigs="last",
    )

    srt = sr.single_relation_table()
    mrt = sr.multiple_relations_table()
    sst = sr.single_statistic_table(
        stat="accuracy",
        type="mean_years",
        rows=["ret", "agg_sigs"],
        columns=["xcat", "freq"],
    )

    print(srt)
    print(mrt)
    print(sst)

    # Basic Signal Returns showing for multiple input values

    sr = SignalReturnRelations(
        dfd,
        rets=["XR", "XRH"],
        sigs=["CRY", "INFL", "GROWTH"],
        sig_neg=[True, True, False],
        cosp=True,
        freqs=["M", "Q"],
        agg_sigs=["last", "mean"],
        blacklist=black,
    )

    sr.accuracy_bars(sigs=["CRY", "INFL_NEG"], type="signals", title="Accuracy")
    sr.correlation_bars(sigs=["CRY", "INFL_NEG"], type="signals", title="Correlation")

    srt = sr.single_relation_table(
        ret="XRH", xcat="INFL_NEG", freq="Q", agg_sigs="last"
    )
    mrt = sr.multiple_relations_table()
    sst = sr.single_statistic_table(stat="pearson", show_heatmap=True)

    print(srt)
    print(mrt)
    print(sst)

    # Specifying specific arguments for each of the Signal Return Functions

    srt = sr.single_relation_table(ret="XR", xcat="CRY_NEG", freq="Q", agg_sigs="last")
    print(srt)

    mrt = sr.multiple_relations_table(
        rets=["XR", "GROWTH"], xcats="INFL", freqs=["M", "Q"], agg_sigs=["last", "mean"]
    )
    print(mrt)

    sst = sr.single_statistic_table(
        stat="auc",
        rows=["ret", "xcat", "freq"],
        columns=["agg_sigs"],
        type="mean_cids",
    )
    print(sst)
