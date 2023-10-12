"""
Module for analysing and visualizing signal and a return series.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics as skm
from scipy import stats
from typing import List, Union, Tuple
from datetime import timedelta
from collections import defaultdict
import warnings

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df, categories_df


class SignalBase:
    def __init__(
        self,
        df: pd.DataFrame,
        ret: Union[str, List[str]],
        sig: Union[str, List[str]],
        cosp: bool = False,
        start: str = None,
        end: str = None,
        blacklist: dict = None,
        freq: Union[str, List[str]] = "M",
        agg_sig: Union[str, List[str]] = "last",
        fwin: int = 1,
        slip: int = 0,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"DataFrame expected and not {type(df)}.")

        df["real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

        self.dic_freq = {
            "D": "daily",
            "W": "weekly",
            "M": "monthly",
            "Q": "quarterly",
            "A": "annual",
        }

        required_columns = ["cid", "xcat", "real_date", "value"]

        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                "Dataframe columns must be of value: 'cid', 'xcat','real_date' and 'value'"
            )

        freq_error = f"Frequency parameter must be one of {list(self.dic_freq.keys())}."
        if isinstance(freq, list):
            for f in freq:
                if not f in self.dic_freq.keys():
                    raise ValueError(freq_error)
        else:
            if not freq in self.dic_freq.keys():
                raise ValueError(freq_error)

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

        self.ret = ret
        self.freq = freq

        assert isinstance(cosp, bool), f"<bool> object expected and not {type(cosp)}."
        self.cosp = cosp
        self.start = start
        self.end = end
        self.blacklist = blacklist
        self.fwin = fwin

        self.sig = sig
        self.slip = slip
        self.agg_sig = agg_sig

        self.xcats = list(df["xcat"].unique())
        self.df = df

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

    @classmethod
    def apply_slip(
        self,
        target_df: pd.DataFrame,
        slip: int,
        cids: List[str],
        xcats: List[str],
        metrics: List[str],
    ) -> pd.DataFrame:
        """
        Applied a slip, i.e. a negative lag, to the target DataFrame
        for the given cross-sections and categories, on the given metrics.

        :param <pd.DataFrame> target_df: DataFrame to which the slip is applied.
        :param <int> slip: Slip to be applied.
        :param <List[str]> cids: List of cross-sections.
        :param <List[str]> xcats: List of categories.
        :param <List[str]> metrics: List of metrics to which the slip is applied.
        :return <pd.DataFrame> target_df: DataFrame with the slip applied.
        :raises <TypeError>: If the provided parameters are not of the expected type.
        :raises <ValueError>: If the provided parameters are semantically incorrect.
        """

        target_df = target_df.copy(deep=True)
        if not (isinstance(slip, int) and slip >= 0):
            raise ValueError("Slip must be a non-negative integer.")

        if cids is None:
            cids = target_df["cid"].unique().tolist()
        if xcats is None:
            xcats = target_df["xcat"].unique().tolist()

        sel_tickers: List[str] = [f"{cid}_{xcat}" for cid in cids for xcat in xcats]
        target_df["tickers"] = target_df["cid"] + "_" + target_df["xcat"]

        if not set(sel_tickers).issubset(set(target_df["tickers"].unique())):
            warnings.warn(
                "Tickers targetted for applying slip are not present in the DataFrame.\n"
                f"Missing tickers: {sorted(list(set(sel_tickers) - set(target_df['tickers'].unique())))}"
            )

        slip: int = slip.__neg__()

        target_df[metrics] = target_df.groupby("tickers")[metrics].shift(slip)
        target_df = target_df.drop(columns=["tickers"])

        return target_df

    # NOTE THAT THE ORIGINAL __table__stats does not have a ret argument, so this needs to be rectified during inheritance

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

    # NOTE THAT THE ORIGINAL __output__table__ does not have a ret or sig argument, so this needs to be rectified during inheritance

    def __output_table__(self, cs_type: str = "cids", ret=None, sig=None, srt=False):
        """
        Creates a DataFrame with information on the signal-return relation across
        cross-sections or years and, additionally, the panel.

        :param <str> cs_type: the segmentation type.

        """

        if ret is None:
            ret = self.ret
        if sig is None:
            sig = self.sig

        # Analysis completed exclusively on the primary signal.
        r = [ret]
        if isinstance(sig, list):
            r += sig
        else:
            r.append(sig)
        df = self.df[r]

        # Will remove any timestamps where both the signal & return are not realised.
        # Applicable even if communal sampling has been applied given the alignment
        # excludes the return category.
        df = df.dropna(how="any")

        if cs_type == "cids":
            css = self.cids
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


class SignalsReturns(SignalBase):
    """
    Class for analysing and visualizing signal and a return series.

    :param <pd.Dataframe> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value.
    :param <str, List[str]> rets: one or several target return categories.
    :param <str, List[str]> sigs: list of signal categories to be considered
    :param <int, List[int[> signs: list of signs (direction of impact) to be applied, 
        corresponding to signs.
        Default is 1. i.e. impact is supposed to be positive.
        When -1 is chosen for one or all list elements the signal category
        hat category is taken in negative terms.
    :param <int> slip: slippage of signal application in days. This effectively lags
    the signal series, i.e. values are recorded on a future date, simulating
    time it takes to trade on the signal
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
    :param <str, List[str]> freqs: letters denoting frequency at which the series are to 
        be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
        The return series will always be summed over the sample period.
        The signal series will be aggregated according to the values of `agg_sigs`.
    :param <int, List[int[> agg_sigs: aggregation method applied to the signal values in 
        down-sampling. The default is "last". Alternatives are "mean", "median" and "sum".
        If a single aggregation type is chosen for multiple signal categories it is 
        applied to all of them.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        rets: Union[str, List[str]],
        sigs: Union[str, List[str]],
        signs: Union[int, List[int]] = 1,
        slip: int = 0,
        cosp: bool = False,
        start: str = None,
        end: str = None,
        blacklist: dict = None,
        freqs: Union[str, List[str]] = "M",
        agg_sigs: Union[int, List[int]] = "last",
    ):
        super().__init__(
            df=df,
            ret=rets,
            sig=sigs,
            slip=slip,
            cosp=cosp,
            start=start,
            end=end,
            blacklist=blacklist,
            freq=freqs,
            agg_sig=agg_sigs,
        )
        self.df = df
        self.original_df = df.copy()

        if isinstance(self.sig, list):
            for sig in self.sig:
                assert (
                    sig in self.xcats
                ), "Primary signal must be available in the DataFrame."
                self.signals = sig
        else:
            assert (
                self.sig in self.xcats
            ), "Primary signal must be available in the DataFrame."
            self.signals = [self.sig]

        if isinstance(self.ret, list):
            self.xcats = self.sig + self.ret
        else:
            self.xcats = self.signals + [self.ret]

        self.signs = signs

    def single_relation_table(self, ret=None, xcat=None, freq=None, agg_sigs=None):
        """if isinstance(self.sigs, list):
            sig = self.sigs[0]
        else:
            sig = self.sigs"""
        if ret is None:
            ret = self.ret if not isinstance(self.ret, list) else self.ret[0]
        if freq is None:
            freq = self.freq if not isinstance(self.freq, list) else self.freq[0]
        if agg_sigs is None:
            agg_sigs = (
                self.agg_sig if not isinstance(self.agg_sig, list) else self.agg_sig[0]
            )
        if xcat is None:
            sig = self.sig if not isinstance(self.sig, list) else self.sig[0]
            xcat = [sig, ret]
        else:
            sig = xcat[0]

        cids: List[str] = None
        dfd = reduce_df(
            self.original_df,
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
            target_df=dfd, slip=self.slip, cids=cids, xcats=xcat, metrics=metric_cols
        )

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
            xcat_aggs=[agg_sigs, "sum"],
        )
        self.df = df
        self.cids = list(np.sort(self.df.index.get_level_values(0).unique()))

        if not isinstance(self.signs, list):
            self.signs = [self.signs]

        if -1 in self.signs:
            self.df.loc[:, self.signals] *= -1
            s_copy = self.signals.copy()

            self.signals = [s + "_NEG" for s in self.signals]
            self.sig += "_NEG"
            self.df.rename(columns=dict(zip(s_copy, self.signals)), inplace=True)

        return self.__output_table__(cs_type="cids", ret=ret, sig=sig, srt=True).round(
            decimals=5
        )

    def multiple_relations_table(
        self, rets=None, xcats=None, freqs=None, agg_sigs=None
    ):
        if rets is None:
            rets = self.ret
        if freqs is None:
            freqs = self.freq
        if agg_sigs is None:
            agg_sigs = self.agg_sig
        if xcats is None:
            xcats = self.xcats

        if not isinstance(rets, list):
            rets = [rets]

        df_out = pd.DataFrame()

        xcats = [x for x in xcats if x in self.sig]

        index = []
        for ret in rets:
            for xcat in xcats:
                index.append(f"{ret}/{xcat}")
                df_out = pd.concat(
                    [df_out, self.single_relation_table(ret=ret, xcat=[xcat, ret])]
                )

        df_out.index = index
        return df_out

    def single_statistic_table(
        self,
        stat: str,
        type: str = "panel",
        rows: List[str] = ["xcat", "agg_sigs"],
        columns: List[str] = ["ret", "freq"],
    ):
        type_values = ["panel", "mean_years", "mean_cids", "pr_years", "pr_cids"]
        rows_values = ["xcat", "ret", "freq", "agg_sigs"]

        for row in rows:
            if not row in rows_values:
                raise ValueError(f"Rows must only contain {rows_values}")
        for column in columns:
            if not column in rows_values:
                raise ValueError(f"Columns must only contain {rows_values}")

        if type == "panel":
            x = 1
            """
            For each xcat 
                For each agg_sigs
                    For each ret
                        For each freq
                            calculate statistic like in single relation table
            """
        elif type == "mean_years":
            x = 1
            """
            For each xcat 
                For each agg_sigs
                    For each ret
                        For each freq
                            calculate statistic like in 2nd row of summary table
            """
        elif type == "mean_cids":
            x = 1
        elif type == "pr_years":
            x = 1
        elif type == "pr_cids":
            x = 1
        else:
            raise ValueError(f"Type must be one of {type_values}")

        x += 1

        return 0
    
    def calculate_statistic(self, stat):
        stat_values = [
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
        if not stat in stat_values:
            raise ValueError(f"Stat must be one of {stat_values}")


class SignalReturnRelations(SignalBase):

    """
    Class for analysing and visualizing signal and a return series.

    :param <pd.Dataframe> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value.
    :param <str> ret: return category.
    :param <str> sig: primary signal category for which detailed relational statistics
        can be calculated.
    :param <str, List[str]> rival_sigs: "rival signals" categoies whose basic relational
        statistics can be calculated for comparison with the primary signal category. The
        table, if rival signals are defined, will be generated upon instantiation of the
        object.
        N.B.: parameters that are applied for sig (e.g. sig_neg, freq, and agg_sig) are 
        equally applied to all rival signals.
    :param <bool> sig_neg: if set to True puts the signal in negative terms for all
        analysis. Default is False.
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
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
        The return series will always be summed over the sample period.
        The signal series will be aggregated according to the value of agg_sig.
    :param <str> agg_sig: aggregation method applied to the signal values in down-
        sampling. The default is "last".
        If defined, the additional signals will also use the same aggregation method for
        any down-sampling.
    :param <int> fwin: forward window of return category in base periods. Default is 1.
        This conceptually corresponds to the holding period of a position in
        accordance with the signal.
    :param <int> slip: implied slippage of feature availability for relationship with
        the target category. This mimics the relationship between trading signals and
        returns, which is often characterized by a delay due to the setup of of positions.
        Technically, this is a negative lag (early arrival) of the target category
        in working days prior to any frequency conversion. Default is 0.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        ret: str,
        sig: str,
        rival_sigs: Union[str, List[str]] = None,
        cids: List[str] = None,
        sig_neg: bool = False,
        cosp: bool = False,
        start: str = None,
        end: str = None,
        blacklist: dict = None,
        freq: str = "M",
        agg_sig: str = "last",
        fwin: int = 1,
        slip: int = 0,
    ):
        super().__init__(
            df=df,
            ret=ret,
            sig=sig,
            slip=slip,
            cosp=cosp,
            start=start,
            end=end,
            blacklist=blacklist,
            freq=freq,
            agg_sig=agg_sig,
            fwin=fwin,
        )
        assert (
            self.sig in self.xcats
        ), "Primary signal must be available in the DataFrame."

        signals = [self.sig]
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

        self.xcats = self.signals + [self.ret]

        dfd = reduce_df(
            df=df,
            xcats=self.xcats,
            cids=cids,
            start=self.start,
            end=self.end,
            blacklist=self.blacklist,
        )

        # Since there may be any metrics in the DF at this point, simply apply slip to all.
        metric_cols: List[str] = list(
            set(dfd.columns.tolist()) - set(["real_date", "xcat", "cid"])
        )
        dfd: pd.DataFrame = self.apply_slip(
            target_df=dfd,
            slip=self.slip,
            cids=cids,
            xcats=self.xcats,
            metrics=metric_cols,
        )

        # Naturally, only applicable if rival signals have been passed.
        if self.cosp and len(signals) > 1:
            dfd = self.__communal_sample__(df=dfd)

        self.dfd = dfd

        df = categories_df(
            dfd,
            xcats=self.xcats,
            cids=cids,
            val="value",
            start=None,
            end=None,
            freq=self.freq,
            blacklist=None,
            lag=1,
            fwin=self.fwin,
            xcat_aggs=[self.agg_sig, "sum"],
        )
        self.df = df
        self.cids = list(np.sort(self.df.index.get_level_values(0).unique()))

        if sig_neg:
            self.df.loc[:, self.signals] *= -1
            s_copy = self.signals.copy()

            self.signals = [s + "_NEG" for s in self.signals]
            self.sig += "_NEG"
            self.df.rename(columns=dict(zip(s_copy, self.signals)), inplace=True)

        if len(self.signals) > 1:
            self.df_sigs = self.__rival_sigs__()

        self.df_cs = self.__output_table__(cs_type="cids", ret=ret, sig=self.sig)
        self.df_ys = self.__output_table__(cs_type="years", ret=ret, sig=self.sig)

    def __communal_sample__(self, df: pd.DataFrame):
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
            cid_df = cid_df[self.signals + [self.ret]]

            final_df = pd.DataFrame(
                data=np.empty(shape=cid_df.shape),
                columns=cid_df.columns,
                index=cid_df.index,
            )
            final_df.loc[:, :] = np.NaN

            # Return category is preserved.
            final_df.loc[:, self.ret] = cid_df[self.ret]

            intersection_df = cid_df.loc[:, self.signals].droplevel(level=0)
            # Intersection exclusively across the signals.
            intersection_df = intersection_df.dropna(how="any")
            s_date = intersection_df.index[0]
            e_date = intersection_df.index[-1]

            final_df.loc[
                (c, s_date):(c, e_date), self.signals
            ] = intersection_df.to_numpy()
            storage.append(final_df)

        df = pd.concat(storage)
        df = df.stack().reset_index().sort_values(["cid", "xcat", "real_date"])
        df.columns = ["cid", "real_date", "xcat", "value"]

        return df[["cid", "xcat", "real_date", "value"]]

    def __rival_sigs__(self):
        """
        Produces the panel-level table for the additional signals.
        """

        df_out = pd.DataFrame(index=self.signals, columns=self.metrics)
        df = self.df

        if isinstance(self.ret, list):
            ret = self.ret[0]
        else:
            ret = self.ret

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
        type: str = "cross_section",
        title: str = None,
        size: Tuple[float] = None,
        legend_pos: str = "best",
    ):
        """
        Plot bar chart for the overall and balanced accuracy metrics.

        :param <str> type: type of segment over which bars are drawn. Either
            "cross_section" (default), "years" or "signals".
        :param <str> title: chart header - default will be applied if none is chosen.
        :param <Tuple[float]> size: 2-tuple of width and height of plot - default will be
            applied if none is chosen.
        :param <str> legend_pos: position of legend box. Default is 'best'.
            See the documentation of matplotlib.pyplot.legend.

        """

        assert type in ["cross_section", "years", "signals"]

        if type == "cross_section":
            df_xs = self.df_cs
        elif type == "years":
            df_xs = self.df_ys
        else:
            df_xs = self.df_sigs

        dfx = df_xs[~df_xs.index.isin(["PosRatio"])]

        if title is None:
            refsig = "various signals" if type == "signals" else self.sig
            title = (
                f"Accuracy for sign prediction of {self.ret} based on {refsig} "
                f"at {self.dic_freq[self.freq]} frequency."
            )
        if size is None:
            size = (np.max([dfx.shape[0] / 2, 8]), 6)

        plt.style.use("seaborn")
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

        plt.title(title)
        plt.legend(loc=legend_pos)
        plt.show()

    def correlation_bars(
        self,
        type: str = "cross_section",
        title: str = None,
        size: Tuple[float] = None,
        legend_pos: str = "best",
    ):
        """
        Plot correlation coefficients and significance.

        :param <str> type: type of segment over which bars are drawn. Either
            "cross_section" (default), "years" or "signals".
        :param <str> title: chart header. Default will be applied if none is chosen.
        :param <Tuple[float]> size: 2-tuple of width and height of plot.
            Default will be applied if none is chosen.
        :param <str> legend_pos: position of legend box. Default is 'best'.
            See matplotlib.pyplot.legend.

        """
        assert type in ["cross_section", "years", "signals"]

        if type == "cross_section":
            df_xs = self.df_cs
        elif type == "years":
            df_xs = self.df_ys
        else:
            df_xs = self.df_sigs

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
            refsig = "various signals" if type == "signals" else self.sig
            title = (
                f"Positive correlation probability of {self.ret} "
                f"and lagged {refsig} at {self.dic_freq[self.freq]} frequency."
            )
        if size is None:
            size = (np.max([dfx.shape[0] / 2, 8]), 6)

        plt.style.use("seaborn")
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

        plt.title(title)
        plt.legend(loc=legend_pos)
        plt.show()

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


if __name__ == "__main__":

    cids = ["AUD", "CAD", "GBP", "NZD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
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
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 0, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 0, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 0, 2, 0.8, 0.5]

    black = {"AUD": ["2006-01-01", "2015-12-31"], "GBP": ["2012-01-01", "2100-01-01"]}

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    sr = SignalsReturns(
        dfd,
        rets=["XR"],
        sigs=["CRY", "GROWTH"],
        signs=[1, 1],
        cosp=True,
        freqs=["M", "Q"],
        agg_sigs=["last", "mean"],
    )

    df_single = sr.single_relation_table()
    df_mult = sr.multiple_relations_table()
    df_sst = sr.single_statistic_table("bal_accuracy")

    # Additional signals.
    srn = SignalReturnRelations(
        dfd,
        ret="XR",
        sig="CRY",
        rival_sigs=None,
        sig_neg=True,
        cosp=True,
        freq="M",
        start="2002-01-01",
    )

    dfsum = srn.summary_table()

    r_sigs = ["INFL", "GROWTH"]
    srn = SignalReturnRelations(
        dfd,
        ret="XR",
        sig="CRY",
        rival_sigs=r_sigs,
        sig_neg=True,
        cosp=True,
        freq="M",
        start="2002-01-01",
    )
    dfsum = srn.summary_table()

    df_sigs = srn.signals_table(sigs=["CRY_NEG", "INFL_NEG"])
    df_sigs_all = srn.signals_table()

    srn.accuracy_bars(
        type="signals",
        title="Accuracy measure between target return, XR,"
        " and the respective signals, ['CRY', 'INFL'"
        ", 'GROWTH'].",
    )
