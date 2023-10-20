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
from macrosynergy.management.utils import apply_slip as apply_slip_util

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
        """
        Signal base is used as a parent class for both SignalReturns and 
        SignalReturnRelations to inherit variables and methods from.

        :param <pd.Dataframe> df: standardized DataFrame with the following necessary
            columns: 'cid', 'xcat', 'real_date' and 'value.
        :param <str> ret: return category.
        :param <str> sig: primary signal category for which detailed relational statistics
            can be calculated.
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
        self.original_df = df.copy()

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
        Function used to call the apply slip method that is defined in management/utils.py
        """
        return apply_slip_util(
            df=df, slip=slip, cids=cids, xcats=xcats, metrics=metrics, raise_error=False
        )

    @staticmethod
    def is_list_of_strings(variable):
        """
        Function used to test whether a variable is a list of strings, to avoid the compiler saying a 
        string is a list of characters
        """
        return isinstance(variable, list) and all(
            isinstance(item, str) for item in variable
        )
    
    def manipulate_df(self, xcat, freq, agg_sig, sig, sst=False, df_result=None):

        cids = None
        dfd = reduce_df(
            self.df,
            xcats=xcat,
            cids=cids,
            start=self.start,
            end=self.end,
            blacklist=self.blacklist,
        )
        metric_cols: List[str] = list(
            set(dfd.columns.tolist())
            - set(["real_date", "xcat", "cid"])
        )
        dfd: pd.DataFrame = self.apply_slip(
            df=dfd,
            slip=self.slip,
            cids=cids,
            xcats=xcat,
            metrics=metric_cols,
        )

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
        self.cids = list(
            np.sort(self.df.index.get_level_values(0).unique())
        )

        if not isinstance(self.signs, list):
            self.signs = [self.signs]

        self.signals = [sig]

        if -1 in self.signs and self.signs[self.sig.index(sig)] == -1:
            original_name = sig + "/" + agg_sig
            self.df.loc[:, self.signals] *= -1
            s_copy = self.signals.copy()

            self.signals = [s + "_NEG" for s in self.signals]
            sig += "_NEG"
            self.df.rename(
                columns=dict(zip(s_copy, self.signals)), inplace=True
            )
            if sst:
                new_name = sig + "/" + agg_sig
                df_result.rename(
                    index={original_name: new_name}, inplace=True
                )
        return df_result

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
            ret = self.ret if not isinstance(self.ret, list) else self.ret[0]
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