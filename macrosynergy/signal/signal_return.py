"""
Module for analysing and visualizing signal and a return series.
"""
import pandas as pd
from typing import List, Union, Tuple, Dict, Optional
import numpy as np

from macrosynergy.management.simulate import make_qdf
import macrosynergy.visuals as msv


class SignalsReturns():
    """
    Class for analysing and visualizing signal and a return series.

    :param <pd.Dataframe> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value.
    :param <str, List[str]> rets: one or several target return categories.
    :param <str, List[str]> sigs: list of signal categories to be considered
    :param <int, List[int[> signs: list of signs (direction of impact) to be applied,
        corresponding to signs. Default is 1. i.e. impact is supposed to be positive.
        When -1 is chosen for one or all list elements the signal category
        that categories' values are applied in negative terms.
    :param <int> slip: slippage of signal application in working days. This effectively
        lags the signal series, i.e. values are recorded on a future date, simulating
        time it takes to adjust positions in accordance with the signal
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
        may be sampled.
        This must be a selection of 'D', 'W', 'M', 'Q', 'A'. Default is only 'M'.
        The return series will always be summed over the sample period.
        The signal series will be aggregated according to the values of `agg_sigs`.
    :param <str, List[str]> agg_sigs: aggregation method applied to the signal values in
        down-sampling. The default is "last". Alternatives are "mean", "median" and "sum".
        If a single aggregation type is chosen for multiple signal categories it is
        applied to all of them.
    :param <str, List[str]> cids: list of cross-sections to be considered. Default is
        None in which case all cross-sections in the provided `df` are used. `cids` may
        be specified as a list of strings or a single string; only `cids` available for
        all `xcats` are used.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rets: Union[str, List[str]],
        sigs: Union[str, List[str]],
        signs: Union[int, List[int]] = 1,
        slip: int = 0,
        cosp: bool = False,
        start: Optional[str] = None,
        end: Optional[str] = None,
        blacklist: Optional[dict] = None,
        freqs: Union[str, List[str]] = "M",
        agg_sigs: Union[int, List[int]] = "last",
        cids: Union[str, List[str]] = None,
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
            cids=cids,
        )
        self.df = df.copy()

        if not self.is_list_of_strings(rets):
            self.ret = [rets]

        if not self.is_list_of_strings(sigs):
            self.sig = [sigs]

        for sig in self.sig:
            assert (
                sig in self.xcats
            ), "Primary signal must be available in the DataFrame."

        for ret in self.ret:
            assert (
                ret in self.xcats
            ), "Target return must be available in the DataFrame."

        self.xcats = self.sig + self.ret

        self.signs = signs if isinstance(signs, list) else [signs]
        for sign in self.signs:
            if not sign in [-1, 1]:
                raise TypeError("Sign must be either 1 or -1.")

        if len(self.signs) < len(self.sig):
            self.signs.extend([1] * (len(self.sig) - len(self.signs)))

        if len(self.signs) > len(self.sig):
            raise ValueError("Signs must have a length less than or equal to signals")
        self.signals = self.sig


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

    # Basic Signal Returns showing if just single input values

    sr = SignalsReturns(
        dfd,
        rets="XR",
        sigs="CRY",
        freqs="M",
        start="2002-01-01",
        agg_sigs="last",
    )

    srt = sr.single_relation_table()
    mrt = sr.multiple_relations_table()
    sst = sr.single_statistic_table(stat="accuracy")

    print(srt)
    print(mrt)
    print(sst)

    # Basic Signal Returns showing for multiple input values

    sr = SignalsReturns(
        dfd,
        rets=["XR", "XRH"],
        sigs=["CRY", "INFL", "GROWTH"],
        signs=[1, -1],
        cosp=True,
        freqs=["M", "Q"],
        agg_sigs=["last", "mean"],
        blacklist=black,
    )

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
    )
    print(sst)
