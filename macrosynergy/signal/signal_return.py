"""
Module for analysing and visualizing signal and a return series.
"""
import pandas as pd
from typing import List, Union, Tuple, Dict

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.signal.signal_base import SignalBase


class SignalsReturns(SignalBase):
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
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rets: Union[str, List[str]],
        sigs: Union[str, List[str]],
        signs: Union[int, List[int]] = [1],
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

        self.cids = None

        if not self.is_list_of_strings(rets):
            self.ret = [rets]

        if not self.is_list_of_strings(sigs):
            self.sig = [sigs]

        self.xcats = self.sig + self.ret

        self.signs = signs if isinstance(signs, list) else [signs]

        if len(self.signs) > len(self.sig):
            ValueError("Signs must have a length less than or equal to signals")
        self.signals = self.sig

    def single_relation_table(self, ret=None, xcat=None, freq=None, agg_sigs=None):
        """
        Computes all the statistics for one specific signal-return relation:

        :param <str> ret: single target return category
        :param <str> xcat: single signal category to be considered
        :param <str> freq: letter denoting single frequency at which the series will
            be sampled.
            This must be one of the frequencies selected for the class.
            If not specified uses the freq stored in the class.
        :param <str> agg_sigs: aggregation method applied to the signal values in
        down-sampling.
        """
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
        elif isinstance(xcat, list):
            sig = xcat[0]
        else:
            sig = xcat
            xcat = [sig, ret]

        self.signals = [sig]

        self.manipulate_df(xcat=xcat, freq=freq, agg_sig=agg_sigs, sig=sig)

        df_result = self.__output_table__(
            cs_type="cids", ret=ret, sig=sig, srt=True
        ).round(decimals=5)

        self.df = self.original_df

        return df_result

    def multiple_relations_table(
        self, rets=None, xcats=None, freqs=None, agg_sigs=None
    ):
        """
        Calculates all the statistics for each return and signal category specified with
        each frequency and aggregation
        method, note that if none are defined it does this for all categories,
        frequencies and aggregation methods that
        were stored in the class.

        :param <str, List[str]> rets: target return category
        :param <str, List[str]> xcats: signal categories to be considered
        :param <str, List[str]> freqs: letters denoting frequency at which the series
        are to be sampled
        This must be one of 'D', 'W', 'M', 'Q', 'A'. If not specified uses the freq
        stored in the class
        :param <str, List[str]> agg_sigs: aggregation methods applied to the signal
        values in down-sampling
        """
        if rets is None:
            rets = self.ret
        if freqs is None:
            freqs = self.freq
        if agg_sigs is None:
            agg_sigs = self.agg_sig
        if not isinstance(agg_sigs, list):
            agg_sigs = [agg_sigs]
        if xcats is None:
            xcats = self.xcats
        if not isinstance(xcats, list):
            xcats = [xcats]

        if not isinstance(rets, list):
            rets = [rets]

        xcats = [x for x in xcats if x in self.sig]

        index = [
            f"{ret}/{xcat}/{agg_sig}/{freq}"
            for freq in freqs
            for agg_sig in agg_sigs
            for ret in rets
            for xcat in xcats
        ]

        df_out = pd.concat(
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
        df_out.index = index

        return df_out

    def define_rows_and_columns(rows, columns):
        """
        Order of table will always be:
        1) xcat
        2) ret
        3) freq
        4) agg_sigs
        """

    def single_statistic_table(
        self,
        stat: str,
        type: str = "panel",
        rows: List[str] = ["xcat", "agg_sigs"],
        columns: List[str] = ["ret", "freq"],
    ):
        """
        Creates a table which shows the specified statistic for each row and
        column specified as arguments:

        :param stat: type of statistic to be displayed. (this can be any of
        the column names of summary_table)
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
        """
        self.df = self.original_df

        if not "agg_sigs" in rows and not "agg_sigs" in columns:
            agg_sigs = ["last"]
        if not "freqs" in rows and not "freqs" in columns:
            freqs = ["Q"]

        if isinstance(self.freq, list):
            freqs = self.freq
        else:
            freqs = [self.freq]
        if self.is_list_of_strings(self.agg_sig):
            agg_sigs = self.agg_sig
        else:
            agg_sigs = [self.agg_sig]

        type_values = ["panel", "mean_years", "mean_cids", "pr_years", "pr_cids"]
        rows_values = ["xcat", "ret", "freq", "agg_sigs"]

        if not type in type_values:
            raise ValueError(f"Type must be one of {type_values}")
        for row in rows:
            if not row in rows_values:
                raise ValueError(f"Rows must only contain {rows_values}")
        for column in columns:
            if not column in rows_values:  # Rows values is the same as columns values
                raise ValueError(f"Columns must only contain {rows_values}")

        rets = self.ret if isinstance(self.ret, list) else [self.ret]
        sigs = self.sig if isinstance(self.sig, list) else [self.sig]

        rows.sort(reverse=True)
        columns.sort(reverse=True)
        rows_dict = {"xcat": sigs, "ret": rets, "freq": freqs, "agg_sigs": agg_sigs}

        rows_names, columns_names = self.set_df_labels(rows_dict, rows, columns)

        df_result = pd.DataFrame(columns=columns_names, index=rows_names)

        # Define cs_type and type_index mappings
        cs_type_mapping = {"panel": 0, "mean_years": 1, "pr_years": 2}
        type_mapping = {
            "mean_years": "years",
            "pr_years": "years",
            "mean_cids": "cids",
            "pr_cids": "cids",
        }

        loop_tuples: List[Tuple[str, str, str, str]] = [
            (ret, sig, freq, agg_sig)
            for ret in rets
            for sig in sigs
            for freq in freqs
            for agg_sig in agg_sigs
        ]

        for ret, sig, freq, agg_sig in loop_tuples:
            sig_original = sig
            hash = f"{ret}/{sig}/{freq}/{agg_sig}"

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

            # Determine cs_type and type_index
            cs_type = type_mapping.get(type, "cids")
            type_index = cs_type_mapping.get(type, 1)

            # Retrieve output table and update df_result
            df_out = self.__output_table__(cs_type=cs_type, ret=ret, sig=sig)
            single_stat = df_out.iloc[type_index][stat]
            row = self.get_rowcol(hash, rows)
            column = self.get_rowcol(hash, columns)
            df_result[column][row] = single_stat

            # Reset self.df and sig to original values
            self.df = self.original_df
            sig = sig_original

        return df_result

    def set_df_labels(self, rows_dict, rows, columns):
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

    def get_rowcol(self, hash, rowcols):
        """
        Calculates which row/column the hash belongs to.

        :param <str> hash: hash of the statistic.
        :param <List[str]> rowcols: list of strings specifying which of the categories
        are in the rows/columns of the dataframe.
        """
        rowcol = ""
        if "xcat" in rowcols:
            rowcol += hash.split("/")[1] + "/"
        if "ret" in rowcols:
            rowcol += hash.split("/")[0] + "/"
        if "freq" in rowcols:
            rowcol += hash.split("/")[2] + "/"
        if "agg_sigs" in rowcols:
            rowcol += hash.split("/")[3] + "/"

        result = rowcol[:-1]
        return result


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
        rets=["XR", "GROWTH"],
        sigs=["CRY", "INFL"],
        signs=[1, 1],
        cosp=True,
        freqs=["M", "Q"],
        agg_sigs=["last", "mean"],
        blacklist=black,
    )

    srt = sr.single_relation_table()
    mrt = sr.multiple_relations_table()
    sst = sr.single_statistic_table(stat="accuracy")

    print(srt)
    print(mrt)
    print(sst)

    # Specifying specific arguments for each of the Signal Return Functions

    srt = sr.single_relation_table(ret="GROWTH", xcat="CRY", freq="Q", agg_sigs="last")
    print(srt)

    mrt = sr.multiple_relations_table(
        rets=["XR", "GROWTH"], xcats="CRY", freqs=["M", "Q"], agg_sigs=["last", "mean"]
    )
    print(mrt)

    sst = sr.single_statistic_table(
        stat="accuracy",
        rows=["xcat"],
        columns=["ret", "freq", "agg_sigs"],
    )
    print(sst)
