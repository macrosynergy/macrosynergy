from typing import Dict, List, Any

import pandas as pd
import numpy as np
from macrosynergy.management.utils import qdf_to_ticker_df, get_xcat



# class InformationStateChanges(object):
#     """
#     # Functions for operations on sparse data
#     # class SparseIndicators(Dict) ...:
#     # """
#     def __init__(self, values: Dict[str, pd.DataFrame] = dict()):
#         # TODO store start and end dates per ticker...
#         self.values: Dict[str, pd.DataFrame] = values

#     @classmethod
#     def from_tickers(cls, tickers: List[str]) -> "InformationStateChanges":
#         return cls({ticker: pd.DataFrame for ticker in tickers})

#     def __setitem__(self, item: str, value: pd.DataFrame):
#         self.values[item] = value

#     def __getitem__(self, item: str) -> pd.DataFrame:
#         return self.values[item]

#     def __setattr__(self, key: str, value: pd.DataFrame):
#         self.values[key] = value

#     def __getattr__(self, item) -> pd.DataFrame:
#         return self.values[item]

#     def __getstate__(self):
#         return self.values

#     def __setstate__(self, values: Dict[str, pd.DataFrame]):
#         self.values = values

#     def keys(self):
#         return self.values.keys()

#     def items(self):
#         return self.values.items()

#     def to_dense(self) -> pd.DataFrame:
#         # TODO convert to QuantamentalDataFrame
#         pass


def _get_diff_data(
    ticker: str,
    p_value: pd.DataFrame,
    p_eop: pd.DataFrame,
    p_grading: pd.DataFrame,
) -> pd.DataFrame:
    # calculate basic density stats
    diff_mask = p_value.diff(axis=0).abs() > 0.0
    diff_density = 100 * diff_mask[ticker].sum() / (~p_value[ticker].isna()).sum()
    fvi = p_value[ticker].first_valid_index().strftime("%Y-%m-%d")
    lvi = p_value[ticker].last_valid_index().strftime("%Y-%m-%d")
    dtrange_str = f"{fvi} : {lvi}"
    ddict = {
        "diff_density": diff_density,
        "date_range": dtrange_str,
    }

    dates = p_value[ticker].index[diff_mask[ticker]]

    # create the diff dataframe
    df_temp = pd.concat(
        (
            p_value.loc[dates, ticker].to_frame("value"),
            p_eop.loc[dates, ticker].to_frame("eop_lag"),
            p_grading.loc[dates, ticker].to_frame("grading"),
        ),
        axis=1,
    )
    df_temp["eop"] = df_temp.index - pd.to_timedelta(df_temp["eop_lag"], unit="D")
    df_temp["release"] = df_temp["eop_lag"].diff(periods=1) < 0

    df_temp = df_temp.sort_index().reset_index()
    df_temp["count"] = df_temp.index

    df_temp = pd.merge(
        left=df_temp,
        right=df_temp.groupby(["eop"], as_index=False)["count"].min(),
        on=["eop"],
        how="outer",
        suffixes=(None, "_min"),
    )
    df_temp["version"] = df_temp["count"] - df_temp["count_min"]
    df_temp["diff"] = df_temp["value"].diff(periods=1)
    df_temp = df_temp.set_index("real_date")[
        ["value", "eop", "version", "grading", "diff"]
    ]

    return df_temp, ddict


def create_delta_data(
    df: pd.DataFrame, return_density_stats: bool = False
) -> pd.DataFrame:
    assert len(df["cid"].unique()) == 1
    # split into value, eop and grading
    p_value = qdf_to_ticker_df(df, value_column="value")
    p_eop = qdf_to_ticker_df(df, value_column="eop_lag")
    p_grading = qdf_to_ticker_df(df, value_column="grading")
    assert set(p_value.columns) == set(p_eop.columns) == set(p_grading.columns)

    # create dicts to store the dataframes and density stats
    isc_dict: Dict[str, Any] = {}
    density_stats: Dict[str, Dict[str, Any]] = {}
    for ticker in p_value.columns:
        df_temp, ddict = _get_diff_data(ticker, p_value, p_eop, p_grading)
        # use the xcat name in the output
        _xcat = get_xcat(ticker)
        isc_dict[_xcat] = df_temp
        density_stats[_xcat] = ddict

    # flatten the density stats
    _dstats_flat = [
        (k, v["diff_density"], v["date_range"]) for k, v in density_stats.items()
    ]

    density_stats_df = (
        pd.DataFrame(_dstats_flat, columns=["ticker", "changes_density", "date_range"])
        .sort_values(by="changes_density", ascending=False)
        .reset_index(drop=True)
    )
    if return_density_stats:
        return isc_dict, density_stats_df
    else:
        return isc_dict


def calculate_score_on_sparse_indicator(
    isc: Dict[str, pd.DataFrame], weight: str = None
):
    # TODO make into a method on InformationStateChanges?
    # TODO adjust score by eop_lag (business days?) to get a native frequency...
    # TODO convert below operation into a function call?
    # Operations on a per key in data dictionary
    for key, v in isc.items():
        mask_rel = v["version"] == 0
        s = v.loc[mask_rel, "diff"]
        # TODO exponential weights (requires knowledge of frequency...)
        std = s.expanding(min_periods=10).std()

        columns = [kk for kk in v.columns if kk != "std"]
        v = pd.merge(
            left=v[columns],
            right=std.to_frame("std"),
            how="left",
            left_index=True,
            right_index=True,
        )
        v["std"] = v["std"].ffill()
        v["zscore"] = v["diff"] / v["std"]
        # TODO write as function (nominator and denominator)
        # Weighted according to changes in information states (not underlying frequency)

        # TODO print options
        changes = v.index.diff()
        print(
            f"{key:30s} days between releases: {changes.min().days:4d} to {changes.max().days:4d} - median {changes.median().days:4d}, linear {changes.median().days/365.25:.4f}, square root {np.sqrt(changes.median().days/365.25):.4f}"
        )
        v["zscore_norm_linear"] = v["zscore"] * v.index.diff().days / 365.24
        v["zscore_norm_squared"] = v["zscore"] * np.sqrt(v.index.diff().days / 365.24)
        isc[key] = v

    # TODO clearer exposition
    return isc


def sparse_to_dense(
    isc: Dict[str, pd.DataFrame],
    value_column: str,
    min_period: pd.Timestamp,
    max_period: pd.Timestamp,
    postfix: str = None,
    add_eop: bool = True,
) -> pd.DataFrame:
    # TODO store real_date min and max in object...
    pz = (
        pd.concat(
            [v[value_column].to_frame(k) for k, v in isc.items()]
            + [
                pd.DataFrame(
                    data=0,
                    index=pd.date_range(
                        start=min_period,
                        end=max_period,
                        freq="B",
                        inclusive="both",
                    ),
                    columns=["rdate"],
                )
            ],
            axis=1,
        )
        .fillna(0)
        .drop(["rdate"], axis=1)
    )

    # Winsorise and remove insignificant values
    pz = (pz / (pz.cumsum(axis=0).abs() > 1e-12).astype(int)).clip(lower=-10, upper=10)

    pz.columns.name = "xcat"
    pz.index.name = "real_date"

    df = (
        pz.stack()
        .to_frame("value")
        .reset_index()
        .assign(cid="USD")[["cid", "xcat", "real_date", "value"]]
    )

    if add_eop:
        p_eop = (
            pd.concat(
                [v["eop"].to_frame(k) for k, v in isc.items()]
                + [
                    pd.DataFrame(
                        data=0,
                        index=pd.date_range(
                            start=df.real_date.min(),
                            end=df.real_date.max(),
                            freq="B",
                            inclusive="both",
                        ),
                        columns=["rdate"],
                    )
                ],
                axis=1,
            )
            .ffill()
            .drop(["rdate"], axis=1)
        )
        p_eop.index.name = "real_date"
        p_eop.columns.name = "xcat"
        df = pd.merge(
            left=df,
            right=p_eop.stack().to_frame("eop").reset_index(),
            how="left",
            on=["real_date", "xcat"],
        )
        df["eop_lag"] = (df["real_date"] - df["eop"]).dt.days

    if postfix:
        df["xcat"] += postfix
    # TODO add eop! (and grading!)
    return df


def temporal_aggregator_exponential(
    df: pd.DataFrame, halflife: int = 5, cid: str = "USD", winsorise: float = None
) -> pd.DataFrame:
    p = df.pivot(index="real_date", columns="xcat", values="value")
    if winsorise:
        p = p.clip(lower=-winsorise, upper=winsorise)
    # Exponential moving average weights (check implementation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
    dfa = p.ewm(halflife=halflife).mean().stack().to_frame("value").reset_index()
    dfa["xcat"] += f"EWM{halflife:d}D"
    dfa["cid"] = cid
    return dfa


def temporal_aggregator_mean(
    df: pd.DataFrame, periods: int = 21, cid: str = "USD", winsorise: float = None
) -> pd.DataFrame:
    p = df.pivot(index="real_date", columns="xcat", values="value")
    if winsorise:
        p = p.clip(lower=-winsorise, upper=winsorise)
    # Exponential moving average weights (check implementation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html)
    dfa = p.rolling(periods=periods).mean().stack().to_frame("value").reset_index()
    dfa["xcat"] += f"MA{periods:d}D"
    dfa["cid"] = cid
    return dfa


# Aggreagte per period (temporal aggregator)
# xcats = [cc + "_N" for cc in growth + inflation + labour + sentiment + financial]
# dfx = msm.reduce_df(df, xcats=xcats, cids=["USD"])


def temporal_aggregator_period(
    isc: Dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    pz = (
        pd.concat(
            [v["zscore_norm_squared"].to_frame(k) for k, v in isc.items()]
            + [
                pd.DataFrame(
                    data=0,
                    index=pd.date_range(
                        start=start,
                        end=end,
                        freq="B",
                        inclusive="both",
                    ),
                    columns=["rdate"],
                )
            ],
            axis=1,
        )
        .fillna(0)
        .drop(["rdate"], axis=1)
    )

    # Winsorise and remove insignificant values
    pz = (pz / (pz.cumsum(axis=0).abs() > 1e-12).astype(int)).clip(lower=-10, upper=10)

    pz.columns.name = "ticker"
    pz.index.name = "real_date"

    # Map out the eop dates
    p_eop = (
        pd.concat(
            [v["eop"].to_frame(k) for k, v in isc.items()]
            + [
                pd.DataFrame(
                    data=0,
                    index=pd.date_range(
                        start=start,
                        end=end,
                        freq="B",
                        inclusive="both",
                    ),
                    columns=["rdate"],
                )
            ],
            axis=1,
        )
        .ffill()
        .drop(["rdate"], axis=1)
    )
    p_eop.index.name = "real_date"
    p_eop.columns.name = "ticker"

    # TODO use zscore instead fo zscore_norm_linear!
    dfa = (
        pd.merge(
            left=pz.stack().to_frame("value").reset_index(),
            right=p_eop.stack().to_frame("eop").reset_index(),
            how="left",
            on=["real_date", "ticker"],
        )
        .sort_values(by=["ticker", "real_date"])
        .rename(columns={"ticker": "xcat"})
    )
    dfa["cumsum"] = dfa.groupby(["xcat", "eop"])["value"].cumsum()
    dfa["cid"] = "USD"

    # Aggregate (cumulatively) on a per period basis
    dfa["csum"] = dfa.groupby(["xcat", "eop"])["value"].cumsum()
    dfa["xcat"] += "_NCSUM"

    return dfa[["real_date", "cid", "xcat", "csum"]].rename(columns={"csum": "value"})


if __name__ == "__main__":
    ...
