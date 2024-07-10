from typing import Dict, List

import pandas as pd
import numpy as np


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


def create_delta_data(df: pd.DataFrame)-> Dict[str, pd.DataFrame]:
    """Create delta data (information state changes)

    :param df: QuantamentalDataFrame
    :return: dictionary of changes

    """
    # TODO check QuantamentalDataFrame and not pd.DataFrame as input
    # TODO store reduce/compact form, together with start/end dates, and timestamps (changes to vintages) essentially the "delta-database"
    # TODO group together for releases (find common releases)

    # TODO check unique (and pivot is possible) - single cid
    p_value = df.pivot(index="real_date", columns="xcat", values="value")
    p_eop = df.pivot(index="real_date", columns="xcat", values="eop_lag")
    p_grading = df.pivot(index="real_date", columns="xcat", values="grading")
    
    # Create dictionary of changes
    isc: Dict[str, pd.DataFrame] = {ticker: pd.DataFrame for ticker in p_value.columns}
    
    mask: pd.DataFrame = p_value.diff(axis=0).abs() > 0
    
    store = []
    print(f"\nDensity of JPMaQS Data\n{'='*60:s}")
    print(f"Percent changes (release or revisions) out of total indicator values\n{'-'*60:s}")
    for ticker in sorted(p_value.columns):
        # print(f"{ticker:30s} {100*mask[ticker].sum()/(~p_value[ticker].isnull()).sum():8.2f}% for period {p_value[ticker].first_valid_index():%Y-%m-%d}/{p_value[ticker].last_valid_index():%Y-%m-%d}")
        store.append((ticker, 100*mask[ticker].sum()/(~p_value[ticker].isnull()).sum(), f"{p_value[ticker].first_valid_index():%Y-%m-%d}/{p_value[ticker].last_valid_index():%Y-%m-%d}"))
        dates = p_value.index[mask[ticker]]
        df_tmp = pd.concat(
            (
                p_value.loc[dates, ticker].to_frame("value"),
                p_eop.loc[dates, ticker].to_frame("eop_lag"),
                p_grading.loc[dates, ticker].to_frame("grading"),
            ),
            axis=1
        )
        df_tmp["eop"] = df_tmp.index - pd.to_timedelta(df_tmp["eop_lag"], unit="D")
        df_tmp["release"] = df_tmp["eop_lag"].diff(periods=1) < 0  # if version == 0
    
        df_tmp = df_tmp.sort_index().reset_index()
        df_tmp["count"] = df_tmp.index
        df_tmp = pd.merge(left=df_tmp, right=df_tmp.groupby(["eop"], as_index=False)["count"].min(), on=["eop"], how="outer", suffixes=(None, "_min"))
        df_tmp["version"] = df_tmp["count"] - df_tmp["count_min"]
        df_tmp["diff"] = df_tmp["value"].diff(periods=1)
        df_tmp = df_tmp.set_index("real_date")[["value", "eop", "version", "grading", "diff"]]  # version = 0 => release
        isc[ticker] = df_tmp
    
    # TODO store as state / print
    print(
        pd.DataFrame(
            store, columns=["Ticker", "Changes", "Period"]
        ).sort_values(by="Changes", ascending=False).reset_index(drop=True)
    )

    return isc


def calculate_score_on_sparse_indicator(isc: Dict[str, pd.DataFrame], weight: str = None):
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
            how='left',
            left_index=True,
            right_index=True
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
        v["zscore_norm_linear"] = v["zscore"] * v.index.diff().days/365.24
        v["zscore_norm_squared"] = v["zscore"] * np.sqrt(v.index.diff().days/365.24)
        isc[key] = v

    # TODO clearer exposition
    return isc


def sparse_to_dense(
        isc: Dict[str, pd.DataFrame],
        value_column: str,
        min_period: pd.Timestamp,
        max_period: pd.Timestamp,
        postfix: str = None,
        add_eop: bool = True
    ) -> pd.DataFrame:
    # TODO store real_date min and max in object...
    pz = pd.concat(
        [
            v[value_column].to_frame(k) for k, v in isc.items()
        ] + [
            pd.DataFrame(
                data=0,
                index=pd.date_range(
                    start=min_period,
                    end=max_period,
                    freq="B",
                    inclusive='both',
                ),
                columns=["rdate"]
            )
        ],
        axis=1
    ).fillna(0).drop(["rdate"], axis=1)

    # Winsorise and remove insignificant values
    pz = (pz / (pz.cumsum(axis=0).abs() > 1e-12).astype(int)).clip(lower=-10, upper=10)
    
    pz.columns.name = "xcat"
    pz.index.name = "real_date"
    
    df = (
        pz
        .stack()
        .to_frame("value")
        .reset_index()
        .assign(cid="USD")
        [["cid", "xcat", "real_date", "value"]]
    )
    
    if add_eop:
        p_eop = pd.concat(
            [
                v["eop"].to_frame(k) for k, v in isc.items()
            ] + [
                pd.DataFrame(
                    data=0,
                    index=pd.date_range(
                        start=df.real_date.min(),
                        end=df.real_date.max(),
                        freq="B",
                        inclusive='both',
                    ),
                    columns=["rdate"]
                )
            ],
            axis=1
        ).ffill().drop(["rdate"], axis=1)
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
        df: pd.DataFrame,
        halflife: int = 5,
        cid: str = "USD",
        winsorise: float =None
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
        df: pd.DataFrame, periods: int = 21, cid: str = "USD", winsorise: float =None
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
        isc: Dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp
    ) -> pd.DataFrame:
    pz = pd.concat(
        [
            v["zscore_norm_squared"].to_frame(k) for k, v in isc.items()
        ] + [
            pd.DataFrame(
                data=0,
                index=pd.date_range(
                    start=start,
                    end=end,
                    freq="B",
                    inclusive='both',
                ),
                columns=["rdate"]
            )
        ],
        axis=1
    ).fillna(0).drop(["rdate"], axis=1)
    
    # Winsorise and remove insignificant values
    pz = (pz / (pz.cumsum(axis=0).abs() > 1e-12).astype(int)).clip(lower=-10, upper=10)
    
    pz.columns.name = "ticker"
    pz.index.name = "real_date"
    
    # Map out the eop dates
    p_eop = pd.concat(
        [
            v["eop"].to_frame(k) for k, v in isc.items()
        ] + [
            pd.DataFrame(
                data=0,
                index=pd.date_range(
                    start=start,
                    end=end,
                    freq="B",
                    inclusive='both',
                ),
                columns=["rdate"]
            )
        ],
        axis=1
    ).ffill().drop(["rdate"], axis=1)
    p_eop.index.name = "real_date"
    p_eop.columns.name = "ticker"
    
    # TODO use zscore instead fo zscore_norm_linear!
    dfa = pd.merge(
        left=pz.stack().to_frame("value").reset_index(),
        right=p_eop.stack().to_frame("eop").reset_index(),
        how="left",
        on=["real_date", "ticker"],
    ).sort_values(by=["ticker", "real_date"]).rename(columns={"ticker": "xcat"})
    dfa["cumsum"] = dfa.groupby(["xcat", "eop"])["value"].cumsum()
    dfa["cid"] = "USD"
        
    # Aggregate (cumulatively) on a per period basis
    dfa["csum"] = dfa.groupby(["xcat", "eop"])["value"].cumsum()
    dfa["xcat"] += "_NCSUM"

    return dfa[["real_date", "cid", "xcat", "csum"]].rename(columns={"csum": "value"})
