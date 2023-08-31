"""
Function for visualising the `eop_lag`, `mop_lag` or `grading` metrics for a given
set of cross sections and extended categories.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional
from macrosynergy.management.simulate_quantamental_data import make_test_df
from macrosynergy.management.utils import is_valid_iso_date
from macrosynergy.management.shape_dfs import reduce_df


def view_metrics(
    df: pd.DataFrame,
    xcat: str,
    cids: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    freq: str = "M",
    agg: str = "mean",
    metric: str = "eop_lag",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float]] = (14, None),
) -> None:
    """
    A function to visualise the `eop_lag`, `mop_lag` or `grading` metrics for a given
    JPMaQS dataset. It generates a heatmap, where the x-axis is the observation
    date, the y-axis is the ticker, and the colour is the lag value.

    :param <pd.Dataframe> df: standardized DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and 'grading', 'eop_lag' or 'mop_lag'.
    :param str xcat: extended category whose lags are to be visualized.
    :param <List[str]> cids: cross sections to visualize. Default is all in DataFrame.
    :param <str> start: earliest date in ISO format. Default is earliest available.
    :param <str> end: latest date in ISO format. Default is latest available.
    :param <str> metric: name of metric to be visualized. Must be "eop_lag" (default)
        "mop_lag" or "grading".
    :param <str> freq: frequency of data. Must be one of "D", "W", "M", "Q", "A".
        Default is "M".
    :param <str> agg: aggregation method. Must be one of "mean" (default), "median",
        "min", "max", "first" or "last".
    :param <str> title: string of chart title; if none given default title is printed.
    :param <Tuple[float]> figsize: Tuple (w, h) of width and height of graph.
        Default is None, meaning it is set in accordance with df.

    :return: None

    :raises TypeError: if any of the inputs are of the wrong type.
    :raises ValueError: if any of the inputs are semantically incorrect.
    """
    # Validating inputs
    # First check if the standard valid and expected columns are present
    if not isinstance(metric, str):
        raise TypeError("`metric` must be a string")
    elif metric not in ["eop_lag", "mop_lag", "grading"]:
        raise ValueError("`metric` must be either 'eop_lag', 'mop_lag' or 'grading'")

    expc_cols: List[str] = ["cid", "xcat", "real_date", metric]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a DataFrame")
    elif df.empty:
        raise ValueError("`df` must not be empty")
    elif not set(expc_cols).issubset(df.columns):
        raise ValueError(
            f"`df` must have columns 'cid', 'xcat', 'real_date' and '{metric}'"
        )
    else:
        # Filter to only necessary columns
        df = df.copy()[expc_cols]

    # Check if xcat is a single string and is in the DataFrame

    if not isinstance(xcat, str):
        raise TypeError("`xcat` must be a string")
    else:
        if not xcat in df["xcat"].unique():
            raise ValueError(f"No data for `xcat` = {xcat} in DataFrame `df`")

    # Check if cids is a list of strings and is in the DataFrame
    if cids is not None:
        if (
            (not isinstance(cids, list))
            or (not all(isinstance(cid, str) for cid in cids))
            or (len(cids) == 0)
        ):
            raise TypeError("`cids` must be a list of strings")

        if not set(cids).issubset(df["cid"].unique()):
            raise ValueError("All cids in `cids` must be in the DataFrame `df`")

    else:
        # In case cids is None, set it to all cids in the DataFrame
        cids = list(df["cid"].unique())

    # Validate start and end dates
    df["real_date"]: pd.DatetimeIndex = pd.to_datetime(df["real_date"])
    min_date: pd.Timestamp = df["real_date"].min()
    max_date: pd.Timestamp = df["real_date"].max()
    if start is None:
        start: str = min_date.strftime("%Y-%m-%d")
    if end is None:
        end: str = max_date.strftime("%Y-%m-%d")

    for varx, namex in zip([start, end], ["start", "end"]):
        if not is_valid_iso_date(varx):
            raise ValueError(f"`{namex}` must be a valid ISO date string")

    # Validate freq and agg
    if not isinstance(freq, str):
        raise TypeError("`freq` must be a string")
    else:
        freq: str = freq.upper()
        if freq not in ["D", "W", "M", "Q", "A"]:
            raise ValueError("`freq` must be one of 'D', 'W', 'M', 'Q' or 'A'")

    if not isinstance(agg, str):
        raise TypeError("`agg` must be a string")
    else:
        agg: str = agg.lower()
        if agg not in ["mean", "median", "min", "max", "first", "last"]:
            raise ValueError(
                "`agg` must be one of 'mean', 'median', 'min', 'max', 'first' or 'last'"
            )

    # Validate title and figsize tuple.
    # Figsize tuple allows second value to be None,
    # which is then set to the number of cids
    if title is not None:
        if not isinstance(title, str):
            raise TypeError("`title` must be a string")

    if not isinstance(figsize, tuple):
        if not isinstance(figsize, list) or len(figsize) != 2:
            raise TypeError("`figsize` must be a tuple or list of length 2")

    if not isinstance(figsize[0], (int, float)):
        raise ValueError("First element of `figsize` must be a float")

    if figsize[1] is None:
        figsize = (figsize[0], len(cids) + 1) # +1 to adjust for labels

    if not isinstance(figsize[1], (int, float)):
        raise ValueError("Second element of `figsize` must be a float")

    # Actual plotting code

    # filter and reduce the df to the required cids, xcat, start and end
    filtered_df: pd.DataFrame = reduce_df(
        df=df, cids=cids, xcats=[xcat], start=start, end=end
    )
    # group by cid, then set the index to real_date, then resample to the required freq and agg
    filtered_df: pd.DataFrame = (
        filtered_df.groupby(["cid", "xcat"])
        .apply(
            lambda x: x.set_index("real_date")
            .resample(freq)
            .agg(agg, numeric_only=True)
        )
        .reset_index()
    )

    filtered_df["real_date"]: pd.Series = filtered_df["real_date"].dt.strftime(
        "%Y-%m-%d"
    )

    pivoted_df: pd.DataFrame = filtered_df.pivot_table(
        index="cid", columns="real_date", values=metric
    )

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    max_mes: float = max(1, filtered_df[metric].max())
    min_mes: float = min(0, filtered_df[metric].min())
    sns.heatmap(
        pivoted_df,
        cmap=sns.color_palette("light:red", as_cmap=True),
        vmin=min_mes,
        vmax=max_mes,
        ax=ax,
    )

    if title is None:
        title = f"Visualising {metric} for {xcat} from {start} to {end}"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cross Sections")
    plt.show()


if __name__ == "__main__":
    test_cids: List[str] = ["USD",]#"EUR", "GBP"]
    test_xcats: List[str] = ["FX", "IR"]
    dfE: pd.DataFrame = make_test_df(
        cids=test_cids, xcats=test_xcats, style="sharp-hill"
    )

    dfM: pd.DataFrame = make_test_df(
        cids=test_cids, xcats=test_xcats, style="four-bit-sine"
    )

    dfG: pd.DataFrame = make_test_df(cids=test_cids, xcats=test_xcats, style="sine")

    dfE.rename(columns={"value": "eop_lag"}, inplace=True)
    dfM.rename(columns={"value": "mop_lag"}, inplace=True)
    dfG.rename(columns={"value": "grading"}, inplace=True)
    mergeon = ["cid", "xcat", "real_date"]
    dfx: pd.DataFrame = pd.merge(pd.merge(dfE, dfM, on=mergeon), dfG, on=mergeon)

    view_metrics(
        df=dfx,
        xcat="FX",
    )
    view_metrics(
        df=dfx,
        xcat="IR",
        metric="mop_lag",
    )
    view_metrics(
        df=dfx,
        xcat="IR",
        metric="grading",
    )
