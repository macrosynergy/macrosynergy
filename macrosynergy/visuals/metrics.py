"""
Function for visualising the `eop_lag`, `mop_lag` or `grading` metrics for a given set
of cross sections and extended categories. 

.. code-block:: python

    import macrosynergy.visuals as msv
    ...
    msv.view_metrics(df, xcats="IR", cids=["USD", "EUR", "GBP"], metric='eop_lag')

"""

from typing import List, Optional, Tuple
import pandas as pd
import seaborn as sns

from macrosynergy.visuals import Heatmap


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
    JPMaQS dataset. It generates a heatmap, where the x-axis is the observation date,
    the y-axis is the ticker, and the colour is the lag value.

    Parameters
    ----------
    df : ~pandas.DataFrame
        standardized Qauntamental dataframe.
    xcat : str
        extended category whose lags are to be visualized.
    cids : List[str]
        cross sections to visualize. Default is all in DataFrame.
    start : str
        earliest date in ISO format. Default is earliest available.
    end : str
        latest date in ISO format. Default is latest available.
    metric : str
        name of metric to be visualized. Must be "eop_lag" (default) "mop_lag" or
        "grading".
    freq : str
        frequency of data. Must be one of "D", "W", "M", "Q", "A". Default is "M".
    agg : str
        aggregation method. Must be one of "mean" (default), "median", "min", "max",
        "first" or "last".
    title : str
        string of chart title; if none given default title is printed.
    figsize : Tuple[float]
        Tuple (w, h) of width and height of graph. Default is None, meaning it is set in
        accordance with df.

    Raises
    ------
    TypeError
        if any of the inputs are of the wrong type.
    ValueError
        if any of the inputs are semantically incorrect.

    Returns
    -------
    None
        None
    """

    heatmap = Heatmap(
        df=df,
        cids=cids,
        xcats=[xcat],
        metrics=[metric],
        start=start,
        end=end,
    )

    if title is None:
        title = f"Visualising {metric} for {xcat} from {heatmap.start} to {heatmap.end}"

    if not isinstance(figsize, tuple):
        if not isinstance(figsize, list) or len(figsize) != 2:
            raise TypeError("`figsize` must be a tuple or list of length 2")

    if not isinstance(figsize[0], (int, float)):
        raise ValueError("First element of `figsize` must be a float")

    if figsize[1] is None:
        figsize = (figsize[0], len(heatmap.cids) + 1)  # +1 to adjust for labels

    if not isinstance(figsize[1], (int, float)):
        raise ValueError("Second element of `figsize` must be a float")

    heatmap.plot_metric(
        x_axis_column="real_date",
        y_axis_column="cid",
        metric=metric,
        freq=freq,
        agg=agg,
        title=title,
        figsize=figsize,
        x_axis_label="Date",
        y_axis_label="Cross Sections",
        cmap=sns.color_palette("light:red", as_cmap=True),
        rotate_xticks=90,
    )


if __name__ == "__main__":
    from macrosynergy.management.simulate import make_test_df

    test_cids: List[str] = ["USD", "EUR", "GBP"]
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
