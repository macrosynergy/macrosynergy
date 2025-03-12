"""
Functions used to visualize cross-correlation functions.
"""

from numbers import Number
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


import numpy as np
import matplotlib.pyplot as plt

from macrosynergy.visuals import FacetPlot


def plot_ccf(
    df: pd.DataFrame,
    cids: List[str],
    xcats: str,
    lags: Union[int, Sequence] = 30,
    alpha: float = 0.05,
    remove_zero_predictor: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[Dict[str, List[str]]] = None,
    figsize: Tuple[float, float] = (16, 9),
    title: Optional[str] = None,
    share_x: bool = True,
    share_y: bool = True,
    zero: bool = False,
    **kwargs,
):
    """
    Plots a facet grid of autocorrelation functions for a given xcat and multiple cids.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame with columns ['real_date', 'cid', 'xcat', 'value'].
    cids : List[str]
        List of cids to plot.
    xcats : str
        The xcat to filter and plot ccfs for.
    lags : Union[int, Sequence], default=30
        Number of lags for ccf calculation. If an integer, the lags from 1 to lags are plotted.
        If a sequence is provided, the lags are plotted as given.
    alpha : float, default=0.05
        Significance level for the confidence intervals.
    remove_zero_predictor : bool, default=False
        Remove zeros from the input series.
    blacklist : dict
        cross-sections with date ranges that should be excluded from the data frame. If
        one cross-section has several blacklist periods append numbers to the cross-section
        code.
    start : str
        ISO-8601 formatted date string. Select data from this date onwards. If None, all
        dates are selected.
    end : str
        ISO-8601 formatted date string. Select data up to and including this date. If
        None, all dates are selected.
    figsize : Tuple[float, float], default=(16,9)
        Figure size for the plot.
    title : Optional[str], default=None
        Title for the plot.
    share_x : bool, default=True
        Share x-axis across all subplots.
    share_y : bool, default=True
        Share y-axis across all subplots.
    zero : bool, default=False
        Include the zero lag in the plot.
    kwargs : Dict
        Additional keyword arguments for the plot passed directly to Facetplot.lineplot.
    """

    # _checks_plot_ccf(
    #     df=df,
    #     cids=cids,
    #     xcat=xcat,
    #     lags=lags,
    #     alpha=alpha,
    #     remove_zero_predictor=remove_zero_predictor,
    #     start=start,
    #     end=end,
    #     blacklist=blacklist,
    #     figsize=figsize,
    #     title=title,
    #     share_x=share_x,
    #     share_y=share_y,
    # )

    if title is None:
        title = f"Cross-correlation for {xcats}"

    plot_func = _plot_ccf_wrapper
    plot_func_kwargs = {
        "lags": lags,
        "alpha": alpha,
        "zero": zero,
        "signal_xcat": xcats[0],
        "target_xcat": xcats[1],
    }

    _plot_ccf(
        df=df,
        cids=cids,
        xcats=xcats,
        plot_func=plot_func,
        plot_func_kwargs=plot_func_kwargs,
        remove_zero_predictor=remove_zero_predictor,
        start=start,
        end=end,
        blacklist=blacklist,
        figsize=figsize,
        title=title,
        share_x=share_x,
        share_y=share_y,
        **kwargs,
    )


def _plot_ccf(
    df: pd.DataFrame,
    cids: List[str],
    xcats: str,
    plot_func: Callable,
    plot_func_kwargs: Dict,
    remove_zero_predictor: bool = False,
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[Dict[str, List[str]]] = None,
    figsize: Tuple[float, float] = (16, 9),
    title: Optional[str] = None,
    share_x: bool = True,
    share_y: bool = True,
    **kwargs,
):

    with FacetPlot(
        df=df,
        xcats=xcats,
        cids=cids,
        intersect=True,
        start=start,
        end=end,
        blacklist=blacklist,
        tickers=None,
        metrics=["value"],
    ) as fp:

        if remove_zero_predictor:
            fp.df = fp.df.loc[fp.df["value"] != 0]

        if len(fp.cids) <= 3:
            kwargs["ncols"] = len(fp.cids)

        fp.cids = sorted(fp.cids)

        fp.lineplot(
            plot_func=plot_func,
            plot_func_kwargs=plot_func_kwargs,
            share_x=share_x,
            share_y=share_y,
            figsize=figsize,
            title=title,
            cid_grid=True,
            interpolate=True,
            **kwargs,
        )


def _plot_ccf_wrapper(
    df, plt_dict, signal_xcat, target_xcat, ax=None, lags=[0, 1, 2, 3], **kwargs
):
    """
    Wrapper function to manually compute and plot cross-correlation.

    Parameters:
    - df: Multi-indexed DataFrame (cid, xcat) with a 'value' column.
    - plt_dct: Dictionary containing ticker names ("Y" and optionally "X"),
               where the ticker format is "cid_xcat" (e.g., "AUD_FXXR").
    - ax: Matplotlib axis (optional).
    - lags: Number of lags to compute.
    - kwargs: Expected to contain:
        - "Y": Target ticker in the format "cid_xcat"
        - "X": signal ticker in the format "cid_xcat"
    """

    # Extract cid and xcat from target ticker (e.g., "AUD_FXXR" → cid="AUD", xcat="FXXR")
    cid = plt_dict["Y"][0].split("_")[0]

    # Filter the DataFrame to get the target series
    target_df = (
        df.loc[(cid, target_xcat), ["real_date", "value"]]
        .rename(columns={"value": "value_target"})
        .reset_index(drop=True)
    )
    signal_df = (
        df.loc[(cid, signal_xcat), ["real_date", "value"]]
        .rename(columns={"value": "value_signal"})
        .reset_index(drop=True)
    )

    # Merge on 'real_date' to align both series
    merged_df = target_df.merge(signal_df, on="real_date")

    # # Extract aligned series
    # target_series = merged_df["value_target"]
    # signal_series = merged_df["value_signal"]

    # Compute cross-correlation manually
    cross_corrs = []
    lags=[0, 1, 2, 3]
    for lag in lags:
        shifted_signal = merged_df["value_signal"].shift(lag)  # Shift predictor forward
        shifted_target = merged_df["value_target"]

        valid_data = merged_df.dropna(subset=["value_signal", "value_target"])

        if len(valid_data) > 0:
            corr = shifted_signal.corr(shifted_target)  # Compute Pearson correlation
        else:
            corr = np.nan  # Avoid errors when no valid data

        cross_corrs.append(corr)

    # Plot the manually computed cross-correlation
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.stem(lags, cross_corrs)
    ax.axhline(0, color="black", linestyle="--", lw=1)
    # ax.set_xlabel("Lag")
    # ax.set_ylabel("Cross-Correlation")
    ax.set_title(f"Cross-Correlation: {signal_xcat} vs. {target_xcat}")
    plt.xticks(lags)
    # plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    return ax


def _checks_plot_ccf(
    df: pd.DataFrame,
    cids: List[str],
    xcat: str,
    lags: int = 30,
    alpha=0.05,
    remove_zero_predictor: bool = False,
    method="ywm",
    start: Optional[str] = None,
    end: Optional[str] = None,
    blacklist: Optional[Dict[str, List[str]]] = None,
    figsize: Tuple[float, float] = (16, 9),
    title: Optional[str] = None,
    share_x: bool = True,
    share_y: bool = True,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("`df` must be a pandas DataFrame.")

    if len(df.columns) < 4:
        df = df.copy().reset_index()

    if not isinstance(lags, (int, np.ndarray, list, tuple)):
        raise TypeError("`lags` must be an integer.")

    if not isinstance(alpha, float):
        raise TypeError("`alpha` must be a number.")

    if not isinstance(remove_zero_predictor, bool):
        raise TypeError("`remove_zero_predictor` must be a boolean.")

    if start is None:
        start: str = pd.Timestamp(df["real_date"].min()).strftime("%Y-%m-%d")

    if end is None:
        end: str = pd.Timestamp(df["real_date"].max()).strftime("%Y-%m-%d")

    if not isinstance(xcat, str):
        raise TypeError("`xcat` must be a string.")

    if xcat not in df["xcat"].unique():
        raise ValueError(f"`xcat` {xcat} not found in the DataFrame.")

    if isinstance(cids, str):
        cids: List[str] = [cids]

    if not isinstance(cids, list):
        raise TypeError("`cids` must be a list.")

    if not all(isinstance(cid, str) for cid in cids):
        raise TypeError("All elements in `cids` must be strings.")

    if blacklist:
        if not isinstance(blacklist, dict):
            raise TypeError("`blacklist` must be a dictionary.")
        for key, value in blacklist.items():
            if not isinstance(key, str):
                raise TypeError("Keys in `blacklist` must be strings.")
            if not isinstance(value, list):
                raise TypeError("Values in `blacklist` must be lists.")

    valid_methods = ["ywm", "ywmle", "yw", "ywadjusted", "ols", "ols-adjusted"]
    if method not in valid_methods:
        raise ValueError(f"Invalid value for method. Must be one of {valid_methods}.")

    if not isinstance(figsize, tuple):
        raise TypeError("`figsize` must be a tuple.")

    if title is not None and not isinstance(title, str):
        raise TypeError("`title` must be a string.")

    if not isinstance(share_x, bool):
        raise TypeError("`share_x` must be a boolean.")

    if not isinstance(share_y, bool):
        raise TypeError("`share_y` must be a boolean.")


if __name__ == "__main__":
    import numpy as np

    from macrosynergy.management.simulate import make_test_df
    from macrosynergy.visuals import FacetPlot

    np.random.seed(42)

    cids: List[str] = [
        "USD",
        "EUR",
        "GBP",
        "AUD",
        "CAD",
        "JPY",
        "CHF",
        "NZD",
        "SEK",
        "NOK",
        "DKK",
        "INR",
    ]
    xcats: List[str] = [
        "FXXR",
        "EQXR",
        "RIR",
        "IR",
        "REER",
        "CPI",
        "PPI",
        "M2",
        "M1",
        "M0",
        "FXVOL",
        "FX",
    ]
    sel_cids: List[str] = [
        "USD",
        "EUR",
        "GBP",
        "AUD",
        "CAD",
        "JPY",
        "CHF",
        "NZD",
    ]  # ["USD", "EUR", "GBP"]
    sel_xcats: List[str] = ["FXXR", "EQXR", "RIR", "IR"]
    r_styles: List[str] = [
        "linear",
        "decreasing-linear",
        "sharp-hill",
        "sine",
        "four-bit-sine",
    ]
    df: pd.DataFrame = make_test_df(
        cids=list(set(cids) - set(sel_cids)),
        xcats=xcats,
        start="2000-01-01",
    )

    for rstyle, xcatx in zip(r_styles, sel_xcats):
        dfB: pd.DataFrame = make_test_df(
            cids=sel_cids,
            xcats=[xcatx],
            start="2000-01-01",
            style=rstyle,
        )
        df: pd.DataFrame = pd.concat([df, dfB], axis=0)

    for ix, cidx in enumerate(sel_cids):
        df.loc[df["cid"] == cidx, "value"] = (
            ((df[df["cid"] == cidx]["value"]) * (ix + 1)).reset_index(drop=True).copy()
        )

    for ix, xcatx in enumerate(sel_xcats):
        df.loc[df["xcat"] == xcatx, "value"] = (
            ((df[df["xcat"] == xcatx]["value"]) * (ix * 10 + 1))
            .reset_index(drop=True)
            .copy()
        )
    df.loc[df["xcat"] == "EQXR", "value"] *= (np.arange(len(df.loc[df["xcat"] == "EQXR", "value"])) % 20 == 0)
    df["grading"] = np.nan

    plot_ccf(
        df,
        cids=sel_cids,
        xcats=["EQXR", "FXXR"],
        # title="ccf Facet Plot",
        remove_zero_predictor=False,
        lags=5,
    )
