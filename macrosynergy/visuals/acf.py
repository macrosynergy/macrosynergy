"""
Functions used to visualize autocorrelation and partial autocorrelation functions.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union, Sequence

import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf as plot_acf_sm
from statsmodels.graphics.tsaplots import plot_pacf as plot_pacf_sm

from macrosynergy.visuals import FacetPlot


def plot_acf(
    df: pd.DataFrame,
    cids: List[str],
    xcat: str,
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
    auto_ylims: bool = True,
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
    xcat : str
        The xcat to filter and plot ACFs for.
    lags : Union[int, Sequence], default=30
        Number of lags for ACF calculation. If an integer, the lags from 1 to lags are plotted.
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
    auto_ylims : bool, default=True
        Automatically set the y-axis limits for each subplot.
    kwargs : Dict
        Additional keyword arguments for the plot passed directly to Facetplot.lineplot.
    """

    _checks_plot_acf(
        df=df,
        cids=cids,
        xcat=xcat,
        lags=lags,
        alpha=alpha,
        remove_zero_predictor=remove_zero_predictor,
        start=start,
        end=end,
        blacklist=blacklist,
        figsize=figsize,
        title=title,
        share_x=share_x,
        share_y=share_y,
    )

    if title is None:
        title = f"Autocorrelation Function (ACF) for {xcat}"

    plot_func = _statsmodels_plot_acf_wrapper
    plot_func_kwargs = {
        "lags": lags,
        "alpha": alpha,
        "zero": zero,
        "auto_ylims": auto_ylims,
    }

    _plot_acf(
        df=df,
        cids=cids,
        xcat=xcat,
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


def plot_pacf(
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
    zero: bool = False,
    auto_ylims: bool = True,
    **kwargs,
):
    """
    Plots a facet grid of partial autocorrelation functions for a given xcat and multiple cids.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame with columns ['real_date', 'cid', 'xcat', 'value'].
    cids : List[str]
        List of cids to plot.
    xcat : str
        The xcat to filter and plot PACFs for.
    lags : Union[int, Sequence], default=30
        Number of lags for PACF calculation. If an integer, the lags from 1 to lags are plotted.
        If a sequence is provided, the lags are plotted as given.
    alpha : float, default=0.05
        Significance level for the confidence intervals.
    remove_zero_predictor : bool, default=False
        Remove zeros from the input series.
    method : str, default='ywm'
        Method for Statsmodel's PACF calculation. Must be one of ['ywm', 'ywmle', 'yw', 'ywadjusted', 'ols', 'ols-adjusted'].
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
    auto_ylims : bool, default=True
        Automatically set the y-axis limits for each subplot.
    kwargs : Dict
        Additional keyword arguments for the plot passed directly to Facetplot.lineplot.
    """
    _checks_plot_acf(
        df=df,
        cids=cids,
        xcat=xcat,
        lags=lags,
        alpha=alpha,
        remove_zero_predictor=remove_zero_predictor,
        method=method,
        start=start,
        end=end,
        blacklist=blacklist,
        figsize=figsize,
        title=title,
        share_x=share_x,
        share_y=share_y,
    )

    if title is None:
        title = f"Partial Autocorrelation Function (PACF) for {xcat}"

    plot_func = _statsmodels_plot_pacf_wrapper
    plot_func_kwargs = {
        "lags": lags,
        "alpha": alpha,
        "method": method,
        "zero": zero,
        "auto_ylims": auto_ylims,
    }

    _plot_acf(
        df=df,
        cids=cids,
        xcat=xcat,
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


def _plot_acf(
    df: pd.DataFrame,
    cids: List[str],
    xcat: str,
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
        xcats=[xcat],
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


def _statsmodels_plot_acf_wrapper(df, plt_dict, ax, **kwargs):
    """
    Wrapper function for statsmodels plot_acf.
    """
    y = plt_dict["Y"][0]
    cid, xcat = str(y).split("_", 1)
    selected_df = df.loc[cid, xcat]
    plot_acf_sm(x=selected_df["value"], ax=ax, title=cid, **kwargs)


def _statsmodels_plot_pacf_wrapper(df, plt_dict, ax, **kwargs):
    """
    Wrapper function for statsmodels plot_pacf.
    """
    y = plt_dict["Y"][0]
    cid, xcat = str(y).split("_", 1)
    selected_df = df.loc[cid, xcat]
    plot_pacf_sm(x=selected_df["value"], ax=ax, title=cid, **kwargs)


def _checks_plot_acf(
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
        "four-bit-sine",
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

    df["value"] = df["value"] * (np.arange(len(df)) % 20 == 0)
    df["grading"] = np.nan

    plot_acf(
        df,
        cids=sel_cids,
        xcat="FXXR",
        # title="ACF Facet Plot",
        remove_zero_predictor=True,
        lags=[5, 6, 7],
        share_y=True,
    )

    plot_pacf(
        df,
        cids=sel_cids,
        xcat="FXXR",
        title="ACF Facet Plot",
        remove_zero_predictor=True,
        zero=True,
        lags=[5, 6, 7],
        share_y=True,
    )
