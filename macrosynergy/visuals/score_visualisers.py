from typing import Dict, List, Optional, Tuple
import warnings

from matplotlib import cm, pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns

from macrosynergy.management.utils.df_utils import (
    reduce_df,
    update_df,
    _map_to_business_day_frequency,
)
from macrosynergy.management.types import QuantamentalDataFrame
from macrosynergy.panel import linear_composite, make_zn_scores


class ScoreVisualisers:
    """
    Class to visualize the z-scores and linear composite of specified categories and
    cross-sections.

    Parameters
    ----------
    df : ~pandas.DataFrame
        A standardized JPMaQS with the following columns: 'cid', 'xcat', 'real_date',
        and at least one metric from - 'value', 'grading', 'eop_lag', or 'mop_lag'.
    cids : List[str], optional
        A list of cross-section identifiers to select from the DataFrame. If None, all
        cross-sections in the frame are selected. Default is None.
    xcats : List[str], optional
        A list of category tickers to select from the DataFrame. If None, all categories
        are selected. Default is None.
    xcat_labels : Dict[str, str], optional
        A dictionary mapping category tickers (keys) to their labels (values). Default is
        None.
    xcat_comp : str
        The name of the composite category. Default is 'Composite'.
    weights : List[float]
        A list of weights for the linear composite. Default is equal weights. The length
        of the list must be equal to the number of categories in xcats. If weights do not
        add up to 1, they are normalized.
    normalize_weights : bool
        If True (default), normalize weights if they do not add to one.
    signs : List[float]
        A list of signs in order to use both negative and positive values of categories
        for the linear composite. This must have the same length as weights and xcats, and
        correspondes to the order of categories in xcats. Default is all positive.
    blacklist : Dict[str, str]
        A dictionary of cross-sections (keys) and date ranges (values) that should be
        excluded. If one cross-section has several blacklist periods append numbers to the
        cross-section identifier.
    complete_xcats : bool
        If True, all xcats must have data for the composite to be calculated. Default is
        False, which means that the composite is calculate if at least one category has
        data.
    no_zn_scores : bool
        Per default, all categories are scored before they are averaged into the
        composite. If True, the class does not calculate scores and takes the average of the
        original categiries. This is useful if those are already score or of similar scale.
    rescore_composite : bool
        If True, the composite is re-scored to a normal unit scale. Default is False.
    sequential : bool
        if True (default) score parameters (neutral level and mean absolute deviation)
        are estimated sequentially with concurrently available information only.
    min_obs : int
        the minimum number of observations required to calculate zn_scores. Default is
        261. The parameter is only applicable if the "sequential" parameter is set to True.
        Otherwise the neutral level and the mean absolute deviation are both computed in-
        sample and will use the full sample.
    iis : bool
        if True (default) zn-scores are also calculated for the initial sample period
        defined by min-obs on an in-sample basis to avoid losing history. This is irrelevant
        if sequential is set to False.
    neutral : str
        The method to calculate the neutral score. Default is 'zero'. Alternatives are
        'mean', 'median' or a number.
    pan_weight : float
        The weight of panel (versus individual cross section) for calculating the
        z-score parameters, i.e. the neutral level and the mean absolute deviation. Default
        is 1, i.e. panel data are the basis for the parameters. Lowest possible value is 0,
        i.e. parameters are all specific to cross section.
    thresh : float
        The threshold value beyond which scores are winsorized, i.e. contained at that
        threshold. The threshold is the maximum absolute score value that the function is
        allowed to produce. The minimum threshold is 1 mean absolute deviation.
    est_freq : str
        the frequency at which mean absolute deviations or means are re-estimated. The
        options are daily, weekly, monthly & quarterly: "D", "W", "M", "Q". Default is
        monthly. Re-estimation is performed at period end.
    postfix : str
        The string appended to category name for output; default is "_ZN".
    """

    def __init__(
        self,
        df: pd.DataFrame,
        xcats: List[str] = None,
        cids: List[str] = None,
        xcat_labels: Dict[str, str] = None,
        xcat_comp: str = "Composite",
        weights: List[float] = None,
        blacklist: Dict[str, str] = None,
        sequential: bool = True,
        iis: bool = True,
        neutral: str = "mean",
        pan_weight: float = 1,
        thresh: float = 3,
        min_obs: int = 261,
        est_freq: str = "m",
        postfix: str = "_ZN",
        normalize_weights: bool = True,
        signs: Optional[List[float]] = None,
        complete_xcats: bool = False,
        no_zn_scores: bool = False,
        rescore_composite: bool = False,
    ):
        self._validate_params(cids, xcats, xcat_comp)

        self.cids = cids if cids else list(df["cid"].unique())
        self.weights = weights
        self.postfix = postfix
        if no_zn_scores:
            self.postfix = ""
        self.xcat_comp = xcat_comp + self.postfix
        df = QuantamentalDataFrame(df)
        self.df = self._create_df(
            df,
            xcats,
            blacklist,
            sequential,
            iis,
            neutral,
            pan_weight,
            thresh,
            min_obs,
            est_freq,
            postfix,
            no_zn_scores,
        )
        self.old_xcats = [xcat_comp] + xcats
        self.xcats = self._apply_postfix(xcats)
        for xcat in self.xcats:
            if xcat not in self.df["xcat"].unique():
                self.xcats.remove(xcat)
                warnings.warn(f"{xcat} not in the DataFrame")

        composite_df = linear_composite(
            self.df,
            xcats=self.xcats,
            cids=self.cids,
            weights=self.weights,
            normalize_weights=normalize_weights,
            signs=signs,
            blacklist=blacklist,
            complete_xcats=complete_xcats,
            new_xcat=self.xcat_comp,
        )

        if rescore_composite:
            composite_df = make_zn_scores(
                composite_df,
                xcat=self.xcat_comp,
                sequential=sequential,
                cids=self.cids,
                blacklist=blacklist,
                iis=iis,
                neutral=neutral,
                pan_weight=pan_weight,
                thresh=thresh,
                min_obs=min_obs,
                est_freq=est_freq,
                postfix="",
            )

        self.df = update_df(self.df, composite_df)
        self.xcats = [self.xcat_comp] + self.xcats
        self.xcat_labels = xcat_labels

    def _validate_params(self, cids, xcats, xcat_comp):
        if cids and (
            not isinstance(cids, list) or not all(isinstance(cid, str) for cid in cids)
        ):
            raise TypeError("cids must be a list of strings")

        if xcats and (
            not isinstance(xcats, list)
            or not all(isinstance(xcat, str) for xcat in xcats)
        ):
            raise TypeError("xcats must be a list of strings")

        if not isinstance(xcat_comp, str):
            raise TypeError("xcat_comp must be a string")

    def _create_df(
        self,
        df,
        xcats,
        blacklist,
        sequential,
        iis,
        neutral,
        pan_weight,
        thresh,
        min_obs,
        est_freq,
        postfix,
        no_zn_scores,
    ):
        """
        Helper function to create the DataFrame with z-scores.
        """
        if no_zn_scores:
            return reduce_df(df, xcats=xcats, cids=self.cids, blacklist=blacklist)

        result_df = None
        for xcat in xcats:
            dfzm = make_zn_scores(
                df,
                xcat=xcat,
                sequential=sequential,
                cids=self.cids,
                blacklist=blacklist,
                iis=iis,
                neutral=neutral,
                pan_weight=pan_weight,
                thresh=thresh,
                min_obs=min_obs,
                est_freq=est_freq,
                postfix=postfix,
            )
            result_df = update_df(result_df, dfzm) if result_df is not None else dfzm
        return result_df

    def _plot_heatmap(
        self,
        df: pd.DataFrame,
        title: str,
        title_fontsize: int = 20,
        annot: bool = True,
        xticks=None,
        yticks_rotation: Optional[int] = None,
        figsize=(20, 10),
        round_decimals: int = 2,
        cmap: str = None,
        cmap_range: Tuple[float, float] = None,
        horizontal_divider: bool = False,
        vertical_divider: bool = False,
        divider_position: int = None,
    ):
        """
        Helper function to plot the heatmap.
        """
        fig, ax = plt.subplots(figsize=figsize)

        cmap = cmap or "coolwarm_r"
        vmax = np.nanmax(np.abs(df.values))
        vmin = -vmax
        cmap_range = cmap_range or (vmin, vmax)

        sns.heatmap(
            df,
            cmap=cmap,
            annot=annot,
            xticklabels="auto",
            yticklabels="auto",
            fmt=f".{round_decimals}f",
            ax=ax,
            vmin=cmap_range[0],
            vmax=cmap_range[1],
        )

        ax.set_title(title, fontsize=title_fontsize)

        if horizontal_divider:
            ax.hlines([divider_position], *ax.get_xlim(), linewidth=2, color="black")
        if vertical_divider:
            ax.vlines([divider_position], *ax.get_ylim(), linewidth=2, color="black")

        plt.xticks(**(xticks or {"rotation": 45, "ha": "right"}))
        if yticks_rotation is not None:
            plt.yticks(rotation=yticks_rotation)
        plt.tight_layout()
        plt.show()

    def _apply_postfix(self, items: List[str]) -> List[str]:
        return [
            item + self.postfix if not item.endswith(self.postfix) else item
            for item in items
        ]

    def _strip_postfix(self, items: List[str]) -> List[str]:
        return [
            item[: -len(self.postfix)] if item.endswith(self.postfix) else item
            for item in items
        ]

    def view_snapshot(
        self,
        cids: List[str] = None,
        xcats: List[str] = None,
        transpose: bool = False,
        date: str = None,
        annot: bool = True,
        title: str = None,
        title_fontsize: int = 20,
        figsize: tuple = (20, 10),
        xcat_labels: Dict[str, str] = None,
        xticks: dict = None,
        yticks_rotation: Optional[int] = None,
        round_decimals: int = 2,
        cmap: str = None,
        cmap_range: Tuple[float, float] = None,
        sort_by_composite: bool = False,
        composite_to_end: bool = False,
        return_as_df: bool = False,
    ):
        """
        View heatmap of the scores at the specified or latest available date.

        Parameters
        ----------
        cids : List[str]
            A list of cross-section identifiers to select from the DataFrame. If None,
            all cross-sections in the frame are selected.
        xcats : List[str]
            A list of category tickers to select from the DataFrame. If None, all
            categories are selected.
        transpose : bool
            If True, transpose the snapshot so cross-section identifiers are on the
            x-axis and category tickers are on the y-axis.
        date : str
            ISO-8601 formatted date. The date of the snapshot. If None, the latest date
            in the DataFrame is selected. Unless the date is today, then the latest date is
            set to the previous business day.
        annot : bool
            If True, annotate the heatmap.
        title : str
            The title of the heatmap.
        title_fontsize : int
            The fontsize of the title.
        figsize : tuple
            The size of the figure.
        xcat_labels : dict
            A dictionary mapping category tickers to their labels.
        xticks : dict
            A dictionary of arguments to label the x axis.
        yticks_rotation : int
            The rotation of the y-axis labels.
        round_decimals : int
            The number of decimals to round the scores to.
        cmap : str
            The colormap of the heatmap.
        cmap_range : tuple
            The range of the colormap.
        return_as_df : bool
            If True, return the DataFrame instead of plotting it.
        """

        cids = cids or self.cids
        xcats = xcats or self.xcats
        xcats = self._apply_postfix(xcats)
        xcat_labels = xcat_labels or self.xcat_labels

        if date:
            date = pd.to_datetime(date)
        else:
            if (
                self.df["real_date"].max().normalize()
                == pd.Timestamp.today().normalize()
            ):
                date = pd.Timestamp.today() - pd.tseries.offsets.BDay(1)
            else:
                date = self.df["real_date"].max()

        date = date.strftime("%Y-%m-%d")

        df = self.df[
            (self.df["xcat"].isin(xcats))
            & (self.df["cid"].isin(cids))
            & (self.df["real_date"] == date)
        ]
        dfw = df.pivot(index="cid", columns="xcat", values="value")
        dfw = dfw.reindex(cids)
        dfw.columns.name = None
        dfw.index.name = None

        composite_zscore = self.xcat_comp

        if sort_by_composite:
            if composite_zscore not in xcats:
                raise ValueError(
                    "Composite category must be in xcats to sort by composite"
                )
            dfw = dfw.sort_values(by=composite_zscore, ascending=False)
        if composite_zscore in xcats:
            if composite_to_end:
                ordering = [xcat for xcat in xcats if xcat != composite_zscore] + [composite_zscore]
            else:
                ordering = [composite_zscore] + [xcat for xcat in xcats if xcat != composite_zscore]
            dfw = dfw[ordering]
        else:
            dfw = dfw[xcats]

        if xcat_labels:
            dfw.columns = [
                xcat_labels.get(
                    self._strip_postfix([xcat])[0],
                    xcat_labels.get(self._apply_postfix([xcat])[0], xcat),
                )
                for xcat in dfw.columns
            ]

        # Drop columns and rows with all NaNs
        dfw = dfw.dropna(axis=1, how="all")
        dfw = dfw.dropna(axis=0, how="all")

        if transpose:
            dfw = dfw.transpose()

        if return_as_df:
            return dfw

        title = title or f"Snapshot for {date}"

        horizontal_divider = transpose and composite_zscore in xcats
        vertical_divider = not transpose and composite_zscore in xcats
        divider_position = None
        if composite_zscore in xcats:
            if composite_to_end:
                divider_position = len(xcats) - 1
            else:
                divider_position = 1

        self._plot_heatmap(
            dfw,
            title=title,
            annot=annot,
            xticks=xticks,
            yticks_rotation=yticks_rotation,
            figsize=figsize,
            title_fontsize=title_fontsize,
            round_decimals=round_decimals,
            cmap=cmap,
            cmap_range=cmap_range,
            horizontal_divider=horizontal_divider,
            vertical_divider=vertical_divider,
            divider_position=divider_position,
        )

    def view_score_evolution(
        self,
        xcat: str,
        freq: str = "A",
        cids: List[str] = None,
        include_latest_period: bool = True,
        include_latest_day: bool = True,
        start: str = None,
        transpose: bool = False,
        annot: bool = True,
        title: str = None,
        title_fontsize: int = 20,
        xticks: dict = None,
        figsize: tuple = (20, 10),
        cmap: str = None,
        cmap_range: Tuple[float, float] = None,
        round_decimals: int = 2,
        return_as_df: bool = False,
    ):
        """
        View the evolution of the scores for the specified category and cross-sections.

        Parameters
        ----------
        xcat : str
            The category to view the evolution of.
        freq : str
            The frequency of the evolution. Must be 'Q', 'A', or 'BA'.
        cids : List[str]
            A list of cross-section identifiers to select from the DataFrame. If None,
            all in the dataframe are selected.
        include_latest_period : bool
            If True, include the latest period in the evolution.
        include_latest_day : bool
            If True, include the latest day in the evolution. If the latest date is
            today, then the latest date is set to the previous business day.
        date : str
            ISO-8601 formatted date. The date of the snapshot. If None, the latest date
            in the DataFrame is selected.
        start : str
            ISO-8601 formatted date. Select data from this date onwards. If None, all
            dates are selected.
        transpose : bool
            If True, transpose the evolution so cross-section identifiers are on the
            x-axis and dates are on the y-axis.
        annot : bool
            If True, annotate the heatmap.
        title : str
            The title of the heatmap.
        title_fontsize : int
            The fontsize of the title.
        xticks : dict
            A dictionary of arguments to label the x axis.
        figsize : tuple
            The size of the figure.
        round_decimals : int
            The number of decimals to round the scores to.
        cmap : str
            The colormap of the heatmap.
        cmap_range : tuple
            The range of the colormap.
        return_as_df : bool
            If True, return the DataFrame instead of plotting it.
        """

        cids = cids or self.cids
        xcat = xcat if xcat.endswith(self.postfix) else xcat + self.postfix

        freq = "2AS" if freq == "BA" else _map_to_business_day_frequency(freq)

        if not (freq in ["2AS", "BA", "A"] or freq.startswith("BQ")):
            raise ValueError("freq must be 'Q', 'A', or 'BA'")

        df = self.df[self.df["xcat"] == xcat]
        df = df[df["cid"].isin(cids)]
        df = df if start is None else df[df["real_date"] >= start]

        dfw = df.pivot(index="real_date", columns="cid", values="value")
        dfw.columns.name = None
        dfw.index.name = None
        dfw_resampled = dfw.resample(freq, origin="start").mean()

        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            if (
                df["real_date"].max().normalize()
                == pd.Timestamp.today().normalize()
            ):
                dfw_resampled.loc[
                    df["real_date"].max() - pd.tseries.offsets.BDay(1)
                ] = dfw.loc[
                    df["real_date"].max() - pd.tseries.offsets.BDay(1)
                ]
                print(
                    "Latest day: ",
                    df["real_date"].max() - pd.tseries.offsets.BDay(1),
                )
            else:
                dfw_resampled.loc[df["real_date"].max()] = dfw.loc[
                    df["real_date"].max()
                ]
                print("Latest day: ", df["real_date"].max())
            if freq in ["Q", "BQ", "BQE"]:
                dfw_resampled.index = list(
                    dfw_resampled.index.to_period("Q").strftime("%YQ%q")[:-1]
                ) + ["Latest"]
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y")[:-1]) + [
                    "Latest"
                ]
        else:
            if freq in ["Q", "BQ", "BQE"]:
                dfw_resampled.index = list(
                    dfw_resampled.index.to_period("Q").strftime("%YQ%q")
                )
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y"))

        dfw_resampled = dfw_resampled.transpose()
        dfw_resampled = dfw_resampled.reindex(cids)

        # Drop columns and rows with all NaNs
        dfw_resampled = dfw_resampled.dropna(axis=1, how="all")
        dfw_resampled = dfw_resampled.dropna(axis=0, how="all")

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        if return_as_df:
            return dfw_resampled

        title = title or f"Evolution for {xcat}"

        self._plot_heatmap(
            dfw_resampled,
            title=title,
            annot=annot,
            xticks=xticks,
            figsize=figsize,
            title_fontsize=title_fontsize,
            round_decimals=round_decimals,
            cmap=cmap,
            cmap_range=cmap_range,
        )

    def view_cid_evolution(
        self,
        cid: str,
        xcats: List[str],
        freq: str,
        include_latest_period: bool = True,
        include_latest_day: bool = True,
        start: str = None,
        transpose: bool = False,
        annot: bool = True,
        title: str = None,
        title_fontsize: int = 20,
        figsize: tuple = (20, 10),
        xticks: dict = None,
        xcat_labels: Dict[str, str] = None,
        cmap: str = None,
        cmap_range: Tuple[float, float] = None,
        round_decimals: int = 2,
        composite_to_end: bool = False,
        return_as_df: bool = False,
    ):
        """
        View the evolution of the scores for the specified cross-section and categories.

        Parameters
        ----------
        cid : str
            The cross-section to view the evolution of.
        xcats : List[str]
            A list of category tickers to select from the DataFrame. If None, all are
            selected.
        freq : str
            The frequency of the evolution. Must be 'Q', 'A', or 'BA'.
        include_latest_period : bool
            If True, include the latest period in the evolution.
        include_latest_day : bool
            If True, include the latest day in the evolution. If the latest date is
            today, then the latest date is set to the previous business day.
        start : str
            ISO-8601 formatted date. Select data from this date onwards. If None, all
            dates are selected.
        transpose : bool
            If True, transpose the evolution so xcats are on the x-axis and dates are on
            the y-axis.
        annot : bool
            If True, annotate the heatmap.
        title : str
            The title of the heatmap.
        title_fontsize : int
            The fontsize of the title.
        xticks : dict
            A dictionary of arguments to label the x axis.
        figsize : tuple
            The size of the figure.
        xcat_labels : dict
            A dictionary mapping xcats to their labels.
        round_decimals : int
            The number of decimals to round the scores to.
        cmap : str
            The colormap of the heatmap.
        cmap_range : tuple
            The range of the colormap.
        return_as_df : bool
            If True, return the DataFrame instead of plotting it.
        """

        if not isinstance(cid, str):
            raise TypeError("cid must be a string")

        freq = "2AS" if freq == "BA" else _map_to_business_day_frequency(freq)

        if not (freq in ["2AS", "BA", "A"] or freq.startswith("BQ")):
            raise ValueError("freq must be 'Q', 'A', or 'BA'")

        xcat_labels = xcat_labels or self.xcat_labels
        xcats = self._apply_postfix(xcats)

        df = self.df[self.df["cid"] == cid]
        df = df if start is None else df[df["real_date"] >= start]
        df = df[df["xcat"].isin(xcats)]
        
        # If there is an xcat that does not exist in the DataFrame, remove it and warn
        for xcat in xcats:
            if xcat not in df["xcat"].unique():
                xcats.remove(xcat)
                warnings.warn(f"{xcat} not in the DataFrame")

        dfw = df.pivot(index="real_date", columns="xcat", values="value")
        dfw.columns.name = None
        dfw.index.name = None
        dfw_resampled = dfw.resample(freq).mean()

        if not include_latest_period:
            dfw_resampled = dfw_resampled.iloc[:-1]

        if include_latest_day:
            if (
                df["real_date"].max().normalize()
                == pd.Timestamp.today().normalize()
            ):
                dfw_resampled.loc[
                    df["real_date"].max() - pd.tseries.offsets.BDay(1)
                ] = dfw.loc[
                    df["real_date"].max() - pd.tseries.offsets.BDay(1)
                ]
                print(
                    "Latest day: ",
                    df["real_date"].max() - pd.tseries.offsets.BDay(1),
                )
            else:
                dfw_resampled.loc[df["real_date"].max()] = dfw.loc[
                    df["real_date"].max()
                ]
                print("Latest day: ", df["real_date"].max())
            if freq in ["Q", "BQ", "BQE"]:
                dfw_resampled.index = list(
                    dfw_resampled.index.to_period("Q").strftime("%YQ%q")[:-1]
                ) + ["Latest"]
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y")[:-1]) + [
                    "Latest"
                ]
        else:
            if freq in ["Q", "BQ", "BQE"]:
                dfw_resampled.index = list(
                    dfw_resampled.index.to_period("Q").strftime("%YQ%q")
                )
            else:
                dfw_resampled.index = list(dfw_resampled.index.strftime("%Y"))

        composite_zscore = self.xcat_comp
        if composite_zscore in xcats:
            dfw_resampled = dfw_resampled[
                [composite_zscore]
                + [xcat for xcat in xcats if xcat != composite_zscore]
            ]
        else:
            dfw_resampled = dfw_resampled[xcats]

        if xcat_labels:
            dfw_resampled.columns = [
                xcat_labels.get(
                    self._strip_postfix([xcat])[0],
                    xcat_labels.get(self._apply_postfix([xcat])[0], xcat),
                )
                for xcat in dfw_resampled.columns
            ]

        dfw_resampled = dfw_resampled.transpose()

        # Drop columns and rows with all NaNs
        dfw_resampled = dfw_resampled.dropna(axis=1, how="all")
        dfw_resampled = dfw_resampled.dropna(axis=0, how="all")

        if transpose:
            dfw_resampled = dfw_resampled.transpose()

        if return_as_df:
            return dfw_resampled

        title = title or f"Evolution for {cid}"

        horizontal_divider = not transpose and self.xcat_comp in xcats
        vertical_divider = transpose and self.xcat_comp in xcats
        divider_position = None
        if composite_zscore in xcats:
            if composite_to_end:
                divider_position = len(xcats) - 1
            else:
                divider_position = 1

        self._plot_heatmap(
            dfw_resampled,
            title=title,
            annot=annot,
            xticks=xticks,
            figsize=figsize,
            title_fontsize=title_fontsize,
            round_decimals=round_decimals,
            cmap=cmap,
            cmap_range=cmap_range,
            horizontal_divider=horizontal_divider,
            vertical_divider=vertical_divider,
            divider_position=divider_position,
        )


if __name__ == "__main__":
    cids_dmea = ["FRF", "DEM", "ITL", "ESP", "EUR"]
    cids_dmxe = ["CHF", "GBP", "JPY", "SEK", "USD"]
    cids_dm = cids_dmea + cids_dmxe
    cids_g10 = ["AUD", "DEM", "FRF", "ESP", "ITL", "JPY", "NZD", "GBP", "USD"]
    cids_latm = ["BRL", "CLP", "COP", "MXN", "PEN"]  # Latam sovereigns
    cids_emea = [
        "CZK",
        "HUF",
        "ILS",
        "PLN",
        "RON",
        "ZAR",
        "TRY",
    ]  # EMEA sovereigns
    cids_emas = [
        "CNY",
        "IDR",
        "KRW",
        "MYR",
        "PHP",
        "THB",
    ]  # EM Asia sovereigns
    cids_ea = ["DEM", "FRF", "ESP", "ITL"]  # major Euro currencies before EUR
    cids_em = cids_emea + cids_latm + cids_emas
    cids = cids_dm + cids_em
    main = [
        "GGIEDGDP_NSA",
        # Currency reserve expansion as % of GDP
        "NIIPGDP_NSA",
        # Monetary base expansion as % of GDP
        "CABGDPRATIO_NSA_12MMA",
        # Intervention-driven liquidity expansion as % of GDP, diff over 3 months
        "GGOBGDPRATIO_NSA",
        # Intervention-driven liquidity expansion as % of GDP, diff over 6 months
    ]

    rets = []

    # rets = [
    #     "DU05YXR_NSA",
    #     "DU05YXR_VT10",
    #     "EQXR_NSA",
    #     "EQXR_VT10",
    #     "FXTARGETED_NSA",
    #     "FXUNTRADABLE_NSA",
    #     "FXXR_VT10",
    # ]

    xcats = main + rets

    tickers = [cid + "_" + xcat for cid in cids for xcat in xcats]

    start_date = "1990-01-01"

    import os
    from macrosynergy.download import JPMaQSDownload

    # Retrieve credentials
    client_id: str = os.getenv("DQ_CLIENT_ID")
    client_secret: str = os.getenv("DQ_CLIENT_SECRET")

    with JPMaQSDownload(client_id=client_id, client_secret=client_secret) as dq:
        df = dq.download(
            tickers=tickers,
            start_date=start_date,
            suppress_warning=True,
            metrics=["all"],
            show_progress=True,
        )

    # Remove data for 24th Feb 2025 for "GGIEDGDP_NSA"

    df = df[
        ~(
            (df["cid"] == "USD")
            & (df["xcat"] == "GGIEDGDP_NSA")
            & (df["real_date"] == "2025-02-24")
        )
    ]

    blacklist = {
        "USD": [pd.Timestamp("2020-06-06"), pd.Timestamp("2030-07-23")]
    }

    sv = ScoreVisualisers(
        df,
        cids=cids,
        xcats=xcats,
        thresh=3,
        no_zn_scores=True,
        complete_xcats=False,
        rescore_composite=True,
        blacklist=blacklist,
    )

    sv.view_snapshot(
        cids=["USD", "EUR", "JPY", "GBP", "CHF"],
        xcats=xcats + ["Composite"],
        figsize=(14, 12),
        sort_by_composite=True,
        composite_to_end=True,
        transpose=False,
        yticks_rotation=45,
    )
    sv.view_cid_evolution(cid="USD", xcats=xcats + ["Composite"] , freq="A", transpose=False)
    sv.view_score_evolution(
        xcat="GGIEDGDP_NSA",
        cids=cids,
        freq="BA",
        transpose=False,
        start="2010-01-01",
        title="AHKSJDA",
        include_latest_day=True,
    )
