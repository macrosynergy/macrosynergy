"""
Classes and functions for analyzing and visualizing the relations of two panel categories.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple
from scipy import stats
import statsmodels.api as sm
import warnings

from macrosynergy.management.simulate import make_qdf
from macrosynergy.management.utils import categories_df
from macrosynergy.management.utils import apply_slip as apply_slip_util


class CategoryRelations(object):
    """
    Class for analyzing and visualizing the relations of multiple panel categories.

    :param <pd.DataFrame> df: standardized DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: exactly two extended categories to be analyzed.
        If there is a hypothesized explanatory-dependent relation, the first category
        is the explanatory variable and the second category the explained variable.
    :param <List[str]> cids: cross-sections for which the category relations is being
        analyzed. Default is all in the DataFrame.
    :param <str> start: earliest date in ISO format. Default is None in which case the
        earliest date in the DataFrame will be used.
    :param <str> end: latest date in ISO format. Default is None in which case the
        latest date in the DataFrame will be used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the analysis.
    :param <int> years: number of years over which data are aggregated. Supersedes the
        'freq' parameter and does not allow lags, Default is None, meaning no multi-year
        aggregation.
        Note: for single year labelled plots, better use freq='A' for cleaner labels.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
    :param <int> lag: lag (delay of arrival) of first (explanatory) category in periods
        as set by freq. Default is 0.
        Importantly, for analyses with explanatory and dependent categories, the first
        category takes the role of the explanatory and a positive lag means that the
        explanatory values will be deferred into the future, i.e. relate to future values
        of the explained variable.
    :param <List[str]> xcat_aggs: Exactly two aggregation methods. Default is 'mean' for
        both.
    :param <str> xcat1_chg: time series changes are applied to the first category.
        Default is None. Options are 'diff' (first difference) and 'pch'
        (percentage change). The changes are calculated over the number of
        periods determined by `n_periods`.
    :param <int> n_periods: number of periods over which changes of the first category
        have been calculated. Default is 1.
    :param <int> fwin: forward moving average window of second category. Default is 1,
        i.e no average.
        Importantly, for analysis with explanatory and dependent categories, the second
        takes the role of the dependent and a forward window means that the dependent
        values average forward into the future.
    :param <List[float]> xcat_trims: two-element list with maximum absolute values
        for the two respective categories. Observations with higher values will be
        trimmed, i.e. removed from the analysis (not winsorized!). Default is None
        for both. Trimming is applied after all other transformations.
    :param <int> slip: implied slippage of feature availability for relationship with
        the target category. This mimics the relationship between trading signals and
        returns, which is often characterized by a delay due to the setup of positions.
        Technically, this is a negative lag (early arrival) of the target category
        in working days prior to any frequency conversion. Default is 0.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        xcats: List[str],
        cids: List[str] = None,
        val: str = "value",
        start: str = None,
        end: str = None,
        blacklist: dict = None,
        years: int = None,
        freq: str = "M",
        lag: int = 0,
        fwin: int = 1,
        xcat_aggs: List[str] = ["mean", "mean"],
        xcat1_chg: str = None,
        n_periods: int = 1,
        xcat_trims: List[float] = [None, None],
        slip: int = 0,
    ):
        """Initializes CategoryRelations"""

        if not isinstance(freq, str):
            raise TypeError("freq must be a string.")

        self.xcats: List[str] = xcats
        self.cids: List[str] = cids
        self.val: str = val
        self.freq: str = freq.upper()
        self.lag: int = lag
        self.years: int = years
        self.aggs: List[str] = xcat_aggs
        self.xcat1_chg: str = xcat1_chg
        self.n_periods: int = n_periods
        self.xcat_trims: List[float] = xcat_trims
        self.slip: int = slip

        if self.freq not in ["D", "W", "M", "Q", "A"]:
            raise ValueError("freq must be one of 'D', 'W', 'M', 'Q', 'A'.")
        if not isinstance(val, str):
            raise TypeError("val must be a string.")
        if not {"cid", "xcat", "real_date", val}.issubset(set(df.columns)):
            raise ValueError(
                "`df` must have columns 'cid', 'xcat', 'real_date' and `val`."
            )
        if not isinstance(xcats, (list, tuple)):
            raise TypeError("`xcats` must be a list or a tuple.")
        elif not len(xcats) == 2:
            raise ValueError("`xcats` must have exactly two elements.")
        if not isinstance(slip, int):
            raise TypeError("`slip` must be a non-negative integer.")
        elif slip < 0:
            raise ValueError("`slip` must be a non-negative integer.")

        if not isinstance(xcat_aggs, (list, tuple)):
            raise TypeError("xcat_aggs must be a list or a tuple.")

        # copy DF to avoid side-effects
        df: pd.DataFrame = df.copy()
        # Select the cross-sections available for both categories.
        df.loc[:, "real_date"] = pd.to_datetime(df["real_date"], format="%Y-%m-%d")

        if self.slip != 0:
            metrics_found: List[str] = list(
                set(df.columns) - set(["cid", "xcat", "real_date"])
            )
            df = self.apply_slip(
                df=df,
                slip=self.slip,
                cids=self.cids,
                xcats=[self.xcats[1]],
                metrics=metrics_found,
            )

        # capture warning from intersection_cids, in case the two categories do not
        # share any cross-sections.
        warnings_list = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            shared_cids = CategoryRelations.intersection_cids(df, xcats, cids)
            for warning in w:
                warnings_list.append(str(warning.message))

        # if shared_cids is empty, then the analysis is not possible.
        # The warning from intersection_cids now becomes an error.
        if len(shared_cids) == 0:
            error_message = "The two categories have no shared cross-sections."
            if len(warnings_list) > 0:
                error_message += f"\nPossible reason(s) for error: "
                error_message += "\n".join(warnings_list)

            error_message += "\nPlease check input parameters."
            raise ValueError(error_message)

        # Will potentially contain NaN values if the two categories are defined over
        # time-periods.
        df = categories_df(
            df,
            xcats,
            shared_cids,
            val=val,
            start=start,
            end=end,
            freq=self.freq,
            blacklist=blacklist,
            years=years,
            lag=lag,
            fwin=fwin,
            xcat_aggs=xcat_aggs,
        )

        if xcat1_chg is not None:
            xcat1_error = (
                "Change applied to the explanatory variable must either be "
                "first-differencing, 'diff', or percentage change, 'pch'."
            )
            assert xcat1_chg in ["diff", "pch"], xcat1_error
            n_periods_error = f"<int> expected and not {type(n_periods)}."
            assert isinstance(n_periods, int), n_periods_error

            df = CategoryRelations.time_series(
                df,
                change=xcat1_chg,
                n_periods=n_periods,
                shared_cids=shared_cids,
                expln_var=xcats[0],
            )

        if any([xt is not None for xt in self.xcat_trims]):
            xcat_trim_error = (
                "Two values expected corresponding to the number " "of categories."
            )
            assert len(xcat_trims) == len(xcats), xcat_trim_error

            types = [
                isinstance(elem, (float, int)) and elem >= 0.0 for elem in xcat_trims
            ]
            assert any(types), "Expected two floating point values."

            df = CategoryRelations.outlier_trim(df, xcats, xcat_trims)

        # NaN values will not be handled if both of the above conditions are not
        # satisfied.
        self.df = df.dropna(axis=0, how="any")

    @classmethod
    def intersection_cids(cls, df, xcats, cids):
        """Returns common cross-sections across both categories and specified
        parameter.

        :param <pd.DataFrame> df: standardised DataFrame.
        :param <List[str]> xcats: exactly two extended categories to be checked on.
        :param <List[str]> cids: cross-sections for which the category relation is being
        analyzed.

        :return <List[str]>: usable: List of the common cross-sections across the two
            categories.
        """

        set_1 = set(df[df["xcat"] == xcats[0]]["cid"])
        set_2 = set(df[df["xcat"] == xcats[1]]["cid"])

        miss_1 = list(set(cids).difference(set_1))
        miss_2 = list(set(cids).difference(set_2))

        if len(miss_1) > 0:
            print(f"{xcats[0]} misses: {sorted(miss_1)}.")
            warnings.warn(f"{xcats[0]} misses: {sorted(miss_1)}.", UserWarning)
        if len(miss_2) > 0:
            print(f"{xcats[1]} misses: {sorted(miss_2)}.")
            warnings.warn(f"{xcats[1]} misses: {sorted(miss_2)}.", UserWarning)

        usable = list(set_1.intersection(set_2).intersection(set(cids)))

        return usable

    @staticmethod
    def apply_slip(
        df: pd.DataFrame,
        slip: int,
        cids: List[str],
        xcats: List[str],
        metrics: List[str],
    ) -> pd.DataFrame:
        return apply_slip_util(
            df=df, slip=slip, cids=cids, xcats=xcats, metrics=metrics, raise_error=False
        )

    @classmethod
    def time_series(
        cls,
        df: pd.DataFrame,
        change: str,
        n_periods: int,
        shared_cids: List[str],
        expln_var: str,
    ):
        """Calculates first differences and percent changes.

        :param <pd.DataFrame> df: multi-index DataFrame hosting the two categories: first
            column represents the explanatory variable; second column hosts the dependent
            variable. The DataFrame's index is the real-date and cross-section.
        :param <str> change: type of change to be applied
        :param <int> n_periods: number of base periods in df over which the change is
            applied.
        :param <List[str]> shared_cids: shared cross-sections across the two categories
            and the received list.
        :param <str> expln_var: only the explanatory variable's data series will be
            changed from the raw value series to a difference or percentage change value.

        :return <pd.Dataframe>: df: returns the same multi-index DataFrame but with an
            adjusted series inline with the 'change' parameter.
        """

        df_lists = []
        for c in shared_cids:
            temp_df: pd.DataFrame = df.loc[c].copy()

            if change == "diff":
                temp_df[expln_var] = temp_df[expln_var].diff(periods=n_periods)
            else:
                temp_df[expln_var] = temp_df[expln_var].pct_change(periods=n_periods)

            temp_df["cid"] = c
            temp_df = temp_df.set_index("cid", append=True)
            df_lists.append(temp_df)

        df_ = pd.concat(df_lists)
        df_ = df_.dropna(axis=0, how="any")
        return df_

    @classmethod
    def outlier_trim(cls, df: pd.DataFrame, xcats: List[str], xcat_trims: List[float]):
        """
        Trim outliers from the dataset.

        :param <pd.DataFrame> df: multi-index DataFrame hosting the two categories. The
            transformations, to each series, have already been applied.
        :param <List[str]> xcats: explanatory and dependent variable.
        :param <List[float]> xcat_trims:

        :return <pd.DataFrame> df: returns the same multi-index DataFrame.

        N.B.:
        Outliers are classified as any datapoint whose absolute value exceeds the
        predefined value specified in the field self.xcat_trims. The values will be set
        to NaN, and subsequently excluded from any regression modelling or correlation
        coefficients.
        """

        xcat_dict = dict(zip(xcats, xcat_trims))

        for k, v in xcat_dict.items():
            df[k] = np.where(np.abs(df[k]) < v, df[k], np.nan)

        df = df.dropna(axis=0, how="any")
        return df

    def corr_prob_calc(
        self, df_probability: Union[pd.DataFrame, List[pd.DataFrame]], prob_est: str
    ):
        """
        Compute the correlation coefficient and probability statistics.

        :param <List[pd.DataFrame] or pd.DataFrame> df_probability: pandas DataFrame
            containing the dependent and explanatory variables.
        :param <str> prob_est: type of estimator for probability of significant relation.

        :return <List[tuple(float, float)]>:

        N.B.: The method is able to handle multiple DataFrames, and will return the
        corresponding number of statistics held inside a List.
        """
        if isinstance(df_probability, pd.DataFrame):
            df_probability = [df_probability]

        cpl = []
        for i, df_i in enumerate(df_probability):
            feat = df_i[self.xcats[0]].to_numpy()
            targ = df_i[self.xcats[1]].to_numpy()
            coeff, pval = stats.pearsonr(feat, targ)
            if prob_est == "map":
                X = df_i.loc[:, self.xcats[0]]
                X = sm.add_constant(X)
                y = df_i.loc[:, self.xcats[1]]
                groups = df_i.reset_index().real_date
                re = sm.MixedLM(
                    y,
                    X,
                    groups,
                ).fit(
                    reml=False
                )  # random effects est
                pval = float(re.summary().tables[1].iloc[1, 3])
            row = [np.round(coeff, 3), np.round(1 - pval, 3)]
            cpl.append(row)
        return cpl

    def corr_probability(
        self,
        df_probability: Union[pd.DataFrame, List[pd.DataFrame]],
        prob_est: str,
        time_period: str = "",
        coef_box_loc: str = "upper left",
        ax: plt.Axes = None,
    ):
        """
        Add the computed correlation coefficient and probability to a Matplotlib table.

        :param <List[pd.DataFrame] or pd.DataFrame> df_probability: pandas DataFrame
            containing the dependent and explanatory variables. Able to handle multiple
            DataFrames representing different time-periods of the original series.
        :param <str> prob_est: type of estimator for probability of significant relation.
        :param <str> time_period: indicator used to clarify which time-period the
            statistics are computed for. For example, before 2010 and after 2010: the two
            periods experience very different macroeconomic conditions. The default is
            an empty string.
        :param <str> coef_box_loc: location on the graph of the aforementioned box. The
            default is in the upper left corner.
        :param <bool> prob_bool: boolean parameter which determines whether the
            probability value is included in the table. The default is True.
        :param <plt.Axes> ax: Matplotlib Axes object. If None (default), new
            axes will be created.

        """
        time_period_error = f"<str> expected - received {type(time_period)}."
        assert isinstance(time_period, str), time_period_error

        cpl = self.corr_prob_calc(df_probability=df_probability, prob_est=prob_est)

        fields = [
            f"Correlation\n coefficient {time_period}",
            f"Probability\n of significance {time_period}",
        ]

        if isinstance(df_probability, list) and len(df_probability) == 2:
            row_headers = ["Before 2010", "After 2010"]
            cellC = [
                ["lightsteelblue", "lightsteelblue"],
                ["lightsalmon", "lightsalmon"],
            ]
        else:
            row_headers = None
            cellC = None

        if ax is None:
            data_table = plt.table(
                cellText=cpl,
                cellColours=cellC,
                colLabels=fields,
                cellLoc="center",
                loc=coef_box_loc,
                zorder=10,
            )
        else:
            data_table = ax.table(
                cellText=cpl,
                cellColours=cellC,
                colLabels=fields,
                cellLoc="center",
                loc=coef_box_loc,
                zorder=10,
            )

        return data_table

    def annotate_facet(self, data, **kws):
        """Annotate each graph within the facet grid."""

        x = data[self.xcats[0]].to_numpy()
        y = data[self.xcats[1]].to_numpy()
        coeff, pval = stats.pearsonr(x, y)

        cpl = np.round(coeff, 3)
        fields = "Correlation coefficient: "
        ax = plt.gca()
        ax.text(0.04, 0.1, f"{fields} {cpl}", fontsize=10, transform=ax.transAxes)

    def reg_scatter(
        self,
        title: str = None,
        labels: bool = False,
        size: Tuple[float] = None,
        xlab: str = None,
        ylab: str = None,
        coef_box: str = None,
        coef_box_font_size: int = 0,
        prob_est: str = "pool",
        fit_reg: bool = True,
        reg_ci: int = 95,
        reg_order: int = 1,
        reg_robust: bool = False,
        separator: Union[str, int] = None,
        title_adj: float = 1,
        single_chart: bool = False,
        single_scatter: bool = False,
        ncol: int = None,
        ax: plt.Axes = None,
    ):
        """
        Display scatter-plot and regression line.

        :param <str> title: title of plot. If None (default) an informative title is
            applied.
        :param <bool> labels: assign a cross-section/period label to each dot.
            Default is False.
        :param <Tuple[float]> size: width and height of the figure
        :param <str> xlab: x-axis label. Default is no label.
        :param <str> ylab: y-axis label. Default is no label.
        :param <bool> fit_reg: if True (default) adds a regression line.
        :param <int> reg_ci: size of the confidence interval for the regression estimate.
            Default is 95. Can be None.
        :param <int> reg_order: order of the regression equation. Default is 1 (linear).
        :param <bool> reg_robust: if this will de-weight outliers, which is
            computationally expensive. Default is False.
        :param <str> coef_box: two-purpose parameter. Firstly, if the parameter equals
            None, the correlation coefficient and probability statistics will not be
            included in the graphic. Secondly, if the statistics are to be included,
            pass in the desired location on the graph which, in addition, will act as a
            pseudo-boolean parameter. The options are standard, i.e. 'upper left',
            'lower right' and so forth. Default is None, i.e the statistics are not
            displayed.
        :param <str> prob_est: type of estimator for probability of significant relation.
            The default is "pool", which means that all observation pairs of a panel
            are pooled and the probability is based on that pool.
            The alternative is "map", denoting Macrosynergy panel test. This is based
            on a panel regression with period-specific random effects and greatly
            mitigates the issue of pseudo-replication if panel features and targets
            are correlated across time.
            See also https://research.macrosynergy.com/testing-macro-trading-factors/
        :param <Union[str, int]> separator: allows categorizing the scatter analysis by
            cross-section or integer. In the former case the argument is set to
            "cids" and in the latter case the argument is set to a year [2010, for
            instance] which will subsequently split the time-period into the sample
            before (not including) that year and from (including) that year.
        :param <float> title_adj: parameter that sets top of figure to accommodate title.
            Default is 1.
        :param <bool> single_chart: boolean parameter determining whether the x- and y-
            labels are only written on a single graph of the Facet Grid (useful if there
            are numerous charts, and the labels are excessively long). The default is
            False, and the names of the axis will be displayed on each grid if not
            conflicting with the label for each variable.
        :param <int> ncol: number of columns in the facet grid. Default is None, in which
            case the number of columns is determined by the number of cross-sections.
        :param <plt.Axes> ax: Matplotlib Axes object. If None (default), new figure and
            axes objects will be created. If an Axes object is passed, the plot will be
            drawn on the Axes, and plt.show() will not be called.
        """

        coef_box_loc_error = (
            "The parameter expects a string used to delimit the "
            "location of the box: 'upper left', 'lower right' etc."
        )
        if coef_box is not None:
            assert isinstance(coef_box, str), coef_box_loc_error

        assert prob_est in ["pool", "map"], "prob_est must be 'pool' or 'map'"

        sns.set_theme(style="whitegrid")
        dfx = self.df.copy()

        if title is None and (self.years is None):
            dates = (
                self.df.index.get_level_values("real_date")
                .to_series()
                .dt.strftime("%Y-%m-%d")
            )
            title = (
                f"{self.xcats[0]} and {self.xcats[1]} "
                f"from {dates.min()} to {dates.max()}"
            )
        elif title is None:
            title = f"{self.xcats[0]} and {self.xcats[1]}"

        if ax is not None:
            if not isinstance(ax, plt.Axes):
                raise TypeError("ax must be a matplotlib Axes object.")
            show_plot = False
        else:
            show_plot = True

        set_font_size = False
        if not (isinstance(coef_box_font_size, int) and coef_box_font_size >= 0):
            raise ValueError("coef_box_font_size must be a non-negative integer.")
        if coef_box_font_size == 0:
            set_font_size = True
            coef_box_font_size = 12

        # If "separator" is type Integer, the scatter plot is split across two
        # time-periods where the divisor is the received year.
        if size is None:
            size = (3, 3) if separator == "cids" else (12, 8)
        else:
            if (
                not isinstance(size, tuple)
                or len(size) != 2
                or not all(isinstance(i, (int, float)) for i in size)
            ):
                raise TypeError("size must be a tuple of ints/floats.")

        if isinstance(separator, int):
            year_error = "Separation by years does not work with year groups."
            assert self.years is None, year_error

            if ax is None:
                fig, ax = plt.subplots(figsize=size)

            index_years = dfx.index.get_level_values(1).year
            years_in_df = list(index_years.unique())

            assert separator in years_in_df, "Separator year is not in the range."
            error_sep = "Separator year must not be the first in the range."
            assert separator > np.min(years_in_df), error_sep

            label_set1 = f"before {separator}"
            label_set2 = f"from {separator}"
            dfx1 = dfx[index_years < separator]
            dfx2 = dfx[index_years >= separator]

            sns.regplot(
                data=dfx1,
                x=self.xcats[0],
                y=self.xcats[1],
                ci=reg_ci,
                order=reg_order,
                robust=reg_robust,
                fit_reg=fit_reg,
                scatter_kws={"s": 30, "alpha": 0.5},
                label=label_set1,
                line_kws={"lw": 1},
                ax=ax,
            )
            sns.regplot(
                data=dfx2,
                x=self.xcats[0],
                y=self.xcats[1],
                ci=reg_ci,
                order=reg_order,
                robust=reg_robust,
                fit_reg=fit_reg,
                label=label_set2,
                scatter_kws={"s": 30, "alpha": 0.5},
                line_kws={"lw": 1},
                ax=ax,
            )

            if coef_box is not None:
                data_table = self.corr_probability(
                    df_probability=[dfx1, dfx2],
                    time_period="",
                    coef_box_loc=coef_box,
                    prob_est=prob_est,
                    ax=ax,
                )
                data_table.scale(0.4, 2.5)
                data_table.auto_set_font_size(set_font_size)
                data_table.set_fontsize(coef_box_font_size)

            ax.legend(loc="upper right")
            ax.set_title(title, fontsize=14)
            if xlab is not None:
                ax.set_xlabel(xlab)
            if ylab is not None:
                ax.set_ylabel(ylab)

        elif separator == "cids" and not single_scatter:
            assert isinstance(single_chart, bool)

            dfx_copy = dfx.reset_index()
            n_cids = len(dfx_copy["cid"].unique())

            error_cids = (
                "There must be more than one cross-section to use "
                "separator = 'cids'."
            )
            assert n_cids > 1, error_cids

            # "Wrap" the column variable at this width, so that the column facets span
            # multiple rows. Used to determine the number of grids on each row.
            dict_coln = {2: 2, 5: 3, 8: 4, 30: 5}

            keys_ar = np.array(list(dict_coln.keys()))
            key = keys_ar[keys_ar <= n_cids][-1]
            if ncol is None:
                ncol = dict_coln[key]
            if ncol > n_cids:
                ncol = n_cids

            # The DataFrame is already a standardised DataFrame. Three columns: two
            # categories (dependent & explanatory variable) and the respective
            # cross-sections. The index will be the date timestamp.

            facet_height = size[1]  # height of each facet in inches
            facet_aspect = size[0] / size[1]  # aspect ratio of each facet

            fg = sns.FacetGrid(
                data=dfx_copy,
                col="cid",
                col_wrap=ncol,
                height=facet_height,
                aspect=facet_aspect,
            )
            fg.map(
                sns.regplot,
                self.xcats[0],
                self.xcats[1],
                ci=reg_ci,
                order=reg_order,
                robust=reg_robust,
                fit_reg=fit_reg,
                scatter_kws={"s": 15, "alpha": 0.5, "color": "lightgray"},
                line_kws={"lw": 1},
            )

            if coef_box is not None:
                fg.map_dataframe(self.annotate_facet)

            fg.set_titles(col_template="{col_name}")
            fg.fig.suptitle(title, y=title_adj, fontsize=14)

            if not single_chart:
                if xlab is not None:
                    fg.set_xlabels(xlab, clear_inner=True)
                if ylab is not None:
                    fg.set_ylabels(ylab)
            else:
                error = "Label expected for the respective axis."
                assert xlab is not None, error
                assert ylab is not None, error
                number_of_graphs = len(fg.axes)
                no_columns = fg._ncol
                remainder = int(number_of_graphs % no_columns)

                for i in range(number_of_graphs):
                    fg.axes[i].set_xlabel("")
                    fg.axes[i].set_ylabel("")

                    if remainder == 0:
                        fg.axes[no_columns - 1].set_xlabel(xlab)
                        fg.axes[no_columns - 1].set_ylabel(ylab)
                    else:
                        fg.axes[-remainder].set_xlabel(xlab)
                        fg.axes[-remainder].set_ylabel(ylab)

        elif separator == "cids" and single_scatter:

            assert isinstance(single_chart, bool)

            if (
                coef_box == "upper right"
            ):  # Since otherwise this overlaps with cid legend
                coef_box = "upper left"

            dfx_copy = dfx.reset_index()
            cids = dfx_copy["cid"].unique()
            n_cids = len(cids)

            error_cids = (
                "There must be more than one cross-section to use "
                "separator = 'cids'."
            )
            assert n_cids > 1, error_cids

            if ax is None:
                fig, ax = plt.subplots(figsize=size)

            # Perform a single global regression
            sns.regplot(
                data=dfx_copy,
                x=self.xcats[0],
                y=self.xcats[1],
                ci=reg_ci,
                order=reg_order,
                robust=reg_robust,
                fit_reg=fit_reg,
                scatter=False,  # Do not plot scatter points in regplot
                line_kws={"lw": 1, "color": "black"},
                ax=ax,
            )

            # Color code the scatter points by cid
            for i, cid in enumerate(cids):
                dfx_i = dfx_copy[dfx_copy["cid"] == cid]
                ax.scatter(
                    dfx_i[self.xcats[0]],
                    dfx_i[self.xcats[1]],
                    label=f"{cid}",
                    s=30,
                    alpha=0.5,
                )

            if coef_box is not None:
                data_table = self.corr_probability(
                    df_probability=dfx_copy,
                    time_period="",
                    coef_box_loc=coef_box,
                    prob_est=prob_est,
                    ax=ax,
                )
                data_table.scale(0.4, 2.5)
                data_table.auto_set_font_size(set_font_size)
                data_table.set_fontsize(coef_box_font_size)

            ax.legend(loc="upper right", title="Cids")
            ax.set_title(title, fontsize=14)
            if xlab is not None:
                ax.set_xlabel(xlab)
            if ylab is not None:
                ax.set_ylabel(ylab)

        elif separator is None:
            if ax is None:
                fig, ax = plt.subplots(figsize=size)
            else:
                show_plot = False

            sns.regplot(
                data=dfx,
                x=self.xcats[0],
                y=self.xcats[1],
                ci=reg_ci,
                order=reg_order,
                robust=reg_robust,
                fit_reg=fit_reg,
                scatter_kws={"s": 30, "alpha": 0.5, "color": "lightgray"},
                line_kws={"lw": 1},
                ax=ax,
            )

            if coef_box is not None:
                data_table = self.corr_probability(
                    df_probability=self.df,
                    prob_est=prob_est,
                    coef_box_loc=coef_box,
                    ax=ax,
                )
                data_table.scale(0.4, 2.5)
                data_table.auto_set_font_size(set_font_size)
                data_table.set_fontsize(coef_box_font_size)

            if labels:
                error_freq = "Labels only available for monthly or lower frequencies."
                assert self.freq in ["A", "Q", "M"], error_freq

                df_labs = self.df.dropna().index.to_frame(index=False)
                if self.years is not None:
                    ser_labs = df_labs["cid"] + " " + df_labs["real_date"]
                else:
                    ser_labs = df_labs["cid"] + " "
                    ser_labs += df_labs["real_date"].dt.year.astype(str)
                    if self.freq == "Q":
                        ser_labs += "Q" + df_labs["real_date"].dt.quarter.astype(str)

                    elif self.freq == "M":
                        ser_labs += "-" + df_labs["real_date"].dt.month.astype(str)

                for i in range(self.df.shape[0]):
                    ax.text(
                        x=self.df[self.xcats[0]][i] + 0,
                        y=self.df[self.xcats[1]][i] + 0,
                        s=ser_labs[i],
                        fontdict=dict(color="black", size=8),
                    )

            ax.set_title(title, fontsize=14)
            if xlab is not None:
                ax.set_xlabel(xlab)
            if ylab is not None:
                ax.set_ylabel(ylab)
        else:
            ValueError("Separator must be either a valid year <int> or 'cids' <str>.")

        plt.tight_layout()
        if show_plot:
            plt.show()

    def ols_table(self, type="pool"):
        """
        Print statsmodels regression summaries.
        :param <str> type: type of linear regression summary to print. Default is 'pool'.
            Alternative is 're' for period-specific random effects.

        """
        assert type in ["pool", "re"], "Type must be either 'pool' or 're'."

        x, y = self.df.dropna().iloc[:, 0], self.df.dropna().iloc[:, 1]
        x_fit = sm.add_constant(x)
        groups = self.df.reset_index().real_date
        if type == "pool":
            fit_results = sm.OLS(y, x_fit).fit()
        elif type == "re":
            fit_results = sm.MixedLM(y, x_fit, groups).fit(reml=False)

        print(fit_results.summary())


if __name__ == "__main__":
    cids = ["AUD", "CAD", "GBP", "NZD", "USD"]
    xcats = ["XR", "CRY", "GROWTH", "INFL"]
    df_cids = pd.DataFrame(
        index=cids, columns=["earliest", "latest", "mean_add", "sd_mult"]
    )
    df_cids.loc["AUD"] = ["2000-01-01", "2020-12-31", 0.1, 1]
    df_cids.loc["CAD"] = ["2001-01-01", "2020-11-30", 0, 1]
    df_cids.loc["BRL"] = ["2001-01-01", "2020-11-30", -0.1, 2]
    df_cids.loc["GBP"] = ["2002-01-01", "2020-11-30", 0, 2]
    df_cids.loc["NZD"] = ["2002-01-01", "2020-09-30", -0.1, 2]
    df_cids.loc["USD"] = ["2003-01-01", "2020-12-31", -0.1, 2]

    cols = ["earliest", "latest", "mean_add", "sd_mult", "ar_coef", "back_coef"]
    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc["XR"] = ["2000-01-01", "2020-12-31", 0.1, 1, 0, 0.3]
    df_xcats.loc["CRY"] = ["2000-01-01", "2020-10-30", 1, 2, 0.95, 1]
    df_xcats.loc["GROWTH"] = ["2001-01-01", "2020-10-30", 1, 2, 0.9, 1]
    df_xcats.loc["INFL"] = ["2001-01-01", "2020-10-30", 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    dfd["grading"] = np.ones(dfd.shape[0])
    black = {"AUD": ["2000-01-01", "2003-12-31"], "GBP": ["2018-01-01", "2100-01-01"]}

    # All AUD GROWTH locations.
    filt1 = (dfd["xcat"] == "GROWTH") & (dfd["cid"] == "AUD")
    filt2 = (dfd["xcat"] == "INFL") & (dfd["cid"] == "NZD")

    # Reduced DataFrame.
    dfdx = dfd[~(filt1 | filt2)].copy()
    dfdx["ERA"] = "before 2007"
    dfdx.loc[dfdx["real_date"].dt.year > 2007, "ERA"] = "from 2010"

    cidx = ["AUD", "CAD", "GBP", "USD", "PRY"]

    cr = CategoryRelations(
        dfdx,
        xcats=["CRY", "XR"],
        freq="M",
        lag=1,
        cids=cidx,
        xcat_aggs=["mean", "sum"],
        start="2001-01-01",
        blacklist=black,
        years=None,
    )

    cr.reg_scatter(
        labels=False,
        separator=None,
        title="Carry and Return",
        xlab="Carry",
        ylab="Return",
        coef_box="lower left",
        prob_est="map",
    )

    # years parameter

    cr = CategoryRelations(
        dfdx,
        xcats=["CRY", "XR"],
        freq="M",
        years=5,
        lag=0,
        cids=cidx,
        xcat_aggs=["mean", "sum"],
        start="2001-01-01",
        blacklist=black,
    )

    cr.reg_scatter(
        labels=False,
        separator=None,
        title="Carry and Return, 5-year periods",
        xlab="Carry",
        ylab="Return",
        coef_box="lower left",
        prob_est="map",
    )

    cr = CategoryRelations(
        dfdx,
        xcats=["CRY", "XR"],
        # xcat1_chg="diff",
        freq="M",
        lag=1,
        cids=cidx,
        xcat_aggs=["mean", "sum"],
        start="2001-01-01",
        blacklist=black,
        years=None,
    )

    cr.reg_scatter(
        labels=False,
        separator=2010,
        title="Carry and Return",
        xlab="Carry",
        ylab="Return",
        coef_box="lower left",
        ncol=5,
    )
    cr.reg_scatter(
        labels=False,
        separator="cids",
        title="Composite macro trend pressure indicator and subsequent IRS fixed receiver returns for USD and EUR, since 2000",
        xlab="Composite macro trend pressure indicator",
        ylab="Next month's return on 2-year IRS return, vol-targeted position, %",
        coef_box="lower left",
        ncol=2,
        single_plot=True,
    )

    # Passing Axes object for a subplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    for i in range(2):
        cr.reg_scatter(
            labels=False,
            separator=None,
            title="Carry and Return",
            xlab="Carry",
            ylab="Return",
            coef_box="lower left",
            prob_est="map",
            ax=ax[i],
        )
    plt.show()

    cr.ols_table(type="pool")
    cr.ols_table(type="re")
