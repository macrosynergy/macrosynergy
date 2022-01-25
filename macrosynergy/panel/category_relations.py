import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple
from scipy import stats
import statsmodels.api as sm

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import categories_df


class CategoryRelations:
    """Class for analyzing and visualizing two categories across a panel

    :param <pd.Dataframe> df: standardized data frame with the necessary columns:
        'cid', 'xcat', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: exactly two extended categories to be checked on.
        If there is a hypothesized explanatory-dependent relation, the first category
        is the explanatory variable and the second category the explained variable.
    :param <List[str]> cids: cross-sections for which the category relation is being
        analyzed. Default is all in the dataframe.
    :param <str> start: earliest date in ISO format. Default is None in which case the
        earliest date in the df will be used.
    :param <str> end: latest date in ISO format. Default is None in which case the
        latest date in the df will be used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded from
        the analysis.
    :param <int> years: number of years over which data are aggregated. Supersedes freq
        and does not allow lags, Default is None, meaning no multi-year aggregation.
        Note: for single year labelled plots, better use freq='A' for cleaner labels.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
    :param <int> lag: lag (delay of arrival) of first (explanatory) category in periods
        as set by freq. Default is 0.
        Importantly, for analyses with explanatory and dependent categories, the first
        takes the role of the explanatory and a positive lag means that the explanatory
        values will be deferred into the future.
    :param <List[str]> xcat_aggs: Exactly two aggregation methods. Default is 'mean' for
        both.
    :param <str> xcat1_chg: time series change applied to first category.
        Default is None. Change options are 'diff' (first difference) and 'pchg'
        (percentage change). The changes are calculated over the number of
        periods determined by `n_periods`.
    :param <int> n_periods: number of periods over which changes of the first category
        have been calculated. Default is 1.
    :param <int> fwin: forward moving average window of second category. Default is 1,
        i.e no average.
        Importantly, for analyses with explanatory and dependent categories, the second
        takes the role of the dependent and a forward window means that the dependent
        values average forward into the future.
    :param: <List[float]> xcat_trims: two-element list with maximum absolute values
        for the two respective categories. Observations with higher values will be
        trimmed, i.e. excluded from the analysis. Default is None for both.
        The trimming is applied after all transformations have been applied.

    """

    def __init__(self, df: pd.DataFrame, xcats: List[str], cids: List[str] = None,
                 val: str = 'value', start: str = None, end: str = None,
                 blacklist: dict = None, years = None, freq: str = 'M', lag: int = 0,
                 fwin: int = 1, xcat_aggs: List[str] = ('mean', 'mean'),
                 xcat1_chg: str = None, n_periods: int = 1,
                 xcat_trims: List[float] = [None, None]):

        """Constructs all attributes for the category relationship to be analyzed."""

        self.xcats = xcats
        self.cids = cids 
        self.val = val 
        self.freq = freq
        self.lag = lag
        self.years = years 
        self.aggs = xcat_aggs
        self.xcat1_chg = xcat1_chg
        self.n_periods = n_periods
        self.xcat_trims = xcat_trims

        assert self.freq in ['D', 'W', 'M', 'Q', 'A']
        assert {'cid', 'xcat', 'real_date', val}.issubset(set(df.columns))
        assert len(xcats) == 2, "Expects two fields."

        # Select the cross-sections available for both categories.
        shared_cids = CategoryRelations.intersection_cids(df, xcats, cids)
        df = categories_df(df, xcats, shared_cids, val, start=start,
                           end=end, freq=freq, blacklist=blacklist, years=years,
                           lag=lag, fwin=fwin, xcat_aggs=xcat_aggs)

        if xcat1_chg is not None:

            assert xcat1_chg in ['diff', 'pch']
            assert isinstance(n_periods, int)

            df = CategoryRelations.time_series(df, change=xcat1_chg,
                                               n_periods=n_periods,
                                               shared_cids=shared_cids,
                                               expln_var=xcats[0])

        if self.xcat_trims[0] != None:
            assert len(xcat_trims) == len(xcats), "Two values expected corresponding to " \
                                                  "the number of categories."
            types = [isinstance(elem, float) and elem >= 0.0 for elem in xcat_trims]
            assert all(types), "Expected two floating point values."

            df = CategoryRelations.outlier_trim(df, xcats, xcat_trims)

        self.df = df

    @classmethod
    def intersection_cids(cls, df, xcats, cids):
        """
        Returns list of common cids across categories.

        :return <List[str]>: usable: List of the common cross-sections across the two
            categories.
        """

        set_1 = set(df[df['xcat'] == xcats[0]]['cid'].unique())
        set_2 = set(df[df['xcat'] == xcats[1]]['cid'].unique())

        miss_1 = list(set(cids).difference(set_1))  # cids not available for 1st cat
        miss_2 = list(set(cids).difference(set_2))  # cids not available for 2nd cat

        if len(miss_1) > 0:
            print(f"{xcats[0]} misses: {sorted(miss_1)}.")
        if len(miss_2) > 0:
            print(f"{xcats[1]} misses: {sorted(miss_2)}.")

        usable = list(set_1.intersection(set_2).
                      intersection(set(cids)))  # 3 set intersection

        return usable

    @classmethod
    def time_series(cls, df: pd.DataFrame, change: str, n_periods: int,
                    shared_cids: List[str], expln_var: str):
        """
        Modifying the metric on the explanatory variable: the dataframe's default will be
        the raw value series, defined according to the frequency parameter, but allow for
        additional time-series metrics such as differencing or % change (pchg).

        :param <pd.DataFrame> df: multi-index DataFrame hosting the two categories: first
            column represents the explanatory variable; second column hosts the dependent
            variable. The dataframe's index is the real-date and cross-section.
        :param <str> change:
        :param <int> n_periods:
        :param <List[str]> shared_cids: shared cross-sections across the two categories
            and the received list.
        :param <str> expln_var: only the explanatory variable's data series will be
            changed from the raw value series to a difference or percentage change value.

        :return <pd.Dataframe>: df: returns the same multi-index dataframe but with an
            adjusted series inline with the 'change' parameter.
        """

        df_lists = []
        for c in shared_cids:
            temp_df = df.loc[c]

            explan_col = temp_df[expln_var].to_numpy()
            shift = np.empty(explan_col.size)
            shift[:] = np.nan
            shift[n_periods:] = explan_col[:-n_periods]

            if change == 'diff':
                temp_df[expln_var] -= shift
            else:
                diff = explan_col - shift
                temp_df[expln_var] = diff / shift

            temp_df['cid'] = c
            temp_df = temp_df.set_index('cid', append=True)
            df_lists.append(temp_df)

        df_ = pd.concat(df_lists)
        return df_.dropna(axis=0, how='any')

    @classmethod
    def outlier_trim(cls, df: pd.DataFrame, xcats: List[str], xcat_trims: List[float]):
        """
        Method used to trim any outliers from the dataset - inclusive of both categories.
        Outliers are classified as any datapoint whose absolute value exceeds the
        predefined value specified in the field self.xcat_trims. The values will be set
        to NaN, and subsequently excluded from any regression modelling or correlation
        coefficients.

        :param <pd.DataFrame> df: multi-index DataFrame hosting the two categories. The
            transformations, to each series, have already been applied.
        :param <List[str]> xcats: explanatory and dependent variable.
        :param <List[float]> xcat_trims:

        :return <pd.DataFrame> df: returns the same multi-index dataframe.
        """

        xcat_dict = dict(zip(xcats, xcat_trims))

        for k, v in xcat_dict.items():

            df[k] = np.where(np.abs(df[k]) < v, df[k], np.nan)

        df = df.dropna(axis=0, how='any')
        return df

    def corr_probability(self, coef_box):

        x = self.df[self.xcats[0]].to_numpy()
        y = self.df[self.xcats[1]].to_numpy()
        coeff, pval = stats.pearsonr(x, y)
        cpl = [np.round(coeff, 3), np.round(1 - pval, 3)]
        fields = ["Correlation\n coefficient", "Probability\n of significance"]
        data_table = plt.table(cellText=[cpl], colLabels=fields,
                               cellLoc='center', loc=coef_box)
        
        return data_table

    def reg_scatter(self, title: str = None, labels: bool = False,
                    size: Tuple[float] = (12, 8),
                    xlab: str = None, ylab: str = None, coef_box: str = None,
                    fit_reg: bool = True,
                    reg_ci: int = 95, reg_order: int = 1, reg_robust: bool = False):

        """
        Display scatterplot and regression line.

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
        :param <str> coef_box: gives location of box of correlation coefficient and
            probability. If None (default), no box is shown. Options are standard,
            i.e. 'upper left', 'lower right' and so forth.
        """
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize = size)
        sns.regplot(data=self.df, x=self.xcats[0], y=self.xcats[1],
                    ci=reg_ci, order=reg_order, robust=reg_robust, fit_reg=fit_reg,
                    scatter_kws={'s': 30, 'alpha': 0.5, 'color': 'lightgray'},
                    line_kws={'lw': 1})

        if coef_box is not None:
            data_table = self.corr_probability(coef_box)
            data_table.scale(0.4, 2.5)
            data_table.set_fontsize(12)

        if labels:
            assert self.freq in ['A', 'Q', 'M'], \
                'Labels only available for monthly or lower frequencies'
            df_labs = self.df.dropna().index.to_frame(index=False)
            if self.years is not None:
                ser_labs = df_labs['cid'] + ' ' + df_labs['real_date']
            elif self.freq == 'A':
                ser_labs = df_labs['cid'] + ' ' + df_labs['real_date'].dt.year.\
                    astype(str)
            elif self.freq == 'Q':
                ser_labs = df_labs['cid'] + ' ' + df_labs['real_date'].dt.year.\
                    astype(str) + 'Q' + df_labs['real_date'].dt.quarter.astype(str)
            elif self.freq == 'M':
                ser_labs = df_labs['cid'] + ' ' + df_labs['real_date'].dt.year.\
                    astype(str) + '-' + df_labs['real_date'].dt.month.astype(str)
            for i in range(self.df.shape[0]):
                plt.text(x=self.df[self.xcats[0]][i] + 0,
                         y=self.df[self.xcats[1]][i] + 0, s=ser_labs[i],
                         fontdict=dict(color='black', size=8))

        if title is None and (self.years is None):
            dates = self.df.index.get_level_values('real_date').to_series().\
                dt.strftime('%Y-%m-%d')
            title = f'{self.xcats[0]} and {self.xcats[1]} ' \
                    f'from {dates.min()} to {dates.max()}'
        elif title is None:
            title = f'{self.xcats[0]} and {self.xcats[1]}'

        ax.set_title(title, fontsize = 14)
        if xlab is not None:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)
            
        plt.show()

    def jointplot(self, kind, fit_reg: bool = True, title: str = None,
                  height: float = 6, xlab: str = None, ylab: str = None):

        """
        Display jointplot of chosen type, based on seaborn.jointplot().
        The plot will always be square.

        :param <str> kind: determines type of relational plot inside the joint plot.
            This must be one of one of 'scatter', 'kde', 'hist', or 'hex'.
        :param <bool> fit_reg: if True (default) adds a regression line.
        :param <str> title: title. If None (default) informative title is applied.
        :param <float> height: height and implicit size of figure. Default is 6.
        :param <str> xlab: x-axis label. Default is no label.
        :param <str> ylab: y-axis label. Default is no label.

        """
        assert kind in ['scatter', 'kde', 'hist', 'hex']

        sns.set_theme(style='whitegrid')
        if kind == 'hex':
            sns.set_theme(style='white')

        fg = sns.jointplot(data=self.df,  x=self.xcats[0], y=self.xcats[1],
                           kind=kind, height=height, color='steelblue')
        
        if fit_reg:
            fg.plot_joint(sns.regplot, scatter=False, ci=0.95, color='black',
                          line_kws={'lw': 1, 'linestyle': '--'})

        xlab = xlab if xlab is not None else ''
        ylab = ylab if ylab is not None else ''
        fg.set_axis_labels(xlab, ylab)

        if title is None and (self.years is None):
            dates = self.df.index.get_level_values('real_date').to_series().\
                dt.strftime('%Y-%m-%d')
            title = f'{self.xcats[0]} and {self.xcats[1]} ' \
                    f'from {dates.min()} to {dates.max()}'
        elif title is None:
            title = f'{self.xcats[0]} and {self.xcats[1]}'

        fg.fig.suptitle(title, y=1.02)

        plt.show()

    def ols_table(self):
        """
        Print statsmodel OLS table of pooled regression.

        """

        x, y = self.df.dropna().iloc[:, 0], self.df.dropna().iloc[:, 1]
        x_fit = sm.add_constant(x)
        fit_results = sm.OLS(y, x_fit).fit()
        print(fit_results.summary())


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD', 'USD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index=cids,
                           columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]
    df_cids.loc['USD'] = ['2003-01-01', '2020-12-31', -0.1, 2]

    cols = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef']
    df_xcats = pd.DataFrame(index=xcats, columns=cols)
    df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    filt1 = (dfd['xcat'] == 'GROWTH') & (dfd['cid'] == 'AUD')  # all AUD GROWTH locations
    filt2 = (dfd['xcat'] == 'INFL') & (dfd['cid'] == 'NZD')  # all NZD INFL locations
    dfdx = dfd[~(filt1 | filt2)]  # reduced dataframe

    cidx = ['AUD', 'CAD', 'GBP', 'USD']

    cr = CategoryRelations(dfdx, xcats=['GROWTH', 'INFL'],
                           cids=cidx, xcat_aggs=['mean', 'mean'],
                           start='2005-01-01', blacklist=black,
                           years=3)

    cr = CategoryRelations(dfdx, xcats=['GROWTH', 'INFL'], cids=cidx, freq='M',
                           xcat_aggs=['mean', 'mean'], lag=1,
                           start='2000-01-01', years=None, blacklist=black,
                           xcat1_chg=None, xcat_trims=[2.75, 2.5])

    cr.reg_scatter(labels=False, coef_box='upper left')
    cr.jointplot(kind='hist', xlab='growth', ylab='inflation', height=5)

    cr = CategoryRelations(dfd, xcats=['GROWTH', 'INFL'], cids=cids, freq='M',
                           xcat_aggs=['mean', 'mean'],
                           start='2000-01-01', years=3, blacklist=black)

    cr.reg_scatter(labels=False, coef_box='lower right',
                   title='Growth and inflation', xlab='Growth', ylab='Inflation')
    cr.jointplot(kind='hist', xlab='growth', ylab='inflation', height=5)
