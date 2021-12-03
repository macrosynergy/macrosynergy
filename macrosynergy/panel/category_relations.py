import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import List, Union, Tuple
from sklearn import datasets, linear_model
from scipy import stats
import statsmodels.api as sm
import time

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import categories_df


class CategoryRelations:
    """Class for analyzing and visualizing two categories across a panel
    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: Exactly two extended categories to be checked on.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the
        dataframe.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the data frame.
    :param <int> years: Number of years over which data are aggregate. Supersedes freq
        and does not allow lags, Default is None, meaning no multi-year aggregation.
    :param <str> val: name of column that contains the values of interest. Default is
        'value'.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
    :param <int> lag: Lag (delay of arrival) of second category in periods as set by
        freq. Default is 0.
    :param <int> fwin: Forward moving average window of first category. Default is 1, i.e
        no average.
    :param <List[str]> xcat_aggs: Exactly two aggregation methods. Default is 'mean' for
        both.
    """

    def __init__(self, df: pd.DataFrame, xcats: List[str], cids: List[str] = None, val: str = 'value',
                 start: str = None, end: str = None, blacklist: dict = None, years = None,
                 freq: str = 'M', lag: int = 0, fwin: int = 1, xcat_aggs: List[str] = ('mean', 'mean')):

        """Constructs all attributes for the category relationship to be analyzed"""

        self.xcats = xcats
        self.cids = cids 
        self.val = val 
        self.freq = freq
        self.lag = lag
        self.years = years 
        self.aggs = xcat_aggs 

        assert self.freq in ['D', 'W', 'M', 'Q', 'A']
        assert {'cid', 'xcat', 'real_date', val}.issubset(set(df.columns))
        assert len(xcats) == 2, "Expects two fields."

        shared_cids = self.intersection_cids(df)
        self.df = categories_df(df, xcats, shared_cids, val, start=start,
                                end=end, freq=freq, blacklist=blacklist, years=years,
                                lag=lag, fwin=fwin, xcat_aggs=xcat_aggs)

    def intersection_cids(self, df):
        unique_cat = iter(set(df['xcat'].to_numpy()))
        cat = next(unique_cat)

        df_cat = df[df['xcat'] == cat]
        cids_ = set(df_cat['cid'].to_numpy())

        for res_cat in unique_cat:
            df_temp = df[df['xcat'] == res_cat]
            cross_sections = set(df_temp['cid'].to_numpy())
            cids_ = cids_.intersection(list(df_temp['cid'].to_numpy()))
            print(f"Missing cross-sections on {res_cat} are: {cids_ - cross_sections}.")

        return list(cids_)

    def corr_probability(self, coef_box):

        x = self.df[self.xcats[0]].to_numpy()
        y = self.df[self.xcats[1]].to_numpy()
        coeff, pval = stats.pearsonr(x, y)
        cpl = [np.round(coeff, 3), np.round(1 - pval, 3)]
        fields = ["Correlation\n coefficient", "Probability\n of significance"]
        data_table = plt.table(cellText = [cpl], colLabels = fields,
                               cellLoc = 'center', loc = coef_box)
        
        return data_table

    def reg_scatter(self, title: str = None, labels: bool = False, size: Tuple[float] = (12, 8),
                    xlab: str = None, ylab: str = None, coef_box: str = None, fit_reg: bool = True,
                    reg_ci: int = 95, reg_order: int = 1, reg_robust: bool = False):

        """Display scatterplot and regression line
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
            assert self.freq in ['A', 'Q', 'M'], 'Labels are only possible for monthly or lower frequencies'
            df_labs = self.df.dropna().index.to_frame(index=False)
            if self.years is not None:
                ser_labs = df_labs['cid'] + ' ' + df_labs['real_date']
            elif self.freq == 'A':
                ser_labs = df_labs['cid'] + ' ' + df_labs['real_date'].dt.year.astype(str)
            elif self.freq == 'Q':
                ser_labs = df_labs['cid'] + ' ' + df_labs['real_date'].dt.year.astype(str) + 'Q' \
                           + df_labs['real_date'].dt.quarter.astype(str)
            elif self.freq == 'M':
                ser_labs = df_labs['cid'] + ' ' + df_labs['real_date'].dt.year.astype(str) + '-' \
                           + df_labs['real_date'].dt.month.astype(str)
            for i in range(self.df.shape[0]):
                plt.text(x=self.df[self.xcats[0]][i] + 0, y=self.df[self.xcats[1]][i] + 0, s=ser_labs[i],
                         fontdict=dict(color='black', size=8))

        if title is None and (self.years is None):
            dates = self.df.index.get_level_values('real_date').to_series().dt.strftime('%Y-%m-%d')
            title = f'{self.xcats[0]} and {self.xcats[1]} from {dates.min()} to {dates.max()}'
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

        """Display jointplot of chosen type, based on seaborn.jointplot().
           The plot will always be square.

        :param <str> kind: determines type of relational plot inside the joint plot.
            This must be one of one of 'scatter', 'kde', 'hist', or 'hex'.
        :param <bool> fit_reg: if True (default) adds a regression line.
        :param <str> title: title of plot. If None (default) an informative title is applied.
        :param <float> height: height and implicit size of figure. Default is 6.
        :param <str> xlab: x-axis label. Default is no label.
        :param <str> ylab: y-axis label. Default is no label.

        """
        assert kind in ['scatter', 'kde', 'hist', 'hex']

        sns.set_theme(style = 'whitegrid')
        if kind == 'hex':
            sns.set_theme(style = 'white')

        fg = sns.jointplot(data=self.df,  x=self.xcats[0], y=self.xcats[1],
                           kind=kind, height=height, color='steelblue')
        
        if fit_reg:
            fg.plot_joint(sns.regplot, scatter=False, ci=0.95, color='black',
                          line_kws={'lw': 1, 'linestyle': '--'})

        xlab = xlab if xlab is not None else ''
        ylab = ylab if ylab is not None else ''
        fg.set_axis_labels(xlab, ylab)

        if title is None and (self.years is None):
            dates = self.df.index.get_level_values('real_date').to_series().dt.strftime('%Y-%m-%d')
            title = f'{self.xcats[0]} and {self.xcats[1]} from {dates.min()} to {dates.max()}'
        elif title is None:
            title = f'{self.xcats[0]} and {self.xcats[1]}'

        fg.fig.suptitle(title, y = 1.02)

        plt.show()

    def ols_table(self):
        """Print statsmodel OLS table of pooled regression"""

        x, y = self.df.dropna().iloc[:, 0], self.df.dropna().iloc[:, 1]
        x_fit = sm.add_constant(x)
        fit_results = sm.OLS(y, x_fit).fit()
        print(fit_results.summary())


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index = cids, columns = ['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index = xcats, columns = ['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)
    black = {'AUD': ['2000-01-01', '2003-12-31'], 'GBP': ['2018-01-01', '2100-01-01']}

    cr = CategoryRelations(dfd, xcats=['GROWTH', 'INFL'], cids=cids, freq='M', xcat_aggs=['mean', 'mean'], lag=1,
                           start='2000-01-01', years=None, blacklist=black)
    cr.reg_scatter(labels=False, coef_box='upper left')

    cr = CategoryRelations(dfd, xcats=['GROWTH', 'INFL'], cids=cids, freq='M', xcat_aggs=['mean', 'mean'],
                           start='2000-01-01', years=3, blacklist=black)
    cr.reg_scatter(labels=False, coef_box='lower right',
                   title='Growth and inflation', xlab='Growth', ylab='Inflation')
    cr.jointplot(kind='hist', xlab='growth', ylab='inflation', height=5)
