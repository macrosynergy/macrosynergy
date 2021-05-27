import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df


def categories_df(df: pd.DataFrame, xcats: List[str], cids: List[str] = None, val: str = 'value',
                  start: str = None, end: str = None, blacklist: dict = None, years: int = None,
                  freq: str = 'M', lag: int = 0, xcat_aggs: List[str] = ('mean', 'mean')):

    """Create custom two-categories dataframe suitable for analysis

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: Exactly two extended categories to be checked on.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the dataframe.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.
    :param <int> years: Number of years over which data are aggregate. Supersedes freq and does not allow lags,
        Default is None, meaning no multi-year aggregation.
    :param <str> val: name of column that contains the values of interest. Default is 'value'.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
    :param <int> lag: Lag (delay of arrival) of second category in periods as set by freq. Default is 0.
    :param <List[str]> xcat_aggs: Exactly two aggregation methods. Default is 'mean' for both.

    :return custom dataframe with columns for the two categories indexed by cross sections and periods.

    """

    assert freq in ['D', 'W', 'M', 'Q', 'A']
    assert not (years is not None) & (lag != 0), 'Lags cannot be applied to year groups'

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, blacklist, out_all=True)

    col_names = ['cid', 'xcat', 'real_date', val]
    dfc = pd.DataFrame(columns=col_names)

    if years is None:
        for i in range(2):
            dfw = df[df['xcat'] == xcats[i]].pivot(index='real_date', columns='cid', values=val)
            dfw = dfw.resample(freq).agg(xcat_aggs[i]).shift(lag).reset_index()
            dfx = pd.melt(dfw, id_vars=['real_date'], value_vars=cids, value_name=val)
            dfx['xcat'] = xcats[i]
            dfc = dfc.append(dfx[col_names])
    else:

        s_year = pd.to_datetime(start).year
        e_year = df['real_date'].max().year + 1

        s_years = range(s_year, e_year, years)
        year_groups = {}
        for y in s_years:
            ey = y + years - 1 if (y + years - 1) <= e_year else 'now'
            y_key = f'{y} - {ey}'
            y_value = [i for i in range(y, y + years)]
            year_groups[y_key] = y_value

        def translate(year):
            return np.array(list(year_groups.keys()))[[year in l for l in list(year_groups.values())]][0]

        df['custom_date'] = df['real_date'].dt.year.apply(translate)
        for i in range(2):
            dfx = df[df['xcat'] == xcats[i]]
            dfx = dfx.groupby(['xcat', 'cid', 'custom_date']).agg(xcat_aggs[i]).reset_index()
            dfx = dfx.rename(columns={"custom_date": "real_date"})
            dfc = dfc.append(dfx[col_names])

    return dfc.pivot(index=('cid', 'real_date'), columns='xcat', values=val).dropna()


class CategoryRelations:

    """Class for analyzing and visualizing two categories across a panel

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: Exactly two extended categories to be checked on.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the data frame.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.
    :param <int> years: Number of years over which data are aggregate. Supersedes freq and does not allow lags,
        Default is None, meaning no multi-year aggregation.
    :param <str> val: name of column that contains the values of interest. Default is 'value'.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
    :param <int> lag: Lag (delay of arrival) of second category in periods as set by freq. Default is 0.
    :param <List[str]> xcat_aggs: Exactly two aggregation methods. Default is 'mean' for both.

    :return custom data frame with columns for the two categories indexed by cross sections and periods.

    """

    def __init__(self, df: pd.DataFrame, xcats: List[str], cids: List[str] = None, val: str = 'value',
                 start: str = None, end: str = None, blacklist: dict = None, years=None,
                 freq: str = 'M', lag: int = 0, xcat_aggs: List[str] = ('mean', 'mean')):

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

        self.df = categories_df(df, xcats, cids, val, start=start, end=end, freq=freq, blacklist=blacklist,
                                years=years, lag=lag, xcat_aggs=xcat_aggs)

    def reg_scatter(self, title: str = None, labels: bool = False,
                    size: Tuple[float] = (12, 8), xlab: str = None, ylab: str = None,
                    fit_reg: bool = True, reg_ci: int = 95, reg_order: int = 1, reg_robust: bool = False):

        """Display scatterplot and regression line

        :param <str> title: title of plot. If None (default) an informative title is applied.
        :param <bool> labels: assign a cross-section/period label to each dot. Default is False.
        :param <Tuple[float]> size: width and height of the figure
        :param <str> xlab: x-axis label. Default is no label.
        :param <str> ylab: y-axis label. Default is no label.
        :param <bool> fit_reg: if True (default) adds a regression line.
        :param <int> reg_ci: size of the confidence interval for the regression estimate. Default is 95. Can be None.
        :param <int> reg_order: order of the regression equation. Default is 1 (linear).
        :param <bool> reg_robust: if this will de-weight outliers, which is computationally expensive. Default is False.

        """

        fig, ax = plt.subplots(figsize=size)  # set up figure
        sns.regplot(data=self.df, x=self.xcats[0], y=self.xcats[1],
                    ci=reg_ci, order=reg_order, robust=reg_robust, fit_reg=fit_reg,
                    scatter_kws={'s': 30, 'alpha': 0.5, 'color': 'lightgray'}, line_kws={'lw': 1})

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

        ax.set_title(title, fontsize=14)
        if xlab is not None:
            ax.set_xlabel(xlab)
        if ylab is not None:
            ax.set_ylabel(ylab)
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
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD',] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD',] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP',] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD',] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR',] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY',] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH',] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL',] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfc = categories_df(dfd, xcats=['GROWTH', 'CRY'], cids=cids, freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                        start='2000-01-01', years=5)
    cr = CategoryRelations(dfd, xcats=['GROWTH', 'INFL'], cids=cids, freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                           start='1999-01-01', years=10)
    cr.reg_scatter(labels=True)

    black = {'AUD': ['2000-01-01', '2003-12-31'], 'NZD': ['2018-01-01', '2100-01-01']}
    dfc = categories_df(dfd, xcats=['GROWTH', 'CRY'], cids=cids, freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                        start='2000-01-01', blacklist=black)

    black = {'AUD_1': ['2000-01-01', '2009-12-31'], 'AUD_2': ['2018-01-01', '2100-01-01']}
    dfc = categories_df(dfd, xcats=['GROWTH', 'CRY'], cids=cids, freq='M', lag=0, xcat_aggs=['mean', 'mean'],
                        start='2000-01-01', blacklist=black, years=10)
    cr = CategoryRelations(dfd, xcats=['GROWTH', 'INFL'], cids=cids, freq='M', xcat_aggs=['mean', 'mean'],
                           start='2000-01-01', years=10, blacklist=black)
    cr.reg_scatter(labels=True)
    cr.ols_table()

    dfc.info()
