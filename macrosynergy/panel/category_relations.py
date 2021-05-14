import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df


def categories_df(df: pd.DataFrame, xcats: List[str], cids: List[str] = None, val: str = 'value',
                  start: str = None, end: str = None,
                  freq: str = 'M', lag: int = 0, xcat_aggs: List[str] = ('mean', 'mean')):
    """Create custom two-categories dataframe suitable for analysis"""

    assert freq in ['W', 'M', 'Q', 'A']

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

    col_names = ['cid', 'xcat', 'real_date', val]
    dfc = pd.DataFrame(columns=col_names)
    for i in range(2):
        dfw = df[df['xcat'] == xcats[i]].pivot(index='real_date', columns='cid', values=val)
        dfw = dfw.resample(freq).agg(xcat_aggs[i]).shift(lag).reset_index()
        dfx = pd.melt(dfw, id_vars=['real_date'], value_vars=cids, value_name=val)
        dfx['xcat'] = xcats[i]
        dfc = dfc.append(dfx[col_names])

    return dfc.pivot(index=('cid', 'real_date'), columns='xcat', values=val).dropna()


class categories_relations:
    """Analyze and visualize two categories across a panel"""

    def __init__(self, df: pd.DataFrame, xcats: List[str], cids: List[str] = None, val: str = 'value',
                 start: str = None, end: str = None,
                 freq: str = 'M', lag: int = 0, xcat_aggs: List[str] = ('mean', 'mean')):

        self.xcats = xcats
        self.cids = cids
        self.val = val
        self.freq = freq
        self.lag = lag
        self.aggs = xcat_aggs

        assert self.freq in ['W', 'M', 'Q', 'A']
        assert {'cid', 'xcat', 'real_date', val}.issubset(set(df.columns))

        self.df = categories_df(df, xcats, cids, val, start=start, end=end, freq=freq, lag=lag, xcat_aggs=xcat_aggs)

    def reg_scatter(self, title: str = None, size: Tuple[float] = (12, 8),
                    fit_reg: bool = True, reg_ci: int = 95, reg_order: int = 1, reg_robust: bool = False,
                    xlab = None, ylab = None, labels: bool = False):

        """Display scatterplot and regression line"""

        fig, ax = plt.subplots(figsize=size)  # set up figure
        sns.regplot(data=self.df, x=self.xcats[0], y=self.xcats[1],
                    ci=reg_ci, order=reg_order, robust=reg_robust, fit_reg=fit_reg,
                    scatter_kws={'s': 30, 'alpha': 0.5, 'color': 'lightgray'}, line_kws={'lw': 1})

        if labels:
            assert self.freq in ['A', 'Q', 'M'], 'Labels are only possible for monthly or lower frequencies'
            df_labs = self.df.dropna().index.to_frame(index=False)
            if self.freq == 'A':
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

        if title is None:
            dates = dfc.index.get_level_values('real_date').to_series().dt.strftime('%Y-%m-%d')
            title = f'{self.xcats[0]} and {self.xcats[1]} from {dates.min()} to {dates.max()} '

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
    df_cids.loc['AUD',] = ['2010-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD',] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP',] = ['2012-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD',] = ['2012-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR',] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY',] = ['2011-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH',] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL',] = ['2011-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfc = categories_df(dfd, xcats=['GROWTH', 'CRY'], cids=cids, freq='M', lag=0, xcat_aggs=['mean', 'mean'])
    cr = categories_relations(dfd, xcats=['GROWTH', 'CRY'], cids=cids, freq='M', lag=0, xcat_aggs=['last', 'last'])

    cr.reg_scatter(fit_reg=True, reg_order=1)
    # cr.reg_scatter(labels=True, title='Growth and FX carry', xlab='Growth', ylab='FX carry')
    cr.ols_table()
