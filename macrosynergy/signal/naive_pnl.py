# NaivePnL class
# [1] uses categories_df in same way as SignalReturnRelations
# [2] allows multiple signals
# [3] allows simple signal transformations (z-scoring, trimming, digital, vol-weight)
# [4] allows periodicity of signal to be set
# [5] allows ex-post vol-scaling of Pnl
# [6] allows equally weighted long-only benchmark
# [7] allows to set slippage in days
#
# Produces daily-frequency statistics:
# [1] chart of PnLs
# [2] table of key performance statistics
# Annualized return, ASD, Sharpe, Sortino, Calmar, positive years ratio
# [3] chart of cross_section PnLs
# [3] table of cross-section contributions
#
# Implementation
# [1] at initiation creates df of basic transformations
# [2] pnl_chart() method
# [3] pnl_table() method
# [4] cs_pnl_charts() method
# [5] cs_pnl_table method

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as skm
from scipy import stats

from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


class NaivePnL:

    """Estimates and analyses naive illustrative PnLs with limited signal options and disregarding transaction costs

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value.
    :param <str> ret: return category.
    :param <List[str]> sigs: signal categories.
    :param <List[str]> cids: cross sections to be considered. Default is all in the data frame.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.

    """
    def __init__(self, df: pd.DataFrame, ret: str, sigs: str, cids: List[str] = None,
                 start: str = None, end: str = None, blacklist: dict = None):

        self.ret = ret
        self.sigs = sigs
        xcats = [ret] + sigs
        self.df, self.xcats, self.cids = reduce_df(df, xcats, cids, start, end, blacklist, out_all=True)
        self.pnl_names = []  # list for PnL names

    def make_pnl(self, sig: str, sig_op: str = 'zn_score_pan', pnl_name: str = None,
                 rebal_freq: str = 'daily', rebal_slip=0, vol_scale: float = None):
        """Calculate daily PnL and add to the class instance

        :param <str> sig: name of signal that is the basis for positioning. It is recorded at the end of trading day.
        :param <str> sig_op: signal transformation options; must be one of 'zn_score_pan', 'zn_score_cs', 'binary'.
            Default is 'zn_score_pan', transforms signals to z-scores around their zero value based on the whole panel.
            Option 'zn_score_cs' transforms signals to z-scores around their zero value based on cross-section alone.
            Option 'binary' transforms signals into equal long/shorts (1/-1) across all sections.
        :param <str> pnl_name: name of the Pnl (overwrites existing). Default is none, i.e. a default name is given.
        :param <str> rebal_freq: rebalancing frequency; must be one of 'daily' (default), 'weekly' or 'monthly'.
        :param <str> rebal_slip: rebalancing slippage in days.  Default is 1, which means that it takes one day
            to rebalance the position and that the new position produces PnL from the second day after the signal.
        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given. Default is none.

        """

        assert sig in self.sigs
        assert sig_op in ['zn_score_pan', 'zn_score_cs', 'binary']
        assert rebal_freq in ['daily', 'weekly', 'monthly']

        dfx = self.df[self.df['xcat'].isin([self.ret, sig])]
        dfw = dfx.pivot(index=['cid', 'real_date'], columns='xcat', values='value')

        if sig_op == 'zn_score_pan':
            sda = dfw[sig].abs().mean()
            dfw['psig'] = dfw[sig] / sda
        elif sig_op == 'zn_score_cs':
            zn_score = lambda x: x / np.nanmean(np.abs(x))
            dfw['psig'] = dfw[sig].groupby(level=0).apply(zn_score)
        elif sig_op == 'binary':
            dfw['psig'] = np.sign(dfw[sig])
        dfw['psig'] = dfw['psig'].groupby(level=0).shift(1)  # lag explanatory to arrive one period late
        dfw.reset_index(inplace=True)

        if rebal_freq != 'daily':
            dfw['year'] = dfw['real_date'].dt.year
            if rebal_freq == 'monthly':
                dfw['month'] = dfw['real_date'].dt.month
                rebal_dates = dfw.groupby(['cid', 'year', 'month'])['real_date'].min()  # first trading day per month
            if rebal_freq == 'weekly':
                dfw['week'] = dfw['real_date'].dt.week
                rebal_dates = dfw.groupby(['cid', 'year', 'week'])['real_date'].min()  # first trading day per week
            dfw['sig'] = np.nan
            dfw.loc[dfw['real_date'].isin(rebal_dates), 'sig'] = dfw.loc[dfw['real_date'].isin(rebal_dates), 'psig']
            dfw['sig'] = dfw['sig'].fillna(method='ffill').shift(rebal_slip)
        dfw['value'] = dfw[self.ret] * dfw['sig']

        df_pnl = dfw.loc[:, ['cid', 'real_date', 'value']]  # cross-section PnLs

        df_pnl_all = df_pnl.groupby(['real_date']).sum()  # global PnL as sum of cross-section PnLs
        df_pnl_all = df_pnl_all[df_pnl_all['value'].cumsum() != 0]  # trim zeros from early part of df
        df_pnl_all['cid'] = 'ALL'
        df_pnl_all = df_pnl_all.reset_index()[df_pnl.columns]  # set columns equal to df_pnl...
        df_pnl = df_pnl.append(df_pnl_all)  #... and append

        if vol_scale is not None:
            leverage = vol_scale * (df_pnl_all['value'].std() * np.sqrt(252))**(-1)
            df_pnl['value'] = df_pnl['value'] * leverage

        pnn = ('PNL_' + sig) if pnl_name is None else pnl_name  # set PnL name
        df_pnl['xcat'] = pnn
        if pnn in self.pnl_names:
            self.df = self.df[~(self.df['xcat'] == pnn)]  # remove PnL with same name from main df
        else:
            self.pnl_names = self.pnl_names + pnn

        self.df = self.df.append(df_pnl[self.df.columns]).reset_index(drop=True)  # add new PnL from main dataframe

    def plot_pnls(self, pnl_names: str = None):
        """Plot line charts of global PnLs"""

        pass

    def evaluate_pnls(self, pnl_names: str = None):

        pass

    def plot_cs_pnls(self, pnl_names: str = None):

        pass

    def pnl_names(self):
        """Print list of names of available PnLs in the class instance"""

        print(self.pnl_names)

    def pnl_df(self, pnl_names: list[str] = None,  cs: bool = False):
        """Return data frame with PnLs

        :param <list[str]> pnl_names: list of names of PnLs to be returned. Default is all.
        :param <bool> cs: inclusion of cross section PnLs. Default is False.

        :return custom data frame with PnLs
        """
        selected_pnls = pnl_names if pnl_names is not None else self.pnl_names

        filter_1 = self.df['xcat'].isin(selected_pnls)
        filter_2 = self.df['cid'] == 'ALL' if cs else True

        return self.df[filter_1 & filter_2]




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

    black = {'AUD': ['2006-01-01', '2015-12-31'], 'GBP': ['2012-01-01', '2100-01-01']}

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    npo = NaivePnL(dfd, ret='XR', sigs=['CRY', 'GROWTH', 'INFL'], cids=cids, start='2000-01-01')
    npo.make_pnl('CRY', sig_op='binary', rebal_freq='monthly', vol_scale=10, rebal_slip=1, pnl_name='PNL_CRY_DIG')