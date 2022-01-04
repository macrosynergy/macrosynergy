import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df


class NaivePnL:

    """Computes and collects illustrative PnLs with limited signal options and
    disregarding transaction costs

    :param <pd.Dataframe> df: standardized data frame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <str> ret: return category.
    :param <List[str]> sigs: signal categories.
    :param <List[str]> cids: cross sections to be considered. Default is all in the
        dataframe.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded
        from the dataframe.

    """
    def __init__(self, df: pd.DataFrame, ret: str, sigs: List[str],
                 cids: List[str] = None,
                 start: str = None, end: str = None,
                 blacklist: dict = None):

        self.ret = ret
        self.sigs = sigs
        xcats = [ret] + sigs
        cols = ['cid', 'xcat', 'real_date', 'value']
        self.df, self.xcats, self.cids = reduce_df(df[cols], xcats, cids, start, end,
                                                   blacklist, out_all=True)
        self.df['real_date'] = pd.to_datetime(self.df['real_date'])
        self.pnl_names = []  # list for PnL names
        self.black = blacklist

    def make_pnl(self, sig: str, sig_op: str = 'zn_score_pan',  pnl_name: str = None,
                 rebal_freq: str = 'daily', rebal_slip = 0, vol_scale: float = None,
                 min_obs: int = 252, iis: bool = True,
                 neutral: str = 'zero', thresh: float = None):

        # Todo: implement the four 'pass through arguments to make_zn_score()

        """Calculate daily PnL and add to the main dataframe of the class instance

        :param <str> sig: name of signal that is the basis for positioning. The signal
            is assumed to be recorded at the end of the day prior to position taking.
        :param <str> sig_op: signal transformation options; must be one of
            'zn_score_pan', 'zn_score_cs', or 'binary'.
            Default 'zn_score_pan' transforms raw signals into z-scores around zero value
            based on the whole panel.
            Option 'zn_score_cs' transforms signals to z-scores around zero based on
            cross-section alone.
            Option 'binary' transforms signals into uniform long/shorts (1/-1) across all
            sections.
            N.B.: zn-score here means standardized score with zero being the natural
            neutral level and standardization through division by mean absolute value.
        :param <str> pnl_name: name of the PnL to be generated and stored.
            Default is none, i.e. a default name is given.
            Previously calculated PnLs in the class will be overwritten. This means that
            if a set of PnLs is to be compared they require custom names.
        :param <str> rebal_freq: rebalancing frequency for positions according to signal
            must be one of 'daily' (default), 'weekly' or 'monthly'.
        :param <str> rebal_slip: rebalancing slippage in days.  Default is 1, which means
            that it takes one day to rebalance the position and that the new positions
            produces PnL from the second day after the signal has been recorded.
        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given.
            This for comparative visualization and not out-of-sample. Default is none.
        :param <int> min_obs: the minimum number of observations required to calculate
            zn_scores. Default is 252.
            # Todo: implement in function
        :param <bool> iis: if True (default) zn-scores are also calculated for the initial
            sample period defined by min-obs, on an in-sample basis, to avoid losing history.
            # Todo: implement in function
        :param <str> neutral: method to determine neutral level. Default is 'zero'.
            Alternatives are 'mean' and "median".
            # Todo: implement in function
        :param <float> thresh: threshold value beyond which scores are winsorized,
            i.e. contained at that threshold. Therefore, the threshold is the maximum absolute
            score value that the function is allowed to produce. The minimum threshold is 1
            standard deviation.
            # Todo: implement in function

        """

        assert sig in self.sigs
        assert sig_op in ['zn_score_pan', 'zn_score_cs', 'binary']
        assert rebal_freq in ['daily', 'weekly', 'monthly']

        dfx = self.df[self.df['xcat'].isin([self.ret, sig])]
        dfw = dfx.pivot(index=['cid', 'real_date'], columns='xcat', values='value')

        if sig_op == 'zn_score_pan':
            # Todo: below is in-sample; use make_zn_score() for oos calculation
            # Todo: pass through min_obs, iss, neutral, thresh
            sda = dfw[sig].abs().mean()
            dfw['psig'] = dfw[sig] / sda
        elif sig_op == 'zn_score_cs':  # zn-score based on
            # Todo: below is in-sample; use make_zn_score() for oos calculation
            # Todo: pass through min_obs, iss, neutral, thresh
            zn_score = lambda x: x / np.nanmean(np.abs(x))
            dfw['psig'] = dfw[sig].groupby(level=0).apply(zn_score)
        elif sig_op == 'binary':
            dfw['psig'] = np.sign(dfw[sig])

        # Signal for the following day explains the lag mechanism.
        dfw['psig'] = dfw['psig'].groupby(level=0).shift(1)  # lag explanatory 1 period
        dfw.reset_index(inplace=True)

        if rebal_freq != 'daily':
            dfw['year'] = dfw['real_date'].dt.year
            if rebal_freq == 'monthly':
                dfw['month'] = dfw['real_date'].dt.month
                rebal_dates = dfw.groupby(['cid', 'year', 'month'])['real_date'].\
                    min()  # rebalancing days are first of month
            if rebal_freq == 'weekly':
                dfw['week'] = dfw['real_date'].dt.week
                rebal_dates = dfw.groupby(['cid', 'year', 'week'])['real_date'].\
                    min()  # rebalancing days are first of week
            dfw['sig'] = np.nan
            dfw.loc[dfw['real_date'].isin(rebal_dates), 'sig'] = \
                dfw.loc[dfw['real_date'].isin(rebal_dates), 'psig']
            dfw['sig'] = dfw['sig'].fillna(method='ffill').shift(rebal_slip)
        dfw['value'] = dfw[self.ret] * dfw['sig']

        df_pnl = dfw.loc[:, ['cid', 'real_date', 'value']]  # cross-section PnLs

        df_pnl_all = df_pnl.groupby(['real_date']).sum()  # global PnL as sum
        df_pnl_all = df_pnl_all[df_pnl_all['value'].cumsum() != 0]  # trim early zeros
        df_pnl_all['cid'] = 'ALL'
        df_pnl_all = df_pnl_all.reset_index()[df_pnl.columns]  # columns as in df_pnl...
        df_pnl = df_pnl.append(df_pnl_all)  #... and append

        if vol_scale is not None:
            leverage = vol_scale * (df_pnl_all['value'].std() * np.sqrt(261))**(-1)
            df_pnl['value'] = df_pnl['value'] * leverage

        pnn = ('PNL_' + sig) if pnl_name is None else pnl_name  # set PnL name
        df_pnl['xcat'] = pnn
        if pnn in self.pnl_names:
            self.df = self.df[~(self.df['xcat'] == pnn)]  # remove any PnL with same name
        else:
            self.pnl_names = self.pnl_names + [pnn]

        self.df = self.df.append(df_pnl[self.df.columns]).reset_index(drop=True)

    def plot_pnls(self, pnl_cats: List[str], pnl_cids: List[str] = ['ALL'],
                  start: str = None, end: str = None, figsize: Tuple = (10, 6)):

        """Plot line chart of cumulative PnLs, single PnL, multiple PnL types per
        cross section,  or mutiple cross sections per PnL type.

        :param <List[str]> pnl_cats: list of PnL categories that should be plotted.
        :param <List[str]> pnl_cids: list of cross sections to be plotted;
            default is 'ALL' (global PnL).
            Note: one can only have multiple PnL categories or multiple cross sections,
            not both.
        :param <str> start: start date in ISO format.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.
        :param <Tuple> figsize: tuple of plot width and height. Default is (10,6).
        """

        if pnl_cats is None:
            pnl_cats = self.pnl_names

        assert (len(pnl_cats) == 1) | (len(pnl_cids) == 1)

        dfx = reduce_df(self.df, pnl_cats, pnl_cids, start, end, self.black,
                        out_all=False)

        sns.set_theme(style='whitegrid', palette='colorblind',
                      rc={'figure.figsize': figsize})

        if len(pnl_cids) == 1:
            dfx['cum_value'] = dfx.groupby('xcat').cumsum()
            ax = sns.lineplot(data=dfx, x='real_date', y='cum_value', hue='xcat',
                              estimator=None, lw=1)
            leg = ax.axes.get_legend()
            if len(pnl_cats) > 1:
                leg.set_title('PnL categories for ' + pnl_cids[0])
            else:
                leg.set_title('PnL category for ' + pnl_cids[0])
        else:
            dfx['cum_value'] = dfx.groupby('cid').cumsum()
            ax = sns.lineplot(data=dfx, x='real_date', y='cum_value', hue='cid',
                              estimator=None, lw=1)
            leg = ax.axes.get_legend()
            leg.set_title('Cross sections')

        plt.title('Cumulative naive PnL', fontsize=16)
        plt.xlabel('')
        plt.ylabel('% of risk capital, no compounding')
        plt.axhline(y=0, color='black', linestyle='--', lw=1)
        plt.show()

    def evaluate_pnls(self, pnl_cats: List[str], pnl_cids: List[str] = ['ALL'],
                      start: str = None, end: str = None):

        """Small table of key PnL statistics

        :param <List[str]> pnl_cats: list of PnL categories that should be plotted.
        :param <List[str]> pnl_cids: list of cross sections to be plotted; default is
            'ALL' (global PnL).
            Note: one can only have multiple PnL categories or multiple cross sections,
            not both.
        :param <str> start: start date in format.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.

        :return: standardized dataframe with key PnL performance statistics
        """

        if pnl_cats is None:
            pnl_cats = self.pnl_names

        assert (len(pnl_cats) == 1) | (len(pnl_cids) == 1)

        dfx = reduce_df(self.df, pnl_cats, pnl_cids, start, end, self.black,
                        out_all=False)

        groups = 'xcat' if len(pnl_cids) == 1 else 'cid'
        stats = ['Return (pct ar)', 'St. Dev. (pct ar)', 'Sharpe ratio', 'Sortino ratio',
                 'Max 21-day draw', 'Max 6-month draw', 'Traded months']

        dfw = dfx.pivot(index='real_date', columns=groups, values='value')
        df = pd.DataFrame(columns=dfw.columns, index=stats)

        df.iloc[0, :] = dfw.mean(axis=0) * 261
        df.iloc[1, :] = dfw.std(axis=0) * np.sqrt(261)
        df.iloc[2, :] = df.iloc[0, :] / df.iloc[1, :]
        dsd = dfw.apply(lambda x: np.sqrt(np.sum(x[x < 0]**2)/len(x))) * np.sqrt(261)
        df.iloc[3, :] = df.iloc[0, :] / dsd
        df.iloc[4, :] = dfw.rolling(21).sum().min()
        df.iloc[5, :] = dfw.rolling(6*21).sum().min()
        df.iloc[6, :] = dfw.resample('M').sum().count()

        return df

    def pnl_names(self):
        """Print list of names of available PnLs in the class instance"""

        print(self.pnl_names)

    def pnl_df(self, pnl_names: List[str] = None,  cs: bool = False):
        """Return data frame with PnLs

        :param <List[str]> pnl_names: list of names of PnLs to be returned.
            Default is 'ALL'.
        :param <bool> cs: inclusion of cross section PnLs. Default is False.

        :return custom data frame with PnLs
        """
        selected_pnls = pnl_names if pnl_names is not None else self.pnl_names

        filter_1 = self.df['xcat'].isin(selected_pnls)
        filter_2 = self.df['cid'] == 'ALL' if not cs else True

        return self.df[filter_1 & filter_2]


if __name__ == "__main__":
    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    cols_1 = ['earliest', 'latest', 'mean_add', 'sd_mult']
    df_cids = pd.DataFrame(index=cids, columns=cols_1)
    df_cids.loc['AUD',] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD',] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP',] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD',] = ['2002-01-01', '2020-09-30', -0.1, 2]

    cols_2 = cols_1 + ['ar_coef', 'back_coef']

    df_xcats = pd.DataFrame(index=xcats, columns=cols_2)
    df_xcats.loc['XR',] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY',] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH',] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL',] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    black = {'AUD': ['2006-01-01', '2015-12-31'], 'GBP': ['2012-01-01', '2100-01-01']}
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Initiate instance

    pnl = NaivePnL(dfd, ret='XR', sigs=['CRY', 'GROWTH', 'INFL'],
                   cids=cids, start='2000-01-01', blacklist=black)

    # Make PnLs

    pnl.make_pnl('CRY', sig_op='zn_score_pan', rebal_freq='monthly',
                 vol_scale=10, rebal_slip=1,
                 pnl_name='PNL_CRY_PZN')
    pnl.make_pnl('CRY', sig_op='binary', rebal_freq='monthly',
                 rebal_slip=1, vol_scale=10,
                 pnl_name='PNL_CRY_DIG')
    pnl.make_pnl('GROWTH', sig_op='zn_score_cs', rebal_freq='monthly',
                 rebal_slip=1, vol_scale=10,
                 pnl_name='PNL_GROWTH_IZN')

    # Plot PnLs

    pnl.plot_pnls(pnl_cats=['PNL_CRY_PZN', 'PNL_CRY_DIG', 'PNL_GROWTH_IZN'],
                  pnl_cids=['ALL'], start='2000-01-01')
    pnl.plot_pnls(pnl_cats=['PNL_CRY_PZN'], pnl_cids=['CAD', 'NZD'],
                  start='2000-01-01')

    # Return evaluation and PnL data frames

    df_eval = pnl.evaluate_pnls(
        pnl_cats=['PNL_CRY_PZN', 'PNL_CRY_DIG', 'PNL_GROWTH_IZN'],
        pnl_cids=['ALL'], start='2000-01-01')
    df_pnls = pnl.pnl_df()
    df_pnls.head()
