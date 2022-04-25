import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df
from macrosynergy.panel.make_zn_scores import make_zn_scores


class NaivePnL:

    """
    Computes and collects illustrative PnLs with limited signal options and
    disregarding transaction costs.

    :param <pd.Dataframe> df: standardized DataFrame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value'.
    :param <str> ret: return category.
    :param <List[str]> sigs: signal categories.
    :param <List[str]> cids: cross sections to be considered. Default is all in the
        dataframe.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded
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
        self.cids = cids
        self.df, self.xcats, self.cids = reduce_df(df[cols], xcats, cids, start, end,
                                                   blacklist, out_all=True)
        self.df['real_date'] = pd.to_datetime(self.df['real_date'])
        self.pnl_names = []
        self.black = blacklist

    def make_pnl(self, sig: str, sig_op: str = 'zn_score_pan', pnl_name: str = None,
                 rebal_freq: str = 'daily', rebal_slip = 0, vol_scale: float = None,
                 long_only: bool = False, min_obs: int = 252, iis: bool = True,
                 sequential: bool = True, neutral: str = 'zero', thresh: float = None):

        """
        Calculate daily PnL and add to the main dataframe held on an instance of the
        Class.

        :param <str> sig: name of signal that is the basis for positioning. The signal
            is assumed to be recorded at the end of the day prior to position taking.
        :param <str> sig_op: signal transformation options; must be one of
            'zn_score_pan', 'zn_score_cs', or 'binary'. The default is 'zn_score_pan'.
            'zn_score_pan': transforms raw signals into z-scores around zero value
            based on the whole panel. The neutral level & standard deviation will use the
            cross-section of panels.
            'zn_score_cs': transforms signals to z-scores around zero based on
            cross-section alone.
            'binary': transforms signals into uniform long/shorts (1/-1) across all
            sections.
            N.B.: zn-score here means standardized score with zero being the natural
            neutral level and standardization through division by mean absolute value
            (the standard deviation).
        :param <str> pnl_name: name of the PnL to be generated and stored.
            Default is none, i.e. a default name is given.
            Previously calculated PnLs in the class will be overwritten. This means that
            if a set of PnLs is to be compared they require custom names.
        :param <str> rebal_freq: re-balancing frequency for positions according to signal
            must be one of 'daily' (default), 'weekly' or 'monthly'. The re-balancing is
            only concerned with the signal value on the re-balancing date which is
            delimited by the frequency chosen.
        :param <str> rebal_slip: re-balancing slippage in days. Default is 1 which
            means that it takes one day to re-balance the position and that the new
            positions produce PnL from the following day after the signal has been
            recorded.
        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given.
            This is for comparative visualization and not out-of-sample. Default is none.
        :param <bool> long_only: if True, the long-only returns will be computed which
            act as a basis for comparison against the signal-adjusted returns. Will take
            a long-only position in the category passed to the parameter 'self.ret'.
            Default is False.
        :param <int> min_obs: the minimum number of observations required to calculate
            zn_scores. Default is 252.
        :param <bool> iis: if True (default) zn-scores are also calculated for the initial
            sample period defined by min-obs, on an in-sample basis, to avoid losing
            history.
        :param <bool> sequential: if True (default) score parameters (neutral level and
            standard deviations) are estimated sequentially with concurrently available
            information only.
        :param <str> neutral: method to determine neutral level. Default is 'zero'.
            Alternatives are 'mean' and "median".
        :param <float> thresh: threshold value beyond which scores are winsorized,
            i.e. contained at that threshold. Therefore, the threshold is the maximum
            absolute score value that the function is allowed to produce. The minimum
            threshold is one standard deviation.

        """

        assert sig in self.sigs
        assert sig_op in ['zn_score_pan', 'zn_score_cs', 'binary']
        assert rebal_freq in ['daily', 'weekly', 'monthly']

        dfx = self.df[self.df['xcat'].isin([self.ret, sig])]

        if sig_op == 'binary':
            dfw = dfx.pivot(index=['cid', 'real_date'], columns='xcat', values='value')
            dfw['psig'] = np.sign(dfw[sig])
        else:
            panw = 1 if sig_op == 'zn_score_pan' else 0
            # Utilising the signal to subsequently take a "position" in self.ret.
            # Deviation from the mean as the signal.
            df_ms = make_zn_scores(dfx, xcat=sig, neutral=neutral, pan_weight=panw,
                                   sequential=sequential, min_obs=min_obs, iis=iis,
                                   thresh=thresh)
            df_ms = df_ms.drop('xcat', axis=1)
            df_ms['xcat'] = 'psig'
            dfx_concat = pd.concat([dfx, df_ms])
            dfw = dfx_concat.pivot(index=['cid', 'real_date'], columns='xcat',
                                   values='value')

        # Signal for the following day explains the lag mechanism.
        dfw['psig'] = dfw['psig'].groupby(level=0).shift(1)
        dfw.reset_index(inplace=True)
        dfw = dfw.rename_axis(None, axis=1)

        dfw = dfw.sort_values(['cid', 'real_date'])

        if long_only:
            dfw_long = self.long_only_pnl(dfw=dfw, ret=self.ret, vol_scale=vol_scale)
            self.__dict__['dfw_long'] = dfw_long.reset_index(drop=True)

        if rebal_freq != 'daily':
            dfw['sig'] = self.rebalancing(dfw=dfw, rebal_freq=rebal_freq,
                                          rebal_slip=rebal_slip)
        else:
            dfw = dfw.rename({'psig': 'sig'}, axis=1)
        dfw['value'] = dfw[self.ret] * dfw['sig']

        df_pnl = dfw.loc[:, ['cid', 'real_date', 'value']]

        # Compute the return across the panel. The returns are still computed daily
        # regardless of the re-balancing frequency potentially occurring weekly or
        # monthly.
        df_pnl_all = df_pnl.groupby(['real_date']).sum()
        df_pnl_all = df_pnl_all[df_pnl_all['value'].cumsum() != 0]
        df_pnl_all['cid'] = 'ALL'
        df_pnl_all = df_pnl_all.reset_index()[df_pnl.columns]
        # Will be inclusive of each individual cross-section's signal-adjusted return and
        # the aggregated panel return.
        df_pnl = df_pnl.append(df_pnl_all)

        if vol_scale is not None:
            leverage = vol_scale * (df_pnl_all['value'].std() * np.sqrt(261))**(-1)
            df_pnl['value'] = df_pnl['value'] * leverage

        pnn = ('PNL_' + sig) if pnl_name is None else pnl_name
        df_pnl['xcat'] = pnn
        if pnn in self.pnl_names:
            self.df = self.df[~(self.df['xcat'] == pnn)]
        else:
            self.pnl_names = self.pnl_names + [pnn]

        self.df = self.df.append(df_pnl[self.df.columns]).reset_index(drop=True)

    @staticmethod
    def long_only_pnl(dfw: pd.DataFrame, ret: str, vol_scale: float = None):
        """
        Method used to compute the PnL accrued from simply taking a long-only position in
        the category, 'self.ret'. The returns from the category are not predicated on any
        exogenous signal.

        :param <pd.DataFrame> dfw:
        :param <str> ret: return category.
        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given.
            This is for comparative visualization and not out-of-sample. Default is none.

        :return <pd.DataFrame> dfw_long: standardised dataframe containing exclusively
            the return category, and the long-only panel return.
        """

        dfw_long = dfw[['cid', 'real_date', ret]]
        panel_pnl = dfw_long.groupby(['real_date']).sum()
        panel_pnl = panel_pnl.reset_index(level=0)
        panel_pnl['cid'] = 'ALL'
        dfw_long = dfw_long.append(panel_pnl)
        dfw_long['xcat'] = ret
        dfw_long = dfw_long.rename(columns={ret: "value"})

        if vol_scale:
            leverage = vol_scale * (panel_pnl[ret].std() * np.sqrt(261))**(-1)
            dfw_long['value'] = dfw_long['value'] * leverage

        return dfw_long[['cid', 'xcat', 'real_date', 'value']]

    @staticmethod
    def rebalancing(dfw: pd.DataFrame, rebal_freq: str = 'daily', rebal_slip = 0):
        """
        The signals are calculated daily and for each individual cross-section defined in
        the panel. However, re-balancing a position can occur more infrequently than
        daily. Therefore, produce the re-balancing values according to the more
        infrequent timeline (weekly or monthly).

        :param <pd.Dataframe> dfw: DataFrame with each category represented by a column
            and the daily signal is also included with the column name 'psig'.
        :param <str> rebal_freq: re-balancing frequency for positions according to signal
            must be one of 'daily' (default), 'weekly' or 'monthly'.
        :param <str> rebal_slip: re-balancing slippage in days.

        :return <pd.Series>: will return a pd.Series containing the associated signals
            according to the re-balancing frequency.
        """

        # The re-balancing days are the first of the respective time-periods.
        dfw['year'] = dfw['real_date'].dt.year
        if rebal_freq == 'monthly':
            dfw['month'] = dfw['real_date'].dt.month
            rebal_dates = dfw.groupby(['cid', 'year', 'month'])['real_date'].min()
        elif rebal_freq == 'weekly':
            dfw['week'] = dfw['real_date'].dt.week
            rebal_dates = dfw.groupby(['cid', 'year', 'week'])['real_date'].min()

        # Convert the index, 'cid', to a formal column aligned to the re-balancing dates.
        r_dates_df = rebal_dates.reset_index(level=0)
        r_dates_df.reset_index(drop=True, inplace=True)
        dfw = dfw[['real_date', 'psig', 'cid']]

        # Isolate the required signals on the re-balancing dates. Only concerned with the
        # respective signal on the re-balancing date. However, the produced dataframe
        # will only be defined over the re-balancing dates. Therefore, merge the
        # aforementioned dataframe with the original dataframe such that all business
        # days are included. The intermediary dates, dates between re-balancing dates,
        # will initially be populated by NA values. To ensure the signal is used for the
        # duration between re-balancing dates, forward fill the computed signal over the
        # associated dates.

        # The signal is computed for each individual cross-section. Therefore, merge on
        # the real_date and the cross-section.
        rebal_merge = r_dates_df.merge(dfw, how='left', on=['real_date', 'cid'])
        rebal_merge = dfw[['real_date', 'cid']].merge(rebal_merge, how='left',
                                                      on=['real_date', 'cid'])

        rebal_merge['psig'] = rebal_merge['psig'].fillna(method='ffill').shift(rebal_slip)
        rebal_merge = rebal_merge.sort_values(['cid', 'real_date'])
        sig_series = rebal_merge['psig']

        return sig_series

    def plot_pnls(self, pnl_cats: List[str], pnl_cids: List[str] = ['ALL'],
                  start: str = None, end: str = None, add_long: bool = False,
                  figsize: Tuple = (10, 6), title: str = "Cumulative Naive PnL",
                  xcat_labels: List[str] = None):

        """
        Plot line chart of cumulative PnLs, single PnL, multiple PnL types per
        cross section, or multiple cross sections per PnL type.

        :param <List[str]> pnl_cats: list of PnL categories that should be plotted.
        :param <List[str]> pnl_cids: list of cross sections to be plotted;
            default is 'ALL' (global PnL).
            Note: one can only have multiple PnL categories or multiple cross sections,
            not both.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.
        :param <bool> add_long: if True, long-only benchmark PnLs will be added to the
            plot. The default is False.
        :param <tuple> figsize: tuple of plot width and height. Default is (10,6).
        :param <str> title: allows entering text for a custom chart header.
        :param <List[str]> xcat_labels: custom labels to be used for the PnLs.

        """

        if pnl_cats is None:
            pnl_cats = self.pnl_names

        assert (len(pnl_cats) == 1) | (len(pnl_cids) == 1)
        error_message = "The number of custom labels must match the defined number of " \
                        "categories in pnl_cats."
        if xcat_labels is not None:
            assert(len(xcat_labels) == len(pnl_cats)), error_message
        else:
            xcat_labels = pnl_cats

        dfx = reduce_df(self.df, pnl_cats, pnl_cids, start, end, self.black,
                        out_all=False)
        no_cids = len(pnl_cids)

        if add_long:
            error_message = "Long-only DataFrame missing. The parameter, 'long_only', " \
                            "must be set to True when calling make_pnl() method."
            assert 'dfw_long' in self.__dict__.keys(), error_message

            dfw_long = reduce_df(self.dfw_long, xcats=None, cids=pnl_cids,
                                 start=start, end=end, blacklist=self.black,
                                 out_all=False)
            dfw_long['cum_value'] = dfw_long.groupby('cid').cumsum()

        sns.set_theme(style='whitegrid', palette='colorblind',
                      rc={'figure.figsize': figsize})

        if no_cids == 1:
            dfx['cum_value'] = dfx.groupby('xcat').cumsum()
            if add_long:
                dfx = dfx.append(dfw_long)
                pnl_cats.append(self.ret)

            ax = sns.lineplot(data=dfx, x='real_date', y='cum_value',
                              hue='xcat', hue_order=pnl_cats,
                              estimator=None, lw=1)
            plt.legend(loc='upper left', labels=xcat_labels)
            leg = ax.axes.get_legend()
            leg.set_title('PnL category(s) for ' + pnl_cids[0])

        else:
            dfx['cum_value'] = dfx.groupby('cid').cumsum()
            if add_long:
                dfw_long['cid'] = dfw_long['cid'] + '_' + dfw_long['xcat']
                dfx = dfx.append(dfw_long)

            ax = sns.lineplot(data=dfx, x='real_date', y='cum_value',
                              hue='cid', estimator=None, lw=1)
            leg = ax.axes.get_legend()
            leg.set_title('Cross Sections')

        plt.title(title, fontsize=16)
        plt.xlabel('')
        plt.ylabel('% of risk capital, no compounding')
        plt.axhline(y=0, color='black', linestyle='--', lw=1)
        plt.show()

    def evaluate_pnls(self, pnl_cats: List[str], pnl_cids: List[str] = ['ALL'],
                      start: str = None, end: str = None):

        """
        Small table of key PnL statistics.

        :param <List[str]> pnl_cats: list of PnL categories that should be plotted.
        :param <List[str]> pnl_cids: list of cross sections to be plotted; default is
            'ALL' (global PnL).
            Note: one can only have multiple PnL categories or multiple cross sections,
            not both.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.

        :return <pd.DataFrame>: standardized dataframe with key PnL performance statistics
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
        """
        Print list of names of available PnLs in the class instance.
        """

        print(self.pnl_names)

    def pnl_df(self, pnl_names: List[str] = None,  cs: bool = False):
        """
        Return dataframe with PnLs.

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
    df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    cols_2 = cols_1 + ['ar_coef', 'back_coef']

    df_xcats = pd.DataFrame(index=xcats, columns=cols_2)
    df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    black = {'AUD': ['2006-01-01', '2015-12-31'], 'GBP': ['2012-01-01', '2100-01-01']}
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Initiate instance.

    pnl = NaivePnL(dfd, ret='XR', sigs=['CRY', 'GROWTH', 'INFL'],
                   cids=cids, start='2000-01-01', blacklist=black)

    # Make and plot PnLs to check correct labelling.

    pnl.make_pnl(sig='CRY', sig_op='zn_score_pan', rebal_freq='monthly',
                 vol_scale=5, long_only=True, rebal_slip=1,
                 pnl_name='PNL_CRY_PZN05', min_obs=250, thresh=2)
    pnl.make_pnl(sig='CRY', sig_op='zn_score_pan', rebal_freq='monthly',
                 vol_scale=10, rebal_slip=1,
                 pnl_name='PNL_CRY_PZN10', min_obs=250, thresh=2)
    pnl.make_pnl(sig='CRY', sig_op='zn_score_pan', rebal_freq='monthly',
                 vol_scale=20, rebal_slip=1,
                 pnl_name='PNL_CRY_PZN20', min_obs=250, thresh=2)
    print(pnl.df)

    pnl.plot_pnls(pnl_cats=['PNL_CRY_PZN20', 'PNL_CRY_PZN05', 'PNL_CRY_PZN10'],
                  pnl_cids=['ALL'], start='2000-01-01', add_long=True,
                  title="Custom Title")
    # Test using long-only parameter in succession but requesting on different
    # cross-sections.
    pnl.plot_pnls(pnl_cats=['PNL_CRY_PZN20'], pnl_cids=['CAD', 'NZD'],
                  start='2000-01-01', add_long=True)

    pnl.plot_pnls(pnl_cats=['PNL_CRY_PZN10', 'PNL_CRY_PZN20', 'PNL_CRY_PZN05'],
                  pnl_cids=['ALL'], start='2000-01-01', title="Custom Title",
                  xcat_labels=["cry10", "cry20", "cry5"])

    # Make and plot PnLs for other checks.

    pnl.make_pnl(sig='CRY', sig_op='binary', rebal_freq='monthly',
                 rebal_slip=1, vol_scale=10,
                 pnl_name='PNL_CRY_DIG')
    pnl.make_pnl(sig='GROWTH', sig_op='zn_score_cs', rebal_freq='monthly',
                 rebal_slip=1, vol_scale=10,
                 pnl_name='PNL_GROWTH_IZN')

    pnl.make_pnl(sig='CRY', sig_op='zn_score_pan', rebal_freq='monthly',
                 vol_scale=10, rebal_slip=1,
                 pnl_name='PNL_CRY_PZN', min_obs=250, thresh=1.5)

    pnl.plot_pnls(pnl_cats=['PNL_CRY_PZN', 'PNL_CRY_DIG', 'PNL_GROWTH_IZN'],
                  pnl_cids=['ALL'], start='2000-01-01')

    # Instantiate a new instance to test the long-only functionality.
    pnl = NaivePnL(dfd, ret='XR', sigs=['CRY', 'GROWTH', 'INFL'],
                   cids=cids, start='2000-01-01', blacklist=black)
    pnl.make_pnl(sig='CRY', sig_op='zn_score_pan', rebal_freq='monthly',
                 vol_scale=10, long_only=True, rebal_slip=1,
                 pnl_name='PNL_CRY_PZN', min_obs=250, thresh=1.5)

    pnl.plot_pnls(pnl_cats=['PNL_CRY_PZN'], pnl_cids=['CAD', 'NZD'],
                  start='2000-01-01', add_long=True)

    # Return evaluation and PnL DataFrames.
    df_eval = pnl.evaluate_pnls(
        pnl_cats=['PNL_CRY_PZN'],
        pnl_cids=['ALL'], start='2000-01-01')
    df_pnls = pnl.pnl_df()
    df_pnls.head()
