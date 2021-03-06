
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from typing import List, Union, Tuple
from itertools import product
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
    :param <List[str]> sigs: signal categories. Able to pass in multiple possible signals
        to the Class' constructor and their respective vintages will be held on the
        instance's DataFrame. The signals can subsequently be referenced through the
        self.make_pnl() method which receives a single signal per call.
    :param <List[str]> cids: cross sections that are traded. Default is all in the
        dataframe.
    :param <str, List[str]> bms: list of benchmark tickers for which
        correlations are displayed against PnL strategies.
        If there are not any benchmarks defined on the Class' instance, then the
        parameter, 'bms', in the method self.evaluate_pnls() becomes obsolete.
    :param <str> start: earliest date in ISO format. Default is None and earliest date
        in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df
        is used.
    :param <dict> blacklist: cross-sections with date ranges that should be excluded
        from the dataframe.

    """
    def __init__(self, df: pd.DataFrame, ret: str, sigs: List[str],
                 cids: List[str] = None, bms: Union[str, List[str]] = None,
                 start: str = None, end: str = None,
                 blacklist: dict = None):

        self.dfd = df
        assert isinstance(ret, str), "The return category expects a single <str>."
        self.ret = ret
        self.sigs = sigs
        xcats = [ret] + sigs

        cols = ['cid', 'xcat', 'real_date', 'value']
        self.df, self.xcats, self.cids = reduce_df(df[cols], xcats, cids, start, end,
                                                   blacklist, out_all=True)
        ticker_func = lambda t: t[0] + "_" + t[1]
        self.tickers = list(map(ticker_func, product(self.cids, self.xcats)))

        self.df['real_date'] = pd.to_datetime(self.df['real_date'])
        self.pnl_names = []
        self.signal_df = {}
        self.black = blacklist

        self.bm_bool = isinstance(bms, (str, list))
        if self.bm_bool:
            bms = [bms] if isinstance(bms, str) else bms
            add_bm_list, bm_dict = self.add_bm(df=self.dfd, bms=bms,
                                               tickers=self.tickers)
            self._bm_dict = bm_dict
            self.df = pd.concat([self.df] + add_bm_list)

    @classmethod
    def add_bm(cls, df: pd.DataFrame, bms: List[str],
               tickers: List[str]):
        """
        Return benchmark DataFrames which will be appended to the instance's DataFrame.
        Additionally, populate the benchmark dictionary which is used to host any valid
        benchmarks.

        :param <pd.DataFrame> df: aggregate DataFrame passed into the Class.
        :param <List[str]> bms: benchmark return tickers.
        :param <List[str]> tickers: the available tickers held in the reduced DataFrame.
            The reduced DataFrame consists exclusively of the signal & return categories.
        """

        add_bm_list = []
        bm_dict = {}
        for bm in bms:
            cid, xcat = bm.split("_", 1)
            dfa = df[(df['cid'] == cid) & (df['xcat'] == xcat)]

            if dfa.shape[0] == 0:
                print(f"{bm} has no observations in the DataFrame.")
            else:
                bm_dict[bm] = dfa.pivot(index='real_date', columns='xcat',
                                        values='value').squeeze(axis=0)
                if bm not in tickers:
                    add_bm_list.append(dfa)

        return add_bm_list, bm_dict

    @classmethod
    def make_signal(cls, dfx: pd.DataFrame, sig: str, sig_op: str = 'zn_score_pan',
                    min_obs: int = 252, iis: bool = True, sequential: bool = True,
                    neutral: str = 'zero', thresh: float = None):
        """
        Helper function used to produce the raw signal that forms the basis for
        positioning.

        :param <pd.DataFrame> dfx: DataFrame defined over the return & signal category.
        :param <str> sig: name of the raw signal.
        :param <str> sig_op: signal transformation.
        :param <int> min_obs: the minimum number of observations required to calculate
            zn_scores. Default is 252.
        :param <bool> iis: if True (default) zn-scores are also calculated for the initial
            sample period defined by min_obs, on an in-sample basis, to avoid losing
            history.
        :param <bool> sequential: if True (default) score parameters are estimated
            sequentially with concurrently available information only.
        :param <str> neutral: method to determine neutral level.
        :param <float> thresh: threshold value beyond which scores are winsorized,

        """

        if sig_op == 'binary':
            dfw = dfx.pivot(index=['cid', 'real_date'], columns='xcat', values='value')
            dfw['psig'] = np.sign(dfw[sig])
        else:
            panw = 1 if sig_op == 'zn_score_pan' else 0
            # The re-estimation frequency for the neutral level and standard deviation
            # will be the same as the re-balancing frequency. For instance, if the
            # neutral level is computed weekly, a material change in the signal will only
            # manifest along a similar timeline. Therefore, re-estimation and
            # re-balancing frequencies match.
            df_ms = make_zn_scores(dfx, xcat=sig, neutral=neutral, pan_weight=panw,
                                   sequential=sequential, min_obs=min_obs, iis=iis,
                                   thresh=thresh)

            df_ms = df_ms.drop('xcat', axis=1)
            df_ms['xcat'] = 'psig'

            dfx_concat = pd.concat([dfx, df_ms])
            dfw = dfx_concat.pivot(index=['cid', 'real_date'], columns='xcat',
                                   values='value')

        # Reconstruct the DataFrame to recognise the signal's start date for each
        # individual cross-section
        dfw_list = []
        for c, cid_df in dfw.groupby(level=0):
            first_date = cid_df.loc[:, 'psig'].first_valid_index()
            cid_df = cid_df.loc[first_date:, :]
            dfw_list.append(cid_df)

        return pd.concat(dfw_list)

    @classmethod
    def rebalancing(cls, dfw: pd.DataFrame, rebal_freq: str = 'daily', rebal_slip = 0):
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

        # The re-balancing days are the first of the respective time-periods because of
        # the shift forward by one day applied earlier in the code. Therefore, only
        # concerned with the minimum date of each re-balance period.
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
        # Re-establish the daily date series index where the intermediary dates, between
        # the re-balancing dates, will be populated using a forward fill.
        rebal_merge = dfw[['real_date', 'cid']].merge(rebal_merge, how='left',
                                                      on=['real_date', 'cid'])
        rebal_merge['psig'] = rebal_merge['psig'].fillna(method='ffill').shift(rebal_slip)
        rebal_merge = rebal_merge.sort_values(['cid', 'real_date'])

        rebal_merge = rebal_merge.set_index('real_date')
        sig_series = rebal_merge.drop(['cid'], axis=1)

        return sig_series

    def make_pnl(self, sig: str, sig_op: str = 'zn_score_pan', pnl_name: str = None,
                 rebal_freq: str = 'daily', rebal_slip = 0, vol_scale: float = None,
                 min_obs: int = 261, iis: bool = True, sequential: bool = True,
                 neutral: str = 'zero', thresh: float = None):

        """
        Calculate daily PnL and add to class instance.

        :param <str> sig: name of raw signal that is basis for positioning. The signal
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
            neutral level and standardization through division by mean absolute value.
        :param <str> pnl_name: name of the PnL to be generated and stored.
            Default is None, i.e. a default name is given. The default name will be:
            ('PNL_' + sig).
            Previously calculated PnLs in the class will be overwritten. This means that
            if a set of PnLs is to be compared, each PnL requires a distinct name.
        :param <str> rebal_freq: re-balancing frequency for positions according to signal
            must be one of 'daily' (default), 'weekly' or 'monthly'. The re-balancing is
            only concerned with the signal value on the re-balancing date which is
            delimited by the frequency chosen.
            Additionally, the re-balancing frequency will be applied to make_zn_scores()
            if used as the method to produce the raw signals.
        :param <str> rebal_slip: re-balancing slippage in days. Default is 1 which
            means that it takes one day to re-balance the position and that the new
            positions produce PnL from the second day after the signal has been recorded.
        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given.
            This is for comparative visualization and not out-of-sample. Default is none.
        :param <int> min_obs: the minimum number of observations required to calculate
            zn_scores. Default is 252.
        :param <bool> iis: if True (default) zn-scores are also calculated for the initial
            sample period defined by min_obs, on an in-sample basis, to avoid losing
            history.
        :param <bool> sequential: if True (default) score parameters (neutral level and
            standard deviations) are estimated sequentially with concurrently available
            information only.
        :param <str> neutral: method to determine neutral level. Default is 'zero'.
            Alternatives are 'mean' and "median".
        :param <float> thresh: threshold value beyond which scores are winsorized,
            i.e. contained at that threshold. Therefore, the threshold is the maximum
            absolute score value that the function is allowed to produce. The minimum
            threshold is one standard deviation. Default is no threshold.

        """

        error_sig = f"Signal category missing from the options defined on the class: " \
                    f"{self.sigs}. "
        assert sig in self.sigs, error_sig
        sig_options = ['zn_score_pan', 'zn_score_cs', 'binary']
        error_sig_method = f"The signal transformation method, {sig_op}, is not one of " \
                           f"the options specified: {sig_options}."
        assert sig_op in sig_options, error_sig_method
        freq_params = ['daily', 'weekly', 'monthly']
        freq_error = f"Re-balancing frequency must be one of: {freq_params}."
        assert rebal_freq in freq_params, freq_error

        # DataFrame consisting exclusively of the two types of categories: the return
        # category and associated signal category.
        dfx = self.df[self.df['xcat'].isin([self.ret, sig])]

        dfw = self.make_signal(dfx=dfx, sig=sig, sig_op=sig_op, min_obs=min_obs, iis=iis,
                               sequential=sequential, neutral=neutral, thresh=thresh)

        # Multi-index DataFrame with a natural minimum lag applied.
        dfw['psig'] = dfw['psig'].groupby(level=0).shift(1)
        dfw.reset_index(inplace=True)
        dfw = dfw.rename_axis(None, axis=1)

        dfw = dfw.sort_values(['cid', 'real_date'])

        if rebal_freq != 'daily':
            sig_series = self.rebalancing(dfw=dfw, rebal_freq=rebal_freq,
                                          rebal_slip=rebal_slip)
            dfw['sig'] = np.squeeze(sig_series.to_numpy())
        else:
            dfw = dfw.rename({'psig': 'sig'}, axis=1)

        # The signals are generated across the panel.
        dfw['value'] = dfw[self.ret] * dfw['sig']

        df_pnl = dfw.loc[:, ['cid', 'real_date', 'value']]

        # Compute the return across the panel. The returns are still computed daily
        # regardless of the re-balancing frequency potentially occurring weekly or
        # monthly.
        df_pnl_all = df_pnl.groupby(['real_date']).sum()
        df_pnl_all = df_pnl_all[df_pnl_all['value'].cumsum() != 0]
        # Returns are computed for each cross-section and across the panel.
        df_pnl_all['cid'] = 'ALL'
        df_pnl_all = df_pnl_all.reset_index()[df_pnl.columns]
        # Will be inclusive of each individual cross-section's signal-adjusted return and
        # the aggregated panel return.
        df_pnl = df_pnl.append(df_pnl_all)

        if vol_scale is not None:
            leverage = vol_scale * (df_pnl_all['value'].std() * np.sqrt(261))**(-1)
            df_pnl['value'] = df_pnl['value'] * leverage

        pnn = ('PNL_' + sig) if pnl_name is None else pnl_name
        # Populating the signal dictionary is required for the display methods:
        self.signal_df[pnn] = dfw.loc[:, ['cid', 'real_date', 'sig']]

        df_pnl['xcat'] = pnn
        if pnn in self.pnl_names:
            self.df = self.df[~(self.df['xcat'] == pnn)]
        else:
            self.pnl_names = self.pnl_names + [pnn]

        agg_df = pd.concat([self.df, df_pnl[self.df.columns]])
        self.df = agg_df.reset_index(drop=True)

    def make_long_pnl(self, vol_scale: float = None, label: str = None):
        """
        The long-only returns will be computed which act as a basis for comparison
        against the signal-adjusted returns. Will take a long-only position in the
        category passed to the parameter 'self.ret'.

        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given.
            This is for comparative visualization and not out-of-sample, and is applied
            to the long-only position. Default is none.
        :param <str> label: associated label that will be mapped to the long-only
            DataFrame. The label will be used in the plotting graphic for plot_pnls().
            If a label is not defined, the default will be the name of the return
            category.
        """

        error_vol = "The volatility scale must be a numerical value."
        assert isinstance(vol_scale, (float, int)), error_vol

        if label is None:
            label = self.ret

        dfx = self.df[self.df['xcat'].isin([self.ret])]

        df_long = self.long_only_pnl(dfw=dfx, vol_scale=vol_scale,
                                     label=label)

        self.df = pd.concat([self.df, df_long])

        if label not in self.pnl_names:
            self.pnl_names = self.pnl_names + [label]

        self.df = self.df.reset_index(drop=True)

    @staticmethod
    def long_only_pnl(dfw: pd.DataFrame, vol_scale: float = None, label: str = None):
        """
        Method used to compute the PnL accrued from simply taking a long-only position in
        the category, 'self.ret'. The returns from the category are not predicated on any
        exogenous signal.

        :param <pd.DataFrame> dfw:
        :param <bool> vol_scale: ex-post scaling of PnL to annualized volatility given.
            This is for comparative visualization and not out-of-sample. Default is none.
        :param <str> label: associated label that will be mapped to the long-only
            DataFrame.

        :return <pd.DataFrame> panel_pnl: standardised dataframe containing exclusively
            the return category, and the long-only panel return.
        """

        dfw_long = dfw.reset_index(drop=True)

        panel_pnl = dfw_long.groupby(['real_date']).sum()
        panel_pnl = panel_pnl.reset_index(level=0)
        panel_pnl['cid'] = 'ALL'
        panel_pnl['xcat'] = label

        if vol_scale:
            leverage = vol_scale * (panel_pnl['value'].std() * np.sqrt(261)) ** (-1)
            panel_pnl['value'] = panel_pnl['value'] * leverage

        return panel_pnl[['cid', 'xcat', 'real_date', 'value']]

    def plot_pnls(self, pnl_cats: List[str] = None, pnl_cids: List[str] = ['ALL'],
                  start: str = None, end: str = None, figsize: Tuple = (10, 6),
                  title: str = "Cumulative Naive PnL",
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
            pnl_cats_c = pnl_cats.copy()
            xcat_labels = pnl_cats_c

        dfx = reduce_df(self.df, pnl_cats, pnl_cids, start,
                        end, self.black, out_all=False)

        no_cids = len(pnl_cids)

        sns.set_theme(style='whitegrid', palette='colorblind',
                      rc={'figure.figsize': figsize})

        if no_cids == 1:
            dfx['cum_value'] = dfx.groupby('xcat').cumsum()

            ax = sns.lineplot(data=dfx, x='real_date', y='cum_value',
                              hue='xcat', hue_order=pnl_cats,
                              estimator=None, lw=1)
            plt.legend(loc='upper left', labels=xcat_labels)
            leg = ax.axes.get_legend()
            leg.set_title('PnL category(s) for ' + pnl_cids[0])

        else:
            dfx['cum_value'] = dfx.groupby('cid').cumsum()

            ax = sns.lineplot(data=dfx, x='real_date', y='cum_value',
                              hue='cid', estimator=None, lw=1)
            leg = ax.axes.get_legend()
            leg.set_title('Cross Sections')

        plt.title(title, fontsize=16)
        plt.xlabel('')
        plt.ylabel('% of risk capital, no compounding')
        plt.axhline(y=0, color='black', linestyle='--', lw=1)
        plt.show()

    def signal_heatmap(self, pnl_name: str, pnl_cids: List[str] = None,
                       start: str = None, end: str = None, freq: str = 'm',
                       title: str = "Average applied signal values",
                       x_label: str = "", y_label: str = "",
                       figsize: (float, float) = None):

        """
        Display heatmap of signals across times and cross-sections.

        :param <str> pnl_name: name of naive PnL whose signals are displayed.
            N.B.: Signal is here is the value that actually determines
            the concurrent PnL.
        :param <List[str]> pnl_cids: cross-sections. Default is all available.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.
        :param <str> freq: frequency for which signal average is displayed.
            Default is monthly ('m'). The only alternative is quarterly ('q').
        :param <str> title: allows entering text for a custom chart header.
        :param <str> x_label: label for the x-axis. Default is None.
        :param <str> y_label: label for the y-axis. Default is None.
        :param <(float, float)> figsize: width and height in inches.
            Default is (14, number of cross sections).
        """

        assert isinstance(pnl_name, str), "The method expects to receive a single " \
                                          "PnL name."
        error_cats = f"The PnL passed to 'pnl_name' parameter is not defined. The " \
                     f"possible options are {self.pnl_names}."
        assert pnl_name in self.pnl_names, error_cats

        error_time = "Defined time-period must be monthly ('m') or quarterly ('q')"
        assert isinstance(freq, str) and freq in ['m', 'q'], error_time

        error_cids = f"Cross-sections not available. Available cids are:" \
                     f"{self.cids}."

        if pnl_cids is None:
            pnl_cids = self.cids
        else:
            assert set(pnl_cids) <= set(self.cids), error_cids

        assert isinstance(x_label, str), f"<str> expected - received {type(x_label)}."
        assert isinstance(y_label, str), f"<str> expected - received {type(y_label)}."

        dfx = self.signal_df[pnl_name]
        dfw = dfx.pivot(index='real_date', columns='cid', values='sig')
        dfw = dfw[pnl_cids]

        if start is None:
            start = dfw.index[0]
        elif end is None:
            end = dfw.index[-1]

        dfw = dfw.truncate(before=start, after=end)

        dfw = dfw.resample(freq, axis=0).mean()

        if figsize is None:
            figsize = (14, len(pnl_cids))

        fig, ax = plt.subplots(figsize=figsize)
        dfw = dfw.transpose()
        dfw.columns = [str(d.strftime('%d-%m-%Y')) for d in dfw.columns]
        sns.heatmap(dfw, cmap="vlag_r", center=0)

        ax.set(xlabel=x_label, ylabel=y_label)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(title, fontsize=14)

        plt.show()

    def agg_signal_bars(self, pnl_name: str, freq: str = 'm',
                       metric: str = "direction", title: str = "Directional Bar Chart.",
                       y_label: str = ""):
        """
        Display aggregate signal strength and - potentially - direction.

        :param <str> pnl_name: name of the PnL whose signals are to be visualized.
            N.B.: Signal is here is the value that actually determines
            the concurrent PnL.
        :param <str> freq: frequency at which the signal is visualized. Default is
            monthly ('m'). The alternative is quarterly ('q').
        :param <str> metric: the type of signal value. Default is "direction".
            Alternative is "strength".
        :param <str> title: allows entering text for a custom chart header.
        :param <str> y_label: label for the y-axis. Default is None.

        """

        assert isinstance(pnl_name, str), "The method expects to receive a single " \
                                          "PnL name."
        error_cats = f"The PnL passed to 'pnl_name' parameter is not defined. The " \
                     f"possible options are {self.pnl_names}."
        assert pnl_name in self.pnl_names, error_cats

        error_time = "Defined time-period must either be monthly, m, or quarterly, q."
        assert isinstance(freq, str) and freq in ['m', 'q'], error_time

        metric_error = "The metric must either be 'direction' or 'strength'."
        assert metric in ['direction', 'strength'], metric_error

        dfx = self.signal_df[pnl_name]
        dfw = dfx.pivot(index='real_date', columns='cid', values='sig')

        # The method allows for visually understanding the overall direction of the
        # aggregate signal but also gaining insight into the proportional exposure to the
        # respective signal by measuring the absolute value, the size of the signal.
        # Is the PnL value generated by large returns or a large signal, position ?
        if metric == "strength":
            dfw = dfw.abs()

        dfw = dfw.resample(freq, axis=0).mean()
        # Sum across the timestamps to compute the aggregate signal according to the
        # down-sampling frequency.
        df_s = dfw.sum(axis=1)
        index = np.array(df_s.index)
        df_signal = pd.DataFrame(data=df_s.to_numpy(), columns=['aggregate_signal'],
                                 index=index)

        df_signal = df_signal.reset_index(level=0)
        df_signal = df_signal.rename({'index': ''}, axis='columns')
        dates = [pd.Timestamp(d) for d in df_signal['']]
        df_signal[''] = np.array(dates)

        plt.style.use('ggplot')

        fig, ax = plt.subplots()
        df_signal.plot.bar(x='', y='aggregate_signal', ax=ax, title=title,
                           ylabel=y_label, legend=False)

        ticklabels = [''] * len(df_signal)
        skip = len(df_signal) // 12
        ticklabels[::skip] = df_signal[''].iloc[::skip].dt.strftime('%Y-%m-%d')
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(ticklabels))
        fig.autofmt_xdate()

        def fmt(x, pos=0, max_i=len(ticklabels) - 1):
            i = int(x)
            i = 0 if i < 0 else max_i if i > max_i else i
            return dates[i]

        ax.fmt_xdata = fmt
        plt.show()

    def evaluate_pnls(self, pnl_cats: List[str], pnl_cids: List[str] = ['ALL'],
                      start: str = None, end: str = None):

        """
        Table of key PnL statistics.

        :param <List[str]> pnl_cats: list of PnL categories that should be plotted.
        :param <List[str]> pnl_cids: list of cross-sections to be plotted; default is
            'ALL' (global PnL).
            Note: one can only have multiple PnL categories or multiple cross-sections,
            not both.
        :param <str> start: earliest date in ISO format. Default is None and earliest
            date in df is used.
        :param <str> end: latest date in ISO format. Default is None and latest date
            in df is used.

        :return <pd.DataFrame>: standardized DataFrame with key PnL performance
            statistics
        """

        error_cids = "List of cross-sections expected."
        error_xcats = "List of PnL categories expected."
        assert isinstance(pnl_cids, list), error_cids
        assert isinstance(pnl_cats, list), error_xcats
        assert all([isinstance(elem, str) for elem in pnl_cids]), error_cids
        assert all([isinstance(elem, str) for elem in pnl_cats]), error_xcats

        if pnl_cats is None:
            # The field, self.pnl_names, is a data structure that stores the name of the
            # category assigned to PnL values. Each time make_pnl() method is called, the
            # computed DataFrame will have an associated category established by the
            # logical method: ('PNL_' + sig) if pnl_name is None else pnl_name. Each
            # category will be held in the data structure.
            pnl_cats = self.pnl_names
        else:
            if not set(pnl_cats) <= set(self.pnl_names):
                missing = [pnl for pnl in pnl_cats if pnl not in self.pnl_names]
                pnl_error = f"Received PnL categories have not been defined. The PnL " \
                            f"category(s) which has not been defined is: {missing}."
                raise ValueError(pnl_error)

        assert (len(pnl_cats) == 1) | (len(pnl_cids) == 1)

        dfx = reduce_df(self.df, pnl_cats, pnl_cids, start,
                        end, self.black, out_all=False)

        groups = 'xcat' if len(pnl_cids) == 1 else 'cid'
        stats = ['Return (pct ar)', 'St. Dev. (pct ar)', 'Sharpe Ratio', 'Sortino Ratio',
                 'Max 21-day draw', 'Max 6-month draw', 'Traded Months']

        # If benchmark tickers have been passed into the Class and if the tickers are
        # present in self.dfd.
        list_for_dfbm = []

        if self.bm_bool and bool(self._bm_dict):

            list_for_dfbm = list(self._bm_dict.keys())
            for bm in list_for_dfbm:
                stats.insert(len(stats) - 1, f"{bm} correl")

        dfw = dfx.pivot(index='real_date', columns=groups, values='value')
        df = pd.DataFrame(columns=dfw.columns, index=stats)

        df.iloc[0, :] = dfw.mean(axis=0) * 261
        df.iloc[1, :] = dfw.std(axis=0) * np.sqrt(261)
        df.iloc[2, :] = df.iloc[0, :] / df.iloc[1, :]
        dsd = dfw.apply(lambda x: np.sqrt(np.sum(x[x < 0]**2)/len(x))) * np.sqrt(261)
        df.iloc[3, :] = df.iloc[0, :] / dsd
        df.iloc[4, :] = dfw.rolling(21).sum().min()
        df.iloc[5, :] = dfw.rolling(6*21).sum().min()
        if len(list_for_dfbm) > 0:
            bm_df = pd.concat(list(self._bm_dict.values()),
                              axis=1)
            for i, bm in enumerate(list_for_dfbm):
                correlation = dfw.corrwith(bm_df.iloc[:, i], axis=0,
                                           method='pearson')
                df.iloc[6 + i, :] = correlation

        df.iloc[6 + len(list_for_dfbm), :] = dfw.resample('M').sum().count()

        return df

    def print_pnl_names(self):
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
    cids = ['AUD', 'CAD', 'GBP', 'NZD', 'USD', 'EUR']
    xcats = ['EQXR', 'CRY', 'GROWTH', 'INFL', 'DUXR']

    cols_1 = ['earliest', 'latest', 'mean_add', 'sd_mult']
    df_cids = pd.DataFrame(index=cids, columns=cols_1)
    df_cids.loc['AUD', :] = ['2008-01-03', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD', :] = ['2010-01-03', '2020-11-30', 0, 1]
    df_cids.loc['GBP', :] = ['2012-01-03', '2020-11-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-03', '2020-09-30', -0.1, 2]
    df_cids.loc['USD'] = ['2015-01-03', '2020-12-31', 0.2, 2]
    df_cids.loc['EUR'] = ['2008-01-03', '2020-12-31', 0.1, 2]

    cols_2 = cols_1 + ['ar_coef', 'back_coef']

    df_xcats = pd.DataFrame(index=xcats, columns=cols_2)
    df_xcats.loc['EQXR'] = ['2000-01-03', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH'] = ['2010-01-03', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]
    df_xcats.loc['DUXR'] = ['2000-01-01', '2020-12-31', 0.1, 0.5, 0, 0.1]

    black = {'AUD': ['2006-01-01', '2015-12-31'], 'GBP': ['2022-01-01', '2100-01-01']}
    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    # Instantiate a new instance to test the long-only functionality.
    pnl = NaivePnL(dfd, ret='EQXR', sigs=['CRY', 'GROWTH', 'INFL'],
                   cids=cids, start='2000-01-01', blacklist=black,
                   bms=["EUR_EQXR", "USD_EQXR"])

    pnl.make_pnl(sig='GROWTH', sig_op='zn_score_pan', rebal_freq='monthly',
                 vol_scale=5, rebal_slip=1, pnl_name='PNL_GROWTH_PZN05',
                 min_obs=250, thresh=2)

    pnl.make_pnl(sig='GROWTH', sig_op='zn_score_pan', rebal_freq='monthly',
                 vol_scale=5, rebal_slip=1, pnl_name='PNL_GROWTH_PZN',
                 min_obs=250, thresh=2.5)

    pnl.make_long_pnl(vol_scale=10, label="Long")

    # Able to adjust the start date for any respective visual analysis.
    pnl.plot_pnls(start="2010-01-01")

    # Return evaluation and PnL DataFrames.
    cids_subset = ['ALL']
    # Test the inclusion of a single benchmark correlation.
    df_eval = pnl.evaluate_pnls(
        pnl_cats=['PNL_GROWTH_PZN', 'PNL_GROWTH_PZN05', "Long"],
        pnl_cids=cids_subset
    )
