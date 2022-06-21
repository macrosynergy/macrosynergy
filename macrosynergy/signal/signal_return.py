
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as skm
from scipy import stats 

from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import categories_df


class SignalReturnRelations:

    """
    Class for analyzing and visualizing relations between a signal and the subsequent
    returns.

    :param <pd.Dataframe> df: standardized data frame with the following necessary
        columns: 'cid', 'xcat', 'real_date' and 'value.
    :param <str> ret: return category.
    :param <str> sig: signal category.
    :param <bool> sig_neg: if set to True  puts the signal in negative terms for all
        analyses. Default is False.
    :param <str> start: earliest date in ISO format. Default is None in which case the
        earliest date in the df will be used.
    :param <str> end: latest date in ISO format. Default is None in which case the
        latest date in the df will be used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from
        the data frame. If one cross section has several blacklist periods append numbers
        to the cross section code.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
    :param <str> agg_sig: aggregation method applied to the signal value in down-
        sampling. The default is "last".
    :param <int> fwin: Forward window of return category in base periods. Default is 1.
        This conceptually corresponds to the holding period of a position in
        accordance with the signal.

    """
    def __init__(self, df: pd.DataFrame, ret: str, sig: str, cids: List[str] = None,
                 sig_neg: bool=False,
                 start: str = None, end: str = None, fwin: int = 1,
                 blacklist: dict = None, agg_sig: str = 'last', freq: str = 'M'):

        self.dic_freq = {'D': 'daily', 'W': 'weekly', 'M': 'monthly',
                         'Q': 'quarterly', 'A': 'annual'}
        self.metrics = ['accuracy', 'bal_accuracy', 'pos_sigr', "pos_retr",
                        'pos_prec', 'neg_prec',
                        'pearson', 'pearson_pval', 'kendall', 'kendall_pval']

        self.df = categories_df(df, [sig, ret], cids, 'value', start=start, end=end,
                                freq=freq, blacklist=blacklist,
                                lag=1, fwin=fwin, xcat_aggs=[agg_sig, 'mean'])

        if sig_neg:
            self.df.iloc[:, 1] = -1 * self.df.iloc[:, 1]
            self.sig = sig+"_NEG"
            self.df.rename(columns={sig:self.sig}, inplace=True)
        else:
            self.sig = sig

        self.ret = ret
        self.freq = freq
        self.cids = list(np.sort(self.df.index.get_level_values(0).unique()))
        self.df_cs = self.panel_relations(cs_type='cids')
        self.df_ys = self.panel_relations(cs_type='years')

    def panel_relations(self, cs_type: str = 'cids'):
        """
        Creates a DataFrame with information on the signal-return relation
        across cids or years and the panel.

        """

        assert cs_type in ['cids', 'years']
        df = self.df.dropna(how='any')

        if cs_type == 'cids':
            css = self.cids
        else:
            df['year'] = np.array(df.reset_index(level=1)['real_date'].dt.year)
            css = [str(i) for i in np.sort(df['year'].unique())]

        statms = self.metrics
        df_out = pd.DataFrame(index=['Panel', 'Mean', 'PosRatio'] + css, columns=statms)

        for cs in (css + ['Panel']):
            # Row names of cross-sections or years.
            if cs in css:
                if cs_type == 'cids':
                    df_cs = df.loc[cs]
                else:
                    df_cs = df[df['year'] == float(cs)]
            elif cs == 'Panel':
                df_cs = df

            df_sgs = np.sign(df_cs.loc[:, [self.ret, self.sig]])
            df_sgs = df_sgs[~((df_sgs.iloc[:, 0] == 0) | (df_sgs.iloc[:, 1] == 0))]

            sig = df_sgs[self.sig]
            ret = df_sgs[self.ret]
            df_out.loc[cs, 'accuracy'] = skm.accuracy_score(sig, ret)
            df_out.loc[cs, 'bal_accuracy'] = skm.balanced_accuracy_score(sig, ret)
            df_out.loc[cs, 'pos_sigr'] = np.mean(sig == 1)
            df_out.loc[cs, "pos_retr"] = np.mean(ret == 1)
            df_out.loc[cs, 'pos_prec'] = skm.precision_score(ret, sig, pos_label=1)
            df_out.loc[cs, 'neg_prec'] = skm.precision_score(ret, sig, pos_label=-1)

            ret_vals, sig_vals = df_cs[self.ret], df_cs[self.sig]
            df_out.loc[cs, ['kendall', 'kendall_pval']] = stats.kendalltau(ret_vals,
                                                                           sig_vals)
            df_out.loc[cs, ['pearson', 'pearson_pval']] = stats.pearsonr(ret_vals,
                                                                         sig_vals)

        df_out.loc['Mean', :] = df_out.loc[css, :].mean()

        above50s = statms[0:6]
        df_out.loc['PosRatio', above50s] = (df_out.loc[css, above50s] > 0.5).mean()
        above0s = [statms[i] for i in [6, 8]]
        df_out.loc['PosRatio', above0s] = (df_out.loc[css, above0s] > 0).mean()
        below50s = [statms[i] for i in [7, 9]]
        pos_pvals = np.mean(np.array(df_out.loc[css, below50s] < 0.5)
                            * np.array(df_out.loc[css, above0s] > 0), axis=0)
        df_out.loc['PosRatio', below50s] = pos_pvals  # pos corrs with error prob < 50%
        return df_out.astype('float')

    def cross_section_table(self):
        """
        Returns a dataframe with information on the signal-return relation across
        sections and the panel.
        """

        return self.df_cs.round(decimals=3)

    def yearly_table(self):
        """
        Returns dataframe with information on the signal-return relation across years
        and the panel.
        """
        return self.df_ys.round(decimals=3)

    def accuracy_bars(self, type: str = 'cross_section', title: str = None,
                      size: Tuple[float] = None,
                      legend_pos: str = 'best'):
        """
        Bars of overall and balanced accuracy

        :param <str> type: type of segment over which bars are drawn.
            Must be 'cross_section' (default) or 'years'
        :param <str> title: chart header. Default will be applied if none is chosen.
        :param <Tuple[float]> size: 2-tuple of width and height of plot.
            Default will be applied if none is chosen.
        :param <str> legend_pos: position of legend box. Default is 'best'.
            See matplotlib.pyplot.legend.

        """

        assert type in ['cross_section', 'years']

        df_xs = self.df_cs if type == 'cross_section' else self.df_ys
        dfx = df_xs[~df_xs.index.isin(['PosRatio'])]

        if title is None:
            title = f'Accuracy for sign prediction of {self.ret} based on {self.sig} ' \
                    f'at {self.dic_freq[self.freq]} frequency'
        if size is None:
            size = (np.max([dfx.shape[0]/2, 8]), 6)

        plt.style.use('seaborn')
        plt.figure(figsize=size)
        x_indexes = np.arange(len(dfx.index))  # generic x index
        w = 0.4  # offset parameter, related to width of bar
        plt.bar(x_indexes - w / 2, dfx['accuracy'], label='Accuracy', width=w,
                color='lightblue')
        plt.bar(x_indexes + w / 2, dfx['bal_accuracy'], label='Balanced Accuracy',
                width=w, color='steelblue')
        plt.xticks(ticks=x_indexes, labels=dfx.index,
                   rotation=0)  # customize x ticks/labels
        plt.axhline(y=0.5, color='black', linestyle='-', linewidth=0.5)
        plt.ylim(np.round(np.max(dfx.loc[:, ['accuracy', 'bal_accuracy']].
                                 min().min()-0.03, 0), 2))
        plt.title(title)
        plt.legend(loc=legend_pos)
        plt.show()

    def correlation_bars(self, type: str = 'cross_section', title: str = None,
                         size: Tuple[float] = None,
                         legend_pos: str = 'best'):
        """
        Correlation coefficients and significance.

        :param <str> type: type of segment over which bars are drawn. Must be
            'cross_section' (default) or 'years'
        :param <str> title: chart header. Default will be applied if none is chosen.
        :param <Tuple[float]> size: 2-tuple of width and height of plot.
            Default will be applied if none is chosen.
        :param <str> legend_pos: position of legend box. Default is 'best'.
            See matplotlib.pyplot.legend.

        """

        df_xs = self.df_cs if type == 'cross_section' else self.df_ys
        dfx = df_xs[~df_xs.index.isin(['PosRatio', 'Mean'])]

        pprobs = np.array([(1 - pv) * (np.sign(cc) + 1) / 2
                           for pv, cc in zip(dfx['pearson_pval'], dfx['pearson'])])
        pprobs[pprobs == 0] = 0.01  # token small value for bar
        kprobs = np.array([(1 - pv) * (np.sign(cc) + 1) / 2
                           for pv, cc in zip(dfx['kendall_pval'], dfx['kendall'])])
        kprobs[kprobs == 0] = 0.01  # token small value for bar

        if title is None:
            title = f'Positive correlation probability of {self.ret} ' \
                    f'and lagged {self.sig} ' \
                    f'at {self.dic_freq[self.freq]} frequency'
        if size is None:
            size = (np.max([dfx.shape[0]/2, 8]), 6)

        plt.style.use('seaborn')
        plt.figure(figsize=size)
        x_indexes = np.arange(len(dfx.index))  # generic x index
        w = 0.4  # offset parameter, related to width of bar
        plt.bar(x_indexes - w / 2, pprobs, label='Pearson',
                width=w, color='lightblue')
        plt.bar(x_indexes + w / 2, kprobs, label='Kendall',
                width=w, color='steelblue')
        plt.xticks(ticks=x_indexes, labels=dfx.index, rotation=0)  # custom x tix/labels
        plt.axhline(y=0.95, color='orange', linestyle='--',
                    linewidth=0.5, label='95% probability')
        plt.axhline(y=0.99, color='red', linestyle='--',
                    linewidth=0.5, label='99% probability')
        plt.title(title)
        plt.legend(loc=legend_pos)
        plt.show()

    def summary_table(self):
        """
        Condensed summary table of signal-return relations.
        N.B.:
        The interpretation of the columns is generally as follows:

        accuracy refers accuracy for binary classification, i.e. positive or negative
            return, and gives the ratio of correct prediction of the sign of returns
            to all predictions.
        bal_accuracy refers to balanced accuracy. This is the average of the ratios of
            correctly detected positive returns and correctly detected negative returns.
            The denominators here are the total of actual positive and negative returns
            cases. Technically, this is the average of sensitivity and specificity.
        pos_sigr is the ratio of positive signals to all predictions. It indicates the
            long bias of the signal.
        pos_retr is the ratio of positive returns to all observed returns. It indicates
            the positive bias of the returns.
        pos_prec means positive precision, i.e. the ratio of correct positive return
            predictions to all positive predictions. It indicates how well the positive
            predictions of the signal have fared. Generally, good positive precision is
            easy to accomplish if the ratio of positive returns has been high.
        neg_prec means negative precision, i.e. the ratio of correct negative return
            predictions to all negative predictions. It indicates how well the negative
            predictions of the signal have fared. Generally, good positive precision is
            hard to accomplish if the ratio of positive returns has been high.
        pearson is the Pearson correlation coefficient between signal and subsequent
            return.
        pearson_pval is the probability that the (positive) correlation has been
            accidental, assuming that returns are independently distributed. This
            statistic would be invalid for forward moving averages.
        kendall is the Kendall correlation coefficient between signal and subsequent
            return.
        kendall_pval is the probability that the (positive) correlation has been
            accidental, assuming that returns are independently distributed. This
            statistic would be invalid for forward moving averages.

        The rows have the following meaning:

        Panel refers to the the whole panel of cross sections and sample period,
            excluding unavailable and blacklisted periods.
        Mean years is the mean of the statistic across all years.
        Mean cids is the mean of the statistic across all sections.
        Positive ratio is the ratio of positive years or cross sections for which the
            statistic was above its "neutral" level, i.e. above 0.5 for classification
            ratios and positive correlation probabilities and above 0 for the
            correlation coefficients.
        """

        dfys = self.df_ys.round(decimals=3)
        dfcs = self.df_cs.round(decimals=3)
        dfsum = dfys.iloc[:3, ].append(dfcs.iloc[1:3, ])
        dfsum.index = ["Panel", "Mean years", "Positive ratio",
                       "Mean cids", "Positive ratio"]

        return dfsum


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index=cids,
                           columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD'] = ['2000-01-01', '2020-12-31', 0, 1]
    df_cids.loc['CAD'] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', 0., 2]

    df_xcats = pd.DataFrame(index=xcats,
                            columns=['earliest', 'latest', 'mean_add', 'sd_mult',
                                     'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2000-01-01', '2020-10-30', 0, 2, 0.95, 1]
    df_xcats.loc['GROWTH'] = ['2001-01-01', '2020-10-30', 0, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2001-01-01', '2020-10-30', 0, 2, 0.8, 0.5]

    black = {'AUD': ['2006-01-01', '2015-12-31'], 'GBP': ['2012-01-01', '2100-01-01']}

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    srr = SignalReturnRelations(dfd, sig='CRY', ret='XR', freq='D', blacklist=black)
    srr.summary_table()
    srn = SignalReturnRelations(dfd, sig='CRY', sig_neg=True,
                                ret='XR', freq='D', blacklist=black)
    srn.summary_table()

    srr.correlation_bars(type='cross_section')
    srn.correlation_bars(type='cross_section')

    srr.accuracy_bars(type='cross_section')
    df_cs_stats = srr.cross_section_table()
    df_ys_stats = srr.yearly_table()
    print(df_cs_stats)
    print(df_ys_stats)