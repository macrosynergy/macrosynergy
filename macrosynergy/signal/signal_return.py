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

    """Class for analyzing and visualizing relations between a signal and subsequent return

    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value.
    :param <str> ret: return category.
    :param <str> sig: signal category.
    :param <List[str]> cids: cross sections to be considered. Default is all in the data frame.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <dict> blacklist: cross sections with date ranges that should be excluded from the data frame.
    :param <str> freq: letter denoting frequency at which the series are to be sampled.
        This must be one of 'D', 'W', 'M', 'Q', 'A'. Default is 'M'.
    :param <int> fwin: Forward window of return category in base periods. Default is 1.

    """
    def __init__(self, df: pd.DataFrame, ret: str, sig: str, cids: List[str] = None,
                 start: str = None, end: str = None, fwin: int = 1, blacklist: dict = None,
                 freq: str = 'M'):

        self.df = categories_df(df, [ret, sig], cids, 'value', start=start, end=end, freq=freq, blacklist=blacklist,
                                lag=1, fwin=fwin, xcat_aggs=['mean', 'last'])
        self.ret = ret
        self.sig = sig
        self.freq = freq
        self.cids = list(np.sort(self.df.index.get_level_values(0).unique()))
        self.df_cs = self.panel_relations(cs_type='cids')
        self.df_ys = self.panel_relations(cs_type='years')
        self.dic_freq = {'D': 'daily', 'W': 'weekly', 'M': 'monthly', 'Q': 'quarterly', 'A': 'annual'}

        """Creates a dataframe of return and signal in the appropriate form for subsequent analysis."""

    def panel_relations(self, cs_type: str = 'cids'):
        """Creates a dataframe with information on the signal-return relation across cids/years and the panel."""

        assert cs_type in ['cids', 'years']
        if cs_type == 'cids':
            df = self.df.dropna(how='any')
            css = self.cids
        else:
            df = self.df.dropna(how='any')
            df['year'] = np.array(df.reset_index(level=1)['real_date'].dt.year)
            css = [str(i) for i in df['year'].unique()]

        statms = ['accuracy', 'bal_accuracy', 'f1_score', 'pearson', 'pearson_pval', 'kendall', 'kendall_pval']
        df_out = pd.DataFrame(index=['Panel', 'Mean', 'PosRatio'] + css, columns=statms)

        for cs in (css + ['Panel']):
            if cs in css:
                if cs_type == 'cids':
                    df_cs = df.loc[cs,]
                else:
                    df_cs = df[df['year'] == float(cs)]
            elif cs == 'Panel':
                df_cs = df

            ret_signs, sig_signs = np.sign(df_cs[self.ret]), np.sign(df_cs[self.sig])
            df_out.loc[cs, 'accuracy'] = skm.accuracy_score(sig_signs, ret_signs)
            df_out.loc[cs, 'bal_accuracy'] = skm.balanced_accuracy_score(sig_signs, ret_signs)
            df_out.loc[cs, 'f1_score'] = skm.f1_score(sig_signs, ret_signs, average='weighted')

            ret_vals, sig_vals = df_cs[self.ret], df_cs[self.sig]
            df_out.loc[cs, ['kendall', 'kendall_pval']] = stats.kendalltau(ret_vals, sig_vals)
            df_out.loc[cs, ['pearson', 'pearson_pval']] = stats.pearsonr(ret_vals, sig_vals)

        df_out.loc['Mean', :] = df_out.loc[css, :].mean()

        above50s = statms[0:3]
        df_out.loc['PosRatio', above50s] = (df_out.loc[css, above50s] > 0.5).mean()
        above0s = [statms[i] for i in [3, 5]]
        df_out.loc['PosRatio', above0s] = (df_out.loc[css, above0s] > 0).mean()
        below50s = [statms[i] for i in [4, 6]]
        pos_pvals = np.mean(np.array(df_out.loc[css, below50s] < 0.5) * np.array(df_out.loc[css, above0s] > 0), axis=0)
        df_out.loc['PosRatio', below50s] = pos_pvals  # positive correlations with error probabilities < 50%
        return df_out

    def cross_section_table(self):
        """Returns a dataframe with information on the signal-return relation across sections and the panel."""
        return self.df_cs.round(decimals=3)

    def yearly_table(self):
        """Returns dataframe with information on the signal-return relation across years and the panel."""
        return self.df_ys.round(decimals=3)

    def accuracy_bars(self, type: str = 'cross_section', title: str = None, size: Tuple[float] = None,
                      legend_pos: str = 'best'):
        """Bars of overall and balanced accuracy

        :param <str> type: type of segment over which bars are drawn. Must be 'cross_section' (default) or 'years'
        :param <str> title: chart header. Default will be applied if none is chosen.
        :param <Tuple[float]> size: 2-tuple of width and height of plot.  Default will be applied if none is chosen.
        :param <str> legend_pos: position of legend box. Default is 'best'. See matplotlib.pyplot.legend.

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
        plt.bar(x_indexes - w / 2, dfx['accuracy'], label='Accuracy', width=w, color='lightblue')
        plt.bar(x_indexes + w / 2, dfx['bal_accuracy'], label='Balanced Accuracy', width=w, color='steelblue')
        plt.xticks(ticks=x_indexes, labels=dfx.index, rotation=0)  # customize x ticks/labels
        plt.axhline(y=0.5, color='black', linestyle='-', linewidth=0.5)
        plt.ylim(np.round(np.max(dfx.loc[:, ['accuracy', 'bal_accuracy']].min().min()-0.03, 0), 2))
        plt.title(title)
        plt.legend(loc=legend_pos)
        plt.show()

    def correlation_bars(self, type: str = 'cross_section', title: str = None, size: Tuple[float] = None,
                         legend_pos: str = 'best'):
        """Correlation coefficients and significance

        :param <str> type: type of segment over which bars are drawn. Must be 'cross_section' (default) or 'years'
        :param <str> title: chart header. Default will be applied if none is chosen.
        :param <Tuple[float]> size: 2-tuple of width and height of plot.  Default will be applied if none is chosen.
        :param <str> legend_pos: position of legend box. Default is 'best'. See matplotlib.pyplot.legend.

        """

        df_xs = self.df_cs if type == 'cross_section' else self.df_ys
        dfx = df_xs[~df_xs.index.isin(['PosRatio', 'Mean'])]

        pprobs = np.array([(1 - pv) * (np.sign(cc) + 1) / 2 for pv, cc in zip(dfx['pearson_pval'], dfx['pearson'])])
        pprobs[pprobs == 0] = 0.01  # token small value for bar
        kprobs = np.array([(1 - pv) * (np.sign(cc) + 1) / 2 for pv, cc in zip(dfx['kendall_pval'], dfx['kendall'])])
        kprobs[kprobs == 0] = 0.01  # token small value for bar

        if title is None:
            title = f'Positive correlation probability of {self.ret} and lagged {self.sig} ' \
                    f'at {self.dic_freq[self.freq]} frequency'
        if size is None:
            size = (np.max([dfx.shape[0]/2, 8]), 6)

        plt.style.use('seaborn')
        plt.figure(figsize=size)
        x_indexes = np.arange(len(dfx.index))  # generic x index
        w = 0.4  # offset parameter, related to width of bar
        plt.bar(x_indexes - w / 2, pprobs, label='Pearson', width=w, color='lightblue')
        plt.bar(x_indexes + w / 2, kprobs, label='Kendall', width=w, color='steelblue')
        plt.xticks(ticks=x_indexes, labels=dfx.index, rotation=0)  # customize x ticks/labels
        plt.axhline(y=0.95, color='orange', linestyle='--', linewidth=0.5, label='95% probability')
        plt.axhline(y=0.99, color='red', linestyle='--', linewidth=0.5, label='99% probability')
        plt.title(title)
        plt.legend(loc=legend_pos)
        plt.show()


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

    srr = SignalReturnRelations(dfd, sig='CRY', ret='XR', freq='D', blacklist=black)
    srr.correlation_bars(type='cross_section')
    srr.accuracy_bars(type='cross_section')
    df_cs_stats = srr.cross_section_table()
    df_ys_stats = srr.yearly_table()
    print(df_cs_stats)
    print(df_ys_stats)