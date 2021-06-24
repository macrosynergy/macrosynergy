# shows various relation metrics between signals and (subsequent returns)
# rows are cross sections or year + average/panel/pos_ratio
# columns are metrics: obs/ pearson, spearman, probability/ signal_pr, return_pr/ accuracy, sensitivity, specificity,
# balanced accuracy
# give choice of frequency (d, m)

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

    """Class for analyzing and visualizing relations between a signal and return indicator

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
                 start: str = None, end: str = None, fwin: int = 1, blacklist: dict = None, years=None,
                 freq: str = 'M'):

        self.df = categories_df(df, [ret, sig], cids, 'value', start=start, end=end, freq=freq, blacklist=blacklist,
                                lag=1, fwin=fwin, xcat_aggs=['mean', 'last'])
        self.ret = ret
        self.sig = sig
        self.cids = list(np.sort(self.df.index.get_level_values(0).unique()))
        self.df_cs = self.panel_relations(cs_type='cids')
        self.df_ys = self.panel_relations(cs_type='years')
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

        statms = ['accuracy', 'bal_accuracy', 'f1_score', 'pearson', 'pearson_prob', 'kendall', 'kendall_prob']
        df_out = pd.DataFrame(index=['Panel', 'Average', 'PosRatio'] + css, columns=statms)

        for cs in (css + ['Panel']):
            if cs in css:
                if cs_type == 'cids':
                    df_cs = df.loc[cs,]
                else:
                    df_cs = df[df['year'] == float(cs)]
            elif cs == 'Panel':
                df_cs = df

            ret_signs, sig_signs = np.sign(df_cs[self.ret]), np.sign(df_cs[self.sig])
            df_out.loc[cs, 'accuracy'] = skm.accuracy_score(ret_signs, sig_signs)
            df_out.loc[cs, 'bal_accuracy'] = skm.balanced_accuracy_score(ret_signs, sig_signs)
            df_out.loc[cs, 'f1_score'] = skm.f1_score(ret_signs, sig_signs, average='weighted')

            ret_vals, sig_vals = df_cs[self.ret], df_cs[self.sig]
            df_out.loc[cs, ['kendall', 'kendall_prob']] = stats.kendalltau(ret_vals, sig_vals)
            df_out.loc[cs, ['pearson', 'pearson_prob']] = stats.pearsonr(ret_vals, sig_vals)

        df_out.loc['Average', :] = df_out.loc[css, :].mean()

        above50s = statms[0:3]
        df_out.loc['PosRatio', above50s] = (df_out.loc[css, above50s] > 0.5).mean()
        above0s = [statms[i] for i in [3, 5]]
        df_out.loc['PosRatio', above0s] = (df_out.loc[css, above0s] > 0).mean()
        below50s = [statms[i] for i in [4, 6]]
        pos_probs = np.mean(np.array(df_out.loc[css, below50s] < 0.5) * np.array(df_out.loc[css, above0s] > 0), axis=0)
        df_out.loc['PosRatio', below50s] = pos_probs  # positive correlations with error probabilities < 50%
        return df_out

    def cross_section_table(self):
        """Returns a dataframe with information on the signal-return relation across sections and the panel."""
        return self.df_cs.round(decimals=3)

    def yearly_table(self):
        """Returns dataframe with information on the signal-return relation across years and the panel."""
        return self.df_ys.round(decimals=3)

    def accuracy_bars(self):
        pass

    def correlation_bars(self):
        pass


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

    srr = SignalReturnRelations(dfd, sig='CRY', ret='XR')
    df_cs_stats = srr.cross_section_table()
    df_ys_stats = srr.yearly_table()

    print(srr)
