import numpy as np
import pandas as pd
from typing import List, Union
from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.shape_dfs import reduce_df

def hedge_ratio(df: pd.DataFrame, xcat: List[str] = None, cids: List[str] = None,
                hedge_return: str = None, start: str = None, end: str = None,
                meth: str = 'ols', oos: bool = True, rfreq: str = 'm',
                minobs: int = 24, blacklist: dict = None):

    """
    Return dataframe of hedge ratios for one or more return categories.
    
    :param <pd.Dataframe> df: standardized data frame with the following necessary columns:
        'cid', 'xcats', 'real_date' and 'value. Will contain all of the data across all
        macroeconomic fields.
    :param <str> xcat:  extended category denoting the return series for which the
        hedge ratios are calculated. In order to hedge against the main position, compute
        hedging ratios for the residual cross-sections defined on the panel. For
        instance, a possible strategy would be to hedge a range of local equity index
        positions against the S&P 500.
    :param <List[str]> cids: cross sections for which hedge ratios are calculated;
        default is all available for the category.
    :param <str> hedge_return: ticker of return of the hedge asset or basket.
    :param <bool> oos: if True (default) hedge ratio are calculated out-of-sample,
        i.e. for the period subsequent to the estimation period at the given
        re-estimation frequency.
    :param <str> refreq: re-estimation frequency. Frequency at which hedge ratios are
        re-estimated. The re-estimation is conducted at the end of the period and
        fills all days of the following period.
    :param <int> min_obs: the minimum number of observations required.
    :param <str> meth: method to estimate hedge ratio. At present the only method is
        OLS regression.
    
    N.B.: A hedge ratio is defined as the sensitivity of the main return in respect
    to the hedge return
    
    """

    cols = ['cid', 'xcat', 'real_date', 'value']
    assert list(df.columns) == cols


if __name__ == "__main__":
    cids = ['AUD', 'CAD', 'GBP', 'USD', 'NZD']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']

    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add',
                                                'sd_mult'])

    df_cids.loc['AUD'] = ['2010-01-01', '2020-12-31', 0.5, 2]
    df_cids.loc['CAD'] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP'] = ['2012-01-01', '2020-11-30', -0.2, 0.5]
    df_cids.loc['USD'] = ['2013-01-01', '2020-09-30', -0.2, 0.5]
    df_cids.loc['NZD'] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR'] = ['2010-01-01', '2020-12-31', 0, 1, 0, 0.3]
    df_xcats.loc['CRY'] = ['2011-01-01', '2020-10-30', 1, 2, 0.9, 0.5]
    df_xcats.loc['GROWTH'] = ['2012-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL'] = ['2013-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)