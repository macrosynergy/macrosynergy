"""
Functions for visualizing data grading and blacklisted periods from a quantamental DataFrame.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
import macrosynergy.visuals as msv


def heatmap_grades(df: pd.DataFrame, xcats:  List[str],  cids: List[str] = None,
                   start: str = '2000-01-01', end: str = None, grade: str = 'grading',
                   title: str = None, size: Tuple[float] = None):

    """Displays a heatmap of grading

    :param <pd.Dataframe> df: standardized DataFrame with the necessary columns:
        'cid', 'xcat', 'real_date' and 'grading'.
    :param List[str] xcats: extended categorys to be checked on.
    :param <List[str]> cids: cross sections to visualize. Default is all in  DataFrame.
    :param <str> start: earliest date in ISO format. Default is earliest available.
    :param <str> end: latest date in ISO format. Default is latest available.
    :param <str> grade: name of column that contains the grades. Default is 'grading'.
    :param <str> title: string of chart title; defaults depend on type of range plot.
    :param <Tuple[float]> size: Tuple of width and height of graph.
        Default is None, meaning it is set in accordance with df.

    """
    msv.view.grades(
        df=df,
        xcats=xcats,
        cids=cids,
        start=start,
        end=end,
        grade=grade,
        title=title,
        figsize=size
    )


if __name__ == "__main__":

    np.random.seed(0)
    
    cids = ['NZD', 'AUD', 'CAD', 'GBP']
    xcats = ['XR', 'CRY', 'GROWTH', 'INFL']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest',
                                                'mean_add', 'sd_mult'])
    df_cids.loc['AUD',] = ['2000-01-01', '2020-12-31', 0.1, 1]
    df_cids.loc['CAD',] = ['2001-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP',] = ['2002-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD',] = ['2002-01-01', '2020-09-30', -0.1, 2]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add',
                                                  'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR',] = ['2000-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY',] = ['2000-01-01', '2020-10-30', 1, 2, 0.95, 1]
    df_xcats.loc['GROWTH',] = ['2001-01-01', '2020-10-30', 1, 2, 0.9, 1]
    df_xcats.loc['INFL',] = ['2001-01-01', '2020-10-30', 1, 2, 0.8, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfd['grading'] = '3'
    filter_date = dfd['real_date'] >= pd.to_datetime('2010-01-01')
    filter_cid = dfd['cid'].isin(['NZD', 'AUD'])
    dfd.loc[filter_date & filter_cid, 'grading'] = '1'
    filter_date = dfd['real_date'] >= pd.to_datetime('2013-01-01')
    filter_xcat = dfd['xcat'].isin(['CRY', 'GROWTH'])
    dfd.loc[filter_date & filter_xcat, 'grading'] = '2.1'
    filter_xcat = dfd['xcat'] == 'XR'
    dfd.loc[filter_xcat, 'grading'] = 1

    heatmap_grades(dfd, xcats=['CRY', 'GROWTH', 'INFL'], cids=cids)

    dfd.info()
