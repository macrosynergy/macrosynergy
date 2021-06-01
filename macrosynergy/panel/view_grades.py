import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df


def view_grades(df: pd.DataFrame, xcat: str,  cids: List[str] = None,
                start: str = '2000-01-01', end: str = None, grade: str = 'grading',
                title: str = None, ylab: str = None, size: Tuple[float] = (16, 8)):

    """Displays stacked bars of grade composition for category

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcat', 'real_date' and at least one column with values of interest.
    :param <str> xcat: extended categorys to be checked on.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the dataframe.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <str> grade: name of column that contains the grades Default is 'grading'.
    :param <str> title: string of chart title; defaults depend on type of range plot.
    :param <str> ylab: y label. Default is no label.
    :param <Tuple[float]> size: Tuple of width and height of graph. Default is (16, 8).

    """
    df, xcats, cids = reduce_df(df, [xcat], cids, start, end, out_all=True)
    df = df[['cid', 'real_date', grade]]

    s_date = df['real_date'].min().strftime('%Y-%m-%d')
    e_date = df['real_date'].max().strftime('%Y-%m-%d')

    if title is None:
        title = f'Daily grade counts for {xcat} from {s_date} to {e_date}'
    if ylab is None:
        ylab = ""

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(figsize=size)

    grades = sorted(df[grade].unique(), reverse=True)
    sns.countplot(x="cid", hue=grade, hue_order=grades, data=df, palette='Blues')

    ax.set_title(title,  fontdict={'fontsize': 16})
    ax.set_xlabel("")
    ax.set_ylabel(ylab)
    ax.xaxis.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])
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

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    dfd['grading'] = '3'
    dfd.loc[dfd['real_date'] >= pd.to_datetime('2015-01-01'), 'grading'] = '2.1'
    dfd.loc[dfd['real_date'] >= pd.to_datetime('2016-01-01'), 'grading'] = '1'

    view_grades(dfd, 'CRY', cids=cids[:3], start='2012-01-01')

    dfd.info()
