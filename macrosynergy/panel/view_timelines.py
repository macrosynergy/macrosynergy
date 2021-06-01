import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Union, Tuple

from macrosynergy.management.simulate_quantamental_data import make_qdf
from macrosynergy.management.check_availability import reduce_df


def view_timelines(df: pd.DataFrame, xcats: List[str] = None,  cids: List[str] = None,
                   start: str = '2000-01-01', end: str = None, val: str = 'value', cumsum: bool = False,
                   title: str = None, title_adj=0.95,
                   ncol: int = 3, same_y: bool = True, size: Tuple[float] = (12, 7), aspect: float = 1.7):

    """Display facet grid of time lines of one or more categories

    :param <pd.Dataframe> df: standardized dataframe with the following necessary columns:
        'cid', 'xcats', 'real_date' and at least one column with values of interest.
    :param <List[str]> xcats: extended categories to be checked on. Default is all in the dataframe.
    :param <List[str]> cids: cross sections to be checked on. Default is all in the dataframe.
        If this contains only one cross section a single chart is created rather than a facet grid.
    :param <str> start: earliest date in ISO format. Default is None and earliest date in df is used.
    :param <str> end: latest date in ISO format. Default is None and latest date in df is used.
    :param <str> val: name of column that contains the values of interest. Default is 'value'.
    :param <bool> cumsum: chart the cumulative sum of the value. Default is False.
    :param <str> title: chart heading. Default is no title.
    :param <float> title_adj: parameter that positions title relative to the facet. Default is 0.95.
    :param <int> ncol: number of columns in facet. Default is 3.
    :param <bool> same_y: if True (default) all axis plots in the facet share the same y axis.
    :param <Tuple[float]> size: two-element tuple setting width/height of figure. Default is (12, 7).
    :param <float> aspect: width-height ratio for plots in facet.

    """

    df, xcats, cids = reduce_df(df, xcats, cids, start, end, out_all=True)

    if cumsum:
        df[val] = df.sort_values(['cid', 'xcat', "real_date"])[['cid', 'xcat', val]].groupby(['cid', 'xcat']).cumsum()

    sns.set(rc={'figure.figsize': size}, style='darkgrid')
    if len(cids) == 1:
        ax = sns.lineplot(data=df, x='real_date', y=val, hue='xcat', ci=None)
        plt.axhline(y=0, c=".5")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[0:], labels=labels[0:])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if title is not None:
            plt.title(title)
    else:
        fg = sns.FacetGrid(data=df, col='cid', col_wrap=ncol, sharey=same_y, aspect=aspect, col_order=cids)
        fg.map_dataframe(sns.lineplot, x='real_date', y=val, hue='xcat', ci=None)
        fg.map(plt.axhline, y=0, c=".5")
        fg.set_titles(col_template='{col_name}')
        fg.add_legend()
        if title is not None:
            fg.fig.subplots_adjust(top=title_adj)
            fg.fig.suptitle(title, fontsize=20)
    plt.show()


if __name__ == "__main__":

    cids = ['AUD', 'CAD', 'GBP', 'NZD']
    xcats = ['XR', 'CRY']
    df_cids = pd.DataFrame(index=cids, columns=['earliest', 'latest', 'mean_add', 'sd_mult'])
    df_cids.loc['AUD', ] = ['2010-01-01', '2020-12-31', 0.2, 0.2]
    df_cids.loc['CAD', ] = ['2011-01-01', '2020-11-30', 0, 1]
    df_cids.loc['GBP', ] = ['2012-01-01', '2020-11-30', 0, 2]
    df_cids.loc['NZD', ] = ['2012-01-01', '2020-09-30', -0.1, 3]

    df_xcats = pd.DataFrame(index=xcats, columns=['earliest', 'latest', 'mean_add', 'sd_mult', 'ar_coef', 'back_coef'])
    df_xcats.loc['XR', ] = ['2010-01-01', '2020-12-31', 0.1, 1, 0, 0.3]
    df_xcats.loc['CRY', ] = ['2011-01-01', '2020-10-30', 1, 2, 0.95, 0.5]

    dfd = make_qdf(df_cids, df_xcats, back_ar=0.75)

    view_timelines(dfd, xcats=['XR', 'CRY'], cids=cids[0], ncol=1, title='AUD return and carry')
    view_timelines(dfd, xcats=['CRY'], cids=cids, ncol=2, title='Carry')
    view_timelines(dfd, xcats=['CRY', 'XR'], cids=cids, ncol=2, title='Carry and return')
    view_timelines(dfd, xcats=['XR'], cids=cids, ncol=2, cumsum=True, same_y=False, aspect=2)


    dfd.info()